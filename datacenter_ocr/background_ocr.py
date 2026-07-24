from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
import time
from typing import Any, Callable, Literal
import uuid

from datacenter_ocr.monitoring_records import build_monitoring_rows
from datacenter_ocr.ocr_processing import (
    CellOCRResult,
    process_measurement_cells_with_blank_detection,
)
from datacenter_ocr.processing_metrics import ProcessingMetrics
from datacenter_ocr.sheet_queue import (
    ArchivedPreparedMonitoringSheet,
    restore_prepared_sheet,
)


OCRJobState = Literal[
    "queued",
    "loading_model",
    "running",
    "succeeded",
    "failed",
]
ModelFactory = Callable[[], Any]
CellProcessor = Callable[..., list[CellOCRResult]]
RowBuilder = Callable[[list[CellOCRResult]], list[dict[str, Any]]]


@dataclass(frozen=True)
class OCRJobRequest:
    """Immutable input for one explicitly submitted sheet OCR job."""

    owner_id: str
    sheet_id: str
    source_filename: str
    geometry_mode: str
    prepared_sheet: ArchivedPreparedMonitoringSheet
    processing_metrics: ProcessingMetrics
    cells_per_batch: int
    recognition_strategy: str = "adaptive"


@dataclass(frozen=True)
class OCRJobSnapshot:
    """Thread-safe, lightweight status exposed to the Streamlit UI."""

    job_id: str
    owner_id: str
    sheet_id: str
    source_filename: str
    geometry_mode: str
    state: OCRJobState
    processed_count: int
    total_count: int
    error_message: str
    created_at: float
    started_at: float | None
    finished_at: float | None


@dataclass(frozen=True)
class OCRJobResult:
    """Completed canonical OCR data collected by the Streamlit main thread."""

    job_id: str
    owner_id: str
    sheet_id: str
    geometry_mode: str
    prepared_sheet: ArchivedPreparedMonitoringSheet
    cell_results: list[CellOCRResult]
    monitoring_rows: list[dict[str, Any]]
    processing_metrics: ProcessingMetrics


@dataclass
class _ModelResource:
    model: Any
    construction_seconds: float
    prediction_call_count: int = 0


@dataclass
class _OCRJobRecord:
    request: OCRJobRequest
    state: OCRJobState
    processed_count: int
    total_count: int
    error_message: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: OCRJobResult | None = None
    future: Future[None] | None = None


def _create_paddle_model() -> Any:
    """Construct PaddleOCR lazily inside the single worker thread."""

    from paddleocr import TextRecognition

    print("[OCR MODEL] Loading PaddleOCR model into background worker...")
    return TextRecognition(device="cpu")


class BackgroundOCRManager:
    """
    Run explicitly submitted sheet OCR jobs sequentially in one worker.

    The worker owns model inference and never reads or writes Streamlit
    session state. The main thread polls snapshots and applies completed
    results to the matching sheet identifier.
    """

    def __init__(
        self,
        *,
        model_factory: ModelFactory = _create_paddle_model,
        cell_processor: CellProcessor = (
            process_measurement_cells_with_blank_detection
        ),
        row_builder: RowBuilder = build_monitoring_rows,
    ) -> None:
        self._model_factory = model_factory
        self._cell_processor = cell_processor
        self._row_builder = row_builder
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="monitoring-ocr",
        )
        self._lock = Lock()
        self._jobs: dict[str, _OCRJobRecord] = {}
        self._model_resource: _ModelResource | None = None

    def submit(self, request: OCRJobRequest) -> str:
        """Queue one sheet, rejecting duplicate active jobs for that sheet."""

        with self._lock:
            duplicate = next(
                (
                    job_id
                    for job_id, record in self._jobs.items()
                    if record.request.owner_id == request.owner_id
                    and record.request.sheet_id == request.sheet_id
                    and record.state
                    in ("queued", "loading_model", "running")
                ),
                None,
            )
            if duplicate is not None:
                return duplicate

            job_id = uuid.uuid4().hex
            self._jobs[job_id] = _OCRJobRecord(
                request=request,
                state="queued",
                processed_count=0,
                total_count=len(request.prepared_sheet.cells),
                error_message="",
                created_at=time.time(),
            )
            future = self._executor.submit(self._execute_job, job_id)
            self._jobs[job_id].future = future
            return job_id

    def snapshot(
        self,
        job_id: str,
        owner_id: str,
    ) -> OCRJobSnapshot | None:
        """Return one job only when it belongs to the requesting session."""

        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.request.owner_id != owner_id:
                return None
            return self._snapshot(job_id, record)

    def snapshots_for_owner(self, owner_id: str) -> tuple[OCRJobSnapshot, ...]:
        """Return this session's jobs in explicit submission order."""

        with self._lock:
            snapshots = [
                self._snapshot(job_id, record)
                for job_id, record in self._jobs.items()
                if record.request.owner_id == owner_id
            ]
        return tuple(sorted(snapshots, key=lambda item: item.created_at))

    def consume_result(
        self,
        job_id: str,
        owner_id: str,
    ) -> OCRJobResult | None:
        """Remove and return one successful result to the main UI thread."""

        with self._lock:
            record = self._jobs.get(job_id)
            if (
                record is None
                or record.request.owner_id != owner_id
                or record.state != "succeeded"
                or record.result is None
            ):
                return None
            result = record.result
            del self._jobs[job_id]
            return result

    def discard_terminal_job(self, job_id: str, owner_id: str) -> bool:
        """Remove a completed or failed record before retry or session cleanup."""

        with self._lock:
            record = self._jobs.get(job_id)
            if (
                record is None
                or record.request.owner_id != owner_id
                or record.state not in ("succeeded", "failed")
            ):
                return False
            del self._jobs[job_id]
            return True

    def shutdown(self, *, wait: bool = True) -> None:
        """Release the worker in tests or controlled process shutdown."""

        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _snapshot(
        self,
        job_id: str,
        record: _OCRJobRecord,
    ) -> OCRJobSnapshot:
        return OCRJobSnapshot(
            job_id=job_id,
            owner_id=record.request.owner_id,
            sheet_id=record.request.sheet_id,
            source_filename=record.request.source_filename,
            geometry_mode=record.request.geometry_mode,
            state=record.state,
            processed_count=record.processed_count,
            total_count=record.total_count,
            error_message=record.error_message,
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
        )

    def _set_state(self, job_id: str, state: OCRJobState) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.state = state
            if state in ("loading_model", "running") and record.started_at is None:
                record.started_at = time.time()

    def _update_progress(
        self,
        job_id: str,
        processed_count: int,
        total_count: int,
    ) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.state = "running"
            record.processed_count = processed_count
            record.total_count = total_count

    def _get_model_resource(self) -> _ModelResource:
        if self._model_resource is None:
            construction_start = time.perf_counter()
            model = self._model_factory()
            self._model_resource = _ModelResource(
                model=model,
                construction_seconds=round(
                    time.perf_counter() - construction_start,
                    6,
                ),
            )
        return self._model_resource

    def _execute_job(self, job_id: str) -> None:
        try:
            self._set_state(job_id, "loading_model")
            with self._lock:
                request = self._jobs[job_id].request

            model_was_already_constructed = self._model_resource is not None
            model_resource = self._get_model_resource()
            metrics = request.processing_metrics
            metrics.model_was_warm = model_was_already_constructed
            metrics.model_construction_seconds = (
                0.0
                if model_was_already_constructed
                else model_resource.construction_seconds
            )
            prepared_sheet = restore_prepared_sheet(request.prepared_sheet)
            self._set_state(job_id, "running")
            results = self._cell_processor(
                model=model_resource.model,
                cells=prepared_sheet.cells,
                cells_per_batch=request.cells_per_batch,
                progress_callback=lambda processed, total: (
                    self._update_progress(job_id, processed, total)
                ),
                metrics=metrics,
                recognition_strategy=request.recognition_strategy,
            )
            model_resource.prediction_call_count += (
                metrics.model_predict_call_count
            )

            record_start = time.perf_counter()
            rows = self._row_builder(results)
            metrics.monitoring_record_construction_seconds = round(
                time.perf_counter() - record_start,
                6,
            )
            metrics.recalculate_total()
            result = OCRJobResult(
                job_id=job_id,
                owner_id=request.owner_id,
                sheet_id=request.sheet_id,
                geometry_mode=request.geometry_mode,
                prepared_sheet=request.prepared_sheet,
                cell_results=results,
                monitoring_rows=rows,
                processing_metrics=metrics,
            )
            with self._lock:
                record = self._jobs[job_id]
                record.processed_count = len(request.prepared_sheet.cells)
                record.total_count = len(request.prepared_sheet.cells)
                record.state = "succeeded"
                record.finished_at = time.time()
                record.result = result
        except Exception as error:
            with self._lock:
                record = self._jobs.get(job_id)
                if record is not None:
                    record.state = "failed"
                    record.error_message = (
                        f"{type(error).__name__}: {error}"
                    )
                    record.finished_at = time.time()
