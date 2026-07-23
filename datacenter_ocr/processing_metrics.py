from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time


PROCESS_STARTED_AT = time.perf_counter()

TOTAL_STAGE_TIMING_FIELDS = (
    "upload_decoding_seconds",
    "document_table_detection_seconds",
    "perspective_warp_seconds",
    "measurement_cell_extraction_seconds",
    "blank_detection_seconds",
    "ocr_preprocessing_seconds",
    "ocr_prediction_seconds",
    "postprocessing_seconds",
    "verification_seconds",
    "monitoring_record_construction_seconds",
    "ui_thumbnail_preparation_seconds",
    "model_construction_seconds",
)


@dataclass
class ProcessingMetrics:
    """Stage timings and counters for one monitoring-sheet run.

    The metrics are observational only. They do not influence OCR,
    verification, review, or export decisions.
    """

    schema_version: int = 2
    source_filename: str = ""
    uploaded_fingerprint: str = ""
    extraction_geometry_mode: str = "fixed"
    recognition_strategy: str = "consensus"
    uploaded_width: int = 0
    uploaded_height: int = 0
    process_uptime_seconds: float = 0.0
    model_was_warm: bool | None = None

    upload_decoding_seconds: float = 0.0
    document_table_detection_seconds: float = 0.0
    perspective_warp_seconds: float = 0.0
    measurement_cell_extraction_seconds: float = 0.0
    blank_detection_seconds: float = 0.0
    ocr_preprocessing_seconds: float = 0.0
    ocr_prediction_seconds: float = 0.0
    postprocessing_seconds: float = 0.0
    verification_seconds: float = 0.0
    monitoring_record_construction_seconds: float = 0.0
    ui_thumbnail_preparation_seconds: float = 0.0
    model_construction_seconds: float = 0.0
    first_prediction_warmup_seconds: float | None = None
    total_sheet_processing_seconds: float = 0.0

    total_cell_count: int = 0
    filled_cell_count: int = 0
    blank_cell_count: int = 0
    ocr_input_image_count: int = 0
    model_predict_call_count: int = 0
    requested_batch_size: int = 0
    result_batch_count: int = 0
    adaptive_first_pass_cell_count: int = 0
    adaptive_fallback_cell_count: int = 0

    def add_seconds(self, field_name: str, elapsed_seconds: float) -> None:
        """Add elapsed time to a named timing field."""

        if not hasattr(self, field_name):
            raise AttributeError(f"Unknown processing metric: {field_name}")

        current_value = getattr(self, field_name)
        if current_value is None:
            current_value = 0.0

        setattr(
            self,
            field_name,
            round(float(current_value) + float(elapsed_seconds), 6),
        )

    def capture_process_uptime(self) -> None:
        """Record approximate process uptime from module import time."""

        self.process_uptime_seconds = round(
            time.perf_counter() - PROCESS_STARTED_AT,
            6,
        )

    def recalculate_total(self) -> None:
        """Sum non-overlapping observed stage timings for this sheet run."""

        self.total_sheet_processing_seconds = round(
            sum(float(getattr(self, field_name)) for field_name in TOTAL_STAGE_TIMING_FIELDS),
            6,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable metrics mapping."""

        return asdict(self)


def write_processing_metrics_json(
    metrics: ProcessingMetrics,
    output_path: Path,
) -> None:
    """Write processing metrics without modifying production results."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
