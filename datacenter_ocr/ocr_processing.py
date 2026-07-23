from __future__ import annotations
from collections.abc import Callable

import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datacenter_ocr.blank_cell_detection import analyze_cell_for_blankness
from typing import Any

import cv2
import numpy as np
from paddleocr import TextRecognition

from datacenter_ocr.cell_preprocessing import create_ocr_variants
from datacenter_ocr.numeric_postprocessing import correct_numeric_prediction
from datacenter_ocr.processing_metrics import ProcessingMetrics
from datacenter_ocr.verification import verify_cell_results


PRODUCTION_VARIANTS = (
    "original",
    "grayscale",
    "contrast",
)

OCR_SCALE = 4
OCR_PADDING = 16
OCR_BATCH_SIZE = 16


@dataclass
class CellOCRResult:
    """
    Final OCR result for one monitoring-table cell.
    """

    filename: str
    day: int
    point: int
    reading_type: str

    predictions: dict[str, str]
    confidences: dict[str, float]

    consensus_prediction: str
    agreement_count: int
    average_consensus_confidence: float

    final_value: str
    needs_review: bool
    review_reason: str
    is_blank: bool = False
    blank_ink_ratio: float = 0.0
    raw_predictions: dict[str, str] = field(default_factory=dict)
    ocr_uncertainty_reasons: tuple[str, ...] = ()
    human_verified: bool = False
    verification_reasons: tuple[str, ...] = ()
    review_categories: tuple[str, ...] = ()
    blocking_errors: tuple[str, ...] = ()
    required_confirmation_reasons: tuple[str, ...] = ()
    operational_warnings: tuple[str, ...] = ()
    informational_notices: tuple[str, ...] = ()
    blocks_export: bool = False
    format_is_valid: bool = True
    within_absolute_limits: bool = True
    operational_severity: str = "not applicable"
    is_statistical_anomaly: bool = False
    has_blank_mismatch: bool = False
    ocr_uncertain: bool = False
    postprocessing_status: str = "not_recorded"
    candidate_interpretations: tuple[str, ...] = ()


def normalize_numeric_text(
    text: str,
) -> str:
    """
    Normalize common OCR character confusions while preserving
    only digits, a decimal point, and an optional minus sign.
    """

    replacements = {
        "O": "0",
        "o": "0",
        "Q": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        ",": ".",
        "·": ".",
        "•": ".",
        "_": ".",
    }

    replaced_text = "".join(
        replacements.get(character, character)
        for character in text
    )

    numeric_text = re.sub(
        pattern=r"[^0-9.\-]",
        repl="",
        string=replaced_text,
    )

    is_negative = numeric_text.startswith("-")

    numeric_text = numeric_text.replace(
        "-",
        "",
    )

    if numeric_text.count(".") > 1:
        first_decimal_index = numeric_text.find(".")

        numeric_text = (
            numeric_text[: first_decimal_index + 1]
            + numeric_text[
                first_decimal_index + 1:
            ].replace(".", "")
        )

    if is_negative:
        numeric_text = "-" + numeric_text

    return numeric_text


def ensure_bgr(
    image: np.ndarray,
) -> np.ndarray:
    """
    Ensure the image has three BGR color channels.
    """

    if len(image.shape) == 2:
        return cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )

    return image.copy()


def prepare_for_ocr(
    image: np.ndarray,
) -> np.ndarray:
    """
    Enlarge a cell and add white padding around it.
    """

    bgr_image = ensure_bgr(image)

    enlarged_image = cv2.resize(
        bgr_image,
        None,
        fx=OCR_SCALE,
        fy=OCR_SCALE,
        interpolation=cv2.INTER_CUBIC,
    )

    padded_image = cv2.copyMakeBorder(
        enlarged_image,
        OCR_PADDING,
        OCR_PADDING,
        OCR_PADDING,
        OCR_PADDING,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    return padded_image


def extract_result_payload(
    result: Any,
) -> dict[str, Any]:
    """
    Convert PaddleOCR's result object into a normal dictionary.
    """

    payload = result.json

    if callable(payload):
        payload = payload()

    if isinstance(payload, str):
        payload = json.loads(payload)

    if not isinstance(payload, dict):
        raise TypeError(
            "Unexpected PaddleOCR result format."
        )

    if "res" in payload:
        payload = payload["res"]

    return payload


def recognize_prepared_images(
    model: TextRecognition,
    images: list[np.ndarray],
    metrics: ProcessingMetrics | None = None,
) -> list[dict[str, Any]]:
    """
    Run text recognition on a batch of prepared images.
    """

    prediction_start = time.perf_counter()
    prediction_results = list(
        model.predict(
            input=images,
            batch_size=OCR_BATCH_SIZE,
        )
    )
    prediction_elapsed = time.perf_counter() - prediction_start

    if metrics is not None:
        metrics.add_seconds("ocr_prediction_seconds", prediction_elapsed)
        metrics.ocr_input_image_count += len(images)
        metrics.model_predict_call_count += 1
        metrics.requested_batch_size = OCR_BATCH_SIZE
        metrics.result_batch_count += math.ceil(
            len(images) / OCR_BATCH_SIZE
        )
        if (
            metrics.model_was_warm is False
            and metrics.first_prediction_warmup_seconds is None
        ):
            metrics.first_prediction_warmup_seconds = round(
                prediction_elapsed,
                6,
            )

    if len(prediction_results) != len(images):
        raise RuntimeError(
            "OCR returned a different number of results "
            "than the number of submitted images."
        )

    recognized_results = []

    for result in prediction_results:
        payload = extract_result_payload(result)

        raw_text = str(
            payload.get("rec_text", "")
        ).strip()

        confidence = float(
            payload.get("rec_score", 0.0)
            or 0.0
        )

        recognized_results.append(
            {
                "raw_text": raw_text,
                "normalized_text": normalize_numeric_text(
                    raw_text
                ),
                "confidence": confidence,
            }
        )

    return recognized_results


def choose_consensus(
    variant_results: list[dict[str, Any]],
) -> tuple[str, int, float, bool]:
    """
    Choose the prediction supported by the largest number
    of preprocessing variants.

    Returns:
        selected prediction
        number of supporting variants
        average confidence of supporters
        whether the result is ambiguous
    """

    vote_counts = Counter(
        result["normalized_text"]
        for result in variant_results
    )

    highest_vote_count = max(
        vote_counts.values()
    )

    winning_predictions = [
        prediction
        for prediction, count in vote_counts.items()
        if count == highest_vote_count
    ]

    ambiguous = (
        len(winning_predictions) > 1
    )

    prediction_confidences = {}

    for prediction in winning_predictions:
        matching_confidences = [
            result["confidence"]
            for result in variant_results
            if result["normalized_text"] == prediction
        ]

        prediction_confidences[prediction] = (
            sum(matching_confidences)
            / len(matching_confidences)
        )

    selected_prediction = max(
        winning_predictions,
        key=lambda prediction: (
            prediction_confidences[prediction]
        ),
    )

    selected_confidence = (
        prediction_confidences[
            selected_prediction
        ]
    )

    return (
        selected_prediction,
        highest_vote_count,
        selected_confidence,
        ambiguous,
    )


def _build_cell_result_from_variants(
    cell: dict[str, Any],
    variant_results: list[dict[str, Any]],
) -> CellOCRResult:
    """Build one result from already-recognized preprocessing variants."""

    (
        consensus_prediction,
        agreement_count,
        consensus_confidence,
        consensus_is_ambiguous,
    ) = choose_consensus(variant_results)
    correction = correct_numeric_prediction(
        prediction=consensus_prediction,
        reading_type=cell["reading_type"],
    )
    review_reasons: list[str] = []
    if consensus_is_ambiguous:
        review_reasons.append("The OCR variants produced an unresolved tie.")
    if correction.needs_review or correction.changed:
        review_reasons.append(correction.reason)

    return CellOCRResult(
        filename=cell["filename"],
        day=cell["day"],
        point=cell["point"],
        reading_type=cell["reading_type"],
        predictions={
            result["variant"]: result["normalized_text"]
            for result in variant_results
        },
        confidences={
            result["variant"]: round(result["confidence"], 4)
            for result in variant_results
        },
        raw_predictions={
            result["variant"]: result["raw_text"]
            for result in variant_results
        },
        consensus_prediction=consensus_prediction,
        agreement_count=agreement_count,
        average_consensus_confidence=round(consensus_confidence, 4),
        final_value=correction.corrected_text,
        needs_review=bool(review_reasons),
        review_reason=(
            " ".join(review_reasons) if review_reasons else correction.reason
        ),
        ocr_uncertainty_reasons=tuple(review_reasons),
        postprocessing_status=correction.status,
        candidate_interpretations=correction.candidate_interpretations,
    )


def process_measurement_cells(
    model: TextRecognition,
    cells: list[dict[str, Any]],
    metrics: ProcessingMetrics | None = None,
) -> list[CellOCRResult]:
    """
    Recognize all supplied cells and return structured results.
    """

    preprocessing_start = time.perf_counter()
    jobs = []

    for cell in cells:
        variants = create_ocr_variants(
            cell["image"]
        )

        for variant_name in PRODUCTION_VARIANTS:
            prepared_image = prepare_for_ocr(
                variants[variant_name]
            )

            jobs.append(
                {
                    "filename": cell["filename"],
                    "day": cell["day"],
                    "point": cell["point"],
                    "reading_type": cell["reading_type"],
                    "variant": variant_name,
                    "image": prepared_image,
                }
            )

    if metrics is not None:
        metrics.add_seconds(
            "ocr_preprocessing_seconds",
            time.perf_counter() - preprocessing_start,
        )

    prepared_images = [
        job["image"]
        for job in jobs
    ]

    recognized_results = recognize_prepared_images(
        model=model,
        images=prepared_images,
        metrics=metrics,
    )

    postprocessing_start = time.perf_counter()
    grouped_jobs = defaultdict(list)

    for job, recognized_result in zip(
        jobs,
        recognized_results,
    ):
        grouped_jobs[job["filename"]].append(
            {
                **job,
                **recognized_result,
            }
        )

    final_results = [
        _build_cell_result_from_variants(
            cell,
            grouped_jobs[cell["filename"]],
        )
        for cell in cells
    ]

    if metrics is not None:
        metrics.add_seconds(
            "postprocessing_seconds",
            time.perf_counter() - postprocessing_start,
        )

    verification_start = time.perf_counter()
    verified_results = verify_cell_results(final_results)
    if metrics is not None:
        metrics.add_seconds(
            "verification_seconds",
            time.perf_counter() - verification_start,
        )
    return verified_results


def process_measurement_cells_adaptive(
    model: TextRecognition,
    cells: list[dict[str, Any]],
    metrics: ProcessingMetrics | None = None,
) -> list[CellOCRResult]:
    """Create fast grayscale proposals and retry unsafe text with consensus.

    A single grayscale result is retained only when PaddleOCR returned an
    already-valid one-decimal value without character normalization or numeric
    repair. These proposals deliberately keep 1/3 agreement and therefore
    require human confirmation. Empty, malformed, normalized, or out-of-range
    first-pass text is rerun through the existing three-variant consensus path.
    """

    preprocessing_start = time.perf_counter()
    grayscale_jobs: list[dict[str, Any]] = []
    for cell in cells:
        grayscale = create_ocr_variants(cell["image"])["grayscale"]
        grayscale_jobs.append(
            {
                "cell": cell,
                "image": prepare_for_ocr(grayscale),
            }
        )
    if metrics is not None:
        metrics.add_seconds(
            "ocr_preprocessing_seconds",
            time.perf_counter() - preprocessing_start,
        )
        metrics.adaptive_first_pass_cell_count += len(cells)

    grayscale_results = recognize_prepared_images(
        model=model,
        images=[job["image"] for job in grayscale_jobs],
        metrics=metrics,
    )

    fast_results: dict[str, CellOCRResult] = {}
    fallback_cells: list[dict[str, Any]] = []
    first_pass_variants: dict[str, dict[str, Any]] = {}
    for job, recognized in zip(grayscale_jobs, grayscale_results):
        cell = job["cell"]
        first_pass_variants[cell["filename"]] = {
            "filename": cell["filename"],
            "day": cell["day"],
            "point": cell["point"],
            "reading_type": cell["reading_type"],
            "variant": "grayscale",
            **recognized,
        }
        raw_text = recognized["raw_text"]
        normalized_text = recognized["normalized_text"]
        correction = correct_numeric_prediction(
            normalized_text,
            cell["reading_type"],
        )
        safe_single_pass = (
            raw_text == normalized_text
            and correction.status == "valid_unchanged"
            and not correction.changed
            and not correction.needs_review
        )
        if not safe_single_pass:
            fallback_cells.append(cell)
            continue

        fast_results[cell["filename"]] = CellOCRResult(
            filename=cell["filename"],
            day=cell["day"],
            point=cell["point"],
            reading_type=cell["reading_type"],
            predictions={"grayscale": normalized_text},
            raw_predictions={"grayscale": raw_text},
            confidences={"grayscale": round(recognized["confidence"], 4)},
            consensus_prediction=normalized_text,
            agreement_count=1,
            average_consensus_confidence=round(recognized["confidence"], 4),
            final_value=correction.corrected_text,
            needs_review=True,
            review_reason=(
                "Fast OCR proposal used one image variant and requires "
                "confirmation against the extracted crop."
            ),
            ocr_uncertainty_reasons=(
                "Fast OCR proposal used one image variant to reduce processing "
                "time.",
            ),
            postprocessing_status=correction.status,
            candidate_interpretations=correction.candidate_interpretations,
        )

    if metrics is not None:
        metrics.adaptive_fallback_cell_count += len(fallback_cells)

    fallback_results: list[CellOCRResult] = []
    if fallback_cells:
        fallback_preprocessing_start = time.perf_counter()
        fallback_jobs: list[dict[str, Any]] = []
        for cell in fallback_cells:
            variants = create_ocr_variants(cell["image"])
            for variant_name in ("original", "contrast"):
                fallback_jobs.append(
                    {
                        "filename": cell["filename"],
                        "day": cell["day"],
                        "point": cell["point"],
                        "reading_type": cell["reading_type"],
                        "variant": variant_name,
                        "image": prepare_for_ocr(variants[variant_name]),
                    }
                )
        if metrics is not None:
            metrics.add_seconds(
                "ocr_preprocessing_seconds",
                time.perf_counter() - fallback_preprocessing_start,
            )
        recognized_fallbacks = recognize_prepared_images(
            model=model,
            images=[job["image"] for job in fallback_jobs],
            metrics=metrics,
        )
        fallback_variants: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for cell in fallback_cells:
            fallback_variants[cell["filename"]].append(
                first_pass_variants[cell["filename"]]
            )
        for job, recognized in zip(fallback_jobs, recognized_fallbacks):
            fallback_variants[job["filename"]].append(
                {**job, **recognized}
            )
        fallback_postprocessing_start = time.perf_counter()
        fallback_results = [
            _build_cell_result_from_variants(
                cell,
                fallback_variants[cell["filename"]],
            )
            for cell in fallback_cells
        ]
        if metrics is not None:
            metrics.add_seconds(
                "postprocessing_seconds",
                time.perf_counter() - fallback_postprocessing_start,
            )

    result_by_filename = {
        **fast_results,
        **{result.filename: result for result in fallback_results},
    }
    ordered_results = [result_by_filename[cell["filename"]] for cell in cells]
    return verify_cell_results(ordered_results)

def process_measurement_cells_in_batches(
    model: TextRecognition,
    cells: list[dict[str, Any]],
    cells_per_batch: int = 32,
    progress_callback: Callable[[int, int], None] | None = None,
    metrics: ProcessingMetrics | None = None,
    recognition_strategy: str = "consensus",
) -> list[CellOCRResult]:
    """
    Process a large collection of cells in smaller batches.

    This reduces memory usage and supports progress reporting
    in both command-line scripts and the future Streamlit app.
    """

    if cells_per_batch <= 0:
        raise ValueError(
            "cells_per_batch must be greater than zero."
        )
    if recognition_strategy not in {"consensus", "adaptive"}:
        raise ValueError(
            "recognition_strategy must be 'consensus' or 'adaptive'."
        )

    all_results: list[CellOCRResult] = []

    total_cells = len(cells)

    for start_index in range(
        0,
        total_cells,
        cells_per_batch,
    ):
        end_index = min(
            start_index + cells_per_batch,
            total_cells,
        )

        cell_batch = cells[
            start_index:end_index
        ]

        processor = (
            process_measurement_cells_adaptive
            if recognition_strategy == "adaptive"
            else process_measurement_cells
        )
        batch_results = processor(model=model, cells=cell_batch, metrics=metrics)

        all_results.extend(
            batch_results
        )

        processed_count = end_index

        if progress_callback is not None:
            progress_callback(
                processed_count,
                total_cells,
            )

    verification_start = time.perf_counter()
    verified_results = verify_cell_results(all_results)
    if metrics is not None:
        metrics.add_seconds(
            "verification_seconds",
            time.perf_counter() - verification_start,
        )
    return verified_results

def process_measurement_cells_with_blank_detection(
    model: TextRecognition,
    cells: list[dict[str, Any]],
    cells_per_batch: int = 32,
    progress_callback: Callable[[int, int], None] | None = None,
    metrics: ProcessingMetrics | None = None,
    recognition_strategy: str = "consensus",
) -> list[CellOCRResult]:
    """
    Classify blank cells before OCR.

    Blank cells receive an empty final value and are not submitted
    to PaddleOCR. Filled cells continue through the normal
    multi-variant recognition pipeline.
    """

    if not cells:
        if metrics is not None:
            metrics.total_cell_count = 0
            metrics.filled_cell_count = 0
            metrics.blank_cell_count = 0
        return []

    if recognition_strategy not in {"consensus", "adaptive"}:
        raise ValueError(
            "recognition_strategy must be 'consensus' or 'adaptive'."
        )
    if metrics is not None:
        metrics.recognition_strategy = recognition_strategy

    total_cells = len(cells)

    filled_cells = []
    blank_results: dict[str, CellOCRResult] = {}
    ink_ratios: dict[str, float] = {}

    blank_detection_start = time.perf_counter()

    for cell in cells:
        blank_analysis = analyze_cell_for_blankness(
            cell["image"]
        )

        filename = cell["filename"]

        ink_ratios[filename] = round(
            blank_analysis.ink_ratio,
            6,
        )

        if not blank_analysis.is_blank:
            filled_cells.append(cell)
            continue

        blank_results[filename] = CellOCRResult(
            filename=filename,
            day=cell["day"],
            point=cell["point"],
            reading_type=cell["reading_type"],

            predictions={
                "original": "",
                "grayscale": "",
                "contrast": "",
            },

            raw_predictions={
                "original": "",
                "grayscale": "",
                "contrast": "",
            },

            confidences={
                "original": 0.0,
                "grayscale": 0.0,
                "contrast": 0.0,
            },

            consensus_prediction="",
            agreement_count=0,
            average_consensus_confidence=0.0,

            final_value="",
            needs_review=False,
            review_reason="Cell classified as blank.",

            is_blank=True,
            blank_ink_ratio=ink_ratios[filename],
            postprocessing_status="skipped_blank",
        )

    blank_count = len(blank_results)

    if metrics is not None:
        metrics.add_seconds(
            "blank_detection_seconds",
            time.perf_counter() - blank_detection_start,
        )
        metrics.total_cell_count = total_cells
        metrics.blank_cell_count = blank_count
        metrics.filled_cell_count = len(filled_cells)

    print(
        f"Blank-cell detector skipped "
        f"{blank_count}/{total_cells} cells."
    )

    print(
        f"Submitting {len(filled_cells)} "
        f"filled cells to OCR."
    )

    # Blank cells are already considered processed.
    if progress_callback is not None:
        progress_callback(
            blank_count,
            total_cells,
        )

    def report_filled_progress(
        processed_filled_count: int,
        total_filled_count: int,
    ) -> None:
        """
        Convert filled-cell progress into whole-sheet progress.
        """

        del total_filled_count

        if progress_callback is not None:
            progress_callback(
                blank_count + processed_filled_count,
                total_cells,
            )

    filled_results = (
        process_measurement_cells_in_batches(
            model=model,
            cells=filled_cells,
            cells_per_batch=cells_per_batch,
            progress_callback=report_filled_progress,
            metrics=metrics,
            recognition_strategy=recognition_strategy,
        )
    )

    result_by_filename = dict(
        blank_results
    )

    for result in filled_results:
        result_with_blank_score = replace(
            result,
            is_blank=False,
            blank_ink_ratio=ink_ratios[
                result.filename
            ],
        )

        result_by_filename[
            result.filename
        ] = result_with_blank_score

    # Restore the original table-cell order.
    ordered_results = []

    for cell in cells:
        filename = cell["filename"]

        if filename not in result_by_filename:
            raise RuntimeError(
                f"No result was generated for {filename}."
            )

        ordered_results.append(
            result_by_filename[filename]
        )

    if len(ordered_results) != total_cells:
        raise RuntimeError(
            "The number of final results does not match "
            "the number of extracted cells."
        )

    verification_start = time.perf_counter()
    verified_results = verify_cell_results(ordered_results)
    if metrics is not None:
        metrics.add_seconds(
            "verification_seconds",
            time.perf_counter() - verification_start,
        )
    return verified_results
