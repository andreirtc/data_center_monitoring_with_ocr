from __future__ import annotations
from collections.abc import Callable

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datacenter_ocr.blank_cell_detection import analyze_cell_for_blankness
from typing import Any

import cv2
import numpy as np
from paddleocr import TextRecognition

from datacenter_ocr.cell_preprocessing import create_ocr_variants
from datacenter_ocr.numeric_postprocessing import correct_numeric_prediction


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
) -> list[dict[str, Any]]:
    """
    Run text recognition on a batch of prepared images.
    """

    prediction_results = list(
        model.predict(
            input=images,
            batch_size=OCR_BATCH_SIZE,
        )
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


def process_measurement_cells(
    model: TextRecognition,
    cells: list[dict[str, Any]],
) -> list[CellOCRResult]:
    """
    Recognize all supplied cells and return structured results.
    """

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

    prepared_images = [
        job["image"]
        for job in jobs
    ]

    recognized_results = recognize_prepared_images(
        model=model,
        images=prepared_images,
    )

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

    final_results = []

    for cell in cells:
        filename = cell["filename"]
        variant_results = grouped_jobs[filename]

        (
            consensus_prediction,
            agreement_count,
            consensus_confidence,
            consensus_is_ambiguous,
        ) = choose_consensus(
            variant_results
        )

        correction = correct_numeric_prediction(
            prediction=consensus_prediction,
            reading_type=cell["reading_type"],
        )

        review_reasons = []

        if consensus_is_ambiguous:
            review_reasons.append(
                "The OCR variants produced an unresolved tie."
            )

        if correction.needs_review:
            review_reasons.append(
                correction.reason
            )

        needs_review = bool(
            review_reasons
        )

        if needs_review:
            final_value = correction.corrected_text
            review_reason = " ".join(
                review_reasons
            )
        else:
            final_value = correction.corrected_text
            review_reason = correction.reason

        predictions = {
            result["variant"]: result["normalized_text"]
            for result in variant_results
        }

        confidences = {
            result["variant"]: round(
                result["confidence"],
                4,
            )
            for result in variant_results
        }

        final_results.append(
            CellOCRResult(
                filename=filename,
                day=cell["day"],
                point=cell["point"],
                reading_type=cell["reading_type"],
                predictions=predictions,
                confidences=confidences,
                consensus_prediction=consensus_prediction,
                agreement_count=agreement_count,
                average_consensus_confidence=round(
                    consensus_confidence,
                    4,
                ),
                final_value=final_value,
                needs_review=needs_review,
                review_reason=review_reason,
            )
        )

    return final_results

def process_measurement_cells_in_batches(
    model: TextRecognition,
    cells: list[dict[str, Any]],
    cells_per_batch: int = 32,
    progress_callback: Callable[[int, int], None] | None = None,
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

        batch_results = process_measurement_cells(
            model=model,
            cells=cell_batch,
        )

        all_results.extend(
            batch_results
        )

        processed_count = end_index

        if progress_callback is not None:
            progress_callback(
                processed_count,
                total_cells,
            )

    return all_results

def process_measurement_cells_with_blank_detection(
    model: TextRecognition,
    cells: list[dict[str, Any]],
    cells_per_batch: int = 32,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[CellOCRResult]:
    """
    Classify blank cells before OCR.

    Blank cells receive an empty final value and are not submitted
    to PaddleOCR. Filled cells continue through the normal
    multi-variant recognition pipeline.
    """

    if not cells:
        return []

    total_cells = len(cells)

    filled_cells = []
    blank_results: dict[str, CellOCRResult] = {}
    ink_ratios: dict[str, float] = {}

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
        )

    blank_count = len(blank_results)

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

    return ordered_results