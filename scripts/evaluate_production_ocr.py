from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import Any

import cv2
from paddleocr import TextRecognition

from datacenter_ocr.ocr_processing import process_measurement_cells

from datacenter_ocr.config import PROJECT_FOLDER

SAMPLE_CELLS_FOLDER = (
    PROJECT_FOLDER
    / "output"
    / "sample_cells"
)

LABELS_PATH = (
    PROJECT_FOLDER
    / "local_benchmark"
    / "labels.csv"
)

REPORT_PATH = (
    PROJECT_FOLDER
    / "local_benchmark"
    / "production_ocr_results.csv"
)


CELL_FILENAME_PATTERN = re.compile(
    r"day_(\d+)_point_(\d+)_(temperature|humidity)\.png"
)


def load_expected_values() -> dict[str, str]:
    """
    Load manually verified values from labels.csv.
    """

    with LABELS_PATH.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        return {
            row["filename"]: row["expected_value"].strip()
            for row in reader
        }


def load_sample_cells() -> list[dict[str, Any]]:
    """
    Load sample images and derive their metadata from filenames.
    """

    cells = []

    for cell_path in sorted(
        SAMPLE_CELLS_FOLDER.glob("*.png")
    ):
        filename_match = CELL_FILENAME_PATTERN.fullmatch(
            cell_path.name
        )

        if filename_match is None:
            print(
                f"Skipping unexpected filename: "
                f"{cell_path.name}"
            )
            continue

        day_text, point_text, reading_type = (
            filename_match.groups()
        )

        cell_image = cv2.imread(
            str(cell_path)
        )

        if cell_image is None:
            raise FileNotFoundError(
                f"Could not load cell:\n{cell_path}"
            )

        cells.append(
            {
                "filename": cell_path.name,
                "day": int(day_text),
                "point": int(point_text),
                "reading_type": reading_type,
                "image": cell_image,
            }
        )

    if not cells:
        raise FileNotFoundError(
            "No sample cells were found.\n"
            f"Expected location: {SAMPLE_CELLS_FOLDER}"
        )

    return cells


def save_report(
    evaluated_rows: list[dict[str, Any]],
) -> None:
    """
    Save the production OCR evaluation.
    """

    fieldnames = [
        "filename",
        "expected_value",
        "original_prediction",
        "grayscale_prediction",
        "contrast_prediction",
        "consensus_prediction",
        "agreement_count",
        "average_consensus_confidence",
        "final_value",
        "needs_review",
        "review_reason",
        "final_value_correct",
    ]

    with REPORT_PATH.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            evaluated_rows
        )


def main() -> None:
    """
    Evaluate the production OCR processor.
    """

    expected_values = load_expected_values()
    cells = load_sample_cells()

    print(
        f"Loaded {len(cells)} sample cells."
    )

    print(
        "Loading PaddleOCR recognition model..."
    )

    model = TextRecognition(
        device="cpu"
    )

    start_time = time.perf_counter()

    results = process_measurement_cells(
        model=model,
        cells=cells,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    evaluated_rows = []

    for result in results:
        expected_value = expected_values[
            result.filename
        ]

        final_value_correct = (
            result.final_value
            == expected_value
        )

        evaluated_rows.append(
            {
                "filename": result.filename,
                "expected_value": expected_value,
                "original_prediction": result.predictions[
                    "original"
                ],
                "grayscale_prediction": result.predictions[
                    "grayscale"
                ],
                "contrast_prediction": result.predictions[
                    "contrast"
                ],
                "consensus_prediction": (
                    result.consensus_prediction
                ),
                "agreement_count": result.agreement_count,
                "average_consensus_confidence": (
                    result.average_consensus_confidence
                ),
                "final_value": result.final_value,
                "needs_review": result.needs_review,
                "review_reason": result.review_reason,
                "final_value_correct": final_value_correct,
            }
        )

    automatically_accepted = [
        row
        for row in evaluated_rows
        if not row["needs_review"]
    ]

    correct_automatic_acceptances = [
        row
        for row in automatically_accepted
        if row["final_value_correct"]
    ]

    unsafe_automatic_acceptances = [
        row
        for row in automatically_accepted
        if not row["final_value_correct"]
    ]

    review_rows = [
        row
        for row in evaluated_rows
        if row["needs_review"]
    ]

    total_correct = sum(
        1
        for row in evaluated_rows
        if row["final_value_correct"]
    )

    print()
    print("PRODUCTION OCR EVALUATION")
    print("-" * 70)

    print(
        f"Total final-value accuracy: "
        f"{total_correct}/{len(evaluated_rows)} "
        f"({total_correct / len(evaluated_rows) * 100:.2f}%)"
    )

    print(
        f"Automatically accepted: "
        f"{len(automatically_accepted)}/"
        f"{len(evaluated_rows)}"
    )

    if automatically_accepted:
        print(
            f"Correct automatic acceptances: "
            f"{len(correct_automatic_acceptances)}/"
            f"{len(automatically_accepted)} "
            f"("
            f"{len(correct_automatic_acceptances) / len(automatically_accepted) * 100:.2f}"
            f"%)"
        )

    print(
        f"Unsafe automatic acceptances: "
        f"{len(unsafe_automatic_acceptances)}"
    )

    print(
        f"Manual review required: "
        f"{len(review_rows)}/"
        f"{len(evaluated_rows)}"
    )

    print(
        f"OCR processing time: "
        f"{elapsed_seconds:.2f} seconds"
    )

    print()
    print("REVIEW AND INCORRECT ROWS")
    print("-" * 70)

    for row in evaluated_rows:
        if (
            row["needs_review"]
            or not row["final_value_correct"]
        ):
            print(
                f"{row['filename']} | "
                f"Expected: {row['expected_value']} | "
                f"Final: {row['final_value']} | "
                f"Agreement: {row['agreement_count']} | "
                f"Review: {row['needs_review']}"
            )

            print(
                f"  Predictions: "
                f"original={row['original_prediction']}, "
                f"grayscale={row['grayscale_prediction']}, "
                f"contrast={row['contrast_prediction']}"
            )

            print(
                f"  Reason: {row['review_reason']}"
            )

    save_report(
        evaluated_rows
    )

    print()
    print(
        "Detailed report saved to:\n"
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()