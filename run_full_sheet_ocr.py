from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from paddleocr import TextRecognition

from config import (
    OUTPUT_FOLDER,
    TEST_IMAGE_PATH,
)
from image_processing import (
    load_image,
    save_image,
)
from monitoring_records import (
    build_monitoring_rows,
)
from ocr_processing import (
    CellOCRResult,
    process_measurement_cells_in_batches,
)
from sheet_processing import (
    prepare_monitoring_sheet,
)


FULL_SHEET_OCR_FOLDER = (
    OUTPUT_FOLDER
    / "full_sheet_ocr"
)

CELL_RESULTS_PATH = (
    FULL_SHEET_OCR_FOLDER
    / "cell_results.csv"
)

MONITORING_ROWS_PATH = (
    FULL_SHEET_OCR_FOLDER
    / "monitoring_rows.csv"
)

REVIEW_CELLS_FOLDER = (
    FULL_SHEET_OCR_FOLDER
    / "review_cells"
)

CELLS_PER_PROCESSING_BATCH = 32


def report_progress(
    processed_count: int,
    total_count: int,
) -> None:
    """Print OCR progress after every cell batch."""

    percentage = (
        processed_count
        / total_count
        * 100
    )

    print(
        f"OCR progress: "
        f"{processed_count}/{total_count} cells "
        f"({percentage:.1f}%)"
    )


def save_cell_results(
    results: list[CellOCRResult],
    output_path: Path,
) -> None:
    """
    Save detailed OCR information for every individual cell.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "filename",
        "day",
        "point",
        "reading_type",

        "original_prediction",
        "grayscale_prediction",
        "contrast_prediction",

        "original_confidence",
        "grayscale_confidence",
        "contrast_confidence",

        "consensus_prediction",
        "agreement_count",
        "average_consensus_confidence",

        "final_value",
        "needs_review",
        "review_reason",
    ]

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "filename": result.filename,
                    "day": result.day,
                    "point": result.point,
                    "reading_type": result.reading_type,

                    "original_prediction": (
                        result.predictions["original"]
                    ),
                    "grayscale_prediction": (
                        result.predictions["grayscale"]
                    ),
                    "contrast_prediction": (
                        result.predictions["contrast"]
                    ),

                    "original_confidence": (
                        result.confidences["original"]
                    ),
                    "grayscale_confidence": (
                        result.confidences["grayscale"]
                    ),
                    "contrast_confidence": (
                        result.confidences["contrast"]
                    ),

                    "consensus_prediction": (
                        result.consensus_prediction
                    ),
                    "agreement_count": (
                        result.agreement_count
                    ),
                    "average_consensus_confidence": (
                        result.average_consensus_confidence
                    ),

                    "final_value": result.final_value,
                    "needs_review": result.needs_review,
                    "review_reason": result.review_reason,
                }
            )


def save_monitoring_rows(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save the combined 248 day-and-point records.
    """

    if not rows:
        raise ValueError(
            "There are no monitoring rows to save."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def save_review_cells(
    results: list[CellOCRResult],
    cells: list[dict[str, Any]],
) -> int:
    """
    Save only the crop images requiring manual review.
    """

    cell_images = {
        cell["filename"]: cell["image"]
        for cell in cells
    }

    saved_count = 0

    for result in results:
        if not result.needs_review:
            continue

        cell_image = cell_images[
            result.filename
        ]

        save_image(
            cell_image,
            REVIEW_CELLS_FOLDER
            / result.filename,
        )

        saved_count += 1

    return saved_count


def main() -> None:
    """
    Run production OCR on all 496 monitoring cells.
    """

    print(
        "Loading monitoring-sheet image..."
    )

    original_image = load_image(
        TEST_IMAGE_PATH
    )

    print(
        "Preparing and extracting the monitoring table..."
    )

    prepared_sheet = prepare_monitoring_sheet(
        original_image
    )

    print(
        f"Extracted {len(prepared_sheet.cells)} "
        f"measurement cells."
    )

    print(
        "Loading PaddleOCR recognition model..."
    )

    model = TextRecognition(
        device="cpu"
    )

    start_time = time.perf_counter()

    cell_results = (
        process_measurement_cells_in_batches(
            model=model,
            cells=prepared_sheet.cells,
            cells_per_batch=(
                CELLS_PER_PROCESSING_BATCH
            ),
            progress_callback=report_progress,
        )
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    monitoring_rows = build_monitoring_rows(
        cell_results
    )

    save_cell_results(
        results=cell_results,
        output_path=CELL_RESULTS_PATH,
    )

    save_monitoring_rows(
        rows=monitoring_rows,
        output_path=MONITORING_ROWS_PATH,
    )

    save_image(
        prepared_sheet.warped_table,
        FULL_SHEET_OCR_FOLDER
        / "warped_table.png",
    )

    save_image(
        prepared_sheet.measurement_grid_overlay,
        FULL_SHEET_OCR_FOLDER
        / "measurement_grid_overlay.png",
    )

    review_cell_count = save_review_cells(
        results=cell_results,
        cells=prepared_sheet.cells,
    )

    automatically_accepted_count = sum(
        1
        for result in cell_results
        if not result.needs_review
    )

    review_row_count = sum(
        1
        for row in monitoring_rows
        if row["needs_review"]
    )

    print()
    print("FULL-SHEET OCR SUMMARY")
    print("-" * 70)

    print(
        f"Total cell results: "
        f"{len(cell_results)}"
    )

    print(
        f"Automatically accepted cells: "
        f"{automatically_accepted_count}"
    )

    print(
        f"Cells requiring review: "
        f"{review_cell_count}"
    )

    print(
        f"Structured monitoring rows: "
        f"{len(monitoring_rows)}"
    )

    print(
        f"Rows containing a review item: "
        f"{review_row_count}"
    )

    print(
        f"OCR processing time: "
        f"{elapsed_seconds:.2f} seconds"
    )

    print()
    print(
        "Detailed cell results:\n"
        f"{CELL_RESULTS_PATH}"
    )

    print()
    print(
        "Structured monitoring rows:\n"
        f"{MONITORING_ROWS_PATH}"
    )

    print()
    print(
        "Review-cell images:\n"
        f"{REVIEW_CELLS_FOLDER}"
    )


if __name__ == "__main__":
    main()