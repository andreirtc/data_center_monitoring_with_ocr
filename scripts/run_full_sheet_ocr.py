from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path
from typing import Any

from paddleocr import TextRecognition

from datacenter_ocr.config import (
    OUTPUT_FOLDER,
    TEST_IMAGE_PATH,
)
from datacenter_ocr.diagnostic_export import write_cell_results_csv
from datacenter_ocr.image_processing import (
    load_image,
    save_image,
)
from datacenter_ocr.monitoring_records import (
    build_monitoring_rows,
)
from datacenter_ocr.ocr_processing import (
    CellOCRResult,
    process_measurement_cells_with_blank_detection,
)
from datacenter_ocr.processing_metrics import (
    ProcessingMetrics,
    write_processing_metrics_json,
)
from datacenter_ocr.sheet_processing import (
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

PROCESSING_METRICS_PATH = (
    FULL_SHEET_OCR_FOLDER
    / "processing_metrics.json"
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

    write_cell_results_csv(results, output_path)


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

    metrics = ProcessingMetrics(
        source_filename=str(TEST_IMAGE_PATH),
        uploaded_fingerprint=hashlib.sha256(
            TEST_IMAGE_PATH.read_bytes()
        ).hexdigest(),
        model_was_warm=False,
    )
    metrics.capture_process_uptime()

    print(
        "Loading monitoring-sheet image..."
    )

    decoding_start = time.perf_counter()
    original_image = load_image(
        TEST_IMAGE_PATH
    )
    metrics.upload_decoding_seconds = round(
        time.perf_counter() - decoding_start,
        6,
    )
    metrics.uploaded_height, metrics.uploaded_width = original_image.shape[:2]

    print(
        "Preparing and extracting the monitoring table..."
    )

    prepared_sheet = prepare_monitoring_sheet(
        original_image,
        metrics=metrics,
    )

    print(
        f"Extracted {len(prepared_sheet.cells)} "
        f"measurement cells."
    )

    print(
        "Loading PaddleOCR recognition model..."
    )

    model_start = time.perf_counter()
    model = TextRecognition(
        device="cpu"
    )
    metrics.model_construction_seconds = round(
        time.perf_counter() - model_start,
        6,
    )

    start_time = time.perf_counter()

    cell_results = (
        process_measurement_cells_with_blank_detection(
            model=model,
            cells=prepared_sheet.cells,
            cells_per_batch=(
                CELLS_PER_PROCESSING_BATCH
            ),
            progress_callback=report_progress,
            metrics=metrics,
        )
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    record_start = time.perf_counter()
    monitoring_rows = build_monitoring_rows(
        cell_results
    )
    metrics.monitoring_record_construction_seconds = round(
        time.perf_counter() - record_start,
        6,
    )
    metrics.recalculate_total()

    save_cell_results(
        results=cell_results,
        output_path=CELL_RESULTS_PATH,
    )

    write_processing_metrics_json(
        metrics,
        PROCESSING_METRICS_PATH,
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

    blank_cell_count = sum(
        1
        for result in cell_results
        if result.is_blank
    )

    ocr_processed_count = sum(
        1
        for result in cell_results
        if not result.is_blank
    )

    automatically_accepted_count = sum(
        1
        for result in cell_results
        if (
            not result.is_blank
            and not result.needs_review
        )
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
        f"Blank cells skipped: "
        f"{blank_cell_count}"
    )

    print(
        f"Filled cells sent to OCR: "
        f"{ocr_processed_count}"
    )

    print(
        f"Automatically accepted filled cells: "
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

    print()
    print(
        "Stage processing metrics:\n"
        f"{PROCESSING_METRICS_PATH}"
    )


if __name__ == "__main__":
    main()
