from __future__ import annotations

import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from paddleocr import TextRecognition

from datacenter_ocr.cell_preprocessing import create_ocr_variants

from datacenter_ocr.config import PROJECT_FOLDER

SAMPLE_CELLS_FOLDER = (
    PROJECT_FOLDER
    / "output"
    / "sample_cells"
)

LOCAL_BENCHMARK_FOLDER = (
    PROJECT_FOLDER
    / "local_benchmark"
)

LABELS_PATH = (
    LOCAL_BENCHMARK_FOLDER
    / "labels.csv"
)

REPORT_PATH = (
    LOCAL_BENCHMARK_FOLDER
    / "ocr_results.csv"
)

OCR_PREVIEW_FOLDER = (
    LOCAL_BENCHMARK_FOLDER
    / "ocr_inputs"
)

BATCH_SIZE = 16
OCR_SCALE = 4
OCR_PADDING = 16


def create_label_template() -> bool:
    """
    Create an empty CSV containing every sample-cell filename.

    Returns:
        True if a new template was created.
        False if the labels file already exists.
    """

    if LABELS_PATH.exists():
        return False

    LOCAL_BENCHMARK_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    cell_paths = sorted(
        SAMPLE_CELLS_FOLDER.glob("*.png")
    )

    if not cell_paths:
        raise FileNotFoundError(
            "No sample cells were found.\n"
            f"Expected location: {SAMPLE_CELLS_FOLDER}\n"
            "Run main.py first."
        )

    with LABELS_PATH.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "filename",
                "expected_value",
            ],
        )

        writer.writeheader()

        for cell_path in cell_paths:
            writer.writerow(
                {
                    "filename": cell_path.name,
                    "expected_value": "",
                }
            )

    return True


def load_labels() -> list[dict[str, str]]:
    """
    Read and validate the manually confirmed OCR answers.
    """

    with LABELS_PATH.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )

        labels = list(reader)

    if not labels:
        raise ValueError(
            "The labels file contains no rows."
        )

    missing_labels = [
        row["filename"]
        for row in labels
        if not row["expected_value"].strip()
    ]

    if missing_labels:
        missing_text = "\n".join(
            missing_labels
        )

        raise ValueError(
            "Fill in expected_value for every cell "
            "before running the benchmark:\n"
            f"{missing_text}"
        )

    return labels


def normalize_numeric_text(
    text: str,
) -> str:
    """
    Normalize common OCR character confusions while
    retaining only digits, a decimal point and minus sign.

    This does not assume that the OCR result is correct.
    It only makes comparison more consistent.
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

    # Keep only the first decimal point.
    if numeric_text.count(".") > 1:
        first_decimal_index = numeric_text.find(".")

        numeric_text = (
            numeric_text[: first_decimal_index + 1]
            + numeric_text[first_decimal_index + 1 :].replace(
                ".",
                "",
            )
        )

    if is_negative:
        numeric_text = "-" + numeric_text

    return numeric_text


def ensure_bgr(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert grayscale images into three-channel BGR images.
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
    Enlarge a cell and add white space around it.

    All preprocessing variants receive the same enlargement
    and padding so the comparison remains fair.
    """

    bgr_image = ensure_bgr(
        image
    )

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
    Convert a PaddleOCR Result object into a normal dictionary.
    """

    payload = result.json

    if callable(payload):
        payload = payload()

    if isinstance(payload, str):
        payload = json.loads(
            payload
        )

    if not isinstance(payload, dict):
        raise TypeError(
            "Unexpected PaddleOCR result format."
        )

    # PaddleOCR normally nests recognition output under "res".
    if "res" in payload:
        payload = payload["res"]

    return payload


def build_jobs(
    labels: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """
    Create one OCR job per cell and preprocessing variant.
    """

    jobs: list[dict[str, Any]] = []

    OCR_PREVIEW_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    for label_row in labels:
        filename = label_row["filename"]
        expected_value = label_row["expected_value"].strip()

        cell_path = (
            SAMPLE_CELLS_FOLDER
            / filename
        )

        cell_image = cv2.imread(
            str(cell_path)
        )

        if cell_image is None:
            raise FileNotFoundError(
                f"Could not load sample cell:\n{cell_path}"
            )

        variants = create_ocr_variants(
            cell_image
        )

        for variant_name, variant_image in variants.items():
            ocr_input = prepare_for_ocr(
                variant_image
            )

            preview_filename = (
                f"{cell_path.stem}_"
                f"{variant_name}.png"
            )

            preview_path = (
                OCR_PREVIEW_FOLDER
                / preview_filename
            )

            cv2.imwrite(
                str(preview_path),
                ocr_input,
            )

            jobs.append(
                {
                    "filename": filename,
                    "expected_value": expected_value,
                    "variant": variant_name,
                    "image": ocr_input,
                    "preview_path": preview_path,
                }
            )

    return jobs


def run_benchmark(
    jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Run PaddleOCR recognition on every prepared job.
    """

    print("Loading PaddleOCR recognition model...")

    # The model is created once and reused for every image.
    model = TextRecognition(
        device="cpu"
    )

    input_images = [
        job["image"]
        for job in jobs
    ]

    print(
        f"Recognizing {len(input_images)} "
        f"prepared cell images..."
    )

    start_time = time.perf_counter()

    prediction_results = list(
        model.predict(
            input=input_images,
            batch_size=BATCH_SIZE,
        )
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    if len(prediction_results) != len(jobs):
        raise RuntimeError(
            "The number of PaddleOCR results does not "
            "match the number of submitted images."
        )

    benchmark_results: list[dict[str, Any]] = []

    for job, prediction_result in zip(
        jobs,
        prediction_results,
    ):
        payload = extract_result_payload(
            prediction_result
        )

        raw_text = str(
            payload.get(
                "rec_text",
                "",
            )
        ).strip()

        confidence = float(
            payload.get(
                "rec_score",
                0.0,
            )
            or 0.0
        )

        normalized_prediction = normalize_numeric_text(
            raw_text
        )

        normalized_expected = normalize_numeric_text(
            job["expected_value"]
        )

        exact_match = (
            normalized_prediction
            == normalized_expected
        )

        same_digits_without_decimal = (
            not exact_match
            and normalized_prediction.replace(".", "")
            == normalized_expected.replace(".", "")
        )

        benchmark_results.append(
            {
                "filename": job["filename"],
                "variant": job["variant"],
                "expected_value": normalized_expected,
                "raw_ocr_text": raw_text,
                "normalized_prediction": normalized_prediction,
                "confidence": round(confidence, 4),
                "exact_match": exact_match,
                "decimal_only_error": same_digits_without_decimal,
                "preview_path": str(job["preview_path"]),
            }
        )

    print(
        f"OCR processing time: "
        f"{elapsed_seconds:.2f} seconds"
    )

    return benchmark_results


def save_report(
    results: list[dict[str, Any]],
) -> None:
    """Save every OCR prediction to a CSV report."""

    LOCAL_BENCHMARK_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "filename",
        "variant",
        "expected_value",
        "raw_ocr_text",
        "normalized_prediction",
        "confidence",
        "exact_match",
        "decimal_only_error",
        "preview_path",
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
            results
        )


def print_summary(
    results: list[dict[str, Any]],
) -> None:
    """
    Print accuracy statistics for each preprocessing variant.
    """

    grouped_results: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for result in results:
        grouped_results[
            result["variant"]
        ].append(result)

    print()
    print("OCR BENCHMARK SUMMARY")
    print("-" * 60)

    for variant_name in sorted(grouped_results):
        variant_results = grouped_results[
            variant_name
        ]

        total = len(
            variant_results
        )

        exact_matches = sum(
            1
            for result in variant_results
            if result["exact_match"]
        )

        decimal_errors = sum(
            1
            for result in variant_results
            if result["decimal_only_error"]
        )

        accuracy = (
            exact_matches
            / total
            * 100
        )

        average_confidence = (
            sum(
                result["confidence"]
                for result in variant_results
            )
            / total
        )

        print(
            f"{variant_name:10} | "
            f"Exact: {exact_matches:2}/{total} "
            f"({accuracy:6.2f}%) | "
            f"Decimal errors: {decimal_errors:2} | "
            f"Average confidence: "
            f"{average_confidence:.4f}"
        )

    print("-" * 60)
    print(
        f"Detailed report saved to:\n{REPORT_PATH}"
    )


def main() -> None:
    """Create labels or execute the OCR benchmark."""

    template_created = create_label_template()

    if template_created:
        print(
            "Created the OCR label template:\n"
            f"{LABELS_PATH}\n\n"
            "Open the CSV and enter the correct handwritten "
            "value for every row, then run this script again."
        )

        return

    labels = load_labels()

    jobs = build_jobs(
        labels
    )

    print(
        f"Created {len(jobs)} OCR jobs "
        f"from {len(labels)} labelled cells."
    )

    results = run_benchmark(
        jobs
    )

    save_report(
        results
    )

    print_summary(
        results
    )


if __name__ == "__main__":
    main()