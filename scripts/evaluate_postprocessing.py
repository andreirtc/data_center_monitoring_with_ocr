from __future__ import annotations

import csv
from pathlib import Path

from datacenter_ocr.numeric_postprocessing import (
    correct_numeric_prediction,
    infer_reading_type,
)

from datacenter_ocr.config import PROJECT_FOLDER

ENSEMBLE_REPORT_PATH = (
    PROJECT_FOLDER
    / "local_benchmark"
    / "ensemble_analysis.csv"
)

POSTPROCESSING_REPORT_PATH = (
    PROJECT_FOLDER
    / "local_benchmark"
    / "postprocessing_analysis.csv"
)


def text_to_bool(value: str) -> bool:
    """Convert CSV Boolean text into a Python bool."""

    return value.strip().lower() == "true"


def load_ensemble_results() -> list[dict[str, str]]:
    """Load the previously generated ensemble report."""

    if not ENSEMBLE_REPORT_PATH.exists():
        raise FileNotFoundError(
            "Ensemble report not found:\n"
            f"{ENSEMBLE_REPORT_PATH}\n"
            "Run analyze_ocr_results.py first."
        )

    with ENSEMBLE_REPORT_PATH.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        return list(reader)


def evaluate_rows(
    ensemble_rows: list[dict[str, str]],
) -> list[dict]:
    """
    Apply numeric post-processing to each consensus prediction.
    """

    evaluated_rows = []

    for row in ensemble_rows:
        filename = row["filename"]
        expected_value = row["expected_value"]
        consensus_prediction = row[
            "consensus_prediction"
        ]

        reading_type = infer_reading_type(
            filename
        )

        correction = correct_numeric_prediction(
            prediction=consensus_prediction,
            reading_type=reading_type,
        )

        corrected_prediction = (
            correction.corrected_text
        )

        corrected_is_exact = (
            corrected_prediction
            == expected_value
        )

        evaluated_rows.append(
            {
                "filename": filename,
                "reading_type": reading_type,
                "expected_value": expected_value,
                "consensus_prediction": consensus_prediction,
                "consensus_correct": text_to_bool(
                    row["consensus_correct"]
                ),
                "corrected_prediction": corrected_prediction,
                "correction_changed": correction.changed,
                "needs_review": correction.needs_review,
                "safe_to_accept": not correction.needs_review,
                "correction_reason": correction.reason,
                "corrected_is_exact": corrected_is_exact,
            }
        )

    return evaluated_rows


def print_summary(
    rows: list[dict],
) -> None:
    """Print accuracy before and after numeric correction."""

    total = len(rows)

    consensus_correct = sum(
        1
        for row in rows
        if row["consensus_correct"]
    )

    corrected_correct = sum(
        1
        for row in rows
        if row["corrected_is_exact"]
    )

    changed_count = sum(
        1
        for row in rows
        if row["correction_changed"]
    )

    review_count = sum(
        1
        for row in rows
        if row["needs_review"]
    )

    automatically_accepted_rows = [
        row
        for row in rows
        if not row["needs_review"]
    ]

    automatically_accepted_count = len(
        automatically_accepted_rows
    )

    correct_auto_accepts = sum(
        1
        for row in automatically_accepted_rows
        if row["corrected_is_exact"]
    )

    unsafe_auto_accepts = sum(
        1
        for row in automatically_accepted_rows
        if not row["corrected_is_exact"]
    )

    print(
        f"Automatically accepted: "
        f"{automatically_accepted_count}/{total}"
    )

    if automatically_accepted_count:
        auto_accept_accuracy = (
            correct_auto_accepts
            / automatically_accepted_count
            * 100
        )

        print(
            f"Correct among automatically accepted: "
            f"{correct_auto_accepts}/"
            f"{automatically_accepted_count} "
            f"({auto_accept_accuracy:.2f}%)"
        )

    print(
        f"Unsafe automatic acceptances: "
        f"{unsafe_auto_accepts}"
    )

    print()
    print("NUMERIC POST-PROCESSING SUMMARY")
    print("-" * 70)

    print(
        f"Consensus accuracy: "
        f"{consensus_correct}/{total} "
        f"({consensus_correct / total * 100:.2f}%)"
    )

    print(
        f"After correction: "
        f"{corrected_correct}/{total} "
        f"({corrected_correct / total * 100:.2f}%)"
    )

    print(
        f"Automatically changed: "
        f"{changed_count}/{total}"
    )

    print(
        f"Still requiring review: "
        f"{review_count}/{total}"
    )

    print()
    print("CORRECTIONS AND REMAINING ERRORS")
    print("-" * 70)

    for row in rows:
        if (
            row["correction_changed"]
            or not row["corrected_is_exact"]
            or row["needs_review"]
        ):
            print(
                f"{row['filename']} | "
                f"Expected: {row['expected_value']} | "
                f"Consensus: {row['consensus_prediction']} | "
                f"After: {row['corrected_prediction']} | "
                f"Review: {row['needs_review']}"
            )

            print(
                f"  Reason: "
                f"{row['correction_reason']}"
            )


def save_report(
    rows: list[dict],
) -> None:
    """Save detailed post-processing results."""

    if not rows:
        raise ValueError(
            "There are no post-processing rows to save."
        )

    fieldnames = list(
        rows[0].keys()
    )

    with POSTPROCESSING_REPORT_PATH.open(
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


def main() -> None:
    """Evaluate monitoring-specific OCR corrections."""

    ensemble_rows = load_ensemble_results()

    evaluated_rows = evaluate_rows(
        ensemble_rows
    )

    print_summary(
        evaluated_rows
    )

    save_report(
        evaluated_rows
    )

    print()
    print(
        "Detailed report saved to:\n"
        f"{POSTPROCESSING_REPORT_PATH}"
    )


if __name__ == "__main__":
    main()