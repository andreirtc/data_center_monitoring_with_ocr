from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from datacenter_ocr.config import PROJECT_FOLDER

REPORT_PATH = (
    PROJECT_FOLDER
    / "local_benchmark"
    / "ocr_results.csv"
)

ANALYSIS_PATH = (
    PROJECT_FOLDER
    / "local_benchmark"
    / "ensemble_analysis.csv"
)


def load_results() -> list[dict]:
    """Load the detailed OCR benchmark report."""

    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            "OCR report not found.\n"
            f"Expected location: {REPORT_PATH}\n"
            "Run ocr_benchmark.py first."
        )

    with REPORT_PATH.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def text_to_bool(value: str) -> bool:
    """Convert CSV text such as 'True' into a Boolean value."""

    return value.strip().lower() == "true"


def choose_consensus_prediction(
    cell_results: list[dict],
) -> tuple[str, int]:
    """
    Select the prediction supported by the most variants.

    When two predictions have the same vote count, choose the one
    with the higher average OCR confidence.
    """

    predictions = [
        result["normalized_prediction"]
        for result in cell_results
    ]

    vote_counts = Counter(predictions)

    highest_vote_count = max(
        vote_counts.values()
    )

    tied_predictions = [
        prediction
        for prediction, count in vote_counts.items()
        if count == highest_vote_count
    ]

    if len(tied_predictions) == 1:
        return tied_predictions[0], highest_vote_count

    average_confidence = {}

    for prediction in tied_predictions:
        matching_confidences = [
            float(result["confidence"])
            for result in cell_results
            if result["normalized_prediction"] == prediction
        ]

        average_confidence[prediction] = (
            sum(matching_confidences)
            / len(matching_confidences)
        )

    selected_prediction = max(
        tied_predictions,
        key=lambda prediction: average_confidence[prediction],
    )

    return selected_prediction, highest_vote_count


def analyze_results(
    results: list[dict],
) -> list[dict]:
    """Compare preprocessing variants for each labelled cell."""

    grouped_by_filename = defaultdict(list)

    for result in results:
        grouped_by_filename[
            result["filename"]
        ].append(result)

    analysis_rows = []

    for filename in sorted(grouped_by_filename):
        cell_results = grouped_by_filename[filename]

        expected_value = cell_results[0][
            "expected_value"
        ]

        any_variant_correct = any(
            text_to_bool(result["exact_match"])
            for result in cell_results
        )

        consensus_prediction, vote_count = (
            choose_consensus_prediction(cell_results)
        )

        consensus_correct = (
            consensus_prediction
            == expected_value
        )

        row = {
            "filename": filename,
            "expected_value": expected_value,
            "any_variant_correct": any_variant_correct,
            "consensus_prediction": consensus_prediction,
            "consensus_vote_count": vote_count,
            "consensus_correct": consensus_correct,
        }

        for result in cell_results:
            variant = result["variant"]

            row[f"{variant}_prediction"] = (
                result["normalized_prediction"]
            )

            row[f"{variant}_confidence"] = (
                result["confidence"]
            )

            row[f"{variant}_correct"] = (
                result["exact_match"]
            )

        analysis_rows.append(row)

    return analysis_rows


def print_summary(
    benchmark_results: list[dict],
    analysis_rows: list[dict],
) -> None:
    """Print variant, ensemble, and confidence statistics."""

    variants = sorted(
        {
            result["variant"]
            for result in benchmark_results
        }
    )

    print()
    print("VARIANT ACCURACY")
    print("-" * 70)

    for variant in variants:
        variant_results = [
            result
            for result in benchmark_results
            if result["variant"] == variant
        ]

        correct_results = [
            result
            for result in variant_results
            if text_to_bool(result["exact_match"])
        ]

        incorrect_results = [
            result
            for result in variant_results
            if not text_to_bool(result["exact_match"])
        ]

        correct_count = len(correct_results)
        total_count = len(variant_results)

        average_correct_confidence = (
            sum(
                float(result["confidence"])
                for result in correct_results
            )
            / correct_count
            if correct_results
            else 0
        )

        average_incorrect_confidence = (
            sum(
                float(result["confidence"])
                for result in incorrect_results
            )
            / len(incorrect_results)
            if incorrect_results
            else 0
        )

        print(
            f"{variant:10} | "
            f"{correct_count:2}/{total_count} correct | "
            f"correct confidence: "
            f"{average_correct_confidence:.4f} | "
            f"wrong confidence: "
            f"{average_incorrect_confidence:.4f}"
        )

    total_cells = len(analysis_rows)

    any_variant_correct_count = sum(
        1
        for row in analysis_rows
        if row["any_variant_correct"]
    )

    consensus_correct_count = sum(
        1
        for row in analysis_rows
        if row["consensus_correct"]
    )

    print()
    print("ENSEMBLE ANALYSIS")
    print("-" * 70)

    print(
        "At least one variant correct: "
        f"{any_variant_correct_count}/{total_cells} "
        f"({any_variant_correct_count / total_cells * 100:.2f}%)"
    )

    print(
        "Consensus prediction correct: "
        f"{consensus_correct_count}/{total_cells} "
        f"({consensus_correct_count / total_cells * 100:.2f}%)"
    )

    print()
    print("DISAGREEMENTS")
    print("-" * 70)

    for row in analysis_rows:
        predictions = []

        for key, value in row.items():
            if key.endswith("_prediction") and key != "consensus_prediction":
                predictions.append(value)

        unique_predictions = set(predictions)

        if len(unique_predictions) > 1:
            print(
                f"{row['filename']} | "
                f"Expected: {row['expected_value']} | "
                f"Consensus: {row['consensus_prediction']} "
                f"({row['consensus_vote_count']} votes)"
            )


def save_analysis(
    analysis_rows: list[dict],
) -> None:
    """Save the cell-by-cell ensemble analysis."""

    if not analysis_rows:
        raise ValueError(
            "There are no analysis rows to save."
        )

    fieldnames = list(
        analysis_rows[0].keys()
    )

    with ANALYSIS_PATH.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(analysis_rows)


def main() -> None:
    """Run the OCR-result analysis."""

    benchmark_results = load_results()

    analysis_rows = analyze_results(
        benchmark_results
    )

    print_summary(
        benchmark_results,
        analysis_rows,
    )

    save_analysis(
        analysis_rows
    )

    print()
    print(
        "Detailed ensemble report saved to:\n"
        f"{ANALYSIS_PATH}"
    )


if __name__ == "__main__":
    main()