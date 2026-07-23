from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from uuid import uuid4

import cv2
import numpy as np

from datacenter_ocr.geometry_benchmark import (
    COMPARISON_FIELDNAMES,
    LABEL_FIELDNAMES,
    RESULT_FIELDNAMES,
    assess_calibrated_safety,
    benchmark_identities,
    calculate_metrics,
    compare_geometry_rows,
    create_label_rows,
    evaluate_prediction_result,
    load_mode_cells,
    load_trusted_labels,
    summarize_mode,
    validate_complete_labels,
    write_csv_rows,
)
from scripts.run_geometry_ocr_benchmark import (
    production_ocr_settings,
    run_benchmark,
)


@contextmanager
def _workspace_scratch() -> object:
    """Use an ignored workspace folder; Windows sandbox ACLs reject tempfile."""

    scratch_root = Path.cwd() / "output" / "test_geometry_benchmark"
    scratch_path = scratch_root / uuid4().hex
    scratch_path.mkdir(parents=True)
    try:
        yield scratch_path
    finally:
        shutil.rmtree(scratch_path, ignore_errors=True)


def _complete_label(identity: object, root: Path) -> dict[str, object]:
    return {
        "item_number": identity.item_number,
        "sheet_id": identity.sheet_id,
        "source_image": identity.source_image,
        "filename": identity.filename,
        "day": identity.day,
        "point": identity.point,
        "reading_type": identity.reading_type,
        "expected_value": "22.0" if identity.reading_type == "temperature" else "55.0",
        "expected_blank": False,
        "fixed_crop_path": "fixed.png",
        "calibrated_crop_path": "calibrated.png",
        "context_crop_path": "context.png",
        "notes": "",
        "root": str(root),
    }


def _metric_row(
    *,
    item_number: int,
    sheet_id: str,
    day_group: str,
    point_group: str,
    reading_type: str = "temperature",
    correct: bool = True,
    needs_review: bool = False,
    unsafe: bool = False,
    expected_blank: bool = False,
    predicted_blank: bool = False,
    alignment_failure: bool = False,
) -> dict[str, object]:
    automatic = not needs_review
    return {
        "item_number": item_number,
        "sheet_id": sheet_id,
        "filename": f"cell_{item_number}.png",
        "day": {"top": 1, "middle": 16, "bottom": 31}[day_group],
        "point": {"left": 1, "center": 4, "right": 8}[point_group],
        "reading_type": reading_type,
        "day_group": day_group,
        "point_group": point_group,
        "geometry_mode": "fixed",
        "expected_value": "" if expected_blank else "22.0",
        "expected_blank": expected_blank,
        "predicted_blank": predicted_blank,
        "final_verified_value": "" if predicted_blank else ("22.0" if correct else "23.0"),
        "final_value_correct": correct,
        "consensus_correct": correct,
        "postprocessing_correct": correct,
        "original_correct": correct,
        "grayscale_correct": correct,
        "contrast_correct": correct,
        "needs_review": needs_review,
        "automatic_acceptance": automatic,
        "correct_automatic_acceptance": automatic and correct,
        "unsafe_automatic_acceptance": unsafe,
        "crop_alignment_failure": alignment_failure,
    }


class GeometryBenchmarkSelectionTests(unittest.TestCase):
    def test_selection_has_expected_stable_identities(self) -> None:
        identities = benchmark_identities()
        self.assertEqual(54, len(identities))
        self.assertEqual(tuple(range(1, 55)), tuple(item.item_number for item in identities))
        self.assertEqual(
            "sample__day_01_point_01_temperature.png", identities[0].filename
        )
        self.assertEqual(
            "may_2026__day_31_point_08_humidity.png", identities[-1].filename
        )
        self.assertEqual(54, len({item.filename for item in identities}))

    def test_label_template_reuses_only_exact_sheet_cell_matches(self) -> None:
        identities = benchmark_identities()
        trusted = {
            ("sample", "day_01_point_01_temperature.png"): ("20.0", False)
        }
        rows, reused, missing = create_label_rows(identities, trusted)
        self.assertEqual(1, reused)
        self.assertEqual(53, missing)
        self.assertEqual("20.0", rows[0]["expected_value"])
        april_match = next(
            row
            for row in rows
            if row["filename"] == "april_2026__day_01_point_01_temperature.png"
        )
        self.assertEqual("", april_match["expected_value"])
        self.assertEqual("false", april_match["expected_blank"])
        self.assertEqual(list(LABEL_FIELDNAMES), list(rows[0]))

    def test_legacy_trusted_labels_are_scoped_to_sample(self) -> None:
        with _workspace_scratch() as temporary_directory:
            labels_path = temporary_directory / "labels.csv"
            labels_path.write_text(
                "filename,expected_value\n"
                "day_01_point_01_temperature.png,20.0\n",
                encoding="utf-8",
            )
            trusted = load_trusted_labels(labels_path)
        self.assertEqual(
            {("sample", "day_01_point_01_temperature.png"): ("20.0", False)},
            trusted,
        )

    def test_runner_refuses_missing_labels_before_creating_output(self) -> None:
        identities = benchmark_identities()
        rows, _, _ = create_label_rows(identities, {})
        with _workspace_scratch() as temporary_directory:
            root = temporary_directory
            labels_path = root / "labels.csv"
            output_path = root / "results"
            write_csv_rows(labels_path, LABEL_FIELDNAMES, rows)
            with self.assertRaisesRegex(ValueError, "Missing ground-truth labels"):
                run_benchmark(labels_path, output_path)
            self.assertFalse(output_path.exists())

    def test_valid_blank_requires_explicit_true_and_empty_value(self) -> None:
        identity = benchmark_identities()[0]
        row = _complete_label(identity, Path("."))
        row["expected_value"] = ""
        row["expected_blank"] = "true"
        validated = validate_complete_labels([row], identities=[identity])
        self.assertTrue(validated[0]["expected_blank"])
        self.assertEqual("", validated[0]["expected_value"])


class GeometryBenchmarkParityTests(unittest.TestCase):
    def test_fixed_and_calibrated_use_identical_ocr_settings(self) -> None:
        self.assertEqual(production_ocr_settings(), production_ocr_settings())
        settings = production_ocr_settings()
        self.assertEqual(
            ["original", "grayscale", "contrast"],
            settings["preprocessing_variants"],
        )
        self.assertEqual(32, settings["cells_per_production_batch"])
        self.assertEqual("consensus", settings["recognition_strategy"])
        self.assertEqual(
            "adaptive",
            production_ocr_settings("adaptive")["recognition_strategy"],
        )

    def test_prediction_cells_never_contain_ground_truth(self) -> None:
        identity = benchmark_identities()[0]
        with _workspace_scratch() as temporary_directory:
            root = temporary_directory
            image = np.full((20, 40, 3), 255, dtype=np.uint8)
            cv2.imwrite(str(root / "fixed.png"), image)
            cv2.imwrite(str(root / "calibrated.png"), image)
            label = _complete_label(identity, root)
            cells = load_mode_cells(root / "labels.csv", [label], "fixed")
        self.assertEqual(1, len(cells))
        self.assertNotIn("expected_value", cells[0])
        self.assertNotIn("expected_blank", cells[0])
        self.assertEqual(
            {"filename", "day", "point", "reading_type", "sheet_id", "image"},
            set(cells[0]),
        )


class GeometryBenchmarkMetricTests(unittest.TestCase):
    def test_metrics_detect_unsafe_acceptance_reviews_and_blanks(self) -> None:
        rows = [
            _metric_row(
                item_number=1,
                sheet_id="sample",
                day_group="top",
                point_group="left",
                correct=True,
            ),
            _metric_row(
                item_number=2,
                sheet_id="sample",
                day_group="middle",
                point_group="center",
                correct=False,
                unsafe=True,
            ),
            _metric_row(
                item_number=3,
                sheet_id="sample",
                day_group="bottom",
                point_group="right",
                correct=True,
                needs_review=True,
                expected_blank=True,
                predicted_blank=True,
                alignment_failure=True,
            ),
        ]
        metrics = calculate_metrics(
            rows, ocr_input_count=6, total_ocr_seconds=3.0
        )
        self.assertEqual(1, metrics["unsafe_automatic_acceptances"])
        self.assertEqual(1, metrics["correct_automatic_acceptances"])
        self.assertEqual(1, metrics["review_count"])
        self.assertEqual(1, metrics["false_review_count"])
        self.assertEqual(0.0, metrics["review_precision"])
        self.assertEqual(1.0, metrics["blank_precision"])
        self.assertEqual(1.0, metrics["blank_recall"])
        self.assertEqual(1, metrics["crop_alignment_failure_count"])
        self.assertEqual(2.0, metrics["average_ocr_inputs_per_cell"])
        self.assertEqual(1.0, metrics["average_ocr_time_per_cell_seconds"])

    def test_result_evaluation_marks_wrong_unreviewed_value_unsafe(self) -> None:
        label = {
            "item_number": 1,
            "sheet_id": "sample",
            "filename": "sample__cell.png",
            "day": 1,
            "point": 1,
            "reading_type": "temperature",
            "expected_value": "22.0",
            "expected_blank": False,
        }
        result = SimpleNamespace(
            is_blank=False,
            final_value="23.0",
            consensus_prediction="23.0",
            agreement_count=3,
            average_consensus_confidence=0.99,
            postprocessing_status="unchanged_valid",
            review_categories=(),
            blocks_export=False,
            needs_review=False,
            review_reason="No issues.",
            predictions={"original": "23.0", "grayscale": "23.0", "contrast": "23.0"},
            raw_predictions={"original": "23.0", "grayscale": "23.0", "contrast": "23.0"},
            confidences={"original": 0.99, "grayscale": 0.99, "contrast": 0.99},
        )
        row = evaluate_prediction_result(label, result, "fixed", "23.0", False)
        self.assertFalse(row["final_value_correct"])
        self.assertTrue(row["unsafe_automatic_acceptance"])

    def test_per_sheet_and_position_summaries_are_complete(self) -> None:
        rows = [
            _metric_row(
                item_number=1,
                sheet_id="sample",
                day_group="top",
                point_group="left",
            ),
            _metric_row(
                item_number=2,
                sheet_id="april_2026",
                day_group="middle",
                point_group="center",
                reading_type="humidity",
            ),
            _metric_row(
                item_number=3,
                sheet_id="may_2026",
                day_group="bottom",
                point_group="right",
            ),
        ]
        summary = summarize_mode(
            rows,
            ocr_input_count=9,
            total_ocr_seconds=1.5,
            total_processing_seconds=2.0,
        )
        self.assertEqual(
            {"sample", "april_2026", "may_2026"},
            set(summary["breakdowns"]["sheet_id"]),
        )
        self.assertEqual(
            {"top", "middle", "bottom"},
            set(summary["breakdowns"]["day_group"]),
        )
        self.assertEqual(
            {"left", "center", "right"},
            set(summary["breakdowns"]["point_group"]),
        )

    def test_comparison_and_output_schemas_are_deterministic(self) -> None:
        fixed = _metric_row(
            item_number=1,
            sheet_id="sample",
            day_group="top",
            point_group="left",
            correct=False,
            needs_review=True,
        )
        calibrated = dict(fixed)
        calibrated.update(
            {
                "final_value_correct": True,
                "final_verified_value": "22.0",
                "needs_review": False,
            }
        )
        comparison = compare_geometry_rows([fixed], [calibrated])[0]
        self.assertEqual("improved", comparison["correctness_change"])
        self.assertEqual("review_to_automatic", comparison["review_disposition_change"])
        self.assertEqual(tuple(COMPARISON_FIELDNAMES), tuple(comparison))
        self.assertEqual(len(RESULT_FIELDNAMES), len(set(RESULT_FIELDNAMES)))
        self.assertEqual(len(LABEL_FIELDNAMES), len(set(LABEL_FIELDNAMES)))

    def test_safety_requires_accuracy_safety_and_both_new_sheet_gains(self) -> None:
        fixed_rows = [
            _metric_row(
                item_number=index,
                sheet_id=sheet,
                day_group="top",
                point_group="left",
                correct=(sheet == "sample"),
                alignment_failure=True,
            )
            for index, sheet in enumerate(
                ("sample", "april_2026", "may_2026"), start=1
            )
        ]
        calibrated_rows = [dict(row) for row in fixed_rows]
        for row in calibrated_rows:
            row["crop_alignment_failure"] = False
            if row["sheet_id"] != "sample":
                row["final_value_correct"] = True
                row["correct_automatic_acceptance"] = True
        fixed_summary = summarize_mode(
            fixed_rows,
            ocr_input_count=9,
            total_ocr_seconds=1.0,
            total_processing_seconds=1.0,
        )
        calibrated_summary = summarize_mode(
            calibrated_rows,
            ocr_input_count=9,
            total_ocr_seconds=1.0,
            total_processing_seconds=1.0,
        )
        assessment = assess_calibrated_safety(fixed_summary, calibrated_summary)
        self.assertTrue(assessment["calibrated_geometry_acceptable"])


if __name__ == "__main__":
    unittest.main()
