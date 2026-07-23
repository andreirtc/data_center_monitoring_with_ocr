from __future__ import annotations

import inspect
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from datacenter_ocr.blank_cell_detection import (
    BLANK_ANALYSIS_CANVAS_HEIGHT,
    BLANK_ANALYSIS_CANVAS_WIDTH,
    BlankCellAnalysis,
    analyze_cell_for_blankness,
    create_blank_analysis_canvas,
    is_likely_border_artifact_blank,
)
from datacenter_ocr.geometry_benchmark import (
    TELEMETRY_FIELDNAMES,
    BenchmarkIdentity,
    _calibrated_cell_telemetry,
    _telemetry_csv_row,
    build_blank_analysis_comparison,
    build_hybrid_rows,
    select_geometry_from_alignment_metrics,
    summarize_blank_analysis,
)
from datacenter_ocr.local_grid import BoundaryCurve
from datacenter_ocr.ocr_processing import (
    CellOCRResult,
    apply_likely_blank_proposal,
)
from datacenter_ocr.sheet_processing import prepare_monitoring_sheet
from scripts.run_geometry_ocr_benchmark import (
    counterbalanced_execution_orders,
    summarize_timing_trials,
)


def _curve(
    axis: str,
    index: int,
    expected: float,
    source: str,
    confidence: float,
) -> BoundaryCurve:
    return BoundaryCurve(
        axis=axis,
        index=index,
        expected_position=expected,
        sample_coordinates=(0.0, 100.0),
        sampled_positions=(expected, expected),
        sample_confidences=(confidence, confidence),
        sample_sources=(source, source),
        direct_detection_count=2 if source == "detected" else 0,
        expected_sample_count=2,
        confidence=confidence,
        uses_fixed_fallback=source == "fixed_fallback",
        uses_interpolation=source.startswith("interpolated"),
    )


def _hybrid_row(
    *,
    filename: str,
    sheet_id: str,
    proposed: str,
    predicted_blank: bool = False,
    blocks_export: bool = False,
    automatic: bool = True,
) -> dict[str, object]:
    return {
        "item_number": 1,
        "sheet_id": sheet_id,
        "filename": filename,
        "geometry_mode": "fixed",
        "selected_geometry": "fixed",
        "proposed_final_value": proposed,
        "predicted_blank": predicted_blank,
        "blocks_export": blocks_export,
        "automatic_acceptance": automatic,
        "needs_review": blocks_export,
        "review_categories": "",
        "review_reason": "Existing disposition.",
        "correct_automatic_acceptance": automatic,
        "unsafe_automatic_acceptance": False,
    }


class NormalizedBlankAnalysisTests(unittest.TestCase):
    def test_normalized_analysis_uses_equal_sized_canvases(self) -> None:
        fixed = np.full((36, 110, 3), 246, dtype=np.uint8)
        calibrated = cv2.resize(fixed, (101, 29), interpolation=cv2.INTER_CUBIC)
        fixed_canvas = create_blank_analysis_canvas(fixed)
        calibrated_canvas = create_blank_analysis_canvas(calibrated)
        expected_shape = (
            BLANK_ANALYSIS_CANVAS_HEIGHT,
            BLANK_ANALYSIS_CANVAS_WIDTH,
            3,
        )
        self.assertEqual(expected_shape, fixed_canvas.shape)
        self.assertEqual(expected_shape, calibrated_canvas.shape)

    def test_visually_equivalent_crops_have_identical_blank_disposition(self) -> None:
        fixed = np.full((36, 110, 3), 255, dtype=np.uint8)
        cv2.putText(
            fixed,
            "23.2",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        calibrated = cv2.resize(fixed, (101, 29), interpolation=cv2.INTER_CUBIC)
        fixed_analysis = analyze_cell_for_blankness(fixed)
        calibrated_analysis = analyze_cell_for_blankness(calibrated)
        self.assertFalse(fixed_analysis.is_blank)
        self.assertEqual(fixed_analysis.is_blank, calibrated_analysis.is_blank)
        self.assertEqual(
            fixed_analysis.cleaned_ink_mask.shape,
            calibrated_analysis.cleaned_ink_mask.shape,
        )

    def test_no_false_blanks_among_labeled_filled_synthetic_cells(self) -> None:
        blank = np.full((36, 110, 3), 255, dtype=np.uint8)
        filled = blank.copy()
        cv2.putText(
            filled,
            "42.4",
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        labels = [
            {
                "item_number": 1,
                "sheet_id": "synthetic",
                "filename": "blank.png",
                "day": 1,
                "point": 1,
                "reading_type": "temperature",
                "expected_blank": True,
            },
            {
                "item_number": 2,
                "sheet_id": "synthetic",
                "filename": "filled.png",
                "day": 1,
                "point": 1,
                "reading_type": "humidity",
                "expected_blank": False,
            },
        ]
        cells_by_mode = {
            "fixed": [
                {**labels[0], "image": blank},
                {**labels[1], "image": filled},
            ],
            "calibrated": [
                {
                    **labels[0],
                    "image": cv2.resize(blank, (101, 29)),
                },
                {
                    **labels[1],
                    "image": cv2.resize(filled, (101, 29)),
                },
            ],
        }
        rows = build_blank_analysis_comparison(labels, cells_by_mode)
        summary = summarize_blank_analysis(rows)
        self.assertEqual([], summary["fixed"]["filled_cells_newly_classified_blank"])
        self.assertEqual(
            [], summary["calibrated"]["filled_cells_newly_classified_blank"]
        )


class LikelyBlankProposalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analysis = BlankCellAnalysis(
            is_blank=False,
            ink_ratio=0.006,
            significant_component_count=1,
            largest_component_ratio=0.006,
            largest_component_width_ratio=0.04,
            largest_component_height_ratio=0.35,
            largest_component_aspect_ratio=0.3,
            analysis_width=112,
            analysis_height=40,
            used_normalized_canvas=True,
            cleaned_ink_mask=np.zeros((40, 112), dtype=np.uint8),
        )
        self.result = CellOCRResult(
            filename="day_01_point_01_temperature.png",
            day=1,
            point=1,
            reading_type="temperature",
            predictions={"original": "1", "grayscale": "1", "contrast": ""},
            raw_predictions={"original": "1", "grayscale": "I", "contrast": ""},
            confidences={"original": 0.7, "grayscale": 0.6, "contrast": 0.0},
            consensus_prediction="1",
            agreement_count=2,
            average_consensus_confidence=0.65,
            final_value="1",
            needs_review=True,
            review_reason="No safe correction.",
            postprocessing_status="no_safe_correction",
        )

    def test_combined_evidence_proposes_blank_and_clears_old_value(self) -> None:
        proposed = apply_likely_blank_proposal(
            self.result,
            self.analysis,
            {"filename": self.result.filename},
        )

        self.assertTrue(proposed.is_blank)
        self.assertEqual("", proposed.final_value)
        self.assertFalse(proposed.human_verified)
        self.assertEqual("likely_blank", proposed.postprocessing_status)
        self.assertIn("only border-like ink", proposed.review_reason)

    def test_ocr_text_one_is_not_sufficient_by_itself(self) -> None:
        stronger_ink = SimpleNamespace(
            **{
                **self.analysis.__dict__,
                "ink_ratio": 0.04,
            }
        )

        proposed = is_likely_border_artifact_blank(
            stronger_ink,
            self.result.predictions,
            self.result.raw_predictions,
            (),
            has_serious_geometry_warning=False,
        )

        self.assertFalse(proposed)

    def test_geometry_warning_prevents_likely_blank_proposal(self) -> None:
        proposed = apply_likely_blank_proposal(
            self.result,
            self.analysis,
            {
                "filename": self.result.filename,
                "geometry_rejection_reason": "Crop bounds were rejected.",
            },
        )

        self.assertFalse(proposed.is_blank)
        self.assertEqual("1", proposed.final_value)

    def test_non_line_ocr_variant_prevents_likely_blank_proposal(self) -> None:
        proposed = is_likely_border_artifact_blank(
            self.analysis,
            {**self.result.predictions, "contrast": "7"},
            self.result.raw_predictions,
            (),
            has_serious_geometry_warning=False,
        )

        self.assertFalse(proposed)


class GeometryTelemetryTests(unittest.TestCase):
    def test_boundary_sources_and_complete_telemetry_schema(self) -> None:
        calibration = SimpleNamespace(
            horizontal_curves=(
                _curve("horizontal", 0, 0.0, "detected", 0.9),
                _curve(
                    "horizontal", 1, 36.0, "interpolated_along_boundary", 0.6
                ),
            ),
            vertical_curves=(
                _curve(
                    "vertical", 0, 0.0, "interpolated_from_neighbors", 0.4
                ),
                _curve("vertical", 1, 110.0, "fixed_fallback", 0.0),
            ),
        )
        cell = {
            "day": 1,
            "point": 1,
            "reading_type": "temperature",
            "x1": 0,
            "y1": 0,
            "x2": 110,
            "y2": 36,
            "source_quadrilateral": ((1, 2), (109, 1), (108, 35), (2, 36)),
            "local_geometry_confidence": 0.4,
        }
        geometry = _calibrated_cell_telemetry(cell, calibration)
        self.assertEqual(
            ["detected", "interpolated", "interpolated", "fixed_fallback"],
            geometry["boundary_sources"],
        )
        self.assertEqual(2, geometry["interpolation_count"])
        self.assertEqual(1, geometry["fallback_count"])

        identity = BenchmarkIdentity(
            1,
            "synthetic",
            "synthetic.png",
            "synthetic__cell.png",
            "cell.png",
            1,
            1,
            "temperature",
        )
        blank_fields = {
            "is_blank": False,
            "ink_ratio": 0.1,
            "component_count": 2,
            "largest_component_ratio": 0.05,
            "analysis_width": 112,
            "analysis_height": 40,
        }
        row = _telemetry_csv_row(
            identity=identity,
            crop_mode="calibrated",
            crop_image=np.full((29, 101, 3), 255, dtype=np.uint8),
            geometry=geometry,
            blank_before=blank_fields,
            blank_after=blank_fields,
            sheet_metrics={
                "material_drift_score": 0.4,
                "material_drift_threshold": 0.3,
                "requires_calibration": True,
            },
        )
        self.assertEqual(tuple(TELEMETRY_FIELDNAMES), tuple(row))


class HybridGeometryPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stable_metrics = {
            "stable": {
                "material_drift_score": 0.10,
                "material_drift_threshold": 0.30,
            }
        }
        self.drifted_metrics = {
            "drifted": {
                "material_drift_score": 0.45,
                "material_drift_threshold": 0.30,
            }
        }

    def test_sheet_policy_uses_metrics_not_filename(self) -> None:
        low = {"material_drift_score": 0.2, "material_drift_threshold": 0.3}
        high = {"material_drift_score": 0.4, "material_drift_threshold": 0.3}
        self.assertEqual("fixed", select_geometry_from_alignment_metrics(low))
        self.assertEqual("calibrated", select_geometry_from_alignment_metrics(high))

    def test_value_disagreement_requires_confirmation(self) -> None:
        fixed = _hybrid_row(filename="cell.png", sheet_id="stable", proposed="23.2")
        calibrated = _hybrid_row(
            filename="cell.png", sheet_id="stable", proposed="22.2"
        )
        hybrid, counts = build_hybrid_rows(
            [fixed], [calibrated], self.stable_metrics
        )
        self.assertTrue(hybrid[0]["needs_review"])
        self.assertEqual(1, counts["disagreement_confirmation_count"])

    def test_blank_disagreement_requires_confirmation(self) -> None:
        fixed = _hybrid_row(filename="cell.png", sheet_id="stable", proposed="")
        calibrated = _hybrid_row(
            filename="cell.png",
            sheet_id="stable",
            proposed="",
            predicted_blank=True,
        )
        hybrid, counts = build_hybrid_rows(
            [fixed], [calibrated], self.stable_metrics
        )
        self.assertTrue(hybrid[0]["needs_review"])
        self.assertEqual(1, counts["blank_disagreement_count"])

    def test_blocking_automatic_disagreement_requires_confirmation(self) -> None:
        fixed = _hybrid_row(
            filename="cell.png",
            sheet_id="stable",
            proposed="24.",
            blocks_export=True,
            automatic=False,
        )
        calibrated = _hybrid_row(
            filename="cell.png", sheet_id="stable", proposed="24.4"
        )
        hybrid, counts = build_hybrid_rows(
            [fixed], [calibrated], self.stable_metrics
        )
        self.assertTrue(hybrid[0]["blocks_export"])
        self.assertEqual(1, counts["blocking_automatic_disagreement_count"])

    def test_calibrated_selection_always_requires_confirmation(self) -> None:
        fixed = _hybrid_row(filename="cell.png", sheet_id="drifted", proposed="42.9")
        calibrated = _hybrid_row(
            filename="cell.png", sheet_id="drifted", proposed="42.9"
        )
        hybrid, counts = build_hybrid_rows(
            [fixed], [calibrated], self.drifted_metrics
        )
        self.assertEqual("calibrated", hybrid[0]["selected_geometry"])
        self.assertTrue(hybrid[0]["needs_review"])
        self.assertFalse(hybrid[0]["automatic_acceptance"])
        self.assertEqual(1, counts["geometry_triggered_confirmation_count"])


class BenchmarkTimingAndDefaultTests(unittest.TestCase):
    def test_execution_order_is_counterbalanced(self) -> None:
        self.assertEqual(
            (("fixed", "calibrated"), ("calibrated", "fixed")),
            counterbalanced_execution_orders(),
        )

    def test_warmup_is_excluded_from_timed_average(self) -> None:
        summary = summarize_timing_trials(
            [
                {
                    "ocr_input_count": 9,
                    "ocr_time_seconds": 2.0,
                    "processing_time_seconds": 3.0,
                },
                {
                    "ocr_input_count": 9,
                    "ocr_time_seconds": 4.0,
                    "processing_time_seconds": 5.0,
                },
            ]
        )
        self.assertEqual(3.0, summary["total_ocr_time_seconds"])
        self.assertEqual(4.0, summary["total_processing_time_seconds"])
        self.assertNotIn("warmup", summary)

    def test_production_geometry_default_remains_fixed(self) -> None:
        default = inspect.signature(prepare_monitoring_sheet).parameters[
            "geometry_mode"
        ].default
        self.assertEqual("fixed", default)


if __name__ == "__main__":
    unittest.main()
