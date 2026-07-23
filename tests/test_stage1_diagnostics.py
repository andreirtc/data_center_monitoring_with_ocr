from __future__ import annotations

import ast
import csv
from dataclasses import asdict
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from datacenter_ocr.cell_extraction import (
    create_cell_filename,
    extract_measurement_cells,
)
from datacenter_ocr.config import (
    CELL_BOTTOM_PADDING_RATIO,
    CELL_HORIZONTAL_MARGIN_RATIO,
    CELL_TOP_PADDING_RATIO,
    CELL_VERTICAL_MARGIN_RATIO,
    STANDARD_TABLE_HEIGHT,
    STANDARD_TABLE_WIDTH,
)
from datacenter_ocr.diagnostic_export import (
    DIAGNOSTIC_CELL_RESULT_FIELDS,
    write_cell_results_csv,
)
from datacenter_ocr.grid_diagnostics import (
    ALIGNMENT_REPORT_FILENAME,
    CONTACT_SHEET_FILENAME,
    GRID_OVERLAY_FILENAME,
    LINE_DIAGNOSTIC_OVERLAY_FILENAME,
    WARPED_TABLE_FILENAME,
    calculate_alignment_report,
    create_diagnostic_contact_sheet,
    diagnostic_cell_keys,
    save_grid_diagnostic_outputs,
)
from datacenter_ocr.image_processing import draw_measurement_boxes
from datacenter_ocr.local_grid import align_measurement_boxes_to_printed_rows
from datacenter_ocr.ocr_processing import (
    CellOCRResult,
    process_measurement_cells,
)
from datacenter_ocr.processing_metrics import ProcessingMetrics
from datacenter_ocr.sheet_processing import PreparedMonitoringSheet
from datacenter_ocr.table_layout import build_measurement_boxes


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_prepared_sheet(*, include_lines: bool = True) -> PreparedMonitoringSheet:
    """Build a deterministic standardized sheet without company imagery."""

    image = np.full(
        (STANDARD_TABLE_HEIGHT, STANDARD_TABLE_WIDTH, 3),
        255,
        dtype=np.uint8,
    )
    boxes = build_measurement_boxes(STANDARD_TABLE_WIDTH, STANDARD_TABLE_HEIGHT)
    if include_lines:
        x_positions = sorted({box[key] for box in boxes for key in ("x1", "x2")})
        y_positions = sorted({box[key] for box in boxes for key in ("y1", "y2")})
        for x_position in x_positions:
            cv2.line(
                image,
                (x_position, y_positions[0]),
                (x_position, y_positions[-1]),
                (0, 0, 0),
                2,
            )
        for y_position in y_positions:
            cv2.line(
                image,
                (x_positions[0], y_position),
                (x_positions[-1], y_position),
                (0, 0, 0),
                2,
            )
    cells = extract_measurement_cells(
        image=image,
        measurement_boxes=boxes,
        horizontal_margin_ratio=CELL_HORIZONTAL_MARGIN_RATIO,
        vertical_margin_ratio=CELL_VERTICAL_MARGIN_RATIO,
        top_padding_ratio=CELL_TOP_PADDING_RATIO,
        bottom_padding_ratio=CELL_BOTTOM_PADDING_RATIO,
    )
    return PreparedMonitoringSheet(
        detection_preview=image.copy(),
        warped_table=image,
        measurement_grid_overlay=draw_measurement_boxes(image, boxes),
        measurement_boxes=boxes,
        cells=cells,
    )


def make_verified_result() -> CellOCRResult:
    return CellOCRResult(
        filename="day_01_point_01_temperature.png",
        day=1,
        point=1,
        reading_type="temperature",
        raw_predictions={
            "original": "22.O",
            "grayscale": "22.0",
            "contrast": "220",
        },
        predictions={
            "original": "22.0",
            "grayscale": "22.0",
            "contrast": "220",
        },
        confidences={"original": 0.91, "grayscale": 0.92, "contrast": 0.70},
        consensus_prediction="22.0",
        agreement_count=2,
        average_consensus_confidence=0.915,
        final_value="22.0",
        needs_review=True,
        review_reason="Confirm OCR.",
        is_blank=False,
        blank_ink_ratio=0.123456,
        ocr_uncertainty_reasons=("Variant disagreement.",),
        human_verified=False,
        verification_reasons=("Confirm OCR.",),
        review_categories=("ocr_uncertainty", "operational_warning"),
        blocking_errors=("Malformed value.",),
        required_confirmation_reasons=("Confirm OCR.",),
        operational_warnings=("Temperature is elevated.",),
        informational_notices=("Diagnostic notice.",),
        blocks_export=True,
        format_is_valid=True,
        within_absolute_limits=True,
        operational_severity="warning",
        is_statistical_anomaly=False,
        has_blank_mismatch=False,
        ocr_uncertain=True,
        postprocessing_status="valid_unchanged",
        candidate_interpretations=("22.0",),
    )


class GridDiagnosticTests(unittest.TestCase):
    def test_row_sequence_alignment_recovers_scanner_shift(self) -> None:
        image = np.full(
            (STANDARD_TABLE_HEIGHT, STANDARD_TABLE_WIDTH, 3),
            255,
            dtype=np.uint8,
        )
        boxes = build_measurement_boxes(STANDARD_TABLE_WIDTH, STANDARD_TABLE_HEIGHT)
        x_positions = sorted(
            {box[key] for box in boxes for key in ("x1", "x2")}
        )
        # A scanner-selected inner table border places Day 1 much higher than
        # the reference prior. The extra header line must not become Day 1.
        cv2.line(image, (x_positions[0], 104), (x_positions[-1], 104), (0, 0, 0), 2)
        expected_boundaries = [142 + index * 36 for index in range(32)]
        for y_position in expected_boundaries:
            cv2.line(
                image,
                (x_positions[0], y_position),
                (x_positions[-1], y_position),
                (0, 0, 0),
                2,
            )

        aligned_boxes, alignment = align_measurement_boxes_to_printed_rows(
            image,
            boxes,
        )

        self.assertTrue(alignment.used_detected_alignment)
        self.assertEqual(32, alignment.strong_boundary_count)
        self.assertAlmostEqual(142, alignment.y_boundaries[0], delta=2)
        self.assertAlmostEqual(1258, alignment.y_boundaries[-1], delta=2)
        self.assertEqual(496, len(aligned_boxes))
        self.assertEqual(alignment.y_boundaries[0], aligned_boxes[0]["y1"])
        self.assertEqual(alignment.y_boundaries[-1], aligned_boxes[-1]["y2"])

    def test_row_sequence_alignment_falls_back_without_strong_lines(self) -> None:
        image = np.full(
            (STANDARD_TABLE_HEIGHT, STANDARD_TABLE_WIDTH, 3),
            255,
            dtype=np.uint8,
        )
        boxes = build_measurement_boxes(STANDARD_TABLE_WIDTH, STANDARD_TABLE_HEIGHT)

        aligned_boxes, alignment = align_measurement_boxes_to_printed_rows(
            image,
            boxes,
        )

        self.assertFalse(alignment.used_detected_alignment)
        self.assertIn("strong evidence", alignment.fallback_reason)
        self.assertEqual(boxes, aligned_boxes)

    def test_expected_496_cell_identity_mapping(self) -> None:
        prepared = make_prepared_sheet()
        self.assertEqual(496, len(prepared.measurement_boxes))
        self.assertEqual(496, len(prepared.cells))
        filenames = {cell["filename"] for cell in prepared.cells}
        self.assertEqual(496, len(filenames))
        for box, cell in zip(prepared.measurement_boxes, prepared.cells):
            self.assertEqual(create_cell_filename(box), cell["filename"])
            for key in ("day", "point", "reading_type", "x1", "y1", "x2", "y2"):
                self.assertEqual(box[key], cell[key])
            self.assertGreater(cell["image"].size, 0)

    def test_contact_sheet_selection_and_association(self) -> None:
        expected = {
            (day, point, reading_type)
            for day in (1, 16, 31)
            for point in (1, 4, 8)
            for reading_type in ("temperature", "humidity")
        }
        self.assertEqual(expected, set(diagnostic_cell_keys()))
        prepared = make_prepared_sheet()
        report = calculate_alignment_report(prepared)
        contact_sheet = create_diagnostic_contact_sheet(prepared, report)
        self.assertEqual((1170, 2820, 3), contact_sheet.shape)
        self.assertLess(int(contact_sheet.min()), 255)

    def test_report_schema_and_exact_synthetic_line_matches(self) -> None:
        report = calculate_alignment_report(make_prepared_sheet())
        required = {
            "standardized_image_width",
            "standardized_image_height",
            "expected_horizontal_boundaries",
            "detected_horizontal_boundaries",
            "expected_vertical_boundaries",
            "detected_vertical_boundaries",
            "horizontal_boundaries",
            "vertical_boundaries",
            "horizontal_measurements",
            "vertical_measurements",
            "left_to_right_horizontal_row_drift",
            "top_to_bottom_vertical_column_drift",
            "line_spacing_variation",
            "provisional_alignment_score",
            "alignment_score_is_calibrated",
        }
        self.assertTrue(required.issubset(report))
        self.assertEqual(32, report["expected_horizontal_line_count"])
        self.assertEqual(17, report["expected_vertical_line_count"])
        self.assertEqual(32, report["matched_horizontal_line_count"])
        self.assertEqual(17, report["matched_vertical_line_count"])
        self.assertFalse(report["alignment_score_is_calibrated"])
        for boundary in report["horizontal_boundaries"] + report["vertical_boundaries"]:
            self.assertIn("pixel_error", boundary)
            self.assertIn("normalized_error", boundary)
            self.assertIn("continuity", boundary)

    def test_unmatched_lines_are_safe_and_not_distant_matches(self) -> None:
        report = calculate_alignment_report(make_prepared_sheet(include_lines=False))
        self.assertEqual(0, report["matched_horizontal_line_count"])
        self.assertEqual(0, report["matched_vertical_line_count"])
        self.assertTrue(
            all(value is None for value in report["detected_horizontal_boundaries"])
        )
        self.assertTrue(
            all(value is None for value in report["detected_vertical_boundaries"])
        )

    def test_diagnostic_output_paths(self) -> None:
        prepared = make_prepared_sheet()
        folder = PROJECT_ROOT / "output" / "grid_diagnostic" / "_test_artifacts"
        folder.mkdir(parents=True, exist_ok=True)
        expected_names = {
            WARPED_TABLE_FILENAME,
            GRID_OVERLAY_FILENAME,
            LINE_DIAGNOSTIC_OVERLAY_FILENAME,
            CONTACT_SHEET_FILENAME,
            ALIGNMENT_REPORT_FILENAME,
        }
        try:
            outputs = save_grid_diagnostic_outputs(prepared, folder)
            actual_paths = set(asdict(outputs).values())
            self.assertEqual(expected_names, {path.name for path in actual_paths})
            self.assertTrue(all(path.is_file() for path in actual_paths))
        finally:
            for output_name in expected_names:
                (folder / output_name).unlink(missing_ok=True)
            folder.rmdir()

    def test_cli_diagnostic_import_graph_has_no_ocr_model_import(self) -> None:
        for relative_path in (
            "scripts/diagnose_grid_alignment.py",
            "datacenter_ocr/grid_diagnostics.py",
            "datacenter_ocr/sheet_processing.py",
        ):
            tree = ast.parse((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))
            imported_modules = {
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            imported_modules.update(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            self.assertFalse(
                any(module.startswith("paddleocr") for module in imported_modules),
                relative_path,
            )


class MetricsAndExportTests(unittest.TestCase):
    def test_timing_result_schema(self) -> None:
        metrics = ProcessingMetrics()
        expected_fields = {
            "upload_decoding_seconds",
            "document_table_detection_seconds",
            "perspective_warp_seconds",
            "measurement_cell_extraction_seconds",
            "blank_detection_seconds",
            "ocr_preprocessing_seconds",
            "ocr_prediction_seconds",
            "postprocessing_seconds",
            "verification_seconds",
            "monitoring_record_construction_seconds",
            "ui_thumbnail_preparation_seconds",
            "total_sheet_processing_seconds",
            "model_construction_seconds",
            "first_prediction_warmup_seconds",
            "filled_cell_count",
            "blank_cell_count",
            "ocr_input_image_count",
            "model_predict_call_count",
            "requested_batch_size",
            "result_batch_count",
            "process_uptime_seconds",
            "model_was_warm",
            "uploaded_width",
            "uploaded_height",
            "uploaded_fingerprint",
        }
        self.assertTrue(expected_fields.issubset(metrics.to_dict()))

    def test_total_timing_sums_only_non_overlapping_stages(self) -> None:
        metrics = ProcessingMetrics(
            upload_decoding_seconds=1.0,
            document_table_detection_seconds=2.0,
            ocr_prediction_seconds=3.0,
            model_construction_seconds=4.0,
            first_prediction_warmup_seconds=3.0,
        )

        metrics.recalculate_total()

        self.assertEqual(10.0, metrics.total_sheet_processing_seconds)

    def test_inference_counters_do_not_change_ocr_output(self) -> None:
        class FakeResult:
            json = {"res": {"rec_text": "22.0", "rec_score": 0.99}}

        class FakeModel:
            def predict(self, *, input: list[np.ndarray], batch_size: int):
                del batch_size
                return [FakeResult() for _ in input]

        cell = {
            "filename": "day_01_point_01_temperature.png",
            "day": 1,
            "point": 1,
            "reading_type": "temperature",
            "image": np.full((40, 80, 3), 255, dtype=np.uint8),
        }
        baseline = process_measurement_cells(FakeModel(), [cell])
        metrics = ProcessingMetrics(model_was_warm=True)
        instrumented = process_measurement_cells(FakeModel(), [cell], metrics=metrics)
        self.assertEqual(baseline, instrumented)
        self.assertEqual(3, metrics.ocr_input_image_count)
        self.assertEqual(1, metrics.model_predict_call_count)
        self.assertEqual(16, metrics.requested_batch_size)
        self.assertEqual(1, metrics.result_batch_count)

    def test_diagnostic_csv_serializes_all_verification_fields(self) -> None:
        output_path = PROJECT_ROOT / "output" / "_stage1_cell_results_test.csv"
        try:
            write_cell_results_csv([make_verified_result()], output_path)
            with output_path.open(newline="", encoding="utf-8") as csv_file:
                row = next(csv.DictReader(csv_file))
        finally:
            output_path.unlink(missing_ok=True)
        self.assertEqual(set(DIAGNOSTIC_CELL_RESULT_FIELDS), set(row))
        self.assertEqual("22.O", json.loads(row["raw_predictions_json"])["original"])
        self.assertEqual(
            ["Malformed value."],
            json.loads(row["blocking_errors_json"]),
        )
        self.assertEqual(
            ["Confirm OCR."],
            json.loads(row["required_confirmation_reasons_json"]),
        )
        self.assertEqual(
            ["Temperature is elevated."],
            json.loads(row["operational_warnings_json"]),
        )
        self.assertEqual(
            ["Diagnostic notice."],
            json.loads(row["informational_notices_json"]),
        )
        self.assertEqual(
            ["ocr_uncertainty", "operational_warning"],
            json.loads(row["review_categories_json"]),
        )


if __name__ == "__main__":
    unittest.main()
