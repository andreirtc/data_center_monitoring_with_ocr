from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from datacenter_ocr.ocr_processing import CellOCRResult
from datacenter_ocr.monitoring_records import (
    attach_crop_data_urls,
    build_monitoring_rows,
)
from datacenter_ocr.numeric_postprocessing import correct_numeric_prediction
from datacenter_ocr.day_verification import cell_display_status
from datacenter_ocr.review_workflow import (
    apply_monitoring_table_edits,
    apply_quick_review_controls,
    apply_review_action,
    apply_sparse_patches,
    clamp_review_state,
)
from datacenter_ocr.verification import (
    ANOMALY_CATEGORY,
    BLANK_MISMATCH_CATEGORY,
    FORMAT_CATEGORY,
    OCR_CATEGORY,
    OPERATIONAL_CATEGORY,
    RANGE_CATEGORY,
    validate_reading_value,
    verify_cell_results,
)


def make_result(
    *,
    day: int,
    point: int,
    reading_type: str,
    value: str,
    confidence: float = 0.99,
    agreement: int = 3,
    is_blank: bool = False,
    raw_value: str | None = None,
    predictions: dict[str, str] | None = None,
) -> CellOCRResult:
    filename = f"day_{day:02d}_point_{point:02d}_{reading_type}.png"
    raw = value if raw_value is None else raw_value
    prediction_values = predictions or {
        variant: value for variant in ("original", "grayscale", "contrast")
    }
    return CellOCRResult(
        filename=filename,
        day=day,
        point=point,
        reading_type=reading_type,
        predictions=prediction_values,
        raw_predictions=(
            prediction_values
            if predictions is not None and raw_value is None
            else {
                variant: raw
                for variant in ("original", "grayscale", "contrast")
            }
        ),
        confidences={variant: confidence for variant in ("original", "grayscale", "contrast")},
        consensus_prediction=value,
        agreement_count=agreement,
        average_consensus_confidence=confidence,
        final_value="" if is_blank else value,
        needs_review=False,
        review_reason="",
        is_blank=is_blank,
    )


class ValueValidationTests(unittest.TestCase):
    def test_malformed_values_are_not_silently_normalized(self) -> None:
        for malformed in ("22.", "22", "53.33", "2a.0", "22..0", "22,0"):
            with self.subTest(malformed=malformed):
                result = validate_reading_value(
                    malformed, "temperature", allow_blank=False
                )
                self.assertIsNotNone(result.error)
                self.assertIsNone(result.normalized_value)

    def test_absolute_limits_are_separate_from_format(self) -> None:
        result = validate_reading_value("51.0", "temperature", allow_blank=False)
        self.assertTrue(result.format_is_valid)
        self.assertFalse(result.within_absolute_limits)


class NumericPostprocessingTests(unittest.TestCase):
    def test_three_digit_temperature_decimal_is_prefilled_for_review(self) -> None:
        for source, expected in (("493", "49.3"), ("227", "22.7")):
            with self.subTest(source=source):
                correction = correct_numeric_prediction(source, "temperature")
                self.assertEqual(expected, correction.corrected_text)
                self.assertEqual("decimal_inferred", correction.status)
                self.assertTrue(correction.needs_review)
                self.assertIn("Decimal point inferred", correction.reason)

    def test_out_of_range_three_digit_temperature_stays_invalid(self) -> None:
        correction = correct_numeric_prediction("568", "temperature")

        self.assertEqual("568", correction.corrected_text)
        self.assertEqual("three_digit_inference_invalid", correction.status)
        self.assertTrue(correction.needs_review)
        self.assertFalse(correction.candidate_interpretations)

    def test_three_digit_humidity_uses_dd_d_business_rule(self) -> None:
        correction = correct_numeric_prediction("568", "humidity")

        self.assertEqual("56.8", correction.corrected_text)
        self.assertEqual("decimal_inferred", correction.status)
        self.assertTrue(correction.needs_review)

    def test_four_digits_and_excess_precision_remain_unresolved(self) -> None:
        for source in ("4114", "2217", "41.14"):
            with self.subTest(source=source):
                correction = correct_numeric_prediction(source, "temperature")
                self.assertEqual(source, correction.corrected_text)
                self.assertTrue(correction.needs_review)
                self.assertIn("Too many digits", correction.reason)

    def test_inferred_decimal_remains_confirmation_required(self) -> None:
        correction = correct_numeric_prediction("493", "temperature")
        result = replace(
            make_result(
                day=1,
                point=1,
                reading_type="temperature",
                value=correction.corrected_text,
            ),
            postprocessing_status=correction.status,
            ocr_uncertainty_reasons=(correction.reason,),
        )

        verified = verify_cell_results([result])[0]

        self.assertFalse(verified.human_verified)
        self.assertTrue(verified.required_confirmation_reasons)
        self.assertTrue(verified.blocks_export)
        self.assertEqual("critical", verified.operational_severity)
        self.assertTrue(verified.operational_warnings)
        self.assertEqual(
            "Decimal inferred",
            cell_display_status(verified, day_confirmed=False).label,
        )

    def test_three_digit_inference_requires_ascii_digits(self) -> None:
        correction = correct_numeric_prediction("４９３", "temperature")

        self.assertEqual("４９３", correction.corrected_text)
        self.assertNotEqual("decimal_inferred", correction.status)


class ContextVerificationTests(unittest.TestCase):
    def test_three_of_three_low_confidence_still_requires_review(self) -> None:
        result = make_result(
            day=1,
            point=1,
            reading_type="temperature",
            value="22.0",
            confidence=0.50,
        )
        verified = verify_cell_results([result])[0]
        self.assertTrue(verified.needs_review)
        self.assertIn(OCR_CATEGORY, verified.review_categories)
        self.assertTrue(verified.required_confirmation_reasons)
        self.assertTrue(verified.blocks_export)

    def test_elevated_temperature_has_operational_category(self) -> None:
        result = make_result(
            day=1, point=1, reading_type="temperature", value="30.0"
        )
        verified = verify_cell_results([result])[0]
        self.assertEqual("alarming", verified.operational_severity)
        self.assertIn(OPERATIONAL_CATEGORY, verified.review_categories)
        self.assertTrue(verified.operational_warnings)
        self.assertFalse(verified.needs_review)
        self.assertFalse(verified.blocks_export)

    def test_minor_high_confidence_disagreement_is_informational(self) -> None:
        result = make_result(
            day=1,
            point=1,
            reading_type="humidity",
            value="60.8",
            agreement=2,
            predictions={
                "original": "60.87",
                "grayscale": "60.8",
                "contrast": "60.8",
            },
        )
        verified = verify_cell_results([result])[0]
        self.assertIn(OCR_CATEGORY, verified.review_categories)
        self.assertTrue(verified.informational_notices)
        self.assertFalse(verified.required_confirmation_reasons)
        self.assertFalse(verified.blocks_export)

    def test_malformed_and_range_values_are_explicitly_blocking(self) -> None:
        results = verify_cell_results(
            [
                make_result(
                    day=1, point=1, reading_type="temperature", value="22."
                ),
                make_result(
                    day=1, point=2, reading_type="temperature", value="51.0"
                ),
            ]
        )
        self.assertIn(FORMAT_CATEGORY, results[0].review_categories)
        self.assertIn(RANGE_CATEGORY, results[1].review_categories)
        self.assertTrue(all(result.blocking_errors for result in results))
        self.assertTrue(all(result.blocks_export for result in results))
        human_confirmed = verify_cell_results(
            [replace(result, human_verified=True) for result in results]
        )
        self.assertTrue(all(result.blocks_export for result in human_confirmed))

    def test_contextual_outlier_is_flagged(self) -> None:
        results = [
            make_result(
                day=1,
                point=point,
                reading_type="temperature",
                value="40.0" if point == 1 else "22.0",
            )
            for point in range(1, 6)
        ]
        verified = verify_cell_results(results)
        outlier = next(result for result in verified if result.point == 1)
        self.assertTrue(outlier.is_statistical_anomaly)
        self.assertIn(ANOMALY_CATEGORY, outlier.review_categories)
        self.assertTrue(outlier.required_confirmation_reasons)
        self.assertTrue(outlier.blocks_export)

    def test_blank_mismatch_flags_both_readings(self) -> None:
        results = [
            make_result(
                day=1,
                point=1,
                reading_type="temperature",
                value="",
                is_blank=True,
            ),
            make_result(
                day=1, point=1, reading_type="humidity", value="50.0"
            ),
        ]
        verified = verify_cell_results(results)
        self.assertTrue(all(result.has_blank_mismatch for result in verified))
        self.assertTrue(
            all(BLANK_MISMATCH_CATEGORY in result.review_categories for result in verified)
        )
        self.assertTrue(all(result.blocking_errors for result in verified))
        self.assertTrue(all(result.blocks_export for result in verified))

    def test_one_sided_likely_blank_preserves_mismatch_blocking(self) -> None:
        results = [
            replace(
                make_result(
                    day=1,
                    point=1,
                    reading_type="temperature",
                    value="",
                    is_blank=True,
                ),
                postprocessing_status="likely_blank",
            ),
            make_result(
                day=1,
                point=1,
                reading_type="humidity",
                value="50.0",
            ),
        ]

        verified = verify_cell_results(results)

        self.assertTrue(all(result.has_blank_mismatch for result in verified))
        self.assertTrue(all(result.blocks_export for result in verified))


class StateUpdateTests(unittest.TestCase):
    def test_quick_review_correction_needs_no_dropdown_action(self) -> None:
        results = verify_cell_results(
            [
                make_result(
                    day=1, point=1, reading_type="temperature", value="22."
                )
            ]
        )
        filename = results[0].filename

        outcome = apply_quick_review_controls(
            results,
            {filename: {"corrected_value": "22.0"}},
        )

        self.assertFalse(outcome.errors)
        self.assertEqual("22.0", outcome.results[0].final_value)
        self.assertTrue(outcome.results[0].human_verified)
        self.assertFalse(outcome.results[0].blocks_export)

    def test_quick_review_confirm_current_action(self) -> None:
        results = verify_cell_results(
            [
                make_result(
                    day=1,
                    point=1,
                    reading_type="temperature",
                    value="22.0",
                    confidence=0.50,
                )
            ]
        )

        outcome = apply_quick_review_controls(
            results,
            {results[0].filename: {"confirm_current": True}},
        )

        self.assertFalse(outcome.errors)
        self.assertTrue(outcome.results[0].human_verified)
        self.assertFalse(outcome.results[0].blocks_export)

    def test_quick_review_mark_blank_clears_old_value(self) -> None:
        results = verify_cell_results(
            [
                make_result(
                    day=1,
                    point=1,
                    reading_type="temperature",
                    value="22.0",
                    confidence=0.50,
                )
            ]
        )

        outcome = apply_quick_review_controls(
            results,
            {results[0].filename: {"mark_blank": True}},
        )

        self.assertFalse(outcome.errors)
        self.assertEqual("", outcome.results[0].final_value)
        self.assertTrue(outcome.results[0].is_blank)
        self.assertTrue(outcome.results[0].human_verified)

    def test_quick_review_untouched_item_remains_unresolved(self) -> None:
        results = verify_cell_results(
            [
                make_result(
                    day=1,
                    point=1,
                    reading_type="temperature",
                    value="22.0",
                    confidence=0.50,
                )
            ]
        )

        outcome = apply_quick_review_controls(
            results,
            {
                results[0].filename: {
                    "corrected_value": "",
                    "confirm_current": False,
                    "mark_blank": False,
                }
            },
        )

        self.assertEqual(0, outcome.changed_count)
        self.assertFalse(outcome.results[0].human_verified)
        self.assertTrue(outcome.results[0].blocks_export)

    def test_quick_review_conflicting_actions_are_rejected(self) -> None:
        results = verify_cell_results(
            [
                make_result(
                    day=1,
                    point=1,
                    reading_type="temperature",
                    value="22.0",
                    confidence=0.50,
                )
            ]
        )

        outcome = apply_quick_review_controls(
            results,
            {
                results[0].filename: {
                    "corrected_value": "23.0",
                    "confirm_current": True,
                }
            },
        )

        self.assertEqual(0, outcome.changed_count)
        self.assertIn("Day 1, Point 1, Temperature", outcome.errors[0])
        self.assertIn("Choose only one", outcome.errors[0])
        self.assertEqual("22.0", outcome.results[0].final_value)
        self.assertFalse(outcome.results[0].human_verified)

    def test_correction_resolves_malformed_value_and_preserves_human_state(self) -> None:
        original = verify_cell_results(
            [
                make_result(
                    day=1, point=1, reading_type="temperature", value="22."
                )
            ]
        )
        filename = original[0].filename

        rejected = apply_review_action(original, filename, "Confirm current")
        self.assertTrue(rejected.errors)

        corrected = apply_review_action(
            original, filename, "Enter correction", "22.0"
        )
        self.assertFalse(corrected.errors)
        self.assertEqual("22.0", corrected.results[0].final_value)
        self.assertTrue(corrected.results[0].human_verified)
        self.assertFalse(corrected.results[0].needs_review)

    def test_verified_operational_warning_is_recorded_but_resolved(self) -> None:
        original = verify_cell_results(
            [
                make_result(
                    day=1, point=1, reading_type="temperature", value="31.0"
                )
            ]
        )
        outcome = apply_review_action(
            original, original[0].filename, "Confirm current"
        )
        self.assertFalse(outcome.results[0].needs_review)
        self.assertIn(OPERATIONAL_CATEGORY, outcome.results[0].review_categories)
        self.assertTrue(outcome.results[0].human_verified)

    def test_valid_field_saves_when_another_edited_field_is_invalid(self) -> None:
        results = verify_cell_results(
            [
                make_result(
                    day=1, point=1, reading_type="temperature", value="22.0"
                ),
                make_result(
                    day=1, point=1, reading_type="humidity", value="50.0"
                ),
            ]
        )
        invalid_row = {
            "Day": 1,
            "Point": 1,
            "Temperature": "23.0",
            "Humidity": "50.",
            "Temperature Blank": False,
            "Humidity Blank": False,
        }
        outcome = apply_monitoring_table_edits(results, [invalid_row])
        self.assertTrue(outcome.errors)
        self.assertEqual("23.0", outcome.results[0].final_value)
        self.assertEqual("50.0", outcome.results[1].final_value)
        self.assertEqual(1, outcome.changed_count)

    def test_image_assisted_table_submits_only_sparse_changed_field(self) -> None:
        results = verify_cell_results(
            [
                make_result(
                    day=1, point=1, reading_type="temperature", value="22.0"
                ),
                make_result(
                    day=1, point=1, reading_type="humidity", value="50.0"
                ),
            ]
        )

        outcome = apply_monitoring_table_edits(
            results,
            [{"Day": 1, "Point": 1, "Temperature": "23.0"}],
        )

        self.assertEqual(1, outcome.changed_count)
        self.assertEqual("23.0", outcome.results[0].final_value)
        self.assertTrue(outcome.results[0].human_verified)
        self.assertEqual("50.0", outcome.results[1].final_value)
        self.assertFalse(outcome.results[1].human_verified)

    def test_both_blank_flags_clear_values_and_resolve_row(self) -> None:
        results = verify_cell_results(
            [
                make_result(
                    day=1, point=1, reading_type="temperature", value="22.0"
                ),
                make_result(
                    day=1, point=1, reading_type="humidity", value="50.0"
                ),
                make_result(
                    day=1, point=2, reading_type="temperature", value="22."
                ),
                make_result(
                    day=1, point=2, reading_type="humidity", value="50.0"
                ),
            ]
        )
        outcome = apply_monitoring_table_edits(
            results,
            [
                {
                    "Day": 1,
                    "Point": 1,
                    "Temperature Blank": True,
                    "Humidity Blank": True,
                }
            ],
        )
        self.assertFalse(outcome.errors)
        blank_pair = [result for result in outcome.results if result.point == 1]
        self.assertTrue(all(result.is_blank for result in blank_pair))
        self.assertTrue(all(result.final_value == "" for result in blank_pair))
        self.assertFalse(any(result.blocks_export for result in blank_pair))
        unrelated = next(result for result in outcome.results if result.point == 2)
        self.assertTrue(unrelated.blocks_export)

    def test_one_blank_flag_remains_blocking_after_confirmation(self) -> None:
        results = verify_cell_results(
            [
                make_result(
                    day=1, point=1, reading_type="temperature", value="22.0"
                ),
                make_result(
                    day=1, point=1, reading_type="humidity", value="50.0"
                ),
            ]
        )
        outcome = apply_monitoring_table_edits(
            results,
            [{"Day": 1, "Point": 1, "Temperature Blank": True}],
        )
        self.assertEqual("", outcome.results[0].final_value)
        self.assertTrue(all(result.has_blank_mismatch for result in outcome.results))
        for result in list(outcome.results):
            outcome = apply_review_action(
                outcome.results, result.filename, "Confirm current"
            )
        self.assertTrue(all(result.human_verified for result in outcome.results))
        self.assertTrue(all(result.blocks_export for result in outcome.results))

    def test_partial_batch_review_preserves_untouched_item(self) -> None:
        results = verify_cell_results(
            [
                make_result(
                    day=1,
                    point=point,
                    reading_type="temperature",
                    value="22.0",
                    confidence=0.50,
                )
                for point in (1, 2)
            ]
        )
        outcome = apply_quick_review_controls(
            results,
            {results[0].filename: {"confirm_current": True}},
        )
        self.assertFalse(outcome.results[0].blocks_export)
        self.assertTrue(outcome.results[1].blocks_export)
        self.assertFalse(outcome.results[1].human_verified)

    def test_review_and_table_updates_share_sparse_patch_engine(self) -> None:
        results = verify_cell_results(
            [
                make_result(
                    day=1, point=1, reading_type="temperature", value="22.0"
                ),
                make_result(
                    day=1, point=1, reading_type="humidity", value="50.0"
                ),
            ]
        )
        with patch(
            "datacenter_ocr.review_workflow.apply_sparse_patches",
            wraps=apply_sparse_patches,
        ) as sparse_engine:
            apply_review_action(
                results, results[0].filename, "Enter correction", "23.0"
            )
            apply_monitoring_table_edits(
                results,
                [{"Day": 1, "Point": 1, "Humidity": "51.0"}],
            )
        self.assertEqual(2, sparse_engine.call_count)

    def test_review_state_clamps_page_and_filename(self) -> None:
        filenames = [f"cell_{index}.png" for index in range(16)]
        page, selected = clamp_review_state(
            filenames,
            page=9,
            selected_filename="removed.png",
        )
        self.assertEqual(2, page)
        self.assertEqual("cell_15.png", selected)

    def test_full_state_graph_rebuilds_from_updated_cell_results(self) -> None:
        results = verify_cell_results(
            [
                make_result(
                    day=day,
                    point=point,
                    reading_type=reading_type,
                    value="22.0" if reading_type == "temperature" else "50.0",
                )
                for day in range(1, 32)
                for point in range(1, 9)
                for reading_type in ("temperature", "humidity")
            ]
        )
        edited_row = {
            "Day": 1,
            "Point": 1,
            "Temperature": "25.0",
            "Humidity": "50.0",
            "Temperature Blank": False,
            "Humidity Blank": False,
        }
        outcome = apply_monitoring_table_edits(results, [edited_row])
        rows = build_monitoring_rows(outcome.results)
        first_row = rows[0]

        self.assertEqual(248, len(rows))
        self.assertEqual("25.0", first_row["temperature"])
        self.assertEqual("alert", first_row["temperature_severity"])
        self.assertFalse(first_row["needs_review"])
        self.assertFalse(first_row["blocks_export"])
        self.assertEqual("Operational warning", first_row["status"])

    def test_rejected_table_edit_rebuilds_canonical_display_value(self) -> None:
        results = verify_cell_results(
            [
                make_result(
                    day=day,
                    point=point,
                    reading_type=reading_type,
                    value="22.0" if reading_type == "temperature" else "50.0",
                )
                for day in range(1, 32)
                for point in range(1, 9)
                for reading_type in ("temperature", "humidity")
            ]
        )

        rejected = apply_monitoring_table_edits(
            results,
            [{"Day": 1, "Point": 1, "Temperature": "25."}],
        )
        rebuilt_rows = build_monitoring_rows(rejected.results)

        self.assertTrue(rejected.errors)
        self.assertEqual(0, rejected.changed_count)
        self.assertEqual("22.0", rebuilt_rows[0]["temperature"])
        self.assertFalse(rebuilt_rows[0]["temperature_human_verified"])


class CropLookupTests(unittest.TestCase):
    def test_crop_lookup_uses_each_readings_stable_filename(self) -> None:
        monitoring_rows = [
            {
                "temperature_filename": "temperature-stable.png",
                "humidity_filename": "humidity-stable.png",
            }
        ]
        crop_urls = {
            "temperature-stable.png": "data:image/png;base64,temp",
            "humidity-stable.png": "data:image/png;base64,humidity",
        }

        enriched = attach_crop_data_urls(monitoring_rows, crop_urls)

        self.assertEqual(
            "data:image/png;base64,temp", enriched[0]["temperature_crop"]
        )
        self.assertEqual(
            "data:image/png;base64,humidity", enriched[0]["humidity_crop"]
        )


if __name__ == "__main__":
    unittest.main()
