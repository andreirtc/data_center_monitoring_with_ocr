from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from datacenter_ocr.day_verification import (
    DayVerificationState,
    apply_day_submission,
    build_day_scroll_request,
    consume_day_scroll_request,
    invalidate_days_for_changes,
    next_unverified_day,
    previous_day,
    results_for_day,
    state_for_sheet,
    summarize_export_readiness,
)
from datacenter_ocr.ocr_processing import CellOCRResult
from datacenter_ocr.verification import verify_cell_results


def make_result(
    day: int,
    point: int,
    reading_type: str,
    value: str,
    *,
    confidence: float = 0.99,
) -> CellOCRResult:
    filename = f"day_{day:02d}_point_{point:02d}_{reading_type}.png"
    predictions = {
        variant: value for variant in ("original", "grayscale", "contrast")
    }
    return CellOCRResult(
        filename=filename,
        day=day,
        point=point,
        reading_type=reading_type,
        predictions=predictions,
        raw_predictions=predictions,
        confidences={variant: confidence for variant in predictions},
        consensus_prediction=value,
        agreement_count=3,
        average_consensus_confidence=confidence,
        final_value=value,
        needs_review=False,
        review_reason="",
    )


def make_day(day: int, *, temperature: str = "22.0") -> list[CellOCRResult]:
    return verify_cell_results(
        [
            make_result(
                day,
                point,
                reading_type,
                temperature if reading_type == "temperature" else "50.0",
            )
            for point in range(1, 9)
            for reading_type in ("temperature", "humidity")
        ]
    )


def make_sheet() -> list[CellOCRResult]:
    return verify_cell_results(
        [
            make_result(
                day,
                point,
                reading_type,
                "22.0" if reading_type == "temperature" else "50.0",
            )
            for day in range(1, 32)
            for point in range(1, 9)
            for reading_type in ("temperature", "humidity")
        ]
    )


class DayStructureTests(unittest.TestCase):
    def test_day_contains_exactly_sixteen_filename_ordered_results(self) -> None:
        ordered = results_for_day(list(reversed(make_day(1))), 1)

        self.assertEqual(16, len(ordered))
        self.assertEqual(
            [
                (point, reading_type)
                for point in range(1, 9)
                for reading_type in ("temperature", "humidity")
            ],
            [(result.point, result.reading_type) for result in ordered],
        )
        self.assertEqual(
            "day_01_point_01_temperature.png",
            ordered[0].filename,
        )

    def test_incomplete_day_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 16"):
            results_for_day(make_day(1)[:-1], 1)


class DaySubmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = make_day(1) + make_day(2)
        self.state = DayVerificationState("sheet-a")

    def test_partial_day_correction_leaves_unrelated_readings_untouched(self) -> None:
        target = self.results[0]
        before = {result.filename: result for result in self.results}
        outcome = apply_day_submission(
            self.results,
            self.state,
            1,
            {target.filename: {"value": "23.0", "is_blank": False}},
            confirm_day=False,
        )

        updated = {result.filename: result for result in outcome.results}
        self.assertEqual("23.0", updated[target.filename].final_value)
        self.assertTrue(updated[target.filename].human_verified)
        self.assertFalse(outcome.day_confirmed)
        self.assertFalse(outcome.state.confirmed_days)
        for filename, result in before.items():
            if filename != target.filename:
                self.assertEqual(result, updated[filename])

    def test_blank_confirmation_clears_old_value(self) -> None:
        temperature = self.results[0]
        humidity = self.results[1]
        outcome = apply_day_submission(
            self.results,
            self.state,
            1,
            {
                temperature.filename: {"value": "22.0", "is_blank": True},
                humidity.filename: {"value": "50.0", "is_blank": True},
            },
            confirm_day=False,
        )
        updated = {result.filename: result for result in outcome.results}
        self.assertEqual("", updated[temperature.filename].final_value)
        self.assertEqual("", updated[humidity.filename].final_value)
        self.assertTrue(updated[temperature.filename].is_blank)
        self.assertTrue(updated[humidity.filename].is_blank)

    def test_one_sided_blank_mismatch_prevents_confirmation(self) -> None:
        temperature = self.results[0]
        outcome = apply_day_submission(
            self.results,
            self.state,
            1,
            {temperature.filename: {"value": "22.0", "is_blank": True}},
            confirm_day=True,
            advance_after_confirmation=True,
        )
        self.assertFalse(outcome.day_confirmed)
        self.assertNotIn(1, outcome.state.confirmed_days)
        self.assertEqual(1, outcome.next_day)
        self.assertTrue(
            any(result.has_blank_mismatch for result in results_for_day(outcome.results, 1))
        )

    def test_malformed_value_prevents_confirmation(self) -> None:
        target = self.results[0]
        outcome = apply_day_submission(
            self.results,
            self.state,
            1,
            {target.filename: {"value": "22.", "is_blank": False}},
            confirm_day=True,
        )
        self.assertTrue(outcome.errors)
        self.assertFalse(outcome.day_confirmed)
        self.assertNotIn(1, outcome.state.confirmed_days)

    def test_operational_warning_allows_confirmation(self) -> None:
        results = make_day(1, temperature="30.0")
        outcome = apply_day_submission(
            results,
            self.state,
            1,
            {},
            confirm_day=True,
        )
        self.assertTrue(outcome.day_confirmed)
        self.assertIn(1, outcome.state.confirmed_days)
        self.assertTrue(
            any(result.operational_warnings for result in outcome.results)
        )

    def test_confirm_and_next_advances_to_next_unverified_day(self) -> None:
        state = replace(self.state, confirmed_days=frozenset({2, 3}))
        outcome = apply_day_submission(
            self.results,
            state,
            1,
            {},
            confirm_day=True,
            advance_after_confirmation=True,
        )
        self.assertTrue(outcome.day_confirmed)
        self.assertEqual(4, outcome.next_day)

    def test_day_confirmation_accepts_valid_inferred_decimal(self) -> None:
        inferred = replace(
            self.results[0],
            final_value="49.3",
            postprocessing_status="decimal_inferred",
            ocr_uncertainty_reasons=(
                "Decimal point inferred from three-digit OCR text.",
            ),
            human_verified=False,
        )
        results = verify_cell_results([inferred, *self.results[1:]])

        outcome = apply_day_submission(
            results,
            self.state,
            1,
            {},
            confirm_day=True,
            advance_after_confirmation=True,
        )

        self.assertTrue(outcome.day_confirmed)
        confirmed = next(
            result
            for result in outcome.results
            if result.filename == inferred.filename
        )
        self.assertTrue(confirmed.human_verified)
        self.assertEqual("49.3", confirmed.final_value)

    def test_save_day_stays_on_current_day(self) -> None:
        target = self.results[0]
        outcome = apply_day_submission(
            self.results,
            self.state,
            1,
            {target.filename: {"value": "23.0", "is_blank": False}},
            confirm_day=False,
            advance_after_confirmation=False,
        )

        self.assertEqual(1, outcome.next_day)
        self.assertFalse(outcome.day_confirmed)

    def test_navigation_does_not_run_ocr(self) -> None:
        with patch(
            "datacenter_ocr.ocr_processing."
            "process_measurement_cells_with_blank_detection"
        ) as ocr:
            self.assertEqual(1, previous_day(1))
            self.assertEqual(2, next_unverified_day(1, {1}))
            apply_day_submission(
                self.results,
                self.state,
                1,
                {},
                confirm_day=False,
            )
        ocr.assert_not_called()


class DayStateAndExportTests(unittest.TestCase):
    def test_one_time_scroll_request_matches_sheet_and_destination(self) -> None:
        request = build_day_scroll_request("sheet-a", 2)

        should_scroll, remaining = consume_day_scroll_request(
            request,
            "sheet-a",
            2,
        )

        self.assertTrue(should_scroll)
        self.assertIsNone(remaining)

    def test_scroll_request_waits_for_destination_and_discards_other_sheet(
        self,
    ) -> None:
        request = build_day_scroll_request("sheet-a", 2)
        should_scroll, remaining = consume_day_scroll_request(
            request,
            "sheet-a",
            1,
        )
        self.assertFalse(should_scroll)
        self.assertEqual(request, remaining)

        should_scroll, remaining = consume_day_scroll_request(
            request,
            "sheet-b",
            2,
        )
        self.assertFalse(should_scroll)
        self.assertIsNone(remaining)

    def test_edit_invalidates_prior_confirmation_only_for_changed_day(self) -> None:
        results = make_day(1) + make_day(2)
        state = DayVerificationState("sheet-a", frozenset({1, 2}))
        changed_filename = results_for_day(results, 1)[0].filename

        invalidated = invalidate_days_for_changes(
            state,
            results,
            [changed_filename],
        )

        self.assertEqual(frozenset({2}), invalidated.confirmed_days)

    def test_session_state_is_separated_by_sheet_fingerprint(self) -> None:
        current = DayVerificationState("sheet-a", frozenset({1, 2, 3}))
        self.assertIs(current, state_for_sheet("sheet-a", current))
        replacement = state_for_sheet("sheet-b", current)
        self.assertEqual("sheet-b", replacement.sheet_fingerprint)
        self.assertFalse(replacement.confirmed_days)

    def test_export_blocked_until_all_days_are_confirmed(self) -> None:
        results = make_sheet()
        readiness = summarize_export_readiness(
            results,
            DayVerificationState("sheet-a", frozenset(range(1, 31))),
        )
        self.assertFalse(readiness.ready)
        self.assertEqual((31,), readiness.unconfirmed_days)

    def test_export_allowed_after_all_days_and_blockers_resolve(self) -> None:
        results = make_sheet()
        readiness = summarize_export_readiness(
            results,
            DayVerificationState("sheet-a", frozenset(range(1, 32))),
        )
        self.assertTrue(readiness.ready)
        self.assertFalse(readiness.unconfirmed_days)
        self.assertEqual(0, readiness.blocking_cell_count)
        self.assertEqual(0, readiness.blank_mismatch_count)


if __name__ == "__main__":
    unittest.main()
