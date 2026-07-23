from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from datacenter_ocr.ocr_processing import CellOCRResult
from datacenter_ocr.review_workflow import CellPatch, apply_sparse_patches
from datacenter_ocr.verification import validate_reading_value


EXPECTED_DAY_COUNT = 31
EXPECTED_POINT_COUNT = 8
EXPECTED_RESULTS_PER_DAY = EXPECTED_POINT_COUNT * 2


@dataclass(frozen=True)
class DayVerificationState:
    """Human-confirmed days belonging to one uploaded sheet fingerprint."""

    sheet_fingerprint: str
    confirmed_days: frozenset[int] = frozenset()


@dataclass(frozen=True)
class DaySubmissionOutcome:
    """Canonical result and navigation state after one day-form submission."""

    results: list[CellOCRResult]
    state: DayVerificationState
    errors: tuple[str, ...]
    changed_filenames: tuple[str, ...]
    day_confirmed: bool
    next_day: int


@dataclass(frozen=True)
class ExportReadiness:
    """Day-level and cell-level conditions required for Excel export."""

    confirmed_days: tuple[int, ...]
    unconfirmed_days: tuple[int, ...]
    blocking_cell_count: int
    attention_item_count: int
    blank_mismatch_count: int
    operational_warning_count: int
    invalid_final_value_count: int
    ready: bool


@dataclass(frozen=True)
class CellDisplayStatus:
    """Compact human-verification status for one measurement cell."""

    label: str
    reason: str


def state_for_sheet(
    sheet_fingerprint: str,
    current_state: DayVerificationState | None = None,
) -> DayVerificationState:
    """Return existing state only when it belongs to the current sheet."""

    if (
        current_state is not None
        and current_state.sheet_fingerprint == sheet_fingerprint
    ):
        return current_state
    return DayVerificationState(sheet_fingerprint=sheet_fingerprint)


def results_for_day(
    results: Sequence[CellOCRResult],
    day: int,
) -> list[CellOCRResult]:
    """Return exactly eight temperature/humidity pairs in form order."""

    if not 1 <= day <= EXPECTED_DAY_COUNT:
        raise ValueError(f"Day must be between 1 and {EXPECTED_DAY_COUNT}.")

    day_results = [result for result in results if result.day == day]
    identities = {
        (result.point, result.reading_type) for result in day_results
    }
    expected_identities = {
        (point, reading_type)
        for point in range(1, EXPECTED_POINT_COUNT + 1)
        for reading_type in ("temperature", "humidity")
    }
    if (
        len(day_results) != EXPECTED_RESULTS_PER_DAY
        or identities != expected_identities
    ):
        raise ValueError(
            f"Day {day} must contain exactly {EXPECTED_RESULTS_PER_DAY} "
            "unique temperature/humidity results."
        )

    reading_order = {"temperature": 0, "humidity": 1}
    return sorted(
        day_results,
        key=lambda result: (result.point, reading_order[result.reading_type]),
    )


def previous_day(day: int) -> int:
    """Move backward without wrapping before Day 1."""

    return max(1, min(EXPECTED_DAY_COUNT, day) - 1)


def next_unverified_day(
    day: int,
    confirmed_days: Iterable[int],
) -> int:
    """Find the next unconfirmed day, wrapping once when necessary."""

    current_day = max(1, min(EXPECTED_DAY_COUNT, day))
    confirmed = set(confirmed_days)
    candidates = list(range(current_day + 1, EXPECTED_DAY_COUNT + 1))
    candidates.extend(range(1, current_day + 1))
    return next(
        (candidate for candidate in candidates if candidate not in confirmed),
        current_day,
    )


def invalidate_days_for_changes(
    state: DayVerificationState,
    results: Sequence[CellOCRResult],
    changed_filenames: Iterable[str],
) -> DayVerificationState:
    """Require reconfirmation for days changed through any editing surface."""

    changed = set(changed_filenames)
    changed_days = {
        result.day for result in results if result.filename in changed
    }
    if not changed_days:
        return state
    return DayVerificationState(
        sheet_fingerprint=state.sheet_fingerprint,
        confirmed_days=frozenset(state.confirmed_days - changed_days),
    )


def day_is_confirmable(
    results: Sequence[CellOCRResult],
    day: int,
) -> bool:
    """Return whether all 16 readings can be explicitly confirmed as a day."""

    try:
        day_results = results_for_day(results, day)
    except ValueError:
        return False

    for result in day_results:
        validation = validate_reading_value(
            "" if result.is_blank else result.final_value,
            result.reading_type,
            allow_blank=result.is_blank,
        )
        if (
            validation.error is not None
            or result.blocks_export
            or result.has_blank_mismatch
            or not result.human_verified
        ):
            return False
    return True


def _patch_for_day_control(
    result: CellOCRResult,
    control: Mapping[str, Any] | None,
    *,
    confirm_day: bool,
) -> CellPatch | None:
    if control is None:
        return (
            CellPatch(result.filename, human_verified=True)
            if confirm_day
            else None
        )

    proposed_blank = bool(control.get("is_blank", result.is_blank))
    proposed_value = str(control.get("value", result.final_value)).strip()
    value_changed = proposed_value != result.final_value
    blank_changed = proposed_blank != result.is_blank
    if not (confirm_day or value_changed or blank_changed):
        return None

    return CellPatch(
        filename=result.filename,
        final_value=proposed_value,
        is_blank=proposed_blank,
        human_verified=True,
    )


def apply_day_submission(
    results: list[CellOCRResult],
    state: DayVerificationState,
    day: int,
    controls: Mapping[str, Mapping[str, Any]],
    *,
    confirm_day: bool,
    advance_after_confirmation: bool = False,
) -> DaySubmissionOutcome:
    """Apply current-day controls through the canonical sparse patch engine."""

    current_day_results = results_for_day(results, day)
    current_filenames = {result.filename for result in current_day_results}
    unknown_filenames = sorted(set(controls) - current_filenames)
    input_errors = tuple(
        f"Day {day} form contains an unknown reading: {filename}"
        for filename in unknown_filenames
    )
    patches = [
        patch
        for result in current_day_results
        if (
            patch := _patch_for_day_control(
                result,
                controls.get(result.filename),
                confirm_day=confirm_day,
            )
        )
        is not None
    ]

    update = apply_sparse_patches(results, patches)
    errors = tuple(dict.fromkeys(input_errors + update.errors))
    next_state = invalidate_days_for_changes(
        state,
        update.results,
        update.changed_filenames,
    )
    confirmed = False
    if confirm_day and not errors and day_is_confirmable(update.results, day):
        next_state = DayVerificationState(
            sheet_fingerprint=state.sheet_fingerprint,
            confirmed_days=frozenset(set(next_state.confirmed_days) | {day}),
        )
        confirmed = True

    target_day = (
        next_unverified_day(day, next_state.confirmed_days)
        if confirmed and advance_after_confirmation
        else day
    )
    return DaySubmissionOutcome(
        results=update.results,
        state=next_state,
        errors=errors,
        changed_filenames=update.changed_filenames,
        day_confirmed=confirmed,
        next_day=target_day,
    )


def summarize_export_readiness(
    results: Sequence[CellOCRResult],
    state: DayVerificationState,
) -> ExportReadiness:
    """Summarize explicit day confirmation and existing export blockers."""

    applicable_days = tuple(range(1, EXPECTED_DAY_COUNT + 1))
    confirmed_days = tuple(
        day for day in applicable_days if day in state.confirmed_days
    )
    unconfirmed_days = tuple(
        day for day in applicable_days if day not in state.confirmed_days
    )
    blocking_cell_count = sum(bool(result.blocking_errors) for result in results)
    attention_item_count = sum(
        bool(result.required_confirmation_reasons) and not result.human_verified
        for result in results
    )
    mismatch_rows = {
        (result.day, result.point)
        for result in results
        if result.has_blank_mismatch
    }
    operational_warning_count = sum(
        bool(result.operational_warnings) for result in results
    )
    invalid_final_value_count = 0
    for result in results:
        validation = validate_reading_value(
            "" if result.is_blank else result.final_value,
            result.reading_type,
            allow_blank=result.is_blank,
        )
        if validation.error is not None:
            invalid_final_value_count += 1

    unresolved_export_cells = sum(result.blocks_export for result in results)
    ready = not any(
        (
            unconfirmed_days,
            blocking_cell_count,
            attention_item_count,
            mismatch_rows,
            invalid_final_value_count,
            unresolved_export_cells,
        )
    )
    return ExportReadiness(
        confirmed_days=confirmed_days,
        unconfirmed_days=unconfirmed_days,
        blocking_cell_count=blocking_cell_count,
        attention_item_count=attention_item_count,
        blank_mismatch_count=len(mismatch_rows),
        operational_warning_count=operational_warning_count,
        invalid_final_value_count=invalid_final_value_count,
        ready=ready,
    )


def cell_display_status(
    result: CellOCRResult,
    *,
    day_confirmed: bool,
    geometry_warning: str = "",
) -> CellDisplayStatus:
    """Choose one compact status without hiding secondary reasons."""

    if result.blocking_errors or result.has_blank_mismatch:
        return CellDisplayStatus(
            "Blocking error",
            " ".join(result.blocking_errors) or result.review_reason,
        )
    if result.is_blank:
        reason = "Verified blank." if day_confirmed else "Blank; confirm this day."
        return CellDisplayStatus("Blank", reason)
    if (
        (result.required_confirmation_reasons and not result.human_verified)
        or geometry_warning
    ):
        reasons = (
            list(result.required_confirmation_reasons)
            + ([geometry_warning] if geometry_warning else [])
        )
        return CellDisplayStatus("Needs attention", " ".join(reasons))
    if result.operational_warnings:
        return CellDisplayStatus(
            "Operational warning",
            " ".join(result.operational_warnings),
        )
    if day_confirmed:
        return CellDisplayStatus("Confirmed", "Day explicitly confirmed.")
    return CellDisplayStatus("Needs attention", "Awaiting day confirmation.")


def build_verification_audit(
    state: DayVerificationState,
    readiness: ExportReadiness,
    geometry_mode: str,
) -> dict[str, Any]:
    """Return display/download audit data without changing the Excel template."""

    return {
        "schema_version": 1,
        "sheet_fingerprint": state.sheet_fingerprint,
        "geometry_mode": geometry_mode,
        "confirmed_days": list(readiness.confirmed_days),
        "unconfirmed_days": list(readiness.unconfirmed_days),
        "blocking_cell_count": readiness.blocking_cell_count,
        "attention_item_count": readiness.attention_item_count,
        "blank_mismatch_count": readiness.blank_mismatch_count,
        "operational_warning_count": readiness.operational_warning_count,
        "invalid_final_value_count": readiness.invalid_final_value_count,
        "export_ready": readiness.ready,
    }
