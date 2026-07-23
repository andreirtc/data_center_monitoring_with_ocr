from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from datacenter_ocr.ocr_processing import CellOCRResult
from datacenter_ocr.verification import validate_reading_value, verify_cell_results


_UNSET = object()


@dataclass(frozen=True)
class CellPatch:
    """A sparse update for one canonical OCR result."""

    filename: str
    final_value: object = _UNSET
    is_blank: bool | None = None
    human_verified: bool | None = None


@dataclass(frozen=True)
class UpdateOutcome:
    """Result of applying valid sparse updates and rejecting invalid ones."""

    results: list[CellOCRResult]
    errors: tuple[str, ...] = ()
    changed_count: int = 0
    changed_filenames: tuple[str, ...] = ()


def _cell_label(result: CellOCRResult) -> str:
    return (
        f"Day {result.day}, Point {result.point}, "
        f"{result.reading_type.title()}"
    )


def apply_sparse_patches(
    results: list[CellOCRResult],
    patches: list[CellPatch],
) -> UpdateOutcome:
    """Apply valid filename-keyed patches independently, then reverify once."""

    updated_by_filename = {result.filename: result for result in results}
    errors: list[str] = []
    changed_filenames: list[str] = []

    for patch in patches:
        current = updated_by_filename.get(patch.filename)
        if current is None:
            errors.append(f"Unknown OCR result: {patch.filename}")
            continue

        proposed_is_blank = (
            current.is_blank if patch.is_blank is None else patch.is_blank
        )
        proposed_value = current.final_value

        if proposed_is_blank:
            proposed_value = ""
        elif patch.final_value is not _UNSET:
            validation = validate_reading_value(
                patch.final_value,
                current.reading_type,
                allow_blank=False,
            )
            if validation.error is not None or validation.normalized_value is None:
                errors.append(
                    f"{_cell_label(current)}: "
                    f"{validation.error or 'Enter a value.'}"
                )
                continue
            proposed_value = validation.normalized_value
        elif current.is_blank and patch.is_blank is False:
            errors.append(
                f"{_cell_label(current)}: Enter a value when removing blank status."
            )
            continue

        proposed_human_verified = (
            current.human_verified
            if patch.human_verified is None
            else patch.human_verified
        )
        updated = replace(
            current,
            final_value=proposed_value,
            is_blank=proposed_is_blank,
            human_verified=proposed_human_verified,
            postprocessing_status=(
                "manually_corrected"
                if (
                    proposed_value != current.final_value
                    or proposed_is_blank != current.is_blank
                )
                else current.postprocessing_status
            ),
        )

        if updated == current:
            continue

        updated_by_filename[patch.filename] = updated
        if patch.filename not in changed_filenames:
            changed_filenames.append(patch.filename)

    updated_results = [
        updated_by_filename[result.filename]
        for result in results
    ]
    return UpdateOutcome(
        results=verify_cell_results(updated_results),
        errors=tuple(dict.fromkeys(errors)),
        changed_count=len(changed_filenames),
        changed_filenames=tuple(changed_filenames),
    )


def apply_review_actions(
    results: list[CellOCRResult],
    actions: dict[str, tuple[str, object]],
) -> UpdateOutcome:
    """Apply only explicit, non-default review actions as sparse patches."""

    result_lookup = {result.filename: result for result in results}
    patches: list[CellPatch] = []
    errors: list[str] = []

    for filename, (action, corrected_value) in actions.items():
        if action == "Leave unresolved":
            continue

        current = result_lookup.get(filename)
        if current is None:
            errors.append(f"Unknown review item: {filename}")
            continue

        if action == "Confirm current":
            if not current.is_blank:
                validation = validate_reading_value(
                    current.final_value,
                    current.reading_type,
                    allow_blank=False,
                )
                if validation.error is not None:
                    errors.append(f"{_cell_label(current)}: {validation.error}")
                    continue
            patches.append(CellPatch(filename, human_verified=True))
        elif action == "Enter correction":
            patches.append(
                CellPatch(
                    filename,
                    final_value=corrected_value,
                    is_blank=False,
                    human_verified=True,
                )
            )
        elif action == "Mark blank":
            patches.append(
                CellPatch(filename, is_blank=True, human_verified=True)
            )
        else:
            errors.append(f"{_cell_label(current)}: Unknown action {action!r}.")

    outcome = apply_sparse_patches(results, patches)
    return replace(
        outcome,
        errors=tuple(dict.fromkeys(errors + list(outcome.errors))),
    )


def apply_quick_review_controls(
    results: list[CellOCRResult],
    controls: dict[str, dict[str, object]],
) -> UpdateOutcome:
    """Interpret keyboard-friendly quick-review controls and apply valid items.

    Correction text, confirmation, and blank selection are mutually exclusive.
    Items with no selected control are intentionally left unresolved.
    """

    result_lookup = {result.filename: result for result in results}
    actions: dict[str, tuple[str, object]] = {}
    errors: list[str] = []

    for filename, item_controls in controls.items():
        current = result_lookup.get(filename)
        if current is None:
            errors.append(f"Unknown review item: {filename}")
            continue

        corrected_value = str(item_controls.get("corrected_value", "")).strip()
        confirm_current = bool(item_controls.get("confirm_current", False))
        mark_blank = bool(item_controls.get("mark_blank", False))
        selected_count = sum(
            (bool(corrected_value), confirm_current, mark_blank)
        )

        if selected_count > 1:
            errors.append(
                f"{_cell_label(current)}: Choose only one of correction, "
                "confirm proposed, or mark blank."
            )
        elif corrected_value:
            actions[filename] = ("Enter correction", corrected_value)
        elif confirm_current:
            actions[filename] = ("Confirm current", "")
        elif mark_blank:
            actions[filename] = ("Mark blank", "")

    outcome = apply_review_actions(results, actions)
    return replace(
        outcome,
        errors=tuple(dict.fromkeys(errors + list(outcome.errors))),
    )


def apply_review_action(
    results: list[CellOCRResult],
    filename: str,
    action: str,
    corrected_value: object = "",
) -> UpdateOutcome:
    """Compatibility wrapper for applying one review action."""

    return apply_review_actions(
        results,
        {filename: (action, corrected_value)},
    )


def apply_monitoring_table_edits(
    results: list[CellOCRResult],
    edited_rows: list[dict[str, Any]],
) -> UpdateOutcome:
    """Convert sparse changed table fields into canonical filename patches."""

    result_lookup = {
        (result.day, result.point, result.reading_type): result
        for result in results
    }
    patches: list[CellPatch] = []
    errors: list[str] = []

    for row in edited_rows:
        day = int(row["Day"])
        point = int(row["Point"])

        for reading_type, value_column, blank_column in (
            ("temperature", "Temperature", "Temperature Blank"),
            ("humidity", "Humidity", "Humidity Blank"),
        ):
            value_changed = value_column in row
            blank_changed = blank_column in row
            if not value_changed and not blank_changed:
                continue

            current = result_lookup.get((day, point, reading_type))
            if current is None:
                errors.append(
                    f"Day {day}, Point {point}, {reading_type.title()}: "
                    "Unknown monitoring reading."
                )
                continue

            proposed_is_blank = (
                bool(row[blank_column]) if blank_changed else current.is_blank
            )
            proposed_value = row[value_column] if value_changed else _UNSET
            patches.append(
                CellPatch(
                    filename=current.filename,
                    final_value=proposed_value,
                    is_blank=proposed_is_blank,
                    human_verified=True,
                )
            )

    outcome = apply_sparse_patches(results, patches)
    return replace(
        outcome,
        errors=tuple(dict.fromkeys(errors + list(outcome.errors))),
    )


def clamp_review_state(
    filenames: list[str],
    page: int,
    selected_filename: str | None,
    *,
    page_size: int = 15,
) -> tuple[int, str | None]:
    """Clamp review pagination and detailed selection after queue changes."""

    if not filenames:
        return 1, None

    page_count = max(1, (len(filenames) + page_size - 1) // page_size)
    clamped_page = min(max(page, 1), page_count)
    if selected_filename in filenames:
        return clamped_page, selected_filename

    page_start = (clamped_page - 1) * page_size
    return clamped_page, filenames[page_start]
