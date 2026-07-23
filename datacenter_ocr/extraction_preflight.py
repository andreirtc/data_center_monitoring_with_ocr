from __future__ import annotations

from typing import Any, Sequence

from datacenter_ocr.sheet_processing import PreparedMonitoringSheet


REPRESENTATIVE_DAYS = (1, 16, 31)
REPRESENTATIVE_POINT = 4


def build_alignment_preflight_summary(
    fixed_sheet: PreparedMonitoringSheet,
    calibrated_sheet: PreparedMonitoringSheet,
    alignment_report: dict[str, Any],
) -> dict[str, Any]:
    """Build concise geometry-only metrics and warnings for user selection."""

    calibration = calibrated_sheet.grid_calibration
    if calibration is None:
        raise ValueError("The calibrated extraction preview has no calibration.")

    fallback_count = (
        calibration.fallback_horizontal_boundary_count
        + calibration.fallback_vertical_boundary_count
    )
    low_confidence_count = (
        calibration.low_confidence_horizontal_boundary_count
        + calibration.low_confidence_vertical_boundary_count
    )
    warnings: list[str] = []
    notices: list[str] = []
    row_alignment = fixed_sheet.row_sequence_alignment
    if row_alignment is not None and row_alignment.used_detected_alignment:
        notices.append(
            "The straight grid was anchored to a complete printed Day 1-31 "
            f"sequence ({row_alignment.strong_boundary_count}/32 strong "
            "boundaries)."
        )
    elif row_alignment is not None and row_alignment.fallback_reason:
        warnings.append(
            "Printed row-sequence evidence was not strong enough, so the "
            f"reference row span was retained: {row_alignment.fallback_reason}"
        )
    if calibrated_sheet.invalid_geometry_cell_count:
        warnings.append(
            f"{calibrated_sheet.invalid_geometry_cell_count} calibrated cell(s) "
            "failed geometry validation and use fixed fallback crops."
        )
    if fallback_count:
        warnings.append(
            f"{fallback_count} local boundary estimate(s) use fixed fallback."
        )
    if low_confidence_count:
        warnings.append(
            f"{low_confidence_count} local boundary estimate(s) have low confidence."
        )
    if not warnings:
        notices.append(
            "No calibrated-cell rejection, fallback, or low-confidence warning "
            "was detected. Visual verification is still required."
        )

    return {
        "fixed_cell_count": len(fixed_sheet.cells),
        "calibrated_cell_count": len(calibrated_sheet.cells),
        "provisional_alignment_score": alignment_report.get(
            "provisional_alignment_score"
        ),
        "matched_horizontal_lines": alignment_report.get(
            "matched_horizontal_line_count"
        ),
        "expected_horizontal_lines": alignment_report.get(
            "expected_horizontal_line_count"
        ),
        "matched_vertical_lines": alignment_report.get(
            "matched_vertical_line_count"
        ),
        "expected_vertical_lines": alignment_report.get(
            "expected_vertical_line_count"
        ),
        "fallback_boundary_count": fallback_count,
        "low_confidence_boundary_count": low_confidence_count,
        "invalid_calibrated_cell_count": (
            calibrated_sheet.invalid_geometry_cell_count
        ),
        "row_sequence_alignment_used": bool(
            row_alignment and row_alignment.used_detected_alignment
        ),
        "warnings": tuple(warnings),
        "notices": tuple(notices),
        "notice": alignment_report.get("alignment_score_notice", ""),
    }


def representative_crop_pairs(
    fixed_sheet: PreparedMonitoringSheet,
    calibrated_sheet: PreparedMonitoringSheet,
) -> list[dict[str, Any]]:
    """Return stable fixed/calibrated crop pairs from top, middle, and bottom."""

    fixed_lookup = {cell["filename"]: cell for cell in fixed_sheet.cells}
    calibrated_lookup = {
        cell["filename"]: cell for cell in calibrated_sheet.cells
    }
    pairs: list[dict[str, Any]] = []
    for day in REPRESENTATIVE_DAYS:
        for reading_type in ("temperature", "humidity"):
            filename = (
                f"day_{day:02d}_point_{REPRESENTATIVE_POINT:02d}_"
                f"{reading_type}.png"
            )
            try:
                fixed_cell = fixed_lookup[filename]
                calibrated_cell = calibrated_lookup[filename]
            except KeyError as error:
                raise ValueError(
                    f"Missing representative extraction crop: {error.args[0]}"
                ) from error
            pairs.append(
                {
                    "day": day,
                    "position": (
                        "Top" if day == 1 else "Middle" if day == 16 else "Bottom"
                    ),
                    "reading_type": reading_type,
                    "filename": filename,
                    "fixed_image": fixed_cell["image"],
                    "calibrated_image": calibrated_cell["image"],
                }
            )
    return pairs


def geometry_warning_for_filename(
    cells: Sequence[dict[str, Any]],
    filename: str,
) -> str:
    """Return a concise calibrated-geometry warning for one stable filename."""

    cell = next((item for item in cells if item.get("filename") == filename), None)
    if cell is None:
        return "Extracted crop is missing for this reading."
    rejection_reason = str(cell.get("geometry_rejection_reason", "")).strip()
    if rejection_reason:
        return f"Calibrated crop rejected: {rejection_reason}"
    if bool(cell.get("uses_fallback_geometry", False)):
        return "Calibrated extraction uses one or more fixed fallback boundaries."
    return ""
