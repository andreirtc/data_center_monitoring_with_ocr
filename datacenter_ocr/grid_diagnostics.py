from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np

from datacenter_ocr.image_processing import load_image, save_image
from datacenter_ocr.sheet_processing import (
    PreparedMonitoringSheet,
    prepare_monitoring_sheet,
)


DIAGNOSTIC_DAYS = (1, 16, 31)
DIAGNOSTIC_POINTS = (1, 4, 8)

WARPED_TABLE_FILENAME = "warped_table.png"
GRID_OVERLAY_FILENAME = "measurement_grid_overlay.png"
LINE_DIAGNOSTIC_OVERLAY_FILENAME = "grid_line_diagnostic_overlay.png"
CONTACT_SHEET_FILENAME = "diagnostic_contact_sheet.png"
ALIGNMENT_REPORT_FILENAME = "alignment_report.json"


@dataclass(frozen=True)
class GridDiagnosticOutputs:
    """Paths written by one geometry-only diagnostic run."""

    warped_table: Path
    measurement_grid_overlay: Path
    line_diagnostic_overlay: Path
    contact_sheet: Path
    alignment_report: Path


def diagnostic_cell_keys() -> tuple[tuple[int, int, str], ...]:
    """Return the stable 18-cell selection used by the contact sheet."""

    return tuple(
        (day, point, reading_type)
        for day in DIAGNOSTIC_DAYS
        for point in DIAGNOSTIC_POINTS
        for reading_type in ("temperature", "humidity")
    )


def _expected_boundaries(
    measurement_boxes: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    x_positions = sorted(
        {
            int(position)
            for box in measurement_boxes
            for position in (box["x1"], box["x2"])
        }
    )
    y_positions = sorted(
        {
            int(position)
            for box in measurement_boxes
            for position in (box["y1"], box["y2"])
        }
    )
    return x_positions, y_positions


def _line_masks(image: np.ndarray, row_height: float, column_width: float) -> tuple[np.ndarray, np.ndarray]:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, inverted = cv2.threshold(
        grayscale,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(12, round(column_width * 0.60)), 1),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(8, round(row_height * 0.65))),
    )
    horizontal_mask = cv2.morphologyEx(
        inverted,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )
    vertical_mask = cv2.morphologyEx(
        inverted,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )
    return horizontal_mask, vertical_mask


def _best_local_position(
    projection: np.ndarray,
    expected_position: int,
    search_radius: int,
    normalization_length: int,
    minimum_confidence: float,
) -> tuple[int | None, float]:
    lower = max(0, expected_position - search_radius)
    upper = min(len(projection), expected_position + search_radius + 1)
    if lower >= upper:
        return None, 0.0

    local_projection = projection[lower:upper]
    local_index = int(np.argmax(local_projection))
    detected_position = lower + local_index
    confidence = float(local_projection[local_index]) / max(
        normalization_length,
        1,
    )
    if confidence < minimum_confidence:
        return None, round(confidence, 6)
    return detected_position, round(min(confidence, 1.0), 6)


def _measure_horizontal_boundary(
    index: int,
    expected_position: int,
    horizontal_mask: np.ndarray,
    x_positions: list[int],
    row_height: float,
) -> dict[str, Any]:
    search_radius = max(3, round(row_height * 0.45))
    local_positions: list[dict[str, Any]] = []

    for point_index in range(8):
        x1 = x_positions[point_index * 2]
        x2 = x_positions[point_index * 2 + 2]
        projection = (
            horizontal_mask[:, x1:x2].sum(axis=1).astype(np.float64)
            / 255.0
        )
        detected, confidence = _best_local_position(
            projection=projection,
            expected_position=expected_position,
            search_radius=search_radius,
            normalization_length=x2 - x1,
            minimum_confidence=0.15,
        )
        if detected is not None:
            local_positions.append(
                {
                    "coordinate": round((x1 + x2) / 2),
                    "detected_position": detected,
                    "confidence": confidence,
                }
            )

    continuity = len(local_positions) / 8
    matched = len(local_positions) >= 4
    detected_position = (
        round(median(item["detected_position"] for item in local_positions))
        if matched
        else None
    )
    pixel_error = (
        detected_position - expected_position
        if detected_position is not None
        else None
    )
    confidence = (
        float(median(item["confidence"] for item in local_positions))
        * continuity
        if local_positions
        else 0.0
    )
    normalized_error = (
        pixel_error / row_height
        if pixel_error is not None
        else None
    )

    return {
        "index": index,
        "expected_position": expected_position,
        "detected_position": detected_position,
        "matched": matched,
        "low_confidence": matched and (
            continuity < 0.75
            or confidence < 0.25
            or (
                normalized_error is not None
                and abs(normalized_error) > 0.25
            )
        ),
        "pixel_error": pixel_error,
        "absolute_pixel_error": abs(pixel_error) if pixel_error is not None else None,
        "normalized_error": (
            round(normalized_error, 6)
            if normalized_error is not None
            else None
        ),
        "continuity": round(continuity, 6),
        "confidence": round(confidence, 6),
        "search_radius": search_radius,
        "local_positions": local_positions,
    }


def _measure_vertical_boundary(
    index: int,
    expected_position: int,
    vertical_mask: np.ndarray,
    y_positions: list[int],
    column_width: float,
) -> dict[str, Any]:
    search_radius = max(4, round(column_width * 0.35))
    data_top = y_positions[0]
    data_bottom = y_positions[-1]
    band_edges = np.linspace(data_top, data_bottom, 7).round().astype(int)
    local_positions: list[dict[str, Any]] = []

    for band_index in range(6):
        y1 = int(band_edges[band_index])
        y2 = int(band_edges[band_index + 1])
        projection = (
            vertical_mask[y1:y2, :].sum(axis=0).astype(np.float64)
            / 255.0
        )
        detected, confidence = _best_local_position(
            projection=projection,
            expected_position=expected_position,
            search_radius=search_radius,
            normalization_length=y2 - y1,
            minimum_confidence=0.12,
        )
        if detected is not None:
            local_positions.append(
                {
                    "coordinate": round((y1 + y2) / 2),
                    "detected_position": detected,
                    "confidence": confidence,
                }
            )

    continuity = len(local_positions) / 6
    matched = len(local_positions) >= 3
    detected_position = (
        round(median(item["detected_position"] for item in local_positions))
        if matched
        else None
    )
    pixel_error = (
        detected_position - expected_position
        if detected_position is not None
        else None
    )
    confidence = (
        float(median(item["confidence"] for item in local_positions))
        * continuity
        if local_positions
        else 0.0
    )
    normalized_error = (
        pixel_error / column_width
        if pixel_error is not None
        else None
    )

    return {
        "index": index,
        "expected_position": expected_position,
        "detected_position": detected_position,
        "matched": matched,
        "low_confidence": matched and (
            continuity < 0.75
            or confidence < 0.25
            or (
                normalized_error is not None
                and abs(normalized_error) > 0.25
            )
        ),
        "pixel_error": pixel_error,
        "absolute_pixel_error": abs(pixel_error) if pixel_error is not None else None,
        "normalized_error": (
            round(normalized_error, 6)
            if normalized_error is not None
            else None
        ),
        "continuity": round(continuity, 6),
        "confidence": round(confidence, 6),
        "search_radius": search_radius,
        "local_positions": local_positions,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values), percentile)), 6)


def _spacing_variation(boundaries: list[dict[str, Any]]) -> dict[str, float | int | None]:
    detected = [
        (boundary["index"], boundary["detected_position"])
        for boundary in boundaries
        if boundary["detected_position"] is not None
    ]
    spacings = [
        float(second_position - first_position)
        for (first_index, first_position), (second_index, second_position) in zip(
            detected,
            detected[1:],
        )
        if second_index == first_index + 1
    ]
    if not spacings:
        return {
            "adjacent_spacing_count": 0,
            "mean_pixels": None,
            "standard_deviation_pixels": None,
            "coefficient_of_variation": None,
        }
    mean_spacing = float(np.mean(spacings))
    standard_deviation = float(np.std(spacings))
    return {
        "adjacent_spacing_count": len(spacings),
        "mean_pixels": round(mean_spacing, 6),
        "standard_deviation_pixels": round(standard_deviation, 6),
        "coefficient_of_variation": round(
            standard_deviation / mean_spacing if mean_spacing else 0.0,
            6,
        ),
    }


def _boundary_summary(boundaries: list[dict[str, Any]]) -> dict[str, Any]:
    absolute_errors = [
        float(boundary["absolute_pixel_error"])
        for boundary in boundaries
        if boundary["absolute_pixel_error"] is not None
    ]
    normalized_errors = [
        abs(float(boundary["normalized_error"]))
        for boundary in boundaries
        if boundary["normalized_error"] is not None
    ]
    continuities = [float(boundary["continuity"]) for boundary in boundaries]
    return {
        "expected_count": len(boundaries),
        "matched_count": sum(bool(boundary["matched"]) for boundary in boundaries),
        "unmatched_count": sum(not bool(boundary["matched"]) for boundary in boundaries),
        "low_confidence_count": sum(
            bool(boundary["low_confidence"]) for boundary in boundaries
        ),
        "median_error_pixels": _percentile(absolute_errors, 50),
        "p95_error_pixels": _percentile(absolute_errors, 95),
        "maximum_error_pixels": max(absolute_errors) if absolute_errors else None,
        "median_normalized_error": _percentile(normalized_errors, 50),
        "p95_normalized_error": _percentile(normalized_errors, 95),
        "maximum_normalized_error": (
            max(normalized_errors) if normalized_errors else None
        ),
        "mean_continuity": (
            round(float(np.mean(continuities)), 6) if continuities else 0.0
        ),
        "spacing_variation": _spacing_variation(boundaries),
    }


def _drift_summary(boundaries: list[dict[str, Any]]) -> dict[str, Any]:
    drift_values: list[float] = []
    per_boundary: list[dict[str, Any]] = []
    for boundary in boundaries:
        positions = boundary["local_positions"]
        drift = None
        if len(positions) >= 2:
            drift = float(
                positions[-1]["detected_position"]
                - positions[0]["detected_position"]
            )
            drift_values.append(abs(drift))
        per_boundary.append(
            {
                "index": boundary["index"],
                "signed_drift_pixels": drift,
            }
        )
    return {
        "per_boundary": per_boundary,
        "median_absolute_drift_pixels": _percentile(drift_values, 50),
        "p95_absolute_drift_pixels": _percentile(drift_values, 95),
        "maximum_absolute_drift_pixels": (
            max(drift_values) if drift_values else None
        ),
    }


def _provisional_score(
    horizontal_summary: dict[str, Any],
    vertical_summary: dict[str, Any],
) -> float:
    summaries = (horizontal_summary, vertical_summary)
    coverage = float(
        np.mean(
            [
                summary["matched_count"] / summary["expected_count"]
                for summary in summaries
            ]
        )
    )
    continuity = float(np.mean([summary["mean_continuity"] for summary in summaries]))
    median_errors = [
        (
            float(summary["median_normalized_error"])
            if summary["median_normalized_error"] is not None
            else 1.0
        )
        for summary in summaries
    ]
    error_quality = max(0.0, 1.0 - min(1.0, float(np.mean(median_errors)) / 0.35))
    spacing_values = [
        summary["spacing_variation"]["coefficient_of_variation"]
        for summary in summaries
        if summary["spacing_variation"]["coefficient_of_variation"] is not None
    ]
    spacing_quality = (
        max(0.0, 1.0 - min(1.0, float(np.mean(spacing_values)) / 0.20))
        if spacing_values
        else 0.0
    )
    score = 100.0 * (
        0.35 * coverage
        + 0.25 * continuity
        + 0.25 * error_quality
        + 0.15 * spacing_quality
    )
    return round(score, 2)


def calculate_alignment_report(
    prepared_sheet: PreparedMonitoringSheet,
) -> dict[str, Any]:
    """Measure nearby printed grid lines using fixed coordinates as priors."""

    image_height, image_width = prepared_sheet.warped_table.shape[:2]
    x_positions, y_positions = _expected_boundaries(
        prepared_sheet.measurement_boxes
    )
    row_height = float(median(np.diff(y_positions)))
    column_width = float(median(np.diff(x_positions)))
    horizontal_mask, vertical_mask = _line_masks(
        prepared_sheet.warped_table,
        row_height,
        column_width,
    )

    horizontal_boundaries = [
        _measure_horizontal_boundary(
            index,
            position,
            horizontal_mask,
            x_positions,
            row_height,
        )
        for index, position in enumerate(y_positions)
    ]
    vertical_boundaries = [
        _measure_vertical_boundary(
            index,
            position,
            vertical_mask,
            y_positions,
            column_width,
        )
        for index, position in enumerate(x_positions)
    ]
    horizontal_summary = _boundary_summary(horizontal_boundaries)
    vertical_summary = _boundary_summary(vertical_boundaries)
    row_alignment = prepared_sheet.row_sequence_alignment

    return {
        "schema_version": 1,
        "standardized_image_width": image_width,
        "standardized_image_height": image_height,
        "expected_row_height_pixels": round(row_height, 6),
        "expected_column_width_pixels": round(column_width, 6),
        "expected_horizontal_boundaries": y_positions,
        "detected_horizontal_boundaries": [
            boundary["detected_position"] for boundary in horizontal_boundaries
        ],
        "expected_vertical_boundaries": x_positions,
        "detected_vertical_boundaries": [
            boundary["detected_position"] for boundary in vertical_boundaries
        ],
        "matched_horizontal_line_count": horizontal_summary["matched_count"],
        "expected_horizontal_line_count": len(y_positions),
        "matched_vertical_line_count": vertical_summary["matched_count"],
        "expected_vertical_line_count": len(x_positions),
        "horizontal_boundaries": horizontal_boundaries,
        "vertical_boundaries": vertical_boundaries,
        "horizontal_measurements": horizontal_summary,
        "vertical_measurements": vertical_summary,
        "left_to_right_horizontal_row_drift": _drift_summary(
            horizontal_boundaries
        ),
        "top_to_bottom_vertical_column_drift": _drift_summary(
            vertical_boundaries
        ),
        "line_spacing_variation": {
            "horizontal": horizontal_summary["spacing_variation"],
            "vertical": vertical_summary["spacing_variation"],
        },
        "provisional_alignment_score": _provisional_score(
            horizontal_summary,
            vertical_summary,
        ),
        "alignment_score_is_calibrated": False,
        "alignment_score_notice": (
            "The alignment score is diagnostic and uncalibrated; the score "
            "itself never gates OCR or export. Strong printed-row evidence may "
            "still anchor the extraction span as reported above."
        ),
        "row_sequence_alignment": (
            {
                "used_detected_alignment": row_alignment.used_detected_alignment,
                "detected_top": row_alignment.detected_top,
                "detected_bottom": row_alignment.detected_bottom,
                "detected_spacing": row_alignment.detected_spacing,
                "strong_boundary_count": row_alignment.strong_boundary_count,
                "median_line_strength": row_alignment.median_line_strength,
                "fallback_reason": row_alignment.fallback_reason,
            }
            if row_alignment is not None
            else None
        ),
    }


def _boundary_color(boundary: dict[str, Any]) -> tuple[int, int, int]:
    if not boundary["matched"]:
        return (0, 0, 255)
    if boundary["low_confidence"]:
        return (0, 215, 255)
    return (0, 210, 0)


def draw_grid_line_diagnostic_overlay(
    prepared_sheet: PreparedMonitoringSheet,
    report: dict[str, Any],
) -> np.ndarray:
    """Draw expected and locally detected boundaries on the warped table."""

    overlay = prepared_sheet.warped_table.copy()
    x_positions = report["expected_vertical_boundaries"]
    y_positions = report["expected_horizontal_boundaries"]
    x_min, x_max = x_positions[0], x_positions[-1]
    y_min, y_max = y_positions[0], y_positions[-1]
    expected_color = (255, 255, 0)

    for boundary in report["horizontal_boundaries"]:
        expected_y = boundary["expected_position"]
        cv2.line(overlay, (x_min, expected_y), (x_max, expected_y), expected_color, 1)
        color = _boundary_color(boundary)
        points = np.asarray(
            [
                (item["coordinate"], item["detected_position"])
                for item in boundary["local_positions"]
            ],
            dtype=np.int32,
        )
        if len(points) >= 2:
            cv2.polylines(overlay, [points], False, color, 3)
        cv2.putText(
            overlay,
            f"H{boundary['index']:02d}",
            (x_min + 3, max(16, expected_y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    for boundary in report["vertical_boundaries"]:
        expected_x = boundary["expected_position"]
        cv2.line(overlay, (expected_x, y_min), (expected_x, y_max), expected_color, 1)
        color = _boundary_color(boundary)
        points = np.asarray(
            [
                (item["detected_position"], item["coordinate"])
                for item in boundary["local_positions"]
            ],
            dtype=np.int32,
        )
        if len(points) >= 2:
            cv2.polylines(overlay, [points], False, color, 3)
        cv2.putText(
            overlay,
            f"V{boundary['index']:02d}",
            (max(0, expected_x + 2), y_min + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        overlay,
        "Expected: cyan | detected: green | weak: yellow | unmatched: red",
        (max(8, x_min), max(24, y_min - 22)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return overlay


def _nearest_local_position(
    boundary: dict[str, Any],
    coordinate: int,
) -> int | None:
    positions = boundary["local_positions"]
    if not positions:
        return boundary["detected_position"]
    return min(
        positions,
        key=lambda item: abs(int(item["coordinate"]) - coordinate),
    )["detected_position"]


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            max(1, round(image.shape[1] * scale)),
            max(1, round(image.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    y_offset = (height - resized.shape[0]) // 2
    x_offset = (width - resized.shape[1]) // 2
    canvas[
        y_offset : y_offset + resized.shape[0],
        x_offset : x_offset + resized.shape[1],
    ] = resized
    return canvas


def _context_crop(
    prepared_sheet: PreparedMonitoringSheet,
    box: dict[str, Any],
    report: dict[str, Any],
) -> np.ndarray:
    width = box["x2"] - box["x1"]
    height = box["y2"] - box["y1"]
    image_height, image_width = prepared_sheet.warped_table.shape[:2]
    x1 = max(0, round(box["x1"] - width * 0.75))
    x2 = min(image_width, round(box["x2"] + width * 0.75))
    y1 = max(0, round(box["y1"] - height * 0.90))
    y2 = min(image_height, round(box["y2"] + height * 0.90))
    context = prepared_sheet.warped_table[y1:y2, x1:x2].copy()

    expected_color = (255, 255, 0)
    cv2.rectangle(
        context,
        (box["x1"] - x1, box["y1"] - y1),
        (box["x2"] - x1, box["y2"] - y1),
        expected_color,
        2,
    )
    x_boundaries = {
        boundary["expected_position"]: boundary
        for boundary in report["vertical_boundaries"]
    }
    y_boundaries = {
        boundary["expected_position"]: boundary
        for boundary in report["horizontal_boundaries"]
    }
    center_x = round((box["x1"] + box["x2"]) / 2)
    center_y = round((box["y1"] + box["y2"]) / 2)
    detected_color = (255, 0, 255)
    for expected_x in (box["x1"], box["x2"]):
        detected_x = _nearest_local_position(x_boundaries[expected_x], center_y)
        if detected_x is not None and x1 <= detected_x < x2:
            cv2.line(
                context,
                (detected_x - x1, 0),
                (detected_x - x1, context.shape[0] - 1),
                detected_color,
                2,
            )
    for expected_y in (box["y1"], box["y2"]):
        detected_y = _nearest_local_position(y_boundaries[expected_y], center_x)
        if detected_y is not None and y1 <= detected_y < y2:
            cv2.line(
                context,
                (0, detected_y - y1),
                (context.shape[1] - 1, detected_y - y1),
                detected_color,
                2,
            )
    return context


def create_diagnostic_contact_sheet(
    prepared_sheet: PreparedMonitoringSheet,
    report: dict[str, Any],
) -> np.ndarray:
    """Create the fixed 3x3 paired-crop geometry contact sheet."""

    cell_lookup = {
        (cell["day"], cell["point"], cell["reading_type"]): cell
        for cell in prepared_sheet.cells
    }
    tile_width = 940
    tile_height = 390
    header_height = 42
    row_height = 170
    tiles: list[np.ndarray] = []

    for day in DIAGNOSTIC_DAYS:
        for point in DIAGNOSTIC_POINTS:
            tile = np.full((tile_height, tile_width, 3), 255, dtype=np.uint8)
            cv2.putText(
                tile,
                f"Day {day:02d} | Point {point} | cyan expected | magenta detected",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.63,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            for reading_index, reading_type in enumerate(("temperature", "humidity")):
                cell = cell_lookup[(day, point, reading_type)]
                y_offset = header_height + reading_index * row_height
                cv2.putText(
                    tile,
                    f"{reading_type.title()} | {cell['filename']}",
                    (12, y_offset + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                normal_panel = _fit_image(cell["image"], 330, 132)
                context_panel = _fit_image(
                    _context_crop(prepared_sheet, cell, report),
                    560,
                    132,
                )
                tile[y_offset + 30 : y_offset + 162, 12:342] = normal_panel
                tile[y_offset + 30 : y_offset + 162, 365:925] = context_panel
                cv2.putText(
                    tile,
                    "normal OCR crop",
                    (14, y_offset + 158),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (90, 90, 90),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    tile,
                    "wider context",
                    (367, y_offset + 158),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (90, 90, 90),
                    1,
                    cv2.LINE_AA,
                )
            cv2.rectangle(tile, (0, 0), (tile_width - 1, tile_height - 1), (70, 70, 70), 2)
            tiles.append(tile)

    contact_rows = [
        cv2.hconcat(tiles[row_start : row_start + 3])
        for row_start in range(0, 9, 3)
    ]
    return cv2.vconcat(contact_rows)


def save_grid_diagnostic_outputs(
    prepared_sheet: PreparedMonitoringSheet,
    output_folder: Path,
    report: dict[str, Any] | None = None,
) -> GridDiagnosticOutputs:
    """Write all geometry-only diagnostic artifacts."""

    output_folder.mkdir(parents=True, exist_ok=True)
    alignment_report = report or calculate_alignment_report(prepared_sheet)
    line_overlay = draw_grid_line_diagnostic_overlay(
        prepared_sheet,
        alignment_report,
    )
    contact_sheet = create_diagnostic_contact_sheet(
        prepared_sheet,
        alignment_report,
    )
    outputs = GridDiagnosticOutputs(
        warped_table=output_folder / WARPED_TABLE_FILENAME,
        measurement_grid_overlay=output_folder / GRID_OVERLAY_FILENAME,
        line_diagnostic_overlay=output_folder / LINE_DIAGNOSTIC_OVERLAY_FILENAME,
        contact_sheet=output_folder / CONTACT_SHEET_FILENAME,
        alignment_report=output_folder / ALIGNMENT_REPORT_FILENAME,
    )
    save_image(prepared_sheet.warped_table, outputs.warped_table)
    save_image(
        prepared_sheet.measurement_grid_overlay,
        outputs.measurement_grid_overlay,
    )
    save_image(line_overlay, outputs.line_diagnostic_overlay)
    save_image(contact_sheet, outputs.contact_sheet)
    outputs.alignment_report.write_text(
        json.dumps(alignment_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return outputs


def run_grid_diagnostics(
    image_path: Path,
    output_folder: Path,
) -> tuple[GridDiagnosticOutputs, dict[str, Any]]:
    """Prepare one sheet and save diagnostics without importing PaddleOCR."""

    original_image = load_image(image_path)
    prepared_sheet = prepare_monitoring_sheet(original_image)
    report = calculate_alignment_report(prepared_sheet)
    report["source_image"] = str(image_path)
    outputs = save_grid_diagnostic_outputs(
        prepared_sheet,
        output_folder,
        report,
    )
    return outputs, report
