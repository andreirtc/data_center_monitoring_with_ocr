from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import median
from typing import Any, Literal

import cv2
import numpy as np


GeometryMode = Literal["fixed", "calibrated"]

MAXIMUM_LOCAL_OFFSET_RATIO = 0.42
MINIMUM_SIZE_RATIO = 0.50
MAXIMUM_SIZE_RATIO = 1.50
MAXIMUM_CURVATURE_CHANGE_RATIO = 0.30
ROW_SEQUENCE_MINIMUM_STRONG_BOUNDARIES = 30
ROW_SEQUENCE_MINIMUM_MEDIAN_STRENGTH = 0.70


@dataclass(frozen=True)
class BoundaryCurve:
    """One locally sampled horizontal or vertical grid boundary."""

    axis: Literal["horizontal", "vertical"]
    index: int
    expected_position: float
    sample_coordinates: tuple[float, ...]
    sampled_positions: tuple[float, ...]
    sample_confidences: tuple[float, ...]
    sample_sources: tuple[str, ...]
    direct_detection_count: int
    expected_sample_count: int
    confidence: float
    uses_fixed_fallback: bool = False
    uses_interpolation: bool = False
    rejection_reason: str = ""

    def position_at(self, coordinate: float) -> float:
        """Interpolate this boundary at a local perpendicular coordinate."""

        return float(
            np.interp(
                coordinate,
                np.asarray(self.sample_coordinates, dtype=np.float64),
                np.asarray(self.sampled_positions, dtype=np.float64),
            )
        )


@dataclass(frozen=True)
class RowSequenceAlignment:
    """Geometry-only evidence used to anchor the 31 daily rows."""

    y_boundaries: tuple[int, ...]
    used_detected_alignment: bool
    detected_top: float
    detected_bottom: float
    detected_spacing: float
    strong_boundary_count: int
    median_line_strength: float
    fallback_reason: str = ""


@dataclass(frozen=True)
class LocalGridCalibration:
    """Safe local boundary curves and their 32-by-17 intersection mesh."""

    fixed_x_boundaries: tuple[int, ...]
    fixed_y_boundaries: tuple[int, ...]
    horizontal_curves: tuple[BoundaryCurve, ...]
    vertical_curves: tuple[BoundaryCurve, ...]
    intersections: np.ndarray
    expected_row_height: float
    expected_column_width: float
    minimum_row_height: float
    maximum_row_height: float
    minimum_column_width: float
    maximum_column_width: float
    rejected_horizontal_boundaries: tuple[int, ...]
    rejected_vertical_boundaries: tuple[int, ...]

    @property
    def fallback_horizontal_boundary_count(self) -> int:
        return sum(curve.uses_fixed_fallback for curve in self.horizontal_curves)

    @property
    def fallback_vertical_boundary_count(self) -> int:
        return sum(curve.uses_fixed_fallback for curve in self.vertical_curves)

    @property
    def low_confidence_horizontal_boundary_count(self) -> int:
        return sum(
            curve.confidence < 0.35 or curve.direct_detection_count < 6
            for curve in self.horizontal_curves
        )

    @property
    def low_confidence_vertical_boundary_count(self) -> int:
        return sum(
            curve.confidence < 0.35 or curve.direct_detection_count < 4
            for curve in self.vertical_curves
        )


def expected_grid_boundaries(
    measurement_boxes: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    """Return the fixed x and y boundaries represented by 496 boxes."""

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


def _line_masks(
    image: np.ndarray,
    row_height: float,
    column_width: float,
) -> tuple[np.ndarray, np.ndarray]:
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
    return (
        cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel),
        cv2.morphologyEx(inverted, cv2.MORPH_OPEN, vertical_kernel),
    )


def _row_sequence_likelihood(
    horizontal_mask: np.ndarray,
    x_positions: list[int],
) -> np.ndarray:
    """Combine line evidence across all eight point pairs."""

    band_likelihoods: list[np.ndarray] = []
    for point_index in range(8):
        x1 = x_positions[point_index * 2]
        x2 = x_positions[point_index * 2 + 2]
        projection = (
            horizontal_mask[:, x1:x2].sum(axis=1).astype(np.float32)
            / 255.0
            / max(1, x2 - x1)
        )
        locally_maximized = cv2.dilate(
            projection.reshape(-1, 1),
            np.ones((9, 1), dtype=np.uint8),
        ).reshape(-1)
        band_likelihoods.append(locally_maximized)
    return np.median(np.asarray(band_likelihoods), axis=0)


def align_measurement_boxes_to_printed_rows(
    image: np.ndarray,
    measurement_boxes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], RowSequenceAlignment]:
    """Anchor straight rows to a complete 32-boundary printed sequence.

    Some scanner inputs cause table detection to select the border below the
    title instead of the complete form border. Reference-scaled rows then start
    one or more days late. This geometry-only search fits all 32 boundaries at
    once, favors a final boundary at the bottom of the warped table, and falls
    back to the reference rows unless evidence is strong across the sheet.
    """

    image_height = image.shape[0]
    x_positions, expected_y_positions = expected_grid_boundaries(
        measurement_boxes
    )
    if len(x_positions) != 17 or len(expected_y_positions) != 32:
        raise ValueError("Expected a 17-by-32 monitoring measurement grid.")

    row_height = float(median(np.diff(expected_y_positions)))
    column_width = float(median(np.diff(x_positions)))
    horizontal_mask, _ = _line_masks(image, row_height, column_width)
    likelihood = _row_sequence_likelihood(horizontal_mask, x_positions)

    spacing_candidates = np.arange(
        row_height * 0.90,
        row_height * 1.23,
        0.1,
        dtype=np.float32,
    )
    top_candidates = np.arange(
        max(0, round(image_height * 0.04)),
        min(image_height, round(image_height * 0.20) + 1),
        dtype=np.float32,
    )
    boundary_indices = np.arange(32, dtype=np.float32)
    candidate_positions = (
        top_candidates[None, :, None]
        + spacing_candidates[:, None, None] * boundary_indices[None, None, :]
    )
    valid_positions = candidate_positions < image_height - 0.5
    sampled_positions = np.clip(
        np.rint(candidate_positions).astype(np.int32),
        0,
        image_height - 1,
    )
    sampled_strengths = likelihood[sampled_positions]
    sampled_strengths = np.where(valid_positions, sampled_strengths, 0.0)
    strong_counts = np.count_nonzero(sampled_strengths > 0.20, axis=2)
    evidence_scores = np.minimum(sampled_strengths, 0.75).sum(axis=2)
    evidence_scores += strong_counts * 0.5
    candidate_bottoms = (
        top_candidates[None, :]
        + spacing_candidates[:, None] * 31.0
    )
    bottom_penalty = np.abs(candidate_bottoms - (image_height - 1)) / max(
        1.0,
        row_height * 0.15,
    )
    final_scores = evidence_scores - bottom_penalty
    best_spacing_index, best_top_index = np.unravel_index(
        int(np.argmax(final_scores)),
        final_scores.shape,
    )

    detected_top = float(top_candidates[best_top_index])
    detected_spacing = float(spacing_candidates[best_spacing_index])
    detected_bottom = detected_top + detected_spacing * 31.0
    best_strengths = sampled_strengths[best_spacing_index, best_top_index]
    strong_boundary_count = int(strong_counts[best_spacing_index, best_top_index])
    available_strengths = best_strengths[
        valid_positions[best_spacing_index, best_top_index]
    ]
    median_line_strength = (
        float(np.median(available_strengths))
        if available_strengths.size
        else 0.0
    )

    fallback_reasons: list[str] = []
    if strong_boundary_count < ROW_SEQUENCE_MINIMUM_STRONG_BOUNDARIES:
        fallback_reasons.append(
            f"only {strong_boundary_count}/32 boundaries had strong evidence"
        )
    if median_line_strength < ROW_SEQUENCE_MINIMUM_MEDIAN_STRENGTH:
        fallback_reasons.append(
            f"median line strength was {median_line_strength:.3f}"
        )
    if abs(detected_bottom - (image_height - 1)) > row_height:
        fallback_reasons.append(
            "the fitted bottom boundary was too far from the table edge"
        )

    if fallback_reasons:
        row_alignment = RowSequenceAlignment(
            y_boundaries=tuple(expected_y_positions),
            used_detected_alignment=False,
            detected_top=round(detected_top, 3),
            detected_bottom=round(detected_bottom, 3),
            detected_spacing=round(detected_spacing, 3),
            strong_boundary_count=strong_boundary_count,
            median_line_strength=round(median_line_strength, 6),
            fallback_reason="; ".join(fallback_reasons) + ".",
        )
        return [dict(box) for box in measurement_boxes], row_alignment

    fitted_bottom = min(float(image_height - 1), detected_bottom)
    fitted_boundaries = tuple(
        int(value)
        for value in np.rint(np.linspace(detected_top, fitted_bottom, 32))
    )
    aligned_boxes = []
    for box in measurement_boxes:
        row_index = int(box["day"]) - 1
        aligned_boxes.append(
            {
                **box,
                "y1": fitted_boundaries[row_index],
                "y2": fitted_boundaries[row_index + 1],
            }
        )
    row_alignment = RowSequenceAlignment(
        y_boundaries=fitted_boundaries,
        used_detected_alignment=True,
        detected_top=round(detected_top, 3),
        detected_bottom=round(detected_bottom, 3),
        detected_spacing=round(detected_spacing, 3),
        strong_boundary_count=strong_boundary_count,
        median_line_strength=round(median_line_strength, 6),
    )
    return aligned_boxes, row_alignment


def _local_candidate(
    projection: np.ndarray,
    expected_position: float,
    search_radius: int,
    normalization_length: int,
    minimum_confidence: float,
) -> tuple[float | None, float]:
    lower = max(0, round(expected_position) - search_radius)
    upper = min(len(projection), round(expected_position) + search_radius + 1)
    if lower >= upper:
        return None, 0.0
    local_projection = projection[lower:upper]
    local_index = int(np.argmax(local_projection))
    detected_position = float(lower + local_index)
    confidence = float(local_projection[local_index]) / max(normalization_length, 1)
    if confidence < minimum_confidence:
        return None, round(confidence, 6)
    return detected_position, round(min(confidence, 1.0), 6)


def _reject_isolated_candidates(
    positions: list[float | None],
    spacing: float,
) -> list[float | None]:
    """Reject isolated local peaks that break an otherwise smooth curve."""

    filtered = positions.copy()
    for index in range(1, len(positions) - 1):
        current = positions[index]
        previous = positions[index - 1]
        following = positions[index + 1]
        if current is None or previous is None or following is None:
            continue
        neighbor_prediction = (previous + following) / 2.0
        if abs(current - neighbor_prediction) > spacing * 0.25:
            filtered[index] = None
    return filtered


def _curve_from_candidates(
    *,
    axis: Literal["horizontal", "vertical"],
    index: int,
    expected_position: float,
    coordinates: list[float],
    positions: list[float | None],
    confidences: list[float],
    spacing: float,
    minimum_direct_count: int,
) -> BoundaryCurve | None:
    bounded_positions = [
        (
            position
            if position is not None
            and abs(position - expected_position)
            <= spacing * MAXIMUM_LOCAL_OFFSET_RATIO
            else None
        )
        for position in positions
    ]
    bounded_positions = _reject_isolated_candidates(bounded_positions, spacing)
    reliable_indices = [
        candidate_index
        for candidate_index, position in enumerate(bounded_positions)
        if position is not None
    ]
    if len(reliable_indices) < minimum_direct_count:
        return None

    reliable_coordinates = np.asarray(
        [coordinates[candidate_index] for candidate_index in reliable_indices],
        dtype=np.float64,
    )
    reliable_positions = np.asarray(
        [bounded_positions[candidate_index] for candidate_index in reliable_indices],
        dtype=np.float64,
    )
    interpolated_positions = np.interp(
        np.asarray(coordinates, dtype=np.float64),
        reliable_coordinates,
        reliable_positions,
    )
    sampled_confidences: list[float] = []
    sources: list[str] = []
    for candidate_index, position in enumerate(bounded_positions):
        if position is not None:
            sampled_confidences.append(confidences[candidate_index])
            sources.append("detected")
            continue
        nearest_confidence = max(
            0.0,
            max(
                (
                    confidences[reliable_index]
                    * max(0.0, 1.0 - abs(reliable_index - candidate_index) * 0.20)
                    for reliable_index in reliable_indices
                ),
                default=0.0,
            ),
        )
        sampled_confidences.append(nearest_confidence * 0.75)
        sources.append("interpolated_along_boundary")

    offsets = interpolated_positions - expected_position
    if len(offsets) >= 3 and np.max(np.abs(np.diff(offsets, n=2))) > (
        spacing * MAXIMUM_CURVATURE_CHANGE_RATIO
    ):
        return None

    direct_confidences = [confidences[item] for item in reliable_indices]
    confidence = float(median(direct_confidences)) * (
        len(reliable_indices) / len(coordinates)
    )
    return BoundaryCurve(
        axis=axis,
        index=index,
        expected_position=expected_position,
        sample_coordinates=tuple(coordinates),
        sampled_positions=tuple(float(value) for value in interpolated_positions),
        sample_confidences=tuple(round(value, 6) for value in sampled_confidences),
        sample_sources=tuple(sources),
        direct_detection_count=len(reliable_indices),
        expected_sample_count=len(coordinates),
        confidence=round(confidence, 6),
        uses_interpolation=len(reliable_indices) < len(coordinates),
    )


def _fixed_curve(
    *,
    axis: Literal["horizontal", "vertical"],
    index: int,
    expected_position: float,
    coordinates: list[float],
    reason: str,
) -> BoundaryCurve:
    return BoundaryCurve(
        axis=axis,
        index=index,
        expected_position=expected_position,
        sample_coordinates=tuple(coordinates),
        sampled_positions=tuple(expected_position for _ in coordinates),
        sample_confidences=tuple(0.0 for _ in coordinates),
        sample_sources=tuple("fixed_fallback" for _ in coordinates),
        direct_detection_count=0,
        expected_sample_count=len(coordinates),
        confidence=0.0,
        uses_fixed_fallback=True,
        rejection_reason=reason,
    )


def _interpolate_missing_curves(
    *,
    curves: list[BoundaryCurve | None],
    axis: Literal["horizontal", "vertical"],
    expected_positions: list[int],
    coordinates: list[float],
    spacing: float,
) -> list[BoundaryCurve]:
    completed: list[BoundaryCurve] = []
    reliable_indices = [index for index, curve in enumerate(curves) if curve is not None]
    for index, curve in enumerate(curves):
        if curve is not None:
            completed.append(curve)
            continue
        previous = max(
            (candidate for candidate in reliable_indices if candidate < index),
            default=None,
        )
        following = min(
            (candidate for candidate in reliable_indices if candidate > index),
            default=None,
        )
        neighbors = [
            neighbor
            for neighbor in (previous, following)
            if neighbor is not None and abs(neighbor - index) <= 2
        ]
        if not neighbors:
            completed.append(
                _fixed_curve(
                    axis=axis,
                    index=index,
                    expected_position=expected_positions[index],
                    coordinates=coordinates,
                    reason="Insufficient nearby reliable line evidence.",
                )
            )
            continue

        neighbor_offsets = []
        for neighbor in neighbors:
            neighbor_curve = curves[neighbor]
            assert neighbor_curve is not None
            neighbor_offsets.append(
                np.asarray(neighbor_curve.sampled_positions)
                - neighbor_curve.expected_position
            )
        inferred_offsets = np.mean(np.vstack(neighbor_offsets), axis=0)
        if np.max(np.abs(inferred_offsets)) > spacing * MAXIMUM_LOCAL_OFFSET_RATIO:
            completed.append(
                _fixed_curve(
                    axis=axis,
                    index=index,
                    expected_position=expected_positions[index],
                    coordinates=coordinates,
                    reason="Neighbor interpolation exceeded the local search bound.",
                )
            )
            continue
        neighbor_confidence = min(
            curves[neighbor].confidence  # type: ignore[union-attr]
            for neighbor in neighbors
        )
        completed.append(
            BoundaryCurve(
                axis=axis,
                index=index,
                expected_position=expected_positions[index],
                sample_coordinates=tuple(coordinates),
                sampled_positions=tuple(
                    float(expected_positions[index] + offset)
                    for offset in inferred_offsets
                ),
                sample_confidences=tuple(
                    round(neighbor_confidence * 0.50, 6) for _ in coordinates
                ),
                sample_sources=tuple("interpolated_from_neighbors" for _ in coordinates),
                direct_detection_count=0,
                expected_sample_count=len(coordinates),
                confidence=round(neighbor_confidence * 0.50, 6),
                uses_interpolation=True,
                rejection_reason="Direct line evidence was weak or missing.",
            )
        )
    return completed


def _fallback_boundary(curve: BoundaryCurve, reason: str) -> BoundaryCurve:
    return replace(
        curve,
        sampled_positions=tuple(
            curve.expected_position for _ in curve.sample_coordinates
        ),
        sample_confidences=tuple(0.0 for _ in curve.sample_coordinates),
        sample_sources=tuple("fixed_fallback" for _ in curve.sample_coordinates),
        confidence=0.0,
        uses_fixed_fallback=True,
        uses_interpolation=False,
        rejection_reason=reason,
    )


def enforce_monotonic_boundaries(
    curves: list[BoundaryCurve],
    evaluation_coordinates: list[float],
    expected_spacing: float,
) -> tuple[list[BoundaryCurve], tuple[int, ...]]:
    """Revert unsafe curves until all local boundary sizes are sensible."""

    minimum_spacing = expected_spacing * MINIMUM_SIZE_RATIO
    maximum_spacing = expected_spacing * MAXIMUM_SIZE_RATIO
    corrected = curves.copy()
    rejected: set[int] = set()
    for _ in range(len(curves) * 2):
        unsafe_pair: tuple[int, int] | None = None
        for first_index in range(len(corrected) - 1):
            gaps = [
                corrected[first_index + 1].position_at(coordinate)
                - corrected[first_index].position_at(coordinate)
                for coordinate in evaluation_coordinates
            ]
            if min(gaps) < minimum_spacing or max(gaps) > maximum_spacing:
                unsafe_pair = (first_index, first_index + 1)
                break
        if unsafe_pair is None:
            return corrected, tuple(sorted(rejected))
        candidates = [
            index for index in unsafe_pair if not corrected[index].uses_fixed_fallback
        ]
        if not candidates:
            break
        rejected_index = min(
            candidates,
            key=lambda index: (
                corrected[index].confidence,
                corrected[index].direct_detection_count,
            ),
        )
        corrected[rejected_index] = _fallback_boundary(
            corrected[rejected_index],
            "Boundary reverted because local spacing was implausible.",
        )
        rejected.add(rejected_index)
    return corrected, tuple(sorted(rejected))


def _build_intersections(
    horizontal_curves: list[BoundaryCurve],
    vertical_curves: list[BoundaryCurve],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    intersections = np.zeros(
        (len(horizontal_curves), len(vertical_curves), 2),
        dtype=np.float32,
    )
    for row_index, horizontal_curve in enumerate(horizontal_curves):
        for column_index, vertical_curve in enumerate(vertical_curves):
            x_position = vertical_curve.expected_position
            y_position = horizontal_curve.expected_position
            for _ in range(4):
                y_position = horizontal_curve.position_at(x_position)
                x_position = vertical_curve.position_at(y_position)
            intersections[row_index, column_index] = (
                np.clip(x_position, 0, image_width - 1),
                np.clip(y_position, 0, image_height - 1),
            )
    return intersections


def calibrate_local_grid(
    image: np.ndarray,
    measurement_boxes: list[dict[str, Any]],
) -> LocalGridCalibration:
    """Build a deterministic, locally curved grid from bounded line evidence."""

    image_height, image_width = image.shape[:2]
    x_positions, y_positions = expected_grid_boundaries(measurement_boxes)
    row_height = float(median(np.diff(y_positions)))
    column_width = float(median(np.diff(x_positions)))
    horizontal_mask, vertical_mask = _line_masks(image, row_height, column_width)

    horizontal_coordinates = [
        float(round((x_positions[index * 2] + x_positions[index * 2 + 2]) / 2))
        for index in range(8)
    ]
    horizontal_candidates: list[BoundaryCurve | None] = []
    for boundary_index, expected_y in enumerate(y_positions):
        positions: list[float | None] = []
        confidences: list[float] = []
        for point_index in range(8):
            x1 = x_positions[point_index * 2]
            x2 = x_positions[point_index * 2 + 2]
            projection = horizontal_mask[:, x1:x2].sum(axis=1).astype(float) / 255.0
            position, confidence = _local_candidate(
                projection,
                expected_y,
                max(3, round(row_height * 0.45)),
                x2 - x1,
                0.15,
            )
            positions.append(position)
            confidences.append(confidence)
        horizontal_candidates.append(
            _curve_from_candidates(
                axis="horizontal",
                index=boundary_index,
                expected_position=expected_y,
                coordinates=horizontal_coordinates,
                positions=positions,
                confidences=confidences,
                spacing=row_height,
                minimum_direct_count=4,
            )
        )

    data_top, data_bottom = y_positions[0], y_positions[-1]
    band_edges = np.linspace(data_top, data_bottom, 7).round().astype(int)
    vertical_coordinates = [
        float(round((band_edges[index] + band_edges[index + 1]) / 2))
        for index in range(6)
    ]
    vertical_candidates: list[BoundaryCurve | None] = []
    for boundary_index, expected_x in enumerate(x_positions):
        positions = []
        confidences = []
        for band_index in range(6):
            y1 = int(band_edges[band_index])
            y2 = int(band_edges[band_index + 1])
            projection = vertical_mask[y1:y2, :].sum(axis=0).astype(float) / 255.0
            position, confidence = _local_candidate(
                projection,
                expected_x,
                max(4, round(column_width * 0.35)),
                y2 - y1,
                0.12,
            )
            positions.append(position)
            confidences.append(confidence)
        vertical_candidates.append(
            _curve_from_candidates(
                axis="vertical",
                index=boundary_index,
                expected_position=expected_x,
                coordinates=vertical_coordinates,
                positions=positions,
                confidences=confidences,
                spacing=column_width,
                minimum_direct_count=3,
            )
        )

    horizontal_curves = _interpolate_missing_curves(
        curves=horizontal_candidates,
        axis="horizontal",
        expected_positions=y_positions,
        coordinates=horizontal_coordinates,
        spacing=row_height,
    )
    vertical_curves = _interpolate_missing_curves(
        curves=vertical_candidates,
        axis="vertical",
        expected_positions=x_positions,
        coordinates=vertical_coordinates,
        spacing=column_width,
    )
    horizontal_curves, rejected_horizontal = enforce_monotonic_boundaries(
        horizontal_curves,
        [float(value) for value in x_positions],
        row_height,
    )
    vertical_curves, rejected_vertical = enforce_monotonic_boundaries(
        vertical_curves,
        [float(value) for value in y_positions],
        column_width,
    )
    intersections = _build_intersections(
        horizontal_curves,
        vertical_curves,
        image_width,
        image_height,
    )
    row_heights = np.diff(intersections[:, :, 1], axis=0)
    column_widths = np.diff(intersections[:, :, 0], axis=1)
    return LocalGridCalibration(
        fixed_x_boundaries=tuple(x_positions),
        fixed_y_boundaries=tuple(y_positions),
        horizontal_curves=tuple(horizontal_curves),
        vertical_curves=tuple(vertical_curves),
        intersections=intersections,
        expected_row_height=row_height,
        expected_column_width=column_width,
        minimum_row_height=round(float(row_heights.min()), 6),
        maximum_row_height=round(float(row_heights.max()), 6),
        minimum_column_width=round(float(column_widths.min()), 6),
        maximum_column_width=round(float(column_widths.max()), 6),
        rejected_horizontal_boundaries=rejected_horizontal,
        rejected_vertical_boundaries=rejected_vertical,
    )


def _fixed_quadrilateral(box: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            [box["x1"], box["y1"]],
            [box["x2"], box["y1"]],
            [box["x2"], box["y2"]],
            [box["x1"], box["y2"]],
        ],
        dtype=np.float32,
    )


def _cell_indices(box: dict[str, Any]) -> tuple[int, int]:
    row_index = int(box["day"]) - 1
    column_index = (int(box["point"]) - 1) * 2
    if box["reading_type"] == "humidity":
        column_index += 1
    return row_index, column_index


def cell_source_quadrilateral(
    calibration: LocalGridCalibration,
    box: dict[str, Any],
) -> np.ndarray:
    """Return top-left, top-right, bottom-right, bottom-left corners."""

    row_index, column_index = _cell_indices(box)
    return np.asarray(
        [
            calibration.intersections[row_index, column_index],
            calibration.intersections[row_index, column_index + 1],
            calibration.intersections[row_index + 1, column_index + 1],
            calibration.intersections[row_index + 1, column_index],
        ],
        dtype=np.float32,
    )


def validate_cell_quadrilateral(
    quadrilateral: np.ndarray,
    calibration: LocalGridCalibration,
    image_width: int,
    image_height: int,
) -> str | None:
    """Return a safety rejection reason, or None for a valid local cell."""

    if quadrilateral.shape != (4, 2) or not np.isfinite(quadrilateral).all():
        return "Cell quadrilateral is not finite and four-cornered."
    if (
        np.any(quadrilateral[:, 0] < 0)
        or np.any(quadrilateral[:, 0] >= image_width)
        or np.any(quadrilateral[:, 1] < 0)
        or np.any(quadrilateral[:, 1] >= image_height)
    ):
        return "Cell quadrilateral would sample outside the image."
    top_width = float(np.linalg.norm(quadrilateral[1] - quadrilateral[0]))
    bottom_width = float(np.linalg.norm(quadrilateral[2] - quadrilateral[3]))
    left_height = float(np.linalg.norm(quadrilateral[3] - quadrilateral[0]))
    right_height = float(np.linalg.norm(quadrilateral[2] - quadrilateral[1]))
    if min(top_width, bottom_width) < calibration.expected_column_width * MINIMUM_SIZE_RATIO:
        return "Cell column width is implausibly small."
    if max(top_width, bottom_width) > calibration.expected_column_width * MAXIMUM_SIZE_RATIO:
        return "Cell column width is implausibly large."
    if min(left_height, right_height) < calibration.expected_row_height * MINIMUM_SIZE_RATIO:
        return "Cell row height is implausibly small."
    if max(left_height, right_height) > calibration.expected_row_height * MAXIMUM_SIZE_RATIO:
        return "Cell row height is implausibly large."
    if abs(float(cv2.contourArea(quadrilateral))) < (
        calibration.expected_column_width
        * calibration.expected_row_height
        * MINIMUM_SIZE_RATIO
    ):
        return "Cell quadrilateral is degenerate."
    return None


def _warp_local_cell(
    image: np.ndarray,
    quadrilateral: np.ndarray,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    destination = np.asarray(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    transformation = cv2.getPerspectiveTransform(quadrilateral, destination)
    warped = cv2.warpPerspective(
        image,
        transformation,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    border_inset = max(1, min(2, output_height // 10))
    if output_width > border_inset * 2 and output_height > border_inset * 2:
        warped = warped[
            border_inset : output_height - border_inset,
            border_inset : output_width - border_inset,
        ].copy()
    return warped


def _wide_context_crop(image: np.ndarray, quadrilateral: np.ndarray) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    x_min, y_min = np.floor(quadrilateral.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(quadrilateral.max(axis=0)).astype(int)
    width = max(1, x_max - x_min)
    height = max(1, y_max - y_min)
    x1 = max(0, x_min - round(width * 0.75))
    x2 = min(image_width, x_max + round(width * 0.75))
    y1 = max(0, y_min - round(height * 0.90))
    y2 = min(image_height, y_max + round(height * 0.90))
    context = image[y1:y2, x1:x2].copy()
    local_quad = quadrilateral - np.asarray([x1, y1], dtype=np.float32)
    cv2.polylines(
        context,
        [np.round(local_quad).astype(np.int32)],
        True,
        (255, 0, 255),
        2,
    )
    return context


def extract_calibrated_measurement_cells(
    image: np.ndarray,
    measurement_boxes: list[dict[str, Any]],
    fixed_cells: list[dict[str, Any]],
    calibration: LocalGridCalibration,
) -> tuple[list[dict[str, Any]], int]:
    """Extract all 496 stable identities from safe local quadrilaterals."""

    fixed_lookup = {cell["filename"]: cell for cell in fixed_cells}
    image_height, image_width = image.shape[:2]
    calibrated_cells: list[dict[str, Any]] = []
    rejected_count = 0
    for box in measurement_boxes:
        row_index, column_index = _cell_indices(box)
        filename = (
            f"day_{box['day']:02d}_point_{box['point']:02d}_"
            f"{box['reading_type']}.png"
        )
        fixed_cell = fixed_lookup[filename]
        quadrilateral = cell_source_quadrilateral(calibration, box)
        rejection_reason = validate_cell_quadrilateral(
            quadrilateral,
            calibration,
            image_width,
            image_height,
        )
        boundary_curves = (
            calibration.horizontal_curves[row_index],
            calibration.horizontal_curves[row_index + 1],
            calibration.vertical_curves[column_index],
            calibration.vertical_curves[column_index + 1],
        )
        fallback_boundaries = tuple(
            f"{curve.axis[0]}{curve.index}"
            for curve in boundary_curves
            if curve.uses_fixed_fallback
        )
        uses_interpolation = any(curve.uses_interpolation for curve in boundary_curves)
        confidence = min(curve.confidence for curve in boundary_curves)
        if rejection_reason is not None:
            rejected_count += 1
            quadrilateral = _fixed_quadrilateral(box)
            calibrated_image = fixed_cell["image"].copy()
            fallback_boundaries = tuple(
                sorted(set((*fallback_boundaries, "invalid_cell_fixed_quad")))
            )
            confidence = 0.0
        else:
            calibrated_image = _warp_local_cell(
                image,
                quadrilateral,
                max(8, int(box["x2"] - box["x1"])),
                max(8, int(box["y2"] - box["y1"])),
            )
        calibrated_cells.append(
            {
                **box,
                "filename": filename,
                "image": calibrated_image,
                "fixed_image": fixed_cell["image"],
                "calibrated_image": calibrated_image,
                "wide_context_image": _wide_context_crop(image, quadrilateral),
                "source_quadrilateral": tuple(
                    tuple(round(float(coordinate), 3) for coordinate in point)
                    for point in quadrilateral
                ),
                "fallback_boundaries": fallback_boundaries,
                "uses_fallback_geometry": bool(fallback_boundaries),
                "uses_interpolated_geometry": uses_interpolation,
                "local_geometry_confidence": round(float(confidence), 6),
                "geometry_rejection_reason": rejection_reason or "",
                "geometry_mode": "calibrated",
            }
        )
    return calibrated_cells, rejected_count


def draw_calibrated_grid_overlay(
    image: np.ndarray,
    calibration: LocalGridCalibration,
) -> np.ndarray:
    """Draw the accepted curved grid, highlighting interpolation and fallback."""

    overlay = image.copy()
    for curve in (*calibration.horizontal_curves, *calibration.vertical_curves):
        if curve.uses_fixed_fallback:
            color = (0, 0, 255)
        elif curve.uses_interpolation or curve.confidence < 0.35:
            color = (0, 215, 255)
        else:
            color = (0, 210, 0)
        if curve.axis == "horizontal":
            points = np.asarray(
                [
                    (round(coordinate), round(curve.position_at(coordinate)))
                    for coordinate in calibration.fixed_x_boundaries
                ],
                dtype=np.int32,
            )
        else:
            points = np.asarray(
                [
                    (round(curve.position_at(coordinate)), round(coordinate))
                    for coordinate in calibration.fixed_y_boundaries
                ],
                dtype=np.int32,
            )
        cv2.polylines(overlay, [points], False, color, 2)
    legend_text = (
        "Local: green direct | yellow interpolated/weak | red fixed fallback"
    )
    cv2.rectangle(
        overlay,
        (35, 5),
        (1170, 38),
        (255, 255, 255),
        -1,
    )
    cv2.putText(
        overlay,
        legend_text,
        (45, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    return overlay
