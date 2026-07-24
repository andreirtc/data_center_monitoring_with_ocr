from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


RotationMode = Literal["none", "clockwise", "counterclockwise"]
AUTO_ORIENTATION_MAXIMUM_WIDTH = 1200
AUTO_ORIENTATION_MINIMUM_SCORE = 0.10
AUTO_ORIENTATION_MINIMUM_MARGIN = 0.25


@dataclass(frozen=True)
class OrientationDecision:
    """Geometry-only orientation choice for one monitoring-sheet image."""

    rotation: RotationMode
    confident: bool
    clockwise_score: float
    counterclockwise_score: float
    score_margin: float


def rotate_monitoring_image(
    image: np.ndarray,
    rotation: RotationMode,
) -> np.ndarray:
    """Apply one lossless right-angle rotation to a monitoring sheet."""

    if rotation == "none":
        return image
    if rotation == "clockwise":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "counterclockwise":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unknown document rotation: {rotation}")


def _balanced_difference(first: float, second: float) -> float:
    """Return a bounded difference that is stable across scan darkness."""

    return (first - second) / max(first + second, 1e-6)


def _landscape_layout_score(image: np.ndarray) -> float:
    """
    Score the known landscape form without reading text or OCR ground truth.

    Correctly oriented sheets have their dense title/header above the mostly
    blank footer and their 16-column measurement grid to the left of the
    wider logged-by and remarks area.
    """

    height, width = image.shape[:2]
    if height >= width:
        return -1.0

    scale = min(AUTO_ORIENTATION_MAXIMUM_WIDTH / width, 1.0)
    if scale < 1.0:
        working = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        working = image

    height, width = working.shape[:2]
    grayscale = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    _, inverted = cv2.threshold(
        grayscale,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    ink = inverted.astype(np.float32) / 255.0

    top_end = max(1, round(height * 0.20))
    bottom_start = min(height - 1, round(height * 0.80))
    measurement_end = max(1, round(width * 0.72))

    top_ink = float(ink[:top_end].mean())
    bottom_ink = float(ink[bottom_start:].mean())
    measurement_ink = float(ink[:, :measurement_end].mean())
    right_side_ink = float(ink[:, measurement_end:].mean())

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(12, round(height * 0.025))),
    )
    vertical_lines = cv2.morphologyEx(
        inverted,
        cv2.MORPH_OPEN,
        vertical_kernel,
    ).astype(np.float32) / 255.0
    measurement_vertical = float(
        vertical_lines[:, :measurement_end].mean()
    )
    right_side_vertical = float(
        vertical_lines[:, measurement_end:].mean()
    )

    score = (
        0.65 * _balanced_difference(top_ink, bottom_ink)
        + 0.20
        * _balanced_difference(measurement_ink, right_side_ink)
        + 0.15
        * _balanced_difference(measurement_vertical, right_side_vertical)
    )
    return round(float(score), 6)


def detect_monitoring_sheet_orientation(
    image: np.ndarray,
) -> OrientationDecision:
    """
    Choose a landscape orientation for a portrait monitoring-sheet scan.

    Landscape inputs remain untouched. Ambiguous portrait inputs also remain
    untouched so the user can choose a manual orientation before preflight.
    """

    if image is None or image.size == 0:
        raise ValueError("No document image was provided for orientation.")

    height, width = image.shape[:2]
    if width >= height:
        return OrientationDecision(
            rotation="none",
            confident=True,
            clockwise_score=0.0,
            counterclockwise_score=0.0,
            score_margin=0.0,
        )

    clockwise = rotate_monitoring_image(image, "clockwise")
    counterclockwise = rotate_monitoring_image(image, "counterclockwise")
    clockwise_score = _landscape_layout_score(clockwise)
    counterclockwise_score = _landscape_layout_score(counterclockwise)

    if clockwise_score >= counterclockwise_score:
        proposed_rotation: RotationMode = "clockwise"
        winning_score = clockwise_score
        losing_score = counterclockwise_score
    else:
        proposed_rotation = "counterclockwise"
        winning_score = counterclockwise_score
        losing_score = clockwise_score

    margin = winning_score - losing_score
    confident = (
        winning_score >= AUTO_ORIENTATION_MINIMUM_SCORE
        and margin >= AUTO_ORIENTATION_MINIMUM_MARGIN
    )
    return OrientationDecision(
        rotation=proposed_rotation if confident else "none",
        confident=confident,
        clockwise_score=clockwise_score,
        counterclockwise_score=counterclockwise_score,
        score_margin=round(margin, 6),
    )
