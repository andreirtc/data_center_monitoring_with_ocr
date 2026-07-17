from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BlankCellAnalysis:
    """
    Measurements produced while determining whether a cell is blank.
    """

    is_blank: bool
    ink_ratio: float
    significant_component_count: int
    cleaned_ink_mask: np.ndarray


def analyze_cell_for_blankness(
    cell_image: np.ndarray,
    ink_ratio_threshold: float = 0.003,
    minimum_component_area: int = 6,
) -> BlankCellAnalysis:
    """
    Estimate whether a cell contains meaningful handwriting.

    Table borders are removed only for blankness analysis.
    The original cell remains untouched for OCR.
    """

    if cell_image is None or cell_image.size == 0:
        raise ValueError(
            "Cannot analyze an empty cell image."
        )

    if len(cell_image.shape) == 2:
        grayscale_image = cell_image.copy()
    else:
        grayscale_image = cv2.cvtColor(
            cell_image,
            cv2.COLOR_BGR2GRAY,
        )

    blurred_image = cv2.GaussianBlur(
        grayscale_image,
        (3, 3),
        0,
    )

    # White ink on a black background makes pixel counting easier.
    _, inverted_binary = cv2.threshold(
        blurred_image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    height, width = inverted_binary.shape[:2]

    horizontal_kernel_width = max(
        8,
        round(width * 0.55),
    )

    vertical_kernel_height = max(
        8,
        round(height * 0.55),
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_kernel_width, 1),
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_kernel_height),
    )

    horizontal_lines = cv2.morphologyEx(
        inverted_binary,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    vertical_lines = cv2.morphologyEx(
        inverted_binary,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    detected_lines = cv2.bitwise_or(
        horizontal_lines,
        vertical_lines,
    )

    # Slightly enlarge the line mask to include anti-aliased edges.
    detected_lines = cv2.dilate(
        detected_lines,
        np.ones(
            (3, 3),
            dtype=np.uint8,
        ),
        iterations=1,
    )

    non_line_ink = cv2.subtract(
        inverted_binary,
        detected_lines,
    )

    (
        component_count,
        component_labels,
        component_statistics,
        _,
    ) = cv2.connectedComponentsWithStats(
        non_line_ink,
        connectivity=8,
    )

    cleaned_ink_mask = np.zeros_like(
        non_line_ink
    )

    significant_component_count = 0

    # Component zero is the background, so iteration starts at one.
    for component_index in range(
        1,
        component_count,
    ):
        component_area = component_statistics[
            component_index,
            cv2.CC_STAT_AREA,
        ]

        if component_area < minimum_component_area:
            continue

        cleaned_ink_mask[
            component_labels == component_index
        ] = 255

        significant_component_count += 1

    significant_ink_pixels = cv2.countNonZero(
        cleaned_ink_mask
    )

    total_pixels = height * width

    ink_ratio = (
        significant_ink_pixels
        / total_pixels
    )

    is_blank = (
        ink_ratio < ink_ratio_threshold
    )

    return BlankCellAnalysis(
        is_blank=is_blank,
        ink_ratio=ink_ratio,
        significant_component_count=(
            significant_component_count
        ),
        cleaned_ink_mask=cleaned_ink_mask,
    )