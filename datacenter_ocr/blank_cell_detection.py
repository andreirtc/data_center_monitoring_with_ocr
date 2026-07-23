from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


BLANK_ANALYSIS_CANVAS_WIDTH = 112
BLANK_ANALYSIS_CANVAS_HEIGHT = 40
BLANK_ANALYSIS_BORDER_EXCLUSION_RATIO = 0.06
BLANK_ANALYSIS_CANVAS_EDGE_RATIO = 0.05
LIKELY_BLANK_MAXIMUM_INK_RATIO = 0.012
LIKELY_BLANK_MAXIMUM_COMPONENT_RATIO = 0.012
LIKELY_BLANK_MAXIMUM_COMPONENT_WIDTH_RATIO = 0.08
LIKELY_BLANK_MAXIMUM_COMPONENT_ASPECT_RATIO = 0.5
VERTICAL_LINE_OCR_TOKENS = frozenset(("", "1", "I", "l", "|"))


@dataclass(frozen=True)
class BlankCellAnalysis:
    """
    Measurements produced while determining whether a cell is blank.
    """

    is_blank: bool
    ink_ratio: float
    significant_component_count: int
    largest_component_ratio: float
    largest_component_width_ratio: float
    largest_component_height_ratio: float
    largest_component_aspect_ratio: float
    analysis_width: int
    analysis_height: int
    used_normalized_canvas: bool
    cleaned_ink_mask: np.ndarray


def create_blank_analysis_canvas(
    cell_image: np.ndarray,
    canvas_width: int = BLANK_ANALYSIS_CANVAS_WIDTH,
    canvas_height: int = BLANK_ANALYSIS_CANVAS_HEIGHT,
    border_exclusion_ratio: float = BLANK_ANALYSIS_BORDER_EXCLUSION_RATIO,
) -> np.ndarray:
    """Place a cell on a stable, aspect-preserving blank-analysis canvas.

    Both fixed and perspective-warped crops are evaluated at the same size.
    A proportional perimeter is excluded before resizing, and the same canvas
    edge is cleared afterward. ``INTER_AREA`` is used consistently for every
    crop so geometry-specific source dimensions do not change the denominator
    or connected-component scale.
    """

    if cell_image is None or cell_image.size == 0:
        raise ValueError("Cannot normalize an empty cell image.")
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("Blank-analysis canvas dimensions must be positive.")
    if not 0.0 <= border_exclusion_ratio < 0.5:
        raise ValueError("border_exclusion_ratio must be between 0.0 and 0.5.")

    source_height, source_width = cell_image.shape[:2]
    excluded_x = round(source_width * border_exclusion_ratio)
    excluded_y = round(source_height * border_exclusion_ratio)
    x1 = excluded_x
    x2 = source_width - excluded_x
    y1 = excluded_y
    y2 = source_height - excluded_y
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Blank-analysis border exclusion removed the whole crop.")

    inner_image = cell_image[y1:y2, x1:x2]
    inner_height, inner_width = inner_image.shape[:2]
    scale = min(canvas_width / inner_width, canvas_height / inner_height)
    resized_width = max(1, round(inner_width * scale))
    resized_height = max(1, round(inner_height * scale))
    resized_image = cv2.resize(
        inner_image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas_shape = (
        (canvas_height, canvas_width)
        if len(cell_image.shape) == 2
        else (canvas_height, canvas_width, cell_image.shape[2])
    )
    canvas = np.full(canvas_shape, 255, dtype=cell_image.dtype)
    x_offset = (canvas_width - resized_width) // 2
    y_offset = (canvas_height - resized_height) // 2
    canvas[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized_image

    canvas_edge = max(
        1,
        round(min(canvas_width, canvas_height) * BLANK_ANALYSIS_CANVAS_EDGE_RATIO),
    )
    canvas[:canvas_edge] = 255
    canvas[-canvas_edge:] = 255
    canvas[:, :canvas_edge] = 255
    canvas[:, -canvas_edge:] = 255
    return canvas


def analyze_cell_for_blankness(
    cell_image: np.ndarray,
    ink_ratio_threshold: float = 0.003,
    minimum_component_area: int = 6,
    normalize_analysis_canvas: bool = True,
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

    analysis_image = (
        create_blank_analysis_canvas(cell_image)
        if normalize_analysis_canvas
        else cell_image.copy()
    )

    if len(analysis_image.shape) == 2:
        grayscale_image = analysis_image.copy()
    else:
        grayscale_image = cv2.cvtColor(
            analysis_image,
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
    largest_component_area = 0
    largest_component_width = 0
    largest_component_height = 0

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
        if component_area > largest_component_area:
            largest_component_area = int(component_area)
            largest_component_width = int(
                component_statistics[component_index, cv2.CC_STAT_WIDTH]
            )
            largest_component_height = int(
                component_statistics[component_index, cv2.CC_STAT_HEIGHT]
            )

    significant_ink_pixels = cv2.countNonZero(
        cleaned_ink_mask
    )

    total_pixels = height * width

    ink_ratio = (
        significant_ink_pixels
        / total_pixels
    )
    largest_component_ratio = largest_component_area / total_pixels
    largest_component_width_ratio = largest_component_width / width
    largest_component_height_ratio = largest_component_height / height
    largest_component_aspect_ratio = (
        largest_component_width / largest_component_height
        if largest_component_height
        else 0.0
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
        largest_component_ratio=largest_component_ratio,
        largest_component_width_ratio=largest_component_width_ratio,
        largest_component_height_ratio=largest_component_height_ratio,
        largest_component_aspect_ratio=largest_component_aspect_ratio,
        analysis_width=width,
        analysis_height=height,
        used_normalized_canvas=normalize_analysis_canvas,
        cleaned_ink_mask=cleaned_ink_mask,
    )


def is_likely_border_artifact_blank(
    analysis: BlankCellAnalysis,
    normalized_predictions: dict[str, str],
    raw_predictions: dict[str, str],
    candidate_interpretations: tuple[str, ...],
    *,
    has_serious_geometry_warning: bool,
) -> bool:
    """Return whether combined evidence supports a reviewable blank proposal.

    A line-like OCR result alone is deliberately insufficient. The crop must
    also contain only one small, narrow component, have very little residual
    ink, lack a plausible numeric interpretation, and have safe geometry.
    """

    available_texts = [
        str(text).strip()
        for predictions in (normalized_predictions, raw_predictions)
        for text in predictions.values()
    ]
    line_like_ocr = (
        bool(available_texts)
        and any(text for text in available_texts)
        and all(text in VERTICAL_LINE_OCR_TOKENS for text in available_texts)
    )
    return all(
        (
            not analysis.is_blank,
            analysis.ink_ratio <= LIKELY_BLANK_MAXIMUM_INK_RATIO,
            analysis.significant_component_count == 1,
            analysis.largest_component_ratio
            <= LIKELY_BLANK_MAXIMUM_COMPONENT_RATIO,
            analysis.largest_component_width_ratio
            <= LIKELY_BLANK_MAXIMUM_COMPONENT_WIDTH_RATIO,
            analysis.largest_component_aspect_ratio
            <= LIKELY_BLANK_MAXIMUM_COMPONENT_ASPECT_RATIO,
            line_like_ocr,
            not candidate_interpretations,
            not has_serious_geometry_warning,
        )
    )
