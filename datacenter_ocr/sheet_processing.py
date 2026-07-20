from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datacenter_ocr.cell_extraction import extract_measurement_cells
from datacenter_ocr.config import (
    CANNY_HIGH_THRESHOLD,
    CANNY_LOW_THRESHOLD,
    CELL_BOTTOM_PADDING_RATIO,
    CELL_HORIZONTAL_MARGIN_RATIO,
    CELL_TOP_PADDING_RATIO,
    CELL_VERTICAL_MARGIN_RATIO,
    MAXIMUM_IMAGE_WIDTH,
    MINIMUM_TABLE_AREA_RATIO,
    POLYGON_APPROXIMATION_RATIO,
    STANDARD_TABLE_HEIGHT,
    STANDARD_TABLE_WIDTH,
    TABLE_ROI_BOTTOM_RATIO,
    TABLE_ROI_LEFT_RATIO,
    TABLE_ROI_RIGHT_RATIO,
    TABLE_ROI_TOP_RATIO,
)
from datacenter_ocr.image_processing import (
    connect_edges,
    convert_to_grayscale,
    detect_edges,
    draw_document_detection,
    draw_measurement_boxes,
    find_table_contour,
    resize_to_maximum_width,
    warp_perspective,
)
from datacenter_ocr.table_layout import build_measurement_boxes


@dataclass
class PreparedMonitoringSheet:
    """
    Images and structured cells produced from one uploaded form.
    """

    detection_preview: np.ndarray
    warped_table: np.ndarray
    measurement_grid_overlay: np.ndarray
    measurement_boxes: list[dict]
    cells: list[dict]


def prepare_monitoring_sheet(
    original_image: np.ndarray,
) -> PreparedMonitoringSheet:
    """
    Detect, straighten, standardize, and divide one monitoring sheet.

    The function accepts an OpenCV image array rather than a file path.
    This allows it to process images loaded from disk or uploaded
    through Streamlit.
    """

    if original_image is None:
        raise ValueError(
            "No monitoring-sheet image was provided."
        )

    if original_image.size == 0:
        raise ValueError(
            "The monitoring-sheet image is empty."
        )

    # Use a smaller copy for faster table detection.
    resized_image = resize_to_maximum_width(
        image=original_image,
        maximum_width=MAXIMUM_IMAGE_WIDTH,
    )

    grayscale_image = convert_to_grayscale(
        resized_image
    )

    raw_edges = detect_edges(
        grayscale_image=grayscale_image,
        low_threshold=CANNY_LOW_THRESHOLD,
        high_threshold=CANNY_HIGH_THRESHOLD,
    )

    connected_edges = connect_edges(
        raw_edges
    )

    table_contour = find_table_contour(
        connected_edges=connected_edges,
        left_ratio=TABLE_ROI_LEFT_RATIO,
        right_ratio=TABLE_ROI_RIGHT_RATIO,
        top_ratio=TABLE_ROI_TOP_RATIO,
        bottom_ratio=TABLE_ROI_BOTTOM_RATIO,
        minimum_area_ratio=MINIMUM_TABLE_AREA_RATIO,
        approximation_ratio=POLYGON_APPROXIMATION_RATIO,
    )

    if table_contour is None:
        raise ValueError(
            "The monitoring table could not be detected. "
            "Use a clearer photo showing the complete form."
        )

    detection_preview = draw_document_detection(
        resized_image,
        table_contour,
    )

    # Detection occurred on the resized copy. Convert the contour
    # coordinates back to the original high-resolution image.
    original_width = original_image.shape[1]
    resized_width = resized_image.shape[1]

    resize_scale = (
        resized_width
        / original_width
    )

    original_table_contour = (
        table_contour.astype("float32")
        / resize_scale
    )

    warped_table = warp_perspective(
        image=original_image,
        contour=original_table_contour,
        output_width=STANDARD_TABLE_WIDTH,
        output_height=STANDARD_TABLE_HEIGHT,
    )

    warped_height, warped_width = (
        warped_table.shape[:2]
    )

    measurement_boxes = build_measurement_boxes(
        image_width=warped_width,
        image_height=warped_height,
    )

    cells = extract_measurement_cells(
        image=warped_table,
        measurement_boxes=measurement_boxes,
        horizontal_margin_ratio=(
            CELL_HORIZONTAL_MARGIN_RATIO
        ),
        vertical_margin_ratio=(
            CELL_VERTICAL_MARGIN_RATIO
        ),
        top_padding_ratio=(
            CELL_TOP_PADDING_RATIO
        ),
        bottom_padding_ratio=(
            CELL_BOTTOM_PADDING_RATIO
        ),
    )

    measurement_grid_overlay = (
        draw_measurement_boxes(
            image=warped_table,
            measurement_boxes=measurement_boxes,
        )
    )

    return PreparedMonitoringSheet(
        detection_preview=detection_preview,
        warped_table=warped_table,
        measurement_grid_overlay=(
            measurement_grid_overlay
        ),
        measurement_boxes=measurement_boxes,
        cells=cells,
    )