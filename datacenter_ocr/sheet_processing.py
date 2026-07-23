from __future__ import annotations

from dataclasses import dataclass
import time

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
from datacenter_ocr.processing_metrics import ProcessingMetrics
from datacenter_ocr.local_grid import (
    GeometryMode,
    LocalGridCalibration,
    RowSequenceAlignment,
    align_measurement_boxes_to_printed_rows,
    calibrate_local_grid,
    draw_calibrated_grid_overlay,
    extract_calibrated_measurement_cells,
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
    geometry_mode: GeometryMode = "fixed"
    grid_calibration: LocalGridCalibration | None = None
    row_sequence_alignment: RowSequenceAlignment | None = None
    invalid_geometry_cell_count: int = 0


def prepare_monitoring_sheet(
    original_image: np.ndarray,
    metrics: ProcessingMetrics | None = None,
    geometry_mode: GeometryMode = "fixed",
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

    detection_start = time.perf_counter()

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

    if metrics is not None:
        metrics.add_seconds(
            "document_table_detection_seconds",
            time.perf_counter() - detection_start,
        )

    if geometry_mode not in ("fixed", "calibrated"):
        raise ValueError(f"Unknown geometry mode: {geometry_mode}")

    # Detection occurred on the resized copy. Convert the contour
    # coordinates back to the original high-resolution image.
    warp_start = time.perf_counter()
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

    if metrics is not None:
        metrics.add_seconds(
            "perspective_warp_seconds",
            time.perf_counter() - warp_start,
        )

    extraction_start = time.perf_counter()
    warped_height, warped_width = (
        warped_table.shape[:2]
    )

    measurement_boxes = build_measurement_boxes(
        image_width=warped_width,
        image_height=warped_height,
    )
    measurement_boxes, row_sequence_alignment = (
        align_measurement_boxes_to_printed_rows(
            warped_table,
            measurement_boxes,
        )
    )

    fixed_cells = extract_measurement_cells(
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

    grid_calibration = None
    invalid_geometry_cell_count = 0
    if geometry_mode == "calibrated":
        grid_calibration = calibrate_local_grid(
            warped_table,
            measurement_boxes,
        )
        cells, invalid_geometry_cell_count = extract_calibrated_measurement_cells(
            image=warped_table,
            measurement_boxes=measurement_boxes,
            fixed_cells=fixed_cells,
            calibration=grid_calibration,
        )
        measurement_grid_overlay = draw_calibrated_grid_overlay(
            warped_table,
            grid_calibration,
        )
    else:
        cells = fixed_cells
        measurement_grid_overlay = draw_measurement_boxes(
            image=warped_table,
            measurement_boxes=measurement_boxes,
        )

    if metrics is not None:
        metrics.add_seconds(
            "measurement_cell_extraction_seconds",
            time.perf_counter() - extraction_start,
        )

    return PreparedMonitoringSheet(
        detection_preview=detection_preview,
        warped_table=warped_table,
        measurement_grid_overlay=(
            measurement_grid_overlay
        ),
        measurement_boxes=measurement_boxes,
        cells=cells,
        geometry_mode=geometry_mode,
        grid_calibration=grid_calibration,
        row_sequence_alignment=row_sequence_alignment,
        invalid_geometry_cell_count=invalid_geometry_cell_count,
    )


def prepare_calibrated_monitoring_sheet(
    fixed_sheet: PreparedMonitoringSheet,
) -> PreparedMonitoringSheet:
    """Calibrate an already warped fixed sheet without repeating table detection."""

    calibration = calibrate_local_grid(
        fixed_sheet.warped_table,
        fixed_sheet.measurement_boxes,
    )
    calibrated_cells, invalid_count = extract_calibrated_measurement_cells(
        image=fixed_sheet.warped_table,
        measurement_boxes=fixed_sheet.measurement_boxes,
        fixed_cells=fixed_sheet.cells,
        calibration=calibration,
    )
    return PreparedMonitoringSheet(
        detection_preview=fixed_sheet.detection_preview,
        warped_table=fixed_sheet.warped_table,
        measurement_grid_overlay=draw_calibrated_grid_overlay(
            fixed_sheet.warped_table,
            calibration,
        ),
        measurement_boxes=fixed_sheet.measurement_boxes,
        cells=calibrated_cells,
        geometry_mode="calibrated",
        grid_calibration=calibration,
        row_sequence_alignment=fixed_sheet.row_sequence_alignment,
        invalid_geometry_cell_count=invalid_count,
    )
