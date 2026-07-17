from pathlib import Path

import cv2
import numpy as np


def crop_measurement_cell(
    image: np.ndarray,
    measurement_box: dict,
    horizontal_margin_ratio: float,
    vertical_margin_ratio: float,
    top_padding_ratio: float,
    bottom_padding_ratio: float,
) -> np.ndarray:
    """
    Crop one handwritten measurement from the clean warped table.

    A small inner margin is applied to reduce the printed cell borders.
    """

    x1 = measurement_box["x1"]
    y1 = measurement_box["y1"]
    x2 = measurement_box["x2"]
    y2 = measurement_box["y2"]

    cell_width = x2 - x1
    cell_height = y2 - y1

    if cell_width <= 0 or cell_height <= 0:
        raise ValueError(
            f"Invalid measurement box: {measurement_box}"
        )

    horizontal_margin = max(
        0,
        round(cell_width * horizontal_margin_ratio),
    )

    vertical_margin = max(
        0,
        round(cell_height * vertical_margin_ratio),
    )

    top_padding = max(
        0,
        round(cell_height * top_padding_ratio),
    )
    
    bottom_padding = max(
        0,
        round(cell_height * bottom_padding_ratio),
    )

    inner_x1 = x1 + horizontal_margin
    inner_x2 = x2 - horizontal_margin
    
    # Extend slightly beyond the original row boundaries.
    inner_y1 = y1 + vertical_margin - top_padding
    inner_y2 = y2 - vertical_margin + bottom_padding

    image_height, image_width = image.shape[:2]

    # Keep the coordinates inside the actual image boundaries.
    inner_x1 = max(0, inner_x1)
    inner_y1 = max(0, inner_y1)
    inner_x2 = min(image_width, inner_x2)
    inner_y2 = min(image_height, inner_y2)

    if inner_x1 >= inner_x2 or inner_y1 >= inner_y2:
        raise ValueError(
            f"The inner crop became empty: {measurement_box}"
        )

    # NumPy image slicing uses:
    # image[y_start:y_end, x_start:x_end]
    cropped_cell = image[
        inner_y1:inner_y2,
        inner_x1:inner_x2,
    ].copy()

    return cropped_cell


def create_cell_filename(
    measurement_box: dict,
) -> str:
    """
    Create a descriptive and consistently sortable filename.
    """

    day = measurement_box["day"]
    point = measurement_box["point"]
    reading_type = measurement_box["reading_type"]

    return (
        f"day_{day:02d}_"
        f"point_{point:02d}_"
        f"{reading_type}.png"
    )


def extract_selected_cells(
    image: np.ndarray,
    measurement_boxes: list[dict],
    selected_days: list[int],
    selected_points: list[int],
    output_folder: Path,
    horizontal_margin_ratio: float,
    vertical_margin_ratio: float,
    top_padding_ratio: float,
    bottom_padding_ratio: float,
) -> list[dict]:
    """
    Extract and save only the selected days and monitoring points.

    Returns metadata describing every saved crop.
    """

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    extracted_cells = []

    for measurement_box in measurement_boxes:
        day = measurement_box["day"]
        point = measurement_box["point"]

        if day not in selected_days:
            continue

        if point not in selected_points:
            continue

        cropped_cell = crop_measurement_cell(
            image=image,
            measurement_box=measurement_box,
            horizontal_margin_ratio=horizontal_margin_ratio,
            vertical_margin_ratio=vertical_margin_ratio,
            top_padding_ratio=top_padding_ratio,
            bottom_padding_ratio=bottom_padding_ratio,
        )

        filename = create_cell_filename(
            measurement_box
        )

        output_path = output_folder / filename

        saved_successfully = cv2.imwrite(
            str(output_path),
            cropped_cell,
        )

        if not saved_successfully:
            raise OSError(
                f"Could not save cell image:\n{output_path}"
            )

        extracted_cells.append(
            {
                **measurement_box,
                "filename": filename,
                "output_path": output_path,
                "image": cropped_cell,
            }
        )

    return extracted_cells