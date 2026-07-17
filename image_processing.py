from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image from a file.

    Raises:
        FileNotFoundError: When OpenCV cannot open the image.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not load the image:\n{image_path}"
        )

    return image


def resize_to_maximum_width(
    image: np.ndarray,
    maximum_width: int,
) -> np.ndarray:
    """
    Resize an image only when it is wider than maximum_width.

    The aspect ratio is preserved.
    """

    height, width = image.shape[:2]

    if width <= maximum_width:
        return image.copy()

    scale = maximum_width / width

    resized_image = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    return resized_image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR color image into grayscale."""

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


def detect_edges(
    grayscale_image: np.ndarray,
    low_threshold: int,
    high_threshold: int,
) -> np.ndarray:
    """
    Create the original thin Canny edges.

    This function does not thicken or connect the edges.
    """

    blurred_image = cv2.GaussianBlur(
        grayscale_image,
        (5, 5),
        0,
    )

    edges = cv2.Canny(
        blurred_image,
        low_threshold,
        high_threshold,
    )

    return edges

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    connected_edges = cv2.dilate(
        edges,
        kernel,
        iterations=1,
    )

    return connected_edges

def connect_edges(
    edges: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Make broken edges thicker and more connected.

    This output is intended for contour detection,
    not for OCR recognition.
    """

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8,
    )

    connected_edges = cv2.dilate(
        edges,
        kernel,
        iterations=iterations,
    )

    return connected_edges

def find_table_contour(
    connected_edges: np.ndarray,
    left_ratio: float,
    right_ratio: float,
    top_ratio: float,
    bottom_ratio: float,
    minimum_area_ratio: float,
    approximation_ratio: float,
) -> np.ndarray | None:
    """
    Find the large four-sided border surrounding the monitoring table.

    The search is limited to a region of interest so that the page edges,
    background, and footer do not interfere with table detection.
    """

    image_height, image_width = connected_edges.shape[:2]

    # Convert ratios into actual pixel coordinates.
    x1 = int(image_width * left_ratio)
    x2 = int(image_width * right_ratio)
    y1 = int(image_height * top_ratio)
    y2 = int(image_height * bottom_ratio)

    # Crop only the expected table area.
    search_region = connected_edges[
        y1:y2,
        x1:x2,
    ]

    contours, _ = cv2.findContours(
        search_region,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    region_height, region_width = search_region.shape[:2]

    minimum_area = (
        region_height
        * region_width
        * minimum_area_ratio
    )

    for contour in contours:
        contour_area = cv2.contourArea(contour)

        if contour_area < minimum_area:
            continue

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        approximate_polygon = cv2.approxPolyDP(
            contour,
            approximation_ratio * perimeter,
            True,
        )

        if (
            len(approximate_polygon) == 4
            and cv2.isContourConvex(approximate_polygon)
        ):
            # The coordinates currently refer to the cropped ROI.
            # Add the ROI offset to convert them back into
            # coordinates for the complete resized image.
            approximate_polygon[:, 0, 0] += x1
            approximate_polygon[:, 0, 1] += y1

            return approximate_polygon

    return None


def draw_document_detection(
    image: np.ndarray,
    document_contour: np.ndarray | None,
) -> np.ndarray:
    """
    Draw the detected document outline and corner points.
    """

    result = image.copy()

    if document_contour is None:
        return result

    cv2.drawContours(
        result,
        [document_contour],
        -1,
        (0, 255, 0),
        4,
    )

    corners = document_contour.reshape(4, 2)

    for corner_number, (x, y) in enumerate(
        corners,
        start=1,
    ):
        x = int(x)
        y = int(y)

        cv2.circle(
            result,
            (x, y),
            10,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            result,
            str(corner_number),
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )

    return result


def save_image(
    image: np.ndarray,
    output_path: Path,
) -> None:
    """Save an image and raise an error when writing fails."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:
        raise OSError(
            f"Could not save image to:\n{output_path}"
        )