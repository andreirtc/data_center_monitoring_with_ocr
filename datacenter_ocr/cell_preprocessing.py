import cv2
import numpy as np


def convert_cell_to_grayscale(
    cell_image: np.ndarray,
) -> np.ndarray:
    """Convert a measurement-cell image to grayscale."""

    # A grayscale image already has only two dimensions:
    # height and width.
    if len(cell_image.shape) == 2:
        return cell_image.copy()

    return cv2.cvtColor(
        cell_image,
        cv2.COLOR_BGR2GRAY,
    )


def enhance_cell_contrast(
    grayscale_image: np.ndarray,
) -> np.ndarray:
    """
    Improve local contrast so faint handwriting becomes clearer.

    CLAHE means Contrast Limited Adaptive Histogram Equalization.
    """

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(4, 4),
    )

    return clahe.apply(
        grayscale_image
    )


def create_binary_cell(
    contrast_image: np.ndarray,
) -> np.ndarray:
    """
    Convert the cell into a black-and-white image.

    Otsu's method automatically selects the threshold value.
    """

    blurred_image = cv2.GaussianBlur(
        contrast_image,
        (3, 3),
        0,
    )

    _, binary_image = cv2.threshold(
        blurred_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return binary_image


def remove_table_lines(
    binary_image: np.ndarray,
) -> np.ndarray:
    """
    Experimentally detect and remove long horizontal
    and vertical printed lines.

    This may also remove parts of handwriting, so its result
    must be compared with the other variants.
    """

    # The binary image has black ink on white paper.
    # Invert it so ink and table lines become white.
    inverted_image = cv2.bitwise_not(
        binary_image
    )

    height, width = inverted_image.shape[:2]

    horizontal_kernel_width = max(
        10,
        round(width * 0.60),
    )

    vertical_kernel_height = max(
        8,
        round(height * 0.65),
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
        inverted_image,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    vertical_lines = cv2.morphologyEx(
        inverted_image,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    detected_lines = cv2.bitwise_or(
        horizontal_lines,
        vertical_lines,
    )

    handwriting_without_lines = cv2.subtract(
        inverted_image,
        detected_lines,
    )

    return cv2.bitwise_not(
        handwriting_without_lines
    )


def add_panel_label(
    image: np.ndarray,
    label: str,
    scale: int,
) -> np.ndarray:
    """
    Enlarge one image and add a title above it
    for side-by-side comparison.
    """

    if len(image.shape) == 2:
        panel_image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )
    else:
        panel_image = image.copy()

    enlarged_image = cv2.resize(
        panel_image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )

    labelled_panel = cv2.copyMakeBorder(
        enlarged_image,
        35,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    cv2.putText(
        labelled_panel,
        label,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return labelled_panel

def create_ocr_variants(
    cell_image: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Create the preprocessing versions that will be tested by OCR.

    The line-removed version is intentionally excluded because
    our visual test showed that it damages handwritten digits.
    """

    grayscale_image = convert_cell_to_grayscale(
        cell_image
    )

    contrast_image = enhance_cell_contrast(
        grayscale_image
    )

    binary_image = create_binary_cell(
        contrast_image
    )

    return {
        "original": cell_image.copy(),
        "grayscale": grayscale_image,
        "contrast": contrast_image,
        "binary": binary_image,
    }

def create_preprocessing_comparison(
    cell_image: np.ndarray,
    preview_scale: int = 4,
) -> np.ndarray:
    """
    Build one horizontal image comparing all processing variants.
    """

    ocr_variants = create_ocr_variants(
        cell_image
    )

    line_removed_image = remove_table_lines(
        ocr_variants["binary"]
    )

    variants = [
        ("Original", ocr_variants["original"]),
        ("Grayscale", ocr_variants["grayscale"]),
        ("Contrast", ocr_variants["contrast"]),
        ("Binary", ocr_variants["binary"]),
        ("Lines removed", line_removed_image),
    ]

    panels = []

    for label, variant_image in variants:
        panel = add_panel_label(
            image=variant_image,
            label=label,
            scale=preview_scale,
        )

        panels.append(panel)

    comparison_image = cv2.hconcat(
        panels
    )

    return comparison_image