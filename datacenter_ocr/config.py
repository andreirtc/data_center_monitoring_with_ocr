from pathlib import Path


# Folder containing this project.
PROJECT_FOLDER = Path(__file__).resolve().parent.parent

# Input and output locations.
TEST_IMAGE_PATH = PROJECT_FOLDER / "test_images" / "sample.png"
OUTPUT_FOLDER = PROJECT_FOLDER / "output"

# Image processing settings.
MAXIMUM_IMAGE_WIDTH = 1200
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
MINIMUM_DOCUMENT_AREA_RATIO = 0.20

TABLE_ROI_LEFT_RATIO = 0.025
TABLE_ROI_RIGHT_RATIO = 0.975
TABLE_ROI_TOP_RATIO = 0.065
TABLE_ROI_BOTTOM_RATIO = 0.80

# The detected table must occupy at least 25% of the search region.
MINIMUM_TABLE_AREA_RATIO = 0.25

# Controls how strongly OpenCV simplifies a contour into corners.
POLYGON_APPROXIMATION_RATIO = 0.02

# Final standardized table dimensions used for cell extraction and OCR.
STANDARD_TABLE_WIDTH = 2400
STANDARD_TABLE_HEIGHT = 1260

# Inner margins used when cropping handwritten values.
#
# These exclude most printed cell borders while retaining
# the handwritten digits and decimal points.
CELL_HORIZONTAL_MARGIN_RATIO = 0.00
CELL_VERTICAL_MARGIN_RATIO = 0.00

# Extend the crop beyond the printed row border.
# The bottom receives more allowance because handwriting
# commonly drops below the baseline.
CELL_TOP_PADDING_RATIO = 0.03
CELL_BOTTOM_PADDING_RATIO = 0.06

# Small controlled sample used while calibrating extraction.
SAMPLE_DAYS = [1, 2, 3]
SAMPLE_POINTS = [1, 2, 3]

TEMPLATES_FOLDER = (
    PROJECT_FOLDER
    / "templates"
)

EXCEL_TEMPLATE_PATH = (
    TEMPLATES_FOLDER
    / "Toyota_Data_Center_Temperature_Monitoring_Template.xlsx"
)