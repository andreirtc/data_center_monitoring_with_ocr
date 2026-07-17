from pathlib import Path


# Folder containing this project.
PROJECT_FOLDER = Path(__file__).resolve().parent

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