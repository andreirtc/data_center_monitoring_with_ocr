# These reference coordinates were measured from the verified
# perspective-corrected monitoring table.
#
# The original verified warped-table reference was:
# width  = 1097 pixels
# height = 577 pixels

REFERENCE_TABLE_WIDTH = 1097
REFERENCE_TABLE_HEIGHT = 577

# Boundaries of the 16 measurement columns:
#
# Point 1 Temp, Point 1 Humidity,
# Point 2 Temp, Point 2 Humidity,
# ...
# Point 8 Temp, Point 8 Humidity
#
# Seventeen boundaries are required to create sixteen columns.
REFERENCE_MEASUREMENT_X_BOUNDS = [
    20,
    70,
    120,
    170,
    218,
    267,
    315,
    362,
    410,
    458,
    504,
    552,
    598,
    646,
    692,
    740,
    786,
]

# Vertical range containing Day 1 through Day 31.
REFERENCE_DATA_TOP = 98
REFERENCE_DATA_BOTTOM = 570

DAY_COUNT = 31
POINT_COUNT = 8


def scale_coordinate(
    reference_coordinate: int,
    reference_size: int,
    actual_size: int,
) -> int:
    """
    Convert a coordinate measured on the reference image
    into the equivalent coordinate on the current image.
    """

    ratio = reference_coordinate / reference_size

    scaled_coordinate = round(
        ratio * actual_size
    )

    return scaled_coordinate


def build_measurement_boxes(
    image_width: int,
    image_height: int,
) -> list[dict]:
    """
    Build the coordinates for all 496 temperature
    and humidity measurement cells.
    """

    # Convert the reference x positions into coordinates
    # for the current standardized image.
    x_bounds = []

    for reference_x in REFERENCE_MEASUREMENT_X_BOUNDS:
        scaled_x = scale_coordinate(
            reference_coordinate=reference_x,
            reference_size=REFERENCE_TABLE_WIDTH,
            actual_size=image_width,
        )

        x_bounds.append(scaled_x)

    # Convert the top and bottom data positions.
    data_top = scale_coordinate(
        reference_coordinate=REFERENCE_DATA_TOP,
        reference_size=REFERENCE_TABLE_HEIGHT,
        actual_size=image_height,
    )

    data_bottom = scale_coordinate(
        reference_coordinate=REFERENCE_DATA_BOTTOM,
        reference_size=REFERENCE_TABLE_HEIGHT,
        actual_size=image_height,
    )

    # There are 31 equally spaced daily rows.
    row_height = (
        data_bottom - data_top
    ) / DAY_COUNT

    measurement_boxes = []

    for day_index in range(DAY_COUNT):
        day_number = day_index + 1

        y1 = round(
            data_top + day_index * row_height
        )

        y2 = round(
            data_top + (day_index + 1) * row_height
        )

        # Sixteen measurement columns:
        # 8 temperature and 8 humidity.
        for column_index in range(POINT_COUNT * 2):
            x1 = x_bounds[column_index]
            x2 = x_bounds[column_index + 1]

            point_number = (
                column_index // 2
            ) + 1

            if column_index % 2 == 0:
                reading_type = "temperature"
            else:
                reading_type = "humidity"

            measurement_box = {
                "day": day_number,
                "point": point_number,
                "reading_type": reading_type,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }

            measurement_boxes.append(
                measurement_box
            )

    return measurement_boxes