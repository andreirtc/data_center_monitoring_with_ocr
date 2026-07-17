import cv2

from config import (
    CANNY_HIGH_THRESHOLD,
    CANNY_LOW_THRESHOLD,
    MAXIMUM_IMAGE_WIDTH,
    MINIMUM_TABLE_AREA_RATIO,
    OUTPUT_FOLDER,
    POLYGON_APPROXIMATION_RATIO,
    TABLE_ROI_BOTTOM_RATIO,
    TABLE_ROI_LEFT_RATIO,
    TABLE_ROI_RIGHT_RATIO,
    TABLE_ROI_TOP_RATIO,
    TEST_IMAGE_PATH,
)

from image_processing import (
    connect_edges,
    convert_to_grayscale,
    detect_edges,
    draw_document_detection,
    find_table_contour,
    load_image,
    resize_to_maximum_width,
    save_image,
    warp_perspective,
)


def main() -> None:
    """Run the current document-processing pipeline."""

    print("Loading image...")

    # 1. Load the original image

    original_image = load_image(
        TEST_IMAGE_PATH
    )

    # 2. Resize the image

    resized_image = resize_to_maximum_width(
        original_image,
        MAXIMUM_IMAGE_WIDTH,
    )

    # 3. Convert the image to grayscale

    grayscale_image = convert_to_grayscale(
        resized_image
    )

    # 4. Create thin canny edges

    raw_edges = detect_edges(
        grayscale_image,
        CANNY_LOW_THRESHOLD,
        CANNY_HIGH_THRESHOLD,
    )

    # 5. Connect broken edges for contour detection

    connected_edges = connect_edges(
        raw_edges,
        kernel_size=3,
        iterations=1,
    )

    # 6. Find the outer table border

    table_contour = find_table_contour(
        connected_edges=connected_edges,
        left_ratio=TABLE_ROI_LEFT_RATIO,
        right_ratio=TABLE_ROI_RIGHT_RATIO,
        top_ratio=TABLE_ROI_TOP_RATIO,
        bottom_ratio=TABLE_ROI_BOTTOM_RATIO,
        minimum_area_ratio=MINIMUM_TABLE_AREA_RATIO,
        approximation_ratio=POLYGON_APPROXIMATION_RATIO,
    )

    # 7. Draw the detected table

    detected_image = draw_document_detection(
        resized_image,
        table_contour,
    )

    # 8. Save the output images

    save_image(
        resized_image,
        OUTPUT_FOLDER / "resized.png",
    )

    save_image(
        grayscale_image,
        OUTPUT_FOLDER / "grayscale.png",
    )

    save_image(
        raw_edges,
        OUTPUT_FOLDER / "detected_edges.png",
    )

    save_image(
        connected_edges,
        OUTPUT_FOLDER / "connected_edges.png",
    )

    save_image(
        detected_image,
        OUTPUT_FOLDER / "detected_document.png",
    )

    # 9. Print the detection result

    if table_contour is None:
        print("No four-corner table border was detected.")
        print("Check the ROI settings and connected edge image.")

    else:
        print("Table border detected successfully.")

        warped_table = warp_perspective(
            resized_image,
            table_contour,
        )

        save_image(
            warped_table,
            OUTPUT_FOLDER / "warped_table.png",
        )

        warped_height, warped_width = warped_table.shape[:2]

        print(
            f"Warped table size: "
            f"{warped_width} x {warped_height}"
        )

        cv2.imshow(
            "Warped Table",
            warped_table,
        )

        corners = table_contour.reshape(4, 2)

        for number, (x, y) in enumerate(
            corners,
            start=1,
        ):
            print(
                f"Corner {number}: "
                f"x={int(x)}, y={int(y)}"
            )

    # 10. Display the outputs

    cv2.imshow(
        "Raw Canny Edges",
        raw_edges,
    )

    cv2.imshow(
        "Connected Edges",
        connected_edges,
    )

    cv2.imshow(
        "Detected Table",
        detected_image,
    )

    print("Press any key inside an image window to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()