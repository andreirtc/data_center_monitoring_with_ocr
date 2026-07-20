from __future__ import annotations

from collections import Counter

from datacenter_ocr.config import (
    OUTPUT_FOLDER,
    TEST_IMAGE_PATH,
)
from datacenter_ocr.image_processing import (
    load_image,
    save_image,
)
from datacenter_ocr.sheet_processing import (
    prepare_monitoring_sheet,
)


EXPECTED_CELL_COUNT = 496
EXPECTED_READING_TYPE_COUNT = 248


SPOT_CHECK_KEYS = {
    (1, 1, "temperature"),
    (1, 1, "humidity"),
    (16, 4, "temperature"),
    (16, 4, "humidity"),
    (31, 8, "temperature"),
    (31, 8, "humidity"),
}


def main() -> None:
    """
    Validate extraction across the beginning, middle,
    and end of the monitoring table.
    """

    original_image = load_image(
        TEST_IMAGE_PATH
    )

    prepared_sheet = prepare_monitoring_sheet(
        original_image
    )

    cells = prepared_sheet.cells

    reading_type_counts = Counter(
        cell["reading_type"]
        for cell in cells
    )

    unique_filenames = {
        cell["filename"]
        for cell in cells
    }

    print()
    print("FULL-SHEET STRUCTURE VALIDATION")
    print("-" * 60)

    print(
        f"Total measurement cells: "
        f"{len(cells)}"
    )

    print(
        f"Temperature cells: "
        f"{reading_type_counts['temperature']}"
    )

    print(
        f"Humidity cells: "
        f"{reading_type_counts['humidity']}"
    )

    print(
        f"Unique filenames: "
        f"{len(unique_filenames)}"
    )

    if len(cells) != EXPECTED_CELL_COUNT:
        raise ValueError(
            "The processor did not generate exactly "
            f"{EXPECTED_CELL_COUNT} measurement cells."
        )

    if (
        reading_type_counts["temperature"]
        != EXPECTED_READING_TYPE_COUNT
    ):
        raise ValueError(
            "The processor did not generate exactly "
            f"{EXPECTED_READING_TYPE_COUNT} "
            "temperature cells."
        )

    if (
        reading_type_counts["humidity"]
        != EXPECTED_READING_TYPE_COUNT
    ):
        raise ValueError(
            "The processor did not generate exactly "
            f"{EXPECTED_READING_TYPE_COUNT} "
            "humidity cells."
        )

    if len(unique_filenames) != EXPECTED_CELL_COUNT:
        raise ValueError(
            "Duplicate cell filenames were generated."
        )

    validation_folder = (
        OUTPUT_FOLDER
        / "full_sheet_validation"
    )

    save_image(
        prepared_sheet.detection_preview,
        validation_folder
        / "detection_preview.png",
    )

    save_image(
        prepared_sheet.warped_table,
        validation_folder
        / "warped_table.png",
    )

    save_image(
        prepared_sheet.measurement_grid_overlay,
        validation_folder
        / "measurement_grid_overlay.png",
    )

    saved_spot_checks = 0

    for cell in cells:
        cell_key = (
            cell["day"],
            cell["point"],
            cell["reading_type"],
        )

        if cell_key not in SPOT_CHECK_KEYS:
            continue

        save_image(
            cell["image"],
            validation_folder
            / "spot_checks"
            / cell["filename"],
        )

        saved_spot_checks += 1

    print(
        f"Saved spot-check crops: "
        f"{saved_spot_checks}"
    )

    print(
        "Structure validation passed."
    )

    print()
    print(
        "Review the validation images in:\n"
        f"{validation_folder}"
    )


if __name__ == "__main__":
    main()