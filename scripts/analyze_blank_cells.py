from __future__ import annotations

import csv

from datacenter_ocr.blank_cell_detection import (
    analyze_cell_for_blankness,
)
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


BLANK_ANALYSIS_FOLDER = (
    OUTPUT_FOLDER
    / "blank_cell_analysis"
)

ANALYSIS_REPORT_PATH = (
    BLANK_ANALYSIS_FOLDER
    / "blank_cell_scores.csv"
)


def main() -> None:
    """
    Measure and visually classify every extracted cell.
    """

    original_image = load_image(
        TEST_IMAGE_PATH
    )

    prepared_sheet = prepare_monitoring_sheet(
        original_image
    )

    analysis_rows = []

    predicted_blank_count = 0
    predicted_filled_count = 0

    for cell in prepared_sheet.cells:
        analysis = analyze_cell_for_blankness(
            cell["image"]
        )

        if analysis.is_blank:
            classification = "predicted_blank"
            predicted_blank_count += 1
        else:
            classification = "predicted_filled"
            predicted_filled_count += 1

        save_image(
            cell["image"],
            BLANK_ANALYSIS_FOLDER
            / classification
            / cell["filename"],
        )

        # Save the cleaned mask so we can understand why
        # the detector classified the cell that way.
        save_image(
            analysis.cleaned_ink_mask,
            BLANK_ANALYSIS_FOLDER
            / "cleaned_masks"
            / classification
            / cell["filename"],
        )

        analysis_rows.append(
            {
                "filename": cell["filename"],
                "day": cell["day"],
                "point": cell["point"],
                "reading_type": cell["reading_type"],
                "is_blank": analysis.is_blank,
                "ink_ratio": round(
                    analysis.ink_ratio,
                    6,
                ),
                "significant_component_count": (
                    analysis.significant_component_count
                ),
            }
        )

    ANALYSIS_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ANALYSIS_REPORT_PATH.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                analysis_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            analysis_rows
        )

    print()
    print("BLANK-CELL ANALYSIS")
    print("-" * 60)

    print(
        f"Total cells: "
        f"{len(analysis_rows)}"
    )

    print(
        f"Predicted blank: "
        f"{predicted_blank_count}"
    )

    print(
        f"Predicted filled: "
        f"{predicted_filled_count}"
    )

    print()
    print(
        "Review predicted blank cells in:\n"
        f"{BLANK_ANALYSIS_FOLDER / 'predicted_blank'}"
    )

    print()
    print(
        "Review predicted filled cells in:\n"
        f"{BLANK_ANALYSIS_FOLDER / 'predicted_filled'}"
    )

    print()
    print(
        "Detailed scores:\n"
        f"{ANALYSIS_REPORT_PATH}"
    )


if __name__ == "__main__":
    main()