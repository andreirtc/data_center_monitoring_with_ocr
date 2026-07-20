from __future__ import annotations

import hashlib
import time
from typing import Any

import re
from dataclasses import replace
from datacenter_ocr.numeric_postprocessing import (
    is_valid_value,
)

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from paddleocr import TextRecognition

from datacenter_ocr.monitoring_records import (
    build_monitoring_rows,
)
from datacenter_ocr.ocr_processing import (
    CellOCRResult,
    process_measurement_cells_with_blank_detection,
)
from datacenter_ocr.sheet_processing import (
    PreparedMonitoringSheet,
    prepare_monitoring_sheet,
)


CELLS_PER_PROCESSING_BATCH = 32


st.set_page_config(
    page_title="Data Center Monitoring OCR",
    page_icon="📋",
    layout="wide",
)

@st.cache_resource
def load_ocr_model() -> TextRecognition:
    print(
        "[OCR MODEL] Loading PaddleOCR model into memory..."
    )
    """
    Load PaddleOCR once and reuse it across Streamlit reruns.
    """

    return TextRecognition(
        device="cpu"
    )


def decode_uploaded_image(
    uploaded_bytes: bytes,
) -> np.ndarray:
    """
    Decode uploaded PNG or JPEG bytes into an OpenCV image.
    """

    byte_array = np.frombuffer(
        uploaded_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        byte_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "The uploaded file could not be decoded as an image."
        )

    return image


def convert_bgr_to_rgb(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert OpenCV's BGR image format for Streamlit display.
    """

    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


def create_file_fingerprint(
    uploaded_bytes: bytes,
) -> str:
    """
    Create a unique identifier for the uploaded image.
    """

    return hashlib.sha256(
        uploaded_bytes
    ).hexdigest()


def clear_previous_results() -> None:
    """
    Clear results when the user uploads a different image.
    """

    keys_to_remove = [
        "prepared_sheet",
        "cell_results",
        "monitoring_rows",
        "processing_seconds",
        "processed_fingerprint",
    ]

    for key in keys_to_remove:
        st.session_state.pop(
            key,
            None,
        )


def create_display_dataframe(
    monitoring_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Create the simplified table shown to the user.
    """

    dataframe = pd.DataFrame(
        monitoring_rows
    )

    display_columns = [
        "day",
        "point",
        "temperature",
        "humidity",
        "needs_review",
        "temperature_is_blank",
        "humidity_is_blank",
        "temperature_needs_review",
        "humidity_needs_review",
    ]

    existing_columns = [
        column
        for column in display_columns
        if column in dataframe.columns
    ]

    return dataframe[
        existing_columns
    ].copy()


def count_blank_cells(
    results: list[CellOCRResult],
) -> int:
    return sum(
        1
        for result in results
        if result.is_blank
    )


def count_review_cells(
    results: list[CellOCRResult],
) -> int:
    return sum(
        1
        for result in results
        if result.needs_review
    )


def count_accepted_filled_cells(
    results: list[CellOCRResult],
) -> int:
    return sum(
        1
        for result in results
        if (
            not result.is_blank
            and not result.needs_review
        )
    )

def validate_manual_value(
    raw_value: str,
    reading_type: str,
) -> tuple[str | None, str | None]:
    """
    Validate a manually entered temperature or humidity value.

    Returns:
        normalized value and no error, or
        no value and an error message.
    """

    cleaned_value = (
        raw_value
        .strip()
        .replace(",", ".")
    )

    if not cleaned_value:
        return None, None

    # Accept whole numbers or values with exactly
    # one digit after the decimal point.
    if re.fullmatch(
        r"-?\d+(?:\.\d)?",
        cleaned_value,
    ) is None:
        return (
            None,
            "Use a number with no more than one "
            "decimal place, such as 21.7.",
        )

    normalized_value = (
        f"{float(cleaned_value):.1f}"
    )

    if not is_valid_value(
        normalized_value,
        reading_type,
    ):
        if reading_type == "temperature":
            valid_range = "10.0 to 50.0"
        else:
            valid_range = "0.0 to 100.0"

        return (
            None,
            f"The value must be within "
            f"{valid_range}.",
        )

    return normalized_value, None


def apply_manual_corrections(
    results: list[CellOCRResult],
    corrections: dict[str, str],
) -> list[CellOCRResult]:
    """
    Apply verified values without modifying the original
    CellOCRResult objects in place.
    """

    updated_results = []

    for result in results:
        corrected_value = corrections.get(
            result.filename
        )

        if corrected_value is None:
            updated_results.append(
                result
            )

            continue

        updated_result = replace(
            result,
            final_value=corrected_value,
            needs_review=False,
            review_reason=(
                "Manually verified by the user."
            ),
        )

        updated_results.append(
            updated_result
        )

    return updated_results


def build_cell_image_lookup(
    prepared_sheet: PreparedMonitoringSheet,
) -> dict[str, np.ndarray]:
    """
    Map each generated filename to its extracted cell image.
    """

    return {
        cell["filename"]: cell["image"]
        for cell in prepared_sheet.cells
    }

st.title(
    "Data Center Monthly Monitoring OCR"
)

st.write(
    "Upload a photo or scanned image of the monthly monitoring "
    "sheet. The application will straighten the form, extract the "
    "temperature and humidity readings, and flag uncertain values "
    "for manual review."
)

uploaded_file = st.file_uploader(
    "Upload monitoring sheet",
    type=[
        "png",
        "jpg",
        "jpeg",
    ],
    help=(
        "The complete monitoring table should be visible. "
        "Avoid strong glare, blur, shadows, or cropped borders."
    ),
)


if uploaded_file is None:
    st.info(
        "Upload a PNG, JPG, or JPEG image to begin."
    )

    st.stop()


uploaded_bytes = uploaded_file.getvalue()

uploaded_fingerprint = create_file_fingerprint(
    uploaded_bytes
)


if (
    st.session_state.get("uploaded_fingerprint")
    != uploaded_fingerprint
):
    clear_previous_results()

    st.session_state[
        "uploaded_fingerprint"
    ] = uploaded_fingerprint


try:
    uploaded_image = decode_uploaded_image(
        uploaded_bytes
    )
except ValueError as error:
    st.error(
        str(error)
    )

    st.stop()


st.subheader(
    "Uploaded image"
)

st.image(
    convert_bgr_to_rgb(
        uploaded_image
    ),
    caption=uploaded_file.name,
    use_container_width=True,
)


process_button_clicked = st.button(
    "Process monitoring sheet",
    type="primary",
)


if process_button_clicked:
    previous_processed_fingerprint = (
        st.session_state.get(
            "processed_fingerprint"
        )
    )

    # Prevent the exact same uploaded image from being
    # processed again unnecessarily.
    if (
        previous_processed_fingerprint
        == uploaded_fingerprint
    ):
        st.info(
            "This uploaded image has already been processed. "
            "The existing results are shown below."
        )

    else:
        progress_bar = st.progress(
            0
        )

        progress_message = st.empty()

        try:
            progress_message.write(
                "Detecting and straightening "
                "the monitoring table..."
            )

            prepared_sheet = (
                prepare_monitoring_sheet(
                    uploaded_image
                )
            )

            st.session_state[
                "prepared_sheet"
            ] = prepared_sheet

            progress_message.write(
                "Loading the OCR recognition model..."
            )

            ocr_model = load_ocr_model()

            processing_start = (
                time.perf_counter()
            )

            def update_progress(
                processed_count: int,
                total_count: int,
            ) -> None:
                """
                Update the Streamlit progress bar
                during OCR processing.
                """

                if total_count <= 0:
                    progress_percentage = 0
                else:
                    progress_percentage = int(
                        processed_count
                        / total_count
                        * 100
                    )

                progress_bar.progress(
                    min(
                        progress_percentage,
                        100,
                    )
                )

                progress_message.write(
                    f"Processing cells: "
                    f"{processed_count}/"
                    f"{total_count}"
                )

            cell_results = (
                process_measurement_cells_with_blank_detection(
                    model=ocr_model,
                    cells=prepared_sheet.cells,
                    cells_per_batch=(
                        CELLS_PER_PROCESSING_BATCH
                    ),
                    progress_callback=(
                        update_progress
                    ),
                )
            )

            monitoring_rows = (
                build_monitoring_rows(
                    cell_results
                )
            )

            processing_seconds = (
                time.perf_counter()
                - processing_start
            )

            st.session_state[
                "cell_results"
            ] = cell_results

            st.session_state[
                "monitoring_rows"
            ] = monitoring_rows

            st.session_state[
                "processing_seconds"
            ] = processing_seconds

            # Record which uploaded image produced
            # the currently stored results.
            st.session_state[
                "processed_fingerprint"
            ] = uploaded_fingerprint

            progress_bar.progress(
                100
            )

            progress_message.success(
                "Monitoring sheet processed "
                "successfully."
            )

        except ValueError as error:
            progress_message.empty()

            st.error(
                str(error)
            )

        except Exception as error:
            progress_message.empty()

            st.error(
                "An unexpected error occurred "
                "while processing the monitoring sheet."
            )

            with st.expander(
                "Technical error details"
            ):
                st.exception(
                    error
                )

prepared_sheet: PreparedMonitoringSheet | None = (
    st.session_state.get(
        "prepared_sheet"
    )
)

cell_results: list[CellOCRResult] | None = (
    st.session_state.get(
        "cell_results"
    )
)

monitoring_rows: list[dict[str, Any]] | None = (
    st.session_state.get(
        "monitoring_rows"
    )
)


if (
    prepared_sheet is not None
    and cell_results is not None
    and monitoring_rows is not None
):
    st.divider()

    st.header(
        "Processing results"
    )

    blank_cell_count = count_blank_cells(
        cell_results
    )

    review_cell_count = count_review_cells(
        cell_results
    )

    accepted_cell_count = (
        count_accepted_filled_cells(
            cell_results
        )
    )

    review_row_count = sum(
        1
        for row in monitoring_rows
        if row["needs_review"]
    )

    metric_columns = st.columns(
        5
    )

    metric_columns[0].metric(
        "Total cells",
        len(cell_results),
    )

    metric_columns[1].metric(
        "Blank cells",
        blank_cell_count,
    )

    metric_columns[2].metric(
        "Accepted filled cells",
        accepted_cell_count,
    )

    metric_columns[3].metric(
        "Review cells",
        review_cell_count,
    )

    metric_columns[4].metric(
        "Processing time",
        (
            f"{st.session_state['processing_seconds']:.1f} s"
        ),
    )

    st.caption(
        f"{review_row_count} of "
        f"{len(monitoring_rows)} monitoring rows "
        f"contain at least one review item."
    )

    preview_tab, table_tab, review_tab = st.tabs(
        [
            "Sheet previews",
            "Monitoring table",
            "Review items",
        ]
    )

    with preview_tab:
        st.subheader(
            "Detected monitoring table"
        )

        st.image(
            convert_bgr_to_rgb(
                prepared_sheet.detection_preview
            ),
            use_container_width=True,
        )

        st.subheader(
            "Straightened monitoring table"
        )

        st.image(
            convert_bgr_to_rgb(
                prepared_sheet.warped_table
            ),
            use_container_width=True,
        )

        with st.expander(
            "Show measurement-grid overlay"
        ):
            st.image(
                convert_bgr_to_rgb(
                    prepared_sheet.measurement_grid_overlay
                ),
                use_container_width=True,
            )

    with table_tab:
        display_dataframe = create_display_dataframe(
            monitoring_rows
        )

        st.dataframe(
            display_dataframe,
            use_container_width=True,
            hide_index=True,
        )

    with review_tab:
        review_results = [
            result
            for result in cell_results
            if result.needs_review
        ]

        if not review_results:
            st.success(
                "All detected readings have been reviewed."
            )

        else:
            st.warning(
                f"{len(review_results)} individual readings "
                f"require manual verification."
            )

            st.caption(
                "You may correct only some items and save them. "
                "Unfilled items will remain in the review list."
            )

            cell_image_lookup = (
                build_cell_image_lookup(
                    prepared_sheet
                )
            )

            correction_inputs: dict[
                str,
                str,
            ] = {}

            with st.form(
                "manual_review_form"
            ):
                for result in review_results:
                    with st.container(
                        border=True
                    ):
                        st.markdown(
                            f"### Day {result.day} — "
                            f"Point {result.point} — "
                            f"{result.reading_type.title()}"
                        )

                        image_column, details_column = (
                            st.columns(
                                [1, 2]
                            )
                        )

                        with image_column:
                            cell_image = (
                                cell_image_lookup[
                                    result.filename
                                ]
                            )

                            st.image(
                                convert_bgr_to_rgb(
                                    cell_image
                                ),
                                caption=(
                                    "Extracted handwritten cell"
                                ),
                                use_container_width=True,
                            )

                        with details_column:
                            consensus_text = (
                                result.consensus_prediction
                                or "(empty)"
                            )

                            st.markdown(
                                f"**OCR consensus:** "
                                f"`{consensus_text}`"
                            )

                            st.markdown(
                                "**Variant predictions:**"
                            )

                            st.write(
                                "Original:",
                                result.predictions[
                                    "original"
                                ]
                                or "(empty)",
                            )

                            st.write(
                                "Grayscale:",
                                result.predictions[
                                    "grayscale"
                                ]
                                or "(empty)",
                            )

                            st.write(
                                "Contrast:",
                                result.predictions[
                                    "contrast"
                                ]
                                or "(empty)",
                            )

                            st.write(
                                f"Agreement: "
                                f"{result.agreement_count}/3"
                            )

                            st.write(
                                f"Reason: "
                                f"{result.review_reason}"
                            )

                            correction_inputs[
                                result.filename
                            ] = st.text_input(
                                "Verified value",
                                value="",
                                placeholder=(
                                    "Example: 53.3"
                                ),
                                key=(
                                    "review_"
                                    f"{uploaded_fingerprint}_"
                                    f"{result.filename}"
                                ),
                            )

                save_corrections_clicked = (
                    st.form_submit_button(
                        "Save verified values",
                        type="primary",
                    )
                )

            if save_corrections_clicked:
                valid_corrections = {}
                validation_errors = []

                for result in review_results:
                    raw_value = correction_inputs[
                        result.filename
                    ]

                    # Empty inputs are intentionally skipped.
                    if not raw_value.strip():
                        continue

                    (
                        normalized_value,
                        validation_error,
                    ) = validate_manual_value(
                        raw_value=raw_value,
                        reading_type=(
                            result.reading_type
                        ),
                    )

                    if validation_error is not None:
                        validation_errors.append(
                            (
                                f"Day {result.day}, "
                                f"Point {result.point}, "
                                f"{result.reading_type.title()}: "
                                f"{validation_error}"
                            )
                        )

                        continue

                    if normalized_value is not None:
                        valid_corrections[
                            result.filename
                        ] = normalized_value

                if validation_errors:
                    st.error(
                        "Some entered values are invalid."
                    )

                    for validation_error in (
                        validation_errors
                    ):
                        st.write(
                            f"- {validation_error}"
                        )

                elif not valid_corrections:
                    st.warning(
                        "Enter at least one verified value "
                        "before saving."
                    )

                else:
                    updated_cell_results = (
                        apply_manual_corrections(
                            results=cell_results,
                            corrections=(
                                valid_corrections
                            ),
                        )
                    )

                    updated_monitoring_rows = (
                        build_monitoring_rows(
                            updated_cell_results
                        )
                    )

                    st.session_state[
                        "cell_results"
                    ] = updated_cell_results

                    st.session_state[
                        "monitoring_rows"
                    ] = updated_monitoring_rows

                    remaining_review_count = sum(
                        1
                        for result
                        in updated_cell_results
                        if result.needs_review
                    )

                    st.success(
                        f"Saved "
                        f"{len(valid_corrections)} "
                        f"verified value(s). "
                        f"{remaining_review_count} "
                        f"review item(s) remain."
                    )

                    st.rerun()