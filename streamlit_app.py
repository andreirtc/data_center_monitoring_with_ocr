from __future__ import annotations

import hashlib
import time
from typing import Any

import re
from dataclasses import replace
from datacenter_ocr.numeric_postprocessing import (
    is_valid_value,
    validate_final_reading,
)

from datacenter_ocr.excel_export import (
    create_monitoring_workbook,
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
        "table_editor_version",
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
    verified_blank_filenames: set[str],
) -> list[CellOCRResult]:
    """
    Apply manually verified numeric values or blank-cell decisions.
    """

    updated_results = []

    for result in results:
        if result.filename in verified_blank_filenames:
            updated_results.append(
                replace(
                    result,
                    final_value="",
                    needs_review=False,
                    review_reason=(
                        "Manually verified as a blank cell."
                    ),
                    is_blank=True,
                )
            )

            continue

        corrected_value = corrections.get(
            result.filename
        )

        if corrected_value is None:
            updated_results.append(
                result
            )

            continue

        updated_results.append(
            replace(
                result,
                final_value=corrected_value,
                needs_review=False,
                review_reason=(
                    "Manually verified by the user."
                ),
                is_blank=False,
            )
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

def revalidate_cell_results(
    results: list[CellOCRResult],
) -> list[CellOCRResult]:
    """
    Validate every OCR result using the final workbook rules.

    This catches malformed automatically accepted values such
    as '22.' that did not previously appear in Review items.
    """

    validated_results = []

    for result in results:
        if result.is_blank:
            validated_results.append(
                replace(
                    result,
                    final_value="",
                )
            )

            continue

        normalized_value, validation_error = (
            validate_final_reading(
                value=result.final_value,
                reading_type=result.reading_type,
            )
        )

        if validation_error is not None:
            existing_reason = (
                result.review_reason.strip()
            )

            validation_reason = (
                "Final validation: "
                f"{validation_error}"
            )

            if (
                existing_reason
                and validation_reason
                not in existing_reason
            ):
                combined_reason = (
                    f"{existing_reason} "
                    f"{validation_reason}"
                )
            else:
                combined_reason = (
                    validation_reason
                )

            validated_results.append(
                replace(
                    result,
                    needs_review=True,
                    review_reason=combined_reason,
                )
            )

            continue

        validated_results.append(
            replace(
                result,
                final_value=(
                    normalized_value
                    if normalized_value is not None
                    else ""
                ),
            )
        )

    return validated_results

def create_editable_monitoring_dataframe(
    monitoring_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Create the full editable monitoring table.

    The user may correct any value, even if OCR originally
    accepted it automatically.
    """

    dataframe = pd.DataFrame(
        monitoring_rows
    )

    editable_dataframe = dataframe[
        [
            "day",
            "point",
            "temperature",
            "humidity",
            "temperature_is_blank",
            "humidity_is_blank",
            "temperature_needs_review",
            "humidity_needs_review",
        ]
    ].copy()

    editable_dataframe = (
        editable_dataframe.rename(
            columns={
                "day": "Day",
                "point": "Point",
                "temperature": "Temperature",
                "humidity": "Humidity",
                "temperature_is_blank": (
                    "Temperature Blank"
                ),
                "humidity_is_blank": (
                    "Humidity Blank"
                ),
                "temperature_needs_review": (
                    "Temperature Review"
                ),
                "humidity_needs_review": (
                    "Humidity Review"
                ),
            }
        )
    )

    return editable_dataframe


def apply_monitoring_table_edits(
    results: list[CellOCRResult],
    edited_dataframe: pd.DataFrame,
) -> tuple[
    list[CellOCRResult],
    list[str],
    int,
]:
    """
    Apply user edits from the complete monitoring table.

    Only changed values are marked as manually verified.
    Unchanged OCR results retain their existing review status.
    """

    result_lookup = {
        (
            result.day,
            result.point,
            result.reading_type,
        ): result
        for result in results
    }

    proposed_updates: dict[
        tuple[int, int, str],
        CellOCRResult,
    ] = {}

    validation_errors = []
    changed_count = 0

    for _, dataframe_row in (
        edited_dataframe.iterrows()
    ):
        day = int(
            dataframe_row["Day"]
        )

        point = int(
            dataframe_row["Point"]
        )

        reading_settings = [
            (
                "temperature",
                "Temperature",
                "Temperature Blank",
            ),
            (
                "humidity",
                "Humidity",
                "Humidity Blank",
            ),
        ]

        for (
            reading_type,
            value_column,
            blank_column,
        ) in reading_settings:
            key = (
                day,
                point,
                reading_type,
            )

            current_result = result_lookup[
                key
            ]

            manually_blank = bool(
                dataframe_row[blank_column]
            )

            raw_value = dataframe_row[
                value_column
            ]

            if manually_blank:
                proposed_value = ""
                proposed_is_blank = True

            else:
                (
                    normalized_value,
                    validation_error,
                ) = validate_final_reading(
                    value=raw_value,
                    reading_type=reading_type,
                )

                if validation_error is not None:
                    validation_errors.append(
                        (
                            f"Day {day}, Point {point}, "
                            f"{reading_type.title()}: "
                            f"{validation_error}"
                        )
                    )

                    continue

                if normalized_value is None:
                    validation_errors.append(
                        (
                            f"Day {day}, Point {point}, "
                            f"{reading_type.title()}: "
                            "enter a value or mark the "
                            "cell as blank."
                        )
                    )

                    continue

                proposed_value = normalized_value
                proposed_is_blank = False

            value_changed = (
                str(current_result.final_value)
                != proposed_value
            )

            blank_status_changed = (
                current_result.is_blank
                != proposed_is_blank
            )

            if not (
                value_changed
                or blank_status_changed
            ):
                proposed_updates[key] = (
                    current_result
                )

                continue

            changed_count += 1

            if proposed_is_blank:
                updated_reason = (
                    "Manually verified as blank "
                    "in the monitoring table."
                )
            else:
                updated_reason = (
                    "Manually verified in the "
                    "editable monitoring table."
                )

            proposed_updates[key] = replace(
                current_result,
                final_value=proposed_value,
                is_blank=proposed_is_blank,
                needs_review=False,
                review_reason=updated_reason,
            )

    # Do not apply partial table changes when any
    # validation error exists.
    if validation_errors:
        return (
            results,
            validation_errors,
            0,
        )

    updated_results = []

    for result in results:
        key = (
            result.day,
            result.point,
            result.reading_type,
        )

        updated_results.append(
            proposed_updates.get(
                key,
                result,
            )
        )

    updated_results = revalidate_cell_results(
        updated_results
    )

    return (
        updated_results,
        [],
        changed_count,
    )

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

            cell_results = revalidate_cell_results(
                cell_results
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

    preview_tab, table_tab, review_tab, export_tab = st.tabs(
        [
            "Sheet previews",
            "Monitoring table",
            "Review items",
            "Export Excel",
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
        st.subheader(
            "Editable monitoring table"
        )

        st.write(
            "Review and correct any reading directly in the "
            "table. Day and Point are locked. You may also "
            "mark incorrectly detected readings as blank."
        )

        invalid_or_review_count = sum(
            1
            for result in cell_results
            if result.needs_review
        )

        if invalid_or_review_count:
            st.warning(
                f"{invalid_or_review_count} reading(s) "
                "currently require attention."
            )
        else:
            st.success(
                "All values currently pass final validation."
            )

        show_attention_only = st.checkbox(
            "Show only rows requiring attention",
            value=False,
            key=(
                "show_attention_"
                f"{uploaded_fingerprint}"
            ),
        )

        editable_dataframe = (
            create_editable_monitoring_dataframe(
                monitoring_rows
            )
        )

        if show_attention_only:
            editable_dataframe = (
                editable_dataframe[
                    (
                        editable_dataframe[
                            "Temperature Review"
                        ]
                    )
                    |
                    (
                        editable_dataframe[
                            "Humidity Review"
                        ]
                    )
                ].copy()
            )

        editor_version = (
            st.session_state.get(
                "table_editor_version",
                0,
            )
        )

        edited_dataframe = st.data_editor(
            editable_dataframe,
            use_container_width=True,
            hide_index=True,
            disabled=[
                "Day",
                "Point",
                "Temperature Review",
                "Humidity Review",
            ],
            column_config={
                "Day": st.column_config.NumberColumn(
                    "Day",
                    format="%d",
                ),
                "Point": st.column_config.NumberColumn(
                    "Point",
                    format="%d",
                ),
                "Temperature": (
                    st.column_config.TextColumn(
                        "Temperature",
                        help=(
                            "Use exactly one decimal place, "
                            "such as 22.0."
                        ),
                    )
                ),
                "Humidity": (
                    st.column_config.TextColumn(
                        "Humidity",
                        help=(
                            "Use exactly one decimal place, "
                            "such as 53.3."
                        ),
                    )
                ),
                "Temperature Blank": (
                    st.column_config.CheckboxColumn(
                        "Temp Blank",
                        help=(
                            "Check when the temperature "
                            "cell is actually empty."
                        ),
                    )
                ),
                "Humidity Blank": (
                    st.column_config.CheckboxColumn(
                        "Humidity Blank",
                        help=(
                            "Check when the humidity "
                            "cell is actually empty."
                        ),
                    )
                ),
                "Temperature Review": (
                    st.column_config.CheckboxColumn(
                        "Temp Review",
                    )
                ),
                "Humidity Review": (
                    st.column_config.CheckboxColumn(
                        "Humidity Review",
                    )
                ),
            },
            key=(
                "monitoring_editor_"
                f"{uploaded_fingerprint}_"
                f"{editor_version}"
            ),
        )

        save_table_changes_clicked = st.button(
            "Save monitoring table changes",
            type="primary",
            key=(
                "save_table_"
                f"{uploaded_fingerprint}"
            ),
        )

        if save_table_changes_clicked:
            (
                updated_cell_results,
                table_validation_errors,
                changed_count,
            ) = apply_monitoring_table_edits(
                results=cell_results,
                edited_dataframe=edited_dataframe,
            )

            if table_validation_errors:
                st.error(
                    "Some table values could not be saved."
                )

                for table_error in (
                    table_validation_errors
                ):
                    st.write(
                        f"- {table_error}"
                    )

                st.info(
                    "Correct the highlighted value, then press "
                    "Enter or click outside the cell again."
                )

            elif changed_count > 0:
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

                st.session_state[
                    "table_editor_version"
                ] = (
                    editor_version + 1
                )

                remaining_review_count = sum(
                    1
                    for result in updated_cell_results
                    if result.needs_review
                )

                st.toast(
                    f"Saved {changed_count} table change(s). {remaining_review_count} reading(s) still require attention."
                )

                st.rerun()

            else:
                st.caption(
                    "Changes save automatically after you press "
                    "Enter, press Tab, or click outside the cell."
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

            blank_inputs: dict[
                str,
                bool,
            ] = {}

            with st.form(
                "manual_review_form",
                enter_to_submit=True,
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

                            blank_inputs[
                                result.filename
                            ] = st.checkbox(
                                "This cell is actually blank",
                                key=(
                                    "review_blank_"
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
                verified_blank_filenames = set()
                validation_errors = []

                for result in review_results:
                    if blank_inputs[result.filename]:
                        verified_blank_filenames.add(
                            result.filename
                        )

                        continue

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

                elif (
                    not valid_corrections
                    and not verified_blank_filenames
                ):
                    st.info(
                        "No verified values were entered. "
                        "Nothing was saved."
                    )

                else:
                    updated_cell_results = (
                        apply_manual_corrections(
                            results=cell_results,
                            corrections=valid_corrections,
                            verified_blank_filenames=(
                                verified_blank_filenames
                            ),
                        )
                    )

                    updated_cell_results = (
                        revalidate_cell_results(
                            updated_cell_results
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
                        f"{len(valid_corrections)} value correction(s) and "
                        f"{len(verified_blank_filenames)} "
                        f"blank-cell confirmation(s). "
                        f"{remaining_review_count} "
                        f"review item(s) remain."
                    )

                    st.rerun()

    with export_tab:
        st.subheader(
            "Export final company workbook"
        )

        unresolved_results = [
            result
            for result in cell_results
            if result.needs_review
        ]

        if unresolved_results:
            st.warning(
                f"{len(unresolved_results)} reading(s) "
                "still require correction or verification."
            )

            st.write(
                "Complete all review items before "
                "downloading the final Excel file."
            )

        else:
            st.success(
                "All flagged readings have been reviewed."
            )

            st.write(
                "The exported workbook will use the official "
                "company template and preserve its logo, "
                "formatting, formulas, borders, and print layout."
            )

            month_year_value = st.text_input(
                "Month/Year",
                placeholder=(
                    "Example: July 2026"
                ),
                key=(
                    "export_month_year_"
                    f"{uploaded_fingerprint}"
                ),
            )

            final_values_confirmed = st.checkbox(
                "I have reviewed the final temperature "
                "and humidity values.",
                key=(
                    "export_confirmed_"
                    f"{uploaded_fingerprint}"
                ),
            )

            if not final_values_confirmed:
                st.info(
                    "Review the Monitoring table, then "
                    "confirm the values to enable download."
                )

            else:
                try:
                    completed_workbook = (
                        create_monitoring_workbook(
                            monitoring_rows=(
                                monitoring_rows
                            ),
                            month_year=(
                                month_year_value
                            ),
                        )
                    )

                except (
                    ValueError,
                    FileNotFoundError,
                ) as error:
                    st.error(
                        str(error)
                    )

                else:
                    st.download_button(
                        label=(
                            "Download completed Excel file"
                        ),
                        data=completed_workbook,
                        file_name=(
                            "Data_Center_Temperature_"
                            "Monitoring_Final.xlsx"
                        ),
                        mime=(
                            "application/vnd.openxmlformats-"
                            "officedocument.spreadsheetml.sheet"
                        ),
                        type="primary",
                    )

                    st.caption(
                        "The original template stored in the "
                        "templates folder is not modified."
                    )