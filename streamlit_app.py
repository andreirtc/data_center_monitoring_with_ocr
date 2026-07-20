from __future__ import annotations

import hashlib
import time
from typing import Any

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
        review_rows = [
            row
            for row in monitoring_rows
            if row["needs_review"]
        ]

        if not review_rows:
            st.success(
                "No monitoring rows require manual review."
            )
        else:
            st.warning(
                f"{len(review_rows)} monitoring rows "
                f"require manual verification."
            )

            review_dataframe = (
                create_display_dataframe(
                    review_rows
                )
            )

            st.dataframe(
                review_dataframe,
                use_container_width=True,
                hide_index=True,
            )