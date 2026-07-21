from __future__ import annotations

import base64
import hashlib
import time
from collections import Counter
from typing import Any

import cv2
import numpy as np
import pandas as pd
from paddleocr import TextRecognition
import streamlit as st

from datacenter_ocr.excel_export import create_monitoring_workbook
from datacenter_ocr.monitoring_records import (
    attach_crop_data_urls,
    build_monitoring_rows,
)
from datacenter_ocr.ocr_processing import (
    CellOCRResult,
    process_measurement_cells_with_blank_detection,
)
from datacenter_ocr.review_workflow import (
    apply_monitoring_table_edits,
    apply_quick_review_controls,
    apply_review_actions,
    clamp_review_state,
)
from datacenter_ocr.sheet_processing import (
    PreparedMonitoringSheet,
    prepare_monitoring_sheet,
)


CELLS_PER_PROCESSING_BATCH = 32
REVIEW_ITEMS_PER_PAGE = 15
TABLE_ROWS_PER_PAGE = 40
REVIEW_ACTIONS = [
    "Leave unresolved",
    "Confirm current",
    "Enter correction",
    "Mark blank",
]


st.set_page_config(
    page_title="Data Center Monitoring OCR",
    page_icon=":material/document_scanner:",
    layout="wide",
)


@st.cache_resource
def load_ocr_model() -> TextRecognition:
    """Load PaddleOCR once and reuse it across Streamlit reruns."""

    print("[OCR MODEL] Loading PaddleOCR model into memory...")
    return TextRecognition(device="cpu")


def decode_uploaded_image(uploaded_bytes: bytes) -> np.ndarray:
    """Decode uploaded PNG or JPEG bytes into an OpenCV image."""

    byte_array = np.frombuffer(uploaded_bytes, dtype=np.uint8)
    image = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file could not be decoded as an image.")
    return image


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV's BGR image format for Streamlit display."""

    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def create_file_fingerprint(uploaded_bytes: bytes) -> str:
    """Create a stable identifier for an uploaded image."""

    return hashlib.sha256(uploaded_bytes).hexdigest()


def clear_previous_results() -> None:
    """Clear canonical results when the user uploads a different image."""

    for key in (
        "prepared_sheet",
        "cell_results",
        "monitoring_rows",
        "processing_seconds",
        "processed_fingerprint",
        "table_editor_version",
        "cell_crop_data_urls",
        "review_flash",
        "table_flash",
    ):
        st.session_state.pop(key, None)


def count_blank_cells(results: list[CellOCRResult]) -> int:
    return sum(result.is_blank for result in results)


def count_review_cells(results: list[CellOCRResult]) -> int:
    return sum(result.blocks_export for result in results)


def count_accepted_filled_cells(results: list[CellOCRResult]) -> int:
    return sum(
        not result.is_blank and not result.blocks_export for result in results
    )


def build_cell_image_lookup(
    prepared_sheet: PreparedMonitoringSheet,
) -> dict[str, np.ndarray]:
    """Map each result filename to its extracted source cell."""

    return {cell["filename"]: cell["image"] for cell in prepared_sheet.cells}


def encode_crop_data_url(image: np.ndarray) -> str:
    """Encode one extracted cell as a PNG data URL for ImageColumn."""

    encoded, png_buffer = cv2.imencode(".png", image)
    if not encoded:
        raise ValueError("An extracted cell crop could not be encoded.")
    payload = base64.b64encode(png_buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def build_cell_crop_data_url_lookup(
    prepared_sheet: PreparedMonitoringSheet,
) -> dict[str, str]:
    """Build filename-keyed data URLs without introducing another data model."""

    return {
        cell["filename"]: encode_crop_data_url(cell["image"])
        for cell in prepared_sheet.cells
    }


def create_editable_monitoring_dataframe(
    monitoring_rows: list[dict[str, Any]],
    crop_data_urls: dict[str, str],
) -> pd.DataFrame:
    """Build the image-assisted editor from canonical monitoring rows."""

    rows_with_crops = attach_crop_data_urls(monitoring_rows, crop_data_urls)
    columns = [
        "day",
        "point",
        "temperature_crop",
        "temperature",
        "temperature_is_blank",
        "humidity_crop",
        "humidity",
        "humidity_is_blank",
        "status",
        "status_reason",
    ]
    dataframe = pd.DataFrame(rows_with_crops, columns=columns)
    return dataframe.rename(
        columns={
            "day": "Day",
            "point": "Point",
            "temperature_crop": "Temperature Crop",
            "temperature": "Temperature",
            "temperature_is_blank": "Temperature Blank",
            "humidity_crop": "Humidity Crop",
            "humidity": "Humidity",
            "humidity_is_blank": "Humidity Blank",
            "status": "Status",
            "status_reason": "Reason",
        }
    )


def store_verified_results(
    results: list[CellOCRResult],
    uploaded_fingerprint: str,
) -> None:
    """Store the canonical result graph after a user update."""

    st.session_state["cell_results"] = results
    st.session_state["monitoring_rows"] = build_monitoring_rows(results)
    st.session_state["table_editor_version"] = (
        st.session_state.get("table_editor_version", 0) + 1
    )
    st.session_state.pop(f"export_confirmed_{uploaded_fingerprint}", None)


def review_action_key(uploaded_fingerprint: str, filename: str) -> str:
    return f"review_action_{uploaded_fingerprint}_{filename}"


def review_correction_key(uploaded_fingerprint: str, filename: str) -> str:
    return f"review_correction_{uploaded_fingerprint}_{filename}"


def quick_review_confirm_key(uploaded_fingerprint: str, filename: str) -> str:
    return f"quick_review_confirm_{uploaded_fingerprint}_{filename}"


def quick_review_blank_key(uploaded_fingerprint: str, filename: str) -> str:
    return f"quick_review_blank_{uploaded_fingerprint}_{filename}"


def save_quick_review_callback(
    uploaded_fingerprint: str,
    filenames: list[str],
) -> None:
    """Apply explicit quick-review controls before the normal form rerun."""

    controls = {
        filename: {
            "corrected_value": st.session_state.get(
                review_correction_key(uploaded_fingerprint, filename), ""
            ),
            "confirm_current": st.session_state.get(
                quick_review_confirm_key(uploaded_fingerprint, filename), False
            ),
            "mark_blank": st.session_state.get(
                quick_review_blank_key(uploaded_fingerprint, filename), False
            ),
        }
        for filename in filenames
    }
    outcome = apply_quick_review_controls(
        results=st.session_state["cell_results"],
        controls=controls,
    )
    if outcome.changed_count:
        store_verified_results(outcome.results, uploaded_fingerprint)
        for filename in outcome.changed_filenames:
            for key in (
                review_correction_key(uploaded_fingerprint, filename),
                quick_review_confirm_key(uploaded_fingerprint, filename),
                quick_review_blank_key(uploaded_fingerprint, filename),
            ):
                st.session_state.pop(key, None)

    remaining = sum(result.blocks_export for result in outcome.results)
    if outcome.errors:
        level = "warning" if outcome.changed_count else "error"
    elif outcome.changed_count:
        level = "success"
    else:
        level = "info"
    message = (
        f"Saved {outcome.changed_count} review item(s). "
        f"{remaining} export-blocking item(s) remain."
        if outcome.changed_count or outcome.errors
        else "No explicit review actions were selected."
    )
    st.session_state["review_flash"] = {
        "level": level,
        "message": message,
        "errors": outcome.errors,
    }


def save_review_batch_callback(
    uploaded_fingerprint: str,
    filenames: list[str],
) -> None:
    """Apply only explicit review actions before the normal widget rerun."""

    actions = {}
    for filename in filenames:
        action = st.session_state.get(
            review_action_key(uploaded_fingerprint, filename),
            "Leave unresolved",
        )
        if action == "Leave unresolved":
            continue
        actions[filename] = (
            action,
            st.session_state.get(
                review_correction_key(uploaded_fingerprint, filename),
                "",
            ),
        )

    if not actions:
        st.session_state["review_flash"] = {
            "level": "info",
            "message": "No explicit review actions were selected.",
            "errors": (),
        }
        return

    outcome = apply_review_actions(
        results=st.session_state["cell_results"],
        actions=actions,
    )
    if outcome.changed_count:
        store_verified_results(outcome.results, uploaded_fingerprint)
        for filename in outcome.changed_filenames:
            st.session_state.pop(
                review_action_key(uploaded_fingerprint, filename), None
            )
            st.session_state.pop(
                review_correction_key(uploaded_fingerprint, filename), None
            )

    remaining = sum(result.blocks_export for result in outcome.results)
    level = "warning" if outcome.errors else "success"
    st.session_state["review_flash"] = {
        "level": level,
        "message": (
            f"Saved {outcome.changed_count} review item(s). "
            f"{remaining} export-blocking item(s) remain."
        ),
        "errors": outcome.errors,
    }


def save_monitoring_table_callback(
    uploaded_fingerprint: str,
    editor_key: str,
    base_rows: list[dict[str, Any]],
) -> None:
    """Apply submitted data-editor changes before the submit rerun."""

    editor_state = st.session_state.get(editor_key, {})
    edited_rows = []
    for row_index, changes in editor_state.get("edited_rows", {}).items():
        base_row = base_rows[int(row_index)]
        edited_rows.append(
            {
                "Day": base_row["Day"],
                "Point": base_row["Point"],
                **changes,
            }
        )

    outcome = apply_monitoring_table_edits(
        results=st.session_state["cell_results"],
        edited_rows=edited_rows,
    )
    if outcome.changed_count:
        store_verified_results(outcome.results, uploaded_fingerprint)
    else:
        st.session_state["table_editor_version"] = (
            st.session_state.get("table_editor_version", 0) + 1
        )

    remaining = sum(result.blocks_export for result in outcome.results)
    if outcome.errors:
        level = "warning" if outcome.changed_count else "error"
        message = (
            f"Saved {outcome.changed_count} valid change(s); "
            f"rejected {len(outcome.errors)} field(s)."
        )
    elif outcome.changed_count:
        level = "success"
        message = (
            f"Saved {outcome.changed_count} change(s). "
            f"{remaining} reading(s) still block export."
        )
    else:
        level = "info"
        message = "No table values or blank flags changed."

    st.session_state["table_flash"] = {
        "level": level,
        "message": message,
        "errors": outcome.errors,
    }


st.title("Data Center Monthly Monitoring OCR")
st.write(
    "Upload a photo or scanned monthly monitoring sheet. The application "
    "straightens the form, extracts 496 readings, and queues every value "
    "that needs human verification."
)

uploaded_file = st.file_uploader(
    "Upload monitoring sheet",
    type=["png", "jpg", "jpeg"],
    help=(
        "The complete monitoring table should be visible. Avoid strong "
        "glare, blur, shadows, or cropped borders."
    ),
)

if uploaded_file is None:
    st.info("Upload a PNG, JPG, or JPEG image to begin.")
    st.stop()

uploaded_bytes = uploaded_file.getvalue()
uploaded_fingerprint = create_file_fingerprint(uploaded_bytes)

if st.session_state.get("uploaded_fingerprint") != uploaded_fingerprint:
    clear_previous_results()
    st.session_state["uploaded_fingerprint"] = uploaded_fingerprint

try:
    uploaded_image = decode_uploaded_image(uploaded_bytes)
except ValueError as error:
    st.error(str(error))
    st.stop()

st.subheader("Uploaded image")
st.image(
    convert_bgr_to_rgb(uploaded_image),
    caption=uploaded_file.name,
    width="stretch",
)

if st.button("Process monitoring sheet", type="primary"):
    if st.session_state.get("processed_fingerprint") == uploaded_fingerprint:
        st.info(
            "This uploaded image has already been processed. The existing "
            "results are shown below."
        )
    else:
        progress_bar = st.progress(0)
        progress_message = st.empty()
        try:
            progress_message.write("Detecting and straightening the table...")
            prepared_sheet = prepare_monitoring_sheet(uploaded_image)
            st.session_state["prepared_sheet"] = prepared_sheet

            progress_message.write("Loading the OCR recognition model...")
            ocr_model = load_ocr_model()
            processing_start = time.perf_counter()

            def update_progress(processed_count: int, total_count: int) -> None:
                """Report whole-sheet OCR progress."""

                percentage = int(processed_count / total_count * 100)
                progress_bar.progress(min(percentage, 100))
                progress_message.write(
                    f"Processing cells: {processed_count}/{total_count}"
                )

            cell_results = process_measurement_cells_with_blank_detection(
                model=ocr_model,
                cells=prepared_sheet.cells,
                cells_per_batch=CELLS_PER_PROCESSING_BATCH,
                progress_callback=update_progress,
            )
            monitoring_rows = build_monitoring_rows(cell_results)
            st.session_state["cell_results"] = cell_results
            st.session_state["monitoring_rows"] = monitoring_rows
            st.session_state["processing_seconds"] = (
                time.perf_counter() - processing_start
            )
            st.session_state["processed_fingerprint"] = uploaded_fingerprint
            st.session_state["table_editor_version"] = 0
            progress_bar.progress(100)
            progress_message.success("Monitoring sheet processed successfully.")
        except ValueError as error:
            progress_message.empty()
            st.error(str(error))
        except Exception as error:
            progress_message.empty()
            st.error(
                "An unexpected error occurred while processing the monitoring sheet."
            )
            with st.expander("Technical error details"):
                st.exception(error)

prepared_sheet: PreparedMonitoringSheet | None = st.session_state.get(
    "prepared_sheet"
)
cell_results: list[CellOCRResult] | None = st.session_state.get("cell_results")
monitoring_rows: list[dict[str, Any]] | None = st.session_state.get(
    "monitoring_rows"
)

if (
    prepared_sheet is not None
    and cell_results is not None
    and monitoring_rows is not None
):
    crop_data_urls = st.session_state.get("cell_crop_data_urls")
    if crop_data_urls is None:
        crop_data_urls = build_cell_crop_data_url_lookup(prepared_sheet)
        st.session_state["cell_crop_data_urls"] = crop_data_urls

    st.divider()
    st.header("Processing results")

    blank_count = count_blank_cells(cell_results)
    review_count = count_review_cells(cell_results)
    review_row_count = sum(row["blocks_export"] for row in monitoring_rows)
    metrics = st.columns(5)
    metrics[0].metric("Total cells", len(cell_results))
    metrics[1].metric("Blank cells", blank_count)
    metrics[2].metric(
        "Accepted filled cells", count_accepted_filled_cells(cell_results)
    )
    metrics[3].metric("Review cells", review_count)
    metrics[4].metric(
        "Processing time", f"{st.session_state['processing_seconds']:.1f} s"
    )
    st.caption(
        f"{review_row_count} of {len(monitoring_rows)} monitoring rows "
        "contain at least one export-blocking item."
    )
    category_counts = Counter(
        category
        for result in cell_results
        for category in set(result.review_categories)
    )
    category_labels = {
        "format": "Malformed",
        "absolute_range": "Absolute range",
        "ocr_uncertainty": "OCR uncertainty",
        "blank_mismatch": "Blank mismatch",
        "anomaly": "Anomaly",
        "operational_warning": "Operational warning",
    }
    st.caption(
        "Non-exclusive category counts: "
        + " · ".join(
            f"{label} {category_counts.get(category, 0)}"
            for category, label in category_labels.items()
        )
    )

    table_tab, review_tab, preview_tab, export_tab = st.tabs(
        ["Monitoring table", "Review items", "Sheet previews", "Export Excel"]
    )

    with preview_tab:
        st.subheader("Detected monitoring table")
        st.image(
            convert_bgr_to_rgb(prepared_sheet.detection_preview), width="stretch"
        )
        st.subheader("Straightened monitoring table")
        st.image(convert_bgr_to_rgb(prepared_sheet.warped_table), width="stretch")
        with st.expander("Show measurement-grid overlay"):
            st.image(
                convert_bgr_to_rgb(prepared_sheet.measurement_grid_overlay),
                width="stretch",
            )

    with table_tab:
        st.subheader("Editable monitoring table")
        st.write(
            "Day, Point, Status, and Reason are locked. Temperature, humidity, "
            "and both blank flags remain editable."
        )
        table_flash = st.session_state.pop("table_flash", None)
        if table_flash is not None:
            getattr(st, table_flash["level"])(table_flash["message"])
            for error_message in table_flash["errors"]:
                st.caption(error_message)

        table_filter = st.selectbox(
            "Rows to show",
            [
                "Unresolved",
                "Blocking errors",
                "Required confirmations",
                "Operational warnings",
                "Informational notices",
                "Blank mismatches",
                "All readings",
            ],
            key=f"table_filter_{uploaded_fingerprint}",
        )
        filtered_rows = [
            row
            for row in monitoring_rows
            if (
                (table_filter == "Unresolved" and row["blocks_export"])
                or (
                    table_filter == "Blocking errors"
                    and bool(row["blocking_errors"])
                )
                or (
                    table_filter == "Required confirmations"
                    and bool(row["required_confirmation_reasons"])
                )
                or (
                    table_filter == "Operational warnings"
                    and bool(row["operational_warnings"])
                )
                or (
                    table_filter == "Informational notices"
                    and bool(row["informational_notices"])
                )
                or (
                    table_filter == "Blank mismatches"
                    and row["has_blank_mismatch"]
                )
                or table_filter == "All readings"
            )
        ]

        if not filtered_rows:
            st.info("No monitoring rows match this filter.")
        else:
            table_page_count = max(
                1,
                (len(filtered_rows) + TABLE_ROWS_PER_PAGE - 1)
                // TABLE_ROWS_PER_PAGE,
            )
            filter_slug = table_filter.lower().replace(" ", "_")
            table_page_key = (
                f"monitoring_page_{uploaded_fingerprint}_{filter_slug}"
            )
            requested_table_page = int(
                st.session_state.get(table_page_key, 1)
            )
            st.session_state[table_page_key] = min(
                max(requested_table_page, 1), table_page_count
            )
            table_page = st.pagination(
                table_page_count,
                key=table_page_key,
                persist_state="session",
            )
            table_page_start = (table_page - 1) * TABLE_ROWS_PER_PAGE
            page_rows = filtered_rows[
                table_page_start : table_page_start + TABLE_ROWS_PER_PAGE
            ]
            editable_dataframe = create_editable_monitoring_dataframe(
                page_rows,
                crop_data_urls,
            )
            st.caption(
                f"Page {table_page} of {table_page_count} · showing "
                f"{len(page_rows)} of {len(filtered_rows)} matching rows. "
                "Double-click a crop for the larger image preview."
            )
            editor_version = st.session_state.get("table_editor_version", 0)
            editor_key = (
                f"monitoring_editor_{uploaded_fingerprint}_"
                f"{filter_slug}_{table_page}_{editor_version}"
            )
            base_rows = editable_dataframe.to_dict(orient="records")
            form_key = (
                f"monitoring_form_{uploaded_fingerprint}_"
                f"{filter_slug}_{table_page}_{editor_version}"
            )

            with st.form(form_key):
                st.data_editor(
                    editable_dataframe,
                    width="stretch",
                    height=520,
                    hide_index=True,
                    disabled=[
                        "Day",
                        "Point",
                        "Temperature Crop",
                        "Humidity Crop",
                        "Status",
                        "Reason",
                    ],
                    column_config={
                        "Day": st.column_config.NumberColumn(
                            "Day", format="%d", pinned=True
                        ),
                        "Point": st.column_config.NumberColumn(
                            "Point", format="%d", pinned=True
                        ),
                        "Temperature Crop": st.column_config.ImageColumn(
                            "Temperature crop",
                            help="Double-click to inspect the extracted crop.",
                            width="small",
                        ),
                        "Temperature": st.column_config.TextColumn(
                            "Temperature",
                            help="Use exactly one decimal place, such as 22.0.",
                        ),
                        "Humidity": st.column_config.TextColumn(
                            "Humidity",
                            help="Use exactly one decimal place, such as 53.3.",
                        ),
                        "Temperature Blank": st.column_config.CheckboxColumn(
                            "Temperature blank"
                        ),
                        "Humidity Crop": st.column_config.ImageColumn(
                            "Humidity crop",
                            help="Double-click to inspect the extracted crop.",
                            width="small",
                        ),
                        "Humidity Blank": st.column_config.CheckboxColumn(
                            "Humidity blank"
                        ),
                        "Reason": st.column_config.TextColumn(
                            "Reason", width="large"
                        ),
                    },
                    key=editor_key,
                )
                st.form_submit_button(
                    "Save monitoring table changes",
                    type="primary",
                    on_click=save_monitoring_table_callback,
                    args=(uploaded_fingerprint, editor_key, base_rows),
                )

    with review_tab:
        review_results = [result for result in cell_results if result.blocks_export]
        review_flash = st.session_state.pop("review_flash", None)
        if review_flash is not None:
            getattr(st, review_flash["level"])(review_flash["message"])
            for error_message in review_flash["errors"]:
                st.caption(error_message)

        if not review_results:
            st.success("No readings currently block export.")
        else:
            st.warning(
                f"{len(review_results)} readings contain blocking errors or "
                "unresolved confirmations."
            )
            review_mode = st.segmented_control(
                "Review mode",
                ["Quick Batch Review", "Detailed Review"],
                default="Quick Batch Review",
                required=True,
                key=f"review_mode_{uploaded_fingerprint}",
            )
            result_by_filename = {
                result.filename: result for result in review_results
            }
            cell_images = build_cell_image_lookup(prepared_sheet)

            if review_mode == "Quick Batch Review":
                review_filter = st.selectbox(
                    "Items to show",
                    [
                        "All unresolved",
                        "Blocking errors",
                        "Required confirmations",
                        "OCR uncertainty",
                        "Anomalies",
                        "Blank mismatch",
                    ],
                    key=f"review_filter_{uploaded_fingerprint}",
                )
                filtered_results = [
                    result
                    for result in review_results
                    if (
                        review_filter == "All unresolved"
                        or (
                            review_filter == "Blocking errors"
                            and bool(result.blocking_errors)
                        )
                        or (
                            review_filter == "Required confirmations"
                            and bool(result.required_confirmation_reasons)
                            and not result.human_verified
                        )
                        or (
                            review_filter == "OCR uncertainty"
                            and "ocr_uncertainty" in result.review_categories
                        )
                        or (
                            review_filter == "Anomalies"
                            and "anomaly" in result.review_categories
                        )
                        or (
                            review_filter == "Blank mismatch"
                            and "blank_mismatch" in result.review_categories
                        )
                    )
                ]
                if not filtered_results:
                    st.info("No unresolved items match this filter.")
                else:
                    filenames = [result.filename for result in filtered_results]
                    filter_slug = review_filter.lower().replace(" ", "_")
                    page_key = f"review_page_{uploaded_fingerprint}_{filter_slug}"
                    requested_page = int(st.session_state.get(page_key, 1))
                    clamped_page, _ = clamp_review_state(
                        filenames,
                        requested_page,
                        None,
                        page_size=REVIEW_ITEMS_PER_PAGE,
                    )
                    st.session_state[page_key] = clamped_page
                    page_count = max(
                        1,
                        (len(filtered_results) + REVIEW_ITEMS_PER_PAGE - 1)
                        // REVIEW_ITEMS_PER_PAGE,
                    )
                    page = st.pagination(
                        page_count,
                        key=page_key,
                        persist_state="session",
                    )
                    page_start = (page - 1) * REVIEW_ITEMS_PER_PAGE
                    page_results = filtered_results[
                        page_start : page_start + REVIEW_ITEMS_PER_PAGE
                    ]
                    st.caption(
                        f"Page {page} of {page_count} · showing "
                        f"{len(page_results)} of {len(filtered_results)} matching "
                        "items. Untouched items remain unresolved."
                    )

                    quick_form_key = (
                        f"quick_review_form_{uploaded_fingerprint}_"
                        f"{filter_slug}_{page}"
                    )
                    with st.form(quick_form_key, enter_to_submit=True):
                        for result in page_results:
                            with st.container(border=True):
                                image_column, details_column, action_column = (
                                    st.columns([1, 4, 2])
                                )
                                with image_column:
                                    st.image(
                                        convert_bgr_to_rgb(
                                            cell_images[result.filename]
                                        ),
                                        width=110,
                                    )
                                with details_column:
                                    proposed_value = result.final_value or "(blank)"
                                    st.markdown(
                                        f"**Day {result.day}, Point {result.point}, "
                                        f"{result.reading_type.title()}**  "
                                        f"Proposed: `{proposed_value}`"
                                    )
                                    concise_reasons = (
                                        result.blocking_errors
                                        + (
                                            result.required_confirmation_reasons
                                            if not result.human_verified
                                            else ()
                                        )
                                    )
                                    if concise_reasons:
                                        st.caption(" ".join(concise_reasons))
                                    st.text_input(
                                        "Corrected value",
                                        placeholder="Example: 22.0",
                                        key=review_correction_key(
                                            uploaded_fingerprint, result.filename
                                        ),
                                        persist_state="session",
                                    )
                                with action_column:
                                    st.checkbox(
                                        "Confirm proposed value",
                                        key=quick_review_confirm_key(
                                            uploaded_fingerprint, result.filename
                                        ),
                                        persist_state="session",
                                    )
                                    st.checkbox(
                                        "Mark blank",
                                        key=quick_review_blank_key(
                                            uploaded_fingerprint, result.filename
                                        ),
                                        persist_state="session",
                                    )

                        st.form_submit_button(
                            "Save reviewed items on this page",
                            type="primary",
                            on_click=save_quick_review_callback,
                            args=(
                                uploaded_fingerprint,
                                [result.filename for result in page_results],
                            ),
                        )

            else:
                filenames = list(result_by_filename)
                selected_key = f"review_selected_{uploaded_fingerprint}"
                _, selected_filename = clamp_review_state(
                    filenames,
                    1,
                    st.session_state.get(selected_key),
                    page_size=REVIEW_ITEMS_PER_PAGE,
                )
                st.session_state[selected_key] = selected_filename
                review_filename = st.selectbox(
                    "Review item",
                    filenames,
                    format_func=lambda filename: (
                        f"Day {result_by_filename[filename].day}, "
                        f"Point {result_by_filename[filename].point}, "
                        f"{result_by_filename[filename].reading_type.title()}"
                    ),
                    key=selected_key,
                    persist_state="session",
                )
                current_result = result_by_filename[review_filename]
                current_position = review_results.index(current_result) + 1
                st.caption(f"Item {current_position} of {len(review_results)}")

                image_column, details_column = st.columns([1, 2])
                with image_column:
                    st.image(
                        convert_bgr_to_rgb(cell_images[current_result.filename]),
                        caption="Extracted handwritten cell",
                        width="stretch",
                    )
                with details_column:
                    st.subheader(
                        f"Day {current_result.day} · Point {current_result.point} · "
                        f"{current_result.reading_type.title()}"
                    )
                    proposed_value = current_result.final_value or "(blank)"
                    st.markdown(f"**Proposed final value:** `{proposed_value}`")
                    st.write(
                        f"Agreement: {current_result.agreement_count}/3 · "
                        "Consensus confidence: "
                        f"{current_result.average_consensus_confidence:.1%}"
                    )
                    if current_result.blocking_errors:
                        st.error(" ".join(current_result.blocking_errors))
                    if (
                        current_result.required_confirmation_reasons
                        and not current_result.human_verified
                    ):
                        st.warning(
                            " ".join(
                                current_result.required_confirmation_reasons
                            )
                        )
                    if current_result.operational_warnings:
                        st.warning(" ".join(current_result.operational_warnings))
                    if current_result.informational_notices:
                        st.info(" ".join(current_result.informational_notices))
                    prediction_rows = [
                        {
                            "Variant": variant.title(),
                            "Raw OCR": current_result.raw_predictions.get(variant, ""),
                            "Normalized": current_result.predictions.get(variant, ""),
                            "Confidence": current_result.confidences.get(variant, 0.0),
                        }
                        for variant in ("original", "grayscale", "contrast")
                    ]
                    st.dataframe(
                        prediction_rows,
                        hide_index=True,
                        column_config={
                            "Confidence": st.column_config.NumberColumn(
                                "Confidence", format="percent"
                            )
                        },
                    )

                context_rows = [
                    {
                        "Day": row["day"],
                        "Point": row["point"],
                        "Temperature": row["temperature"],
                        "Humidity": row["humidity"],
                        "Status": row["status"],
                    }
                    for row in monitoring_rows
                    if (
                        row["day"] == current_result.day
                        or (
                            row["point"] == current_result.point
                            and abs(row["day"] - current_result.day) <= 2
                        )
                    )
                ]
                with st.expander("Neighboring and same-day context"):
                    st.dataframe(context_rows, hide_index=True)

                detailed_action = st.selectbox(
                    "Action",
                    REVIEW_ACTIONS,
                    key=review_action_key(
                        uploaded_fingerprint, current_result.filename
                    ),
                    persist_state="session",
                )
                if detailed_action == "Enter correction":
                    st.text_input(
                        "Corrected value",
                        placeholder="Example: 22.0",
                        key=review_correction_key(
                            uploaded_fingerprint, current_result.filename
                        ),
                        persist_state="session",
                    )
                st.button(
                    "Save detailed review",
                    type="primary",
                    on_click=save_review_batch_callback,
                    args=(uploaded_fingerprint, [current_result.filename]),
                )

    with export_tab:
        st.subheader("Export final company workbook")
        blocked_results = [
            result for result in cell_results if result.blocks_export
        ]
        if blocked_results:
            st.warning(
                f"{len(blocked_results)} reading(s) contain blocking errors or "
                "unresolved confirmations."
            )
            st.write(
                "Resolve all export-blocking Review items before downloading."
            )
        else:
            st.success("All flagged readings have been reviewed.")
            st.write(
                "The export uses the official template and preserves its logo, "
                "formatting, formulas, borders, and print layout."
            )
            month_year_value = st.text_input(
                "Month/Year",
                placeholder="Example: July 2026",
                key=f"export_month_year_{uploaded_fingerprint}",
            )
            final_values_confirmed = st.checkbox(
                "I have reviewed the final temperature and humidity values.",
                key=f"export_confirmed_{uploaded_fingerprint}",
            )
            if not final_values_confirmed:
                st.info(
                    "Review the Monitoring table, then confirm the values to "
                    "enable download."
                )
            else:
                try:
                    completed_workbook = create_monitoring_workbook(
                        monitoring_rows=monitoring_rows,
                        month_year=month_year_value,
                    )
                except (ValueError, FileNotFoundError) as error:
                    st.error(str(error))
                else:
                    st.download_button(
                        label="Download completed Excel file",
                        data=completed_workbook,
                        file_name=(
                            "Data_Center_Temperature_Monitoring_Final.xlsx"
                        ),
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "spreadsheetml.sheet"
                        ),
                        type="primary",
                    )
                    st.caption(
                        "The original template stored in templates is not modified."
                    )
