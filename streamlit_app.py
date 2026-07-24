from __future__ import annotations

import base64
import hashlib
import json
import time
from collections import Counter
from dataclasses import replace
from typing import Any
import uuid

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from datacenter_ocr.background_ocr import (
    BackgroundOCRManager,
    OCRJobRequest,
    OCRJobResult,
)
from datacenter_ocr.excel_export import (
    build_excel_mapping_audit,
    create_monitoring_workbook,
)
from datacenter_ocr.day_verification import (
    DayVerificationState,
    apply_day_submission,
    build_day_scroll_request,
    build_verification_audit,
    cell_display_status,
    consume_day_scroll_request,
    invalidate_days_for_changes,
    previous_day,
    results_for_day,
    state_for_sheet,
    summarize_export_readiness,
)
from datacenter_ocr.extraction_preflight import (
    build_alignment_preflight_summary,
    geometry_warning_for_filename,
    recommended_geometry_mode,
    representative_crop_pairs,
)
from datacenter_ocr.grid_diagnostics import calculate_alignment_report
from datacenter_ocr.monitoring_records import (
    attach_crop_data_urls,
    build_monitoring_rows,
)
from datacenter_ocr.ocr_processing import (
    CellOCRResult,
)
from datacenter_ocr.processing_metrics import ProcessingMetrics
from datacenter_ocr.review_workflow import (
    apply_monitoring_table_edits,
    apply_review_actions,
    clamp_review_state,
)
from datacenter_ocr.sheet_processing import (
    PreparedMonitoringSheet,
    prepare_calibrated_monitoring_sheet,
    prepare_monitoring_sheet,
)
from datacenter_ocr.sheet_queue import (
    ArchivedPreparedMonitoringSheet,
    SheetQueueItem,
    archive_prepared_sheet,
    build_sheet_queue_items,
    decode_queue_image,
    resolve_queue_rotation,
    restore_prepared_sheet,
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
ACTIVE_SHEET_STATE_KEYS = (
    "fixed_preflight_sheet",
    "calibrated_preflight_sheet",
    "alignment_report",
    "alignment_preflight_summary",
    "preflight_processing_metrics",
    "prepared_sheet",
    "cell_results",
    "monitoring_rows",
    "processing_seconds",
    "processing_metrics",
    "processed_fingerprint",
    "processed_geometry_mode",
    "prepared_orientation_choice",
    "orientation_preflight_locked",
    "day_verification_state",
    "day_editor_version",
    "day_flash",
    "day_scroll_request",
    "table_editor_version",
    "cell_crop_data_urls",
    "review_flash",
    "table_flash",
)


st.set_page_config(
    page_title="Data Center Monitoring OCR",
    page_icon=":material/document_scanner:",
    layout="wide",
)

_SCROLL_TO_DAY_BANNER = st.components.v2.component(
    "scroll_to_day_verification_banner",
    html="<span aria-hidden='true'></span>",
    css="""
:host {
  display: block;
  height: 1px;
}
""",
    js="""
export default function (component) {
  const message = component.data?.message
  window.requestAnimationFrame(() => {
    const alerts = Array.from(
      document.querySelectorAll('[data-testid="stAlert"], [role="alert"]')
    )
    const target = alerts.find((alert) =>
      message && alert.textContent?.includes(message)
    )
    if (target) {
      target.style.scrollMarginTop = "1rem"
      target.scrollIntoView({ behavior: "smooth", block: "start" })
    }
  })
}
""",
)


@st.cache_resource
def load_background_ocr_manager() -> BackgroundOCRManager:
    """Create one process-wide, single-worker OCR queue."""

    return BackgroundOCRManager()


@st.fragment(run_every="2s")
def render_background_ocr_monitor(owner_id: str) -> None:
    """Show progress and collect an active sheet immediately after OCR."""

    snapshots = load_background_ocr_manager().snapshots_for_owner(owner_id)
    active_sheet_id = st.session_state.get("active_sheet_id")
    if any(
        snapshot.state == "succeeded"
        and snapshot.sheet_id == active_sheet_id
        for snapshot in snapshots
    ):
        # The full rerun collects the finished result before rendering. Limit
        # this to the active sheet so an unrelated completion cannot interrupt
        # edits being made on another sheet.
        st.rerun(scope="app")

    active_snapshots = [
        snapshot
        for snapshot in snapshots
        if snapshot.state
        in ("queued", "loading_model", "running", "succeeded", "failed")
    ]
    if not active_snapshots:
        return

    st.subheader("Background OCR")
    for snapshot in active_snapshots:
        with st.container(border=True):
            if snapshot.state == "queued":
                st.write(f"**{snapshot.source_filename}** · Waiting for OCR")
            elif snapshot.state == "loading_model":
                st.write(
                    f"**{snapshot.source_filename}** · Loading OCR model"
                )
            elif snapshot.state == "running":
                progress = (
                    snapshot.processed_count / snapshot.total_count
                    if snapshot.total_count
                    else 0.0
                )
                st.progress(
                    progress,
                    text=(
                        f"{snapshot.source_filename} · Processing cells "
                        f"{snapshot.processed_count}/{snapshot.total_count}"
                    ),
                )
            elif snapshot.state == "succeeded":
                st.success(
                    f"{snapshot.source_filename} finished OCR. Select that "
                    "sheet when you are ready to load its proposals."
                )
            else:
                st.error(
                    f"{snapshot.source_filename} OCR failed: "
                    f"{snapshot.error_message}"
                )


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV's BGR image format for Streamlit display."""

    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def create_file_fingerprint(uploaded_bytes: bytes) -> str:
    """Create a stable identifier for an uploaded image."""

    return hashlib.sha256(uploaded_bytes).hexdigest()


def clear_previous_results() -> None:
    """Clear the canonical working graph before activating another sheet."""

    for key in ACTIVE_SHEET_STATE_KEYS:
        st.session_state.pop(key, None)


def capture_active_sheet_state(*, archive_images: bool) -> dict[str, Any]:
    """Capture the active canonical graph, optionally compressing image arrays."""

    captured: dict[str, Any] = {}
    archived_by_identity: dict[int, ArchivedPreparedMonitoringSheet] = {}
    for key in ACTIVE_SHEET_STATE_KEYS:
        if key not in st.session_state:
            continue
        value = st.session_state[key]
        if archive_images and isinstance(value, PreparedMonitoringSheet):
            identity = id(value)
            if identity not in archived_by_identity:
                archived_by_identity[identity] = archive_prepared_sheet(value)
            value = archived_by_identity[identity]
        captured[key] = value
    return captured


def restore_sheet_state(saved_state: dict[str, Any]) -> None:
    """Restore active results while leaving optional preflight images archived."""

    restored_by_identity: dict[int, PreparedMonitoringSheet] = {}
    for key, value in saved_state.items():
        if (
            key == "prepared_sheet"
            and isinstance(value, ArchivedPreparedMonitoringSheet)
        ):
            identity = id(value)
            if identity not in restored_by_identity:
                restored_by_identity[identity] = restore_prepared_sheet(value)
            value = restored_by_identity[identity]
        st.session_state[key] = value


def activate_queue_sheet(sheet_id: str) -> None:
    """Archive the previous sheet and activate the selected queue item."""

    queue: dict[str, dict[str, Any]] = st.session_state["sheet_queue"]
    previous_sheet_id = st.session_state.get("active_sheet_id")
    if previous_sheet_id == sheet_id:
        return

    if previous_sheet_id in queue:
        queue[previous_sheet_id]["state"] = capture_active_sheet_state(
            archive_images=True
        )

    clear_previous_results()
    saved_state = queue[sheet_id].get("state", {})
    restore_sheet_state(saved_state)
    queue[sheet_id]["state"] = {}
    st.session_state["active_sheet_id"] = sheet_id
    st.session_state["uploaded_fingerprint"] = sheet_id


def queue_item_state(sheet_id: str) -> dict[str, Any]:
    """Return current state for active or archived queue status rendering."""

    if st.session_state.get("active_sheet_id") == sheet_id:
        return capture_active_sheet_state(archive_images=False)
    queue: dict[str, dict[str, Any]] = st.session_state["sheet_queue"]
    return queue[sheet_id].get("state", {})


def apply_background_ocr_result(result: OCRJobResult) -> None:
    """Attach one completed worker result to its exact queue sheet."""

    queue: dict[str, dict[str, Any]] = st.session_state["sheet_queue"]
    entry = queue.get(result.sheet_id)
    if entry is None:
        return

    active = st.session_state.get("active_sheet_id") == result.sheet_id
    prepared_sheet: PreparedMonitoringSheet | ArchivedPreparedMonitoringSheet
    if active:
        prepared_sheet = restore_prepared_sheet(result.prepared_sheet)
        target_state = st.session_state
    else:
        prepared_sheet = result.prepared_sheet
        target_state = entry.setdefault("state", {})

    target_state["prepared_sheet"] = prepared_sheet
    target_state["cell_results"] = result.cell_results
    target_state["monitoring_rows"] = result.monitoring_rows
    target_state["processing_metrics"] = result.processing_metrics
    target_state["processing_seconds"] = (
        result.processing_metrics.total_sheet_processing_seconds
    )
    target_state["processed_fingerprint"] = result.sheet_id
    target_state["processed_geometry_mode"] = result.geometry_mode
    target_state["day_verification_state"] = DayVerificationState(
        sheet_fingerprint=result.sheet_id
    )
    target_state["day_editor_version"] = 0
    target_state["table_editor_version"] = 0
    target_state.pop("cell_crop_data_urls", None)
    st.session_state[f"selected_day_{result.sheet_id}"] = 1
    entry["ocr_job_id"] = None


def collect_completed_background_jobs(
    manager: BackgroundOCRManager,
    owner_id: str,
) -> None:
    """Collect successful worker results on the Streamlit main thread."""

    queue: dict[str, dict[str, Any]] = st.session_state["sheet_queue"]
    for entry in queue.values():
        job_id = entry.get("ocr_job_id")
        if not job_id:
            continue
        snapshot = manager.snapshot(job_id, owner_id)
        if snapshot is None:
            entry["ocr_job_id"] = None
            continue
        if snapshot.state != "succeeded":
            continue
        result = manager.consume_result(job_id, owner_id)
        if result is not None:
            apply_background_ocr_result(result)


def queue_stage(
    sheet_id: str,
    manager: BackgroundOCRManager,
    owner_id: str,
) -> str:
    """Summarize one sheet without running geometry preparation or OCR."""

    queue: dict[str, dict[str, Any]] = st.session_state["sheet_queue"]
    job_id = queue[sheet_id].get("ocr_job_id")
    if job_id:
        snapshot = manager.snapshot(job_id, owner_id)
        if snapshot is not None:
            if snapshot.state == "queued":
                return "Waiting for OCR"
            if snapshot.state == "loading_model":
                return "Loading OCR model"
            if snapshot.state == "running":
                return (
                    f"OCR running · {snapshot.processed_count}/"
                    f"{snapshot.total_count}"
                )
            if snapshot.state == "succeeded":
                return "OCR finished"
            if snapshot.state == "failed":
                return "OCR failed · Retry"

    state = queue_item_state(sheet_id)
    results = state.get("cell_results")
    if results:
        day_state = state_for_sheet(
            sheet_id,
            state.get("day_verification_state"),
        )
        readiness = summarize_export_readiness(results, day_state)
        if readiness.ready:
            return "Export-ready"
        return f"Verification · {len(readiness.confirmed_days)}/31 days"
    if (
        state.get("fixed_preflight_sheet") is not None
        and state.get("calibrated_preflight_sheet") is not None
    ):
        return "Geometry ready"
    return "Awaiting geometry"


def select_adjacent_queue_sheet(direction: int) -> None:
    """Move the selected queue item without starting processing."""

    order: list[str] = st.session_state.get("sheet_queue_order", [])
    current = st.session_state.get("selected_sheet_id")
    if not order or current not in order:
        return
    current_index = order.index(current)
    next_index = min(max(current_index + direction, 0), len(order) - 1)
    st.session_state["selected_sheet_id"] = order[next_index]


def lock_orientation_for_preflight(uploaded_fingerprint: str) -> None:
    """Freeze the pixel orientation before geometry preparation starts."""

    orientation_key = f"orientation_choice_{uploaded_fingerprint}"
    st.session_state["prepared_orientation_choice"] = st.session_state.get(
        orientation_key,
        "auto",
    )
    st.session_state["orientation_preflight_locked"] = True


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
    *,
    changed_filenames: tuple[str, ...] = (),
    day_state: DayVerificationState | None = None,
) -> None:
    """Store the canonical result graph after a user update."""

    current_state = state_for_sheet(
        uploaded_fingerprint,
        day_state or st.session_state.get("day_verification_state"),
    )
    if day_state is None and changed_filenames:
        current_state = invalidate_days_for_changes(
            current_state,
            results,
            changed_filenames,
        )
    st.session_state["cell_results"] = results
    st.session_state["monitoring_rows"] = build_monitoring_rows(results)
    st.session_state["day_verification_state"] = current_state
    st.session_state["day_editor_version"] = (
        st.session_state.get("day_editor_version", 0) + 1
    )
    st.session_state["table_editor_version"] = (
        st.session_state.get("table_editor_version", 0) + 1
    )


def review_action_key(uploaded_fingerprint: str, filename: str) -> str:
    return f"review_action_{uploaded_fingerprint}_{filename}"


def review_correction_key(uploaded_fingerprint: str, filename: str) -> str:
    return f"review_correction_{uploaded_fingerprint}_{filename}"


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
        store_verified_results(
            outcome.results,
            uploaded_fingerprint,
            changed_filenames=outcome.changed_filenames,
        )
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
        store_verified_results(
            outcome.results,
            uploaded_fingerprint,
            changed_filenames=outcome.changed_filenames,
        )
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


def day_value_key(
    uploaded_fingerprint: str,
    day: int,
    filename: str,
    editor_version: int,
) -> str:
    return (
        f"day_value_{uploaded_fingerprint}_{day}_{filename}_{editor_version}"
    )


def day_blank_key(
    uploaded_fingerprint: str,
    day: int,
    filename: str,
    editor_version: int,
) -> str:
    return (
        f"day_blank_{uploaded_fingerprint}_{day}_{filename}_{editor_version}"
    )


def submit_day_callback(
    uploaded_fingerprint: str,
    day: int,
    editor_version: int,
    confirm_day: bool,
    advance_after_confirmation: bool,
) -> None:
    """Save one day before the form's single normal Streamlit rerun."""

    current_results = st.session_state["cell_results"]
    controls = {
        result.filename: {
            "value": st.session_state.get(
                day_value_key(
                    uploaded_fingerprint,
                    day,
                    result.filename,
                    editor_version,
                ),
                result.final_value,
            ),
            "is_blank": st.session_state.get(
                day_blank_key(
                    uploaded_fingerprint,
                    day,
                    result.filename,
                    editor_version,
                ),
                result.is_blank,
            ),
        }
        for result in results_for_day(current_results, day)
    }
    outcome = apply_day_submission(
        current_results,
        state_for_sheet(
            uploaded_fingerprint,
            st.session_state.get("day_verification_state"),
        ),
        day,
        controls,
        confirm_day=confirm_day,
        advance_after_confirmation=advance_after_confirmation,
    )
    store_verified_results(
        outcome.results,
        uploaded_fingerprint,
        day_state=outcome.state,
    )
    st.session_state[f"selected_day_{uploaded_fingerprint}"] = outcome.next_day

    if outcome.errors:
        level = "warning" if outcome.changed_filenames else "error"
        message = (
            f"Saved {len(outcome.changed_filenames)} valid reading change(s), "
            f"but Day {day} was not confirmed."
        )
    elif outcome.day_confirmed:
        level = "success"
        message = f"Day {day} was saved and explicitly confirmed."
    elif outcome.changed_filenames:
        level = "success"
        message = (
            f"Saved {len(outcome.changed_filenames)} reading change(s) for Day "
            f"{day}. The day still requires confirmation."
        )
    elif not confirm_day:
        level = "success"
        message = f"Day {day} was saved. No values changed."
    else:
        level = "warning"
        message = (
            f"Day {day} was not confirmed because unresolved readings remain."
        )
    st.session_state["day_flash"] = {
        "level": level,
        "message": message,
        "errors": outcome.errors,
    }
    st.session_state["day_scroll_request"] = build_day_scroll_request(
        uploaded_fingerprint,
        outcome.next_day,
    )


def previous_day_callback(uploaded_fingerprint: str) -> None:
    """Navigate without touching OCR results or confirmation state."""

    key = f"selected_day_{uploaded_fingerprint}"
    st.session_state[key] = previous_day(int(st.session_state.get(key, 1)))
    st.session_state.pop("day_scroll_request", None)


st.title("OCR-assisted monitoring sheet encoding")
st.write(
    "OCR prefills the company monitoring sheet. Verify each day beside the "
    "exact extracted handwriting crops before exporting the workbook."
)

background_ocr_manager = load_background_ocr_manager()
background_ocr_owner_id = st.session_state.setdefault(
    "background_ocr_owner_id",
    uuid.uuid4().hex,
)

uploaded_files = st.file_uploader(
    "Upload monitoring sheets",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True,
    help=(
        "Upload one or more PNG, JPG, JPEG, or PDF scans. Every PDF page "
        "becomes a separate queue item. Avoid glare, blur, shadows, or "
        "cropped borders."
    ),
)

sheet_queue: dict[str, dict[str, Any]] = st.session_state.setdefault(
    "sheet_queue",
    {},
)
sheet_queue_order: list[str] = st.session_state.setdefault(
    "sheet_queue_order",
    [],
)
queue_upload_errors: dict[str, str] = st.session_state.setdefault(
    "queue_upload_errors",
    {},
)
known_source_fingerprints = {
    entry["item"].source_fingerprint for entry in sheet_queue.values()
}

for uploaded_file in uploaded_files or ():
    uploaded_bytes = uploaded_file.getvalue()
    source_fingerprint = create_file_fingerprint(uploaded_bytes)
    if (
        source_fingerprint in known_source_fingerprints
        or source_fingerprint in queue_upload_errors
    ):
        continue
    try:
        with st.spinner(f"Adding {uploaded_file.name} to the sheet queue..."):
            new_items = build_sheet_queue_items(
                uploaded_bytes,
                uploaded_file.name,
            )
    except ValueError as error:
        queue_upload_errors[source_fingerprint] = (
            f"{uploaded_file.name}: {error}"
        )
        continue

    for item in new_items:
        if item.sheet_id in sheet_queue:
            continue
        sheet_queue[item.sheet_id] = {
            "item": item,
            "state": {},
            "ocr_job_id": None,
        }
        sheet_queue_order.append(item.sheet_id)
    known_source_fingerprints.add(source_fingerprint)

for error_message in queue_upload_errors.values():
    st.error(error_message)

if not sheet_queue_order:
    st.info("Upload one or more PNG, JPG, JPEG, or PDF scans to begin.")
    st.stop()

collect_completed_background_jobs(
    background_ocr_manager,
    background_ocr_owner_id,
)

selected_sheet_id = st.session_state.get("selected_sheet_id")
if selected_sheet_id not in sheet_queue_order:
    st.session_state["selected_sheet_id"] = sheet_queue_order[0]

st.subheader("Sheet queue")
st.caption(
    "Inspect and process one selected sheet at a time. Switching sheets "
    "preserves its geometry, OCR proposals, corrections, and confirmations; "
    "navigation never starts OCR."
)
selected_sheet_id = st.selectbox(
    "Selected sheet",
    options=sheet_queue_order,
    format_func=lambda sheet_id: sheet_queue[sheet_id]["item"].display_name,
    key="selected_sheet_id",
)

with st.spinner("Switching the active sheet..."):
    activate_queue_sheet(selected_sheet_id)

queue_rows = [
    {
        "Sheet": sheet_queue[sheet_id]["item"].display_name,
        "Stage": queue_stage(
            sheet_id,
            background_ocr_manager,
            background_ocr_owner_id,
        ),
    }
    for sheet_id in sheet_queue_order
]
st.dataframe(
    queue_rows,
    hide_index=True,
    width="stretch",
    column_config={
        "Sheet": st.column_config.TextColumn("Sheet", width="large"),
        "Stage": st.column_config.TextColumn("Stage", width="medium"),
    },
)
render_background_ocr_monitor(background_ocr_owner_id)
selected_index = sheet_queue_order.index(selected_sheet_id)
queue_navigation = st.columns(2)
queue_navigation[0].button(
    "Previous sheet",
    icon=":material/arrow_back:",
    disabled=selected_index == 0,
    on_click=select_adjacent_queue_sheet,
    args=(-1,),
)
queue_navigation[1].button(
    "Next sheet",
    icon=":material/arrow_forward:",
    disabled=selected_index == len(sheet_queue_order) - 1,
    on_click=select_adjacent_queue_sheet,
    args=(1,),
)

active_queue_item: SheetQueueItem = sheet_queue[selected_sheet_id]["item"]
uploaded_fingerprint = active_queue_item.sheet_id
source_display_name = active_queue_item.display_name
upload_decoding_seconds = active_queue_item.decoding_seconds
orientation_choice = "auto"
source_was_portrait = (
    active_queue_item.source_height > active_queue_item.source_width
)
if source_was_portrait:
    orientation_key = f"orientation_choice_{uploaded_fingerprint}"
    st.session_state.setdefault(orientation_key, "auto")
    auto_rotation_labels = {
        "clockwise": "Auto (rotate right)",
        "counterclockwise": "Auto (rotate left)",
        "none": "Auto (needs review)",
    }
    geometry_already_prepared = (
        st.session_state.get("fixed_preflight_sheet") is not None
        or st.session_state.get("calibrated_preflight_sheet") is not None
    )
    if geometry_already_prepared:
        st.session_state[orientation_key] = st.session_state.get(
            "prepared_orientation_choice",
            "auto",
        )
    orientation_is_locked = (
        geometry_already_prepared
        or st.session_state.get("orientation_preflight_locked", False)
    )
    orientation_choice = st.segmented_control(
        "Page orientation",
        options=["auto", "counterclockwise", "clockwise", "none"],
        format_func=lambda choice: {
            "auto": auto_rotation_labels[active_queue_item.auto_rotation],
            "counterclockwise": "Rotate left",
            "clockwise": "Rotate right",
            "none": "Keep portrait",
        }[choice],
        key=orientation_key,
        required=True,
        disabled=orientation_is_locked,
    )
    if orientation_is_locked:
        st.caption(
            "Orientation is locked after extraction geometry is prepared so "
            "saved crops and verification remain associated with the same "
            "pixels."
        )

effective_rotation = resolve_queue_rotation(
    active_queue_item,
    orientation_choice,
)
try:
    uploaded_image = decode_queue_image(
        active_queue_item,
        orientation_choice,
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

if active_queue_item.source_kind == "pdf":
    if active_queue_item.image_source == "embedded_scan":
        st.caption(
            f"Using the original full-page scanner image from PDF page "
            f"{active_queue_item.page_number} of "
            f"{active_queue_item.page_count} at approximately "
            f"{active_queue_item.effective_dpi:.0f} DPI."
        )
    else:
        st.caption(
            f"Rendered PDF page {active_queue_item.page_number} of "
            f"{active_queue_item.page_count} at "
            f"{active_queue_item.render_dpi} DPI for the image-based "
            "extraction pipeline."
        )

if effective_rotation != "none":
    direction = "right" if effective_rotation == "clockwise" else "left"
    if orientation_choice == "auto":
        st.info(
            f"Automatically rotated this portrait scan 90° {direction} using "
            "the monitoring-sheet layout. Inspect the document before "
            "preparing extraction."
        )
    else:
        st.info(f"Applied the selected 90° rotation to the {direction}.")
elif source_was_portrait and not active_queue_item.orientation_confident:
    st.warning(
        "Automatic orientation was inconclusive. Inspect the document and "
        "choose Rotate left or Rotate right before preparing extraction."
    )

uploaded_document_expander = st.expander(
    "Uploaded document",
    expanded=False,
    key=f"uploaded_document_{uploaded_fingerprint}",
    on_change="rerun",
)
if uploaded_document_expander.open:
    with uploaded_document_expander:
        st.image(
            convert_bgr_to_rgb(uploaded_image),
            caption=source_display_name,
            width="stretch",
        )

st.header("1. Extraction preflight")
processed_result_available = (
    st.session_state.get("processed_fingerprint") == uploaded_fingerprint
    and st.session_state.get("cell_results") is not None
)
show_extraction_diagnostics = True
if processed_result_available:
    st.success(
        "Extraction and OCR are complete. Continue directly to verification."
    )
    show_extraction_diagnostics = st.toggle(
        "Show extraction diagnostics or OCR replacement controls",
        value=False,
        key=f"show_extraction_diagnostics_{uploaded_fingerprint}",
        help=(
            "Open this only when you need to inspect the grid overlays, compare "
            "representative crops, or replace this sheet's OCR results."
        ),
    )
else:
    st.write(
        "Prepare both extraction previews first. This geometry-only step does "
        "not load the OCR model."
    )

fixed_preflight_sheet: (
    PreparedMonitoringSheet | ArchivedPreparedMonitoringSheet | None
) = st.session_state.get("fixed_preflight_sheet")
calibrated_preflight_sheet: (
    PreparedMonitoringSheet | ArchivedPreparedMonitoringSheet | None
) = (
    st.session_state.get("calibrated_preflight_sheet")
)
if show_extraction_diagnostics:
    if isinstance(fixed_preflight_sheet, ArchivedPreparedMonitoringSheet):
        fixed_preflight_sheet = restore_prepared_sheet(fixed_preflight_sheet)
        st.session_state["fixed_preflight_sheet"] = fixed_preflight_sheet
    if isinstance(calibrated_preflight_sheet, ArchivedPreparedMonitoringSheet):
        calibrated_preflight_sheet = restore_prepared_sheet(
            calibrated_preflight_sheet
        )
        st.session_state["calibrated_preflight_sheet"] = (
            calibrated_preflight_sheet
        )

if show_extraction_diagnostics and (
    fixed_preflight_sheet is None or calibrated_preflight_sheet is None
):
    if st.button(
        "Prepare extraction preview",
        type="primary",
        icon=":material/grid_view:",
        on_click=lock_orientation_for_preflight,
        args=(uploaded_fingerprint,),
    ):
        image_height, image_width = uploaded_image.shape[:2]
        preflight_metrics = ProcessingMetrics(
            source_filename=source_display_name,
            uploaded_fingerprint=uploaded_fingerprint,
            extraction_geometry_mode="fixed",
            uploaded_width=image_width,
            uploaded_height=image_height,
            upload_decoding_seconds=round(upload_decoding_seconds, 6),
        )
        preflight_metrics.capture_process_uptime()
        try:
            with st.spinner("Preparing fixed and locally calibrated previews..."):
                fixed_preflight_sheet = prepare_monitoring_sheet(
                    uploaded_image,
                    metrics=preflight_metrics,
                    geometry_mode="fixed",
                )
                calibration_start = time.perf_counter()
                calibrated_preflight_sheet = prepare_calibrated_monitoring_sheet(
                    fixed_preflight_sheet
                )
                preflight_metrics.add_seconds(
                    "measurement_cell_extraction_seconds",
                    time.perf_counter() - calibration_start,
                )
                alignment_report = calculate_alignment_report(
                    fixed_preflight_sheet
                )
                alignment_summary = build_alignment_preflight_summary(
                    fixed_preflight_sheet,
                    calibrated_preflight_sheet,
                    alignment_report,
                )
            st.session_state["fixed_preflight_sheet"] = fixed_preflight_sheet
            st.session_state["calibrated_preflight_sheet"] = (
                calibrated_preflight_sheet
            )
            st.session_state["alignment_report"] = alignment_report
            st.session_state["alignment_preflight_summary"] = alignment_summary
            st.session_state["preflight_processing_metrics"] = preflight_metrics
            st.session_state["prepared_orientation_choice"] = (
                orientation_choice
            )
        except ValueError as error:
            st.session_state.pop("orientation_preflight_locked", None)
            st.session_state.pop("prepared_orientation_choice", None)
            st.error(str(error))
        except Exception as error:
            st.session_state.pop("orientation_preflight_locked", None)
            st.session_state.pop("prepared_orientation_choice", None)
            st.error("The extraction preview could not be prepared.")
            with st.expander("Technical error details"):
                st.exception(error)

if (
    show_extraction_diagnostics
    and isinstance(fixed_preflight_sheet, PreparedMonitoringSheet)
    and isinstance(calibrated_preflight_sheet, PreparedMonitoringSheet)
):
    alignment_report = st.session_state.get("alignment_report")
    if alignment_report is not None:
        # Refresh derived UI data after a hot reload so sessions created by an
        # older summary schema do not retain missing keys.
        alignment_summary = build_alignment_preflight_summary(
            fixed_preflight_sheet,
            calibrated_preflight_sheet,
            alignment_report,
        )
        st.session_state["alignment_preflight_summary"] = alignment_summary
    else:
        alignment_summary = st.session_state.get(
            "alignment_preflight_summary",
            {},
        )
    st.caption(
        "Both previews use the same detected Day 1-31 row span when the "
        "printed-line evidence is strong. Fixed keeps straight, evenly spaced "
        "rows; local calibration follows small line bends and column drift."
    )
    overlay_columns = st.columns(2)
    with overlay_columns[0]:
        st.subheader("Fixed-grid overlay")
        st.image(
            convert_bgr_to_rgb(fixed_preflight_sheet.measurement_grid_overlay),
            width="stretch",
        )
    with overlay_columns[1]:
        st.subheader("Locally calibrated-grid overlay")
        st.image(
            convert_bgr_to_rgb(
                calibrated_preflight_sheet.measurement_grid_overlay
            ),
            width="stretch",
        )

    alignment_metrics = st.columns(4)
    alignment_metrics[0].metric(
        "Alignment score",
        alignment_summary["provisional_alignment_score"],
        help="Diagnostic and uncalibrated; inspect the overlays and crops.",
    )
    alignment_metrics[1].metric(
        "Horizontal lines",
        f"{alignment_summary['matched_horizontal_lines']}/"
        f"{alignment_summary['expected_horizontal_lines']}",
    )
    alignment_metrics[2].metric(
        "Vertical lines",
        f"{alignment_summary['matched_vertical_lines']}/"
        f"{alignment_summary['expected_vertical_lines']}",
    )
    alignment_metrics[3].metric(
        "Fallback boundaries",
        alignment_summary["fallback_boundary_count"],
    )
    for warning in alignment_summary.get("warnings", ()):
        st.warning(warning)
    for notice in alignment_summary.get("notices", ()):
        st.info(notice)
    st.caption(alignment_summary.get("notice", ""))

    st.subheader("Representative top, middle, and bottom crops")
    st.caption(
        "Fixed and calibrated crops use the same stable filename. Compare "
        "handwriting placement and neighboring border contamination."
    )
    for crop_pair in representative_crop_pairs(
        fixed_preflight_sheet,
        calibrated_preflight_sheet,
    ):
        label_column, fixed_column, calibrated_column = st.columns([1, 2, 2])
        label_column.markdown(
            f"**{crop_pair['position']}**  \nDay {crop_pair['day']}, Point 4, "
            f"{crop_pair['reading_type'].title()}"
        )
        fixed_column.image(
            convert_bgr_to_rgb(crop_pair["fixed_image"]),
            caption=f"Fixed · {crop_pair['filename']}",
            width="stretch",
        )
        calibrated_column.image(
            convert_bgr_to_rgb(crop_pair["calibrated_image"]),
            caption=f"Calibrated · {crop_pair['filename']}",
            width="stretch",
        )

    recommended_geometry = recommended_geometry_mode(alignment_summary)
    preflight_job_id = sheet_queue[uploaded_fingerprint].get("ocr_job_id")
    preflight_job_snapshot = (
        background_ocr_manager.snapshot(
            preflight_job_id,
            background_ocr_owner_id,
        )
        if preflight_job_id
        else None
    )
    geometry_is_locked_for_job = (
        preflight_job_snapshot is not None
        and preflight_job_snapshot.state
        in ("queued", "loading_model", "running", "succeeded")
    )
    geometry_choice = st.segmented_control(
        "Extraction geometry",
        options=["fixed", "calibrated"],
        default=recommended_geometry,
        required=True,
        format_func=lambda mode: (
            "Fixed extraction" if mode == "fixed" else "Locally calibrated extraction"
        ),
        key=f"geometry_choice_{uploaded_fingerprint}",
        persist_state="session",
        disabled=geometry_is_locked_for_job,
    )
    if geometry_is_locked_for_job:
        st.caption(
            "Extraction geometry is locked to the version submitted to the "
            "background OCR worker."
        )
    if recommended_geometry == "calibrated":
        st.caption(
            "Locally calibrated extraction is recommended because the complete "
            "printed row sequence and all 496 calibrated cells passed geometry "
            "checks. Fixed extraction remains available as a recovery option."
        )
    else:
        st.warning(
            "Fixed extraction is recommended because calibrated geometry did "
            "not pass every default-selection guard. Inspect both overlays."
        )

    current_result_matches_choice = (
        st.session_state.get("processed_fingerprint") == uploaded_fingerprint
        and st.session_state.get("processed_geometry_mode") == geometry_choice
    )
    current_job_id = preflight_job_id
    current_job_snapshot = preflight_job_snapshot
    current_job_pending = (
        current_job_snapshot is not None
        and current_job_snapshot.state
        in ("queued", "loading_model", "running", "succeeded")
    )
    retrying_failed_job = (
        current_job_snapshot is not None
        and current_job_snapshot.state == "failed"
    )
    run_ocr = st.button(
        (
            f"Retry OCR with {geometry_choice} extraction"
            if retrying_failed_job
            else f"Queue OCR with {geometry_choice} extraction"
        ),
        type="primary",
        icon=":material/document_scanner:",
        disabled=current_result_matches_choice or current_job_pending,
    )
    st.caption(
        "One background worker processes explicitly queued sheets in order. "
        "You may switch sheets and continue verification while OCR runs. "
        "The worker uses a grayscale first pass and the existing adaptive "
        "three-variant fallback."
    )
    if current_result_matches_choice:
        st.info(
            f"The current results already use {geometry_choice} extraction. "
            "Preview navigation and geometry selection do not rerun OCR."
        )
    elif current_job_snapshot is not None:
        if current_job_snapshot.state == "queued":
            st.info("This sheet is waiting for the background OCR worker.")
        elif current_job_snapshot.state == "loading_model":
            st.info("The background worker is loading the OCR model.")
        elif current_job_snapshot.state == "running":
            st.info(
                f"This sheet is processing in the background: "
                f"{current_job_snapshot.processed_count}/"
                f"{current_job_snapshot.total_count} cells."
            )
        elif current_job_snapshot.state == "succeeded":
            st.success(
                "Background OCR finished. The results will load on the next "
                "sheet switch or saved action."
            )
        elif current_job_snapshot.state == "failed":
            st.error(
                "Background OCR failed. You may retry this sheet without "
                "affecting the other queue items."
            )
            with st.expander("Technical error details"):
                st.code(current_job_snapshot.error_message)

    if run_ocr:
        selected_sheet = (
            fixed_preflight_sheet
            if geometry_choice == "fixed"
            else calibrated_preflight_sheet
        )
        processing_metrics = replace(
            st.session_state["preflight_processing_metrics"],
            extraction_geometry_mode=geometry_choice,
        )
        try:
            if retrying_failed_job and current_job_id:
                background_ocr_manager.discard_terminal_job(
                    current_job_id,
                    background_ocr_owner_id,
                )
            with st.spinner("Adding this sheet to the background OCR queue..."):
                archived_sheet = archive_prepared_sheet(selected_sheet)
                request = OCRJobRequest(
                    owner_id=background_ocr_owner_id,
                    sheet_id=uploaded_fingerprint,
                    source_filename=source_display_name,
                    geometry_mode=geometry_choice,
                    prepared_sheet=archived_sheet,
                    processing_metrics=processing_metrics,
                    cells_per_batch=CELLS_PER_PROCESSING_BATCH,
                    recognition_strategy="adaptive",
                )
                job_id = background_ocr_manager.submit(request)
                sheet_queue[uploaded_fingerprint]["ocr_job_id"] = job_id
            st.rerun()
        except ValueError as error:
            st.error(str(error))
        except Exception as error:
            st.error(
                "The sheet could not be added to the background OCR queue."
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
active_job_id = sheet_queue[uploaded_fingerprint].get("ocr_job_id")
active_job_snapshot = (
    background_ocr_manager.snapshot(
        active_job_id,
        background_ocr_owner_id,
    )
    if active_job_id
    else None
)
active_job_blocks_saved_results = (
    active_job_snapshot is not None
    and active_job_snapshot.state
    in ("queued", "loading_model", "running", "succeeded")
)
if active_job_blocks_saved_results and cell_results is not None:
    st.warning(
        "This sheet has replacement OCR in progress. Its previous verification "
        "workspace is temporarily read-only; switch to another sheet until "
        "the queued result is collected."
    )

if (
    prepared_sheet is not None
    and cell_results is not None
    and monitoring_rows is not None
    and not active_job_blocks_saved_results
):
    crop_data_urls = st.session_state.get("cell_crop_data_urls")
    if crop_data_urls is None:
        thumbnail_start = time.perf_counter()
        crop_data_urls = build_cell_crop_data_url_lookup(prepared_sheet)
        st.session_state["cell_crop_data_urls"] = crop_data_urls
        processing_metrics = st.session_state.get("processing_metrics")
        if processing_metrics is not None:
            thumbnail_seconds = time.perf_counter() - thumbnail_start
            processing_metrics.add_seconds(
                "ui_thumbnail_preparation_seconds",
                thumbnail_seconds,
            )
            processing_metrics.recalculate_total()
            st.session_state["processing_seconds"] = (
                processing_metrics.total_sheet_processing_seconds
            )

    st.divider()
    st.header("Processing results")

    blank_count = count_blank_cells(cell_results)
    day_state = state_for_sheet(
        uploaded_fingerprint,
        st.session_state.get("day_verification_state"),
    )
    st.session_state["day_verification_state"] = day_state
    readiness = summarize_export_readiness(cell_results, day_state)
    metrics = st.columns(5)
    metrics[0].metric("Blocking cells", readiness.blocking_cell_count)
    metrics[1].metric("Attention items", readiness.attention_item_count)
    metrics[2].metric(
        "Operational warnings", readiness.operational_warning_count
    )
    metrics[3].metric(
        "Confirmed days", f"{len(readiness.confirmed_days)} of 31"
    )
    metrics[4].metric("Unconfirmed days", len(readiness.unconfirmed_days))
    st.caption(
        f"Extraction used: **{prepared_sheet.geometry_mode.title()}** · "
        f"Blank readings: {blank_count} · OCR processing time: "
        f"{st.session_state['processing_seconds']:.1f} s"
    )
    processing_metrics = st.session_state.get("processing_metrics")
    if processing_metrics is not None:
        avoided_inputs = max(
            processing_metrics.filled_cell_count * 3
            - processing_metrics.ocr_input_image_count,
            0,
        )
        input_note = (
            f"Adaptive first-pass reuse avoided {avoided_inputs} inputs."
            if getattr(
                processing_metrics,
                "recognition_strategy",
                "consensus",
            )
            == "adaptive"
            else "This saved result used the full consensus strategy."
        )
        st.caption(
            f"OCR prediction used {processing_metrics.ocr_prediction_seconds:.1f} "
            f"of {processing_metrics.total_sheet_processing_seconds:.1f} seconds "
            f"across {processing_metrics.ocr_input_image_count} model inputs. "
            f"{input_note}"
        )
        with st.expander("Advanced processing diagnostics (troubleshooting)"):
            st.json(processing_metrics.to_dict())
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

    day_tab, table_tab, preview_tab, export_tab = st.tabs(
        [
            "Day Verification",
            "Full Monitoring Table",
            "Sheet Previews",
            "Export Excel",
        ],
        key=f"results_tab_{uploaded_fingerprint}",
        on_change="rerun",
    )

    with day_tab:
        st.subheader("Day Verification")
        st.write(
            "Verify one complete day beside its 16 exact filename-keyed crops. "
            "Corrections save through the same canonical patch engine used by "
            "the table and detailed review."
        )
        st.caption(
            "Session-only work: closing this browser session or stopping "
            "Streamlit clears unfinished verification."
        )
        selected_day_key = f"selected_day_{uploaded_fingerprint}"
        st.session_state.setdefault(selected_day_key, 1)
        selected_day = int(st.session_state[selected_day_key])
        day_flash = st.session_state.pop("day_flash", None)
        if day_flash is not None:
            getattr(st, day_flash["level"])(day_flash["message"])
            for error_message in day_flash["errors"]:
                st.caption(error_message)
        should_scroll, remaining_scroll_request = consume_day_scroll_request(
            st.session_state.get("day_scroll_request"),
            uploaded_fingerprint,
            selected_day,
        )
        if remaining_scroll_request is None:
            st.session_state.pop("day_scroll_request", None)
        else:
            st.session_state["day_scroll_request"] = remaining_scroll_request
        if should_scroll:
            _SCROLL_TO_DAY_BANNER(
                key=(
                    f"day_scroll_{uploaded_fingerprint}_{selected_day}_"
                    f"{st.session_state.get('day_editor_version', 0)}"
                ),
                data={
                    "message": (
                        day_flash["message"] if day_flash is not None else ""
                    )
                },
                height=1,
            )

        st.progress(
            len(readiness.confirmed_days) / 31,
            text=f"{len(readiness.confirmed_days)} of 31 days confirmed",
        )
        with st.container(horizontal=True, vertical_alignment="bottom"):
            selected_day = st.selectbox(
                "Day",
                options=list(range(1, 32)),
                key=selected_day_key,
                persist_state="session",
                width=180,
            )
            if selected_day in day_state.confirmed_days:
                st.badge(
                    "Confirmed",
                    color="green",
                    icon=":material/check_circle:",
                )
            else:
                st.badge(
                    "Unconfirmed",
                    color="orange",
                    icon=":material/pending:",
                )

        current_day_results = results_for_day(cell_results, selected_day)
        result_lookup = {
            (result.point, result.reading_type): result
            for result in current_day_results
        }
        cell_images = build_cell_image_lookup(prepared_sheet)
        editor_version = st.session_state.get("day_editor_version", 0)
        day_form_key = (
            f"day_verification_form_{uploaded_fingerprint}_"
            f"{selected_day}_{editor_version}"
        )
        badge_colors = {
            "Blocking error": "red",
            "Blocking format error": "red",
            "Needs attention": "orange",
            "Operational warning": "yellow",
            "Confirmed": "green",
            "Blank": "gray",
            "Decimal inferred": "blue",
            "Likely blank": "violet",
            "Manually corrected": "blue",
        }

        with st.form(day_form_key, enter_to_submit=True):
            header = st.columns([0.5, 1.3, 1.2, 1.3, 1.2, 1.8])
            for column, label in zip(
                header,
                (
                    "Point",
                    "Temperature crop",
                    "Temperature value",
                    "Humidity crop",
                    "Humidity value",
                    "Status",
                ),
            ):
                column.markdown(f"**{label}**")

            for point in range(1, 9):
                temperature = result_lookup[(point, "temperature")]
                humidity = result_lookup[(point, "humidity")]
                row_columns = st.columns([0.5, 1.3, 1.2, 1.3, 1.2, 1.8])
                row_columns[0].markdown(f"**{point}**")

                for crop_column, result, short_label in (
                    (row_columns[1], temperature, "T"),
                    (row_columns[3], humidity, "H"),
                ):
                    crop_column.image(
                        convert_bgr_to_rgb(cell_images[result.filename]),
                        width="stretch",
                    )
                    with crop_column.popover(
                        f"Enlarge {short_label}",
                        icon=":material/zoom_in:",
                        key=(
                            f"day_crop_{uploaded_fingerprint}_{selected_day}_"
                            f"{result.filename}"
                        ),
                    ):
                        st.image(
                            convert_bgr_to_rgb(cell_images[result.filename]),
                            caption=result.filename,
                            width="stretch",
                        )

                for value_column, result in (
                    (row_columns[2], temperature),
                    (row_columns[4], humidity),
                ):
                    value_column.text_input(
                        (
                            f"Day {selected_day}, Point {point}, "
                            f"{result.reading_type} value"
                        ),
                        value=result.final_value,
                        key=day_value_key(
                            uploaded_fingerprint,
                            selected_day,
                            result.filename,
                            editor_version,
                        ),
                        label_visibility="collapsed",
                        placeholder="Example: 22.0",
                    )
                    value_column.checkbox(
                        f"{result.reading_type.title()} blank",
                        value=result.is_blank,
                        key=day_blank_key(
                            uploaded_fingerprint,
                            selected_day,
                            result.filename,
                            editor_version,
                        ),
                    )

                status_column = row_columns[5]
                status_reasons: list[str] = []
                for result, short_label in (
                    (temperature, "T"),
                    (humidity, "H"),
                ):
                    geometry_warning = (
                        geometry_warning_for_filename(
                            prepared_sheet.cells,
                            result.filename,
                        )
                        if prepared_sheet.geometry_mode == "calibrated"
                        else ""
                    )
                    status = cell_display_status(
                        result,
                        day_confirmed=selected_day in day_state.confirmed_days,
                        geometry_warning=geometry_warning,
                    )
                    status_column.badge(
                        f"{short_label}: {status.label}",
                        color=badge_colors[status.label],
                    )
                    if status.reason:
                        status_reasons.append(f"{short_label}: {status.reason}")
                status_column.caption(" ".join(dict.fromkeys(status_reasons)))

            action_columns = st.columns(
                [1.2, 4.0, 1.2, 1.9],
                vertical_alignment="center",
            )
            action_columns[3].form_submit_button(
                "Confirm Day and Next",
                type="primary",
                icon=":material/check_circle:",
                on_click=submit_day_callback,
                args=(
                    uploaded_fingerprint,
                    selected_day,
                    editor_version,
                    True,
                    True,
                ),
                key=f"confirm_day_{day_form_key}",
                width="stretch",
            )
            action_columns[2].form_submit_button(
                "Save Day",
                icon=":material/save:",
                on_click=submit_day_callback,
                args=(
                    uploaded_fingerprint,
                    selected_day,
                    editor_version,
                    False,
                    False,
                ),
                key=f"save_day_{day_form_key}",
                width="stretch",
            )
            action_columns[0].form_submit_button(
                "Previous Day",
                icon=":material/arrow_back:",
                on_click=previous_day_callback,
                args=(uploaded_fingerprint,),
                key=f"previous_day_{day_form_key}",
                width="stretch",
            )

        advanced_review_container = st.expander(
            "Advanced OCR details for troubleshooting",
            expanded=False,
        )

    if preview_tab.open:
        with preview_tab:
            st.subheader("Detected monitoring table")
            st.image(
                convert_bgr_to_rgb(prepared_sheet.detection_preview),
                width="stretch",
            )
            st.subheader("Straightened monitoring table")
            st.image(
                convert_bgr_to_rgb(prepared_sheet.warped_table),
                width="stretch",
            )
            grid_overlay_expander = st.expander(
                "Show measurement-grid overlay",
                key=f"result_grid_overlay_{uploaded_fingerprint}",
                on_change="rerun",
            )
            if grid_overlay_expander.open:
                with grid_overlay_expander:
                    st.image(
                        convert_bgr_to_rgb(
                            prepared_sheet.measurement_grid_overlay
                        ),
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

    with advanced_review_container:
        review_results = [
            result
            for result in cell_results
            if result.blocks_export or result.review_categories
        ]
        review_flash = st.session_state.pop("review_flash", None)
        if review_flash is not None:
            getattr(st, review_flash["level"])(review_flash["message"])
            for error_message in review_flash["errors"]:
                st.caption(error_message)

        if not review_results:
            st.success("No readings need detailed inspection.")
        else:
            st.info(
                f"{len(review_results)} reading(s) have detailed OCR, warning, "
                "or verification context. Day Verification remains the primary "
                "confirmation workflow."
            )
            review_mode = "Detailed Review"
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

    with export_tab:
        st.subheader("Export final company workbook")
        export_metrics = st.columns(5)
        export_metrics[0].metric(
            "Confirmed days", len(readiness.confirmed_days)
        )
        export_metrics[1].metric(
            "Unconfirmed days", len(readiness.unconfirmed_days)
        )
        export_metrics[2].metric(
            "Blocking cells", readiness.blocking_cell_count
        )
        export_metrics[3].metric(
            "Blank mismatches", readiness.blank_mismatch_count
        )
        export_metrics[4].metric(
            "Operational warnings", readiness.operational_warning_count
        )
        st.caption(
            f"Extraction geometry recorded for this result: "
            f"**{prepared_sheet.geometry_mode.title()}**"
        )
        if readiness.attention_item_count:
            st.warning(
                f"{readiness.attention_item_count} unresolved attention item(s) "
                "still require human confirmation."
            )
        if not readiness.ready:
            st.warning("The workbook is not ready for export.")
            st.write(
                "Confirm every applicable day and resolve blocking values or "
                "blank mismatches. Operational warnings remain visible but do "
                "not independently block export."
            )
        else:
            st.success("All 31 days and final readings are export-ready.")
            st.write(
                "The export uses the official template and preserves its logo, "
                "formatting, formulas, borders, and print layout."
            )
            month_year_value = st.text_input(
                "Month/Year",
                placeholder="Example: July 2026",
                key=f"export_month_year_{uploaded_fingerprint}",
            )
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
                    file_name="Data_Center_Temperature_Monitoring_Final.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    type="primary",
                )
                mapping_audit_rows = build_excel_mapping_audit(monitoring_rows)
                mapping_audit_csv = pd.DataFrame(mapping_audit_rows).rename(
                    columns={
                        "day": "Day",
                        "point": "Point",
                        "reading_type": "Reading Type",
                        "source_filename": "Source Filename",
                        "final_value": "Final Value",
                        "excel_cell": "Excel Cell",
                    }
                ).to_csv(index=False)
                st.download_button(
                    label="Download Excel mapping audit CSV",
                    data=mapping_audit_csv,
                    file_name="Data_Center_Monitoring_Excel_Mapping_Audit.csv",
                    mime="text/csv",
                )
                st.caption(
                    "The mapping audit lists every source crop and its exact "
                    "Excel destination. The original template is not modified."
                )

        audit_data = build_verification_audit(
            day_state,
            readiness,
            prepared_sheet.geometry_mode,
        )
        with st.expander("Advanced verification audit (troubleshooting)"):
            st.json(audit_data)
            st.download_button(
                "Download verification audit JSON",
                data=json.dumps(audit_data, indent=2, sort_keys=True),
                file_name="Data_Center_Monitoring_Verification_Audit.json",
                mime="application/json",
            )
