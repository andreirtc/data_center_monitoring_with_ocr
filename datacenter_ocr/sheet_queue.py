from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

import cv2
import numpy as np

from datacenter_ocr.document_input import decode_monitoring_documents
from datacenter_ocr.document_orientation import (
    RotationMode,
    detect_monitoring_sheet_orientation,
    rotate_monitoring_image,
)
from datacenter_ocr.sheet_processing import PreparedMonitoringSheet


@dataclass(frozen=True)
class SheetQueueItem:
    """One losslessly encoded monitoring sheet in the upload queue."""

    sheet_id: str
    source_fingerprint: str
    source_filename: str
    display_name: str
    page_number: int
    page_count: int
    source_kind: str
    render_dpi: int | None
    image_source: str
    effective_dpi: float | None
    source_width: int
    source_height: int
    auto_rotation: RotationMode
    orientation_confident: bool
    clockwise_orientation_score: float
    counterclockwise_orientation_score: float
    orientation_score_margin: float
    image_png_bytes: bytes
    decoding_seconds: float


@dataclass(frozen=True)
class ArchivedPreparedMonitoringSheet:
    """PNG-compressed form of an inactive prepared sheet."""

    detection_preview_png: bytes
    warped_table_png: bytes
    measurement_grid_overlay_png: bytes
    measurement_boxes: list[dict]
    cells: tuple[dict, ...]
    geometry_mode: str
    grid_calibration: object | None
    row_sequence_alignment: object | None
    invalid_geometry_cell_count: int


def encode_png(image: np.ndarray) -> bytes:
    """Losslessly encode a page so inactive queue items do not retain arrays."""

    encoded, buffer = cv2.imencode(".png", image)
    if not encoded:
        raise ValueError("A monitoring-sheet page could not be encoded.")
    return buffer.tobytes()


def decode_png(png_bytes: bytes, description: str) -> np.ndarray:
    """Decode a lossless queue image with a contextual error."""

    image = cv2.imdecode(
        np.frombuffer(png_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError(f"{description} could not be restored.")
    return image


def resolve_queue_rotation(
    item: SheetQueueItem,
    orientation_choice: str = "auto",
) -> RotationMode:
    """Resolve the automatic or explicit rotation selected for one sheet."""

    if orientation_choice == "auto":
        return item.auto_rotation
    if orientation_choice in ("none", "clockwise", "counterclockwise"):
        return orientation_choice
    raise ValueError(f"Unknown orientation choice: {orientation_choice}")


def decode_queue_image(
    item: SheetQueueItem,
    orientation_choice: str = "auto",
) -> np.ndarray:
    """Restore one selected queue page to the application's BGR format."""

    source_image = decode_png(
        item.image_png_bytes,
        f"{item.display_name} in the upload queue",
    )
    return rotate_monitoring_image(
        source_image,
        resolve_queue_rotation(item, orientation_choice),
    )


def archive_prepared_sheet(
    sheet: PreparedMonitoringSheet,
) -> ArchivedPreparedMonitoringSheet:
    """Compress an inactive prepared sheet while preserving crop metadata."""

    archived_cells: list[dict] = []
    for cell in sheet.cells:
        archived_cell = {
            key: value for key, value in cell.items() if key != "image"
        }
        archived_cell["image_png_bytes"] = encode_png(cell["image"])
        archived_cells.append(archived_cell)

    return ArchivedPreparedMonitoringSheet(
        detection_preview_png=encode_png(sheet.detection_preview),
        warped_table_png=encode_png(sheet.warped_table),
        measurement_grid_overlay_png=encode_png(
            sheet.measurement_grid_overlay
        ),
        measurement_boxes=sheet.measurement_boxes,
        cells=tuple(archived_cells),
        geometry_mode=sheet.geometry_mode,
        grid_calibration=sheet.grid_calibration,
        row_sequence_alignment=sheet.row_sequence_alignment,
        invalid_geometry_cell_count=sheet.invalid_geometry_cell_count,
    )


def restore_prepared_sheet(
    archived: ArchivedPreparedMonitoringSheet,
) -> PreparedMonitoringSheet:
    """Expand a selected sheet's previews and crops back into OpenCV arrays."""

    cells: list[dict] = []
    for archived_cell in archived.cells:
        cell = {
            key: value
            for key, value in archived_cell.items()
            if key != "image_png_bytes"
        }
        cell["image"] = decode_png(
            archived_cell["image_png_bytes"],
            f"Crop {archived_cell.get('filename', '')}",
        )
        cells.append(cell)

    return PreparedMonitoringSheet(
        detection_preview=decode_png(
            archived.detection_preview_png,
            "Document detection preview",
        ),
        warped_table=decode_png(
            archived.warped_table_png,
            "Warped monitoring table",
        ),
        measurement_grid_overlay=decode_png(
            archived.measurement_grid_overlay_png,
            "Measurement-grid overlay",
        ),
        measurement_boxes=archived.measurement_boxes,
        cells=cells,
        geometry_mode=archived.geometry_mode,
        grid_calibration=archived.grid_calibration,
        row_sequence_alignment=archived.row_sequence_alignment,
        invalid_geometry_cell_count=archived.invalid_geometry_cell_count,
    )


def build_sheet_queue_items(
    uploaded_bytes: bytes,
    source_filename: str,
) -> tuple[SheetQueueItem, ...]:
    """Turn one uploaded image or PDF into independently selectable sheets."""

    source_fingerprint = hashlib.sha256(uploaded_bytes).hexdigest()
    decoding_start = time.perf_counter()
    decoded_pages = decode_monitoring_documents(
        uploaded_bytes,
        source_filename,
    )
    decoding_seconds = time.perf_counter() - decoding_start
    per_page_seconds = decoding_seconds / len(decoded_pages)

    items: list[SheetQueueItem] = []
    for page in decoded_pages:
        source_height, source_width = page.image.shape[:2]
        orientation = detect_monitoring_sheet_orientation(page.image)
        page_identity = (
            f"{source_fingerprint}:page:{page.page_number}".encode("ascii")
        )
        sheet_id = hashlib.sha256(page_identity).hexdigest()
        display_name = (
            source_filename
            if page.page_count == 1
            else (
                f"{source_filename} — Page {page.page_number} "
                f"of {page.page_count}"
            )
        )
        items.append(
            SheetQueueItem(
                sheet_id=sheet_id,
                source_fingerprint=source_fingerprint,
                source_filename=source_filename,
                display_name=display_name,
                page_number=page.page_number,
                page_count=page.page_count,
                source_kind=page.source_kind,
                render_dpi=page.render_dpi,
                image_source=page.image_source,
                effective_dpi=page.effective_dpi,
                source_width=source_width,
                source_height=source_height,
                auto_rotation=orientation.rotation,
                orientation_confident=orientation.confident,
                clockwise_orientation_score=orientation.clockwise_score,
                counterclockwise_orientation_score=(
                    orientation.counterclockwise_score
                ),
                orientation_score_margin=orientation.score_margin,
                image_png_bytes=encode_png(page.image),
                decoding_seconds=per_page_seconds,
            )
        )
    return tuple(items)
