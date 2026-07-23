from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from datacenter_ocr.blank_cell_detection import analyze_cell_for_blankness
from datacenter_ocr.grid_diagnostics import (
    DIAGNOSTIC_DAYS,
    DIAGNOSTIC_POINTS,
    calculate_alignment_report,
)
from datacenter_ocr.image_processing import load_image, save_image
from datacenter_ocr.sheet_processing import (
    prepare_calibrated_monitoring_sheet,
    prepare_monitoring_sheet,
)
from datacenter_ocr.verification import validate_reading_value


BENCHMARK_SCHEMA_VERSION = 2
MATERIAL_DRIFT_THRESHOLD = 0.30
READING_TYPES = ("temperature", "humidity")
SHEET_SOURCES = (
    ("sample", "test_images/sample.png"),
    ("april_2026", "test_images/april_2026.png"),
    ("may_2026", "test_images/may_2026.png"),
)
LABEL_FIELDNAMES = (
    "item_number",
    "sheet_id",
    "source_image",
    "filename",
    "day",
    "point",
    "reading_type",
    "expected_value",
    "expected_blank",
    "fixed_crop_path",
    "calibrated_crop_path",
    "context_crop_path",
    "notes",
)
RESULT_FIELDNAMES = (
    "item_number",
    "sheet_id",
    "filename",
    "day",
    "point",
    "reading_type",
    "day_group",
    "point_group",
    "geometry_mode",
    "selected_geometry",
    "expected_value",
    "expected_blank",
    "raw_original_prediction",
    "raw_grayscale_prediction",
    "raw_contrast_prediction",
    "original_prediction",
    "grayscale_prediction",
    "contrast_prediction",
    "original_confidence",
    "grayscale_confidence",
    "contrast_confidence",
    "consensus_prediction",
    "agreement_count",
    "average_consensus_confidence",
    "proposed_final_value",
    "final_verified_value",
    "predicted_blank",
    "blank_ink_ratio",
    "blank_component_count",
    "blank_largest_component_ratio",
    "blank_analysis_width",
    "blank_analysis_height",
    "postprocessing_status",
    "review_categories",
    "blocks_export",
    "needs_review",
    "review_reason",
    "crop_alignment_failure",
    "final_value_correct",
    "consensus_correct",
    "postprocessing_correct",
    "original_correct",
    "grayscale_correct",
    "contrast_correct",
    "automatic_acceptance",
    "correct_automatic_acceptance",
    "unsafe_automatic_acceptance",
)
TELEMETRY_FIELDNAMES = (
    "item_number",
    "sheet_id",
    "filename",
    "day",
    "point",
    "reading_type",
    "crop_mode",
    "crop_width",
    "crop_height",
    "source_corners",
    "corner_displacements",
    "maximum_corner_displacement",
    "boundary_confidences",
    "boundary_sources",
    "interpolation_count",
    "fallback_count",
    "local_row_height",
    "local_column_width",
    "local_geometry_confidence",
    "blank_before_is_blank",
    "blank_before_ink_ratio",
    "blank_before_component_count",
    "blank_before_largest_component_ratio",
    "blank_after_is_blank",
    "blank_after_ink_ratio",
    "blank_after_component_count",
    "blank_after_largest_component_ratio",
    "blank_analysis_width",
    "blank_analysis_height",
    "sheet_material_drift_score",
    "sheet_requires_calibration",
    "sheet_alignment_metrics",
)
BLANK_COMPARISON_FIELDNAMES = (
    "item_number",
    "sheet_id",
    "filename",
    "day",
    "point",
    "reading_type",
    "expected_blank",
    "crop_mode",
    "before_is_blank",
    "after_is_blank",
    "classification_changed",
    "before_ink_ratio",
    "after_ink_ratio",
    "before_component_count",
    "after_component_count",
    "before_largest_component_ratio",
    "after_largest_component_ratio",
    "false_blank_before",
    "false_blank_after",
)
COMPARISON_FIELDNAMES = (
    "item_number",
    "sheet_id",
    "filename",
    "day",
    "point",
    "reading_type",
    "expected_value",
    "expected_blank",
    "fixed_final_value",
    "calibrated_final_value",
    "fixed_correct",
    "calibrated_correct",
    "correctness_change",
    "fixed_needs_review",
    "calibrated_needs_review",
    "changed_review_disposition",
    "review_disposition_change",
    "fixed_unsafe_automatic_acceptance",
    "calibrated_unsafe_automatic_acceptance",
)


@dataclass(frozen=True)
class BenchmarkIdentity:
    """One stable, sheet-scoped benchmark cell identity."""

    item_number: int
    sheet_id: str
    source_image: str
    filename: str
    cell_filename: str
    day: int
    point: int
    reading_type: str


def benchmark_identities() -> tuple[BenchmarkIdentity, ...]:
    """Return the deterministic 54-cell Stage 3A selection."""

    identities: list[BenchmarkIdentity] = []
    item_number = 1
    for sheet_id, source_image in SHEET_SOURCES:
        for day in DIAGNOSTIC_DAYS:
            for point in DIAGNOSTIC_POINTS:
                for reading_type in READING_TYPES:
                    cell_filename = (
                        f"day_{day:02d}_point_{point:02d}_{reading_type}.png"
                    )
                    identities.append(
                        BenchmarkIdentity(
                            item_number=item_number,
                            sheet_id=sheet_id,
                            source_image=source_image,
                            filename=f"{sheet_id}__{cell_filename}",
                            cell_filename=cell_filename,
                            day=day,
                            point=point,
                            reading_type=reading_type,
                        )
                    )
                    item_number += 1
    return tuple(identities)


def _parse_boolean(value: object, field_name: str) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"{field_name} must be either true or false, not {value!r}.")


def _trusted_label_key(row: Mapping[str, str]) -> tuple[str, str]:
    sheet_id = row.get("sheet_id", "").strip() or "sample"
    filename = row.get("filename", "").strip()
    if "__" in filename:
        filename = filename.split("__", 1)[1]
    return sheet_id, filename


def load_trusted_labels(labels_path: Path) -> dict[tuple[str, str], tuple[str, bool]]:
    """Load trusted labels without editing or broadening their sheet identity.

    The existing two-column legacy file belongs to the sample-sheet benchmark,
    so filename-only rows are intentionally scoped to ``sample``.
    """

    if not labels_path.exists():
        return {}
    with labels_path.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))
    trusted: dict[tuple[str, str], tuple[str, bool]] = {}
    for row in rows:
        expected_value = row.get("expected_value", "").strip()
        expected_blank_text = row.get("expected_blank", "").strip()
        expected_blank = (
            _parse_boolean(expected_blank_text, "expected_blank")
            if expected_blank_text
            else False
        )
        if not expected_value and not expected_blank:
            continue
        if expected_value and expected_blank:
            raise ValueError(
                f"Trusted label {_trusted_label_key(row)!r} has both a value and blank=true."
            )
        trusted[_trusted_label_key(row)] = (expected_value, expected_blank)
    return trusted


def _existing_label_edits(labels_path: Path) -> dict[str, dict[str, str]]:
    if not labels_path.exists():
        return {}
    with labels_path.open(newline="", encoding="utf-8-sig") as csv_file:
        return {
            row.get("filename", "").strip(): row
            for row in csv.DictReader(csv_file)
            if row.get("filename", "").strip()
        }


def _relative_artifact_path(folder_name: str, filename: str) -> str:
    return (Path(folder_name) / filename).as_posix()


def create_label_rows(
    identities: Sequence[BenchmarkIdentity],
    trusted_labels: Mapping[tuple[str, str], tuple[str, bool]],
    existing_rows: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[list[dict[str, str]], int, int]:
    """Create a label template while preserving any prior manual entries."""

    existing_rows = existing_rows or {}
    rows: list[dict[str, str]] = []
    reused_count = 0
    missing_count = 0
    for identity in identities:
        trusted = trusted_labels.get((identity.sheet_id, identity.cell_filename))
        existing = existing_rows.get(identity.filename)
        if existing is not None:
            expected_value = existing.get("expected_value", "").strip()
            expected_blank = existing.get("expected_blank", "false").strip() or "false"
            notes = existing.get("notes", "").strip()
        elif trusted is not None:
            expected_value, trusted_blank = trusted
            expected_blank = str(trusted_blank).lower()
            notes = "Reused unchanged trusted label from local_benchmark/labels.csv."
            reused_count += 1
        else:
            expected_value = ""
            expected_blank = "false"
            notes = (
                "MANUAL LABEL REQUIRED: enter expected_value, or leave it empty "
                "and change expected_blank to true."
            )
        if not expected_value and expected_blank.strip().lower() != "true":
            missing_count += 1
        rows.append(
            {
                "item_number": str(identity.item_number),
                "sheet_id": identity.sheet_id,
                "source_image": identity.source_image,
                "filename": identity.filename,
                "day": str(identity.day),
                "point": str(identity.point),
                "reading_type": identity.reading_type,
                "expected_value": expected_value,
                "expected_blank": expected_blank,
                "fixed_crop_path": _relative_artifact_path(
                    "fixed_crops", identity.filename
                ),
                "calibrated_crop_path": _relative_artifact_path(
                    "calibrated_crops", identity.filename
                ),
                "context_crop_path": _relative_artifact_path(
                    "context_crops", identity.filename
                ),
                "notes": notes,
            }
        )
    return rows, reused_count, missing_count


def write_csv_rows(
    output_path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write a deterministic UTF-8 CSV with a fixed field order."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            max(1, round(image.shape[1] * scale)),
            max(1, round(image.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    x_offset = (width - resized.shape[1]) // 2
    y_offset = (height - resized.shape[0]) // 2
    canvas[
        y_offset : y_offset + resized.shape[0],
        x_offset : x_offset + resized.shape[1],
    ] = resized
    return canvas


def _benchmark_context_crop(
    warped_table: np.ndarray,
    fixed_cell: Mapping[str, Any],
    calibrated_cell: Mapping[str, Any],
) -> np.ndarray:
    fixed_quad = np.asarray(
        [
            [fixed_cell["x1"], fixed_cell["y1"]],
            [fixed_cell["x2"], fixed_cell["y1"]],
            [fixed_cell["x2"], fixed_cell["y2"]],
            [fixed_cell["x1"], fixed_cell["y2"]],
        ],
        dtype=np.float32,
    )
    calibrated_quad = np.asarray(
        calibrated_cell["source_quadrilateral"], dtype=np.float32
    )
    combined = np.vstack((fixed_quad, calibrated_quad))
    x_min, y_min = np.floor(combined.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(combined.max(axis=0)).astype(int)
    width = max(1, x_max - x_min)
    height = max(1, y_max - y_min)
    image_height, image_width = warped_table.shape[:2]
    x1 = max(0, x_min - round(width * 1.10))
    x2 = min(image_width, x_max + round(width * 1.10))
    y1 = max(0, y_min - round(height * 1.25))
    y2 = min(image_height, y_max + round(height * 1.25))
    context = warped_table[y1:y2, x1:x2].copy()
    offset = np.asarray([x1, y1], dtype=np.float32)
    cv2.polylines(
        context,
        [np.round(fixed_quad - offset).astype(np.int32)],
        True,
        (255, 255, 0),
        2,
    )
    cv2.polylines(
        context,
        [np.round(calibrated_quad - offset).astype(np.int32)],
        True,
        (255, 0, 255),
        2,
    )
    return context


def create_labeling_contact_sheet(items: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Lay out context, fixed, and calibrated panels for manual labeling."""

    tile_width = 960
    tile_height = 300
    tiles: list[np.ndarray] = []
    for item in items:
        tile = np.full((tile_height, tile_width, 3), 255, dtype=np.uint8)
        title = (
            f"#{int(item['item_number']):02d} | {item['sheet_id']} | "
            f"day {int(item['day']):02d} | point {item['point']} | "
            f"{item['reading_type']}"
        )
        cv2.putText(
            tile,
            title,
            (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            str(item["filename"]),
            (12, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        panels = (
            ("CONTEXT (cyan fixed, magenta calibrated)", item["context_image"], 12, 500),
            ("FIXED CROP", item["fixed_image"], 522, 200),
            ("CALIBRATED CROP", item["calibrated_image"], 734, 214),
        )
        for label, image, x_offset, panel_width in panels:
            cv2.putText(
                tile,
                label,
                (x_offset, 73),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )
            panel = _fit_image(image, panel_width, 198)
            tile[84:282, x_offset : x_offset + panel_width] = panel
        cv2.rectangle(tile, (0, 0), (tile_width - 1, tile_height - 1), (80, 80, 80), 2)
        tiles.append(tile)
    rows = [
        cv2.hconcat(tiles[start : start + 3])
        for start in range(0, len(tiles), 3)
    ]
    return cv2.vconcat(rows)


def _fixed_source_corners(cell: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            [cell["x1"], cell["y1"]],
            [cell["x2"], cell["y1"]],
            [cell["x2"], cell["y2"]],
            [cell["x1"], cell["y2"]],
        ],
        dtype=np.float64,
    )


def _normalized_boundary_source(source: str) -> str:
    if source == "detected":
        return "detected"
    if source == "fixed_fallback":
        return "fixed_fallback"
    return "interpolated"


def _local_boundary_sample(curve: Any, coordinate: float) -> tuple[float, str]:
    sample_index = min(
        range(len(curve.sample_coordinates)),
        key=lambda index: abs(curve.sample_coordinates[index] - coordinate),
    )
    return (
        float(curve.sample_confidences[sample_index]),
        _normalized_boundary_source(str(curve.sample_sources[sample_index])),
    )


def _calibrated_cell_telemetry(
    cell: Mapping[str, Any],
    calibration: Any,
) -> dict[str, Any]:
    row_index = int(cell["day"]) - 1
    column_index = (int(cell["point"]) - 1) * 2
    if cell["reading_type"] == "humidity":
        column_index += 1
    fixed_corners = _fixed_source_corners(cell)
    calibrated_corners = np.asarray(
        cell["source_quadrilateral"], dtype=np.float64
    )
    center_x = float(np.mean(calibrated_corners[:, 0]))
    center_y = float(np.mean(calibrated_corners[:, 1]))
    curves = (
        calibration.horizontal_curves[row_index],
        calibration.horizontal_curves[row_index + 1],
        calibration.vertical_curves[column_index],
        calibration.vertical_curves[column_index + 1],
    )
    local_samples = (
        _local_boundary_sample(curves[0], center_x),
        _local_boundary_sample(curves[1], center_x),
        _local_boundary_sample(curves[2], center_y),
        _local_boundary_sample(curves[3], center_y),
    )
    boundary_confidences = [round(sample[0], 6) for sample in local_samples]
    boundary_sources = [sample[1] for sample in local_samples]
    displacements = calibrated_corners - fixed_corners
    displacement_rows = [
        {
            "dx": round(float(displacement[0]), 6),
            "dy": round(float(displacement[1]), 6),
            "distance": round(float(np.linalg.norm(displacement)), 6),
        }
        for displacement in displacements
    ]
    row_heights = (
        float(np.linalg.norm(calibrated_corners[3] - calibrated_corners[0])),
        float(np.linalg.norm(calibrated_corners[2] - calibrated_corners[1])),
    )
    column_widths = (
        float(np.linalg.norm(calibrated_corners[1] - calibrated_corners[0])),
        float(np.linalg.norm(calibrated_corners[2] - calibrated_corners[3])),
    )
    return {
        "fixed_source_corners": [
            [round(float(value), 6) for value in corner]
            for corner in fixed_corners
        ],
        "calibrated_source_corners": [
            [round(float(value), 6) for value in corner]
            for corner in calibrated_corners
        ],
        "corner_displacements": displacement_rows,
        "maximum_corner_displacement": max(
            item["distance"] for item in displacement_rows
        ),
        "boundary_confidences": boundary_confidences,
        "boundary_sources": boundary_sources,
        "interpolation_count": sum(
            source == "interpolated" for source in boundary_sources
        ),
        "fallback_count": sum(
            source == "fixed_fallback" for source in boundary_sources
        ),
        "local_row_height": round(float(np.mean(row_heights)), 6),
        "local_column_width": round(float(np.mean(column_widths)), 6),
        "local_geometry_confidence": round(
            float(cell.get("local_geometry_confidence", 0.0)), 6
        ),
    }


def _sheet_alignment_metrics(prepared_sheet: Any) -> dict[str, Any]:
    report = calculate_alignment_report(prepared_sheet)
    horizontal = report["horizontal_measurements"]
    vertical = report["vertical_measurements"]
    normalized_p95_values = [
        float(value)
        for value in (
            horizontal["p95_normalized_error"],
            vertical["p95_normalized_error"],
        )
        if value is not None
    ]
    material_drift_score = max(normalized_p95_values, default=1.0)
    return {
        "material_drift_threshold": MATERIAL_DRIFT_THRESHOLD,
        "material_drift_score": round(material_drift_score, 6),
        "requires_calibration": material_drift_score >= MATERIAL_DRIFT_THRESHOLD,
        "horizontal_p95_normalized_error": horizontal[
            "p95_normalized_error"
        ],
        "vertical_p95_normalized_error": vertical["p95_normalized_error"],
        "horizontal_p95_error_pixels": horizontal["p95_error_pixels"],
        "vertical_p95_error_pixels": vertical["p95_error_pixels"],
        "horizontal_matched_count": horizontal["matched_count"],
        "vertical_matched_count": vertical["matched_count"],
        "provisional_alignment_score": report["provisional_alignment_score"],
    }


def select_geometry_from_alignment_metrics(
    sheet_alignment_metrics: Mapping[str, Any],
) -> str:
    """Choose a benchmark crop mode solely from declared drift metrics."""

    drift_score = float(sheet_alignment_metrics["material_drift_score"])
    threshold = float(
        sheet_alignment_metrics.get(
            "material_drift_threshold", MATERIAL_DRIFT_THRESHOLD
        )
    )
    return "calibrated" if drift_score >= threshold else "fixed"


def _blank_analysis_fields(image: np.ndarray, normalize: bool) -> dict[str, Any]:
    analysis = analyze_cell_for_blankness(
        image,
        normalize_analysis_canvas=normalize,
    )
    return {
        "is_blank": analysis.is_blank,
        "ink_ratio": round(float(analysis.ink_ratio), 6),
        "component_count": analysis.significant_component_count,
        "largest_component_ratio": round(
            float(analysis.largest_component_ratio), 6
        ),
        "analysis_width": analysis.analysis_width,
        "analysis_height": analysis.analysis_height,
    }


def _telemetry_csv_row(
    *,
    identity: BenchmarkIdentity,
    crop_mode: str,
    crop_image: np.ndarray,
    geometry: Mapping[str, Any],
    blank_before: Mapping[str, Any],
    blank_after: Mapping[str, Any],
    sheet_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    source_corners = geometry[f"{crop_mode}_source_corners"]
    if crop_mode == "fixed":
        corner_displacements = [
            {"dx": 0.0, "dy": 0.0, "distance": 0.0} for _ in range(4)
        ]
        boundary_confidences = [0.0] * 4
        boundary_sources = ["fixed_fallback"] * 4
        interpolation_count = 0
        fallback_count = 4
        fixed_corners = np.asarray(source_corners, dtype=np.float64)
        local_row_height = float(fixed_corners[3, 1] - fixed_corners[0, 1])
        local_column_width = float(fixed_corners[1, 0] - fixed_corners[0, 0])
        local_geometry_confidence = 0.0
        maximum_displacement = 0.0
    else:
        corner_displacements = geometry["corner_displacements"]
        boundary_confidences = geometry["boundary_confidences"]
        boundary_sources = geometry["boundary_sources"]
        interpolation_count = geometry["interpolation_count"]
        fallback_count = geometry["fallback_count"]
        local_row_height = geometry["local_row_height"]
        local_column_width = geometry["local_column_width"]
        local_geometry_confidence = geometry["local_geometry_confidence"]
        maximum_displacement = geometry["maximum_corner_displacement"]
    return {
        "item_number": identity.item_number,
        "sheet_id": identity.sheet_id,
        "filename": identity.filename,
        "day": identity.day,
        "point": identity.point,
        "reading_type": identity.reading_type,
        "crop_mode": crop_mode,
        "crop_width": crop_image.shape[1],
        "crop_height": crop_image.shape[0],
        "source_corners": json.dumps(source_corners, separators=(",", ":")),
        "corner_displacements": json.dumps(
            corner_displacements, separators=(",", ":")
        ),
        "maximum_corner_displacement": maximum_displacement,
        "boundary_confidences": json.dumps(
            boundary_confidences, separators=(",", ":")
        ),
        "boundary_sources": json.dumps(boundary_sources, separators=(",", ":")),
        "interpolation_count": interpolation_count,
        "fallback_count": fallback_count,
        "local_row_height": local_row_height,
        "local_column_width": local_column_width,
        "local_geometry_confidence": local_geometry_confidence,
        "blank_before_is_blank": blank_before["is_blank"],
        "blank_before_ink_ratio": blank_before["ink_ratio"],
        "blank_before_component_count": blank_before["component_count"],
        "blank_before_largest_component_ratio": blank_before[
            "largest_component_ratio"
        ],
        "blank_after_is_blank": blank_after["is_blank"],
        "blank_after_ink_ratio": blank_after["ink_ratio"],
        "blank_after_component_count": blank_after["component_count"],
        "blank_after_largest_component_ratio": blank_after[
            "largest_component_ratio"
        ],
        "blank_analysis_width": blank_after["analysis_width"],
        "blank_analysis_height": blank_after["analysis_height"],
        "sheet_material_drift_score": sheet_metrics["material_drift_score"],
        "sheet_requires_calibration": sheet_metrics["requires_calibration"],
        "sheet_alignment_metrics": json.dumps(
            dict(sheet_metrics), sort_keys=True, separators=(",", ":")
        ),
    }


def prepare_benchmark_artifacts(
    project_root: Path,
    output_folder: Path,
    trusted_labels_path: Path,
) -> dict[str, Any]:
    """Generate geometry-only Stage 3A crops, labels, contact sheet, and manifest."""

    identities = benchmark_identities()
    identity_lookup = {
        (identity.sheet_id, identity.cell_filename): identity
        for identity in identities
    }
    labels_path = output_folder / "labels.csv"
    trusted_labels = load_trusted_labels(trusted_labels_path)
    label_rows, reused_count, missing_count = create_label_rows(
        identities,
        trusted_labels,
        _existing_label_edits(labels_path),
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    for folder_name in ("fixed_crops", "calibrated_crops", "context_crops"):
        (output_folder / folder_name).mkdir(parents=True, exist_ok=True)

    contact_items: list[dict[str, Any]] = []
    manifest_items: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, Any]] = []
    sheet_metrics_by_id: dict[str, dict[str, Any]] = {}
    for sheet_id, source_image in SHEET_SOURCES:
        original_image = load_image(project_root / source_image)
        fixed_sheet = prepare_monitoring_sheet(original_image, geometry_mode="fixed")
        calibrated_sheet = prepare_calibrated_monitoring_sheet(fixed_sheet)
        if calibrated_sheet.grid_calibration is None:
            raise RuntimeError("Calibrated sheet has no local-grid calibration.")
        sheet_metrics = _sheet_alignment_metrics(fixed_sheet)
        calibration = calibrated_sheet.grid_calibration
        sheet_metrics.update(
            {
                "interpolated_boundary_count": sum(
                    curve.uses_interpolation
                    for curve in (
                        *calibration.horizontal_curves,
                        *calibration.vertical_curves,
                    )
                ),
                "fallback_boundary_count": sum(
                    curve.uses_fixed_fallback
                    for curve in (
                        *calibration.horizontal_curves,
                        *calibration.vertical_curves,
                    )
                ),
                "low_confidence_boundary_count": (
                    calibration.low_confidence_horizontal_boundary_count
                    + calibration.low_confidence_vertical_boundary_count
                ),
                "invalid_geometry_cell_count": (
                    calibrated_sheet.invalid_geometry_cell_count
                ),
            }
        )
        sheet_metrics_by_id[sheet_id] = sheet_metrics
        fixed_lookup = {cell["filename"]: cell for cell in fixed_sheet.cells}
        calibrated_lookup = {
            cell["filename"]: cell for cell in calibrated_sheet.cells
        }
        for cell_filename in (
            identity.cell_filename
            for identity in identities
            if identity.sheet_id == sheet_id
        ):
            identity = identity_lookup[(sheet_id, cell_filename)]
            fixed_cell = fixed_lookup[cell_filename]
            calibrated_cell = calibrated_lookup[cell_filename]
            context_image = _benchmark_context_crop(
                fixed_sheet.warped_table, fixed_cell, calibrated_cell
            )
            fixed_path = output_folder / "fixed_crops" / identity.filename
            calibrated_path = output_folder / "calibrated_crops" / identity.filename
            context_path = output_folder / "context_crops" / identity.filename
            save_image(fixed_cell["image"], fixed_path)
            save_image(calibrated_cell["image"], calibrated_path)
            save_image(context_image, context_path)
            geometry_telemetry = _calibrated_cell_telemetry(
                calibrated_cell,
                calibration,
            )
            fixed_blank_before = _blank_analysis_fields(
                fixed_cell["image"], normalize=False
            )
            fixed_blank_after = _blank_analysis_fields(
                fixed_cell["image"], normalize=True
            )
            calibrated_blank_before = _blank_analysis_fields(
                calibrated_cell["image"], normalize=False
            )
            calibrated_blank_after = _blank_analysis_fields(
                calibrated_cell["image"], normalize=True
            )
            telemetry_rows.extend(
                (
                    _telemetry_csv_row(
                        identity=identity,
                        crop_mode="fixed",
                        crop_image=fixed_cell["image"],
                        geometry=geometry_telemetry,
                        blank_before=fixed_blank_before,
                        blank_after=fixed_blank_after,
                        sheet_metrics=sheet_metrics,
                    ),
                    _telemetry_csv_row(
                        identity=identity,
                        crop_mode="calibrated",
                        crop_image=calibrated_cell["image"],
                        geometry=geometry_telemetry,
                        blank_before=calibrated_blank_before,
                        blank_after=calibrated_blank_after,
                        sheet_metrics=sheet_metrics,
                    ),
                )
            )
            contact_items.append(
                {
                    **asdict(identity),
                    "fixed_image": fixed_cell["image"],
                    "calibrated_image": calibrated_cell["image"],
                    "context_image": context_image,
                }
            )
            manifest_items.append(
                {
                    **asdict(identity),
                    "fixed_crop_path": _relative_artifact_path(
                        "fixed_crops", identity.filename
                    ),
                    "calibrated_crop_path": _relative_artifact_path(
                        "calibrated_crops", identity.filename
                    ),
                    "context_crop_path": _relative_artifact_path(
                        "context_crops", identity.filename
                    ),
                    "fixed_crop_alignment_failure": False,
                    "calibrated_crop_alignment_failure": bool(
                        calibrated_cell.get("geometry_rejection_reason", "")
                    ),
                    "calibrated_geometry_rejection_reason": calibrated_cell.get(
                        "geometry_rejection_reason", ""
                    ),
                    "calibrated_uses_fallback_geometry": bool(
                        calibrated_cell.get("uses_fallback_geometry", False)
                    ),
                    "calibrated_uses_interpolated_geometry": bool(
                        calibrated_cell.get("uses_interpolated_geometry", False)
                    ),
                    "calibrated_local_geometry_confidence": calibrated_cell.get(
                        "local_geometry_confidence", 0.0
                    ),
                    "fixed_crop_dimensions": {
                        "width": fixed_cell["image"].shape[1],
                        "height": fixed_cell["image"].shape[0],
                    },
                    "calibrated_crop_dimensions": {
                        "width": calibrated_cell["image"].shape[1],
                        "height": calibrated_cell["image"].shape[0],
                    },
                    **geometry_telemetry,
                    "fixed_blank_analysis_before": fixed_blank_before,
                    "fixed_blank_analysis_after": fixed_blank_after,
                    "calibrated_blank_analysis_before": calibrated_blank_before,
                    "calibrated_blank_analysis_after": calibrated_blank_after,
                    "sheet_alignment_metrics": sheet_metrics,
                }
            )

    write_csv_rows(labels_path, LABEL_FIELDNAMES, label_rows)
    telemetry_path = output_folder / "geometry_telemetry.csv"
    write_csv_rows(telemetry_path, TELEMETRY_FIELDNAMES, telemetry_rows)
    contact_sheet_path = output_folder / "labeling_contact_sheet.png"
    save_image(create_labeling_contact_sheet(contact_items), contact_sheet_path)
    manifest = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_name": "fixed_vs_calibrated_geometry_ocr",
        "geometry_only_preparation": True,
        "ocr_was_run": False,
        "production_geometry_default": "fixed",
        "selection": {
            "sheets": [sheet_id for sheet_id, _ in SHEET_SOURCES],
            "days": list(DIAGNOSTIC_DAYS),
            "points": list(DIAGNOSTIC_POINTS),
            "reading_types": list(READING_TYPES),
            "item_count": len(identities),
        },
        "trusted_labels_source": trusted_labels_path.as_posix(),
        "trusted_labels_reused": reused_count,
        "missing_labels": missing_count,
        "labels_path": labels_path.as_posix(),
        "labeling_contact_sheet_path": contact_sheet_path.as_posix(),
        "geometry_telemetry_path": telemetry_path.as_posix(),
        "sheet_alignment_metrics": sheet_metrics_by_id,
        "items": manifest_items,
    }
    manifest_path = output_folder / "benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_label_rows(labels_path: Path) -> list[dict[str, str]]:
    """Read labels without importing or initializing OCR."""

    if not labels_path.exists():
        raise FileNotFoundError(f"Benchmark labels were not found: {labels_path}")
    with labels_path.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError("The benchmark labels file contains no rows.")
    return rows


def validate_complete_labels(
    rows: Sequence[Mapping[str, str]],
    identities: Sequence[BenchmarkIdentity] | None = None,
) -> list[dict[str, Any]]:
    """Validate identities and ground truth, refusing incomplete labels."""

    identities = tuple(identities or benchmark_identities())
    expected_by_filename = {identity.filename: identity for identity in identities}
    seen: set[str] = set()
    missing_labels: list[str] = []
    errors: list[str] = []
    validated: list[dict[str, Any]] = []
    for row in rows:
        filename = row.get("filename", "").strip()
        if filename in seen:
            errors.append(f"Duplicate filename: {filename}")
            continue
        seen.add(filename)
        identity = expected_by_filename.get(filename)
        if identity is None:
            errors.append(f"Unexpected benchmark identity: {filename}")
            continue
        try:
            expected_blank = _parse_boolean(
                row.get("expected_blank", ""), "expected_blank"
            )
        except ValueError as error:
            errors.append(f"{filename}: {error}")
            continue
        expected_value = row.get("expected_value", "").strip()
        if expected_blank and expected_value:
            errors.append(
                f"{filename}: blanks must have an empty expected_value."
            )
        elif not expected_blank and not expected_value:
            missing_labels.append(filename)
        elif not expected_blank:
            validation = validate_reading_value(
                expected_value, identity.reading_type, allow_blank=False
            )
            if validation.error is not None:
                errors.append(f"{filename}: invalid ground truth: {validation.error}")
        validated.append(
            {
                **dict(row),
                "item_number": identity.item_number,
                "sheet_id": identity.sheet_id,
                "filename": identity.filename,
                "day": identity.day,
                "point": identity.point,
                "reading_type": identity.reading_type,
                "expected_value": expected_value,
                "expected_blank": expected_blank,
            }
        )
    absent = sorted(set(expected_by_filename) - seen)
    if absent:
        errors.append("Missing selected rows: " + ", ".join(absent))
    if missing_labels:
        errors.append(
            "Missing ground-truth labels (enter expected_value or set "
            "expected_blank=true): " + ", ".join(missing_labels)
        )
    if errors:
        raise ValueError("Benchmark labels are not complete and valid:\n- " + "\n- ".join(errors))
    return sorted(validated, key=lambda row: int(row["item_number"]))


def load_mode_cells(
    labels_path: Path,
    labels: Sequence[Mapping[str, Any]],
    geometry_mode: str,
) -> list[dict[str, Any]]:
    """Load prediction inputs without copying ground truth into OCR jobs."""

    if geometry_mode not in {"fixed", "calibrated"}:
        raise ValueError(f"Unknown geometry mode: {geometry_mode}")
    path_field = f"{geometry_mode}_crop_path"
    cells: list[dict[str, Any]] = []
    for label in labels:
        crop_path = Path(str(label[path_field]))
        if not crop_path.is_absolute():
            crop_path = labels_path.parent / crop_path
        image = cv2.imread(str(crop_path))
        if image is None:
            raise FileNotFoundError(f"Could not read {geometry_mode} crop: {crop_path}")
        cells.append(
            {
                "filename": str(label["filename"]),
                "day": int(label["day"]),
                "point": int(label["point"]),
                "reading_type": str(label["reading_type"]),
                "sheet_id": str(label["sheet_id"]),
                "image": image,
            }
        )
    return cells


def alignment_failure_lookup(labels_path: Path) -> dict[tuple[str, str], bool]:
    """Read geometry rejection facts recorded during crop preparation."""

    manifest_path = labels_path.parent / "benchmark_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: dict[tuple[str, str], bool] = {}
    for item in manifest.get("items", []):
        filename = str(item["filename"])
        failures[("fixed", filename)] = bool(
            item.get("fixed_crop_alignment_failure", False)
        )
        failures[("calibrated", filename)] = bool(
            item.get("calibrated_crop_alignment_failure", False)
        )
    return failures


def load_benchmark_manifest(labels_path: Path) -> dict[str, Any]:
    """Load the geometry-only manifest adjacent to benchmark labels."""

    manifest_path = labels_path.parent / "benchmark_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Benchmark manifest was not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_blank_analysis_comparison(
    labels: Sequence[Mapping[str, Any]],
    cells_by_mode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Evaluate old and normalized blank analysis without invoking OCR."""

    label_lookup = {str(label["filename"]): label for label in labels}
    rows: list[dict[str, Any]] = []
    for crop_mode in ("fixed", "calibrated"):
        for cell in cells_by_mode[crop_mode]:
            filename = str(cell["filename"])
            label = label_lookup[filename]
            before = _blank_analysis_fields(cell["image"], normalize=False)
            after = _blank_analysis_fields(cell["image"], normalize=True)
            expected_blank = bool(label["expected_blank"])
            rows.append(
                {
                    "item_number": int(label["item_number"]),
                    "sheet_id": str(label["sheet_id"]),
                    "filename": filename,
                    "day": int(label["day"]),
                    "point": int(label["point"]),
                    "reading_type": str(label["reading_type"]),
                    "expected_blank": expected_blank,
                    "crop_mode": crop_mode,
                    "before_is_blank": before["is_blank"],
                    "after_is_blank": after["is_blank"],
                    "classification_changed": (
                        before["is_blank"] != after["is_blank"]
                    ),
                    "before_ink_ratio": before["ink_ratio"],
                    "after_ink_ratio": after["ink_ratio"],
                    "before_component_count": before["component_count"],
                    "after_component_count": after["component_count"],
                    "before_largest_component_ratio": before[
                        "largest_component_ratio"
                    ],
                    "after_largest_component_ratio": after[
                        "largest_component_ratio"
                    ],
                    "false_blank_before": (
                        before["is_blank"] and not expected_blank
                    ),
                    "false_blank_after": after["is_blank"] and not expected_blank,
                }
            )
    return sorted(
        rows,
        key=lambda row: (row["crop_mode"], int(row["item_number"])),
    )


def _blank_classification_metrics(
    rows: Sequence[Mapping[str, Any]],
    classification_field: str,
) -> dict[str, Any]:
    expected_blank_count = sum(bool(row["expected_blank"]) for row in rows)
    predicted_blank_count = sum(bool(row[classification_field]) for row in rows)
    true_blank_count = sum(
        bool(row["expected_blank"]) and bool(row[classification_field])
        for row in rows
    )
    false_blank_count = sum(
        not bool(row["expected_blank"]) and bool(row[classification_field])
        for row in rows
    )
    return {
        "cell_count": len(rows),
        "expected_blank_count": expected_blank_count,
        "predicted_blank_count": predicted_blank_count,
        "true_blank_count": true_blank_count,
        "false_blank_count": false_blank_count,
        "blank_precision": _safe_rate(true_blank_count, predicted_blank_count),
        "blank_recall": _safe_rate(true_blank_count, expected_blank_count),
    }


def summarize_blank_analysis(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize before/after blank precision, recall, and changed identities."""

    report: dict[str, Any] = {}
    for crop_mode in ("fixed", "calibrated"):
        mode_rows = [row for row in rows if row["crop_mode"] == crop_mode]
        report[crop_mode] = {
            "before": _blank_classification_metrics(
                mode_rows, "before_is_blank"
            ),
            "after": _blank_classification_metrics(mode_rows, "after_is_blank"),
            "changed_classifications": [
                str(row["filename"])
                for row in mode_rows
                if row["classification_changed"]
            ],
            "filled_cells_newly_classified_blank": [
                str(row["filename"])
                for row in mode_rows
                if row["false_blank_after"]
            ],
        }
    return report


def day_group(day: int) -> str:
    return {1: "top", 16: "middle", 31: "bottom"}[day]


def point_group(point: int) -> str:
    return {1: "left", 4: "center", 8: "right"}[point]


def _prediction_is_correct(
    prediction: str,
    predicted_blank: bool,
    expected_value: str,
    expected_blank: bool,
) -> bool:
    if expected_blank:
        return predicted_blank and prediction == ""
    return not predicted_blank and prediction == expected_value


def evaluate_prediction_result(
    label: Mapping[str, Any],
    result: Any,
    geometry_mode: str,
    proposed_final_value: str,
    crop_alignment_failure: bool,
    blank_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Join completed ground truth to an already-produced OCR result."""

    expected_value = str(label["expected_value"])
    expected_blank = bool(label["expected_blank"])
    predicted_blank = bool(result.is_blank)
    final_correct = _prediction_is_correct(
        result.final_value,
        predicted_blank,
        expected_value,
        expected_blank,
    )
    consensus_correct = _prediction_is_correct(
        result.consensus_prediction,
        predicted_blank,
        expected_value,
        expected_blank,
    )
    postprocessing_correct = _prediction_is_correct(
        proposed_final_value,
        predicted_blank,
        expected_value,
        expected_blank,
    )
    empty_nonblank_crop_proxy = (
        not expected_blank
        and not any(result.predictions.values())
        and not result.consensus_prediction
        and not result.final_value
    )
    crop_alignment_failure = crop_alignment_failure or empty_nonblank_crop_proxy
    automatic_acceptance = not result.needs_review and not result.blocks_export
    blank_analysis = blank_analysis or {}
    row: dict[str, Any] = {
        "item_number": int(label["item_number"]),
        "sheet_id": str(label["sheet_id"]),
        "filename": str(label["filename"]),
        "day": int(label["day"]),
        "point": int(label["point"]),
        "reading_type": str(label["reading_type"]),
        "day_group": day_group(int(label["day"])),
        "point_group": point_group(int(label["point"])),
        "geometry_mode": geometry_mode,
        "selected_geometry": geometry_mode,
        "expected_value": expected_value,
        "expected_blank": expected_blank,
        "consensus_prediction": result.consensus_prediction,
        "agreement_count": result.agreement_count,
        "average_consensus_confidence": result.average_consensus_confidence,
        "proposed_final_value": proposed_final_value,
        "final_verified_value": result.final_value,
        "predicted_blank": predicted_blank,
        "blank_ink_ratio": blank_analysis.get(
            "ink_ratio", getattr(result, "blank_ink_ratio", 0.0)
        ),
        "blank_component_count": blank_analysis.get("component_count", 0),
        "blank_largest_component_ratio": blank_analysis.get(
            "largest_component_ratio", 0.0
        ),
        "blank_analysis_width": blank_analysis.get("analysis_width", 0),
        "blank_analysis_height": blank_analysis.get("analysis_height", 0),
        "postprocessing_status": result.postprocessing_status,
        "review_categories": "|".join(result.review_categories),
        "blocks_export": result.blocks_export,
        "needs_review": result.needs_review,
        "review_reason": result.review_reason,
        "crop_alignment_failure": crop_alignment_failure,
        "final_value_correct": final_correct,
        "consensus_correct": consensus_correct,
        "postprocessing_correct": postprocessing_correct,
        "automatic_acceptance": automatic_acceptance,
        "correct_automatic_acceptance": automatic_acceptance and final_correct,
        "unsafe_automatic_acceptance": automatic_acceptance and not final_correct,
    }
    for variant in ("original", "grayscale", "contrast"):
        prediction = result.predictions.get(variant, "")
        row[f"raw_{variant}_prediction"] = result.raw_predictions.get(variant, "")
        row[f"{variant}_prediction"] = prediction
        row[f"{variant}_confidence"] = result.confidences.get(variant, 0.0)
        row[f"{variant}_correct"] = _prediction_is_correct(
            prediction,
            predicted_blank,
            expected_value,
            expected_blank,
        )
    return row


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def calculate_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    ocr_input_count: int | None = None,
    total_ocr_seconds: float | None = None,
    total_processing_seconds: float | None = None,
) -> dict[str, Any]:
    """Calculate benchmark accuracy, review, blank, alignment, and timing metrics."""

    total = len(rows)
    final_correct = sum(bool(row["final_value_correct"]) for row in rows)
    consensus_correct = sum(bool(row["consensus_correct"]) for row in rows)
    postprocessing_correct = sum(bool(row["postprocessing_correct"]) for row in rows)
    reviews = sum(bool(row["needs_review"]) for row in rows)
    false_reviews = sum(
        bool(row["needs_review"]) and bool(row["final_value_correct"])
        for row in rows
    )
    reviewed_incorrect = reviews - false_reviews
    expected_blanks = sum(bool(row["expected_blank"]) for row in rows)
    predicted_blanks = sum(bool(row["predicted_blank"]) for row in rows)
    true_blanks = sum(
        bool(row["expected_blank"]) and bool(row["predicted_blank"])
        for row in rows
    )
    metrics: dict[str, Any] = {
        "cell_count": total,
        "exact_final_value_correct_count": final_correct,
        "exact_final_value_accuracy": _safe_rate(final_correct, total),
        "exact_ocr_consensus_correct_count": consensus_correct,
        "exact_ocr_consensus_accuracy": _safe_rate(consensus_correct, total),
        "postprocessing_correct_count": postprocessing_correct,
        "postprocessing_accuracy": _safe_rate(postprocessing_correct, total),
        "per_variant_accuracy": {
            variant: _safe_rate(
                sum(bool(row[f"{variant}_correct"]) for row in rows), total
            )
            for variant in ("original", "grayscale", "contrast")
        },
        "correct_automatic_acceptances": sum(
            bool(row["correct_automatic_acceptance"]) for row in rows
        ),
        "unsafe_automatic_acceptances": sum(
            bool(row["unsafe_automatic_acceptance"]) for row in rows
        ),
        "review_count": reviews,
        "review_rate": _safe_rate(reviews, total),
        "review_precision": _safe_rate(reviewed_incorrect, reviews),
        "false_review_count": false_reviews,
        "expected_blank_count": expected_blanks,
        "predicted_blank_count": predicted_blanks,
        "blank_precision": _safe_rate(true_blanks, predicted_blanks),
        "blank_recall": _safe_rate(true_blanks, expected_blanks),
        "crop_alignment_failure_count": sum(
            bool(row["crop_alignment_failure"]) for row in rows
        ),
    }
    if ocr_input_count is not None:
        metrics["ocr_input_count"] = int(ocr_input_count)
        metrics["average_ocr_inputs_per_cell"] = _safe_rate(ocr_input_count, total)
    if total_ocr_seconds is not None:
        metrics["total_ocr_time_seconds"] = round(float(total_ocr_seconds), 6)
        metrics["average_ocr_time_per_cell_seconds"] = (
            round(float(total_ocr_seconds) / total, 6) if total else None
        )
    if total_processing_seconds is not None:
        metrics["total_processing_time_seconds"] = round(
            float(total_processing_seconds), 6
        )
    return metrics


def summarize_mode(
    rows: Sequence[Mapping[str, Any]],
    *,
    ocr_input_count: int,
    total_ocr_seconds: float,
    total_processing_seconds: float,
) -> dict[str, Any]:
    """Report overall and required position-based breakdowns for one mode."""

    dimensions = ("sheet_id", "reading_type", "day_group", "point_group")
    breakdowns: dict[str, dict[str, Any]] = {}
    for dimension in dimensions:
        values = sorted({str(row[dimension]) for row in rows})
        breakdowns[dimension] = {
            value: calculate_metrics(
                [row for row in rows if str(row[dimension]) == value]
            )
            for value in values
        }
    return {
        "overall": calculate_metrics(
            rows,
            ocr_input_count=ocr_input_count,
            total_ocr_seconds=total_ocr_seconds,
            total_processing_seconds=total_processing_seconds,
        ),
        "breakdowns": breakdowns,
    }


def build_hybrid_rows(
    fixed_rows: Sequence[Mapping[str, Any]],
    calibrated_rows: Sequence[Mapping[str, Any]],
    sheet_alignment_metrics: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build a confirmation-only benchmark hybrid without using ground truth.

    Geometry selection reads only declared sheet alignment metrics. Confirmation
    rules read only the two modes' predictions, blank states, and verification
    dispositions. Expected values remain present for later metric calculation
    but never participate in selection or confirmation decisions.
    """

    fixed_lookup = {str(row["filename"]): row for row in fixed_rows}
    calibrated_lookup = {str(row["filename"]): row for row in calibrated_rows}
    if set(fixed_lookup) != set(calibrated_lookup):
        raise ValueError("Fixed and calibrated hybrid identities do not match.")

    counters = {
        "fixed_selected_count": 0,
        "calibrated_selected_count": 0,
        "disagreement_confirmation_count": 0,
        "blank_disagreement_count": 0,
        "blocking_automatic_disagreement_count": 0,
        "geometry_triggered_confirmation_count": 0,
    }
    hybrid_rows: list[dict[str, Any]] = []
    for filename in sorted(
        fixed_lookup, key=lambda value: int(fixed_lookup[value]["item_number"])
    ):
        fixed = fixed_lookup[filename]
        calibrated = calibrated_lookup[filename]
        sheet_id = str(fixed["sheet_id"])
        selected_geometry = select_geometry_from_alignment_metrics(
            sheet_alignment_metrics[sheet_id]
        )
        selected = fixed if selected_geometry == "fixed" else calibrated
        counters[f"{selected_geometry}_selected_count"] += 1

        value_disagreement = (
            str(fixed["proposed_final_value"])
            != str(calibrated["proposed_final_value"])
        )
        blank_disagreement = bool(fixed["predicted_blank"]) != bool(
            calibrated["predicted_blank"]
        )
        blocking_automatic_disagreement = (
            bool(fixed["blocks_export"]) and bool(calibrated["automatic_acceptance"])
        ) or (
            bool(calibrated["blocks_export"]) and bool(fixed["automatic_acceptance"])
        )
        geometry_confirmation = selected_geometry == "calibrated"
        if value_disagreement:
            counters["disagreement_confirmation_count"] += 1
        if blank_disagreement:
            counters["blank_disagreement_count"] += 1
        if blocking_automatic_disagreement:
            counters["blocking_automatic_disagreement_count"] += 1
        if geometry_confirmation:
            counters["geometry_triggered_confirmation_count"] += 1

        confirmation_reasons: list[str] = []
        if value_disagreement:
            confirmation_reasons.append(
                "Fixed and calibrated proposed values differ."
            )
        if blank_disagreement:
            confirmation_reasons.append(
                "Fixed and calibrated blank classifications differ."
            )
        if blocking_automatic_disagreement:
            confirmation_reasons.append(
                "One geometry blocks while the other would be automatic."
            )
        if geometry_confirmation:
            confirmation_reasons.append(
                "Calibrated-selected results require confirmation during Stage 3B."
            )

        hybrid = dict(selected)
        hybrid["geometry_mode"] = "hybrid"
        hybrid["selected_geometry"] = selected_geometry
        if confirmation_reasons:
            categories = [
                category
                for category in str(hybrid.get("review_categories", "")).split("|")
                if category
            ]
            categories.append("benchmark_geometry_confirmation")
            hybrid["review_categories"] = "|".join(dict.fromkeys(categories))
            existing_reason = str(hybrid.get("review_reason", "")).strip()
            hybrid["review_reason"] = " ".join(
                [existing_reason, *confirmation_reasons]
            ).strip()
            hybrid["needs_review"] = True
            hybrid["blocks_export"] = True
            hybrid["automatic_acceptance"] = False
            hybrid["correct_automatic_acceptance"] = False
            hybrid["unsafe_automatic_acceptance"] = False
        hybrid_rows.append(hybrid)
    return hybrid_rows, counters


def compare_geometry_rows(
    fixed_rows: Sequence[Mapping[str, Any]],
    calibrated_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create one fixed-versus-calibrated correctness and review comparison row."""

    fixed_lookup = {str(row["filename"]): row for row in fixed_rows}
    calibrated_lookup = {str(row["filename"]): row for row in calibrated_rows}
    if set(fixed_lookup) != set(calibrated_lookup):
        raise ValueError("Fixed and calibrated benchmark identities do not match.")
    comparisons: list[dict[str, Any]] = []
    for filename in sorted(
        fixed_lookup, key=lambda value: int(fixed_lookup[value]["item_number"])
    ):
        fixed = fixed_lookup[filename]
        calibrated = calibrated_lookup[filename]
        fixed_correct = bool(fixed["final_value_correct"])
        calibrated_correct = bool(calibrated["final_value_correct"])
        if not fixed_correct and calibrated_correct:
            correctness_change = "improved"
        elif fixed_correct and not calibrated_correct:
            correctness_change = "regressed"
        elif fixed_correct:
            correctness_change = "stayed_correct"
        else:
            correctness_change = "stayed_incorrect"
        fixed_review = bool(fixed["needs_review"])
        calibrated_review = bool(calibrated["needs_review"])
        if fixed_review == calibrated_review:
            review_change = "unchanged"
        elif fixed_review:
            review_change = "review_to_automatic"
        else:
            review_change = "automatic_to_review"
        comparisons.append(
            {
                "item_number": fixed["item_number"],
                "sheet_id": fixed["sheet_id"],
                "filename": filename,
                "day": fixed["day"],
                "point": fixed["point"],
                "reading_type": fixed["reading_type"],
                "expected_value": fixed["expected_value"],
                "expected_blank": fixed["expected_blank"],
                "fixed_final_value": fixed["final_verified_value"],
                "calibrated_final_value": calibrated["final_verified_value"],
                "fixed_correct": fixed_correct,
                "calibrated_correct": calibrated_correct,
                "correctness_change": correctness_change,
                "fixed_needs_review": fixed_review,
                "calibrated_needs_review": calibrated_review,
                "changed_review_disposition": fixed_review != calibrated_review,
                "review_disposition_change": review_change,
                "fixed_unsafe_automatic_acceptance": fixed[
                    "unsafe_automatic_acceptance"
                ],
                "calibrated_unsafe_automatic_acceptance": calibrated[
                    "unsafe_automatic_acceptance"
                ],
            }
        )
    return comparisons


def assess_calibrated_safety(
    fixed_summary: Mapping[str, Any],
    calibrated_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply conservative Stage 3A gates without changing production behavior."""

    fixed_overall = fixed_summary["overall"]
    calibrated_overall = calibrated_summary["overall"]
    fixed_sheets = fixed_summary["breakdowns"]["sheet_id"]
    calibrated_sheets = calibrated_summary["breakdowns"]["sheet_id"]
    checks = {
        "exact_accuracy_stable_or_improved": (
            calibrated_overall["exact_final_value_correct_count"]
            >= fixed_overall["exact_final_value_correct_count"]
        ),
        "unsafe_automatic_acceptances_not_increased": (
            calibrated_overall["unsafe_automatic_acceptances"]
            <= fixed_overall["unsafe_automatic_acceptances"]
        ),
        "sample_not_materially_regressed": (
            calibrated_sheets["sample"]["exact_final_value_correct_count"]
            >= fixed_sheets["sample"]["exact_final_value_correct_count"]
            and calibrated_sheets["sample"]["unsafe_automatic_acceptances"]
            <= fixed_sheets["sample"]["unsafe_automatic_acceptances"]
        ),
        "crop_alignment_failures_decreased": (
            calibrated_overall["crop_alignment_failure_count"]
            < fixed_overall["crop_alignment_failure_count"]
        ),
        "april_meaningfully_improved": (
            calibrated_sheets["april_2026"]["exact_final_value_correct_count"]
            > fixed_sheets["april_2026"]["exact_final_value_correct_count"]
        ),
        "may_meaningfully_improved": (
            calibrated_sheets["may_2026"]["exact_final_value_correct_count"]
            > fixed_sheets["may_2026"]["exact_final_value_correct_count"]
        ),
    }
    return {
        "calibrated_geometry_acceptable": all(checks.values()),
        "checks": checks,
        "meaningful_improvement_definition": (
            "At least one additional exact final value among the 18 selected "
            "cells for each of April and May, with all other safety gates met."
        ),
        "notice": (
            "This assessment is benchmark evidence only. It does not change "
            "the production fixed-geometry default."
        ),
    }


def assess_hybrid_safety(
    fixed_summary: Mapping[str, Any],
    calibrated_summary: Mapping[str, Any],
    hybrid_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the declared Stage 3B production-activation gates."""

    fixed_overall = fixed_summary["overall"]
    hybrid_overall = hybrid_summary["overall"]
    fixed_sheets = fixed_summary["breakdowns"]["sheet_id"]
    calibrated_sheets = calibrated_summary["breakdowns"]["sheet_id"]
    hybrid_sheets = hybrid_summary["breakdowns"]["sheet_id"]
    april_fixed = fixed_sheets["april_2026"]["exact_final_value_correct_count"]
    april_calibrated = calibrated_sheets["april_2026"][
        "exact_final_value_correct_count"
    ]
    april_hybrid = hybrid_sheets["april_2026"][
        "exact_final_value_correct_count"
    ]
    possible_april_gain = max(0, april_calibrated - april_fixed)
    retained_april_gain = max(0, april_hybrid - april_fixed)
    april_gain_retention = (
        retained_april_gain / possible_april_gain
        if possible_april_gain
        else 1.0
    )
    checks = {
        "zero_new_unsafe_automatic_acceptances_relative_to_fixed": (
            hybrid_overall["unsafe_automatic_acceptances"]
            <= fixed_overall["unsafe_automatic_acceptances"]
        ),
        "sample_accuracy_at_least_14_of_18": (
            hybrid_sheets["sample"]["exact_final_value_correct_count"] >= 14
        ),
        "blank_precision_remains_100_percent": (
            hybrid_overall["blank_precision"] == 1.0
        ),
        "blank_recall_exceeds_7_of_16": (
            hybrid_overall["expected_blank_count"] == 16
            and hybrid_overall["predicted_blank_count"] > 7
            and hybrid_overall["blank_recall"] > 7 / 16
        ),
        "april_gains_substantially_retained": april_gain_retention >= 0.80,
        "may_has_no_additional_unsafe_automatic_acceptances": (
            hybrid_sheets["may_2026"]["unsafe_automatic_acceptances"]
            <= fixed_sheets["may_2026"]["unsafe_automatic_acceptances"]
        ),
        "crop_failure_proxy_does_not_increase": (
            hybrid_overall["crop_alignment_failure_count"]
            <= fixed_overall["crop_alignment_failure_count"]
        ),
    }
    return {
        "all_production_gates_passed": all(checks.values()),
        "checks": checks,
        "april_gain_retention": round(april_gain_retention, 6),
        "known_undetectable_geometry_error": (
            "A shared fixed/calibrated OCR error such as 42.4 read as 42.9 "
            "cannot be detected by geometry selection. Stage 3B contains it "
            "only because calibrated-selected results remain confirmation-required."
        ),
        "production_activation_recommended": False,
        "notice": (
            "Passing this limited benchmark is necessary but not sufficient "
            "to change the production fixed-geometry default."
        ),
    }
