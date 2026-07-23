from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


DIAGNOSTIC_CELL_RESULT_FIELDS = [
    "filename",
    "day",
    "point",
    "reading_type",
    "is_blank",
    "blank_ink_ratio",
    "blank_component_count",
    "blank_largest_component_ratio",
    "blank_largest_component_width_ratio",
    "blank_largest_component_height_ratio",
    "blank_largest_component_aspect_ratio",
    "raw_predictions_json",
    "normalized_predictions_json",
    "confidences_json",
    "agreement_count",
    "average_consensus_confidence",
    "consensus_prediction",
    "proposed_final_value",
    "postprocessing_status",
    "candidate_interpretations_json",
    "ocr_uncertainty_reasons_json",
    "verification_reasons_json",
    "blocking_errors_json",
    "required_confirmation_reasons_json",
    "operational_warnings_json",
    "informational_notices_json",
    "review_categories_json",
    "human_verified",
    "blocks_export",
    "needs_review",
    "review_reason",
    "format_is_valid",
    "within_absolute_limits",
    "operational_severity",
    "is_statistical_anomaly",
    "has_blank_mismatch",
    "ocr_uncertain",
]


def _json_mapping(value: dict[str, Any]) -> str:
    """Serialize variant-keyed diagnostic values consistently."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_sequence(value: Iterable[Any]) -> str:
    """Serialize ordered diagnostic reasons consistently."""

    return json.dumps(
        list(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def cell_result_to_diagnostic_row(result: Any) -> dict[str, Any]:
    """Convert one CellOCRResult into a complete diagnostic CSV row."""

    return {
        "filename": result.filename,
        "day": result.day,
        "point": result.point,
        "reading_type": result.reading_type,
        "is_blank": result.is_blank,
        "blank_ink_ratio": result.blank_ink_ratio,
        "blank_component_count": result.blank_component_count,
        "blank_largest_component_ratio": result.blank_largest_component_ratio,
        "blank_largest_component_width_ratio": (
            result.blank_largest_component_width_ratio
        ),
        "blank_largest_component_height_ratio": (
            result.blank_largest_component_height_ratio
        ),
        "blank_largest_component_aspect_ratio": (
            result.blank_largest_component_aspect_ratio
        ),
        "raw_predictions_json": _json_mapping(result.raw_predictions),
        "normalized_predictions_json": _json_mapping(result.predictions),
        "confidences_json": _json_mapping(result.confidences),
        "agreement_count": result.agreement_count,
        "average_consensus_confidence": result.average_consensus_confidence,
        "consensus_prediction": result.consensus_prediction,
        "proposed_final_value": result.final_value,
        "postprocessing_status": result.postprocessing_status,
        "candidate_interpretations_json": _json_sequence(
            result.candidate_interpretations
        ),
        "ocr_uncertainty_reasons_json": _json_sequence(
            result.ocr_uncertainty_reasons
        ),
        "verification_reasons_json": _json_sequence(result.verification_reasons),
        "blocking_errors_json": _json_sequence(result.blocking_errors),
        "required_confirmation_reasons_json": _json_sequence(
            result.required_confirmation_reasons
        ),
        "operational_warnings_json": _json_sequence(result.operational_warnings),
        "informational_notices_json": _json_sequence(
            result.informational_notices
        ),
        "review_categories_json": _json_sequence(result.review_categories),
        "human_verified": result.human_verified,
        "blocks_export": result.blocks_export,
        "needs_review": result.needs_review,
        "review_reason": result.review_reason,
        "format_is_valid": result.format_is_valid,
        "within_absolute_limits": result.within_absolute_limits,
        "operational_severity": result.operational_severity,
        "is_statistical_anomaly": result.is_statistical_anomaly,
        "has_blank_mismatch": result.has_blank_mismatch,
        "ocr_uncertain": result.ocr_uncertain,
    }


def write_cell_results_csv(
    results: list[Any],
    output_path: Path,
) -> None:
    """Write complete, machine-readable CellOCRResult diagnostics."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=DIAGNOSTIC_CELL_RESULT_FIELDS,
        )
        writer.writeheader()
        writer.writerows(
            cell_result_to_diagnostic_row(result)
            for result in results
        )
