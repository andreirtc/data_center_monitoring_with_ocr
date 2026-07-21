from __future__ import annotations

from typing import Any

from datacenter_ocr.ocr_processing import CellOCRResult


EXPECTED_DAY_COUNT = 31
EXPECTED_POINT_COUNT = 8
EXPECTED_ROW_COUNT = 248


def attach_crop_data_urls(
    monitoring_rows: list[dict[str, Any]],
    crop_data_urls: dict[str, str],
) -> list[dict[str, Any]]:
    """Attach reading crops by each cell's stable OCR-result filename."""

    rows_with_crops: list[dict[str, Any]] = []
    for row in monitoring_rows:
        temperature_filename = str(row["temperature_filename"])
        humidity_filename = str(row["humidity_filename"])

        try:
            temperature_crop = crop_data_urls[temperature_filename]
            humidity_crop = crop_data_urls[humidity_filename]
        except KeyError as error:
            raise ValueError(
                f"Missing extracted crop for {error.args[0]}."
            ) from error

        rows_with_crops.append(
            {
                **row,
                "temperature_crop": temperature_crop,
                "humidity_crop": humidity_crop,
            }
        )

    return rows_with_crops


def build_monitoring_rows(
    cell_results: list[CellOCRResult],
) -> list[dict[str, Any]]:
    """
    Combine temperature and humidity cell results into
    one row per day and monitoring point.
    """

    grouped_results: dict[
        tuple[int, int],
        dict[str, CellOCRResult],
    ] = {}

    for result in cell_results:
        key = (
            result.day,
            result.point,
        )

        if key not in grouped_results:
            grouped_results[key] = {}

        reading_group = grouped_results[key]

        if result.reading_type in reading_group:
            raise ValueError(
                "Duplicate reading found for "
                f"Day {result.day}, "
                f"Point {result.point}, "
                f"{result.reading_type}."
            )

        reading_group[
            result.reading_type
        ] = result

    monitoring_rows = []

    for day in range(
        1,
        EXPECTED_DAY_COUNT + 1,
    ):
        for point in range(
            1,
            EXPECTED_POINT_COUNT + 1,
        ):
            key = (
                day,
                point,
            )

            if key not in grouped_results:
                raise ValueError(
                    "Missing readings for "
                    f"Day {day}, Point {point}."
                )

            reading_group = grouped_results[
                key
            ]

            if "temperature" not in reading_group:
                raise ValueError(
                    "Missing temperature reading for "
                    f"Day {day}, Point {point}."
                )

            if "humidity" not in reading_group:
                raise ValueError(
                    "Missing humidity reading for "
                    f"Day {day}, Point {point}."
                )

            temperature = reading_group[
                "temperature"
            ]

            humidity = reading_group[
                "humidity"
            ]

            row_blocks_export = (
                temperature.blocks_export
                or humidity.blocks_export
            )
            row_needs_review = row_blocks_export

            row_categories = tuple(
                dict.fromkeys(
                    temperature.review_categories
                    + humidity.review_categories
                )
            )

            row_reasons = tuple(
                dict.fromkeys(
                    temperature.verification_reasons
                    + humidity.verification_reasons
                )
            )

            row_blocking_errors = tuple(
                dict.fromkeys(
                    temperature.blocking_errors
                    + humidity.blocking_errors
                )
            )
            row_confirmation_reasons = tuple(
                dict.fromkeys(
                    (
                        ()
                        if temperature.human_verified
                        else temperature.required_confirmation_reasons
                    )
                    + (
                        ()
                        if humidity.human_verified
                        else humidity.required_confirmation_reasons
                    )
                )
            )
            row_operational_warnings = tuple(
                dict.fromkeys(
                    temperature.operational_warnings
                    + humidity.operational_warnings
                )
            )
            row_informational_notices = tuple(
                dict.fromkeys(
                    temperature.informational_notices
                    + humidity.informational_notices
                )
            )

            monitoring_rows.append(
                {
                    "day": day,
                    "point": point,
                    "temperature": temperature.final_value,
                    "humidity": humidity.final_value,
                    "needs_review": row_needs_review,
                    "blocks_export": row_blocks_export,
                    "blocking_errors": row_blocking_errors,
                    "required_confirmation_reasons": row_confirmation_reasons,
                    "operational_warnings": row_operational_warnings,
                    "informational_notices": row_informational_notices,
                    "status": (
                        "Blocked"
                        if row_blocks_export
                        else (
                            "Operational warning"
                            if row_operational_warnings
                            else (
                                "Ready with notice"
                                if row_informational_notices
                                else "Ready"
                            )
                        )
                    ),
                    "status_reason": " ".join(row_reasons),
                    "review_categories": row_categories,
                    "has_ocr_uncertainty": (
                        temperature.ocr_uncertain
                        or humidity.ocr_uncertain
                    ),
                    "has_operational_warning": (
                        temperature.operational_severity
                        in {"alert", "alarming", "critical"}
                    ),
                    "has_anomaly": (
                        temperature.is_statistical_anomaly
                        or humidity.is_statistical_anomaly
                    ),
                    "has_blank_mismatch": (
                        temperature.has_blank_mismatch
                        or humidity.has_blank_mismatch
                    ),

                    "temperature_is_blank": (
                        temperature.is_blank
                    ),
                    "humidity_is_blank": (
                        humidity.is_blank
                    ),
                    "row_is_blank": (
                        temperature.is_blank
                        and humidity.is_blank
                    ),
                    "temperature_needs_review": (
                        temperature.needs_review
                    ),
                    "humidity_needs_review": (
                        humidity.needs_review
                    ),
                    "temperature_status": (
                        "Blocked"
                        if temperature.blocks_export
                        else "Operational warning"
                        if temperature.operational_warnings
                        else "Ready with notice"
                        if temperature.informational_notices
                        else "Verified"
                        if temperature.human_verified
                        else "Ready"
                    ),
                    "humidity_status": (
                        "Blocked"
                        if humidity.blocks_export
                        else "Operational warning"
                        if humidity.operational_warnings
                        else "Ready with notice"
                        if humidity.informational_notices
                        else "Verified"
                        if humidity.human_verified
                        else "Ready"
                    ),
                    "temperature_status_reason": temperature.review_reason,
                    "humidity_status_reason": humidity.review_reason,
                    "temperature_severity": temperature.operational_severity,
                    "temperature_human_verified": temperature.human_verified,
                    "humidity_human_verified": humidity.human_verified,

                    "temperature_prediction": (
                        temperature.consensus_prediction
                    ),
                    "humidity_prediction": (
                        humidity.consensus_prediction
                    ),

                    "temperature_agreement": (
                        temperature.agreement_count
                    ),
                    "humidity_agreement": (
                        humidity.agreement_count
                    ),

                    "temperature_confidence": (
                        temperature.average_consensus_confidence
                    ),
                    "humidity_confidence": (
                        humidity.average_consensus_confidence
                    ),

                    "temperature_review_reason": (
                        temperature.review_reason
                    ),
                    "humidity_review_reason": (
                        humidity.review_reason
                    ),

                    "temperature_filename": (
                        temperature.filename
                    ),
                    "humidity_filename": (
                        humidity.filename
                    ),
                    
                }
            )

    if len(monitoring_rows) != EXPECTED_ROW_COUNT:
        raise ValueError(
            "Expected exactly "
            f"{EXPECTED_ROW_COUNT} monitoring rows, "
            f"but generated {len(monitoring_rows)}."
        )

    return monitoring_rows
