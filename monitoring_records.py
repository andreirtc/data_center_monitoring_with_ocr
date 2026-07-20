from __future__ import annotations

from typing import Any

from ocr_processing import CellOCRResult


EXPECTED_DAY_COUNT = 31
EXPECTED_POINT_COUNT = 8
EXPECTED_ROW_COUNT = 248


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

            row_needs_review = (
                temperature.needs_review
                or humidity.needs_review
            )

            monitoring_rows.append(
                {
                    "day": day,
                    "point": point,
                    "temperature": temperature.final_value,
                    "humidity": humidity.final_value,
                    "needs_review": row_needs_review,

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