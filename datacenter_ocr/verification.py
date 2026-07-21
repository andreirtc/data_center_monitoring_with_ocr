from __future__ import annotations

from dataclasses import dataclass, replace
from math import isnan
from statistics import median
from typing import Any, Protocol, TypeVar
import re


TEMPERATURE_MIN = 10.0
TEMPERATURE_MAX = 50.0
HUMIDITY_MIN = 0.0
HUMIDITY_MAX = 100.0

MINIMUM_OCR_CONFIDENCE = 0.85
EXPECTED_OCR_VARIANT_COUNT = 3
MINOR_OCR_DISAGREEMENT_MAXIMUM_DELTA = 0.1

TEMPERATURE_ANOMALY_MINIMUM_DELTA = 4.0
HUMIDITY_ANOMALY_MINIMUM_DELTA = 12.0
ANOMALY_MAD_MULTIPLIER = 3.5
MINIMUM_ANOMALY_CONTEXT_SIZE = 4

FORMAT_CATEGORY = "format"
RANGE_CATEGORY = "absolute_range"
OCR_CATEGORY = "ocr_uncertainty"
OPERATIONAL_CATEGORY = "operational_warning"
ANOMALY_CATEGORY = "anomaly"
BLANK_MISMATCH_CATEGORY = "blank_mismatch"

EXACT_READING_PATTERN = re.compile(r"-?\d+\.\d")


@dataclass(frozen=True)
class ValueValidation:
    """Validation facts for one final temperature or humidity value."""

    text: str
    normalized_value: str | None
    numeric_value: float | None
    is_blank: bool
    format_is_valid: bool
    within_absolute_limits: bool
    error: str | None


class VerifiableCellResult(Protocol):
    """Fields required by the verification engine."""

    filename: str
    day: int
    point: int
    reading_type: str
    predictions: dict[str, str]
    raw_predictions: dict[str, str]
    agreement_count: int
    average_consensus_confidence: float
    final_value: str
    is_blank: bool
    human_verified: bool
    ocr_uncertainty_reasons: tuple[str, ...]


CellResultT = TypeVar("CellResultT", bound=VerifiableCellResult)


def _clean_text(value: object) -> str:
    """Convert a widget or dataframe value into stable display text."""

    if value is None:
        return ""

    if isinstance(value, float) and isnan(value):
        return ""

    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _absolute_limits(reading_type: str) -> tuple[float, float]:
    if reading_type == "temperature":
        return TEMPERATURE_MIN, TEMPERATURE_MAX
    if reading_type == "humidity":
        return HUMIDITY_MIN, HUMIDITY_MAX
    raise ValueError(f"Unknown reading type: {reading_type}")


def validate_reading_value(
    value: object,
    reading_type: str,
    *,
    allow_blank: bool = True,
) -> ValueValidation:
    """Validate the exact one-decimal form format and absolute limits.

    The function intentionally does not repair commas, missing digits,
    letters, or excess precision. Ambiguous text must reach a reviewer.
    """

    text = _clean_text(value)

    if not text:
        error = None if allow_blank else "Enter a value or mark the cell as blank."
        return ValueValidation(
            text="",
            normalized_value=None,
            numeric_value=None,
            is_blank=True,
            format_is_valid=allow_blank,
            within_absolute_limits=allow_blank,
            error=error,
        )

    if EXACT_READING_PATTERN.fullmatch(text) is None:
        return ValueValidation(
            text=text,
            normalized_value=None,
            numeric_value=None,
            is_blank=False,
            format_is_valid=False,
            within_absolute_limits=False,
            error="Use exactly one decimal place, such as 22.0 or 53.3.",
        )

    numeric_value = float(text)
    minimum, maximum = _absolute_limits(reading_type)
    within_limits = minimum <= numeric_value <= maximum

    if not within_limits:
        return ValueValidation(
            text=text,
            normalized_value=None,
            numeric_value=numeric_value,
            is_blank=False,
            format_is_valid=True,
            within_absolute_limits=False,
            error=f"The value must be within {minimum:.1f} to {maximum:.1f}.",
        )

    return ValueValidation(
        text=text,
        normalized_value=f"{numeric_value:.1f}",
        numeric_value=numeric_value,
        is_blank=False,
        format_is_valid=True,
        within_absolute_limits=True,
        error=None,
    )


def temperature_severity(value: float | None) -> str:
    """Classify a valid temperature using the monitoring-form bands."""

    if value is None:
        return "not applicable"
    if value < 25.0:
        return "normal"
    if value < 30.0:
        return "alert"
    if value <= 35.0:
        return "alarming"
    return "critical"


def _material_ocr_reasons(result: VerifiableCellResult) -> list[str]:
    """Return OCR concerns that require a person to confirm the value."""

    reasons = list(result.ocr_uncertainty_reasons)

    if result.agreement_count < EXPECTED_OCR_VARIANT_COUNT:
        reasons.append(
            f"OCR variants disagree ({result.agreement_count}/"
            f"{EXPECTED_OCR_VARIANT_COUNT} agreement)."
        )

    if result.average_consensus_confidence < MINIMUM_OCR_CONFIDENCE:
        reasons.append(
            "OCR confidence is low "
            f"({result.average_consensus_confidence:.1%})."
        )

    for variant, raw_text in result.raw_predictions.items():
        normalized_text = result.predictions.get(variant, "")
        raw_text = raw_text.strip()

        if raw_text and raw_text != normalized_text:
            reasons.append(
                f"The {variant} OCR text {raw_text!r} required character normalization."
            )
            break

    return list(dict.fromkeys(reason for reason in reasons if reason))


def _is_minor_high_confidence_disagreement(
    result: VerifiableCellResult,
    validation: ValueValidation,
) -> bool:
    """Recognize a harmless 2/3 disagreement near the accepted value."""

    if (
        result.agreement_count != 2
        or result.average_consensus_confidence < MINIMUM_OCR_CONFIDENCE
        or validation.error is not None
        or validation.numeric_value is None
        or result.ocr_uncertainty_reasons
        or len(result.predictions) != EXPECTED_OCR_VARIANT_COUNT
    ):
        return False

    for variant, raw_text in result.raw_predictions.items():
        if raw_text.strip() != result.predictions.get(variant, ""):
            return False

    prediction_values: list[float] = []
    for prediction in result.predictions.values():
        try:
            prediction_values.append(float(prediction))
        except ValueError:
            return False

    accepted_value = validation.numeric_value
    matching_count = sum(
        abs(value - accepted_value) < 1e-9 for value in prediction_values
    )
    if matching_count != 2:
        return False

    return all(
        abs(value - accepted_value)
        <= MINOR_OCR_DISAGREEMENT_MAXIMUM_DELTA + 1e-9
        for value in prediction_values
    )


def _contextual_anomaly(
    result: VerifiableCellResult,
    numeric_values: dict[str, float],
    results: list[CellResultT],
) -> tuple[bool, str | None]:
    current_value = numeric_values.get(result.filename)
    if current_value is None:
        return False, None

    context_values = [
        numeric_values[peer.filename]
        for peer in results
        if (
            peer.filename in numeric_values
            and peer.filename != result.filename
            and peer.reading_type == result.reading_type
            and (
                peer.day == result.day
                or (
                    peer.point == result.point
                    and abs(peer.day - result.day) <= 2
                )
            )
        )
    ]

    if len(context_values) < MINIMUM_ANOMALY_CONTEXT_SIZE:
        return False, None

    context_median = float(median(context_values))
    absolute_deviations = [
        abs(value - context_median) for value in context_values
    ]
    median_absolute_deviation = float(median(absolute_deviations))
    minimum_delta = (
        TEMPERATURE_ANOMALY_MINIMUM_DELTA
        if result.reading_type == "temperature"
        else HUMIDITY_ANOMALY_MINIMUM_DELTA
    )
    threshold = max(
        minimum_delta,
        ANOMALY_MAD_MULTIPLIER * 1.4826 * median_absolute_deviation,
    )
    delta = abs(current_value - context_median)

    if delta <= threshold:
        return False, None

    return (
        True,
        "Statistical anomaly: "
        f"{current_value:.1f} differs from the local median "
        f"{context_median:.1f} by {delta:.1f}.",
    )


def verify_cell_results(results: list[CellResultT]) -> list[CellResultT]:
    """Recompute all derived verification state for a complete result set."""

    validations: dict[str, ValueValidation] = {}
    numeric_values: dict[str, float] = {}

    for result in results:
        validation = validate_reading_value(
            "" if result.is_blank else result.final_value,
            result.reading_type,
            allow_blank=result.is_blank,
        )
        validations[result.filename] = validation
        if validation.error is None and validation.numeric_value is not None:
            numeric_values[result.filename] = validation.numeric_value

    paired_results = {
        (result.day, result.point, result.reading_type): result
        for result in results
    }
    updated_results: list[CellResultT] = []

    for result in results:
        validation = validations[result.filename]
        blocking_errors: list[str] = []
        confirmation_reasons: list[str] = []
        operational_warnings: list[str] = []
        informational_notices: list[str] = []
        categories: list[str] = []

        if not result.is_blank and validation.is_blank:
            categories.append(FORMAT_CATEGORY)
            blocking_errors.append("The cell has no value and is not marked blank.")
        elif not validation.format_is_valid:
            categories.append(FORMAT_CATEGORY)
            blocking_errors.append(
                f"Malformed final value {validation.text!r}: {validation.error}"
            )
        elif not validation.within_absolute_limits:
            categories.append(RANGE_CATEGORY)
            blocking_errors.append(
                f"Value outside absolute limits: {validation.error}"
            )

        minor_ocr_disagreement = (
            False
            if result.is_blank
            else _is_minor_high_confidence_disagreement(result, validation)
        )
        material_ocr_reasons = (
            [] if result.is_blank else _material_ocr_reasons(result)
        )
        if minor_ocr_disagreement:
            categories.append(OCR_CATEGORY)
            informational_notices.append(
                "Minor high-confidence OCR disagreement; two variants support "
                f"the accepted value {validation.normalized_value}."
            )
            material_ocr_reasons = []
        elif material_ocr_reasons:
            categories.append(OCR_CATEGORY)
            confirmation_reasons.extend(material_ocr_reasons)

        severity = "not applicable"
        if result.reading_type == "temperature":
            severity = temperature_severity(validation.numeric_value)
            if severity in {"alert", "alarming", "critical"}:
                categories.append(OPERATIONAL_CATEGORY)
                operational_warnings.append(
                    f"Operational temperature status is {severity} "
                    f"at {validation.numeric_value:.1f} degrees C."
                )

        is_anomaly, anomaly_reason = _contextual_anomaly(
            result,
            numeric_values,
            results,
        )
        if is_anomaly and anomaly_reason is not None:
            categories.append(ANOMALY_CATEGORY)
            confirmation_reasons.append(anomaly_reason)

        counterpart_type = (
            "humidity" if result.reading_type == "temperature" else "temperature"
        )
        counterpart = paired_results.get((result.day, result.point, counterpart_type))
        has_blank_mismatch = (
            counterpart is not None and result.is_blank != counterpart.is_blank
        )
        if has_blank_mismatch:
            categories.append(BLANK_MISMATCH_CATEGORY)
            blocking_errors.append(
                "Temperature and humidity blank states do not match for this row."
            )

        categories = list(dict.fromkeys(categories))
        blocking_errors = list(dict.fromkeys(blocking_errors))
        confirmation_reasons = list(dict.fromkeys(confirmation_reasons))
        operational_warnings = list(dict.fromkeys(operational_warnings))
        informational_notices = list(dict.fromkeys(informational_notices))
        unresolved_confirmation = bool(confirmation_reasons) and not (
            result.human_verified
        )
        blocks_export = bool(blocking_errors) or unresolved_confirmation
        needs_review = blocks_export
        reasons = list(
            dict.fromkeys(
                blocking_errors
                + confirmation_reasons
                + operational_warnings
                + informational_notices
            )
        )

        if reasons:
            review_reason = " ".join(reasons)
        elif result.human_verified:
            review_reason = "Human verified; no outstanding verification issues."
        elif result.is_blank:
            review_reason = "Cell classified as blank."
        else:
            review_reason = "No verification issues detected."

        updated_results.append(
            replace(
                result,
                final_value=(
                    ""
                    if result.is_blank
                    else (
                        validation.normalized_value
                        if validation.error is None
                        and validation.normalized_value is not None
                        else result.final_value
                    )
                ),
                needs_review=needs_review,
                review_reason=review_reason,
                verification_reasons=tuple(reasons),
                review_categories=tuple(categories),
                blocking_errors=tuple(blocking_errors),
                required_confirmation_reasons=tuple(confirmation_reasons),
                operational_warnings=tuple(operational_warnings),
                informational_notices=tuple(informational_notices),
                blocks_export=blocks_export,
                format_is_valid=validation.format_is_valid,
                within_absolute_limits=validation.within_absolute_limits,
                operational_severity=severity,
                is_statistical_anomaly=is_anomaly,
                has_blank_mismatch=has_blank_mismatch,
                ocr_uncertain=bool(material_ocr_reasons or minor_ocr_disagreement),
            )
        )

    return updated_results


def validate_export_reading(
    value: Any,
    reading_type: str,
    day: int,
    point: int,
) -> float | None:
    """Validate one export value with the same rules used by the UI."""

    validation = validate_reading_value(value, reading_type, allow_blank=True)
    if validation.error is not None:
        raise ValueError(
            f"Day {day}, Point {point}, {reading_type} has an invalid value "
            f"{value!r}: {validation.error}"
        )
    return validation.numeric_value
