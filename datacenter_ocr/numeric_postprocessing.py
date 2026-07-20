from __future__ import annotations

from dataclasses import dataclass

import re


TEMPERATURE_MIN = 10.0
TEMPERATURE_MAX = 50.0

HUMIDITY_MIN = 0.0
HUMIDITY_MAX = 100.0

DECIMAL_PLACES = 1


@dataclass(frozen=True)
class CorrectionResult:
    """
    Result of applying monitoring-form rules to one OCR prediction.
    """

    original_text: str
    corrected_text: str
    changed: bool
    needs_review: bool
    reason: str


def infer_reading_type(filename: str) -> str:
    """
    Determine whether a crop contains temperature or humidity
    based on its descriptive filename.
    """

    lowercase_filename = filename.lower()

    if "temperature" in lowercase_filename:
        return "temperature"

    if "humidity" in lowercase_filename:
        return "humidity"

    raise ValueError(
        f"Could not determine reading type from filename: {filename}"
    )


def is_valid_value(
    text: str,
    reading_type: str,
) -> bool:
    """
    Check whether text can be parsed and falls inside
    the accepted numeric range.
    """

    try:
        value = float(text)
    except ValueError:
        return False

    if reading_type == "temperature":
        return TEMPERATURE_MIN <= value <= TEMPERATURE_MAX

    if reading_type == "humidity":
        return HUMIDITY_MIN <= value <= HUMIDITY_MAX

    raise ValueError(
        f"Unknown reading type: {reading_type}"
    )

def validate_final_reading(
    value: object,
    reading_type: str,
) -> tuple[str | None, str | None]:
    """
    Validate a final reading before display or export.

    Returns:
        normalized value and no error, or
        no value and an error message.
    """

    if value is None:
        return None, None

    text = (
        str(value)
        .strip()
        .replace(",", ".")
    )

    # Pandas may represent empty table cells as NaN.
    if (
        not text
        or text.lower() == "nan"
    ):
        return None, None

    # The company form requires exactly one decimal digit.
    if re.fullmatch(
        r"-?\d+\.\d",
        text,
    ) is None:
        return (
            None,
            "Use exactly one decimal place, "
            "such as 22.0 or 53.3.",
        )

    normalized_value = (
        f"{float(text):.1f}"
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
            f"The value must be within {valid_range}.",
        )

    return normalized_value, None

def trim_extra_decimal_digits(
    text: str,
) -> str:
    """
    Keep only the configured number of decimal digits.

    Example:
        59.51 becomes 59.5
    """

    if "." not in text:
        return text

    whole_part, decimal_part = text.split(
        ".",
        maxsplit=1,
    )

    if len(decimal_part) <= DECIMAL_PLACES:
        return text

    trimmed_decimal = decimal_part[
        :DECIMAL_PLACES
    ]

    return (
        f"{whole_part}."
        f"{trimmed_decimal}"
    )


def insert_decimal_before_last_digit(
    text: str,
) -> str | None:
    """
    Convert a digit-only OCR result into a one-decimal value.

    Example:
        217 becomes 21.7
    """

    if not text.isdigit():
        return None

    if len(text) < 2:
        return None

    return (
        f"{text[:-1]}."
        f"{text[-1]}"
    )


def add_candidate(
    candidates: dict[str, str],
    candidate: str | None,
    reason: str,
    reading_type: str,
) -> None:
    """
    Add a candidate only when it is valid for the reading type.
    """

    if candidate is None:
        return

    candidate = trim_extra_decimal_digits(
        candidate
    )

    if is_valid_value(
        candidate,
        reading_type,
    ):
        candidates[candidate] = reason


def correct_numeric_prediction(
    prediction: str,
    reading_type: str,
) -> CorrectionResult:
    """
    Apply conservative repairs to a normalized OCR prediction.

    Automatic correction occurs only when one valid interpretation exists.
    Multiple possible interpretations are flagged for manual review.
    """

    original_text = prediction.strip()

    if not original_text:
        return CorrectionResult(
            original_text=original_text,
            corrected_text="",
            changed=False,
            needs_review=True,
            reason="OCR returned an empty value.",
        )

    # First handle excessive precision.
    precision_adjusted = trim_extra_decimal_digits(
        original_text
    )

    if is_valid_value(
        precision_adjusted,
        reading_type,
    ):
        if precision_adjusted != original_text:
            return CorrectionResult(
                original_text=original_text,
                corrected_text=precision_adjusted,
                changed=True,
                needs_review=False,
                reason=(
                    "Removed extra decimal digits because "
                    "the form uses one decimal place."
                ),
            )

        return CorrectionResult(
            original_text=original_text,
            corrected_text=original_text,
            changed=False,
            needs_review=False,
            reason="Prediction already follows the expected format.",
        )

    candidates: dict[str, str] = {}

    # Example: 120.0 -> 20.0
    if len(precision_adjusted) > 1:
        without_first_character = (
            precision_adjusted[1:]
        )

        add_candidate(
            candidates=candidates,
            candidate=without_first_character,
            reason=(
                "Removed one leading OCR artifact because "
                "the original value was outside the valid range."
            ),
            reading_type=reading_type,
        )

    # Example: 217 -> 21.7
    decimal_candidate = (
        insert_decimal_before_last_digit(
            precision_adjusted
        )
    )

    add_candidate(
        candidates=candidates,
        candidate=decimal_candidate,
        reason=(
            "Inserted the missing decimal point before "
            "the final digit."
        ),
        reading_type=reading_type,
    )

    # Example: 1217 -> 217 -> 21.7
    if len(precision_adjusted) > 1:
        without_first_character = (
            precision_adjusted[1:]
        )

        combined_candidate = (
            insert_decimal_before_last_digit(
                without_first_character
            )
        )

        add_candidate(
            candidates=candidates,
            candidate=combined_candidate,
            reason=(
                "Removed one leading OCR artifact and "
                "inserted a missing decimal point."
            ),
            reading_type=reading_type,
        )

    # Example: 5331 -> 533 -> 53.3
    #
    # The final character may be an OCR artifact created by
    # a printed border, nearby stroke, or duplicated digit.
    if len(precision_adjusted) > 1:
        without_last_character = (
            precision_adjusted[:-1]
        )

        trailing_artifact_candidate = (
            insert_decimal_before_last_digit(
                without_last_character
            )
        )

        add_candidate(
            candidates=candidates,
            candidate=trailing_artifact_candidate,
            reason=(
                "Removed one trailing OCR artifact and "
                "inserted a missing decimal point."
            ),
            reading_type=reading_type,
        )


    if len(candidates) == 1:
        corrected_text, reason = next(
            iter(candidates.items())
        )

        return CorrectionResult(
            original_text=original_text,
            corrected_text=corrected_text,
            changed=True,
            needs_review=False,
            reason=reason,
        )

    if len(candidates) > 1:
        candidate_text = ", ".join(
            sorted(candidates)
        )

        return CorrectionResult(
            original_text=original_text,
            corrected_text=original_text,
            changed=False,
            needs_review=True,
            reason=(
                "Multiple valid interpretations exist: "
                f"{candidate_text}"
            ),
        )

    return CorrectionResult(
        original_text=original_text,
        corrected_text=original_text,
        changed=False,
        needs_review=True,
        reason=(
            "No safe correction produced a valid "
            f"{reading_type} value."
        ),
    )