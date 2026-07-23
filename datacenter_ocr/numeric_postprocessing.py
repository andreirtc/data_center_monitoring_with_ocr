from __future__ import annotations

from dataclasses import dataclass
import re

from datacenter_ocr.verification import (
    HUMIDITY_MAX,
    HUMIDITY_MIN,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    validate_reading_value,
)

DECIMAL_PLACES = 1
THREE_ASCII_DIGITS_PATTERN = re.compile(r"[0-9]{3}")
FOUR_OR_MORE_ASCII_DIGITS_PATTERN = re.compile(r"[0-9]{4,}")


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
    status: str = "unknown"
    candidate_interpretations: tuple[str, ...] = ()


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

    validation = validate_reading_value(
        value,
        reading_type,
        allow_blank=True,
    )
    return validation.normalized_value, validation.error

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

    if re.fullmatch(r"[0-9]+", text) is None:
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

    validation = validate_reading_value(
        candidate,
        reading_type,
        allow_blank=False,
    )

    if validation.error is None:
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
            status="empty_prediction",
        )

    # Excess precision is ambiguous. Keep it visible for a reviewer
    # instead of silently discarding a possibly meaningful digit.
    precision_adjusted = trim_extra_decimal_digits(
        original_text
    )

    if precision_adjusted != original_text:
        return CorrectionResult(
            original_text=original_text,
            corrected_text=original_text,
            changed=False,
            needs_review=True,
            reason=(
                "Too many digits or decimal places; verify the handwritten "
                "value. No digit was discarded automatically."
            ),
            status="extra_decimal_digits",
            candidate_interpretations=(precision_adjusted,),
        )

    validation = validate_reading_value(
        original_text,
        reading_type,
        allow_blank=False,
    )

    if validation.error is None:
        return CorrectionResult(
            original_text=original_text,
            corrected_text=original_text,
            changed=False,
            needs_review=False,
            reason="Prediction already follows the expected format.",
            status="valid_unchanged",
            candidate_interpretations=(original_text,),
        )

    # Monitoring-sheet readings are written with one decimal place. For an
    # exact three-digit ASCII OCR result, the operator-approved business rule
    # gives DD.D precedence over broader leading/trailing-artifact guesses.
    if THREE_ASCII_DIGITS_PATTERN.fullmatch(original_text):
        inferred_value = insert_decimal_before_last_digit(original_text)
        inferred_validation = validate_reading_value(
            inferred_value,
            reading_type,
            allow_blank=False,
        )
        if (
            inferred_validation.error is None
            and inferred_validation.normalized_value is not None
        ):
            corrected_text = inferred_validation.normalized_value
            return CorrectionResult(
                original_text=original_text,
                corrected_text=corrected_text,
                changed=True,
                needs_review=True,
                reason=(
                    "Decimal point inferred from three-digit OCR text. "
                    "Human verification is required."
                ),
                status="decimal_inferred",
                candidate_interpretations=(corrected_text,),
            )

        return CorrectionResult(
            original_text=original_text,
            corrected_text=original_text,
            changed=False,
            needs_review=True,
            reason=(
                "The inferred decimal value is outside the allowed "
                f"{reading_type} range; verify the handwritten value."
            ),
            status="three_digit_inference_invalid",
            candidate_interpretations=(),
        )

    # A decimal point with a missing or malformed digit cannot be
    # repaired safely by moving or inserting another decimal point.
    if "." in original_text:
        return CorrectionResult(
            original_text=original_text,
            corrected_text=original_text,
            changed=False,
            needs_review=True,
            reason=validation.error or "Malformed numeric prediction.",
            status="malformed_with_decimal",
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

    # Four-or-more digit OCR results are never resolved automatically. The
    # candidates above remain useful evidence for the operator, but deleting
    # a leading or trailing digit is too ambiguous to prefill as final.
    if FOUR_OR_MORE_ASCII_DIGITS_PATTERN.fullmatch(original_text):
        return CorrectionResult(
            original_text=original_text,
            corrected_text=original_text,
            changed=False,
            needs_review=True,
            reason=(
                "Too many digits or decimal places; verify the handwritten "
                "value."
            ),
            status="too_many_digits",
            candidate_interpretations=tuple(sorted(candidates)),
        )


    if len(candidates) == 1:
        corrected_text, reason = next(
            iter(candidates.items())
        )

        return CorrectionResult(
            original_text=original_text,
            corrected_text=corrected_text,
            changed=True,
            needs_review=True,
            reason=(
                f"Proposed {corrected_text} after OCR repair. "
                f"{reason} Human verification is required."
            ),
            status="proposed_correction",
            candidate_interpretations=(corrected_text,),
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
            status="multiple_interpretations",
            candidate_interpretations=tuple(sorted(candidates)),
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
        status="no_safe_correction",
    )
