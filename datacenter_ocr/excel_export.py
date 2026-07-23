from __future__ import annotations

from html import escape
from io import BytesIO
from pathlib import Path
import re
import zipfile
from typing import Any

from datacenter_ocr.config import (
    EXCEL_TEMPLATE_PATH,
)
from datacenter_ocr.verification import validate_export_reading


WORKSHEET_XML_PATH = (
    "xl/worksheets/sheet1.xml"
)

WORKBOOK_XML_PATH = (
    "xl/workbook.xml"
)

DATA_START_ROW = 7
EXPECTED_DAY_COUNT = 31
EXPECTED_POINT_COUNT = 8
EXPECTED_MONITORING_ROW_COUNT = 248


TEMPERATURE_COLUMNS = {
    1: "B",
    2: "D",
    3: "F",
    4: "H",
    5: "J",
    6: "L",
    7: "N",
    8: "P",
}

HUMIDITY_COLUMNS = {
    1: "C",
    2: "E",
    3: "G",
    4: "I",
    5: "K",
    6: "M",
    7: "O",
    8: "Q",
}


def excel_cell_address(
    day: int,
    point: int,
    reading_type: str,
) -> str:
    """Return the official-template address for one identified reading."""

    if not 1 <= day <= EXPECTED_DAY_COUNT:
        raise ValueError(f"Invalid day number: {day}.")
    if not 1 <= point <= EXPECTED_POINT_COUNT:
        raise ValueError(f"Invalid point number: {point}.")

    if reading_type == "temperature":
        column = TEMPERATURE_COLUMNS[point]
    elif reading_type == "humidity":
        column = HUMIDITY_COLUMNS[point]
    else:
        raise ValueError(f"Invalid reading type: {reading_type}.")

    return f"{column}{DATA_START_ROW + day - 1}"


def build_excel_mapping_audit(
    monitoring_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Describe all 496 filename-keyed source-to-workbook destinations.

    The same completeness and export-readiness checks used by workbook
    creation are applied so this audit cannot describe a partial export.
    """

    indexed_rows = _index_monitoring_rows(monitoring_rows)
    audit_rows: list[dict[str, Any]] = []

    for day in range(1, EXPECTED_DAY_COUNT + 1):
        for point in range(1, EXPECTED_POINT_COUNT + 1):
            monitoring_row = indexed_rows[(day, point)]
            for reading_type in ("temperature", "humidity"):
                audit_rows.append(
                    {
                        "day": day,
                        "point": point,
                        "reading_type": reading_type,
                        "source_filename": monitoring_row.get(
                            f"{reading_type}_filename",
                            "",
                        ),
                        "final_value": monitoring_row.get(reading_type, ""),
                        "excel_cell": excel_cell_address(
                            day,
                            point,
                            reading_type,
                        ),
                    }
                )

    expected_reading_count = EXPECTED_MONITORING_ROW_COUNT * 2
    if len(audit_rows) != expected_reading_count:
        raise RuntimeError(
            f"Expected {expected_reading_count} Excel mapping rows, "
            f"but generated {len(audit_rows)}."
        )
    return audit_rows


def _remove_cell_type_attribute(
    attributes: bytes,
) -> bytes:
    """
    Remove an existing Excel cell-type attribute.

    Numeric measurement cells should not be stored as text.
    """

    return re.sub(
        rb'\s+t="[^"]*"',
        b"",
        attributes,
    )


def _replace_existing_cell(
    worksheet_xml: bytes,
    cell_address: str,
    value: float | str | None,
    *,
    inline_string: bool = False,
) -> bytes:
    """
    Replace the contents of one existing Excel cell while
    preserving its original formatting attributes.

    The official template already contains all target cells,
    so a missing cell is treated as a template-layout error.
    """

    encoded_address = re.escape(
        cell_address.encode("ascii")
    )

    cell_pattern = re.compile(
        rb'<c\b'
        rb'(?=[^>]*\br="' + encoded_address + rb'")'
        rb'(?P<self_attributes>[^>]*?)/>'
        rb'|'
        rb'<c\b'
        rb'(?=[^>]*\br="' + encoded_address + rb'")'
        rb'(?P<full_attributes>[^>]*)>'
        rb'(?P<body>.*?)'
        rb'</c>',
        re.DOTALL,
    )

    matches = list(
        cell_pattern.finditer(
            worksheet_xml
        )
    )

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one Excel cell "
            f"for {cell_address}, but found "
            f"{len(matches)}. The workbook template "
            f"may have been changed."
        )

    match = matches[0]

    attributes = (
        match.group("self_attributes")
        if match.group("self_attributes")
        is not None
        else match.group("full_attributes")
    )

    if attributes is None:
        raise RuntimeError(
            f"Could not read cell attributes "
            f"for {cell_address}."
        )

    attributes = _remove_cell_type_attribute(
        attributes
    )

    if inline_string:
        escaped_text = escape(
            "" if value is None else str(value),
            quote=False,
        ).encode("utf-8")

        replacement = (
            b"<c"
            + attributes
            + b' t="inlineStr">'
            + b"<is><t>"
            + escaped_text
            + b"</t></is></c>"
        )

    elif value is None:
        # Keep the styled cell but leave it empty.
        replacement = (
            b"<c"
            + attributes
            + b"/>"
        )

    else:
        numeric_text = (
            f"{float(value):.1f}"
            .encode("ascii")
        )

        replacement = (
            b"<c"
            + attributes
            + b"><v>"
            + numeric_text
            + b"</v></c>"
        )

    return (
        worksheet_xml[: match.start()]
        + replacement
        + worksheet_xml[match.end() :]
    )


def _force_formula_recalculation(
    workbook_xml: bytes,
) -> bytes:
    """
    Tell Excel to recalculate formulas when the workbook opens.

    This updates the Average Temp formulas after new values
    have been inserted into columns B through Q.
    """

    calculation_pattern = re.compile(
        rb'<calcPr\b'
        rb'(?P<attributes>[^>]*)/>'
    )

    match = calculation_pattern.search(
        workbook_xml
    )

    if match is None:
        # The current official template contains calcPr.
        # Returning unchanged XML keeps the workbook usable
        # even if a future version omits it.
        return workbook_xml

    attributes = match.group(
        "attributes"
    )

    attributes = re.sub(
        rb'\s+'
        rb'(calcMode|fullCalcOnLoad|forceFullCalc)'
        rb'="[^"]*"',
        b"",
        attributes,
    )

    replacement = (
        b"<calcPr"
        + attributes
        + b' calcMode="auto"'
        + b' fullCalcOnLoad="1"'
        + b' forceFullCalc="1"/>'
    )

    return (
        workbook_xml[: match.start()]
        + replacement
        + workbook_xml[match.end() :]
    )


def _normalize_reading(
    value: Any,
    reading_type: str,
    day: int,
    point: int,
) -> float | None:
    """
    Convert one final reading into a validated Excel number.

    Blank form cells remain blank.
    """

    return validate_export_reading(
        value=value,
        reading_type=reading_type,
        day=day,
        point=point,
    )


def _index_monitoring_rows(
    monitoring_rows: list[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    """
    Validate and index the complete 31-day, 8-point dataset.
    """

    if len(monitoring_rows) != (
        EXPECTED_MONITORING_ROW_COUNT
    ):
        raise ValueError(
            "Expected exactly "
            f"{EXPECTED_MONITORING_ROW_COUNT} "
            "monitoring rows before Excel export, "
            f"but received {len(monitoring_rows)}."
        )

    missing_export_state = [
        row
        for row in monitoring_rows
        if "blocks_export" not in row
    ]
    if missing_export_state:
        raise ValueError(
            "Monitoring rows are missing explicit export verification state."
        )

    blocked_rows = [
        row
        for row in monitoring_rows
        if bool(row["blocks_export"])
    ]

    if blocked_rows:
        raise ValueError(
            f"{len(blocked_rows)} monitoring "
            "row(s) contain blocking errors or unresolved confirmations. "
            "Resolve all export-blocking items before "
            "exporting the final workbook."
        )

    indexed_rows: dict[
        tuple[int, int],
        dict[str, Any],
    ] = {}

    for row in monitoring_rows:
        day = int(
            row["day"]
        )

        point = int(
            row["point"]
        )

        if not (
            1 <= day <= EXPECTED_DAY_COUNT
        ):
            raise ValueError(
                f"Invalid day number: {day}."
            )

        if not (
            1 <= point
            <= EXPECTED_POINT_COUNT
        ):
            raise ValueError(
                f"Invalid point number: {point}."
            )

        key = (
            day,
            point,
        )

        if key in indexed_rows:
            raise ValueError(
                f"Duplicate monitoring row for "
                f"Day {day}, Point {point}."
            )

        indexed_rows[key] = row

    expected_keys = {
        (day, point)
        for day in range(
            1,
            EXPECTED_DAY_COUNT + 1,
        )
        for point in range(
            1,
            EXPECTED_POINT_COUNT + 1,
        )
    }

    missing_keys = (
        expected_keys
        - set(indexed_rows)
    )

    if missing_keys:
        first_missing_day, first_missing_point = (
            sorted(missing_keys)[0]
        )

        raise ValueError(
            "Monitoring data is incomplete. "
            f"Missing Day {first_missing_day}, "
            f"Point {first_missing_point}."
        )

    return indexed_rows


def create_monitoring_workbook(
    monitoring_rows: list[dict[str, Any]],
    month_year: str = "",
    template_path: Path = EXCEL_TEMPLATE_PATH,
) -> bytes:
    """
    Insert final readings into a copy of the official company
    template and return the completed workbook as bytes.

    The original template file is never modified.
    """

    if not template_path.exists():
        raise FileNotFoundError(
            "Excel template not found:\n"
            f"{template_path}"
        )

    indexed_rows = _index_monitoring_rows(
        monitoring_rows
    )

    output_buffer = BytesIO()

    with zipfile.ZipFile(
        template_path,
        mode="r",
    ) as source_workbook:
        workbook_files = set(
            source_workbook.namelist()
        )

        if (
            WORKSHEET_XML_PATH
            not in workbook_files
        ):
            raise ValueError(
                "The expected Monitoring Form worksheet "
                "was not found in the Excel template."
            )

        worksheet_xml = (
            source_workbook.read(
                WORKSHEET_XML_PATH
            )
        )

        workbook_xml = (
            source_workbook.read(
                WORKBOOK_XML_PATH
            )
        )

        if month_year.strip():
            worksheet_xml = (
                _replace_existing_cell(
                    worksheet_xml=worksheet_xml,
                    cell_address="C3",
                    value=month_year.strip(),
                    inline_string=True,
                )
            )

        for day in range(
            1,
            EXPECTED_DAY_COUNT + 1,
        ):
            for point in range(
                1,
                EXPECTED_POINT_COUNT + 1,
            ):
                monitoring_row = (
                    indexed_rows[
                        (day, point)
                    ]
                )

                temperature = (
                    _normalize_reading(
                        value=monitoring_row.get(
                            "temperature"
                        ),
                        reading_type=(
                            "temperature"
                        ),
                        day=day,
                        point=point,
                    )
                )

                humidity = (
                    _normalize_reading(
                        value=monitoring_row.get(
                            "humidity"
                        ),
                        reading_type="humidity",
                        day=day,
                        point=point,
                    )
                )

                temperature_cell = excel_cell_address(
                    day,
                    point,
                    "temperature",
                )

                humidity_cell = excel_cell_address(
                    day,
                    point,
                    "humidity",
                )

                worksheet_xml = (
                    _replace_existing_cell(
                        worksheet_xml=(
                            worksheet_xml
                        ),
                        cell_address=(
                            temperature_cell
                        ),
                        value=temperature,
                    )
                )

                worksheet_xml = (
                    _replace_existing_cell(
                        worksheet_xml=(
                            worksheet_xml
                        ),
                        cell_address=(
                            humidity_cell
                        ),
                        value=humidity,
                    )
                )

        workbook_xml = (
            _force_formula_recalculation(
                workbook_xml
            )
        )

        with zipfile.ZipFile(
            output_buffer,
            mode="w",
        ) as output_workbook:
            for file_information in (
                source_workbook.infolist()
            ):
                if (
                    file_information.filename
                    == WORKSHEET_XML_PATH
                ):
                    file_data = worksheet_xml

                elif (
                    file_information.filename
                    == WORKBOOK_XML_PATH
                ):
                    file_data = workbook_xml

                else:
                    file_data = (
                        source_workbook.read(
                            file_information.filename
                        )
                    )

                output_workbook.writestr(
                    file_information,
                    file_data,
                )

    output_buffer.seek(
        0
    )

    return output_buffer.getvalue()
