from __future__ import annotations

import hashlib
from io import BytesIO
import unittest
import zipfile
from xml.etree import ElementTree

from datacenter_ocr.config import EXCEL_TEMPLATE_PATH
from datacenter_ocr.excel_export import (
    WORKSHEET_XML_PATH,
    build_excel_mapping_audit,
    create_monitoring_workbook,
    excel_cell_address,
)


def complete_rows() -> list[dict[str, object]]:
    return [
        {
            "day": day,
            "point": point,
            "temperature": "22.0",
            "humidity": "50.0",
            "needs_review": False,
            "blocks_export": False,
        }
        for day in range(1, 32)
        for point in range(1, 9)
    ]


def worksheet_value(workbook_bytes: bytes, address: str) -> str | None:
    """Return the raw value stored at one worksheet coordinate."""

    with zipfile.ZipFile(BytesIO(workbook_bytes)) as workbook:
        worksheet_xml = workbook.read(WORKSHEET_XML_PATH)
    root = ElementTree.fromstring(worksheet_xml)
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    cell = next(
        node
        for node in root.iter(f"{namespace}c")
        if node.attrib.get("r") == address
    )
    value = cell.find(f"{namespace}v")
    return None if value is None else value.text


class ExcelExportTests(unittest.TestCase):
    def test_all_496_readings_keep_their_exact_excel_destination(self) -> None:
        rows = complete_rows()
        for index, row in enumerate(rows):
            row["temperature"] = f"{10.0 + index / 10:.1f}"
            row["humidity"] = f"{50.0 + index / 10:.1f}"
            row["temperature_filename"] = (
                f"day_{row['day']:02d}_point_{row['point']:02d}_temperature.png"
            )
            row["humidity_filename"] = (
                f"day_{row['day']:02d}_point_{row['point']:02d}_humidity.png"
            )

        workbook_bytes = create_monitoring_workbook(rows)
        audit_rows = build_excel_mapping_audit(rows)

        self.assertEqual(496, len(audit_rows))
        self.assertEqual(496, len({row["excel_cell"] for row in audit_rows}))
        for audit_row in audit_rows:
            expected_address = excel_cell_address(
                audit_row["day"],
                audit_row["point"],
                audit_row["reading_type"],
            )
            self.assertEqual(expected_address, audit_row["excel_cell"])
            self.assertEqual(
                audit_row["final_value"],
                worksheet_value(workbook_bytes, expected_address),
            )

    def test_export_preserves_template_and_formula_nodes(self) -> None:
        template_bytes = EXCEL_TEMPLATE_PATH.read_bytes()
        template_hash = hashlib.sha256(template_bytes).hexdigest()
        workbook_bytes = create_monitoring_workbook(
            complete_rows(), month_year="July 2026"
        )

        self.assertTrue(workbook_bytes.startswith(b"PK"))
        self.assertEqual(
            template_hash,
            hashlib.sha256(EXCEL_TEMPLATE_PATH.read_bytes()).hexdigest(),
        )

        with zipfile.ZipFile(BytesIO(template_bytes)) as source:
            source_sheet = source.read(WORKSHEET_XML_PATH)
        with zipfile.ZipFile(BytesIO(workbook_bytes)) as output:
            output_sheet = output.read(WORKSHEET_XML_PATH)
        self.assertEqual(source_sheet.count(b"<f>"), output_sheet.count(b"<f>"))

    def test_export_uses_explicit_blocks_export_state(self) -> None:
        rows = complete_rows()
        rows[0]["needs_review"] = True
        create_monitoring_workbook(rows)

        rows[0]["needs_review"] = False
        rows[0]["blocks_export"] = True
        with self.assertRaisesRegex(ValueError, "blocking errors"):
            create_monitoring_workbook(rows)

        del rows[0]["blocks_export"]
        with self.assertRaisesRegex(ValueError, "explicit export"):
            create_monitoring_workbook(rows)

    def test_export_uses_exact_temperature_and_humidity_coordinates(self) -> None:
        rows = complete_rows()
        values = {
            (1, 1): ("21.1", "51.1"),
            (1, 2): ("22.2", "52.2"),
            (2, 1): ("23.1", "53.1"),
        }
        for row in rows:
            key = (row["day"], row["point"])
            if key in values:
                row["temperature"], row["humidity"] = values[key]

        workbook_bytes = create_monitoring_workbook(rows)
        self.assertEqual("21.1", worksheet_value(workbook_bytes, "B7"))
        self.assertEqual("51.1", worksheet_value(workbook_bytes, "C7"))
        self.assertEqual("22.2", worksheet_value(workbook_bytes, "D7"))
        self.assertEqual("52.2", worksheet_value(workbook_bytes, "E7"))
        self.assertEqual("23.1", worksheet_value(workbook_bytes, "B8"))
        self.assertEqual("53.1", worksheet_value(workbook_bytes, "C8"))

    def test_blank_measurement_is_written_as_an_empty_cell(self) -> None:
        rows = complete_rows()
        rows[0]["temperature"] = ""
        workbook_bytes = create_monitoring_workbook(rows)
        self.assertIsNone(worksheet_value(workbook_bytes, "B7"))
        self.assertEqual("50.0", worksheet_value(workbook_bytes, "C7"))

    def test_export_rejects_format_not_caught_by_caller(self) -> None:
        rows = complete_rows()
        rows[0]["temperature"] = "22."
        with self.assertRaisesRegex(ValueError, "exactly one decimal"):
            create_monitoring_workbook(rows)


if __name__ == "__main__":
    unittest.main()
