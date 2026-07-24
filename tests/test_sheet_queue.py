from __future__ import annotations

import unittest

import cv2
import numpy as np
import pymupdf

from datacenter_ocr.sheet_queue import (
    archive_prepared_sheet,
    build_sheet_queue_items,
    decode_queue_image,
    restore_prepared_sheet,
)
from datacenter_ocr.sheet_processing import PreparedMonitoringSheet


def create_pdf_bytes(page_count: int) -> bytes:
    document = pymupdf.open()
    for page_index in range(page_count):
        page = document.new_page(width=144, height=72)
        color = (1.0, 0.0, 0.0) if page_index == 0 else (0.0, 1.0, 0.0)
        page.draw_rect(page.rect, color=color, fill=color)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


class SheetQueueTests(unittest.TestCase):
    def test_image_becomes_one_lossless_queue_item(self) -> None:
        source = np.full((12, 20, 3), (10, 20, 30), dtype=np.uint8)
        encoded, buffer = cv2.imencode(".png", source)
        self.assertTrue(encoded)

        items = build_sheet_queue_items(buffer.tobytes(), "april.png")

        self.assertEqual(1, len(items))
        self.assertEqual("april.png", items[0].display_name)
        self.assertEqual(1, items[0].page_number)
        np.testing.assert_array_equal(source, decode_queue_image(items[0]))
        np.testing.assert_array_equal(
            cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE),
            decode_queue_image(items[0], "clockwise"),
        )

    def test_multi_page_pdf_becomes_separately_named_queue_items(self) -> None:
        items = build_sheet_queue_items(
            create_pdf_bytes(page_count=2),
            "april-may.pdf",
        )

        self.assertEqual(2, len(items))
        self.assertEqual(
            ["april-may.pdf — Page 1 of 2", "april-may.pdf — Page 2 of 2"],
            [item.display_name for item in items],
        )
        self.assertNotEqual(items[0].sheet_id, items[1].sheet_id)

    def test_queue_identity_is_stable_for_repeated_upload(self) -> None:
        source_bytes = create_pdf_bytes(page_count=2)

        first = build_sheet_queue_items(source_bytes, "scan.pdf")
        second = build_sheet_queue_items(source_bytes, "renamed.pdf")

        self.assertEqual(
            [item.sheet_id for item in first],
            [item.sheet_id for item in second],
        )

    def test_prepared_sheet_archive_restores_images_and_metadata(self) -> None:
        preview = np.full((8, 10, 3), (10, 20, 30), dtype=np.uint8)
        crop = np.full((3, 4, 3), (40, 50, 60), dtype=np.uint8)
        sheet = PreparedMonitoringSheet(
            detection_preview=preview,
            warped_table=preview.copy(),
            measurement_grid_overlay=preview.copy(),
            measurement_boxes=[{"day": 1}],
            cells=[
                {
                    "filename": "day_01_point_01_temperature.png",
                    "day": 1,
                    "image": crop,
                }
            ],
            geometry_mode="calibrated",
            invalid_geometry_cell_count=2,
        )

        restored = restore_prepared_sheet(archive_prepared_sheet(sheet))

        self.assertEqual("calibrated", restored.geometry_mode)
        self.assertEqual(2, restored.invalid_geometry_cell_count)
        self.assertEqual([{"day": 1}], restored.measurement_boxes)
        np.testing.assert_array_equal(preview, restored.detection_preview)
        np.testing.assert_array_equal(crop, restored.cells[0]["image"])


if __name__ == "__main__":
    unittest.main()
