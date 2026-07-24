from __future__ import annotations

import unittest

import cv2
import numpy as np

from datacenter_ocr.document_orientation import (
    detect_monitoring_sheet_orientation,
    rotate_monitoring_image,
)


def create_synthetic_monitoring_sheet() -> np.ndarray:
    """Create the form's asymmetric landscape line layout without text OCR."""

    image = np.full((800, 1200, 3), 255, dtype=np.uint8)
    table_left = 20
    measurement_right = 870
    table_right = 1175
    table_top = 35
    table_bottom = 630

    cv2.rectangle(
        image,
        (table_left, table_top),
        (table_right, table_bottom),
        (0, 0, 0),
        3,
    )
    for row in range(33):
        y = table_top + round(row * (table_bottom - table_top) / 32)
        cv2.line(
            image,
            (table_left, y),
            (table_right, y),
            (0, 0, 0),
            1,
        )
    for column in range(18):
        x = table_left + round(
            column * (measurement_right - table_left) / 17
        )
        cv2.line(
            image,
            (x, table_top),
            (x, table_bottom),
            (0, 0, 0),
            1,
        )
    for x in (970, 1070):
        cv2.line(
            image,
            (x, table_top),
            (x, table_bottom),
            (0, 0, 0),
            1,
        )
    cv2.rectangle(image, (20, 5), (900, 28), (0, 0, 0), -1)
    return image


class DocumentOrientationTests(unittest.TestCase):
    def test_landscape_sheet_is_not_rotated(self) -> None:
        landscape = create_synthetic_monitoring_sheet()

        decision = detect_monitoring_sheet_orientation(landscape)

        self.assertEqual("none", decision.rotation)
        self.assertTrue(decision.confident)

    def test_clockwise_sideways_scan_is_rotated_left(self) -> None:
        landscape = create_synthetic_monitoring_sheet()
        portrait = cv2.rotate(landscape, cv2.ROTATE_90_CLOCKWISE)

        decision = detect_monitoring_sheet_orientation(portrait)
        restored = rotate_monitoring_image(portrait, decision.rotation)

        self.assertEqual("counterclockwise", decision.rotation)
        self.assertTrue(decision.confident)
        np.testing.assert_array_equal(landscape, restored)

    def test_counterclockwise_sideways_scan_is_rotated_right(self) -> None:
        landscape = create_synthetic_monitoring_sheet()
        portrait = cv2.rotate(landscape, cv2.ROTATE_90_COUNTERCLOCKWISE)

        decision = detect_monitoring_sheet_orientation(portrait)
        restored = rotate_monitoring_image(portrait, decision.rotation)

        self.assertEqual("clockwise", decision.rotation)
        self.assertTrue(decision.confident)
        np.testing.assert_array_equal(landscape, restored)

    def test_ambiguous_portrait_is_left_for_manual_review(self) -> None:
        blank_portrait = np.full((800, 500, 3), 255, dtype=np.uint8)

        decision = detect_monitoring_sheet_orientation(blank_portrait)

        self.assertEqual("none", decision.rotation)
        self.assertFalse(decision.confident)


if __name__ == "__main__":
    unittest.main()
