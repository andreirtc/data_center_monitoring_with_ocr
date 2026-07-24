from __future__ import annotations

import unittest

import cv2
import numpy as np
import pymupdf

from datacenter_ocr.document_input import (
    decode_monitoring_document,
    decode_monitoring_documents,
)


def create_pdf_bytes(page_count: int = 1) -> bytes:
    document = pymupdf.open()
    for _ in range(page_count):
        page = document.new_page(width=144, height=72)
        page.draw_rect(
            page.rect,
            color=(1.0, 0.0, 0.0),
            fill=(1.0, 0.0, 0.0),
        )
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def create_scanned_pdf_bytes(image: np.ndarray) -> bytes:
    encoded, buffer = cv2.imencode(".png", image)
    if not encoded:
        raise AssertionError("Test scan could not be encoded.")
    document = pymupdf.open()
    page = document.new_page(width=144, height=216)
    page.insert_image(page.rect, stream=buffer.tobytes())
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


class MonitoringDocumentInputTests(unittest.TestCase):
    def test_png_decodes_to_existing_bgr_image_format(self) -> None:
        source = np.full((12, 20, 3), (10, 20, 30), dtype=np.uint8)
        encoded, buffer = cv2.imencode(".png", source)
        self.assertTrue(encoded)

        decoded = decode_monitoring_document(
            buffer.tobytes(),
            "scan.png",
        )

        self.assertEqual("image", decoded.source_kind)
        self.assertEqual(1, decoded.page_number)
        self.assertEqual(1, decoded.page_count)
        self.assertIsNone(decoded.render_dpi)
        self.assertEqual("uploaded_image", decoded.image_source)
        np.testing.assert_array_equal(source, decoded.image)

    def test_one_page_pdf_renders_at_requested_dpi_in_bgr(self) -> None:
        decoded = decode_monitoring_document(
            create_pdf_bytes(),
            "scan.pdf",
            pdf_render_dpi=72,
        )

        self.assertEqual("pdf", decoded.source_kind)
        self.assertEqual(1, decoded.page_number)
        self.assertEqual(1, decoded.page_count)
        self.assertEqual(72, decoded.render_dpi)
        self.assertEqual("rendered_page", decoded.image_source)
        self.assertEqual(72.0, decoded.effective_dpi)
        self.assertEqual((72, 144, 3), decoded.image.shape)
        blue, green, red = decoded.image[36, 72]
        self.assertLess(blue, 10)
        self.assertLess(green, 10)
        self.assertGreater(red, 245)

    def test_pdf_signature_is_detected_even_with_image_suffix(self) -> None:
        decoded = decode_monitoring_document(
            create_pdf_bytes(),
            "renamed-scan.jpg",
            pdf_render_dpi=72,
        )

        self.assertEqual("pdf", decoded.source_kind)

    def test_multi_page_pdf_is_returned_as_independent_pages(self) -> None:
        decoded_pages = decode_monitoring_documents(
            create_pdf_bytes(page_count=2),
            "three-month-scan.pdf",
            pdf_render_dpi=72,
        )

        self.assertEqual(2, len(decoded_pages))
        self.assertEqual([1, 2], [page.page_number for page in decoded_pages])
        self.assertTrue(all(page.page_count == 2 for page in decoded_pages))
        self.assertTrue(
            all(page.image.shape == (72, 144, 3) for page in decoded_pages)
        )

    def test_full_page_scanner_raster_is_extracted_without_upscaling(
        self,
    ) -> None:
        source = np.full((300, 200, 3), 255, dtype=np.uint8)
        cv2.line(source, (10, 20), (190, 20), (0, 0, 0), 2)

        decoded = decode_monitoring_document(
            create_scanned_pdf_bytes(source),
            "printer-scan.pdf",
            pdf_render_dpi=300,
        )

        self.assertEqual("embedded_scan", decoded.image_source)
        self.assertIsNone(decoded.render_dpi)
        self.assertEqual((300, 200, 3), decoded.image.shape)
        np.testing.assert_array_equal(source, decoded.image)

    def test_single_document_decoder_rejects_multi_page_pdf(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "contains 2 pages.*decode_monitoring_documents",
        ):
            decode_monitoring_document(
                create_pdf_bytes(page_count=2),
                "three-month-scan.pdf",
                pdf_render_dpi=72,
            )

    def test_invalid_or_unsupported_upload_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "could not be opened"):
            decode_monitoring_document(b"not a pdf", "scan.pdf")
        with self.assertRaisesRegex(ValueError, "Unsupported upload type"):
            decode_monitoring_document(b"plain text", "scan.txt")


if __name__ == "__main__":
    unittest.main()
