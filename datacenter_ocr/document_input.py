from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf


PDF_RENDER_DPI = 300
MAXIMUM_RENDERED_PDF_PIXELS = 40_000_000
SUPPORTED_IMAGE_SUFFIXES = frozenset((".jpeg", ".jpg", ".png"))


@dataclass(frozen=True)
class DecodedMonitoringDocument:
    """One decoded monitoring-sheet page ready for the image pipeline."""

    image: np.ndarray
    source_kind: str
    page_number: int
    page_count: int
    render_dpi: int | None
    image_source: str
    effective_dpi: float | None


def _decode_image(uploaded_bytes: bytes) -> DecodedMonitoringDocument:
    """Decode a PNG or JPEG into the application's BGR image format."""

    byte_array = np.frombuffer(uploaded_bytes, dtype=np.uint8)
    image = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            "The uploaded image could not be decoded. Upload a valid PNG, "
            "JPG, or JPEG file."
        )
    return DecodedMonitoringDocument(
        image=image,
        source_kind="image",
        page_number=1,
        page_count=1,
        render_dpi=None,
        image_source="uploaded_image",
        effective_dpi=None,
    )


def _expected_pdf_pixel_count(
    page: pymupdf.Page,
    render_dpi: int,
) -> int:
    """Estimate rendered page pixels before allocating the PDF pixmap."""

    scale = render_dpi / 72.0
    return round(page.rect.width * scale) * round(page.rect.height * scale)


def _decode_full_page_embedded_scan(
    document: pymupdf.Document,
    page: pymupdf.Page,
) -> tuple[np.ndarray, float] | None:
    """Extract an unrotated, full-page scanner raster without upscaling it."""

    if page.rotation != 0 or page.get_text().strip():
        return None

    images = page.get_images(full=True)
    if len(images) != 1:
        return None

    image_info = images[0]
    xref = image_info[0]
    soft_mask_xref = image_info[1]
    pixel_width = int(image_info[2])
    pixel_height = int(image_info[3])
    if (
        soft_mask_xref
        or pixel_width * pixel_height > MAXIMUM_RENDERED_PDF_PIXELS
    ):
        return None

    placements = page.get_image_rects(xref, transform=True)
    if len(placements) != 1:
        return None
    placement, transform = placements[0]
    if (
        transform.a <= 0
        or transform.d <= 0
        or abs(transform.b) > 1e-6
        or abs(transform.c) > 1e-6
    ):
        return None
    page_rect = page.rect
    width_coverage = placement.width / max(page_rect.width, 1.0)
    height_coverage = placement.height / max(page_rect.height, 1.0)
    edge_tolerance_x = page_rect.width * 0.01
    edge_tolerance_y = page_rect.height * 0.01
    if (
        width_coverage < 0.98
        or height_coverage < 0.98
        or abs(placement.x0 - page_rect.x0) > edge_tolerance_x
        or abs(placement.y0 - page_rect.y0) > edge_tolerance_y
        or abs(placement.x1 - page_rect.x1) > edge_tolerance_x
        or abs(placement.y1 - page_rect.y1) > edge_tolerance_y
    ):
        return None

    try:
        extracted = document.extract_image(xref)["image"]
    except Exception:
        return None
    image = cv2.imdecode(
        np.frombuffer(extracted, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        return None

    horizontal_dpi = image.shape[1] / max(page_rect.width / 72.0, 1e-6)
    vertical_dpi = image.shape[0] / max(page_rect.height / 72.0, 1e-6)
    effective_dpi = round((horizontal_dpi + vertical_dpi) / 2.0, 2)
    return image, effective_dpi


def _decode_pdf_pages(
    uploaded_bytes: bytes,
    render_dpi: int,
) -> tuple[DecodedMonitoringDocument, ...]:
    """Render every PDF page into BGR without invoking OCR."""

    if render_dpi <= 0:
        raise ValueError("PDF render DPI must be positive.")

    try:
        document = pymupdf.open(stream=uploaded_bytes, filetype="pdf")
    except Exception as error:
        raise ValueError(
            "The uploaded PDF could not be opened. Upload a valid, "
            "unencrypted PDF scan."
        ) from error

    with document:
        if document.needs_pass:
            raise ValueError(
                "Password-protected PDFs are not supported. Export or scan "
                "the monitoring sheet as an unencrypted PDF."
            )

        page_count = document.page_count
        if page_count < 1:
            raise ValueError(
                "The uploaded PDF does not contain any pages."
            )

        decoded_pages: list[DecodedMonitoringDocument] = []
        for page_index in range(page_count):
            page = document.load_page(page_index)
            embedded_scan = _decode_full_page_embedded_scan(document, page)
            if embedded_scan is not None:
                bgr_image, effective_dpi = embedded_scan
                page_render_dpi: int | None = None
                image_source = "embedded_scan"
            else:
                expected_pixels = _expected_pdf_pixel_count(page, render_dpi)
                if expected_pixels > MAXIMUM_RENDERED_PDF_PIXELS:
                    raise ValueError(
                        f"PDF page {page_index + 1} is too large to render "
                        f"safely at {render_dpi} DPI. Rescan it at 300 DPI on "
                        "a standard paper size."
                    )

                pixmap = page.get_pixmap(
                    dpi=render_dpi,
                    colorspace=pymupdf.csRGB,
                    alpha=False,
                )
                rgb_image = np.frombuffer(
                    pixmap.samples,
                    dtype=np.uint8,
                ).reshape(pixmap.height, pixmap.width, pixmap.n)
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                effective_dpi = float(render_dpi)
                page_render_dpi = render_dpi
                image_source = "rendered_page"

            decoded_pages.append(
                DecodedMonitoringDocument(
                    image=bgr_image,
                    source_kind="pdf",
                    page_number=page_index + 1,
                    page_count=page_count,
                    render_dpi=page_render_dpi,
                    image_source=image_source,
                    effective_dpi=effective_dpi,
                )
            )

    return tuple(decoded_pages)


def decode_monitoring_documents(
    uploaded_bytes: bytes,
    source_filename: str,
    *,
    pdf_render_dpi: int = PDF_RENDER_DPI,
) -> tuple[DecodedMonitoringDocument, ...]:
    """Decode an image or every page of a PDF into monitoring sheets."""

    if not uploaded_bytes:
        raise ValueError("The uploaded file is empty.")

    suffix = Path(source_filename).suffix.lower()
    looks_like_pdf = uploaded_bytes.lstrip().startswith(b"%PDF-")
    if suffix == ".pdf" or looks_like_pdf:
        return _decode_pdf_pages(uploaded_bytes, pdf_render_dpi)
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        return (_decode_image(uploaded_bytes),)

    raise ValueError(
        "Unsupported upload type. Upload a PNG, JPG, JPEG, or PDF."
    )


def decode_monitoring_document(
    uploaded_bytes: bytes,
    source_filename: str,
    *,
    pdf_render_dpi: int = PDF_RENDER_DPI,
) -> DecodedMonitoringDocument:
    """Decode one monitoring sheet, rejecting PDFs with multiple pages."""

    decoded_pages = decode_monitoring_documents(
        uploaded_bytes,
        source_filename,
        pdf_render_dpi=pdf_render_dpi,
    )
    if len(decoded_pages) != 1:
        raise ValueError(
            f"This PDF contains {len(decoded_pages)} pages. Use "
            "decode_monitoring_documents() to receive each page."
        )
    return decoded_pages[0]
