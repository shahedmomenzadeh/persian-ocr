"""Lazy 1-page-at-a-time PDF rasterizer and image loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pypdfium2 as pdfium
from PIL import Image, ImageOps, ImageSequence

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(slots=True)
class PageInput:
    """Single rasterized document page yielded lazily."""

    page_number: int
    image: Image.Image
    native_text: str | None = None
    has_native_text: bool = False


def _load_pdf_pages(path: Path, dpi: int) -> Iterator[PageInput]:
    """Stream PDF pages one by one using pypdfium2 to bound RAM usage."""
    scale = dpi / 72.0
    pdf = pdfium.PdfDocument(str(path))
    try:
        total_pages = len(pdf)
        for idx in range(total_pages):
            page = pdf[idx]
            textpage = None
            bitmap = None
            try:
                textpage = page.get_textpage()
                native_text = textpage.get_text_range() or ""
                has_native = len(native_text.strip()) > 20

                bitmap = page.render(scale=scale)
                pil_img = bitmap.to_pil().convert("RGB")
                yield PageInput(
                    page_number=idx + 1,
                    image=pil_img,
                    native_text=native_text if native_text.strip() else None,
                    has_native_text=has_native,
                )
            finally:
                if bitmap is not None:
                    bitmap.close()
                if textpage is not None:
                    textpage.close()
                page.close()
    finally:
        pdf.close()


def _load_image_pages(path: Path) -> Iterator[PageInput]:
    """Yield single image or frames from multi-page TIFF lazily."""
    with Image.open(path) as img:
        for idx, frame in enumerate(ImageSequence.Iterator(img), start=1):
            oriented = ImageOps.exif_transpose(frame)
            rgb_copy = oriented.convert("RGB")
            yield PageInput(
                page_number=idx,
                image=rgb_copy,
                native_text=None,
                has_native_text=False,
            )


def load_document_pages(path: Path | str, dpi: int = 200) -> Iterator[PageInput]:
    """Yield PageInput objects lazily for a PDF or image file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Input document not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        yield from _load_pdf_pages(file_path, dpi=dpi)
    elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
        yield from _load_image_pages(file_path)
    else:
        raise ValueError(
            f"Unsupported document format {suffix!r}. "
            f"Expected .pdf or one of {sorted(SUPPORTED_IMAGE_EXTENSIONS)}."
        )
