"""Passages avec chevauchement, limités à une page et à des offsets vérifiables."""

import hashlib
from dataclasses import dataclass

from .documents import PDFPage

CHUNK_VERSION = "page-char-v1"


@dataclass(frozen=True)
class DocumentChunk:
    identifier: str
    text: str
    metadata: dict


def chunk_page(page: PDFPage, *, chunk_chars: int = 900, overlap_chars: int = 150) -> list[DocumentChunk]:
    """Ne jamais mélanger deux pages ou attribuer un texte à une autre source."""
    if chunk_chars < 100 or not 0 <= overlap_chars < chunk_chars // 2:
        raise ValueError("chunk_chars >= 100 et 0 <= overlap < la moitié de chunk_chars requis.")
    text, start, chunks = page.text, 0, []
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            boundaries = [position for position in range(start + chunk_chars // 2, end) if text[position].isspace()]
            if boundaries:
                end = boundaries[-1]
        raw = text[start:end]
        actual_start = start + len(raw) - len(raw.lstrip())
        actual_end = end - (len(raw) - len(raw.rstrip()))
        content = text[actual_start:actual_end]
        if content:
            metadata = {**page.metadata, "page": page.page, "page_label": page.page_label,
                        "start_char": actual_start, "end_char": actual_end, "chunk_version": CHUNK_VERSION,
                        "text_is_untrusted_data": True}
            signature = f"{metadata['relative_path']}:{metadata['pdf_sha256']}:{page.page}:{actual_start}:{actual_end}:{CHUNK_VERSION}"
            chunks.append(DocumentChunk(hashlib.sha256(signature.encode()).hexdigest(), content, metadata))
        if end == len(text):
            break
        start = max(start + 1, end - overlap_chars)
        while start < end and start > 0 and not text[start - 1].isspace():
            start += 1
    return chunks
