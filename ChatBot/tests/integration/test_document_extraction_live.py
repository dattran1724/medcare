"""Opt-in real PDF/model smoke test; first run downloads Docling models."""
import os

import pytest

from rag.extract import extract_pdf


pytestmark = pytest.mark.skipif(os.getenv("MEDCARE_TEST_DOCLING") != "1", reason="Docling model tests are opt-in")


def write_layout_pdf(path):
    """Two-column page and a second page, using built-in PDF fonts; no patient data."""
    objects = [b"", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    pages = []
    commands = ["BT /F1 18 Tf 50 750 Td (Synthetic layout test) Tj ET"]
    for x, label in [(50, "LEFT"), (330, "RIGHT")]:
        for index in range(1, 9):
            commands.append(f"BT /F1 11 Tf {x} {700 - index * 20} Td ({label} column line {index}. Synthetic text.) Tj ET")
    for stream in ["\n".join(commands), "BT /F1 18 Tf 50 750 Td (SECOND PAGE) Tj ET\nBT /F1 12 Tf 50 700 Td (Continuation after the columns.) Tj ET"]:
        content = stream.encode("ascii")
        stream_id = len(objects) + 1
        objects.append(b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream")
        page_id = len(objects) + 1
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {stream_id} 0 R >>".encode())
        pages.append(page_id)
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Count {len(pages)} /Kids [{' '.join(f'{p} 0 R' for p in pages)}] >>".encode()
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, item in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode() + item + b"\nendobj\n")
    start = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode())
    path.write_bytes(data)


def test_real_pdf_layout_and_numbered_markdown(tmp_path):
    pytest.importorskip("docling.document_converter")
    source, output = tmp_path / "layout.pdf", tmp_path / "layout.md"
    write_layout_pdf(source)
    device = os.getenv("MEDCARE_TEST_DOCLING_DEVICE", "cpu")
    if device == "cuda":
        import torch
        assert torch.cuda.is_available()
        torch.cuda.reset_peak_memory_stats()
    _, warnings = extract_pdf(source, output, ocr="off", device=device)
    text = output.read_text(encoding="utf-8")
    assert text.count("<!-- page: 1 -->") == text.count("<!-- page: 2 -->") == 1
    assert text.index("LEFT column line 8") < text.index("RIGHT column line 1")
    assert text.index("RIGHT column line 8") < text.index("<!-- page: 2 -->") < text.index("SECOND PAGE")
    assert not warnings
    if device == "cuda":
        assert torch.cuda.max_memory_allocated() > 0
        print(f"CUDA peak allocated MiB: {torch.cuda.max_memory_allocated() / 1024**2:.1f}")
