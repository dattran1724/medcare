"""Native PDF geometry + real Docling serializer; no inference/model downloads."""
import pytest

from rag.extract import render_markdown
from rag.layout import separate_paragraphs


@pytest.mark.parametrize("extra_gap", [False, True])
def test_native_paragraph_spacing_preserves_wrapped_lines_and_content(extra_gap):
    doc = pytest.importorskip("docling_core.types.doc")
    document = doc.DoclingDocument(name="paragraph-spacing")
    document.add_page(page_no=1, size=doc.Size(width=600, height=800))
    lines = ["First paragraph starts here.", "This is its wrapped continuation.",
             'Second paragraph "begins" here.', "Values remain 2.10 and 3.80."]
    bottoms = [700, 688, 668 if extra_gap else 676, 656 if extra_gap else 664]
    rects = [(40, y, 300, y + 10) for y in bottoms]
    class NativeText:
        def count_rects(self): return len(rects)
        def get_rect(self, index): return rects[index]
        def get_text_bounded(self, left, bottom, right, top):
            return lines[bottoms.index(bottom)].replace('"begins"', '\u201cbegins\u201d')
    text = " ".join(lines)
    document.add_text(label=doc.DocItemLabel.TEXT, text=text, prov=doc.ProvenanceItem(page_no=1,
        bbox=doc.BoundingBox(l=40, b=650, r=300, t=715, coord_origin=doc.CoordOrigin.BOTTOMLEFT),
        charspan=(0, len(text))))
    before = document.model_dump()
    result = separate_paragraphs(document, 1, NativeText(), text)
    assert " ".join(result.split()) == text
    assert (lines[1] + "\n\n" + lines[2] in result) == extra_gap
    assert lines[0] + " " + lines[1] in result
    assert document.model_dump() == before


def test_numbered_entries_split_at_pdf_lines_and_keep_wrapped_text():
    doc = pytest.importorskip("docling_core.types.doc")
    document = doc.DoclingDocument(name="numbered-entries")
    document.add_page(page_no=7, size=doc.Size(width=600, height=800))
    lines = ["51. A long procedure title", "continues with values 2.10 and 3.80.",
             "52. Another procedure title", "continues on its next line.",
             "53. The final procedure title"]
    rects = [(40, 700 - i * 12, 300, 710 - i * 12) for i in range(len(lines))]
    class NativeText:
        def count_rects(self): return len(rects)
        def get_rect(self, index): return rects[index]
        def get_text_bounded(self, left, bottom, right, top): return lines[(700 - bottom) // 12]
    text = " ".join(lines)
    document.add_text(label=doc.DocItemLabel.TEXT, text=text, prov=doc.ProvenanceItem(page_no=7,
        bbox=doc.BoundingBox(l=40, b=640, r=300, t=715, coord_origin=doc.CoordOrigin.BOTTOMLEFT),
        charspan=(0, len(text))))
    result = separate_paragraphs(document, 7, NativeText(), text)
    # Markdown may escape the first numbered marker; all source characters remain.
    assert " ".join(result.replace("\\.", ".").split()) == text
    assert lines[1] + "\n\n52" in result
    assert lines[3] + "\n\n53" in result
    assert lines[0] + " " + lines[1] in result


def test_native_fragment_cannot_split_inside_a_word():
    doc = pytest.importorskip("docling_core.types.doc")
    document = doc.DoclingDocument(name="fragmented-native-line")
    document.add_page(page_no=1, size=doc.Size(width=600, height=800))
    lines = ["An introductory paragraph.", "Symptoms include ti",
             "m, with more details for this patient.", "A final line of text."]
    bottoms = [700, 688, 668, 656]
    rects = [(40, y, 300, y + 10) for y in bottoms]
    class NativeText:
        def count_rects(self): return len(rects)
        def get_rect(self, index): return rects[index]
        def get_text_bounded(self, left, bottom, right, top): return lines[bottoms.index(bottom)]
    text = lines[0] + " " + lines[1] + lines[2] + " " + lines[3]
    document.add_text(label=doc.DocItemLabel.TEXT, text=text,
        prov=doc.ProvenanceItem(page_no=1, bbox=doc.BoundingBox(l=40, b=650, r=300, t=715,
            coord_origin=doc.CoordOrigin.BOTTOMLEFT), charspan=(0, len(text))))
    assert separate_paragraphs(document, 1, NativeText(), text) == text


@pytest.fixture
def layout_source(tmp_path):
    pdfium = pytest.importorskip("pypdfium2")
    doc = pytest.importorskip("docling_core.types.doc")
    commands = []
    for x, y, text in [(40, 750, "ISSUING OFFICE"), (350, 750, "REPUBLIC"), (350, 735, "MOTTO"),
        (40, 700, "Reference ABC"), (350, 700, "Date 2026"), (220, 650, "DECISION"),
        (40, 250, "RECIPIENTS"), (350, 255, "ACTING MINISTER"), (350, 240, "DEPUTY MINISTER"),
        (40, 220, "- Item A"), (40, 205, "- Item B"), (350, 205, "SIGNATURE"),
        (40, 160, "AUTHORS"), (40, 120, "NAME"), (350, 120, "AFFILIATION")]:
        commands.append(f"BT /F1 11 Tf {x} {y} Td ({text}) Tj ET")
    stream = "\n".join(commands).encode("ascii")
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Count 1 /Kids [5 0 R] >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Resources << /Font << /F1 3 0 R >> >> /Contents 4 0 R >>"]
    data, offsets = bytearray(b"%PDF-1.4\n"), [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    start = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode())
    path = tmp_path / "layout.pdf"
    path.write_bytes(data)
    document = doc.DoclingDocument(name="layout")
    document.add_page(page_no=1, size=doc.Size(width=600, height=800))
    for text, label, box in [
        ("ISSUING OFFICE REPUBLIC MOTTO", doc.DocItemLabel.SECTION_HEADER, (40, 730, 500, 765)),
        ("Reference ABC Date 2026", doc.DocItemLabel.TEXT, (40, 695, 500, 715)),
        ("DECISION", doc.DocItemLabel.SECTION_HEADER, (220, 645, 300, 665)),
        ("RECIPIENTS ACTING MINISTER DEPUTY MINISTER", doc.DocItemLabel.SECTION_HEADER, (40, 238, 500, 270)),
        ("- Item A - Item B SIGNATURE", doc.DocItemLabel.TEXT, (40, 200, 500, 235)),
        ("AUTHORS", doc.DocItemLabel.SECTION_HEADER, (40, 155, 150, 175)),
        ("NAME AFFILIATION", doc.DocItemLabel.TEXT, (40, 115, 500, 135)),
    ]:
        l, b, r, t = box
        document.add_text(label=label, text=text, prov=doc.ProvenanceItem(page_no=1,
            bbox=doc.BoundingBox(l=l, b=b, r=r, t=t, coord_origin=doc.CoordOrigin.BOTTOMLEFT), charspan=(0, len(text))))
    with pdfium.PdfDocument(path) as native:
        yield document, native


def test_independent_header_and_signature_columns_are_not_interleaved(layout_source):
    document, native = layout_source
    before = document.model_dump()
    text, warnings = render_markdown(document, pdf=native)
    assert text.index("ISSUING OFFICE") < text.index("Reference ABC") < text.index("REPUBLIC") < text.index("Date 2026")
    assert text.index("RECIPIENTS") < text.index("Item A") < text.index("Item B") < text.index("ACTING MINISTER") < text.index("SIGNATURE")
    assert "NAME AFFILIATION" in text  # Do not reinterpret paired body text as independent columns.
    assert document.model_dump() == before
    assert any("separated 2" in warning for warning in warnings)


def test_unsupported_native_text_is_not_used_to_silently_replace_model_text(layout_source):
    document, native = layout_source
    document.texts[0].text += " EXTRA"
    text, warnings = render_markdown(document, pdf=native)
    assert "ISSUING OFFICE REPUBLIC MOTTO EXTRA" in text
    assert any("separated 1" in warning for warning in warnings)
