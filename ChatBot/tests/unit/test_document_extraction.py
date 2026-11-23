"""Export and file-safety checks without model downloads or external requests."""
from types import SimpleNamespace

import pytest

from rag.extract import ExtractionError, build_converter, extract_pdf, main, render_markdown


class Document:
    def __init__(self, pages):
        self.pages = {p: object() for p in pages}
        self.content = pages
        self.pictures = []

    def export_to_markdown(self, *, page_no):
        return self.content[page_no]


def converter(pages, status="success"):
    return SimpleNamespace(convert=lambda path: SimpleNamespace(status=status, document=Document(pages)))


def source_pdf(tmp_path):
    # Tests inject conversion, so no real PDF parser is invoked here.
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\nsynthetic fixture")
    return source


def test_unicode_tables_and_empty_pages_keep_original_numbers(tmp_path):
    source = source_pdf(tmp_path)
    output = tmp_path / "output.md"
    pages = {1: "# Triệu chứng\n\nNội dung tiếng Việt.", 2: "", 3: "| A | B |\n| --- | --- |\n| 1 | 2 |"}
    path, warnings = extract_pdf(source, output, converter=converter(pages))
    markdown = path.read_text(encoding="utf-8")
    assert markdown.index("<!-- page: 1 -->") < markdown.index("<!-- page: 2 -->") < markdown.index("<!-- page: 3 -->")
    assert "Nội dung tiếng Việt." in markdown and pages[3] in markdown
    assert len(warnings) == 1 and "Page 2" in warnings[0]
    assert source.read_bytes().startswith(b"%PDF")


def test_existing_output_requires_opt_in_and_survives_failed_conversion(tmp_path):
    source, output = source_pdf(tmp_path), tmp_path / "output.md"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(ExtractionError, match="already exists"):
        extract_pdf(source, output, converter=converter({1: "new"}))
    with pytest.raises(ExtractionError, match="fully succeed"):
        extract_pdf(source, output, overwrite=True, converter=converter({1: "partial"}, "partial_success"))
    assert output.read_text() == "existing"
    extract_pdf(source, output, overwrite=True, converter=converter({1: "new"}))
    assert "new" in output.read_text()
    assert not list(tmp_path.glob(".extract-*.tmp"))


@pytest.mark.parametrize("pages", [{}, {1: "", 2: "<!-- image -->"}, {1: "a", 3: "b"}])
def test_unreadable_or_incomplete_documents_do_not_publish_markdown(tmp_path, pages):
    source, output = source_pdf(tmp_path), tmp_path / "output.md"
    with pytest.raises(ExtractionError):
        extract_pdf(source, output, converter=converter(pages))
    assert not output.exists()


def test_failed_page_conversion_does_not_publish_partial_output(tmp_path):
    source, output = source_pdf(tmp_path), tmp_path / "output.md"
    with pytest.raises(ExtractionError):
        extract_pdf(source, output, converter=converter({1: "text"}, status="failure"))
    assert not output.exists()


def test_invalid_paths_and_options_are_rejected_before_conversion(tmp_path):
    source = source_pdf(tmp_path)
    def unexpected(path):
        pytest.fail("Converter must not run for invalid input/output")
    service = SimpleNamespace(convert=unexpected)
    for input_path, output, options in [
        (tmp_path / "missing.pdf", tmp_path / "out.md", {}),
        (source, source, {}),
        (source, tmp_path / "out.txt", {}),
        (source, tmp_path / "out.md", {"ocr": "invalid"}),
    ]:
        with pytest.raises(ExtractionError):
            extract_pdf(input_path, output, converter=service, **options)


def test_images_and_cross_page_tables_are_flagged_for_review():
    document = Document({1: "# Text", 2: "More"})
    document.pictures = [object()]
    document.tables = [SimpleNamespace(prov=[SimpleNamespace(page_no=1), SimpleNamespace(page_no=2)])]
    _, warnings = render_markdown(document)
    assert any("Figures" in w for w in warnings) and any("spans multiple" in w for w in warnings)


def test_cli_error_is_nonzero_without_saying_saved(tmp_path, capsys):
    assert main([str(tmp_path / "missing.pdf")]) == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err and "Saved:" not in captured.out


def test_default_batch_processes_all_pdfs_reuses_models_and_continues_on_error(tmp_path, monkeypatch, capsys):
    from rag import extract
    monkeypatch.setattr(extract, "__file__", str(tmp_path / "rag" / "extract.py"))
    source_dir = tmp_path / "rag" / "data" / "pdf"
    source_dir.mkdir(parents=True)
    for name in ("a.pdf", "bad.PDF", "c.pdf", "ignored.txt"):
        (source_dir / name).write_bytes(b"synthetic")
    builds, calls = [], []
    def convert(path):
        calls.append(path.name)
        if path.name == "bad.PDF":
            raise RuntimeError("Synthetic conversion failure")
        return SimpleNamespace(status="success", document=Document({1: path.stem}))
    def build(*args):
        builds.append(1)
        return SimpleNamespace(convert=convert)
    monkeypatch.setattr(extract, "build_converter", build)
    assert extract.main([]) == 1  # A failed file does not imply batch success.
    output_dir = tmp_path / "rag" / "data" / "markdown"
    assert (output_dir / "a.md").exists() and (output_dir / "c.md").exists()
    assert not (output_dir / "bad.md").exists()
    assert calls == ["a.pdf", "bad.PDF", "c.pdf"] and len(builds) == 1
    captured = capsys.readouterr()
    assert "2 saved, 0 skipped, 1 failed" in captured.out


def test_batch_skips_existing_outputs_without_loading_models_and_can_overwrite(tmp_path, monkeypatch):
    from rag import extract
    source_dir, output_dir = tmp_path / "pdf", tmp_path / "markdown"
    source_dir.mkdir()
    output_dir.mkdir()
    (source_dir / "a.pdf").write_bytes(b"synthetic")
    (output_dir / "a.md").write_text("old", encoding="utf-8")
    builds = []
    def build(*args):
        builds.append(1)
        return converter({1: "new"})
    monkeypatch.setattr(extract, "build_converter", build)
    assert extract.main([str(source_dir), "-o", str(output_dir)]) == 0
    assert not builds and (output_dir / "a.md").read_text() == "old"
    assert extract.main([str(source_dir), "-o", str(output_dir), "--overwrite"]) == 0
    assert len(builds) == 1 and "new" in (output_dir / "a.md").read_text()


def test_empty_batch_does_not_load_models(tmp_path, monkeypatch):
    from rag import extract
    monkeypatch.setattr(extract, "build_converter", lambda *a: pytest.fail("No PDFs: do not load models"))
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "a.pdf").write_bytes(b"synthetic")
    assert extract.main([str(tmp_path)]) == 1  # Direct files only; no recursive scan.


def test_cli_single_pdf_remains_supported(tmp_path, monkeypatch):
    from rag import extract
    source, output = source_pdf(tmp_path), tmp_path / "single.md"
    monkeypatch.setattr(extract, "build_converter", lambda *a: converter({1: "Single file"}))
    assert extract.main([str(source), "-o", str(output)]) == 0
    assert "Single file" in output.read_text()


def test_real_docling_markdown_export_preserves_page_markers_and_table():
    doc = pytest.importorskip("docling_core.types.doc")
    # Real serializer test, without PDF inference/model downloads.
    document = doc.DoclingDocument(name="synthetic")
    for page in (1, 2):
        document.add_page(page_no=page, size=doc.Size(width=600, height=800))
    def provenance(page):
        return doc.ProvenanceItem(page_no=page,
            bbox=doc.BoundingBox(l=20, t=100, r=300, b=80, coord_origin=doc.CoordOrigin.BOTTOMLEFT), charspan=(0, 20))
    document.add_text(label=doc.DocItemLabel.SECTION_HEADER, text="Synthetic heading", prov=provenance(1))
    document.add_text(label=doc.DocItemLabel.TEXT, text="Nội dung tiếng Việt.", prov=provenance(1))
    document.add_table(data=doc.TableData(num_rows=1, num_cols=2, table_cells=[
        doc.TableCell(text="A", column_header=True, start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=1),
        doc.TableCell(text="B", column_header=True, start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=1, end_col_offset_idx=2),
    ]), prov=provenance(2))
    markdown, warnings = render_markdown(document)
    assert "<!-- page: 1 -->" in markdown and "<!-- page: 2 -->" in markdown
    assert markdown.index("Nội dung tiếng Việt.") < markdown.index("<!-- page: 2 -->")
    assert "|" in markdown.split("<!-- page: 2 -->")[1]
    assert not warnings


def test_merged_and_headerless_tables_preserve_structure_without_fake_headers():
    doc = pytest.importorskip("docling_core.types.doc")
    document = doc.DoclingDocument(name="table-regression")
    document.add_page(page_no=235, size=doc.Size(width=600, height=800))
    prov = doc.ProvenanceItem(page_no=235,
        bbox=doc.BoundingBox(l=10, t=100, r=300, b=20), charspan=(0, 10))
    document.add_table(data=doc.TableData(num_rows=2, num_cols=2, table_cells=[
        doc.TableCell(text="Shared label", row_span=2, start_row_offset_idx=0, end_row_offset_idx=2,
                      start_col_offset_idx=0, end_col_offset_idx=1),
        doc.TableCell(text="7,4/9", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=1, end_col_offset_idx=2),
        doc.TableCell(text="6,4/8", start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=1, end_col_offset_idx=2),
    ]), prov=prov)
    text, warnings = render_markdown(document)
    rows = [[c.strip() for c in line.split("|")[1:-1]] for line in text.splitlines() if line.startswith("|")]
    assert rows[0] == ["", ""]
    assert rows[2:] == [["Shared label", "7,4/9"], ["Shared label", "6,4/8"]]
    assert "<table" not in text
    assert "<!-- page: 235 -->" in text
    assert not warnings


def test_colspan_header_keeps_one_label_and_empty_answer_column():
    doc = pytest.importorskip("docling_core.types.doc")
    document = doc.DoclingDocument(name="column-regression")
    document.add_page(page_no=1, size=doc.Size(width=600, height=800))
    prov = doc.ProvenanceItem(page_no=1, bbox=doc.BoundingBox(l=10, t=100, r=300, b=20), charspan=(0, 10))
    document.add_table(data=doc.TableData(num_rows=2, num_cols=2, table_cells=[
        doc.TableCell(text="Grouped header", col_span=2, column_header=True,
            start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=2),
        doc.TableCell(text="Question", start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=0, end_col_offset_idx=1),
        doc.TableCell(text="", start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=1, end_col_offset_idx=2),
    ]), prov=prov)
    text, _ = render_markdown(document)
    rows = [[c.strip() for c in line.split("|")[1:-1]] for line in text.splitlines() if line.startswith("|")]
    assert rows[0] == ["Grouped header", "Grouped header"]
    assert rows[2] == ["Question", ""]
    assert "<table" not in text


def test_table_header_levels_spans_and_numeric_values_keep_row_context():
    doc = pytest.importorskip("docling_core.types.doc")
    document = doc.DoclingDocument(name="semantic-table")
    document.add_page(page_no=1, size=doc.Size(width=600, height=800))
    def cell(text, row, col, *, rows=1, cols=1, header=False):
        return doc.TableCell(text=text, row_span=rows, col_span=cols, column_header=header,
            start_row_offset_idx=row, end_row_offset_idx=row + rows,
            start_col_offset_idx=col, end_col_offset_idx=col + cols)
    table = document.add_table(data=doc.TableData(num_rows=4, num_cols=3, table_cells=[
        cell("Type", 0, 0, rows=2, header=True), cell("Allowed", 0, 1, cols=2, header=True),
        cell("A", 1, 1, header=True), cell("B", 1, 2, header=True),
        cell("Case X", 2, 0, rows=2), cell("2.10", 2, 1), cell("", 2, 2),
        cell("3.80", 3, 1), cell("0.001", 3, 2),
    ]), prov=doc.ProvenanceItem(page_no=1, bbox=doc.BoundingBox(l=10, t=100, r=300, b=20), charspan=(0, 10)))
    before = table.model_dump()
    text, _ = render_markdown(document)
    rows = [[c.strip() for c in line.split("|")[1:-1]] for line in text.splitlines() if line.startswith("|")]
    assert rows[0] == ["Type", "Allowed - A", "Allowed - B"]
    assert rows[2:] == [["Case X", "2.10", ""], ["Case X", "3.80", "0.001"]]
    assert table.model_dump() == before


def test_page_range_is_forwarded_and_physical_numbers_are_kept(tmp_path):
    source, output = source_pdf(tmp_path), tmp_path / "sample.md"
    calls = []
    def convert(path, *, page_range):
        calls.append(page_range)
        return SimpleNamespace(status="success", document=Document({235: "Page A", 236: "Page B"}))
    extract_pdf(source, output, page_range=(235, 236), converter=SimpleNamespace(convert=convert))
    assert calls == [(235, 236)]
    assert "<!-- page: 235 -->" in output.read_text() and "<!-- page: 1 -->" not in output.read_text()


def test_merged_source_blocks_restore_paragraphs_without_splitting_sentences():
    doc = pytest.importorskip("docling_core.types.doc")
    document = doc.DoclingDocument(name="paragraphs")
    document.add_page(page_no=1, size=doc.Size(width=600, height=800))
    first, second = "First sentence. Still the same paragraph.", "A separate source paragraph."
    bbox = doc.BoundingBox(l=10, t=100, r=300, b=20)
    item = document.add_text(label=doc.DocItemLabel.TEXT, text=first + " " + second,
        prov=doc.ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, len(first))))
    item.prov.append(doc.ProvenanceItem(page_no=1, bbox=bbox,
        charspan=(len(first) + 1, len(first) + 1 + len(second))))
    before = document.model_dump()
    text, _ = render_markdown(document)
    assert first + "\n\n" + second in text
    assert document.model_dump() == before


def test_cross_page_merged_paragraph_is_not_dumped_on_first_page():
    doc = pytest.importorskip("docling_core.types.doc")
    document = doc.DoclingDocument(name="merged-paragraph")
    for page in (2, 3):
        document.add_page(page_no=page, size=doc.Size(width=600, height=800))
    bbox = doc.BoundingBox(l=10, t=100, r=300, b=20)
    item = document.add_text(label=doc.DocItemLabel.TEXT, text="LEFTPAGE RIGHTPAGE",
        prov=doc.ProvenanceItem(page_no=2, bbox=bbox, charspan=(0, 8)))
    item.prov.append(doc.ProvenanceItem(page_no=3, bbox=bbox, charspan=(9, 18)))
    before = document.model_dump()
    text, warnings = render_markdown(document)
    first, second = text.split("<!-- page: 3 -->")
    assert "LEFTPAGE" in first and "RIGHTPAGE" not in first
    assert "RIGHTPAGE" in second and "LEFTPAGE" not in second
    assert document.model_dump() == before  # Temporary page projection must not corrupt the document.
    assert not warnings


@pytest.mark.parametrize("mode", ["auto", "off", "force"])
def test_real_converter_configuration_without_loading_models(mode):
    pytest.importorskip("docling.document_converter")
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import OcrMode
    service = build_converter(mode, device="cpu")
    options = service.format_to_options[InputFormat.PDF].pipeline_options
    assert options.do_table_structure and not options.enable_remote_services
    assert options.do_ocr == (mode != "off")
    assert options.accelerator_options.device == "cpu"
    assert options.layout_batch_size == options.ocr_batch_size == options.table_batch_size == 1
    assert options.heading_hierarchy_options.enabled
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    assert service.format_to_options[InputFormat.PDF].backend is PyPdfiumDocumentBackend
    if mode != "off":
        assert options.ocr_options.lang == ["vi", "en"]
        assert options.ocr_options.mode == (OcrMode.FULL_PAGE if mode == "force" else OcrMode.DEFAULT)
        assert options.ocr_options.use_gpu is None


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda"])
def test_device_selection_is_delegated_to_docling(device):
    pytest.importorskip("docling.document_converter")
    from docling.datamodel.base_models import InputFormat
    service = build_converter(device=device)
    options = service.format_to_options[InputFormat.PDF].pipeline_options
    assert options.accelerator_options.device == device
    assert options.ocr_options.use_gpu is None


def test_table_line_breaks_and_caption_position_survive_markdown_export():
    doc = pytest.importorskip("docling_core.types.doc")
    document = doc.DoclingDocument(name="multiline-table")
    document.add_page(page_no=3, size=doc.Size(width=600, height=800))
    caption = document.add_text(label=doc.DocItemLabel.CAPTION, text="Note after table",
        prov=doc.ProvenanceItem(page_no=3, bbox=doc.BoundingBox(l=10, t=80, r=300, b=60,
            coord_origin=doc.CoordOrigin.BOTTOMLEFT), charspan=(0, 16)))
    document.add_table(data=doc.TableData(num_rows=2, num_cols=2, table_cells=[
        doc.TableCell(text=text, column_header=row == 0, start_row_offset_idx=row, end_row_offset_idx=row + 1,
            start_col_offset_idx=col, end_col_offset_idx=col + 1)
        for row, values in enumerate([["Drug", "Dose"], ["Drug A\r\nDrug B", "Dose A\nDose B"]])
        for col, text in enumerate(values)]), caption=caption,
        prov=doc.ProvenanceItem(page_no=3, bbox=doc.BoundingBox(l=10, t=500, r=300, b=100,
            coord_origin=doc.CoordOrigin.BOTTOMLEFT), charspan=(0, 0)))
    before = document.model_dump()
    text, _ = render_markdown(document)
    assert "Drug A<br>Drug B | Dose A<br>Dose B" in text
    assert text.index("Drug A") < text.index("Note after table")
    assert document.model_dump() == before


def test_physical_pdf_page_count_mismatch_is_rejected():
    document = Document({1: "text", 2: "more text"})
    with pytest.raises(ExtractionError, match="page count/range"):
        render_markdown(document, pdf=[None, None, None])


@pytest.mark.parametrize("case", ["matching", "ambiguous", "wrong_page", "empty_cell", "partial_text"])
def test_native_geometry_refines_only_a_unique_readable_table(monkeypatch, case):
    from rag.extract import refine_ruled_tables
    pdfplumber = pytest.importorskip("pdfplumber")
    doc = pytest.importorskip("docling_core.types.doc")
    document = doc.DoclingDocument(name="refine-table")
    document.add_page(page_no=1, size=doc.Size(width=600, height=800))
    table = document.add_table(data=doc.TableData(num_rows=1, num_cols=2, table_cells=[
        doc.TableCell(text=["Drug A Drug B", "Dose A Dose B"][col], start_row_offset_idx=0, end_row_offset_idx=1,
            start_col_offset_idx=col, end_col_offset_idx=col + 1) for col in range(2)]),
        prov=doc.ProvenanceItem(page_no=1, bbox=doc.BoundingBox(l=10, t=500, r=300, b=100,
            coord_origin=doc.CoordOrigin.BOTTOMLEFT), charspan=(0, 0)))
    boxes = [(10, 300, 150, 700), (150, 300, 300, 500), (150, 500, 300, 700)]
    values = [["Drug A\nDrug B", "Dose A"], [None, "Dose B"]]
    if case == "empty_cell":
        values = [[None, None], [None, None]]
    if case == "partial_text":
        values = [["Drug A", "Dose A"], [None, "Dose B"]]
    candidate = SimpleNamespace(bbox=(10, 300, 300, 700), cells=boxes,
        rows=[SimpleNamespace(cells=boxes[:2]), SimpleNamespace(cells=[None, boxes[2]])],
        extract=lambda: values)
    candidates = [] if case == "wrong_page" else [candidate, candidate] if case == "ambiguous" else [candidate]
    class Pdf:
        pages = [SimpleNamespace(find_tables=lambda: candidates)]
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(pdfplumber, "open", lambda path: Pdf())
    before = document.model_dump()
    count = refine_ruled_tables(document, "unused.pdf")
    if case != "matching":
        assert count == 0 and document.model_dump() == before
    else:
        assert count == 1
        assert table.data.num_rows == table.data.num_cols == 2
        assert [c.text for c in table.data.table_cells] == ["Drug A\nDrug B", "Dose A", "Dose B"]
        assert table.data.table_cells[0].row_span == 2
        assert table.data.grid[1][0].text == "Drug A\nDrug B"
        from rag.extract import table_rows
        text = table_rows(table)
        assert text.count("- Bảng") == 1  # A vertical span binds its rows into one group.
        assert "Dose A" in text and "Dose B" in text
        assert "1–2" in text


def test_embedding_rows_keep_column_context_blanks_and_multivalue_groups():
    from rag.extract import table_rows
    from rag.chunk import chunk_markdown
    doc = pytest.importorskip("docling_core.types.doc")
    document = doc.DoclingDocument(name="row-context")
    document.add_page(page_no=1, size=doc.Size(width=600, height=800))
    table = document.add_table(data=doc.TableData(num_rows=3, num_cols=2, table_cells=[
        doc.TableCell(text=text, column_header=r == 0, start_row_offset_idx=r, end_row_offset_idx=r + 1,
            start_col_offset_idx=c, end_col_offset_idx=c + 1)
        for r, values in enumerate([["Drug", "Dose"], ["Drug A\nDrug B", "Dose A\nDose B"], ["Drug C", ""]])
        for c, text in enumerate(values)]), prov=doc.ProvenanceItem(page_no=1,
            bbox=doc.BoundingBox(l=10, t=100, r=300, b=20), charspan=(0, 0)))
    before = document.model_dump()
    text, _ = render_markdown(document, table_format="rows")
    assert "<br>" not in text and "| Drug" not in text
    assert "Drug:\n    Drug A\n    Drug B\n  Dose:\n    Dose A\n    Dose B" in text
    assert "(ô trống trong tài liệu)" in text
    assert text.count("- Bảng") == 2 and text.count("Drug A") == 1
    assert table_rows(table) in text
    chunks = chunk_markdown("## Context\n\n" + text, document_id="test", source="test.md", max_tokens=64)
    matching = [c for c in chunks if "Drug A" in c["text"]]
    assert len(matching) == 1 and "Dose B" in matching[0]["text"]
    assert "Context" in matching[0]["text"]
    assert document.model_dump() == before


def test_merged_group_keeps_each_criterion_with_its_score_and_distinct_labels():
    from rag.extract import table_rows
    from rag.chunk import chunk_markdown
    from markdown_it import MarkdownIt
    doc = pytest.importorskip("docling_core.types.doc")
    def cell(text, row, col, rows=1, cols=1, header=False):
        return doc.TableCell(text=text, row_span=rows, col_span=cols, column_header=header,
            start_row_offset_idx=row, end_row_offset_idx=row + rows,
            start_col_offset_idx=col, end_col_offset_idx=col + cols)
    document = doc.DoclingDocument(name="paired-scores")
    document.add_page(page_no=1, size=doc.Size(width=600, height=800))
    table = document.add_table(data=doc.TableData(num_rows=3, num_cols=3, table_cells=[
        cell("Criterion", 0, 0, cols=2, header=True), cell("Score", 0, 2, header=True),
        cell("Shared group", 1, 0, rows=2), cell("Range A", 1, 1), cell("5", 1, 2),
        cell("Range B", 2, 1), cell("10", 2, 2),
    ]), prov=doc.ProvenanceItem(page_no=1, bbox=doc.BoundingBox(l=10, t=100, r=300, b=20), charspan=(0, 0)))
    before = document.model_dump()
    text = table_rows(table)
    assert text.count("Shared group") == 1
    assert "Criterion (cột 1)" in text and "Criterion (cột 2)" in text
    lines = text.splitlines(keepends=True)
    details = ["".join(lines[t.map[0]:t.map[1]]) for t in MarkdownIt().parse(text)
               if t.type == "list_item_open" and t.level == 3]
    assert len(details) == 2
    assert "Range A" in details[0] and "\n      5" in details[0] and "Range B" not in details[0]
    assert "Range B" in details[1] and "\n      10" in details[1] and "Range A" not in details[1]
    chunks = chunk_markdown("## Context\n\n" + text, document_id="test", source="test.md", max_tokens=64)
    assert len(chunks) == 1 and "Shared group" in chunks[0]["text"]
    assert "Range A" in chunks[0]["text"] and "Range B" in chunks[0]["text"]
    assert document.model_dump() == before


def continuation_document():
    doc = pytest.importorskip("docling_core.types.doc")
    document = doc.DoclingDocument(name="continuation")
    for page in (1, 2, 3):
        document.add_page(page_no=page, size=doc.Size(width=600, height=800))
    def provenance(page, top, bottom):
        return doc.ProvenanceItem(page_no=page,
            bbox=doc.BoundingBox(l=40, t=top, r=560, b=bottom, coord_origin=doc.CoordOrigin.TOPLEFT),
            charspan=(0, 0))
    caption = document.add_text(label=doc.DocItemLabel.CAPTION, text="Original table title",
                                prov=provenance(1, 490, 510))
    for page in (1, 2):
        top = 520 if page == 1 else 50
        cells = [doc.TableCell(text=text, column_header=row == 0,
            start_row_offset_idx=row, end_row_offset_idx=row + 1,
            start_col_offset_idx=col, end_col_offset_idx=col + 1,
            bbox=doc.BoundingBox(l=40 if col == 0 else 300, r=300 if col == 0 else 560,
                t=top + row * 20, b=top + 20 if row == 0 else 750, coord_origin=doc.CoordOrigin.TOPLEFT))
            for row, values in enumerate([["Criterion", "Score"], [f"Value {page}", str(page)]])
            for col, text in enumerate(values)]
        document.add_table(data=doc.TableData(num_rows=2, num_cols=2, table_cells=cells),
            prov=provenance(page, top, 750), caption=caption if page == 1 else None)
        document.add_text(label=doc.DocItemLabel.PAGE_FOOTER, text=str(page), prov=provenance(page, 780, 790))
    return document


def test_continuation_shares_identity_and_title_without_mutating_source():
    from rag.extract import table_contexts
    document = continuation_document()
    before = document.model_dump()
    first, second = document.tables
    contexts = table_contexts(document)
    assert contexts[second.self_ref]["id"] == contexts[first.self_ref]["id"]
    assert contexts[second.self_ref]["title"] == "Original table title"
    assert contexts[second.self_ref]["continued_from"] == 1
    text, warnings = render_markdown(document, table_format="rows")
    assert text.count("table_id: table_0;") == 2
    assert "part_id: table_1; continuation: inferred" in text
    assert "Original table title" in text.split("<!-- page: 2 -->")[1]
    assert "<!-- page: 2 -->" in text and any("continuation inferred" in w for w in warnings)
    assert document.model_dump() == before


def test_continuation_chain_retains_root_title_and_page_local_parts():
    from rag.extract import table_contexts
    document = continuation_document()
    second = document.tables[1]
    third = document.add_table(data=second.data.model_copy(deep=True),
        prov=second.prov[0].model_copy(update={"page_no": 3}))
    contexts = table_contexts(document)
    assert contexts[third.self_ref]["id"] == contexts[document.tables[0].self_ref]["id"]
    assert contexts[third.self_ref]["title"] == "Original table title"
    assert contexts[third.self_ref]["continued_from"] == 2


@pytest.mark.parametrize("variant", ["new_caption", "heading", "body_text", "columns", "headers",
                                      "no_anchor", "bottom_gap", "top_gap", "page_gap"])
def test_similar_tables_are_not_joined_without_all_continuation_signals(variant):
    from rag.extract import table_contexts
    doc = pytest.importorskip("docling_core.types.doc")
    document = continuation_document()
    first, second = document.tables
    if variant == "new_caption":
        caption = document.add_text(label=doc.DocItemLabel.CAPTION, text="A new independent table")
        second.captions.append(caption.get_ref())
    elif variant in {"heading", "body_text"}:
        document.add_text(label=doc.DocItemLabel.SECTION_HEADER if variant == "heading" else doc.DocItemLabel.TEXT,
            text="A different section" if variant == "heading" else "Intervening explanation",
            prov=doc.ProvenanceItem(page_no=2, bbox=doc.BoundingBox(l=40, t=20, r=560, b=40,
                coord_origin=doc.CoordOrigin.TOPLEFT), charspan=(0, 0)))
    elif variant == "columns":
        for cell in second.data.table_cells:
            if cell.start_col_offset_idx == 0:
                cell.bbox.r = 400
            else:
                cell.bbox.l = 400
    elif variant == "headers":
        second.data.table_cells[0].text = "Different header"
    elif variant == "no_anchor":
        first.captions.clear()
    elif variant == "bottom_gap":
        first.prov[0].bbox.b = 600
    elif variant == "top_gap":
        second.prov[0].bbox.t = 200
    elif variant == "page_gap":
        second.prov[0].page_no = 3
    contexts = table_contexts(document)
    assert contexts[second.self_ref]["id"] != contexts[first.self_ref]["id"]
    assert contexts[second.self_ref]["status"] != "inferred"
    if variant not in {"new_caption", "headers", "page_gap"}:
        assert contexts[second.self_ref]["status"] == "unconfirmed"
