"""Local Docling PDF-to-Markdown conversion, with physical page markers."""
import argparse
import os
import re
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path


class ExtractionError(RuntimeError):
    """Conversion failed; no successful Markdown output was written."""


def build_converter(ocr="auto", device="auto"):
    """Let Docling select the accelerator and recognize layout, OCR and tables."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.pipeline_options import EasyOcrOptions, OcrMode, PdfPipelineOptions, TableFormerMode
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError:
        raise ExtractionError('Install PDF dependencies: uv pip install --python .venv/Scripts/python.exe -e ".[documents]"') from None
    options = PdfPipelineOptions(do_ocr=ocr != "off", do_table_structure=True,
        accelerator_options=AcceleratorOptions(device=device),
        layout_batch_size=1, ocr_batch_size=1, table_batch_size=1)
    if ocr != "off":
        options.ocr_options = EasyOcrOptions(lang=["vi", "en"],
            mode=OcrMode.FULL_PAGE if ocr == "force" else OcrMode.DEFAULT)
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    options.heading_hierarchy_options.enabled = True
    options.enable_remote_services = False
    return DocumentConverter(allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options,
                                                       backend=PyPdfiumDocumentBackend)})


def refine_ruled_tables(document, source):
    """Read ruled-table cells from native PDF coordinates, including true spans."""
    import pdfplumber
    from docling_core.types.doc import BoundingBox, CoordOrigin, TableCell, TableData

    replaced = 0
    with pdfplumber.open(source) as pdf:
        detected = {}
        for table in document.tables:
            if len(table.prov) != 1:
                continue
            page = table.prov[0].page_no
            box = table.prov[0].bbox.to_top_left_origin(document.pages[page].size.height)
            if page not in detected:
                detected[page] = pdf.pages[page - 1].find_tables()
            matches = []
            for candidate in detected[page]:
                l, t, r, b = candidate.bbox
                intersection = max(0, min(r, box.r) - max(l, box.l)) * max(0, min(b, box.b) - max(t, box.t))
                union = (r - l) * (b - t) + (box.r - box.l) * (box.b - box.t) - intersection
                if union > 0 and intersection / union >= 0.8:
                    matches.append(candidate)
            if len(matches) != 1:
                continue
            candidate = matches[0]
            values = candidate.extract()
            if not any(value for row in values for value in row):
                continue  # A native parser must not blank content recovered by OCR.
            characters = lambda text: Counter(re.sub(r"\W", "", text).casefold())
            original = characters(" ".join(c.text for c in table.data.table_cells))
            native = characters(" ".join(value or "" for row in values for value in row))
            if original - native:
                continue  # Reject partial native text, too; preserve the OCR/model result.
            xs = sorted({x for cell in candidate.cells for x in (cell[0], cell[2])})
            ys = sorted({y for cell in candidate.cells for y in (cell[1], cell[3])})
            headers = max((c.end_row_offset_idx for c in table.data.table_cells if c.column_header), default=0)
            cells = []
            for row_index, row in enumerate(candidate.rows):
                for col_index, cell in enumerate(row.cells):
                    if cell is None:
                        continue  # Covered by a geometric span; never forward-fill unknowns.
                    l, t, r, b = cell
                    x0, x1, y0, y1 = xs.index(l), xs.index(r), ys.index(t), ys.index(b)
                    cells.append(TableCell(text=values[row_index][col_index] or "",
                        start_row_offset_idx=y0, end_row_offset_idx=y1, row_span=y1 - y0,
                        start_col_offset_idx=x0, end_col_offset_idx=x1, col_span=x1 - x0,
                        column_header=y0 < headers,
                        bbox=BoundingBox(l=l, t=t, r=r, b=b, coord_origin=CoordOrigin.TOPLEFT)))
            table.data = TableData(num_rows=len(ys) - 1, num_cols=len(xs) - 1, table_cells=cells)
            replaced += 1
    return replaced


def table_headers(item):
    """Return the leading header rows and literal per-column labels."""
    grid = item.data.grid
    header_count = 0
    while header_count < len(grid) and any(c.column_header for c in grid[header_count]):
        header_count += 1
    labels = [" - ".join(dict.fromkeys(grid[r][c].text.strip() for r in range(header_count)
                                      if grid[r][c].text.strip())) or f"Cột {c + 1} (chưa có tiêu đề)"
              for c in range(item.data.num_cols)]
    return header_count, labels


def table_contexts(document):
    """Infer page continuations conservatively; never merge table cell data."""
    context = {t.self_ref: {"id": t.self_ref.rsplit("/", 1)[-1],
               "title": " ".join(t.caption_text(document).split()), "status": "separate"}
               for t in document.tables}
    tables = sorted((t for t in document.tables if len(t.prov) == 1),
                    key=lambda t: (t.prov[0].page_no,
                        t.prov[0].bbox.to_top_left_origin(document.pages[t.prov[0].page_no].size.height).t))

    def columns(table):
        header_count, _ = table_headers(table)
        grid = table.data.grid
        if header_count >= len(grid) or any(c.bbox is None for c in grid[header_count]):
            return ()
        row = grid[header_count]
        width = document.pages[table.prov[0].page_no].size.width
        edges = [row[0].bbox.l / width] + [c.bbox.r / width for c in row]
        return edges if all(a < b for a, b in zip(edges, edges[1:])) else ()

    for previous, current in zip(tables, tables[1:]):
        p, q = previous.prov[0].page_no, current.prov[0].page_no
        info = context[current.self_ref]
        h1, labels1 = table_headers(previous)
        h2, labels2 = table_headers(current)
        normalize = lambda labels: [" ".join(s.split()).casefold() for s in labels]
        if q != p + 1 or info["title"] or not h1 or not h2 or normalize(labels1) != normalize(labels2):
            continue
        info["status"] = "unconfirmed"
        size1, size2 = document.pages[p].size, document.pages[q].size
        box1 = previous.prov[0].bbox.to_top_left_origin(size1.height)
        box2 = current.prov[0].bbox.to_top_left_origin(size2.height)
        edges1, edges2 = columns(previous), columns(current)
        if (not context[previous.self_ref]["title"] or box1.b < 0.85 * size1.height
                or box2.t > 0.15 * size2.height or not edges1 or len(edges1) != len(edges2)
                or any(abs(a - b) > 0.02 for a, b in zip(edges1, edges2))):
            continue
        # Body text or a new heading between the fragments breaks the chain.
        blocked = False
        for item in document.texts:
            if item.label.value in {"page_header", "page_footer"} or not item.text.strip():
                continue
            for prov in item.prov:
                if prov.page_no not in {p, q}:
                    continue
                box = prov.bbox.to_top_left_origin(document.pages[prov.page_no].size.height)
                if (prov.page_no == p and box.t >= box1.b) or (prov.page_no == q and box.b <= box2.t):
                    blocked = True
        if not blocked:
            info.update(id=context[previous.self_ref]["id"], title=context[previous.self_ref]["title"],
                        status="inferred", continued_from=p)
    return context


def table_rows(item, document=None, context=None):
    """Keep each physical row's field relationships inside its merged-cell group."""
    grid = item.data.grid
    header_count, labels = table_headers(item)
    counts = Counter(labels)
    labels = [f"{label} (cột {c + 1})" if counts[label] > 1 else label for c, label in enumerate(labels)]
    part_id = item.self_ref.rsplit("/", 1)[-1]
    context = context or {}
    identity = context.get("id", part_id)
    title = context.get("title", item.caption_text(document) if document is not None else "")
    title = " ".join(title.split()) or "Bảng chưa có chú thích"
    rows = []

    def fields(row, columns, indent):
        result = []
        for col in columns:
            value = row[col].text.strip() or "(ô trống trong tài liệu)"
            result.append(" " * indent + labels[col].replace("\n", " ") + ":")
            result.extend(" " * (indent + 2) + line.strip() for line in value.splitlines() if line.strip())
        return result

    index = header_count
    while index < len(grid):
        stop = index + 1
        for row_index in range(index, len(grid)):
            if row_index >= stop:
                break
            stop = max(stop, *(cell.end_row_offset_idx for cell in grid[row_index]))
        position = str(index + 1) if stop == index + 1 else f"{index + 1}–{stop}"
        lines = [f"- {title} — hàng nguồn {position}."]
        columns = range(item.data.num_cols)
        if stop == index + 1:
            lines.extend(fields(grid[index], columns, 2))
        else:
            shared = [c for c in columns if grid[index][c].end_row_offset_idx >= stop]
            details = [c for c in columns if c not in shared]
            lines.extend(fields(grid[index], shared, 2))
            if details:
                for row_index in range(index, stop):
                    lines.append(f"  - Hàng nguồn {row_index + 1}:")
                    lines.extend(fields(grid[row_index], details, 4))
        rows.append("\n".join(lines))
        index = stop
    metadata = f"table_id: table_{identity}; part_id: table_{part_id}"
    if context.get("status") in {"inferred", "unconfirmed"}:
        metadata += f"; continuation: {context['status']}"
    if "continued_from" in context:
        metadata += f"; continued_from_page: {context['continued_from']}"
    return f"<!-- {metadata} -->\n\n" + "\n\n".join(rows)


def page_renderer(document, table_format="markdown", contexts=None):
    """Normalize detected cells, then let Docling write Markdown with stable column context."""
    if not hasattr(document, "iterate_items"):
        return lambda page: document.export_to_markdown(page_no=page)
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams, MarkdownTableSerializer
    from docling_core.types.doc import TableCell
    from docling_core.transforms.serializer.common import create_ser_result

    class TableSerializer(MarkdownTableSerializer):
        def serialize(self, *, item, doc_serializer, doc, **kwargs):
            original = item
            # GFM cells cannot contain physical newlines. Keep their boundaries
            # with inline breaks instead of Docling's default space replacement.
            item = item.model_copy(update={"data": item.data.model_copy(update={"table_cells": [
                cell.model_copy(update={"text": cell.text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")})
                if isinstance(cell, TableCell) else cell for cell in item.data.table_cells]})})
            if not any(c.column_header for c in item.data.table_cells):
                # GFM requires a header. Use blank cells, never promote the first data row.
                headers = [TableCell(text="", column_header=True, start_row_offset_idx=0, end_row_offset_idx=1,
                    start_col_offset_idx=col, end_col_offset_idx=col + 1) for col in range(item.data.num_cols)]
                cells = [cell.model_copy(update={"start_row_offset_idx": cell.start_row_offset_idx + 1,
                    "end_row_offset_idx": cell.end_row_offset_idx + 1}) for cell in item.data.table_cells]
                item = item.model_copy(update={"data": item.data.model_copy(update={
                    "num_rows": item.data.num_rows + 1, "table_cells": headers + cells})})
            # Docling's grid expands only actual spans, joins header levels and preserves
            # numeric strings. Real empty cells stay empty; no value-based forward filling.
            result = super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)
            if table_format == "rows":
                text = re.sub(r"(?m)^\|[^\n]*(?:\n\|[^\n]*)*",
                              lambda match: table_rows(original, doc, (contexts or {}).get(original.self_ref)),
                              result.text, count=1)
                return create_ser_result(text=text, span_source=original)
            return result

    return lambda page: MarkdownDocSerializer(doc=document, table_serializer=TableSerializer(),
        params=MarkdownParams(pages={page}, caption_placement="layout", compact_tables=True)).serialize().text


@contextmanager
def page_text(document, page):
    """Restore source block boundaries and project merged text onto its actual page."""
    saved = []
    try:
        for item in getattr(document, "texts", []):
            if len(item.prov) < 2:
                continue
            saved.append((item, item.text, item.orig, item.prov))
            relevant = [p for p in item.prov if p.page_no == page]
            parts = [item.text[p.charspan[0]:p.charspan[1]] for p in relevant]
            # Separate detected source blocks, not sentences or wrapped PDF lines.
            separator = "\n\n" if item.label.value == "text" else " "
            item.text = separator.join(parts)
            item.orig = item.text
            item.prov = relevant
        yield
    finally:
        for item, text, orig, prov in saved:
            item.text, item.orig, item.prov = text, orig, prov


def render_markdown(document, *, pdf=None, page_range=None, table_format="markdown") -> tuple[str, list[str]]:
    """Export each actual page, including empty pages; never renumber missing content."""
    pages = sorted(document.pages)
    if not pages or pages[0] < 1 or pages != list(range(pages[0], pages[-1] + 1)):
        raise ExtractionError("Conversion has missing or invalid page numbers; output was not written")
    if pdf is not None:
        first, last = page_range or (1, len(pdf))
        if pages != list(range(first, min(last, len(pdf)) + 1)):
            raise ExtractionError("PDF page count/range does not match conversion; output was not written")
    sections, warnings = [], []
    contexts = table_contexts(document) if table_format == "rows" and hasattr(document, "iterate_items") else {}
    export_page = page_renderer(document, table_format, contexts)
    for table in getattr(document, "tables", []):
        info = contexts.get(getattr(table, "self_ref", None), {})
        if info.get("status") in {"inferred", "unconfirmed"}:
            warnings.append(f"Page {table.prov[0].page_no}: table continuation {info['status']}; review source")
    has_text = False
    for page in pages:
        with page_text(document, page):
            text = export_page(page).strip()
            if pdf is not None:
                from rag.layout import separate_columns, separate_paragraphs
                native_page = pdf[page - 1]
                native_text = native_page.get_textpage()
                try:
                    text, repairs = separate_columns(document, page, native_text, text)
                    text = separate_paragraphs(document, page, native_text, text)
                finally:
                    native_text.close()
                    native_page.close()
                if repairs:
                    warnings.append(f"Page {page}: separated {repairs} fused column heading blocks; review reading order")
        # Image placeholders and HTML comments alone are not extracted text.
        visible = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()
        if not visible:
            warnings.append(f"Page {page}: no readable text; inspect the PDF or retry with --ocr force")
            text = text + "\n\n<!-- extraction: no readable text; review source page -->"
        else:
            has_text = True
        sections.append(f"<!-- page: {page} -->\n\n{text.strip()}")
    if not has_text:
        raise ExtractionError("No readable text extracted. Inspect the PDF or retry with --ocr force")
    if getattr(document, "pictures", []):
        warnings.append("Figures are represented by placeholders/captions, not interpreted; review them in the PDF")
    if any(len({p.page_no for p in item.prov}) > 1 for item in getattr(document, "tables", [])):
        warnings.append("A table spans multiple pages; page-wise Markdown can repeat it. Review before chunking")
    return "\n\n".join(sections) + "\n", warnings


def extract_pdf(source, output=None, *, ocr="auto", device="auto", overwrite=False,
                page_range=None, converter=None, tables="pdfplumber", table_format="markdown") -> tuple[Path, list[str]]:
    """Convert one PDF. Tests may inject a converter without downloading any models."""
    source = Path(source).resolve()
    if not source.is_file() or source.suffix.lower() != ".pdf":
        raise ExtractionError("Input must be an existing local .pdf file")
    if page_range is not None and (len(page_range) != 2 or any(type(p) is not int for p in page_range)
                                   or not 1 <= page_range[0] <= page_range[1]):
        raise ExtractionError("Page range must be two positive page numbers in increasing order")
    suffix = f".pages-{page_range[0]}-{page_range[1]}" if page_range else ""
    output = (Path(output) if output is not None else
              Path(__file__).parent / "data" / "markdown" / (source.stem + suffix + ".md")).resolve()
    if output == source or output.suffix.lower() != ".md":
        raise ExtractionError("Output must be a .md file distinct from the source PDF")
    if output.exists() and not overwrite:
        raise ExtractionError("Output already exists; use --overwrite to replace it")
    if ocr not in {"auto", "off", "force"}:
        raise ExtractionError("Choose OCR auto/off/force")
    if tables not in {"pdfplumber", "docling"} or table_format not in {"markdown", "rows"}:
        raise ExtractionError("Choose tables pdfplumber/docling and table-format markdown/rows")
    try:
        converter = converter if converter is not None else build_converter(ocr, device)
        result = converter.convert(source, **({"page_range": tuple(page_range)} if page_range else {}))
        status = getattr(result.status, "value", result.status)
        if status != "success":
            raise ExtractionError("PDF conversion did not fully succeed; output was not written")
        if hasattr(result.document, "iterate_items"):
            import pypdfium2
            if tables == "pdfplumber":
                refine_ruled_tables(result.document, source)
            with pypdfium2.PdfDocument(source) as native_pdf:
                markdown, warnings = render_markdown(result.document, pdf=native_pdf, page_range=page_range,
                                                     table_format=table_format)
        else:
            markdown, warnings = render_markdown(result.document, table_format=table_format)
    except ExtractionError:
        raise
    except Exception as error:
        raise ExtractionError(f"PDF conversion failed ({type(error).__name__}); check Docling/OCR setup and model downloads") from None
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                dir=output.parent, prefix=".extract-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(markdown)
        if overwrite:
            os.replace(temporary, output)
        else:
            # Same-filesystem link publishes the complete file and never replaces an existing file.
            os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output, warnings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PDF to Markdown with page markers; no chunking or embedding")
    parser.add_argument("source", type=Path, nargs="?", default=Path(__file__).parent / "data" / "pdf",
                        help="PDF file or folder; default: all PDFs directly in rag/data/pdf")
    parser.add_argument("-o", "--output", type=Path, help="Markdown file for single PDF, or output folder for batch")
    parser.add_argument("--ocr", choices=("auto", "off", "force"), default="auto",
                        help="auto: OCR detected image regions; off: text layer only; force: full-page OCR")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing Markdown only after successful conversion")
    parser.add_argument("--tables", choices=("pdfplumber", "docling"), default="pdfplumber",
                        help="pdfplumber: recover ruled-table geometry and native text; docling: keep model tables")
    parser.add_argument("--table-format", choices=("markdown", "rows"), default="markdown",
                        help="rows: label each source row with column headers for embedding")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="auto: use CUDA when available, otherwise CPU; cuda: require NVIDIA GPU")
    parser.add_argument("--pages", nargs=2, type=int, metavar=("START", "END"),
                        help="Convert only this inclusive physical PDF page range")
    args = parser.parse_args(argv)
    batch = args.source.is_dir()
    if not batch and (not args.source.is_file() or args.source.suffix.lower() != ".pdf"):
        print("Error: input must be an existing PDF file or folder", file=sys.stderr)
        return 1
    if batch and args.output is not None and (args.output.is_file() or args.output.suffix.lower() == ".md"):
        parser.error("Batch output must be a folder, not a Markdown file")
    files = (sorted((p for p in args.source.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
                    key=lambda p: p.name.casefold()) if batch else [args.source])
    if not files:
        print(f"Error: no PDF files found in {args.source}", file=sys.stderr)
        return 1
    output_dir = args.output if batch and args.output is not None else Path(__file__).parent / "data" / "markdown"
    service, saved, skipped, failed = None, 0, 0, 0
    for index, source in enumerate(files, 1):
        suffix = f".pages-{args.pages[0]}-{args.pages[1]}" if args.pages else ""
        output = output_dir / (source.stem + suffix + ".md") if batch else args.output
        if batch and output.is_file() and not args.overwrite:
            print(f"[{index}/{len(files)}] Skipped (already exists): {output}", flush=True)
            skipped += 1
            continue
        print(f"[{index}/{len(files)}] Processing: {source.name}", flush=True)
        try:
            # One converter reuses loaded models for the entire batch.
            if service is None:
                service = build_converter(args.ocr, args.device)
            path, warnings = extract_pdf(source, output, overwrite=args.overwrite,
                page_range=args.pages, converter=service, tables=args.tables, table_format=args.table_format)
        except (ExtractionError, OSError, ValueError) as error:
            print(f"Error [{source.name}]: {error}", file=sys.stderr, flush=True)
            failed += 1
            if service is None:
                break  # Setup failed; repeating it for every PDF cannot help.
            continue
        for warning in warnings:
            print(f"Warning [{source.name}]: {warning}", file=sys.stderr)
        print(f"Saved: {path}", flush=True)
        saved += 1
    print(f"Finished: {saved} saved, {skipped} skipped, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
