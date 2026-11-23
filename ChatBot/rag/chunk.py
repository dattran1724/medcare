"""Split at actual Markdown headings; preserve source sections and their parts."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import tempfile


def split_large_section(text, prefix, max_tokens, count, sentence_tokenizer):
    """Split at Markdown blocks first, then sentences/items/rows; retain source spans."""
    from markdown_it import MarkdownIt
    parser = MarkdownIt().enable("table")
    lines = text.splitlines(keepends=True)
    blocks = [t for t in parser.parse(text) if t.level == 0 and t.nesting != -1 and t.map]
    units, index = [], 0
    while index < len(blocks):
        block, intro, intro_spans = blocks[index], "", []
        index += 1
        if block.type == "heading_open":
            continue  # The full heading path is repeated in prefix.
        if block.type == "paragraph_open" and index < len(blocks) and blocks[index].type in {"table_open", "bullet_list_open", "ordered_list_open"}:
            intro = "".join(lines[block.map[0]:block.map[1]]).rstrip() + "\n\n"
            intro_spans = [tuple(block.map)]
            block = blocks[index]
            index += 1
        a, b = block.map
        body = "".join(lines[a:b]).rstrip("\r\n")
        fits = lambda value: count(prefix + "\n\n" + intro + value) <= max_tokens
        if fits(body):
            fragments = [(body, [(a, b)])]
        elif block.type == "paragraph_open":
            spans = list(sentence_tokenizer.span_tokenize(body))
            fragments = [(body[x:y], [(a + body[:x].count("\n"), a + body[:y].count("\n") + 1)]) for x, y in spans]
        elif block.type in {"bullet_list_open", "ordered_list_open"}:
            fragments = [("".join(body.splitlines(keepends=True)[x:y]).rstrip("\r\n"), [(a + x, a + y)])
                         for t in parser.parse(body) if t.type == "list_item_open" and t.level == 1 and t.map
                         for x, y in [t.map]]
        elif block.type == "table_open":
            rows = body.splitlines()
            header = "\n".join(rows[:2])
            fragments, group, row_start = [], [], 2
            for row_index, row in enumerate(rows[2:], 2):
                if group and not fits(header + "\n" + "\n".join(group + [row])):
                    fragments.append((header + "\n" + "\n".join(group), [(a, a + 2), (a + row_start, a + row_index)]))
                    group, row_start = [], row_index
                group.append(row)
            if group:
                fragments.append((header + "\n" + "\n".join(group), [(a, a + 2), (a + row_start, b)]))
        else:
            fragments = [(body, [(a, b)])]  # Code and other indivisible blocks.
        for value, spans in fragments or [(body, [(a, b)])]:
            units.append((intro + value, intro_spans + spans))

    parts, group, spans = [], [], []
    for value, source_spans in units:
        if group and count(prefix + "\n\n" + "\n\n".join(group + [value])) > max_tokens:
            parts.append((prefix + "\n\n" + "\n\n".join(group), spans))
            group, spans = [], []
        group.append(value)
        spans += source_spans
    if group:
        parts.append((prefix + "\n\n" + "\n\n".join(group), spans))
    return parts


def chunk_markdown(markdown, *, document_id, source, max_tokens=1200):
    from markdown_it import MarkdownIt

    if max_tokens < 64:
        raise ValueError("max_tokens must be >=64")
    import tiktoken
    from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer
    parameters = PunktParameters()
    parameters.abbrev_types = {"bs", "ts", "ths", "gs", "pgs", "gs.ts", "pgs.ts", "dr", "mr", "mrs", "e.g", "i.e"}
    sentence_tokenizer = PunktSentenceTokenizer(parameters)
    encoder = tiktoken.get_encoding("cl100k_base")
    count = lambda value: len(encoder.encode(value, disallowed_special=()))
    lines = markdown.splitlines(keepends=True)
    # Read page markers from actual comments, never from fenced code examples.
    markers, ignored = {}, set()
    tokens = MarkdownIt().parse(markdown)
    for token in tokens:
        if token.type == "html_block" and token.level == 0 and token.map:
            match = re.fullmatch(r"\s*<!--\s*page:\s*(\d+)\s*-->\s*", token.content)
            if match:
                markers[token.map[0]] = int(match[1])
            if token.content.strip().startswith("<!--") and token.content.strip().endswith("-->"):
                ignored.update(range(*token.map))
    page, pages_by_line = None, []
    for index in range(len(lines)):
        if index in markers:
            number = markers[index]
            if number < 1 or (page is not None and number <= page):
                raise ValueError("Page markers must be positive and strictly increasing")
            page = number
        pages_by_line.append(page)

    # Source positions, not title spelling, identify each section occurrence.
    sections, stack, start, path = [], [], 0, []
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.level != 0 or not token.map:
            continue
        line, depth = token.map[0], int(token.tag[1:])
        if line > start:
            sections.append((start, line, path))
        while stack and stack[-1][0] >= depth:
            stack.pop()
        stack.append((depth, tokens[index + 1].content))
        start, path = line, [title for _, title in stack]
    sections.append((start, len(lines), path))
    checksum, chunks = sha256(markdown.encode("utf-8")).hexdigest(), []
    for start, end, path in sections:
        clean_lines = [lines[i] if i not in ignored else "\n" for i in range(start, end)]
        text = "".join(clean_lines).strip("\r\n")
        if not text.strip():
            continue
        # Keep line offsets unchanged for the parser even when comments are blanked.
        section = "".join(clean_lines)
        parts = [(text, [(0, len(clean_lines))])]
        if count(text) > max_tokens:
            parts = split_large_section(section, "\n".join(path), max_tokens, count, sentence_tokenizer) or parts
        section_id = sha256(f"{document_id}:{checksum}:{start}:{end}".encode()).hexdigest()[:24]
        section_pages = sorted({pages_by_line[i] for i in range(start, end)
                                if i not in ignored and lines[i].strip() and pages_by_line[i] is not None})
        for part_index, (value, spans) in enumerate(parts):
            source_indexes = sorted({start + i for a, b in spans for i in range(a, b)
                                     if start + i < end and start + i not in ignored and lines[start + i].strip()})
            pages = sorted({pages_by_line[i] for i in source_indexes if pages_by_line[i] is not None})
            size = count(value)
            warnings = ["oversized_preserved_for_review"] if size > max_tokens else []
            chunks.append(dict(document_id=document_id, source=source, source_sha256=checksum,
                chunk_id=sha256(f"{document_id}:{checksum}:{len(chunks)}:{value}".encode()).hexdigest()[:24],
                chunk_index=len(chunks), text=value, heading_path=path, section_id=section_id,
                section_path=path, part_index=part_index, part_count=len(parts),
                section_line_start=start + 1, section_line_end=end,
                pages=pages, page_start=pages[0] if pages else None, page_end=pages[-1] if pages else None,
                line_start=min(source_indexes) + 1, line_end=max(source_indexes) + 1,
                token_count=size, tokenizer="cl100k_base", needs_review=bool(warnings), warnings=warnings))
            if part_index == 0:
                # One canonical copy per section lets retrieval remove repeated context exactly.
                chunks[-1].update(section_text=text, section_pages=section_pages)
    for record in chunks:
        record["document_chunk_count"] = len(chunks)
    return chunks


def chunk_file(source, output=None, *, overwrite=False, max_tokens=1200):
    source = Path(source).resolve()
    if not source.is_file() or source.suffix.lower() != ".md":
        raise ValueError("Source must be an existing Markdown file")
    output = (Path(output) if output else Path(__file__).parent / "data" / "chunks" / (source.stem + ".jsonl")).resolve()
    if output == source or output.suffix.lower() != ".jsonl":
        raise ValueError("Output must be a separate .jsonl file")
    if output.exists() and not overwrite:
        raise ValueError("Output exists; use --overwrite")
    original = source.read_bytes()
    markdown = original.decode("utf-8-sig")
    chunks = chunk_markdown(markdown, document_id=source.stem, source=source.name, max_tokens=max_tokens)
    if not chunks:
        raise ValueError("No chunkable content")
    for item in chunks:
        item["source_sha256"] = sha256(original).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=output.parent,
                prefix=".chunk-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for chunk in chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False, allow_nan=False) + "\n")
        if overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return output, chunks


def main(argv=None):
    parser = argparse.ArgumentParser(description="Markdown headings to JSONL chunks")
    parser.add_argument("source", nargs="?", type=Path, default=Path(__file__).parent / "data" / "markdown")
    parser.add_argument("-o", "--output", type=Path, help="JSONL file, or output folder for batch")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1200)
    args = parser.parse_args(argv)
    batch = args.source.is_dir()
    if batch and args.output and args.output.suffix.lower() == ".jsonl":
        parser.error("Batch output must be a folder")
    files = sorted(p for p in args.source.iterdir() if p.is_file() and p.suffix.lower() == ".md") if batch else [args.source]
    if not files:
        parser.error("No Markdown files found")
    failed = 0
    for source in files:
        output = ((args.output or Path(__file__).parent / "data" / "chunks") / (source.stem + ".jsonl")) if batch else args.output
        if batch and output.exists() and not args.overwrite:
            print(f"Skipped: {output}")
            continue
        try:
            path, chunks = chunk_file(source, output, overwrite=args.overwrite, max_tokens=args.max_tokens)
            print(f"Saved: {path} ({len(chunks)} chunks; {sum(c['needs_review'] for c in chunks)} oversized)")
        except (ValueError, OSError, ImportError) as error:
            print(f"Error [{source.name}]: {error}", file=sys.stderr)
            failed += 1
    return int(failed > 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
