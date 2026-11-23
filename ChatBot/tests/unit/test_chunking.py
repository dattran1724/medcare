"""Markdown section boundaries, source preservation and local JSONL checks."""
from hashlib import sha256
import json

import pytest

pytest.importorskip("markdown_it")
pytest.importorskip("nltk")
pytest.importorskip("tiktoken")
from rag import chunk


def make(text, **options):
    return chunk.chunk_markdown(text, document_id="sample", source="sample.md", **options)


def test_headings_and_original_table_list_paragraph_format_are_preserved():
    content = "Only if condition X applies:\n\n- Parent condition.\n  - Nested instruction.\n\n| Name | Value |\n| --- | --- |\n| A | 2.10 |\n| B | 3.80 |"
    source = "<!-- page: 1 -->\n# Procedure\n\n## Restrictions\n" + content + "\n\n## Another section\nOther content."
    result = make(source)
    item = next(c for c in result if "Parent condition" in c["text"])
    assert content in item["text"]
    assert item["heading_path"] == ["Procedure", "Restrictions"]
    assert item["pages"] == [1]
    assert "Other content" not in item["text"]


def test_long_paragraph_is_not_cut_or_rewritten():
    paragraph = "Do not proceed unless the condition applies. " * 100
    result = make("# Heading\n\n" + paragraph)
    assert len(result) == 1
    assert paragraph in result[0]["text"]


def test_cross_page_section_and_marker_inside_code():
    source = "<!-- page: 1 -->\n# Section\nFirst part.\n\n```text\n<!-- page: 99 -->\n  preserve indentation\n```\n\n<!-- page: 2 -->\nContinuation."
    result = make(source)
    assert len(result) == 1
    assert result[0]["pages"] == [1, 2]
    assert "<!-- page: 99 -->\n  preserve indentation" in result[0]["text"]
    assert "<!-- page: 1 -->" not in result[0]["text"]
    assert "<!-- page: 2 -->" not in result[0]["text"]


def test_repeated_headings_and_flat_numbering_do_not_lose_or_infer_content():
    source = "## 1. PROCEDURE\nDescription.\n\n## 1. OVERVIEW\nFirst overview.\n\n## 1. OVERVIEW\nSecond overview."
    result = make(source)
    for phrase in ["Description.", "First overview.", "Second overview."]:
        assert sum(c["text"].count(phrase) for c in result) == 1
    assert all(c["heading_path"] == ["1. OVERVIEW"] for c in result if "overview." in c["text"])


@pytest.mark.parametrize("source", ["<!-- page: 0 -->\nText", "<!-- page: 2 -->\nText\n<!-- page: 1 -->\nText"])
def test_invalid_page_order_is_rejected(source):
    with pytest.raises(ValueError, match="Page markers"):
        make(source)


def test_ids_metadata_and_original_source_lines_are_stable():
    source = "<!-- page: 3 -->\n# A\nAlpha.\n\n## B\nBeta."
    result = make(source)
    assert result == make(source)
    assert len({c["chunk_id"] for c in result}) == len(result)
    for index, item in enumerate(result):
        assert item["chunk_index"] == index
        assert item["source"] == "sample.md" and item["pages"] == [3]
        raw = "\n".join(source.splitlines()[item["line_start"] - 1:item["line_end"]])
        assert item["text"] == raw


def test_jsonl_checksum_and_failed_conversion_preserve_existing_output(tmp_path):
    source, output = tmp_path / "source.md", tmp_path / "chunks.jsonl"
    original = b"\xef\xbb\xbf<!-- page: 1 -->\r\n# Heading\r\nBody."
    source.write_bytes(original)
    path, records = chunk.chunk_file(source, output)
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == records
    assert records[0]["source_sha256"] == sha256(original).hexdigest()
    saved = output.read_bytes()
    with pytest.raises(ValueError, match="exists"):
        chunk.chunk_file(source, output)
    source.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No chunkable"):
        chunk.chunk_file(source, output, overwrite=True)
    assert output.read_bytes() == saved


def test_source_control_characters_are_preserved_without_splitter_mapping(tmp_path):
    source, output = tmp_path / "source.md", tmp_path / "chunks.jsonl"
    source.write_text("# Heading\n\x01\nContent.", encoding="utf-8")
    _, records = chunk.chunk_file(source, output)
    assert records[0]["text"] == source.read_bytes().decode("utf-8")


@pytest.mark.parametrize("control", ["\u202a", "\u200e", "\ufeff"])
def test_hidden_direction_markers_do_not_break_source_mapping(control):
    line = f"{control} Trai {control} G\u00e1i \u0110\u1ebb h\u1ed3i......Ng\u00e0y....../......./........."
    result = make("<!-- page: 2 -->\n# Form\n" + line + "\n")
    assert len(result) == 1
    assert line in result[0]["text"]
    assert result[0]["pages"] == [2]


def test_cli_batch_skips_existing_output(tmp_path, monkeypatch):
    source, output = tmp_path / "markdown", tmp_path / "chunks"
    source.mkdir()
    (source / "a.md").write_text("# A\nText", encoding="utf-8")
    assert chunk.main([str(source), "-o", str(output)]) == 0
    monkeypatch.setattr(chunk, "chunk_file", lambda *a, **k: pytest.fail("Must skip existing"))
    assert chunk.main([str(source), "-o", str(output)]) == 0


def test_large_paragraph_splits_only_between_complete_sentences():
    sentences = [f"Statement {i} says do not proceed unless the stated condition applies, with a value of 0.9." for i in range(15)]
    result = make("<!-- page: 2 -->\n# Context\n\n" + " ".join(sentences), max_tokens=80)
    assert len(result) > 1
    for sentence in sentences:
        assert sum(sentence in item["text"] for item in result) == 1
    assert all(item["token_count"] <= 80 and item["pages"] == [2] for item in result)
    assert all(item["text"].startswith("Context\n\n") for item in result)


def test_sentence_abbreviations_and_decimal_values_stay_together():
    sentence = "Dr. Alpha and TS. Beta report a value of 0.9 under the specified conditions."
    result = make("# Context\n" + " ".join([sentence] * 20), max_tokens=70)
    assert sum(item["text"].count(sentence) for item in result) == 20


def test_nested_list_fragments_keep_parent_children_and_introduction():
    introduction = "Only if condition X applies:"
    items = [f"- Parent instruction {i}.\n  - Keep child A{i} with the parent.\n  - Keep child B{i} with the parent." for i in range(10)]
    result = make("<!-- page: 5 -->\n# Conditions\n" + introduction + "\n\n" + "\n".join(items), max_tokens=90)
    assert len(result) > 1
    for item in items:
        assert sum(item in c["text"] for c in result) == 1
    assert all(introduction in c["text"] and c["pages"] == [5] for c in result)
    assert all(c["token_count"] <= 90 for c in result)


def test_table_splits_between_rows_with_original_header_and_context():
    header = "| Name | Value |\n| --- | --- |"
    rows = [f"| Entry {i} | value {i} is 2.10 and 3.80 |" for i in range(20)]
    introduction = "Use this table only when condition X applies:"
    result = make("<!-- page: 8 -->\n# Table\n" + introduction + "\n\n" + header + "\n" + "\n".join(rows), max_tokens=100)
    assert len(result) > 1
    for row in rows:
        assert sum(row in c["text"] for c in result) == 1
    assert all(header in c["text"] and introduction in c["text"] and c["pages"] == [8] for c in result)
    assert all(c["token_count"] <= 100 for c in result)


@pytest.mark.parametrize("body", ["Unbreakable " * 200 + ".", "| A | B |\n| --- | --- |\n| A | " + "value " * 200 + "|", "- Parent\n  - " + "condition " * 200])
def test_indivisible_sentence_row_or_nested_item_is_preserved_and_flagged(body):
    result = make("# Context\n\n" + body, max_tokens=64)
    assert any(body in c["text"] for c in result)
    assert any(c["needs_review"] and "oversized_preserved_for_review" in c["warnings"] for c in result)


def test_comments_removed_but_code_examples_are_preserved():
    result = make("<!-- page: 1 -->\n# Context\nText.\n<!-- image -->\n<!-- extraction: inspect source -->\n```text\n<!-- image -->\n```\n")
    assert len(result) == 1
    assert result[0]["text"].count("<!-- image -->") == 1
    assert "<!-- extraction:" not in result[0]["text"]
    assert "```text\n<!-- image -->\n```" in result[0]["text"]


def test_split_provenance_tracks_the_actual_source_pages():
    first = " ".join(f"First page sentence {i} contains the complete statement." for i in range(20))
    second = " ".join(f"Second page sentence {i} contains another statement." for i in range(20))
    result = make("<!-- page: 1 -->\n# Context\n" + first + "\n\n<!-- page: 2 -->\n" + second, max_tokens=80)
    for item in result:
        expected = sorted(({1} if "First page" in item["text"] else set()) | ({2} if "Second page" in item["text"] else set()))
        assert item["pages"] == expected


def test_large_heading_without_body_is_not_discarded():
    heading = "# " + "Long heading " * 100
    result = make(heading, max_tokens=64)
    assert len(result) == 1 and result[0]["text"] == heading
    assert result[0]["needs_review"]


def test_section_parts_have_stable_identity_and_one_canonical_copy():
    first = " ".join(f"Complete statement {i} has a condition that must remain attached." for i in range(20))
    second = "A different section with exactly the same title."
    source = "<!-- page: 1 -->\n## Repeated\n" + first + "\n\n## Repeated\n" + second
    rows = make(source, max_tokens=80)
    groups = {}
    for row in rows:
        groups.setdefault(row["section_id"], []).append(row)
        assert row["document_chunk_count"] == len(rows)
        assert row["section_path"] == ["Repeated"]
    assert len(groups) == 2
    parts, other = list(groups.values())
    assert len(parts) >= 3 and len(other) == 1
    assert [r["part_index"] for r in parts] == list(range(len(parts)))
    assert all(r["part_count"] == len(parts) for r in parts)
    assert sum("section_text" in r for r in parts) == 1
    assert first in parts[0]["section_text"] and second not in parts[0]["section_text"]
    assert rows == make(source, max_tokens=80)


def test_section_boundaries_do_not_treat_quoted_headings_as_document_sections():
    rows = make("# Root\nBody.\n\n> # Quoted heading\n> Quoted content.\n\n## Child\nNext body.")
    assert len(rows) == 2
    assert "Quoted content" in rows[0]["section_text"]
    assert rows[1]["section_path"] == ["Root", "Child"]
