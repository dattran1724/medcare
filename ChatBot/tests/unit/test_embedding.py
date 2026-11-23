"""Embedding orchestration without model downloads, GPU or external services."""
import json
import pytest

np = pytest.importorskip("numpy")
from rag import embed


class Model:
    max_seq_length = 8192

    def __init__(self):
        self.calls = []

    def tokenizer(self, texts, **options):
        assert options == dict(truncation=False, add_special_tokens=True, return_length=True)
        return {"length": [len(text) + 2 for text in texts]}

    def get_sentence_embedding_dimension(self):
        return 1024

    def encode(self, texts, **options):
        self.calls.append((texts, options))
        return np.ones((len(texts), 1024), dtype=np.float32)


def write_chunks(tmp_path, records=None):
    path = tmp_path / "chunks.jsonl"
    records = records if records is not None else [dict(chunk_id=f"c{i}", text=f"Statement {i}.",
        heading_path=["Section"], pages=[2], source_sha256="source-hash") for i in range(5)]
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def test_batches_preserve_metadata_and_save_normalized_finite_dense_vectors(tmp_path):
    source, model = write_chunks(tmp_path), Model()
    path, total = embed.embed_file(source, tmp_path / "embedded.jsonl", model=model, batch_size=2)
    result = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert total == 5 and [len(texts) for texts, _ in model.calls] == [2, 2, 1]
    for index, record in enumerate(result):
        assert record["chunk_id"] == f"c{index}" and record["text"] == f"Statement {index}."
        assert record["pages"] == [2] and record["source_sha256"] == "source-hash"
        assert record["embedding_text"] == f"Section\n\nStatement {index}."
        assert record["embedding_token_count"] == len(record["embedding_text"]) + 2
        assert record["embedding_model"] == embed.MODEL and record["embedding_revision"] == embed.REVISION
        assert record["embedding_dimensions"] == 1024 and record["embedding_normalized"]
        assert len(record["embedding"]) == 1024
        assert np.isclose(np.linalg.norm(record["embedding"]), 1)
    assert model.calls[0][1]["normalize_embeddings"] is True
    assert model.calls[0][1]["prompt"] == ""
    assert not list(tmp_path.glob(".embed-*.tmp"))


def test_heading_context_already_in_text_is_not_duplicated(tmp_path):
    source = write_chunks(tmp_path, [dict(chunk_id="c1", text="Section\n\nBody", heading_path=["Section"])])
    model = Model()
    embed.embed_file(source, tmp_path / "out.jsonl", model=model)
    assert model.calls[0][0] == ["Section\n\nBody"]


def test_over_limit_input_fails_before_encoding_and_preserves_previous_output(tmp_path):
    source = write_chunks(tmp_path, [dict(chunk_id="long", text="x" * 8191)])
    output = tmp_path / "out.jsonl"
    output.write_text("previous", encoding="utf-8")
    model = Model()
    with pytest.raises(ValueError, match="long.*8193.*8192"):
        embed.embed_file(source, output, model=model, overwrite=True)
    assert not model.calls and output.read_text() == "previous"


@pytest.mark.parametrize("records", [[], [dict(chunk_id="x", text="")], [dict(text="Missing id")],
    [dict(chunk_id="x", text="A"), dict(chunk_id="x", text="B")], [dict(chunk_id="x", text="A", heading_path="bad")]])
def test_invalid_chunks_do_not_publish_output(tmp_path, records):
    source, output = write_chunks(tmp_path, records), tmp_path / "out.jsonl"
    with pytest.raises(ValueError):
        embed.embed_file(source, output, model=Model())
    assert not output.exists()


@pytest.mark.parametrize("failure", ["shape", "nan", "zero", "provider"])
def test_failure_after_first_batch_leaves_existing_output_intact(tmp_path, failure):
    source, output = write_chunks(tmp_path), tmp_path / "out.jsonl"
    output.write_text("previous", encoding="utf-8")
    model, encode = Model(), Model.encode
    def fail_later(texts, **options):
        if model.calls:
            if failure == "provider":
                raise RuntimeError("Synthetic encode failure")
            if failure == "shape":
                return np.ones((len(texts), 10))
            return np.full((len(texts), 1024), np.nan if failure == "nan" else 0)
        return encode(model, texts, **options)
    model.encode = fail_later
    with pytest.raises((ValueError, RuntimeError)):
        embed.embed_file(source, output, model=model, batch_size=2, overwrite=True)
    assert output.read_text() == "previous" and not list(tmp_path.glob(".embed-*.tmp"))


def test_source_and_existing_output_are_protected(tmp_path):
    source, model = write_chunks(tmp_path), Model()
    with pytest.raises(ValueError, match="separate"):
        embed.embed_file(source, source, model=model, overwrite=True)
    output = tmp_path / "out.jsonl"
    output.write_text("previous", encoding="utf-8")
    with pytest.raises(ValueError, match="exists"):
        embed.embed_file(source, output, model=model)
    assert not model.calls


def test_batch_cli_reuses_model_and_skips_existing_outputs(tmp_path, monkeypatch):
    source, output = tmp_path / "chunks", tmp_path / "embeddings"
    source.mkdir()
    write_chunks(source).rename(source / "a.jsonl")
    write_chunks(source).rename(source / "b.jsonl")
    calls = []
    def load(device):
        calls.append(device)
        return Model()
    monkeypatch.setattr(embed, "load_model", load)
    assert embed.main([str(source), "-o", str(output), "--batch-size", "2"]) == 0
    assert calls == ["auto"]
    monkeypatch.setattr(embed, "load_model", lambda *_: pytest.fail("Existing outputs must skip model loading"))
    assert embed.main([str(source), "-o", str(output)]) == 0


def test_invalid_cli_path_does_not_download_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "load_model", lambda *_: pytest.fail("Do not load a model for a missing file"))
    with pytest.raises(SystemExit):
        embed.main([str(tmp_path / "missing.jsonl")])


def test_metadata_refresh_reuses_only_identical_text_and_model(tmp_path):
    source, output = write_chunks(tmp_path), tmp_path / "out.jsonl"
    embed.embed_file(source, output, model=Model())
    records = [json.loads(line) for line in source.read_text().splitlines()]
    for i, row in enumerate(records):
        row.update(chunk_id=f"new-{i}", section_id="section", part_index=i, part_count=len(records))
    records[-1]["text"] = "A genuinely changed input"
    write_chunks(tmp_path, records)
    model = Model()
    embed.embed_file(source, output, model=model, overwrite=True, reuse_existing=True)
    assert len(model.calls) == 1 and len(model.calls[0][0]) == 1
    refreshed = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["chunk_id"] for r in refreshed] == [f"new-{i}" for i in range(5)]
    assert all(r["section_id"] == "section" for r in refreshed)


def test_reuse_rejects_different_revision_or_invalid_cached_vector(tmp_path):
    source, output = write_chunks(tmp_path), tmp_path / "out.jsonl"
    embed.embed_file(source, output, model=Model())
    cached = [json.loads(line) for line in output.read_text().splitlines()]
    cached[0]["embedding_revision"] = "wrong"
    cached[1]["embedding"] = [0.] * 1024
    output.write_text("\n".join(json.dumps(row) for row in cached), encoding="utf-8")
    model = Model()
    embed.embed_file(source, output, model=model, overwrite=True, reuse_existing=True)
    assert sum(len(texts) for texts,_ in model.calls) == 2
