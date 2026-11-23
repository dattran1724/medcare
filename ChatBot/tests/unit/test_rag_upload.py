"""Upload batching/idempotency checks without connecting to Atlas."""
import json

import pytest

pytest.importorskip("pymongo")
from rag import upload


class Collection:
    def __init__(self):
        self.documents, self.batches = {}, []
        self.fail_after = None
        self.deletions = []

    def create_index(self, fields):
        assert fields == [(key, 1) for key in ("document_id", "source_sha256", "section_id", "part_index")]

    def bulk_write(self, operations, ordered):
        assert ordered is True
        if self.fail_after is not None and len(self.batches) >= self.fail_after:
            from pymongo.errors import AutoReconnect
            raise AutoReconnect("Synthetic network failure")
        self.batches.append(len(operations))
        for operation in operations:
            assert operation._upsert is True
            assert operation._filter == {"_id": operation._doc["chunk_id"]}
            self.documents[operation._filter["_id"]] = dict(operation._doc)

    def delete_many(self, filters):
        from types import SimpleNamespace
        self.deletions.append(filters)
        obsolete = [key for key,r in self.documents.items() if r.get("document_id") == filters["document_id"]
                    and key not in filters["_id"]["$nin"]]
        for key in obsolete:
            del self.documents[key]
        return SimpleNamespace(deleted_count=len(obsolete))


def write_source(tmp_path, records=None):
    records = records if records is not None else [dict(chunk_id=f"c{i}", text=f"Text {i}", pages=[2],
        embedding=[1.0] + [0.0] * 1023, embedding_model=upload.MODEL, embedding_revision=upload.REVISION) for i in range(5)]
    path = tmp_path / "embeddings.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def test_batched_upload_preserves_records_and_reruns_without_duplicates(tmp_path):
    source, collection = write_source(tmp_path), Collection()
    assert upload.upload_file(source, collection, batch_size=2) == 5
    assert collection.batches == [2, 2, 1]
    original = [json.loads(line) for line in source.read_text().splitlines()]
    assert collection.documents == {r["chunk_id"]: dict(r, _id=r["chunk_id"]) for r in original}
    original[0]["text"] = "Updated text"
    write_source(tmp_path, original)
    upload.upload_file(source, collection, batch_size=3)
    assert len(collection.documents) == 5 and collection.documents["c0"]["text"] == "Updated text"


@pytest.mark.parametrize("failure", ["missing_id", "duplicate", "empty_text", "dimensions", "nan", "infinity", "zero", "boolean", "model", "revision"])
def test_bad_last_record_is_rejected_before_any_database_write(tmp_path, failure):
    source = write_source(tmp_path)
    records = [json.loads(line) for line in source.read_text().splitlines()]
    record = records[-1]
    if failure == "missing_id": record.pop("chunk_id")
    elif failure == "duplicate": record["chunk_id"] = "c0"
    elif failure == "empty_text": record["text"] = ""
    elif failure == "dimensions": record["embedding"] = [1.0]
    elif failure == "nan": record["embedding"][0] = float("nan")
    elif failure == "infinity": record["embedding"][0] = float("inf")
    elif failure == "zero": record["embedding"] = [0.0] * 1024
    elif failure == "boolean": record["embedding"][0] = True
    elif failure == "model": record["embedding_model"] = "Other"
    elif failure == "revision": record["embedding_revision"] = "Other"
    source, collection = write_source(tmp_path, records), Collection()
    with pytest.raises(ValueError):
        upload.upload_file(source, collection, batch_size=2)
    assert not collection.batches


def test_network_failure_can_be_retried_without_duplicate_documents(tmp_path):
    from pymongo.errors import AutoReconnect
    source, collection = write_source(tmp_path), Collection()
    collection.fail_after = 1
    with pytest.raises(AutoReconnect):
        upload.upload_file(source, collection, batch_size=2)
    assert len(collection.documents) == 2
    collection.fail_after = None
    upload.upload_file(source, collection, batch_size=2)
    assert len(collection.documents) == 5


def test_empty_input_and_invalid_batch_size_do_not_write(tmp_path):
    collection = Collection()
    with pytest.raises(ValueError, match="Empty"):
        upload.upload_file(write_source(tmp_path, []), collection)
    with pytest.raises(ValueError, match="positive"):
        upload.upload_file(write_source(tmp_path), collection, batch_size=0)
    assert not collection.batches


def test_cli_targets_only_rag_collection_and_closes_client(tmp_path, monkeypatch):
    import dotenv
    import pymongo
    source, collection = write_source(tmp_path), Collection()
    visited = []
    class Client:
        def __enter__(self): return self
        def __exit__(self, *args): visited.append("closed")
        @property
        def admin(self): return self
        def command(self, name): visited.append(name)
        def __getitem__(self, name):
            visited.append(name)
            return collection if name == "rag_chunks" else self
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *_: None)
    monkeypatch.setenv("MONGODB_URI", "mongodb://example.invalid")
    monkeypatch.setenv("MONGODB_DATABASE", "medcare_chatbot")
    monkeypatch.setattr(pymongo, "MongoClient", lambda *a, **kw: Client())
    collection.count_documents = lambda query: len(collection.documents)
    assert upload.main([str(source), "--batch-size", "2"]) == 0
    assert visited == ["ping", "medcare_chatbot", "rag_chunks", "closed"]
    assert len(collection.documents) == 5


def test_cli_does_not_print_credentials_from_database_errors(tmp_path, monkeypatch, capsys):
    import dotenv
    import pymongo
    from pymongo.errors import ConnectionFailure
    source = write_source(tmp_path)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *_: None)
    monkeypatch.setenv("MONGODB_URI", "mongodb://example.invalid")
    def fail(*args, **kwargs):
        raise ConnectionFailure("mongodb://user:private-password@example.invalid")
    monkeypatch.setattr(pymongo, "MongoClient", fail)
    assert upload.main([str(source)]) == 1
    assert "private-password" not in capsys.readouterr().err


def complete_manifest(tmp_path):
    source = write_source(tmp_path)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for i,row in enumerate(rows):
        row.update(document_id="doc", source_sha256="version", document_chunk_count=len(rows), chunk_index=i,
                   section_id="section", part_index=i, part_count=len(rows))
    rows[0]["section_text"] = "Complete original section"
    return rows


def test_sync_removes_only_obsolete_ids_after_complete_upload(tmp_path):
    rows, collection = complete_manifest(tmp_path), Collection()
    collection.documents.update(old=dict(document_id="doc"), unrelated=dict(document_id="other"))
    upload.upload_file(write_source(tmp_path,rows), collection, sync=True, batch_size=2)
    assert "old" not in collection.documents and "unrelated" in collection.documents
    assert len(collection.documents) == 6 and len(collection.deletions) == 1


@pytest.mark.parametrize("corruption", ["truncated", "missing_section_part", "mixed_document", "mixed_version", "index"])
def test_sync_rejects_partial_manifest_before_writing_or_deleting(tmp_path, corruption):
    rows = complete_manifest(tmp_path)
    if corruption == "truncated": rows.pop()
    elif corruption == "missing_section_part": rows[-1]["part_count"] += 1
    elif corruption == "mixed_document": rows[-1]["document_id"] = "other"
    elif corruption == "mixed_version": rows[-1]["source_sha256"] = "other"
    else: rows[-1]["chunk_index"] = 999
    collection = Collection()
    with pytest.raises(ValueError):
        upload.upload_file(write_source(tmp_path,rows), collection, sync=True)
    assert not collection.batches and not collection.deletions


def test_sync_never_deletes_old_version_after_a_batch_failure(tmp_path):
    from pymongo.errors import AutoReconnect
    rows, collection = complete_manifest(tmp_path), Collection()
    collection.documents["old"] = dict(document_id="doc")
    collection.fail_after = 1
    with pytest.raises(AutoReconnect):
        upload.upload_file(write_source(tmp_path,rows), collection, sync=True, batch_size=2)
    assert "old" in collection.documents and not collection.deletions
