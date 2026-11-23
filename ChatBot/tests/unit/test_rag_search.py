"""Query embedding and Atlas pipeline checks without network or model downloads."""
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
from rag import search as module


class Model:
    max_seq_length = 8192

    def __init__(self):
        self.calls = []
        self.device = "cpu"
        self.tokenizer = SimpleNamespace(encode=lambda query, **options: list(range(len(query) + 2)))

    def encode(self, texts, **options):
        self.calls.append((texts, options))
        return np.ones((1, 1024), dtype=np.float32)


class Reranker:
    max_seq_length = 8192

    def __init__(self):
        self.tokenizer = lambda queries, texts, **opts: {"length": [len(q) + len(t) + 3 for q, t in zip(queries, texts)]}
        self.pairs = []

    def predict(self, pairs, **options):
        self.pairs = pairs
        return np.arange(len(pairs), dtype=float)


class Collection:
    def aggregate(self, pipeline, **options):
        self.pipeline, self.options = pipeline, options
        return iter([dict(chunk_id="c1", text="Source passage", source="guide.md", pages=[2], score=0.8)])


def test_search_embeds_query_and_returns_source_fields():
    model, collection = Model(), Collection()
    result = module.search("  Example question  ", collection, model=model, reranker=Reranker(), top_k=3)
    assert model.calls[0][0] == ["Example question"]
    assert model.calls[0][1]["prompt"] == "" and model.calls[0][1]["normalize_embeddings"]
    stage = collection.pipeline[0]["$vectorSearch"]
    assert stage["index"] == "rag_vector_index" and stage["path"] == "embedding"
    assert stage["limit"] == 20 and stage["numCandidates"] == 400
    assert len(stage["queryVector"]) == 1024 and np.isclose(np.linalg.norm(stage["queryVector"]), 1)
    assert collection.pipeline[1]["$match"] == dict(embedding_model=module.MODEL, embedding_revision=module.REVISION)
    assert collection.pipeline[2]["$project"]["score"] == {"$meta": "vectorSearchScore"}
    assert "embedding" not in collection.pipeline[2]["$project"]
    assert collection.options == dict(maxTimeMS=30000)
    assert result[0]["source"] == "guide.md" and result[0]["pages"] == [2]


@pytest.mark.parametrize("query,top_k", [("", 5), ("  ", 5), ("query", 0), ("query", 101), ("x" * 8191, 5)])
def test_invalid_or_oversized_query_does_not_encode_or_query_atlas(query, top_k):
    model = Model()
    with pytest.raises(ValueError):
        module.search(query, object(), model=model, reranker=Reranker(), top_k=top_k)
    assert not model.calls


@pytest.mark.parametrize("bad_vector", [np.zeros((1, 1024)), np.full((1, 1024), np.nan), np.ones((1, 10))])
def test_bad_vector_is_rejected_before_atlas(bad_vector):
    model = Model()
    model.encode = lambda *a, **kw: bad_vector
    with pytest.raises(ValueError, match="embedding"):
        module.search("question", object(), model=model, reranker=Reranker())


def test_empty_results_are_returned_without_inventing_an_answer():
    collection = SimpleNamespace(aggregate=lambda *a, **kw: iter([]))
    assert module.search("question", collection, model=Model(), reranker=Reranker()) == []


def test_cli_reuses_existing_database_and_closes_client(tmp_path, monkeypatch, capsys):
    import dotenv
    import pymongo
    visited, collection = [], Collection()
    class Client:
        def __enter__(self): return self
        def __exit__(self, *args): visited.append("closed")
        @property
        def admin(self): return self
        def command(self, command): visited.append(command)
        def __getitem__(self, name):
            visited.append(name)
            return collection if name == "rag_chunks" else self
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *_: None)
    monkeypatch.setenv("MONGODB_URI", "mongodb://example.invalid")
    monkeypatch.setenv("MONGODB_DATABASE", "medcare_chatbot")
    monkeypatch.setattr(pymongo, "MongoClient", lambda *a, **kw: Client())
    monkeypatch.setattr(module, "load_model", lambda *_: Model())
    monkeypatch.setattr(module, "load_reranker", lambda *_: Reranker())
    assert module.main(["question", "--top-k", "3"]) == 0
    assert visited == ["ping", "medcare_chatbot", "rag_chunks", "closed"]
    assert '"source": "guide.md"' in capsys.readouterr().out


def test_cli_database_errors_do_not_expose_credentials(monkeypatch, capsys):
    import dotenv
    import pymongo
    from pymongo.errors import ConnectionFailure
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *_: None)
    monkeypatch.setenv("MONGODB_URI", "mongodb://example.invalid")
    def fail(*a, **kw): raise ConnectionFailure("mongodb://user:private-password@example.invalid")
    monkeypatch.setattr(pymongo, "MongoClient", fail)
    assert module.main(["question"]) == 1
    assert "private-password" not in capsys.readouterr().err


def record(key, text, start, end, *, score=0.5, source="guide.md"):
    return dict(chunk_id=key, text=text, line_start=start, line_end=end, document_id="doc",
                source=source, source_sha256="checksum", heading_path=["Context"], pages=[1], score=score)


def test_reranking_promotes_content_instead_of_returning_heading_only():
    candidates = [record("heading", "## Context", 1, 1, score=.99),
                  record("background", "Background information.", 3, 4, score=.9),
                  record("answer", "The requested treatment information.", 6, 8, score=.7)]
    reranker = Reranker()
    results = module.rank_passages("treatment", candidates, reranker, top_k=2)
    assert [r["chunk_id"] for r in results] == ["answer", "background"]
    assert all("## Context" != pair[1].strip() for pair in reranker.pairs)
    assert results[0]["score"] == .7 and results[0]["rerank_score"] == 1.


def test_deduplication_does_not_merge_distinct_sections_or_sources():
    candidates = [record("a", "Identical wording.", 3, 5), record("a", "Identical wording.", 3, 5),
                  record("overlap", "Identical wording.", 4, 5), record("elsewhere", "Identical wording.", 9, 10),
                  record("other-doc", "Identical wording.", 3, 5, source="other.md")]
    result = module.rank_passages("question", candidates, Reranker(), top_k=5)
    assert {r["chunk_id"] for r in result} == {"a", "elsewhere", "other-doc"}


@pytest.mark.parametrize("headings,expected", [(((1,1,1),(3,3,2),(10,10,1)), ["child"]),
                                              (((1,1,2),(3,3,2),(10,10,2)), [])])
def test_heading_anchor_expands_only_confirmed_subtree(monkeypatch, headings, expected):
    anchor = record("anchor", "# Context", 1, 1)
    child = record("child", "Child content.", 3, 4)
    unrelated = record("unrelated", "Different chapter.", 10, 11)
    class Cursor(list):
        def sort(self, *args): return self
        def limit(self, size): return self[:size]
    class Database:
        def find(self, filters, projection):
            assert filters["document_id"] == "doc" and filters["source_sha256"] == "checksum"
            return Cursor([r.copy() for r in [child, unrelated] if r["line_start"] > filters["line_start"]["$gt"]
                           and r["line_end"] < filters.get("line_end", {}).get("$lt", float("inf"))])
    monkeypatch.setattr(module, "source_headings", lambda *args: headings)
    children = module.heading_children(anchor, Database(), 20)
    assert [r["chunk_id"] for r in children] == expected
    for r in children:
        assert r["via_heading"]["chunk_id"] == "anchor" and r["score"] is None


def test_source_outline_requires_matching_checksum_and_ignores_code_headings(tmp_path, monkeypatch):
    from hashlib import sha256
    folder = tmp_path / "rag" / "data" / "markdown"
    folder.mkdir(parents=True)
    data = b"# Root\n\n```text\n# Not a chapter\n```\n\n## Child\nBody.\n"
    (folder / "guide.md").write_bytes(data)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "rag" / "search.py"))
    module.source_headings.cache_clear()
    assert module.source_headings("guide.md", sha256(data).hexdigest()) == ((1,1,1),(7,7,2))
    assert module.source_headings("guide.md", "wrong-checksum") == ()
    assert module.source_headings("../guide.md", sha256(data).hexdigest()) == ()
    module.source_headings.cache_clear()


def test_no_guessed_child_lookup_without_source_identity():
    assert module.heading_children({"chunk_id":"title", "text":"# Title"}, object(), 20) == []


def test_reranker_does_not_silently_truncate_evidence_or_return_invalid_scores():
    reranker = Reranker()
    reranker.max_seq_length = 10
    with pytest.raises(ValueError, match="context"):
        module.rank_passages("question", [record("c", "Long source passage.", 1, 2)], reranker, 5)
    reranker.max_seq_length = 8192
    reranker.predict = lambda *a, **kw: [np.nan]
    with pytest.raises(ValueError, match="scores"):
        module.rank_passages("question", [record("c", "Source passage.", 1, 2)], reranker, 5)


def test_anchor_context_is_added_to_an_existing_candidate_without_duplication(monkeypatch):
    child = record("child", "Specific source content.", 3, 4)
    anchor = record("title", "# Context", 1, 1)
    class Database:
        def aggregate(self,*a,**kw): return [anchor,child]
    linked = dict(child, via_heading={"chunk_id":"title", "text":"# Context", "score":.9}, score=None)
    monkeypatch.setattr(module, "heading_children", lambda *a: [linked])
    reranker = Reranker()
    result = module.search("question", Database(), model=Model(), reranker=reranker)
    assert len(result) == 1 and result[0]["chunk_id"] == "child"
    assert result[0]["score"] == .5
    assert result[0]["via_heading"]["chunk_id"] == "title"
    assert "# Context" in reranker.pairs[0][1]


class SectionCollection:
    def __init__(self, rows): self.rows, self.filters = rows, []
    def find(self, filters, projection):
        self.filters.append(filters)
        class Cursor(list):
            def sort(self, key, direction): return sorted(self, key=lambda r:r[key])
        return Cursor([dict(r) for r in self.rows if all(r.get(k) == v for k,v in filters.items())])


def section_records():
    rows = [dict(record(f"part{i}", f"Context\n\nComplete source statement {i}.", i*3+1, i*3+3),
                 section_id="s", section_path=["Context"], part_index=i, part_count=3,
                 embedding_model=module.MODEL, embedding_revision=module.REVISION) for i in range(3)]
    rows[0].update(section_text="# Context\n\nComplete source statement 0.\nComplete source statement 1.\nComplete source statement 2.",
                   section_pages=[1], section_line_start=1,section_line_end=9)
    return rows


def test_expansion_returns_entire_section_in_order_without_repeated_context():
    rows = section_records()
    wrong_source = dict(rows[1], document_id="other", chunk_id="wrong")
    database = SectionCollection([rows[2],wrong_source,rows[0],rows[1]])
    result = module.expand_sections([rows[1],rows[2]], database, top_k=5, max_context_tokens=500)
    assert len(result) == 1 and result[0]["section_complete"]
    assert result[0]["text"] == rows[0]["section_text"]
    assert [p["part_index"] for p in result[0]["parts"]] == [0,1,2]
    assert result[0]["text"].count("Context") == 1
    assert result[0]["line_start"] == 1 and result[0]["line_end"] == 9
    assert database.filters[0]["source_sha256"] == "checksum"


def test_missing_parts_are_never_reported_as_complete():
    rows = section_records()
    result = module.expand_sections([rows[1]], SectionCollection(rows[1:]), 5, 500)
    assert result[0]["section_complete"] is False
    assert result[0]["text"] == rows[1]["text"]
    assert len(result[0]["parts"]) == 1


def test_context_budget_returns_whole_parts_and_marks_partial_sections():
    rows = section_records()
    rows[0]["section_text"] = "long section " * 300
    for row in rows:
        row["text"] += " Preserve this entire statement." * 8
    result = module.expand_sections([rows[1]], SectionCollection(rows), 5, 80)
    assert len(result) == 1 and not result[0]["section_complete"]
    assert rows[1]["text"] in result[0]["text"]
    assert result[0]["context_token_count"] <= 80
    assert result[0]["parts"][0]["part_index"] == 1


def test_shared_budget_and_distinct_section_identity():
    one = section_records()
    two = [dict(r,section_id="other",chunk_id="other"+r["chunk_id"]) for r in section_records()]
    database=SectionCollection(one+two)
    results=module.expand_sections([one[1],two[1]],database,5,64)
    assert sum(r["context_token_count"] for r in results) <= 64
    assert len({r["section_id"] for r in results}) == len(results)
