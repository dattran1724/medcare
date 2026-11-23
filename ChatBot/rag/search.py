"""Search uploaded chunks with the same BGE-M3 model used for embedding."""
import argparse
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from rag.embed import DIMENSIONS, MODEL, REVISION, load_model

INDEX = "rag_vector_index"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
FIELDS = {"_id": 0, **{name: 1 for name in (
    "chunk_id", "chunk_index", "document_id", "source_sha256", "text", "heading_path",
    "source", "pages", "page_start", "page_end", "line_start", "line_end", "section_id", "section_path",
    "section_line_start", "section_line_end", "part_index", "part_count")}}


def load_reranker(device):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, revision=RERANK_REVISION, device=str(device),
        processor_kwargs={"model_max_length": 8192},
        model_kwargs={"torch_dtype": "float16" if str(device).startswith("cuda") else "float32"})


def heading_only(record):
    from markdown_it import MarkdownIt
    text = record["text"].strip()
    if text == "\n".join(record.get("heading_path", [])).strip():
        return True
    blocks = [t for t in MarkdownIt().parse(text) if t.level == 0 and t.nesting != -1 and t.map]
    return bool(blocks) and all(t.type == "heading_open" or t.type == "hr" for t in blocks)


@lru_cache(maxsize=32)
def source_headings(source, checksum):
    from markdown_it import MarkdownIt
    root = Path(__file__).parent / "data" / "markdown"
    path = root / source
    if Path(source).name != source or not path.is_file():
        return ()
    data = path.read_bytes()
    if sha256(data).hexdigest() != checksum:
        return ()
    return tuple((t.map[0] + 1, t.map[1], int(t.tag[1:]))
                 for t in MarkdownIt().parse(data.decode("utf-8-sig")) if t.type == "heading_open" and t.level == 0 and t.map)


def heading_children(anchor, collection, limit):
    """Expand only an actual Markdown subtree in the matching source version."""
    if not all(anchor.get(k) for k in ("source", "source_sha256", "document_id", "line_start", "line_end")):
        return []
    headings = source_headings(anchor["source"], anchor["source_sha256"])
    matches = [h for h in headings if anchor["line_start"] <= h[0] <= anchor["line_end"]]
    if not matches:
        return []
    start, end, depth = matches[-1]
    stop = next((line for line, _, level in headings if line > start and level <= depth), None)
    bounds = {"$gt": end}
    filters = dict(document_id=anchor["document_id"], source_sha256=anchor["source_sha256"],
                   embedding_model=MODEL, embedding_revision=REVISION, line_start=bounds)
    if stop is not None:
        filters["line_end"] = {"$lt": stop}
    children = list(collection.find(filters, FIELDS).sort("chunk_index", 1).limit(limit))
    for child in children:
        child["score"] = None  # It has no independent vector-search score.
        child["via_heading"] = {"chunk_id": anchor["chunk_id"], "text": anchor["text"], "score": anchor.get("score")}
    return children


def rank_passages(query, candidates, reranker, top_k):
    import numpy as np
    passages, seen = [], set()
    for record in candidates:
        if record["chunk_id"] in seen or heading_only(record):
            continue
        # Remove exact duplicates only when source ranges actually overlap.
        duplicate = any(record.get("source") == other.get("source")
                        and record.get("source_sha256") == other.get("source_sha256")
                        and record.get("heading_path") == other.get("heading_path")
                        and record.get("line_start", 0) <= other.get("line_end", -1)
                        and other.get("line_start", 0) <= record.get("line_end", -1)
                        and " ".join(record["text"].split()) == " ".join(other["text"].split())
                        for other in passages)
        if not duplicate:
            passages.append(record)
            seen.add(record["chunk_id"])
    if not passages:
        return []
    texts = [r.get("via_heading", {}).get("text", "") + "\n" +
             "\n".join(r.get("heading_path", [])) + "\n\n" + r["text"] for r in passages]
    lengths = reranker.tokenizer([query] * len(texts), texts, truncation=False, return_length=True)["length"]
    if any(length > reranker.max_seq_length for length in lengths):
        raise ValueError("Candidate exceeds reranker context; no evidence is silently truncated")
    scores = np.asarray(reranker.predict([[query, text] for text in texts], batch_size=1,
        show_progress_bar=False, convert_to_numpy=True), dtype=float).reshape(-1)
    if len(scores) != len(passages) or not np.isfinite(scores).all():
        raise ValueError("Invalid reranker scores")
    results = [dict(record, rerank_score=float(score), rerank_model=RERANK_MODEL)
               for record, score in zip(passages, scores, strict=True)]
    return sorted(results, key=lambda r: r["rerank_score"], reverse=True)[:top_k]


def expand_sections(ranked, collection, top_k, max_context_tokens):
    """Return canonical sections, or a bounded contiguous window around the hit."""
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")
    count = lambda text: len(encoder.encode(text, disallowed_special=()))
    results, seen, remaining = [], set(), max_context_tokens
    for hit in ranked:
        identity = tuple(hit.get(k) for k in ("document_id", "source_sha256", "section_id"))
        if identity in seen:
            continue
        section_id = hit.get("section_id")
        parts, complete = [hit], False
        if section_id and identity[0] and identity[1]:
            filters = dict(zip(("document_id", "source_sha256", "section_id"), identity))
            filters.update(embedding_model=MODEL, embedding_revision=REVISION)
            projection = dict(FIELDS, section_text=1, section_pages=1)
            fetched = list(collection.find(filters, projection).sort("part_index", 1))
            expected = hit.get("part_count", 0)
            complete = (expected > 0 and len(fetched) == expected
                        and [p.get("part_index") for p in fetched] == list(range(expected))
                        and all(p.get("part_count") == expected for p in fetched)
                        and isinstance(fetched[0].get("section_text"), str))
            if complete:
                parts = fetched
        full_text = parts[0].get("section_text") if complete else None
        if full_text and count(full_text) <= remaining:
            selected, text, pages = parts, full_text, parts[0].get("section_pages", [])
        else:
            # Never slice a sentence/table row to meet the context budget.
            middle = next((i for i,p in enumerate(parts) if p["chunk_id"] == hit["chunk_id"]), None)
            if middle is None:
                continue
            left, right = middle, middle + 1
            def window(a, b): return "\n\n".join(p["text"] for p in parts[a:b])
            if count(window(left, right)) > remaining:
                continue
            while left > 0 and count(window(left - 1, right)) <= remaining:
                left -= 1
            while right < len(parts) and count(window(left, right + 1)) <= remaining:
                right += 1
            selected, text = parts[left:right], window(left, right)
            pages = sorted({p for part in selected for p in part.get("pages", [])})
            complete = False  # Canonical complete section did not fit / was unavailable.
        result = dict(hit, text=text, pages=pages, page_start=pages[0] if pages else None,
                      page_end=pages[-1] if pages else None, section_complete=complete,
                      parts=[{k:p.get(k) for k in ("chunk_id", "part_index", "pages", "line_start", "line_end")} for p in selected],
                      line_start=min((p["line_start"] for p in selected if "line_start" in p), default=None),
                      line_end=max((p["line_end"] for p in selected if "line_end" in p), default=None))
        result["context_token_count"] = count(text)
        if complete:
            result["line_start"] = parts[0].get("section_line_start", result["line_start"])
            result["line_end"] = parts[0].get("section_line_end", result["line_end"])
        results.append(result)
        remaining -= result["context_token_count"]
        seen.add(identity if section_id else (hit["chunk_id"],))
        if len(results) == top_k:
            break
    return results


def search(query, collection, *, model, reranker, top_k=5, max_context_tokens=4000):
    import numpy as np

    query = query.strip()
    if not query or not 1 <= top_k <= 100 or max_context_tokens < 64:
        raise ValueError("Provide a nonempty query and top_k between 1 and 100")
    length = len(model.tokenizer.encode(query, truncation=False, add_special_tokens=True))
    if length > model.max_seq_length:
        raise ValueError(f"Query exceeds model limit: {length}/{model.max_seq_length} tokens")
    vector = np.asarray(model.encode([query], batch_size=1, prompt="", normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
    if vector.shape != (1, DIMENSIONS) or not np.isfinite(vector).all():
        raise ValueError("Invalid query embedding")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid query embedding norm")
    candidate_count = max(20, top_k * 4)
    pipeline = [
        {"$vectorSearch": {"index": INDEX, "path": "embedding", "queryVector": (vector[0] / norm).tolist(),
                           "numCandidates": 20 * candidate_count, "limit": candidate_count}},
        {"$match": {"embedding_model": MODEL, "embedding_revision": REVISION}},
        {"$project": dict(FIELDS, score={"$meta": "vectorSearchScore"})},
    ]
    candidates = list(collection.aggregate(pipeline, maxTimeMS=30000))
    by_id = {r["chunk_id"]: r for r in candidates}
    for anchor in list(candidates):
        if heading_only(anchor):
            for child in heading_children(anchor, collection, candidate_count):
                if child["chunk_id"] in by_id:
                    by_id[child["chunk_id"]].setdefault("via_heading", child["via_heading"])
                else:
                    candidates.append(child)
                    by_id[child["chunk_id"]] = child
    ranked = rank_passages(query, candidates, reranker, len(candidates))
    return expand_sections(ranked, collection, top_k, max_context_tokens)



def main(argv=None):
    from dotenv import load_dotenv
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError

    parser = argparse.ArgumentParser(description="Search MongoDB RAG chunks with BGE-M3")
    parser.add_argument("query", help="Vietnamese or multilingual search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-tokens", type=int, default=4000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if not args.query.strip() or not 1 <= args.top_k <= 100 or args.max_context_tokens < 64:
        parser.error("Provide a nonempty query and --top-k between 1 and 100")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    uri, database = os.getenv("MONGODB_URI"), os.getenv("MONGODB_DATABASE", "medcare_chatbot")
    if not uri or not database:
        parser.error("Set MONGODB_URI and MONGODB_DATABASE in .env")
    try:
        with MongoClient(uri, serverSelectionTimeoutMS=10000, connectTimeoutMS=10000,
                         socketTimeoutMS=40000, appname="MedCare-RAG-Search") as client:
            client.admin.command("ping")
            model = load_model(args.device)
            reranker = load_reranker(model.device)
            results = search(args.query, client[database]["rag_chunks"], model=model, reranker=reranker, top_k=args.top_k,
                             max_context_tokens=args.max_context_tokens)
            print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
    except PyMongoError as error:
        print(f"MongoDB search failed ({type(error).__name__}); check connection and vector index readiness.", file=sys.stderr)
        return 1
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"Search failed ({type(error).__name__}); check model setup and query/candidate length.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
