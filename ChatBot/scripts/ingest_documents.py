"""Ingest UTF-8 text/Markdown and explicit source metadata; no medical auto-review."""
import argparse
import hashlib
from pathlib import Path
from medical_chatbot.rag import DocumentChunk


def ingest(source, output, *, source_id, title, version, reviewed_by=None, reviewed_at=None,
           chunk_size=1800, overlap=200):
    source, output = Path(source), Path(output)
    if source.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("Only UTF-8 .txt/.md input is supported; convert PDF separately")
    if source.resolve() == output.resolve():
        raise ValueError("Output must not replace the source document")
    if not 100 <= chunk_size <= 10000 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap")
    text = source.read_text(encoding="utf-8-sig")
    chunks = []
    for start in range(0, len(text), chunk_size - overlap):
        content = text[start:start + chunk_size].strip()
        if not content:
            continue
        digest = hashlib.sha256(f"{source_id}:{version}:{start}:{content}".encode("utf-8")).hexdigest()[:20]
        chunks.append(DocumentChunk(chunk_id=digest, source_id=source_id, title=title, version=version,
            content=content, reviewed_by=reviewed_by, reviewed_at=reviewed_at,
            source_url=source.resolve().as_uri()))
        if start + chunk_size >= len(text):
            break
    if not chunks:
        raise ValueError("Source document is empty")
    previous = [DocumentChunk.model_validate_json(line) for line in output.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()] if output.exists() else []
    # One active version per source; old response snapshots retain historical citations.
    previous = [c for c in previous if c.source_id != source_id]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text("\n".join(c.model_dump_json() for c in previous + chunks) + "\n", encoding="utf-8")
    temporary.replace(output)
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--output", default="data/rag/corpus.jsonl")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--reviewed-by")
    parser.add_argument("--reviewed-at", help="ISO timestamp including timezone; owner-controlled review metadata")
    args = vars(parser.parse_args())
    print("Chunks written:", ingest(**args))
