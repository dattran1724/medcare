"""Local BGE-M3 dense embeddings: chunk JSONL in, embeddings JSONL out."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

MODEL = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DIMENSIONS = 1024


def load_model(device="auto"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, revision=REVISION, device=None if device == "auto" else device)
    if model.device.type == "cuda":
        model.half()
    print(f"Embedding model: {MODEL}; device: {model.device}", flush=True)
    return model


def embed_file(source, output=None, *, model, batch_size=4, overwrite=False, reuse_existing=False):
    import numpy as np

    source = Path(source).resolve()
    output = (Path(output) if output else Path(__file__).parent / "data" / "embeddings" / source.name).resolve()
    if source.suffix.lower() != ".jsonl" or not source.is_file():
        raise ValueError("Source must be an existing chunk JSONL file")
    if output == source or output.suffix.lower() != ".jsonl":
        raise ValueError("Output must be a separate JSONL file")
    if output.exists() and not overwrite:
        raise ValueError("Output exists; use --overwrite")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not records or any(not isinstance(r, dict) or not isinstance(r.get("text"), str)
                          or not r["text"].strip() or not isinstance(r.get("chunk_id"), str)
                          or not r["chunk_id"].strip() for r in records):
        raise ValueError("Each chunk needs nonempty text and chunk_id")
    if len({r["chunk_id"] for r in records}) != len(records):
        raise ValueError("Duplicate chunk_id in source")
    texts = []
    for record in records:
        path = record.get("heading_path", [])
        if not isinstance(path, list) or any(not isinstance(h, str) for h in path):
            raise ValueError(f"Invalid heading_path: {record['chunk_id']}")
        context = "\n".join(path)
        body = record["text"]
        texts.append(context + "\n\n" + body if context and not body.startswith(context) else body)
    lengths = model.tokenizer(texts, truncation=False, add_special_tokens=True, return_length=True)["length"]
    for record, length in zip(records, lengths, strict=True):
        if length > model.max_seq_length:
            raise ValueError(f"Chunk {record['chunk_id']}: {length} tokens exceeds model limit {model.max_seq_length}; split it again")
    if model.get_sentence_embedding_dimension() != DIMENSIONS:
        raise ValueError("Expected BGE-M3 with 1024 dimensions")

    cache = {}
    if reuse_existing and output.is_file():
        with output.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                old = json.loads(line)
                if (old.get("embedding_model"), old.get("embedding_revision"), old.get("embedding_dimensions"),
                        old.get("embedding_normalized")) != (MODEL, REVISION, DIMENSIONS, True):
                    continue
                vector = np.asarray(old.get("embedding"), dtype=np.float32)
                if (vector.shape == (DIMENSIONS,) and np.isfinite(vector).all()
                        and np.isclose(np.linalg.norm(vector), 1, atol=1e-4)):
                    cache[(old.get("embedding_text"), old.get("embedding_token_count"))] = vector
    reused = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=output.parent,
                prefix=".embed-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for start in range(0, len(records), batch_size):
                batch = texts[start:start + batch_size]
                vectors = np.zeros((len(batch), DIMENSIONS), dtype=np.float32)
                missing = []
                for offset, text in enumerate(batch):
                    cached = cache.get((text, lengths[start + offset]))
                    if cached is None:
                        missing.append(offset)
                    else:
                        vectors[offset] = cached
                        reused += 1
                if missing:
                    encoded = np.asarray(model.encode([batch[i] for i in missing], batch_size=batch_size, prompt="",
                        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
                    if encoded.shape != (len(missing), DIMENSIONS):
                        raise ValueError("Invalid embedding shape")
                    vectors[missing] = encoded
                if vectors.shape != (len(batch), DIMENSIONS) or not np.isfinite(vectors).all():
                    raise ValueError("Invalid embedding shape or non-finite values")
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                if not np.isfinite(norms).all() or (norms == 0).any():
                    raise ValueError("Zero or invalid embedding norm")
                vectors = vectors / norms
                for offset, vector in enumerate(vectors):
                    index = start + offset
                    result = dict(records[index], embedding=vector.tolist(), embedding_text=texts[index],
                        embedding_model=MODEL, embedding_revision=REVISION, embedding_dimensions=DIMENSIONS,
                        embedding_normalized=True, embedding_token_count=lengths[index])
                    handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                if start % (batch_size * 25) == 0 or start + batch_size >= len(records):
                    print(f"  {source.name}: {min(start + batch_size, len(records))}/{len(records)}", flush=True)
        if overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    if reuse_existing:
        print(f"Reused {reused}/{len(records)} vectors with identical input/model; metadata refreshed", flush=True)
    return output, len(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Chunk JSONL to local BGE-M3 embeddings")
    parser.add_argument("source", nargs="?", type=Path, default=Path(__file__).parent / "data" / "chunks")
    parser.add_argument("-o", "--output", type=Path, help="JSONL file, or output folder for batch")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse vectors only for identical input/model")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    batch = args.source.is_dir()
    if not batch and (not args.source.is_file() or args.source.suffix.lower() != ".jsonl"):
        parser.error("Source must be a chunk JSONL file or folder")
    if batch and args.output and args.output.suffix.lower() == ".jsonl":
        parser.error("Batch output must be a folder")
    files = sorted(p for p in args.source.iterdir() if p.is_file() and p.suffix.lower() == ".jsonl") if batch else [args.source]
    if not files:
        parser.error("No chunk JSONL files found")
    model, failed = None, 0
    for source in files:
        output = ((args.output or Path(__file__).parent / "data" / "embeddings") / source.name) if batch else args.output
        if batch and output.exists() and not args.overwrite:
            print(f"Skipped: {output}")
            continue
        try:
            if model is None:
                model = load_model(args.device)
            path, total = embed_file(source, output, model=model, batch_size=args.batch_size,
                                     overwrite=args.overwrite, reuse_existing=args.reuse_existing)
            print(f"Saved: {path} ({total} embeddings)", flush=True)
        except (ValueError, OSError, ImportError, RuntimeError) as error:
            print(f"Error [{source.name}]: {error}", file=sys.stderr)
            failed += 1
            if model is None:
                break
    return int(failed > 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
