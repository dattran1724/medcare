"""Upsert local BGE-M3 embedding JSONL into MongoDB rag_chunks."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

from rag.embed import DIMENSIONS, MODEL, REVISION

COLLECTION = "rag_chunks"


def upload_file(source, collection, *, batch_size=100, sync=False):
    from pymongo import ReplaceOne

    source = Path(source)
    if source.suffix.lower() != ".jsonl" or not source.is_file():
        raise ValueError("Source must be an embedding JSONL file")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    with source.open(encoding="utf-8-sig") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError("Empty embedding file")
    seen = set()
    # Validate the complete file before its first database write.
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict) or not isinstance(record.get("chunk_id"), str) or not record["chunk_id"].strip():
            raise ValueError(f"Line {index}: missing chunk_id")
        if record["chunk_id"] in seen:
            raise ValueError(f"Duplicate chunk_id: {record['chunk_id']}")
        seen.add(record["chunk_id"])
        vector = record.get("embedding")
        if (not isinstance(vector, list) or len(vector) != DIMENSIONS
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)
                or not any(vector)):
            raise ValueError(f"Chunk {record['chunk_id']}: invalid 1024-dimensional embedding")
        if (record.get("embedding_model"), record.get("embedding_revision")) != (MODEL, REVISION):
            raise ValueError(f"Chunk {record['chunk_id']}: unexpected embedding model/revision")
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            raise ValueError(f"Chunk {record['chunk_id']}: missing text")
    if sync:
        # A truncated/partial file must never authorize removal of an older version.
        document = records[0].get("document_id")
        version = records[0].get("source_sha256")
        if (not document or not version or any(r.get("document_id") != document or r.get("source_sha256") != version
                or r.get("document_chunk_count") != len(records) or r.get("chunk_index") != i
                for i, r in enumerate(records))):
            raise ValueError("Sync requires a complete single-document chunk manifest; rerun chunk and embed")
        sections = {}
        for record in records:
            if not record.get("section_id"):
                raise ValueError("Sync requires section metadata; rerun chunk and embed")
            sections.setdefault(record["section_id"], []).append(record)
        for parts in sections.values():
            if ([p.get("part_index") for p in parts] != list(range(len(parts)))
                    or any(p.get("part_count") != len(parts) for p in parts)
                    or not isinstance(parts[0].get("section_text"), str) or not parts[0]["section_text"].strip()):
                raise ValueError("Sync requires every part of each section")
    for start in range(0, len(records), batch_size):
        operations = [ReplaceOne({"_id": r["chunk_id"]}, dict(r, _id=r["chunk_id"]), upsert=True)
                      for r in records[start:start + batch_size]]
        collection.bulk_write(operations, ordered=True)
    if sync:
        deleted = collection.delete_many({"document_id": document, "_id": {"$nin": list(seen)}})
        print(f"Removed {deleted.deleted_count} obsolete chunks for {document}", flush=True)
    return len(records)


def main(argv=None):
    from dotenv import load_dotenv
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError

    parser = argparse.ArgumentParser(description="Embedding JSONL to MongoDB rag_chunks")
    parser.add_argument("source", nargs="?", type=Path, default=Path(__file__).parent / "data" / "embeddings")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sync", action="store_true", help="Remove obsolete IDs for each fully uploaded document")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    files = sorted(p for p in args.source.glob("*.jsonl") if p.is_file()) if args.source.is_dir() else [args.source]
    if not files or any(not p.is_file() or p.suffix.lower() != ".jsonl" for p in files):
        parser.error("No embedding JSONL files found")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    uri, database = os.getenv("MONGODB_URI"), os.getenv("MONGODB_DATABASE", "medcare_chatbot")
    if not uri or not database:
        parser.error("Set MONGODB_URI and MONGODB_DATABASE in .env")
    total = 0
    try:
        with MongoClient(uri, serverSelectionTimeoutMS=10000, connectTimeoutMS=10000,
                         socketTimeoutMS=60000, appname="MedCare-RAG-Upload") as client:
            client.admin.command("ping")
            collection = client[database][COLLECTION]
            collection.create_index([(key, 1) for key in ("document_id", "source_sha256", "section_id", "part_index")])
            for source in files:
                count = upload_file(source, collection, batch_size=args.batch_size, sync=args.sync)
                total += count
                print(f"Uploaded: {source.name} ({count} chunks)", flush=True)
            print(f"Finished: {total} chunks processed; collection contains {collection.count_documents({})} documents")
    except (ValueError, OSError) as error:
        print(f"Error: {error}. Earlier batches may already be saved; rerun after fixing the input.", file=sys.stderr)
        return 1
    except PyMongoError as error:
        print(f"MongoDB upload failed ({type(error).__name__}); check connection/access and retry. Earlier batches may be saved.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
