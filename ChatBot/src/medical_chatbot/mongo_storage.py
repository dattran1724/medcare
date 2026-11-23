"""Atlas persistence. Model calls run outside short version-checked transactions."""
import json
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from datetime import datetime, timezone
from threading import RLock, local

from medical_chatbot.storage import StorageConflict, StorageError


class MongoStore:
    backend = "mongodb"
    persistent = True

    def __init__(self, uri, database):
        if not database or any(c in database for c in '/\\. "$*<>:|?'):
            raise StorageError("Invalid MONGODB_DATABASE")
        self._uri, self._name = uri, database
        self._client = None
        self._local = local()
        self._lock = RLock()

    def initialize(self):
        with self._lock:
            if self._client is not None:
                return
            try:
                from pymongo import MongoClient
                from pymongo.errors import PyMongoError
            except ImportError:
                raise StorageError('Install the mongodb extra: pip install -e ".[mongodb]"') from None
            client = None
            try:
                client = MongoClient(self._uri, serverSelectionTimeoutMS=10000,
                                     connectTimeoutMS=10000, socketTimeoutMS=20000,
                                     appname="MedCare-Agent1")
                client.admin.command("ping")
                database = client[self._name]
                for name in ("sessions", "turns", "conversations", "messages", "facts", "message_versions", "fact_versions"):
                    # Create collections/indexes outside application transactions.
                    database[name].create_index("conversation_id")
                database.messages.create_index([("conversation_id", 1), ("message_id", 1)], unique=True)
                database.messages.create_index([("conversation_id", 1), ("sequence", 1)], unique=True)
                database.facts.create_index([("conversation_id", 1), ("fact_id", 1)], unique=True)
                database.conversations.create_index([("patient_id", 1), ("updated_at", -1)])
            except (PyMongoError, ValueError):
                if client is not None:
                    client.close()
                raise StorageError("MongoDB unavailable; check URI, database user and Atlas IP access") from None
            self._client, self._db, self._errors = client, database, PyMongoError

    @staticmethod
    def _identity(namespace, key):
        return json.dumps([namespace, key], ensure_ascii=False, separators=(",", ":"))

    def _collection(self, namespace):
        return self._db.sessions if namespace == "sessions" else self._db.turns

    def _context(self):
        context = getattr(self._local, "context", None)
        if context is None:
            raise StorageError("Storage access requires a transaction")
        return context

    @contextmanager
    def transaction(self):
        self.initialize()
        if getattr(self._local, "context", None) is not None:
            raise StorageError("Nested storage transaction is unsupported")
        self._local.context = {"reads": {}, "writes": {}}
        try:
            yield
            if self._context()["writes"]:
                # The callback only performs DB operations, never invokes Gemini.
                with self._client.start_session() as session:
                    from pymongo.read_concern import ReadConcern
                    from pymongo.write_concern import WriteConcern
                    session.with_transaction(self._commit, max_commit_time_ms=10000,
                        read_concern=ReadConcern("snapshot"), write_concern=WriteConcern("majority"))
        except self._errors as error:
            if getattr(error, "code", None) == 11000:
                raise StorageConflict("Concurrent update; read the latest session and retry") from None
            raise StorageError("MongoDB operation failed; no successful commit was confirmed") from None
        finally:
            self._local.context = None

    def get(self, namespace, key, default=None):
        context = self._context()
        identity = self._identity(namespace, key)
        if identity in context["writes"]:
            return deepcopy(context["writes"][identity][2])
        if identity not in context["reads"]:
            record = self._collection(namespace).find_one({"_id": identity})
            try:
                value = self._transform(namespace, record["value"], unpack=True) if record else None
                context["reads"][identity] = (namespace, record["version"] if record else None, value)
            except (KeyError, TypeError):
                raise StorageError("Stored MongoDB record is incompatible or incomplete") from None
        version, value = context["reads"][identity][1:]
        return deepcopy(default if version is None else value)

    def put(self, namespace, key, value):
        identity = self._identity(namespace, key)
        if identity not in self._context()["reads"]:
            self.get(namespace, key)
        # Match SQLite's JSON contract; reject NaN and arbitrary Python objects.
        value = json.loads(json.dumps(value, allow_nan=False))
        self._context()["writes"][identity] = (namespace, key, value)

    def _commit(self, session):
        context = self._context()
        for identity, (namespace, version, _) in context["reads"].items():
            current = self._collection(namespace).find_one({"_id": identity}, {"version": 1}, session=session)
            if (current["version"] if current else None) != version:
                raise StorageConflict("Session changed while processing; read latest state before retrying")
        for identity, (namespace, key, value) in context["writes"].items():
            version = context["reads"][identity][1]
            encoded = self._transform(namespace, value, session=session)
            conversation_id = key if namespace == "sessions" else key[0]
            record = {"_id": identity, "namespace": namespace, "conversation_id": conversation_id,
                      "version": (version or 0) + 1, "value": encoded}
            if version is None:
                self._collection(namespace).insert_one(record, session=session)
            else:
                result = self._collection(namespace).replace_one({"_id": identity, "version": version}, record, session=session)
                if result.matched_count != 1:
                    raise StorageConflict("Concurrent session update")
            if namespace == "sessions":
                conversation = value["conversation"]
                now = datetime.now(timezone.utc)
                self._db.conversations.update_one({"_id": conversation_id}, {
                    "$set": {"conversation_id": conversation_id, "patient_id": conversation["patient_id"],
                        "version": record["version"], "updated_at": now,
                        "message_count": len(conversation["messages"])},
                    "$setOnInsert": {"created_at": now}}, upsert=True, session=session)
                previous = context["reads"][identity][2]
                self._project_history(value, previous, session)

    def _project_history(self, value, previous, session):
        """Queryable current history; immutable versions below preserve old retry snapshots."""
        cid = value["conversation"]["conversation_id"]
        for collection, items, old_items, id_field in (
            (self._db.messages, value["conversation"]["messages"],
             previous["conversation"]["messages"] if previous else [], "message_id"),
            (self._db.facts, value["facts"], previous["facts"] if previous else [], "fact_id"),
        ):
            old = {item[id_field]: (sequence, item) for sequence, item in enumerate(old_items)}
            current_ids = set()
            for sequence, item in enumerate(items):
                current_ids.add(item[id_field])
                if old.get(item[id_field]) == (sequence, item):
                    continue
                identity = self._identity("history", [cid, item[id_field]])
                collection.replace_one({"_id": identity}, {"_id": identity,
                    **item, "conversation_id": cid, "sequence": sequence}, upsert=True, session=session)
            removed = set(old) - current_ids
            if removed:
                collection.delete_many({"conversation_id": cid, id_field: {"$in": list(removed)}}, session=session)

    def history(self, conversation_id, offset, limit):
        self._context()
        if self._db.conversations.find_one({"_id": conversation_id}, {"_id": 1}) is None:
            return None
        return [{k: v for k, v in record.items() if k not in {"_id", "conversation_id", "sequence"}}
                for record in self._db.messages.find({"conversation_id": conversation_id})
                    .sort("sequence", 1).skip(offset).limit(limit)]

    def _transform(self, namespace, value, *, unpack=False, session=None):
        value = deepcopy(value)
        if namespace in {"sessions", "before_turns"}:
            target = value
        elif namespace == "replies":
            target = value["result"]["session"]
        elif namespace == "retry_replies":
            target = value["session"]
        else:
            return value
        cid = target["conversation"]["conversation_id"]
        for collection, container, field, id_field in (
            (self._db.message_versions, target["conversation"], "messages", "message_id"),
            (self._db.fact_versions, target, "facts", "fact_id"),
        ):
            if unpack:
                refs = container.pop(field + "_refs")
                records = {r["_id"]: r["value"] for r in collection.find({"_id": {"$in": refs}}, session=session)}
                if any(ref not in records for ref in refs):
                    raise StorageError("Stored session has missing message/fact references")
                container[field] = [records[ref] for ref in refs]
            else:
                refs = []
                for item in container.pop(field):
                    encoded = json.dumps([cid, item], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                    reference = sha256(encoded.encode("utf-8")).hexdigest()
                    collection.update_one({"_id": reference}, {"$setOnInsert": {
                        "conversation_id": cid, id_field: item[id_field], "value": item}}, upsert=True, session=session)
                    refs.append(reference)
                container[field + "_refs"] = refs
        return value

    def close(self):
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None
