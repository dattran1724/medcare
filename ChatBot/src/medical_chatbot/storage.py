"""Small transactional JSON store for the local API; not a medical-record backend."""
import json
import sqlite3
from contextlib import contextmanager
from threading import RLock


class LocalStore:
    def __init__(self, path: str | None = None):
        self.persistent = bool(path and path != ":memory:")
        self._connection = sqlite3.connect(path or ":memory:", timeout=5,
                                            isolation_level=None, check_same_thread=False)
        self._connection.execute("""CREATE TABLE IF NOT EXISTS agent_data (
            namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
            PRIMARY KEY(namespace, key))""")
        self._lock = RLock()

    @contextmanager
    def transaction(self):
        # BEGIN IMMEDIATE serializes competing updates even across API instances.
        # Local testing intentionally keeps the transaction across provider calls.
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def get(self, namespace: str, key, default=None):
        row = self._connection.execute("SELECT value FROM agent_data WHERE namespace=? AND key=?",
            (namespace, json.dumps(key))).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            raise sqlite3.DatabaseError("Invalid stored JSON") from None

    def put(self, namespace: str, key, value):
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        self._connection.execute("""INSERT INTO agent_data(namespace,key,value) VALUES(?,?,?)
            ON CONFLICT(namespace,key) DO UPDATE SET value=excluded.value""",
            (namespace, json.dumps(key), encoded))

    def close(self):
        with self._lock:
            self._connection.close()
