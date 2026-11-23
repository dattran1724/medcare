"""Opt-in Atlas tests. Fake clinical providers; real MongoDB transactions and indexes.

Run with MEDCARE_TEST_MONGODB=1. Only unique synthetic conversation IDs are used.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from medical_chatbot.api import create_app
from medical_chatbot.agents.intake.graph import build_intake_graph
from medical_chatbot.graph.state import IntakeSession
from medical_chatbot.schemas.conversation import Conversation
from medical_chatbot.storage import StorageConflict, StorageError
from medical_chatbot.mongo_storage import MongoStore


pytestmark = pytest.mark.skipif(os.getenv("MEDCARE_TEST_MONGODB") != "1", reason="Atlas tests are opt-in")


@pytest.fixture
def atlas():
    pytest.importorskip("pymongo")
    from dotenv import dotenv_values
    from pymongo import MongoClient
    config = dotenv_values(".env")
    uri = os.getenv("MONGODB_URI") or config.get("MONGODB_URI")
    database = os.getenv("MONGODB_DATABASE") or config.get("MONGODB_DATABASE") or "medcare_chatbot"
    if not uri:
        pytest.fail("MONGODB_URI is required for opt-in Atlas tests")
    ids, stores = set(), []

    def factory():
        store = MongoStore(uri, database)
        store.initialize()
        stores.append(store)
        return store

    def new_id():
        cid = "integration_" + uuid4().hex
        ids.add(cid)
        return cid

    yield factory, new_id, ids
    # Remove only synthetic documents registered by this fixture, never collections/databases.
    for store in stores:
        store.close()
    if ids:
        with MongoClient(uri, serverSelectionTimeoutMS=10000) as client:
            for name in ("sessions", "turns", "conversations", "messages", "facts", "message_versions", "fact_versions"):
                client[database][name].delete_many({"conversation_id": {"$in": list(ids)}})


def payload(cid):
    return IntakeSession(conversation=Conversation(conversation_id=cid, patient_id="synthetic"))


def providers(*, bad_question=False, calls=None):
    def extract(data):
        if calls is not None:
            calls.append("extract")
        cases = data["clinical_state"]["cases"]
        return json.dumps({"case_result": {"resolution": "OLD_CASE", "target_case_id": cases[0]["case_id"]}
                          if cases else {"resolution": "NEW_CASE", "new_problem": "headache"}})

    def screen(data):
        msg = data["new_message"]
        return json.dumps(dict(result="SAFE", reason="Synthetic screen", evidence=[
            dict(message_id=msg["message_id"], source_text=msg["content"])]))

    def question(data):
        return "invalid" if bad_question else json.dumps(dict(question="Synthetic question?",
            scope_ids=[s["id"] for s in data["approved_scopes"]]))

    return build_intake_graph(extract_json=extract, risk_json=screen, question_json=question)


def test_atlas_api_restart_history_and_idempotency(atlas):
    factory, _, ids = atlas
    calls = []
    first_store = factory()
    with TestClient(create_app(providers(calls=calls), storage=first_store)) as client:
        assert client.get("/health").json()["storage"] == "mongodb"
        cid = client.post("/conversations", json={"patient_id": "synthetic"}).json()["conversation"]["conversation_id"]
        ids.add(cid)
        url = f"/conversations/{cid}"
        body = dict(message_id="m1", content="headache")
        first = client.post(url + "/messages", json=body)
        assert first.status_code == 200 and first.json()["output"]["next_action"] == "ASK_MORE"
        assert client.post(url + "/messages", json=body).json() == first.json()
        assert len(calls) == 1
        assert client.post(url + "/messages", json=body | dict(content="different")).status_code == 409
        assert client.get(url + "/messages?offset=1&limit=1").json()[0]["role"] == "AI"
        saved = client.get(url).json()
    with TestClient(create_app(providers(calls=calls), storage=factory())) as client:
        assert client.get(url).json() == saved
        assert client.post(url + "/messages", json=body).json() == first.json()
        assert len(calls) == 1
        assert len(client.get(url + "/messages").json()) == 2
        assert client.get("/conversations/unknown/messages").status_code == 404


def test_atlas_retry_preserves_old_snapshot_and_cache(atlas):
    factory, _, ids = atlas
    with TestClient(create_app(providers(bad_question=True), storage=factory())) as client:
        cid = client.post("/conversations", json={"patient_id": "synthetic"}).json()["conversation"]["conversation_id"]
        ids.add(cid)
        url = f"/conversations/{cid}/messages"
        response = client.post(url, json=dict(message_id="m1", content="headache"))
        assert response.json()["output"]["status"] == "FAILED"
    store = factory()
    with TestClient(create_app(providers(), storage=store)) as client:
        retried = client.post(url + "/m1/retry", json=dict(retry_id="r1"))
        assert retried.status_code == 200
        assert retried.json()["output"]["next_action"] == "ASK_MORE"
        assert len(retried.json()["session"]["conversation"]["messages"]) == 2
        with store.transaction():
            before = store.get("before_turns", (cid, "m1"))
            assert before["conversation"]["messages"] == []
    with TestClient(create_app(providers(), storage=factory())) as client:
        assert client.post(url + "/m1/retry", json=dict(retry_id="r1")).json() == retried.json()


def test_atlas_commit_failure_rolls_back_all_documents(atlas, monkeypatch):
    factory, new_id, _ = atlas
    store, cid = factory(), new_id()
    def fail(*args):
        raise StorageError("Synthetic failure after session and history writes")
    monkeypatch.setattr(store, "_project_history", fail)
    with pytest.raises(StorageError):
        with store.transaction():
            store.put("sessions", cid, payload(cid).model_dump(mode="json"))
    with store.transaction():
        assert store.get("sessions", cid) is None
        assert store.history(cid, 0, 50) is None


def test_atlas_concurrent_writers_cannot_overwrite_each_other(atlas):
    factory, new_id, _ = atlas
    first, second, cid = factory(), factory(), new_id()
    with first.transaction():
        first.put("sessions", cid, payload(cid).model_dump(mode="json"))
    barrier = Barrier(2)
    def write(pair):
        store, age = pair
        try:
            with store.transaction():
                current = store.get("sessions", cid)
                current["conversation"]["clinical_state"]["patient_info"]["age"] = age
                barrier.wait(timeout=10)
                store.put("sessions", cid, current)
            return "committed"
        except StorageConflict:
            return "conflict"
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, [(first, 30), (second, 40)]))
    assert sorted(results) == ["committed", "conflict"]
    with first.transaction():
        assert first.get("sessions", cid)["conversation"]["clinical_state"]["patient_info"]["age"] in (30, 40)


def test_atlas_snapshot_versions_do_not_duplicate_current_messages(atlas):
    factory, new_id, _ = atlas
    store, cid = factory(), new_id()
    current = payload(cid).model_dump(mode="json")
    msg = dict(message_id="m1", role="USER", content="synthetic", timestamp="2026-11-15T00:00:00Z",
               case_ids=[], asked_requirements=[])
    current["conversation"]["messages"].append(msg)
    with store.transaction():
        store.put("sessions", cid, current)
        store.put("before_turns", (cid, "m2"), current)
    current["conversation"]["messages"][0]["case_ids"] = ["synthetic-case"]
    with store.transaction():
        store.put("sessions", cid, current)
    with store.transaction():
        assert store.history(cid, 0, 50)[0]["case_ids"] == ["synthetic-case"]
        assert store.get("before_turns", (cid, "m2"))["conversation"]["messages"][0]["case_ids"] == []
        assert len(store.history(cid, 0, 50)) == 1
