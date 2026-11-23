"""Local test API with Atlas or SQLite persistence; no patient authentication."""
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import os
import sqlite3
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import Field, ValidationError
from medical_chatbot.schemas.base import NonEmpty, Schema
from medical_chatbot.schemas.conversation import Conversation, Message
from medical_chatbot.agents.intake.graph import IntakeSession, TurnResult, build_intake_graph, run_turn, retry_question
from medical_chatbot.storage import create_store, StorageError, StorageConflict


class CreateConversation(Schema):
    patient_id: NonEmpty


class SendMessage(Schema):
    message_id: NonEmpty
    content: NonEmpty = Field(max_length=10000)


class RetryMessage(Schema):
    retry_id: NonEmpty


def create_app(graph=None, *, database_path: str | None = None, storage=None) -> FastAPI:
    store = storage if storage is not None else create_store(database_path)

    @asynccontextmanager
    async def lifespan(app):
        try:
            if hasattr(store, "initialize"):
                store.initialize()
            yield
        finally:
            store.close()

    app = FastAPI(title="MedCare Agent 1 - local testing", version="0.1.0", lifespan=lifespan)
    workflow = graph if graph is not None else build_intake_graph()

    @app.exception_handler(sqlite3.DatabaseError)
    @app.exception_handler(ValidationError)
    @app.exception_handler(StorageError)
    async def storage_error(request, error):
        return JSONResponse(status_code=503, content={"detail": "Stored session unavailable or storage is busy"})

    @app.exception_handler(StorageConflict)
    async def concurrent_update(request, error):
        return JSONResponse(status_code=409, content={"detail": "Session changed during processing; reload before retrying"})

    def load_session(conversation_id):
        data = store.get("sessions", conversation_id)
        if data is None:
            raise HTTPException(404, "Conversation not found")
        return IntakeSession.model_validate(data)

    def save_result(conversation_id, key, content, result):
        store.put("sessions", conversation_id, result.session.model_dump(mode="json"))
        store.put("replies", key, {"content": content, "result": result.model_dump(mode="json")})

    @app.get("/health")
    def health():
        return {"status": "ok", "storage": store.backend, "clinical_validation": False,
                "rag": True, "rag_corpus_configured": bool(os.getenv("MEDCARE_RAG_PATH")), "agent_2": False}

    @app.post("/conversations", response_model=IntakeSession, status_code=201)
    def create(body: CreateConversation):
        conversation = Conversation(conversation_id=f"conv_{uuid4().hex}", patient_id=body.patient_id)
        session = IntakeSession(conversation=conversation)
        with store.transaction():
            store.put("sessions", conversation.conversation_id, session.model_dump(mode="json"))
        return session

    @app.get("/conversations/{conversation_id}", response_model=IntakeSession)
    def read(conversation_id: str):
        with store.transaction():
            return load_session(conversation_id)

    @app.post("/conversations/{conversation_id}/messages", response_model=TurnResult)
    def send(conversation_id: str, body: SendMessage):
        with store.transaction():
            current = load_session(conversation_id)
            key = (conversation_id, body.message_id)
            cached = store.get("replies", key)
            if cached is not None:
                if cached["content"] != body.content:
                    raise HTTPException(409, "Message ID reused with different content")
                return TurnResult.model_validate(cached["result"])
            message = Message(message_id=body.message_id, role="USER", content=body.content,
                              timestamp=datetime.now(timezone.utc))
            try:
                result = run_turn(workflow, current, message)
            except Exception:
                # Do not expose request payloads, model exceptions or secrets.
                raise HTTPException(503, "Agent processing failed; state was not committed") from None
            save_result(conversation_id, key, body.content, result)
            if result.processing_errors:
                store.put("before_turns", key, current.model_dump(mode="json"))
            return result

    @app.get("/conversations/{conversation_id}/messages", response_model=list[Message])
    def history(conversation_id: str, offset: int = Query(0, ge=0),
                limit: int = Query(50, ge=1, le=200)):
        """Read persisted messages in conversation order, including doctor and AI messages."""
        with store.transaction():
            if hasattr(store, "history"):
                messages = store.history(conversation_id, offset, limit)
                if messages is None:
                    raise HTTPException(404, "Conversation not found")
                return [Message.model_validate(message) for message in messages]
            session = load_session(conversation_id)
            return session.conversation.messages[offset:offset + limit]

    @app.post("/conversations/{conversation_id}/messages/{message_id}/retry", response_model=TurnResult)
    def retry(conversation_id: str, message_id: str, body: RetryMessage):
        with store.transaction():
            key = (conversation_id, message_id)
            retry_key = (*key, body.retry_id)
            current = load_session(conversation_id)
            original_reply = store.get("replies", key)
            if original_reply is None:
                raise HTTPException(404, "Conversation or message not found")
            cached = store.get("retry_replies", retry_key)
            if cached is not None:
                return TurnResult.model_validate(cached)
            previous = TurnResult.model_validate(original_reply["result"])
            attempts = store.get("retry_counts", key, 0)
            if (attempts >= 2 or current != previous.session
                    or not previous.processing_errors or previous.output.risk.result == "DANGEROUS"
                    or previous.output.next_action in {"ASK_MORE", "PASS_TO_AGENT_2"}):
                raise HTTPException(409, "Retry unavailable: state changed, no retryable error, danger or attempt limit")
            try:
                if current.pending_question_for == message_id:
                    result = retry_question(current, message_id,
                        question_json=getattr(workflow, "question_provider", None))
                    result.rag = previous.rag
                    result.rag_skip_reason = previous.rag_skip_reason
                else:
                    original = next(m for m in current.conversation.messages if m.message_id == message_id)
                    # Case tags were assigned after the original request. Replay
                    # raw input against the pre-turn state, where a new case may
                    # not exist yet (HTTP input never accepts client case tags).
                    before = IntakeSession.model_validate(store.get("before_turns", key))
                    result = run_turn(workflow, before, original.model_copy(update={"case_ids": []}))
            except ValueError:
                raise HTTPException(409, "Retry no longer matches the conversation state") from None
            except Exception:
                raise HTTPException(503, "Retry failed; state was not committed") from None
            save_result(conversation_id, key, original_reply["content"], result)
            store.put("retry_replies", retry_key, result.model_dump(mode="json"))
            store.put("retry_counts", key, attempts + 1)
            return result

    return app


app = create_app()
