# MedCare - Agent 1 foundation

Python >=3.11, src layout. Includes validated schemas, structured intake updates,
doctor handoff review, and a minimal requirement/question engine. No running
bundled medical corpus or Agent 2/3 implementation yet. LangGraph now runs intake turns,
with Gemini extraction, experimental risk screening and question wording.
An optional local HTTP API supports Thunder Client tests with memory or SQLite sessions.

Only implemented modules are kept. Empty API/RAG/adapter scaffolding and unused
scripts have been removed; add modules when implementing their functionality.

## Install and test (PowerShell)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest
```

With uv:

```powershell
uv venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
.venv\Scripts\python.exe -m pytest
```

Tests use synthetic data and make no network calls. Dependency installation
requires network access. To test schemas without installing the whole project:

```powershell
uv pip install --python .venv\Scripts\python.exe pydantic pytest
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m pytest -q
```

## Clinical State

The state follows the requested grouped JSON structure, using English field names, defined together in
`src/medical_chatbot/schemas/clinical_state.py`:

- `patient_info`: age, sex, pregnancy_status.
- `medical_history`: conditions, surgical_history, current_medications,
  allergies, family_history.
- `current_case_id`: null or an existing case ID.
- `cases`: cases containing chief_complaint, status,
  response_mode, present_illness, clinical_data, progress_history,
  risk_assessment, missing_information, workflow, and doctor_handoff.

Values remain plain: age is an integer/null, descriptions are strings/null,
and collections are arrays. Each case requires a code-assigned case_id.
English names replace the original Vietnamese keys; the hierarchy is unchanged.
No aliases or duplicate versions of the state are maintained.

```python
from medical_chatbot.schemas.clinical_state import ClinicalState, ClinicalCase

state = ClinicalState(cases=[ClinicalCase(case_id="case_001")])
print(state.model_dump_json(indent=2))
```

See `docs/clinical_state.example.json` for the complete initial JSON.

## Validation choices

- The duplicated character is a single string/null field.
- progress_history starts empty. Actual events require a field, old/new values,
  a timezone-aware timestamp, and a TEMPORAL_CHANGE/CORRECTION type.
- New cases default to AI_ACTIVE; DOCTOR_ACTIVE is explicitly supported.
- Risk defaults to NOT_ASSESSED, never SAFE. False escalation flags at this
  stage do not represent a completed safety assessment.
- Facts and conflicts remain separate processing contracts for provenance,
  denial and uncertainty; they do not wrap values inside Clinical State.
  Fact/Requirement field paths match state fields. The merger returns this
  metadata separately; the API can persist it with SQLite. Empty lists do not
  mean the patient explicitly denied all items.
- RiskResult represents turn assessment status/errors; CaseRisk is the
  case summary. Synchronization is for a future processing step.
- Case workflow/mode describe business state. Graph state holds execution data.
- Undeclared fields, invalid enums, invalid ages and inconsistent references
  are rejected. Schema validation alone does not perform clinical assessment,
  doctor authentication, or semantic verification of extracted facts. Workflow
  changes happen in the clinical functions described below.

## Structured intake updates

`clinical/state_merger.py` provides one pure entry point:

```python
result = update_clinical_state(state, message, extracted,
                               facts=previous_facts, conflicts=previous_conflicts)
```

- Validate raw input with `Message.model_validate` and
  `InputProcessorOutput.model_validate` first. Invalid shapes/forbidden fields
  raise ValidationError; invalid source excerpts appear in `result.rejected`.
- The function returns a new Clinical State, accepted fact history, pending
  conflicts, unbound/uncertain proposals, and rejected facts. Inputs are unchanged.
- Retain and pass the returned facts/conflicts into the next call. These are
  application-owned metadata, not model-supplied data. Persist pending proposals
  too if clarification will span turns. The local API now saves these together.
- Existing case IDs must belong to the supplied state. New case IDs are generated
  in code. A new problem must be a message excerpt; this deliberately conservative
  demo does not support normalized/paraphrased new problem labels yet.
- Unknown case proposals remain pending; patient facts can still be updated.
- Lists append unique values. Vitals update one key at a time. Uncertain facts
  never become confirmed state values; explicit denials stay in fact metadata.
- Conflicting values require CORRECTION/TEMPORAL_CHANGE plus an explicit text cue
  in the source excerpt. Supported cues are defined in `_has_change_evidence`.
  This is conservative validation, not a language understanding or medical model.
  A later proposal with evidence can resolve a matching pending conflict.
- To remove a list element, its original keyed fact metadata must be available;
  the code will not guess how a semantic key maps to an untracked plain string.
- Open clinical values are scalar entries only in this step, preventing nested
  updates hidden inside clinical data. Rich lab/imaging objects remain a future
  extension of the validator, even though the state schema can represent them.
- An applied case fact reopens a resolved case. Accepted medical input invalidates
  earlier SAFE summaries, including patient-level changes. DANGEROUS and doctor
  mode are preserved; this function does not assess risk or authorize PASS.
- Requirement satisfaction is handled separately by the engine below. Full
  routing, message retry/idempotency and external access control are not implemented.

Run the synthetic demo (no extraction model, API or network calls):

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe scripts\demo_state_update.py
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

## Doctor review before returning to AI

Implemented in the existing `clinical/case_manager.py`:

1. `prepare_doctor_handoff(conversation, case_id, doctor_context)` returns a
   review draft and case messages. Official state stays DOCTOR_ACTIVE.
2. The review form edits `draft.content`: summary, conclusion, instructions,
   follow_up. There is no UI or AI summarizer in this step; the initial summary
   simply quotes doctor messages. Closing/cancelling the form makes no change.
3. `confirm_doctor_handoff(conversation, server_draft, edited_content,
   doctor_context, action="RETURN_TO_AI", confirmed_at=...)` returns a new
   Conversation with the approved doctor_handoff and AI_ACTIVE mode together.
   CLOSE_CASE instead marks the case RESOLVED and workflow COMPLETED. Neither
   action sends an AI reply automatically. A SYSTEM message records confirmation.
4. If history or state changed while reviewing, confirmation fails; regenerate
   the draft and ask the doctor to review new data. Drafts stay outside Clinical
   State; only approved content goes into the case. Original messages are retained.

DoctorContext must come from backend authentication of the doctor role,
never from user/LLM fields. Authentication itself is not implemented here.
The draft and its snapshot must be kept server-side; submit only edited content
and the chosen action from the browser. Persistence must atomically compare/save
the conversation version; the in-memory check does not prevent database races.
The latest approved handoff stays in state; immutable revisions/storage remain
for the persistence step. AI can receive it through Agent1Input.conversation.
Risk is not automatically SAFE after handoff, and DANGEROUS is preserved.

## Minimal requirement engine

Existing modules now provide:

- `requirements/engine.py`: `evaluate_requirements` returns a copied state,
  selected requirements and an intake status, without generating question text.
- `requirements/satisfaction.py`: a four-item core catalog (chief complaint,
  onset, duration, course) and semantic matching against accepted facts/state.
- `requirements/question_selector.py`: up to two questions from the highest
  remaining blocking tier, CRITICAL before HIGH; OPTIONAL never blocks.
- `agents/intake/question_generator.py`: Gemini-generated Vietnamese questions,
  using selected requirements, clinical context and recent messages.

After successful question delivery, call `mark_question_sent(result, question)`
and retain its returned state; it records `asked_count` on missing requirements.
Selection and generation alone never mark a requirement asked.
Partial answers satisfy only the matching items. Unasked items take precedence
within the highest priority tier; remaining unanswered items get a clarification
question. Asking once does not imply refusal or trigger doctor handoff.
`record_unavailable_answer(state, case_id, key, status)` records an
explicit UNKNOWN/DECLINED/UNAVAILABLE answer. Such items stay documented and are
skipped while collecting other available information; they do not trigger referral.

Additional requirements are supplied by trusted application code with explicit
applicable/blocking flags. Identical keys in the same scope are deduplicated;
conflicting definitions fail validation. A symptom denial satisfies its exact
key, not every symptom sharing the same field. No medical applicability inference or
complaint-specific medical catalog has been implemented.

Pass accepted fact history/conflicts into the engine and set case_uncertain
from case resolution. Case/conflict clarification precedes routine questions.
DOCTOR_ACTIVE produces NO_REPLY; danger produces a handoff
status. The engine does not run safety monitoring while doctor mode is active;
that remains the caller's responsibility. Optional/nonblocking questions are
not proactively asked in this minimal flow.

No risk result is inferred from a stored SAFE case summary. When information
is sufficient, the result is READY_FOR_REVIEW even if risk is NOT_ASSESSED.
Intake completeness is separate from clinical safety; unavailable answers remain
visible to downstream reviewers. The graph emits PASS_TO_AGENT_2 as a data handoff,
not a safety clearance, diagnosis, or treatment recommendation.
The pure requirement engine does not call RAG or a database. The intake graph and
local API below now coordinate it; delivery retries remain outside this engine.

All authenticated doctors may review any existing doctor-controlled case; there
is no case allowlist. A doctor still cannot confirm another doctor's draft.
The existing demo runs two structured-input turns through merger + engine.

## Gemini question generation

Install optional SDK dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev,gemini]"
Copy-Item .env.example .env
```

Set GEMINI_API_KEY in .env (server only). GEMINI_MODEL defaults to
gemini-2.5-flash-lite and can be changed to a model available to your project.
GEMINI_TIMEOUT_MS defaults to 20000 ms, with one SDK attempt per call.
The code does not enable billing or enforce free-tier eligibility: account tier,
model access and quota are controlled by Google. There is no paid-model fallback.
Check [Google pricing](https://ai.google.dev/gemini-api/docs/pricing) for current
free-tier access and data-use terms. Google lists free-tier inputs/outputs as
usable to improve products; use synthetic data for this demo.

```powershell
$env:PYTHONPATH = "src"
# Offline: show selected requirements, no fixed questions or API requests.
.venv\Scripts\python.exe scripts\demo_state_update.py
# Live: send synthetic messages to Gemini for extraction and question wording.
.venv\Scripts\python.exe scripts\demo_state_update.py --live
```

Application flow:

```python
result = evaluate_requirements(state, facts=facts, conflicts=conflicts)
question = generate_question(result, history=recent_messages, conflicts=conflicts)
if question is not None:
    # Send successfully to the user first, then commit this returned state.
    result = mark_question_sent(result, question)
```

No fixed questions remain, including case/conflict clarification. The model is
instructed to cover only approved scopes, ask in Vietnamese and avoid diagnosis,
treatment or invented facts. JSON shape and exact scope IDs are checked at runtime.
These checks cannot prove semantic/medical correctness; evaluate real model
wording before clinical use. Only the last ten supplied messages are included.
Pass full conflict metadata for conflict clarification, not just the conflict count.

Blocked/empty responses, bad JSON, missing/extra scopes, timeouts and provider
errors raise QuestionGenerationError. There is no hardcoded fallback and no
asked_count update on failure. Callers handle retry/handoff; exceptions exposed
by this module do not include raw provider requests, API keys or patient text.
Tests inject a fake JSON generator and never use the API. Live verification
requires SDK installation, a valid key and model access. This step does not
implement specialist requirement discovery.

## Input Processor and short answers

`agents/intake/input_processor.py` implements `process_input(conversation, message)`.
It supplies clinical state, up to ten recent messages, the immediately preceding
AI question (including its asked_requirements), allowed field paths and the new
USER message to Gemini. A validated InputProcessorOutput is returned without
mutating the conversation. The merger still owns IDs, binding, conflict checks
and state updates. Extracted values are scalar, matching the current merger.

```python
extracted = process_input(conversation, user_message)
merged = update_clinical_state(conversation.clinical_state, user_message, extracted,
                               facts=previous_facts, conflicts=previous_conflicts)
intake = evaluate_requirements(merged.state, facts=merged.facts,
    conflicts=merged.conflicts,
    case_uncertain=extracted.case_result.resolution == "UNCERTAIN")
```

Before sending an AI question, build its history message using
`make_question_message(result, question, message_id=..., timestamp=...)`.
After successful delivery, call mark_question_sent and save both the returned
state and the AI message in Conversation. Building the message alone does not
increment counters. Message.asked_requirements records the exact selected targets
and keys so a short answer such as 'two days' has context. Store question delivery
and state consistently when persistence is added.

Explicit unknown/refused/unavailable answers are proposals with their own source
excerpts. They must reference an actually asked requirement in the preceding AI
message, the resolved case, and a currently missing requirement. The merger saves
the status; the engine decides whether it blocks intake. Missing portions of a
partial answer are not automatically marked unknown or refused.

Wrong source messages, invented excerpts, foreign cases, forbidden fields and
invalid JSON fail extraction. Failures raise sanitized InputProcessingError;
there are no default facts or SAFE results. Shape/provenance checks do not prove
the model interpreted the language correctly. Safety assessment must still inspect
raw user messages independently, including when extraction fails.

The Gemini extraction contract uses one `facts` list. Python assigns each allowed
field to patient-level or case-level data before validating InputProcessorOutput.
This avoids an observed live failure where Gemini placed chief_complaint/frequency
inside patient_facts. Previous injected providers using the two-list format remain
supported; mixed formats and invalid sources/values are rejected. Identical onset
and duration proposals from the same excerpt are not counted as two independent
answers: onset is retained and total duration remains to be collected.

SDK requests allow at most two attempts for HTTP 500/502/503/504 with a short
backoff, using the same configured model. Validation failures are not automatically
retried. The explicit message retry endpoint is separate. Intake completion now
records a short AI acknowledgement alongside PASS_TO_AGENT_2, without claiming
that Agent 2 has run or a doctor has accepted the case.

Live screening uses `evidence_message_ids` rather than model-generated quotations.
The server validates that every ID belongs to a supplied USER message and copies
its original text into the internal evidence. `doctor_requested` refers only to the
new message. Unknown/non-USER IDs remain errors; this establishes provenance, not
medical correctness or proof that the selected message supports the interpretation.

Offline tests inject generate_json. The --live demo now uses real extraction
and real question generation on synthetic messages; it is not an interactive
chatbot. The application still needs to retain fact/conflict/pending metadata,
save history and handle delivery/concurrency. Specialist requirements, risk
assessment and RAG are not implemented by Input Processor itself. The graph now
calls the separate experimental risk screener described below.

## Agent 1 graph and local Thunder Client API

`agents/intake/graph.py` builds a real LangGraph with these sequential nodes:
raw-message screening -> extraction/merge -> routing/question generation.
`graph/state.py` defines transient execution state and the session containing
Conversation plus facts, conflicts, pending proposals and sourced risk metadata.
`agents/intake/risk_assessor.py` calls Gemini for experimental safety screening.

The raw message is screened even if extraction subsequently fails. Invalid JSON,
missing/fake evidence or provider errors cannot become SAFE. Existing danger is
not downgraded. Immediate danger can emit EMERGENCY. Doctor handoff is limited to
danger, explicit doctor requests, or insufficient RAG information for a concrete
specialist question. Provider/extraction/RAG failures return retryable FAILED;
uncertain risk alone does not trigger referral or imply SAFE.
Doctor/pending-doctor modes suppress routine AI replies while
still recording user data and running safety screening. Emergency monitoring
can still produce a safety message.

This is NOT a clinically validated triage system. LLM citations to patient text
are checked for provenance, not medical correctness. There is no bundled medical
corpus or complete complaint-specific catalog. Intake completes when applicable
blocking items have answers or explicit unavailable statuses, the complaint/case
is known, no relevant conflicts remain, and no technical failure or danger blocks
the turn. No doctor dispatch service or Agent 2 is called.
Old pending facts are retained and only individually source-backed confirmations
may bind them to a selected case, as described below.

Install and start locally:

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[dev,api,gemini]"
# Configure GEMINI_API_KEY in .env using .env.example. Do not commit .env.
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m uvicorn medical_chatbot.api:app --host 127.0.0.1 --port 8000
```

Thunder Client steps (Content-Type: application/json):

1. GET `http://127.0.0.1:8000/health`.
2. POST `http://127.0.0.1:8000/conversations`:

```json
{"patient_id": "test_patient_001"}
```

Copy `conversation.conversation_id` from the response.

3. POST `http://127.0.0.1:8000/conversations/CONVERSATION_ID/messages`:

```json
{"message_id": "msg_001", "content": "Tôi đau đầu hai ngày rồi"}
```

The response contains `output` (action, risk, reply or processing error) and
`session` (updated state/history/metadata). Send the next answer to the same URL
using a new message_id. Patient/history IDs and timestamps are server-controlled;
clients cannot submit role, clinical state, mode or model-generated facts.

4. GET `http://127.0.0.1:8000/conversations/CONVERSATION_ID` to inspect the session.
5. API schema/UI: `http://127.0.0.1:8000/docs`.

Repeating the same message ID/content returns the cached result without another
model call; changing content for that ID returns 409. Unknown conversations return
404; invalid request bodies return 422. Unexpected graph failures return sanitized
503 with no state commit. A known question-generation failure returns an output
with status FAILED while retaining the user message and accepted data, and does
not mark requirements asked. Repeating the normal message endpoint reproduces
the saved outcome; use the dedicated retry endpoint below to retry provider work.

### Pending case confirmation

Input Processor receives server-owned pending facts. Its `pending_bindings`
identify an original message ID, field and semantic key, plus an exact confirmation
excerpt from the new message. Code validates both messages and case resolution
before replaying that fact against the selected case. Source/time remain original.
Newer conflicting values are never overwritten by old temporal/correction cues;
they create a pending conflict instead. No blanket assignment of pending facts.
Bindings of UNCERTAIN-status facts are rejected; the user must provide a definite
new answer. The supplied Gemini generator proposes semantic mappings; substring
checks prove provenance, not the correctness of that mapping.

### Retrying a failed operation

POST `/conversations/CONVERSATION_ID/messages/msg_001/retry`:

```json
{"retry_id": "retry_001"}
```

- Question failures retry only wording using saved selection/history; extraction,
  user history and accepted facts are not duplicated.
- Risk/extraction failures may replay the turn from the saved pre-turn snapshot.
  The message retains its original ID/time and replaces the failed turn result.
- Only the current unchanged failed turn is retryable. New input, doctor changes,
  no processing error or an existing DANGEROUS output returns 409.
- Two explicit retries maximum per original message; no background retry loops.
- Repeating the same retry_id returns its cached result without another call.
  Use a new retry_id for the second attempt. Unknown messages return 404.
- Accepted data survives question failures. An unsuccessful retry never marks a
  question asked or overwrites valid danger findings.

### LLM screening versus approved medical risk

RiskResult now exposes assessment_source. Gemini screening is LLM_SCREENING,
including its SAFE result; it is not a document-grounded clinical conclusion.
Screening signals remain in session metadata and can conservatively trigger
review/emergency messaging. They do not write official case SAFE/DANGEROUS fields.
The PASS output contract indicates intake completion and does not require SAFE.
Only RAG/DOCTOR assessments update official case risk conclusions.
Do not expose a screening SAFE as a guarantee that the patient is safe.

An authenticated doctor's confirmed handoff supersedes older provisional screening
signals whose source messages predate that review. New danger and official
RAG/doctor findings are not cleared by this rule. Case handoff functions still
operate through trusted backend calls; no doctor authentication/UI is added here.

### Reviewed edge cases

New emergency signals take precedence over older non-emergency signals even when
extraction fails. Unmapped danger is retained in session.unbound_risk, not assigned
to an arbitrary case or forgotten on the next message. Switching complaints also
cannot bypass unresolved danger. Clearing unbound provisional signals requires a
doctor-reviewed case explicitly containing their original source messages.

Mapped USER messages and confirmed pending-source messages now carry case IDs so
the doctor's review includes the patient history, not just AI/doctor messages.
Clinical numeric values reject NaN/infinity. Progress events carry a semantic key
(e.g. temperature versus pulse); older messages cannot overwrite a newer accepted
value. Temporal cues are matched as words/phrases, so 'unknown' is not 'now'.
Event time still uses message time; actual measurement times require a future
explicit time-extraction contract. These are runtime safeguards, not proof of
medical correctness or full language understanding.

`api.py` is a local development service without authentication. Keep it on
127.0.0.1 and use synthetic data. The stored AI message is a replayable response,
not proof that a remote human received it. Production still requires access
control, data retention/migration policy, backup and proper message delivery.

### Local SQLite persistence

```powershell
$env:MEDCARE_DB_PATH = '.\agent1.sqlite3'
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe -m uvicorn medical_chatbot.api:app --host 127.0.0.1 --port 8000
```

Set MEDCARE_DB_PATH in the shell before startup, not just in .env (the API reads
its storage configuration before any Gemini call). By default the API stores data
in `agent1.sqlite3` in the working directory. Use `MEDCARE_DB_PATH=:memory:` only
for temporary sessions. /health reports sqlite or memory. SQLite retains sessions, accepted
facts, pending/conflicts, replies, pre-turn retry snapshots and retry counts across
restart. Session and reply changes commit in one transaction, and rollback on
failure. Transactions serialize requests across instances sharing the same file;
provider calls hold the transaction, so this is intentionally for low-volume local
tests. Busy/corrupt/incompatible storage returns a sanitized 503. The file is plain
SQLite, not encrypted clinical storage; do not use real patient data for testing.

`GET /conversations/{conversation_id}/messages?offset=0&limit=50` reads the saved
history in conversation order (limit 1-200). `GET /conversations/{conversation_id}`
returns the full session, including clinical state and fact provenance. The existing
`agent_data(namespace, key, value)` table keeps these snapshots and retry records
in the same transaction; no second competing history store is needed.

Tests use injected providers and FastAPI TestClient; no real Gemini calls. Run:

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

## Local medical RAG pipeline

`rag.py` contains document metadata, lexical retrieval, Gemini intake support and
evidence validation. No medical guidance is bundled, and no embedding/vector
database is selected. This lexical baseline supports UTF-8 text/Markdown and
Vietnamese accent normalization; it does not provide semantic synonym search.

Ingest a document (PDF must first be converted to text/Markdown):

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe scripts\ingest_documents.py .\documents\guideline.md --source-id guideline_001 --title "Document title" --version "2026-01"
```

The generated index is `data/rag/corpus.jsonl`. Each chunk has source_id, title,
version, chunk_id, source location and text. Re-ingesting the same source replaces
its active chunks, including older versions, to avoid mixing conflicting editions.
The original document is never overwritten. Corpus files are excluded from git.

Documents default to UNREVIEWED. Only when the corpus owner has actually approved
the source, ingest with BOTH `--reviewed-by` and `--reviewed-at` (ISO timestamp with
timezone). These fields are owner-supplied metadata, not proof of medical review
performed by this software. Do not mark arbitrary/test documents medically approved.
Without reviewed retrieved chunks, the model is not called for a clinical outcome.

Configure before starting uvicorn:

```powershell
$env:MEDCARE_RAG_PATH = '.\data\rag\corpus.jsonl'
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe -m uvicorn medical_chatbot.api:app --host 127.0.0.1 --port 8000
```

Gemini key/model configuration is unchanged. /health distinguishes RAG support
from `rag_corpus_configured`; a configured path does not certify usable documents.
There is no new RAG HTTP endpoint: existing POST /messages invokes retrieval when
extraction/screening identifies a concrete need for specialist follow-up questions.
Completing basic intake does not automatically invoke RAG. Danger or an explicit
doctor request is not delayed for retrieval. Doctor and
pending-doctor cases do not run routine RAG automatically.

Response `rag` shows READY / UNAVAILABLE / INSUFFICIENT / FAILED, supporting
quotes, chunk IDs and source versions. Bad citations, unreviewed sources, fabricated
patient excerpts, unsupported requirement evidence or conflicting requirement
policies fail closed. RAG failures appear in processing_errors and can use the
existing retry endpoint subject to the same danger/staleness/attempt safeguards.

When `rag` is null, `rag_skip_reason` explains why: `extraction_failed`,
`risk_failed`, `case_unbound`, `doctor_mode`, `direct_escalation`, or `not_requested`.
Extraction errors preserve provider/timeout versus validation categories. Schema
errors expose sanitized field locations/types, never raw model responses. SDK
failures expose numeric HTTP status when available; logs contain categories only.
Rejected merge proposals roll back the entire turn's clinical updates while the
USER message remains recorded for retry. Duplicate processing errors are omitted.

Intake field meanings are shared between extraction and question generation:
`onset` is when the problem began; `duration` is total duration;
`episode_duration` is the length of each episode; `frequency` is how often it
occurs; `course` describes improvement/worsening/no change over time. Optional
new fields default to null when reading older sessions; old values are not silently
reclassified. Scalar fact keys are normalized by code; collection keys retain their
item-specific meaning. Source checks remain strict and require exact USER excerpts.

Failed turns return `user_notice` for the client to display, separately from clinical
questions/history. This does not add an AI question or increment asked_count. Risk
validation identifies specific code-owned rule names, e.g.
`risk_evidence_quote_not_exact`. Question retries preserve those error categories;
routing failures are not mislabeled as failed question generation. There is no UI
in this repository: API consumers must render user_notice and expose the retry action.

RAG returns additional requirements, red flags to ask about, and a recommendation.
Red flags to check become CRITICAL requirements, never confirmed patient facts.
The engine runs again; danger is handled before ordinary questions or PASS. SAFE
requires reviewed document citations AND patient evidence. PASS is only the
Agent 1 handoff contract: it does not run Agent 2 or send treatment advice.

Scope/source checks establish provenance, not semantic or medical correctness.
Validate corpus suitability, recall, interpretation and routing before real patient
use. Tests use artificial text and fake models only; they do not validate medical
content. Separate live diagnostics on synthetic conversations exercise actual Gemini
and the local API; their success does not validate medical screening or RAG content.
