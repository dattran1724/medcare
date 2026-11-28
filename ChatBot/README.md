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
DOCTOR_ACTIVE produces NO_REPLY because the doctor is responding directly.
WAITING_DOCTOR keeps AI active: new messages continue intake, and a pending
DANGEROUS assessment is included as a required warning on every AI response.
The first doctor request or danger handoff is written to the `doctor_notifications`
outbox in PENDING state. This repository has no configured doctor delivery
channel or dispatcher, so the outbox must be consumed by a later notification
integration. Optional/nonblocking questions are not proactively asked here.

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

### MongoDB Atlas persistence

Install the MongoDB extra into the project environment:

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[api,gemini,mongodb,dev]"
```

Configure `.env` (real credentials belong only in `.env`, never `.env.example`):

```dotenv
MONGODB_URI="mongodb+srv://USER:ENCODED_PASSWORD@CLUSTER/?retryWrites=true&w=majority"
MONGODB_DATABASE=medcare_chatbot
```

Start the API normally. Configuration loads before storage is selected. A nonempty
MONGODB_URI selects MongoDB and startup validates connectivity/indexes; failure does
not silently fall back to SQLite. `/health` reports `storage: mongodb`. Atlas access
requires a database user with read/write permissions and your IP on its access list.

Collections:

| Collection | Purpose |
| --- | --- |
| conversations | Patient ID, creation/update time, message count, state version |
| messages | Current messages in sequence order, unique conversation/message ID |
| facts | Current facts, unique conversation/fact ID |
| sessions | Clinical state, runtime metadata and references to messages/facts |
| turns | Cached replies, retry counts/results and pre-turn snapshots |
| message_versions / fact_versions | Immutable content-addressed revisions used by snapshots |

Snapshots reference immutable versions so updating a message's case tags does not
rewrite old cached replies or pre-turn data. History pagination reads messages
directly, without loading full clinical sessions. Message and fact bodies are not
embedded repeatedly in session/reply documents; their reference arrays still grow,
so large conversations need retention/archival planning before production use.

Gemini runs outside MongoDB transactions. Only the final short database commit uses
a snapshot/majority transaction with version checks. Concurrent stale writers return
409; failed writes return sanitized 503. Driver transaction retries never rerun Gemini.
Concurrent duplicate requests can both call Gemini, but cannot commit two different
outcomes; retrying the request after reloading returns the already saved outcome.

Existing SQLite files remain untouched. There is no automatic data migration. An
explicit database_path argument still selects SQLite for offline tests; without
MONGODB_URI the SQLite fallback below remains available.

Offline tests do not connect to Atlas. To opt into real Atlas integration tests:

```powershell
$env:PYTHONPATH = 'src'
$env:MEDCARE_TEST_MONGODB = '1'
.venv\Scripts\python.exe -m pytest tests/integration/test_mongodb.py -q -p no:cacheprovider --tb=short
Remove-Item Env:MEDCARE_TEST_MONGODB
```

These tests use fake clinical providers and unique synthetic conversation IDs.
They clean up only their own records, never drop collections/databases, and cover
restart/idempotency, retry snapshots, transaction rollback, concurrent writes and
historical message revisions. If a test is forcibly interrupted, its synthetic
records may remain and can be identified by its synthetic patient/conversation IDs.

### Local SQLite fallback

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

## PDF preparation: Docling to Markdown

Run these commands from the repository root. The root `rag/` package contains
document-preparation tools, separate from the existing Agent 1 retrieval module.
Only PDF-to-Markdown is implemented here; it does not chunk, embed or write MongoDB.

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[documents]"
.venv\Scripts\python.exe -m rag.extract .\rag\data\pdf\guideline.pdf
```

To process all PDFs directly in `rag/data/pdf/`, run without a source argument:

```powershell
.venv\Scripts\python.exe -m rag.extract
```

The batch shares one converter/model instance, skips existing Markdown by default,
continues after individual PDF failures, and prints saved/skipped/failed totals.
Use `--overwrite` to convert existing outputs again, or `--ocr force` for scans.
It does not scan subfolders. You can also pass another input folder; `-o` then
specifies an output folder. A nonzero exit code means the batch had a failure or
contained no PDFs. An explicit single PDF and `-o PATH.md` remain supported.

Default output: `rag/data/markdown/guideline.md`. Override it with `-o PATH.md`.
Both data directories are ignored by Git. The PDF remains untouched. An existing
output is only replaced with `--overwrite`, after successful conversion.

Device selection applies to layout, tables and EasyOCR together:

```powershell
# Default: choose CUDA if available, otherwise CPU (process all PDFs):
.venv\Scripts\python.exe -m rag.extract
# Require an NVIDIA GPU; fail clearly if CUDA-enabled PyTorch is unavailable:
.venv\Scripts\python.exe -m rag.extract --device cuda
# Force CPU:
.venv\Scripts\python.exe -m rag.extract --device cpu
```

Docling selects the accelerator and reports unavailable devices. OCR follows the
same accelerator. Layout/OCR/table batches are limited to one page to
reduce peak VRAM usage on smaller GPUs. GPU support requires a working NVIDIA driver
and CUDA-enabled PyTorch, not just an installed CUDA Toolkit. Low free VRAM can still
cause out-of-memory failures; close other GPU workloads or select CPU. There is no
silent CPU fallback after a CUDA runtime failure.

```powershell
# Native PDF text + layout/table detection, without OCR:
.venv\Scripts\python.exe -m rag.extract .\rag\data\pdf\guideline.pdf --ocr off
# Full-page OCR for scans or broken text layers (EasyOCR Vietnamese + English):
.venv\Scripts\python.exe -m rag.extract .\rag\data\pdf\guideline.pdf --ocr force
```

Default `--ocr auto` enables Docling's default OCR region selection; it does not
guarantee detection of every broken text layer. EasyOCR uses Vietnamese + English.
Docling layout/table and EasyOCR weights may download on first use. Inference
runs locally, with remote conversion services disabled; no Gemini key is required.
The PDF dependencies are optional and are not needed to run Agent 1.

Docling uses its PDFium backend for physical page numbering and native text,
with heading hierarchy recovery enabled. Markdown is exported page by page:

```markdown
<!-- page: 1 -->

# Section heading
Content...

<!-- page: 2 -->

Continuation...
```

Markers are physical PDF page numbers, not printed page labels. Empty pages retain
their number and produce a review warning; wholly unreadable or partially failed
documents are not published. Figures are placeholders/captions, not interpreted.
Cross-page tables may repeat in per-page export and are flagged for review.
By default, pdfplumber reconstructs ruled tables from native PDF lines and text.
Replacement requires a unique same-page table with bounding-box overlap (IoU >= 0.8),
but does not require matching Docling's row/column count. Actual cell rectangles
determine spans; empty cells stay empty. This can repair a misordered model grid
and recover blank answer columns. Borderless, scanned, cross-page and unmatched
tables retain Docling's result; this does not certify their accuracy. Tables with
no native text retain the OCR result. Use `--tables docling` for comparison.

All tables are exported as Markdown pipe tables. Cell line breaks are retained
as inline `<br>` markers; whole HTML tables are not generated. This preserves
lines within an existing cell without guessing which wrapped lines form drug/dose pairs.
Captions are placed before/after their table according to source geometry.
Docling expands actual row/column spans into
the corresponding grid cells, so each data row retains its group context. Multi-level
headers are joined per column (e.g. `Allowed - FEV1`). Tables without detected column
headers get a blank Markdown header; their first data row remains data. Real empty
cells stay empty, and numeric strings such as `2.10` are not converted to numbers.
There is no whole-table HTML output mode or string-based HTML repair. This preserves detected
cell relationships, not visual merged-cell formatting, and cannot recover a cell
or column that the recognition model missed.
Compare complex tables, figures and numeric content against the source before use.

For embedding-oriented text, use `--table-format rows`. Each source data row becomes
one Markdown list item with column labels, any detected caption and explicit blank
cell markers. Multi-level headers are combined per column; duplicate labels include
their source column number. A merged-cell group has shared fields followed by
nested source rows, keeping each criterion paired with its score. Values are never
flattened into separate column lists. A cell listing several drugs/doses stays
intact rather than being guessed into pairs. The existing chunker keeps each top-level list item whole
and adds the heading path when splitting a long section. Native cell line breaks
remain as indented text, without `<br>` tags. Standalone table-ID comments are for
review and are excluded by the current chunker; they are not yet structured chunk metadata.
In `rows` mode, continuation is inferred only between the last/first tables on
adjacent pages with matching nonempty headers and aligned column boundaries
(within 2% of page width). The previous table must reach the bottom 15% of its
page and the next must start in the top 15%, with no intervening body text,
heading or new caption. A known title anchors the chain. Accepted fragments share
a `table_id` and title but keep their own `part_id`, physical page and local row
numbers; cell data is not merged or renumbered. `continuation: inferred` and a
review warning make the heuristic explicit. Similar-header fragments lacking the
other signals retain separate IDs and `continuation: unconfirmed`. New captions
and changed headers start independent tables. This cannot prove every continuation.

```powershell
.venv\Scripts\python.exe -m rag.extract .\rag\data\pdf\guideline.pdf --table-format rows --overwrite
```

Default `--table-format markdown` keeps pipe tables for visual review. Store trial
outputs in `rag/data/review`, rather than keeping both formats in the batch chunk
input folder. Extraction/formatting checks do not measure retrieval accuracy.

For targeted review without reprocessing a long document:

```powershell
.venv\Scripts\python.exe -m rag.extract .\rag\data\pdf\guideline.pdf --pages 235 236 --ocr off
```

The default sample output gets `.pages-235-236.md` in its name and keeps physical
page markers 235/236. Table recognition uses accurate TableFormer mode and
Docling's default cell matching; disabling it lost text on the reviewed PDF and
is not the default. Page count/range is checked against the physical PDF before
publishing. Re-extract old Markdown to repair old page markers; existing files
are not silently rewritten.

Review a known problematic table without replacing the production Markdown:

```powershell
.venv\Scripts\python.exe -m rag.extract .\rag\data\pdf\3hdnititchuynha.pdf --pages 242 244 --ocr off --device cuda -o .\rag\data\review\3hdnititchuynha.updated.pages-242-244.md --overwrite
```

Keep review samples outside `rag/data/markdown` so batch chunking does not ingest
them as duplicate documents. These checks do not establish accuracy on all PDFs.
Merged text is projected by provenance character spans so later-page text is not
dumped onto the first page; detected source blocks retain paragraph breaks.
Within a text block, native PDF line spacing restores paragraph breaks where an
unusually large gap and a uniquely matching line start are present. Only whitespace
is inserted; consecutive numbered entries at native line starts are also separated,
including uniformly spaced lists. Wrapped lines stay joined and tables are excluded. Uniformly spaced
paragraphs and scanned text can still need review. `rag/layout.py` also repairs the
specific pattern of headings fused across a clear vertical gutter, along with their
following column blocks (e.g. agency/national heading or recipients/signature).
It uses native PDF coordinates/text, keeps left-column blocks together before the
right column and requires exact character conservation ignoring whitespace. Tables
are excluded; plain paired body text without a fused heading is not reinterpreted.
Scans without a usable native text layer and unsupported layouts remain unchanged.
Each repair emits a reading-order review warning; this is not a guarantee of correct
layout for every document. Full-page splitting into fixed left/right halves is not used.

Offline tests use injected conversion and the real serializer when installed.
For a real two-column PDF smoke test (first run can download model weights):

```powershell
$env:MEDCARE_TEST_DOCLING = '1'
.venv\Scripts\python.exe -m pytest tests/integration/test_document_extraction_live.py -q -p no:cacheprovider
Remove-Item Env:MEDCARE_TEST_DOCLING
```

To run that smoke test on CUDA, also set `MEDCARE_TEST_DOCLING_DEVICE=cuda` before
pytest. It checks that conversion actually allocates GPU memory, not only that
the configuration requests CUDA. Unset that variable after the test.

## Markdown chunking before embedding

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[chunking]"
.venv\Scripts\python.exe -m rag.chunk .\rag\data\markdown\guideline.md --overwrite
```

`rag/chunk.py` uses MarkdownIt's actual heading positions and levels to retain
source boundaries, formatting and indentation. Repeated titles at different
positions remain different sections. It does not infer hierarchy from numbering,
uppercase text or adjacency, and does not guess missing chapter/disease context.

Only sections above the token target are split further, in this order:

- Markdown paragraphs first; an oversized paragraph splits at sentence boundaries
  using NLTK Punkt with common medical-title abbreviations. It needs no model download.
- Lists split at top-level items, keeping nested children with their parent.
- Tables split at whole rows and repeat their existing column header.

The immediately preceding paragraph is carried with each list/table fragment so
its conditions are retained. Split fragments also carry their heading path in text.
There is no fixed overlap or rewriting. Sentence detection is structural, not a
clinical interpretation; verify conditions and negations when reviewing retrieval.
An indivisible sentence, nested item, table row, introduction or code block that
exceeds the target stays intact with `oversized_preserved_for_review`.

The default target is **1,200 tokens**, measured by reference tokenizer
`cl100k_base`. Set `--max-tokens 800` to change it. The limit is soft; the embedding
step must check its own model tokenizer and input limit. Standalone technical
comments are excluded from embedding text; comments inside code examples stay.

Output: `rag/data/chunks/<name>.jsonl`, one chunk per line with text, heading path,
source pages/line bounds, stable ID, source checksum, token count and review flags.
Each section has `section_id`, `section_path`, zero-based `part_index`, `part_count`
and source section bounds. Only part 0 stores the canonical `section_text` and
`section_pages`, so retrieval can return the exact full section without repeating
introductions or table headers. `document_chunk_count` records the complete file size.
Page metadata includes repeated source introductions/table headers. Line bounds
are conservative when a fragment repeats context or splits a single source line.
Existing files require `--overwrite`; conversion failure leaves the old output intact.
Rebuild old JSONL after changing Markdown or chunk settings.

Without a source argument, all Markdown directly in `rag/data/markdown` is processed.
Batch mode skips existing files; `-o` selects an output folder. Review samples such
as `.pages-*.md` are included too, so select the full document explicitly to avoid
indexing duplicate content. No embedding, LLM, API or database is called.


## Local dense embedding: BGE-M3

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[embedding]"
# All JSONL files directly in rag/data/chunks:
.venv\Scripts\python.exe -m rag.embed
# One file, or regenerate existing results:
.venv\Scripts\python.exe -m rag.embed .\rag\data\chunks\3375_QD-BYT_679213.jsonl --overwrite
# CPU / smaller GPU batch:
.venv\Scripts\python.exe -m rag.embed --device cpu --batch-size 2
```

`rag/embed.py` uses `sentence-transformers` with `BAAI/bge-m3`, pinned to revision
`5617a9f61b028005a4858fdac845db406aefb181`. First execution downloads the model
weights from Hugging Face (roughly 2.3 GB plus tokenizer/config files); later runs
reuse the local cache. Encoding runs locally and needs no Gemini/API key.
Automatic device selection uses Sentence Transformers; CUDA uses float16 weights,
CPU uses the default precision. Default batch size is 4; lower it if GPU memory is limited.

Each chunk's heading path is prepended when it is not already a leading prefix.
The exact input is retained as `embedding_text`; the original `text`, IDs, pages,
source checksum and other chunk metadata remain unchanged. The BGE-M3 tokenizer
checks input length including special tokens before encoding. If any chunk exceeds
`model.max_seq_length` (8,192 for this model), the file fails with its chunk ID;
no content is silently truncated and the previous output remains untouched.

Output: `rag/data/embeddings/<name>.jsonl`. Each record adds a normalized, finite
1,024-dimensional dense `embedding`, model/revision, dimensions, normalization flag
and `embedding_token_count`. The existing chunk `token_count` uses a different
reference tokenizer and is not the embedding input length. Sparse embeddings,
reranking, MongoDB writes and Agent 1 retrieval integration are not part of this script.

Batch mode loads one model, skips existing outputs, and continues after individual
file failures with a nonzero exit code. Use `--overwrite` after changing source chunks.
Add `--reuse-existing` to refresh metadata while reusing cached vectors only when
embedding text, token count, model/revision and normalized finite dimensions match.
Chunk IDs alone are never a reuse key; changed inputs are encoded normally. Keep old
embedding JSONL files in place to use this cache. Queries must later use the same model revision and normalized dense encoding.
Unit tests inject a small fake encoder to verify batches, validation and file safety;
they do not measure real BGE-M3 retrieval quality or GPU performance.


## Upload embeddings to MongoDB Atlas

Use the existing `.env` keys `MONGODB_URI` and `MONGODB_DATABASE`. The target is
always the `rag_chunks` collection; conversation collections are not touched.

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[mongodb]"
# All JSONL files directly in rag/data/embeddings:
.venv\Scripts\python.exe -m rag.upload
# One file:
.venv\Scripts\python.exe -m rag.upload .\rag\data\embeddings\3375_QD-BYT_679213.jsonl
```

Default batch size: 100 (`--batch-size 50` to change it). No model is loaded and no
embedding is generated again. Each file is validated for nonempty IDs/text, unique
IDs, finite nonzero 1,024-dimensional vectors and the pinned BGE-M3 model/revision
before that file's first database write. All embedding and source metadata is retained.

Records use `_id = chunk_id` and batched replacement upserts. Rerunning the same
files updates existing records without creating duplicates. Network failures can
leave earlier batches/files committed; fix the issue and rerun. Upload stops on the
first error with a nonzero exit code, and database errors never print the URI.

Default upload only upserts; it does not delete older IDs. Use `--sync` after
re-chunking. Sync requires a complete document manifest and every section part,
uploads all batches successfully, then deletes obsolete IDs for that `document_id`
only. A failed batch never authorizes cleanup. Other documents, collections and
the Vector Search index are untouched. Upload creates a normal compound index for section lookup. Uploads are not an atomic version switch: avoid querying
mid-migration and wait for Atlas indexing afterwards. The script does not create
or change the Vector Search index. Use the already configured
`rag_vector_index` on `embedding`, 1,024 dimensions, cosine; wait for Atlas indexing
before testing vector queries. Unit tests use a fake collection; no Atlas data is
written by the default test suite.


## Search uploaded RAG chunks

```powershell
.venv\Scripts\python.exe -m rag.search "Chẩn đoán và điều trị tăng huyết áp" --top-k 5
```

`rag/search.py` reads `.env`, `rag_chunks` and `rag_vector_index`; it writes no data.
It reuses the pinned BGE-M3 loader and reranks with multilingual
`BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.
The reranker weights download on first use (roughly 2.3 GB) and are cached locally.
CUDA uses float16 and reranking batch size 1. Use `--device cpu` if GPU memory is
insufficient. An application should keep both models and MongoClient loaded, then
call `search(query, collection, model=model, reranker=reranker)` across requests.
CLI launches still pay model startup costs.

Search retrieves at least 20 candidates (four times `--top-k` when larger). Titles
remain in vector search; they are never deleted from the index. A title-only match
is expanded only if the matching local Markdown checksum and actual heading levels
prove a subtree. Child lookup is constrained by document ID, source checksum and
source line boundaries; it is capped at the candidate count. Without a verified
source outline, the title is not blindly attached to following chunks. It is not
returned alone as evidence. Current flat `##` sources do not provide parent-child
relations merely because headings are adjacent, uppercase or numbered.

The cross-encoder ranks question/passage pairs using actual content and known
heading context. Exact duplicate chunks and identical overlapping source passages
are removed; distinct source sections are retained. Results include `rerank_score`,
original vector `score`, text, source/pages and heading metadata. Expanded children
have `score: null` and `via_heading` instead of an invented vector score.

Inputs are checked against embedding/reranker context limits; evidence is not
silently truncated. No uncalibrated relevance threshold is applied. Scores rank
passages; they are not clinical confidence. An empty result is `[]`. Query errors
return a nonzero exit code. This is retrieval only, without Gemini or Agent 1
integration. Existing data needs section metadata; use the migration commands below.
Validate relevance on representative questions before treating retrieval as reliable.


### Retrieve complete sections after chunking changes

After reranking, results are grouped by document ID, source checksum and section ID.
All parts of a matching section are fetched in `part_index` order. A complete,
consistent set uses the canonical `section_text`, removing repeated context exactly.
The returned `parts` retain individual chunk IDs, pages and line bounds. The score
and chunk ID identify the matched seed; expanded text may contain the whole section.

`--max-context-tokens 4000` sets the shared context budget across results, measured
with reference tokenizer `cl100k_base`. If the complete section does not fit, return
a contiguous window of whole parts around the hit and set `section_complete: false`.
Missing parts or legacy documents without section metadata are also marked incomplete.
No individual sentence/table row is sliced. Partial windows retain original per-part
context rather than guessing how to remove overlapping text.

To update existing data, run in order (no manual deletion):

```powershell
.venv\Scripts\python.exe -m rag.chunk --overwrite
.venv\Scripts\python.exe -m rag.embed --overwrite --reuse-existing
.venv\Scripts\python.exe -m rag.upload --sync
.venv\Scripts\python.exe -m rag.search "Chẩn đoán và điều trị tăng huyết áp" --top-k 5
```

Local writes replace files only after success. Keep the old embedding files until
the metadata refresh succeeds. Upload sync does not delete files, databases or the
Vector Search index. It does not remove documents omitted from the selected local
files. A flat source heading structure still cannot prove missing disease/chapter
relationships; section expansion solves split sections, not missing source hierarchy.

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
