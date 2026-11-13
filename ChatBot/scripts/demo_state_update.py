"""Two synthetic turns: offline fixtures or live Gemini extraction and questions."""
import sys
import argparse
from medical_chatbot.clinical.state_merger import update_clinical_state
from medical_chatbot.schemas.clinical_state import ClinicalState
from medical_chatbot.schemas.conversation import Conversation, Message
from medical_chatbot.schemas.outputs import InputProcessorOutput
from medical_chatbot.requirements.engine import evaluate_requirements, mark_question_sent, make_question_message
from medical_chatbot.agents.intake.question_generator import generate_question, QuestionGenerationError
from medical_chatbot.agents.intake.input_processor import process_input, InputProcessingError
from datetime import datetime


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Synthetic conversation with optional live Gemini extraction/questions")
    parser.add_argument("--live", action="store_true", help="Send synthetic messages to Gemini for extraction and question wording")
    args = parser.parse_args()
    print("MODE:", "LIVE Gemini extraction and questions / synthetic messages" if args.live else "OFFLINE fixtures / no LLM call")
    state = ClinicalState()
    message = Message(message_id="demo1", role="USER", content="headache for two days",
                      timestamp="2026-10-30T10:00:00+07:00")
    extracted = InputProcessorOutput.model_validate({
        "case_result": {"resolution": "NEW_CASE", "new_problem": "headache"},
        "case_facts": [{"field": "present_illness.duration", "key": "duration",
                        "value": "two days", "status": "PRESENT",
                        "source": {"message_id": "demo1", "source_text": "two days"}}],
    })
    if args.live:
        extracted = process_input(Conversation(conversation_id="demo", patient_id="synthetic", clinical_state=state), message)
    print("BEFORE:", state.model_dump_json(indent=2))
    result = update_clinical_state(state, message, extracted)
    print("AFTER:", result.model_dump_json(indent=2))
    intake = evaluate_requirements(result.state, facts=result.facts,
                                  case_uncertain=extracted.case_result.resolution == "UNCERTAIN")
    if args.live:
        question_text = generate_question(intake, history=[message])
        print("TURN 1:", intake.status, question_text)
        if question_text is None:
            return
        question_message = make_question_message(intake, question_text, message_id="question1",
            timestamp=datetime.fromisoformat("2026-10-30T10:00:30+07:00"))
        intake = mark_question_sent(intake, question_text)
        if intake.state.current_case_id is None:
            return
        conversation = Conversation(conversation_id="demo", patient_id="synthetic",
            clinical_state=intake.state, messages=[message, question_message])
    else:
        print("TURN 1:", intake.status, "Selected:", [r.key for r in intake.selected])
    case_id = intake.state.current_case_id
    second_message = Message(message_id="demo2", role="USER",
        content="started yesterday and is unchanged", timestamp="2026-10-30T10:01:00+07:00")
    second_extracted = InputProcessorOutput.model_validate({
        "case_result": {"resolution": "OLD_CASE", "target_case_id": case_id},
        "case_facts": [
            {"field": "present_illness.onset", "key": "onset", "value": "yesterday", "status": "PRESENT",
             "source": {"message_id": "demo2", "source_text": "started yesterday"}},
            {"field": "present_illness.course", "key": "course", "value": "unchanged", "status": "PRESENT",
             "source": {"message_id": "demo2", "source_text": "unchanged"}},
        ],
    })
    if args.live:
        second_extracted = process_input(conversation, second_message)
    second = update_clinical_state(intake.state, second_message, second_extracted,
                                   facts=result.facts, conflicts=result.conflicts)
    final = evaluate_requirements(second.state, facts=second.facts, conflicts=second.conflicts,
                                 case_uncertain=second_extracted.case_result.resolution == "UNCERTAIN")
    print("TURN 2:", final.status, final.question)
    print("Missing:", [r.key for r in final.state.cases[0].missing_information])


if __name__ == "__main__":
    try:
        main()
    except (QuestionGenerationError, InputProcessingError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
