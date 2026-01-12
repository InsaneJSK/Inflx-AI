"""
Conversation agent for AutoStream assistant.

Responsibilities:
- intent detection
- RAG retrieval
- lead data collection
- tool execution
"""

from state_manager import ConversationState
from rag_retriever import retrieve_from_kb
from tools import mock_lead_capture


def detect_intent(user_query: str) -> str:
    """
    Very lightweight rule-based intent detector.
    """

    q = user_query.lower()

    if any(x in q for x in ["price", "plan", "feature", "quality", "limit", "refund", "support"]):
        return "product_query"

    if any(x in q for x in ["sign up", "interested", "buy", "purchase", "subscribe", "demo", "contact"]):
        return "lead_generation"

    return "chitchat"


def update_lead_fields(state: ConversationState, user_query: str):
    """
    Try to update name/email/platform from user input.
    Extremely naive but fine for assignment.
    """

    text = user_query.strip()

    # email
    if "@" in text and "." in text and state.email is None:
        state.email = text
        return

    # platform
    if any(p in text.lower() for p in ["youtube", "instagram", "tiktok", "facebook"]) and state.platform is None:
        state.platform = text
        return

    # name (fallback: any short text without @)
    if state.name is None and "@" not in text:
        state.name = text


def agent_step(state: ConversationState, user_query: str) -> str:
    """
    Main loop of agent logic.
    """

    # add user message
    state.add_turn("user", user_query)

    # detect intent only if not already collecting lead
    if not state.collecting_lead:
        intent = detect_intent(user_query)
        state.last_intent = intent
    else:
        intent = "lead_generation"

    # -------- LEAD GENERATION MODE --------
    if intent == "lead_generation":

        state.collecting_lead = True
        update_lead_fields(state, user_query)

        missing = state.missing_lead_fields()

        # if done -> call tool
        if len(missing) == 0:
            result = mock_lead_capture(state.name, state.email, state.platform)
            state.reset_lead_capture()
            state.add_turn("assistant", "Thanks! Your details have been submitted successfully.")
            return (
                "🎉 Thanks! Your details are submitted.\n"
                "Our team will reach out shortly."
            )

        # otherwise ask next missing field
        next_field = missing[0]
        prompts = {
            "name": "Great! Can I have your full name?",
            "email": "Thanks. What is your email address?",
            "platform": "Which platform do you primarily create content for?"
        }

        reply = prompts[next_field]
        state.add_turn("assistant", reply)
        return reply

    # -------- PRODUCT QUERY (RAG) --------
    elif intent == "product_query":
        answer = retrieve_from_kb(user_query)
        state.add_turn("assistant", answer)
        return answer

    # -------- SMALL TALK FALLBACK --------
    else:
        reply = "Hi! I can help with plans, pricing, features, refunds, or help you sign up."
        state.add_turn("assistant", reply)
        return reply


if __name__ == "__main__":

    state = ConversationState()

    while True:
        user = input("You: ")
        if user.lower() in ["exit", "quit"]:
            break

        response = agent_step(state, user)
        print("Assistant:", response)
