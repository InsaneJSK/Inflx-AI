from typing import Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from google import genai

from intent_classifier import classify_intent
from rag_retriever import retrieve_from_kb
from state_manager import ConversationState
from tools import mock_lead_capture

from dotenv import load_dotenv
load_dotenv()

client = genai.Client()


# =========================
# LangGraph State Schema
# =========================
class AgentState(BaseModel):
    user_message: str = ""
    conversation: ConversationState = Field(default_factory=ConversationState)
    rag_result: Optional[str] = None
    reply: Optional[str] = None


# =========================
# ---- NODES ---------------
# =========================

def intent_node(state: AgentState):
    conv = state.conversation
    user_message = state.user_message

    intent = classify_intent(user_message)
    conv.last_intent = intent
    conv.add_turn("User", user_message)

    return state



def rag_node(state: AgentState):
    answer = retrieve_from_kb(state.user_message)
    state.rag_result = answer
    return state



# =========================
# LLM lead extraction node
# =========================

def lead_collection_node(state: AgentState):
    conv = state.conversation
    user_msg = state.user_message

    conv.collecting_lead = True

    # ---- LLM extraction ----
    extraction_prompt = f"""
You extract structured lead details from free-form text.

Extract ONLY the following fields if explicitly mentioned:
- name
- email
- platform of interest (e.g., LinkedIn, YouTube, Instagram, WhatsApp, Website)

Rules:
- Do NOT invent missing fields
- If a field is missing, output null
- If multiple options exist, choose the clearest one
- Platform can be generic social media or "website"

Return JSON ONLY in the form:
{{
 "name": <string or null>,
 "email": <string or null>,
 "platform": <string or null>
}}

User message:
\"\"\"{user_msg}\"\"\"

Conversation so far (may contain previous info):
Name: {conv.name}
Email: {conv.email}
Platform: {conv.platform}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=extraction_prompt
    )

    import json

    try:
        extracted = json.loads(response.text)
    except Exception:
        extracted = {"name": None, "email": None, "platform": None}

    # ---- update state only if missing ----
    if conv.name is None and extracted.get("name"):
        conv.name = extracted["name"]

    if conv.email is None and extracted.get("email"):
        conv.email = extracted["email"]

    if conv.platform is None and extracted.get("platform"):
        conv.platform = extracted["platform"]

    # ---- check remaining ----
    missing = conv.missing_lead_fields()

    if missing:
        # ask only for missing items
        ask = "Great! To complete your signup, I still need your "
        ask += ", ".join(missing)
        ask += "."
        state.reply = ask
        return state

    # ---- all details present → capture lead ----
    out = mock_lead_capture(conv.name, conv.email, conv.platform)
    conv.reset_lead_capture()

    state.reply = (
        f"🎉 Lead captured successfully!\n\n{out}\n\nOur team will reach out soon."
    )

    return state



def llm_response_node(state: AgentState):
    conv = state.conversation

    prompt = f"""
You are AutoStream SaaS support assistant.

Conversation history:
{conv.history}

User said: "{state.user_message}"

Detected intent: {conv.last_intent}

If rag_result exists, use ONLY that information and don't hallucinate.
RAG result:
{state.rag_result}

Write a friendly reply.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    conv.add_turn("Assistant", text)
    state.reply = text

    return state



# =========================
# ---- ROUTER --------------
# =========================

def router(state: AgentState):
    intent = state.conversation.last_intent

    if intent == "greeting":
        return "llm"

    if intent == "product_inquiry":
        return "rag"

    if intent == "high_intent_lead":
        return "lead"

    return "llm"



# =========================
# ---- GRAPH BUILD ---------
# =========================

graph = StateGraph(AgentState)

graph.add_node("intent", intent_node)
graph.add_node("rag", rag_node)
graph.add_node("lead", lead_collection_node)
graph.add_node("llm", llm_response_node)

graph.set_entry_point("intent")

graph.add_conditional_edges(
    "intent",
    router,
    {
        "rag": "rag",
        "lead": "lead",
        "llm": "llm",
    },
)

graph.add_edge("rag", "llm")
graph.add_edge("lead", "llm")
graph.add_edge("llm", END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# =========================
# ---- DEMO ---------------
# =========================

if __name__ == "__main__":
    state = AgentState()
    thread_id = "demo-thread"

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        state.user_message = user_input

        result = app.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )

        print("Agent:", result["reply"])
