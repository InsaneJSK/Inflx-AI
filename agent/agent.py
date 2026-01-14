from typing import Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import json, re
from agent.intent_classifier import classify_intent
from data.rag_retriever import retrieve_from_kb
from agent.state_manager import ConversationState
from agent.tools import mock_lead_capture
from agent.state_manager import MultiLLM

# client = genai.Client()
llm = MultiLLM()

# LangGraph State Schema
class AgentState(BaseModel):
    user_message: str = ""
    conversation: ConversationState = Field(default_factory=ConversationState)
    rag_result: Optional[str] = None
    reply: Optional[str] = None
    class Config:
        arbitrary_types_allowed = True

# NODES

def intent_node(state: AgentState):
    conv = state.conversation
    user_message = state.user_message

    label = classify_intent(user_message)
    conv.last_intent = label
    conv.add_turn("User", user_message)

    return state

def rag_node(state: AgentState):
    answer = retrieve_from_kb(state.user_message)
    if not answer or not answer.strip():
        state.rag_result = None
        state.conversation.rag_used = False
    else:
        state.rag_result = answer
        state.conversation.rag_used = True

    return state

# LLM lead extraction node

def lead_collection_node(state: AgentState):
    conv = state.conversation
    user_msg = state.user_message
    history_text = "\n".join(f"{turn.role}: {turn.content}" for turn in conv.history)

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
    - Values may exist in the history as well

    Respond ONLY with valid JSON. Do not add any explanation. JSON should follow the form:
    {{
    "name": <string or null>,
    "email": <string or null>,
    "platform": <string or null>
    }}
    History of messages:
    {history_text}
    User message:
    \"\"\"{user_msg}\"\"\"

    Conversation so far (may contain previous info):
    Name: {conv.name}
    Email: {conv.email}
    Platform: {conv.platform}
"""

    # response = client.models.generate_content(
    #     model="gemini-2.5-flash",
    #     contents=extraction_prompt
    # )
    response = llm.invoke(extraction_prompt)
    raw = response.text.strip()
    raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()

    try:
        extracted = json.loads(raw)
    except Exception:
        extracted = {"name": None, "email": None, "platform": None}

    # ---- update state only if missing ----
    def keep(existing, new):
        if new  in [None, "null", "None", ""]:
            return existing
        return new if not existing else existing

    conv.name = keep(conv.name, extracted.get("name"))
    conv.email = keep(conv.email, extracted.get("email"))
    conv.platform = keep(conv.platform, extracted.get("platform"))
    print(f"Name: {conv.name}, Email: {conv.email}, Platform: {conv.platform}")

    # ---- check remaining ----
    missing = conv.missing_lead_fields()

    if missing:
        # ask only for missing items
        ask = "Great! To complete your signup, I still need your "
        ask += ", ".join(missing)
        ask += "."
        state.reply = ask
        conv.add_turn("Assistant", ask)
        return state

    # ---- all details present → capture lead ----
    out = mock_lead_capture(conv.name, conv.email, conv.platform)
    conv.reset_lead_capture()
    conv.last_intent = "post_lead"
    conv.lead_just_captured = True
    state.reply = (
        f"🎉 Lead captured successfully!\n\n{out}\n\nOur team will reach out soon."
    )
    conv.add_turn("Assistant", state.reply)
    return state



def llm_response_node(state: AgentState):
    conv = state.conversation
    history_text = "\n".join(f"{turn.role}: {turn.content}" for turn in conv.history)
    if getattr(conv, "rag_used", False):
        rag_section = f"Use the following official knowledge base info and keep your answer grounded to it:\n{state.rag_result}\n"
    else:
        rag_section = "No reliable info found in the knowledge base. Do NOT invent product details."
    post_lead_note = ""
    if getattr(conv, "lead_just_captured", False):
        post_lead_note = "NOTE: The user has successfully signed up. Do NOT try to sell again, focus on support and answering."
    prompt = f"""
    You are AutoStream SaaS support assistant.

    Conversation history:
    {history_text}

    User said: "{state.user_message}"
    Detected intent: {conv.last_intent}

    {rag_section}

    STRICT RULES:
    - If no info is available, say you don't have that information
    - DO NOT MAKE UP PRICES, FEATURES, OR CLAIMS
    - If user asks something outside the context, say you will connect them to sales
    Write a friendly, to the point and concise reply.
    If you find the user mildly interested in our product, nudge him ever so slightly to try our services but don't be a pushy sales agent.

    {post_lead_note}
    """

    # response = client.models.generate_content(
    #     model="gemini-2.5-flash",
    #     contents=prompt
    # )
    response = llm.invoke(prompt)
    text = response.text.strip()

    conv.add_turn("Assistant", text)
    state.reply = text

    return state

# =========================
# ---- ROUTER --------------
# =========================

def router(state: AgentState):
    conv = state.conversation

    if getattr(conv, "collecting_lead", False):
        return "lead"
    
    intent = conv.last_intent

    if intent == "greeting":
        return "llm"

    if intent == "product_inquiry":
        return "rag"

    if intent == "high_intent_lead":
        conv.collecting_lead = True
        return "lead"
    if intent == "post_lead":
        return "llm"
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
