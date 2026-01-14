"""
UI for Inflx Agent
Uses streamlit for the frontend
"""

import streamlit as st
from agent.agent import app, AgentState

st.set_page_config(page_title="Inflx: AutoStream Support", page_icon="🤖", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center;'>🤖 Smart Support Assistant</h1>
    <p style='text-align: center;'>
    I can help, answer questions, and capture leads when you're interested.
    </p>
    """,
    unsafe_allow_html=True
)

# Session State Initialization
if "state" not in st.session_state:
    st.session_state.state = AgentState()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "ui-thread-1"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat History Display
for role, content in st.session_state.messages:
    if role == "user":
        st.chat_message("user").markdown(content)
    else:
        st.chat_message("assistant").markdown(content)

# Chat Input
user_input = st.chat_input("Type your message…")

if user_input:
    # display user text
    st.session_state.messages.append(("user", user_input))
    st.chat_message("user").markdown(user_input)

    # update LangGraph state
    st.session_state.state.user_message = user_input

    # invoke graph
    result = app.invoke(
        st.session_state.state,
        config={"configurable": {"thread_id": st.session_state.thread_id}},
    )

    reply = result["reply"]

    # show assistant text
    st.session_state.messages.append(("assistant", reply))
    st.chat_message("assistant").markdown(reply)

# ✅ celebration if lead just captured
if getattr(st.session_state.state.conversation, "lead_just_captured", False):
    st.snow()
    # clear the flag so it doesn't snow forever
    st.session_state.state.conversation.lead_just_captured = False
