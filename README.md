# 🤖 Inflx: Social-to-Lead Agentic Workflow
Deployed at: [Streamlit](https://inflx-ai.streamlit.app/) | Demo Video: [Demo](https://youtu.be/GFiyL7PWCDU)
## 1. Project Overview

Inflx is an AI-powered conversational agent designed to convert social media interactions into qualified business leads. Unlike simple chatbots, this agent can:

- Understand user intent (greeting, product inquiry, high-intent lead)  
- Provide accurate, RAG-powered answers from a local knowledge base  
- Detect high-intent users ready to sign up  
- Trigger backend actions like lead capture  

It demonstrates building a real-world GenAI agent capable of **reasoning, tool execution, and conversational memory**.

## 2. Problem Statement

The agent simulates customer interactions for a fictional SaaS product:

**AutoStream** – a platform providing automated video editing and streaming tools for content creators.  

The agent must:

1. Respond accurately to greetings and product questions  
2. Detect users ready to subscribe  
3. Collect user information (name, email, platform)  
4. Call a **mock API** for lead capture **only after all details are collected**  

## 3. Agent Capabilities
### 3.1 Intent Identification
- **Casual greeting**  
- **Product/pricing inquiry**  
- **High-intent lead**  

### 3.2 RAG-Powered Knowledge Retrieval
- Local knowledge base (JSON/Markdown) includes:  

**AutoStream Pricing & Features**  
| Plan  | Price | Videos/Month | Resolution | Extras |
|-------|-------|--------------|------------|--------|
| Basic | $29   | 10           | 720p       | –      |
| Pro   | $79   | Unlimited    | 4K         | AI Captions |

**Company Policies**  
- No refunds after 7 days  
- 24/7 support only on Pro Plan  

### 3.3 Tool Execution – Lead Capture
- Collects **Name, Email, Platform**  
- Calls the mock function **only after collecting all fields**  

```python
def mock_lead_capture(name, email, platform):
    print(f"Lead captured successfully: {name}, {email}, {platform}")
```

## 4. Tech Stack

| Component                       | Technology / Library               |
|---------------------------------|-----------------------------------|
| LLM                             | `gemini-2.5-flash` with `langchain-groq` (Llama 3.1 Instant) as fallback for rate limits |
| Conversational Agent            | `LangGraph`                       |
| Intent Classification            | Custom TfIdf based classifier          |
| Knowledge Retrieval (RAG)       | Custom module without chunking or embedding since the KB was very small                     |
| Lead Capture                     | LLM-based extraction + mock function |
| Frontend                        | `Streamlit`                        |
| Environment & Config            | `.env`, `python-dotenv`           |
| Data Handling                   | `Pydantic` for state management  |

---

## 5. Folder Structure

- Inflx-AI
    - agent
        - `__init__.py`: required for packaging
        - `agent.py`: main file handling entire agent architecture
        - `intent_classifier.py`: Module for classifying intent
        - `state_manager.py`: Definitions for various classes required for the agent
        - `tools.py`: Definitions for the tools to be called (mock_tool resides here)
    - data
        - `__init__.py`: required for packaging
        - `knowledge_base.json`: The provided information in json
        - `rag_retriever.py`: The module handling logic for RAG
    - models
        - `__init__.py`: required for packaging
        - `intent_data.py`: Data required for training
        - `intent.py`: Training of the TfIDf based classifier
    - `app.py`: Streamlit UI for Inflx-AI
    - `requirements.txt`
    - `.env.dist`

## 6. Run Locally
1. **Clone the repository**  
```bash
git clone https://github.com/yourusername/Inflx-AI.git
cd Inflx-AI
```
2. **Install Dependencies**
```bash
python -m venv venv
venv/Scripts/activate.bat #Or source venv/bin/activate for linux/MacOS
pip install -r requirements.txt
```
3. **Set-up `.env` file**

Copy the structure fro `.env.dist` and replace with the required keys

4. **Run the app**
```bash
streamlit run .\app.py
```

5. **Interact with the assistant**
- Type messages in the chat input
- Ask about the Pro Plan features or provide lead information.
- Celebrate lead capture with interactive snow animation.

## 🔹 Key Notes

- **Memory**: Stores up to 5-turn conversation history for context and lead extraction.

- **Lead Capture**: Only extracts fields explicitly mentioned by the user. If missing, the agent will ask politely for the remaining details.

- **LLM Efficiency**: Only calls the LLM when generating responses or performing structured extraction, minimizing API usage.

## Screenshots/Demo

[![Inflx-AI-Demo](https://img.youtube.com/vi/GFiyL7PWCDU/0.jpg)](https://www.youtube.com/watch?v=GFiyL7PWCDU)

### Required Questions

#### 1. Why you chose LangGraph

- Stateful multi-turn conversations: Tracks user context across the last 5 turns.
- Intent-driven routing: Directs chats to custom modules like intent classifier or rag, call tools like lead capture, or general LLM responses.
- easy RAG integration: Answers product and policy questions accurately from a local knowledge base.
- High-intent detection: Identifies users ready to sign up and triggers lead capture tools. LangGraph allows the flow to move around easily
- Safe tool execution: Ensures `mock_lead_capture()` runs only after collecting all required info, allowing it to come back to old state even when digressing to call tools.
- LLM-agnostic & modular: Works with multiple LLMs, keeping the workflow flexible and explainable.
- Real-world deployability: Mirrors a SaaS support workflow while remaining lightweight and maintainable.

#### 2. State Management in Inflx AI

- **AgentState & ConversationState**: Pydantic-based classes manage all conversation data.
- **Turn History**: Each user and assistant message stored as Turn objects; max 5 turns retained.
- **Intent & RAG Tracking**: Last detected intent and RAG usage tracked per conversation.
- **Lead Capture**: Fields for name, email, platform, with helper methods to check missing info and reset state.
- **State Updates**: Every user input and LLM response updates the state, keeping conversations coherent.
- **Pydantic Validation**: Ensures strict typing and prevents inconsistent state, making memory reliable.
- **MultiLLM Integration**: Handles automatic failover between Gemini and Groq LLMs for seamless responses.

3. **WhatsApp Deployment via Webhooks:**

- Use the WhatsApp Business API to receive and send messages.
- Configure a Webhook URL that points to a lightweight server (e.g., FastAPI or Flask).
- Incoming messages from WhatsApp trigger the webhook, which updates the AgentState and invokes the LangGraph agent.
- The agent’s reply is sent back through the WhatsApp API to the user.
- Maintain conversation context in memory or a database to ensure multi-turn dialogue continuity.
- Lead capture and RAG responses function seamlessly as they do in the Streamlit/local setup.
