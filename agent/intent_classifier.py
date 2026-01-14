"""
Intent classifier for AutoStream SaaS support assistant.

Classifies messages into exactly one of three intents:
- "greeting"
- "product_inquiry"
- "high_intent_lead"

Uses:
1. Local keyword/TF-IDF classifier (fast, cheap)
2. Post-processing rules for high-intent overrides
3. LLM fallback (Gemini/Gemma) for low-confidence or ambiguous messages
"""
import string
from models.intent import classify_intent_local
from agent.state_manager import MultiLLM

THRESHOLD = 0.40
llm = MultiLLM()

def clean_text(text: str):
    """Lowers the text, removes extra whitespaces and punctuations"""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def classify_with_gemini(user_message: str) -> str:
    """
    Fallback function.
    Uses Gemini model to classify user message in case of ambiguity into one of three intents.
    """
    prompt = f"""
    You are an intent classification model for a SaaS support assistant.
    Classify the user's intent into exactly one of these:
    - greeting
    - product_inquiry (general questions about product features, pricing, plans, etc.)
    - high_intent_lead (showing interest in signing up, purchasing, or requesting account-related actions)
    Respond ONLY with the one-word label.
    User message: "{user_message}"
    """
    # response = client.models.generate_content(
    #     model="gemini-2.5-flash",
    #     contents=prompt
    # )
    response = llm.invoke(prompt)
    return response.text.strip().lower() if response.text else "unknown"

def classify_intent(user_message: str) -> str:
    """
    Docstring for classify_intent
    
    :param user_message: Message to be classified
    :type user_message: str
    :return: Classified intent
    :rtype: str

    Uses keyword matching to classify the user message into one of three intents:
    - "greeting"; "high_intent_lead"; "product_inquiry"
    If none of the keywords matches, fallbacks to gemini-based detection.
    """
    text = clean_text(user_message)
    label, confidence = classify_intent_local(text)
    if confidence < THRESHOLD:
        label = classify_with_gemini(text)
    print(label)
    return label

if __name__ == "__main__":
    test_messages = [
        "Hello!", #greeting
        "What are your pricing plans?", #product_inquiry
        "I want to sign up for the pro plan.", #high_intent_lead
        "What are your features", #product_inquiry
        "Hello, I would like to know more about your product.", #product_inquiry
        "Good morning, how can I get started?", #high_intent_lead
        "Hola Amigo!", #ambiguous, fallback to gemini
        "I am Jaspreet, i think it might be good for my linkedin"
    ]

    for msg in test_messages:
        intent = classify_intent(msg)
        print(f"Message: '{msg}' => Classified Intent: '{intent}'")
