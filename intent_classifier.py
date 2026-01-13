"""
Classifies intents into exactly one of three classes:
- "greeting"
- "product_inquiry"
- "high_intent_lead"
"""
import re
from google import genai
from dotenv import load_dotenv
load_dotenv()

client = genai.Client()

def contain_word(text: str, word: str) -> bool:
    """Checks if a word is present in the text as a whole word."""
    return re.search(rf'\b{re.escape(word)}\b', text) is not None

def classify_with_gemini(user_message: str) -> str:
    """
    Fallback function.
    Uses Gemini model to classify user message in case of ambiguity into one of three intents.
    :param user_message: Message to be classified
    :type user_message: str
    :return: Classified intent
    :rtype: str
    """

    prompt = f"""
    You are an intent classification model for a SaaS support assistant. The user messages can be of three types:
    1. greeting - User is saying hello or greeting.
    2. product_inquiry - User is asking about product features, pricing, or plans.
    3. high_intent_lead - User is expressing strong interest in signing up, purchasing, or subscribing. This might be as simple as asking to "get started" or "sign up". Sharing personal information like name or email also indicates high intent.
    Classify the user's intent into exactly one of these and respond with only that one word label.
    - greeting
    - product_inquiry
    - high_intent_lead

    User message: "{user_message}"
    """
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text.strip().lower() if response.text else "unknown"

def classify_intent(user_message: str) -> str:
    """
    Docstring for classify_intent
    
    :param user_message: Message to be classified
    :type user_message: str
    :return: Classified intent
    :rtype: str

    Uses keyword matching to classify the user message into one of three intents:
    - "greeting": If the message contains greeting keywords.
    - "high_intent_lead": If the message contains high intent lead keywords.
    - "product_inquiry": If the message contains product/pricing inquiry keywords.
    If none of the keywords matches, fallbacks to gemini-based detection.
    """
    text = user_message.lower().strip()
    greeting_keywords = ["hi", "hello", "hey", "good morning", "good evening"]
    high_intent_keywords = [
        "sign up",
        "subscribe",
        "buy",
        "purchase",
        "get started",
        "i want the pro plan",
        "i want to try",
        "ready to",
        "create account",
        "register",
        "enroll",
        "my name is",
        "youtube",
        "linkedin"
        "instagram",
        "facebook",
        "twitter",
        "email"
    ]
    inquiry_keywords = [
        "price",
        "pricing",
        "plan",
        "cost",
        "features",
        "what do you offer",
        "resolution",
        "limits",
        "refund",
        "support",
        "basic",
        "pro"
    ]

    # high-intent  >  product inquiry  >  greeting (priority order)
    if any(contain_word(text, word) for word in high_intent_keywords):
        return "high_intent_lead"
    if any(contain_word(text, word) for word in inquiry_keywords):
        return "product_inquiry"
    if any(contain_word(text, word) for word in greeting_keywords):
        return "greeting"
    return classify_with_gemini(user_message)

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
