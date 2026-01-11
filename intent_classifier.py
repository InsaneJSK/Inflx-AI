"""
Classifies intents into exactly one of three classes:
- "greeting"
- "product_inquiry"
- "high_intent_lead"
"""
from google import genai
from dotenv import load_dotenv
load_dotenv()

client = genai.Client()

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
    Classify the user's intent into exactly one of these:
    - greeting
    - product_inquiry
    - high_intent_lead

    User message: "{user_message}"

    Respond with only one word label.
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
        "register"
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
    if any(word in text for word in high_intent_keywords):
        return "high_intent_lead"
    if any(word in text for word in inquiry_keywords):
        return "product_inquiry"
    if any(word in text for word in greeting_keywords):
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
        "Hola Amigo!" #ambiguous, fallback to gemini
    ]

    for msg in test_messages:
        intent = classify_intent(msg)
        print(f"Message: '{msg}' => Classified Intent: '{intent}'")
