"""
RAG-based knowledge retriever for Inflx-AI's AutoStream product.
Features:
- NLTK lemmatization of user query
- Multi-attribute detection
- Explicit plan detection (Basic/Pro)
- retrieves from knowledge_base.json
"""

import json
import string
import nltk
from nltk.stem import WordNetLemmatizer

# NLTK setup
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()
KB_PATH = "knowledge_base.json"


def load_kb():
    with open(KB_PATH, "r") as f:
        return json.load(f)


def clean_and_lemmatize(text: str):
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [lemmatizer.lemmatize(tok) for tok in text.split()]


def format_plan(plan_name: str, kb):
    plan_data = kb["AutoStream Pricing & Features"][plan_name]
    lines = [f"{k}: {v}" for k, v in plan_data.items()]
    return f"{plan_name} details:\n" + "\n".join(lines)


def retrieve_from_kb(user_query: str) -> str:
    kb = load_kb()
    lemmas = clean_and_lemmatize(user_query)

    # ---------------- detect explicit plan ----------------
    plan = None
    if "basic" in lemmas:
        plan = "Basic Plan"
    elif "pro" in lemmas:
        plan = "Pro Plan"

    # ---------------- detect generic plan language ----------------
    generic_plan_terms = {"plan", "plans", "pricing", "subscription"}
    generic_plan_mentioned = any(term in lemmas for term in generic_plan_terms)

    # ---------------- attribute detection ----------------
    attribute_map = {
        "price": "Price",
        "cost": "Price",
        "limit": "Limits",
        "video": "Limits",
        "quality": "Quality",
        "resolution": "Quality",
        "feature": "Additional Features",
        "caption": "Additional Features",
        "refund": "Company Policies",
        "support": "Company Policies",
    }

    detected_attributes = {attribute_map[l] for l in lemmas if l in attribute_map}

    # ---------------- policy handling ----------------
    if "Company Policies" in detected_attributes:
        return "Company Policies:\n- " + "\n- ".join(kb["Company Policies"])

    # ---------------- explicit plan mentioned → ALWAYS full plan ----------------
    if plan:
        return format_plan(plan, kb)

    # ---------------- generic plan mentioned → return BOTH full plans ----------------
    if generic_plan_mentioned:
        basic = format_plan("Basic Plan", kb)
        pro = format_plan("Pro Plan", kb)
        return basic + "\n\n" + pro

    # ---------------- attributes mentioned but no specific plan ----------------
    if detected_attributes:
        # return BOTH plans with full info (no partial loss)
        basic = format_plan("Basic Plan", kb)
        pro = format_plan("Pro Plan", kb)
        return basic + "\n\n" + pro

    # ---------------- generic fallback ----------------
    return (
        "AutoStream offers Basic and Pro plans. "
        "Ask about price, limits, quality, features, refunds, or support for more details."
    )


if __name__ == "__main__":
    test_queries = [
        "What is the price of the Pro plan?",
        "Tell me about the limits of the Basic plan.",
        "What features does the Pro plan offer?",
        "How can I get a refund?",
        "What support options are available?",
        "Give me an overview of your plans.",
        "What is the cost?",
        "Tell me about video quality in Pro.",
        "I want info on refunds and support.",
        "Tell me about your plans",
        "What limits and price do plans have?"
    ]

    for query in test_queries:
        result = retrieve_from_kb(query)
        print(f"Query: '{query}'\nRetrieved Info:\n{result}\n{'-'*50}")
