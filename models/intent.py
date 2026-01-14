"""
Trains a classifier for intent detecting using tfidf and logistic regression
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from models.intent_data import training_data

texts = [x[0] for x in training_data]
labels = [x[1] for x in training_data]

intent_classifier = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
    ('clf', LogisticRegression())
])

intent_classifier.fit(texts, labels)

def classify_intent_local(message):
    """Uses tfidf vectors for classification, done locally"""
    label = intent_classifier.predict([message])[0]
    confidence = intent_classifier.predict_proba([message]).max()
    return label, float(confidence)

if __name__ == "__main__":
    test_messages = [
        "Hello!",
        "What are your pricing plans?",
        "I want to sign up for the pro plan.",
        "What are your features",
        "Hello, I would like to know more about your product.",
        "Good morning, how can I get started?",
        "Hola Amigo!",
        "I am Jaspreet, i think it might be good for my linkedin"
    ]

    for msg in test_messages:
        intent, conf = classify_intent_local(msg)
        print(f"Message: '{msg}' => Classified Intent: '{intent}' with confidence {conf:.2f}")
