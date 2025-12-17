# severity_model.py

import re

# -------------------------
# WEIGHTS (tune if needed)
# -------------------------

INTENT_WEIGHTS = {
    "Fraud": 0.9,
    "Identity theft": 0.85,
    "Account management": 0.35,
    "Loan servicing": 0.50,
    "Billing dispute": 0.60,
    "Payment problem": 0.55,
    "Data privacy": 0.75,
    "Card not working": 0.30,
    "Other": 0.20
}

SENTIMENT_WEIGHTS = {
    "negative": 0.8,
    "neutral": 0.4,
    "positive": 0.2,
}

# High-risk keywords
CRITICAL_KEYWORDS = [
    r"fraud", r"scam", r"stolen", r"illegal",
    r"lawsuit", r"court", r"escalate",
    r"chargeback", r"unauthorized", r"dispute"
]

def keyword_weight(text: str) -> float:
    text = text.lower()
    for kw in CRITICAL_KEYWORDS:
        if re.search(kw, text):
            return 0.9
    return 0.3


def severity_score(intent: str, sentiment: str, text: str) -> float:

    intent_w = INTENT_WEIGHTS.get(intent, 0.3)
    sentiment_w = SENTIMENT_WEIGHTS.get(sentiment, 0.4)
    keyword_w = keyword_weight(text)

    score = (0.5 * intent_w) + (0.3 * sentiment_w) + (0.2 * keyword_w)
    return round(score, 3)


# → 0.92
