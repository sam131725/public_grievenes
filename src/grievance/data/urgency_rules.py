"""Weak-label heuristics for urgency/sentiment when gold labels are missing.
Use only to bootstrap; document in the report and validate on a hand-labelled subset."""
CRITICAL = ["fire", "electrocut", "live wire", "sewage overflow", "no water for", "accident", "आग", "करंट", "दुर्घटना"]
HIGH = ["days", "since", "urgent", "children", "hospital", "कई दिन", "अत्यावश्यक"]
NEG = ["not", "no ", "worst", "pathetic", "angry", "नहीं", "खराब"]

def weak_urgency(text: str) -> str:
    t = text.lower()
    if any(k in t for k in CRITICAL): return "critical"
    if any(k in t for k in HIGH): return "high"
    return "medium" if len(t) > 80 else "low"

def weak_sentiment(text: str) -> str:
    t = text.lower()
    return "negative" if any(k in t for k in NEG) else "neutral"
