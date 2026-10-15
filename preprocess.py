"""NLP preprocessing: cleaning, tokenising and stop-word removal (no downloads needed)."""
import re

STOPWORDS = set("""
a about above after again all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how
i if in into is it its itself just me more most my no nor not now of off on once only or other our out over own
same she should so some such than that the their them then there these they this those through to too under until
up very was we were what when where which while who whom why will with would you your said says say also
""".split())

URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punctuation/digits, drop stop words and 1-letter tokens."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


CLICKBAIT_PHRASES = [
    "shocking", "you won't believe", "doctors hate", "miracle", "secret", "they don't want you to know",
    "share before", "share this", "forward this", "deleted", "banned", "exposed", "100% proven",
    "cover-up", "cover up", "wake up", "urgent", "must watch", "viral", "hidden truth",
]


def red_flags(text: str) -> list:
    """Simple rule-based warning signs shown next to the ML prediction."""
    flags = []
    raw = str(text)
    letters = [c for c in raw if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.35:
        flags.append("Excessive ALL-CAPS text (often used to provoke emotion)")
    if raw.count("!") >= 3:
        flags.append("Repeated exclamation marks")
    low = raw.lower()
    hits = [p for p in CLICKBAIT_PHRASES if p in low]
    if hits:
        flags.append("Clickbait / pressure phrases: " + ", ".join(hits[:5]))
    if not re.search(r"according to|said|reported|official|ministry|study|data|spokesperson|statement", low):
        flags.append("No source, official or data is mentioned")
    if len(raw.split()) < 12:
        flags.append("Very short text - hard to verify; add more context")
    return flags
