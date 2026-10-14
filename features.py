"""The OLD way: turn a prompt (text) into 13 numbers for CatBoost. Only used for the comparison in section 5b."""
import re

# The order here is the column order of the training CSV. Never reorder without retraining.
FEATURE_NAMES = [
    "n_chars", "n_words", "n_lines", "n_sentences", "n_question_marks",
    "n_digits", "n_code_symbols", "n_code_fences", "n_code_keywords",
    "avg_word_len", "has_tests", "asks_for_short", "asks_for_long",
]

_CODE_KEYWORDS = re.compile(r"\b(def|class|import|return|function|assert|SELECT|for|while)\b")
_SHORT = re.compile(r"\b(brief|briefly|short|concise|one word|one sentence|yes or no|summari[sz]e)\b", re.I)
_LONG = re.compile(r"\b(explain|detailed|in detail|essay|step by step|write a|implement|story)\b", re.I)


def extract_features(prompt):
    text = prompt if isinstance(prompt, str) else ""
    words = text.split()
    return [
        len(text),
        len(words),
        text.count("\n") + 1,
        len(re.findall(r"[.!?]+", text)),
        text.count("?"),
        sum(c.isdigit() for c in text),
        sum(text.count(c) for c in "{}()[]=<>;_"),
        text.count("```"),
        len(_CODE_KEYWORDS.findall(text)),
        (sum(len(w) for w in words) / len(words)) if words else 0.0,
        int("assert" in text or "test" in text.lower()),
        int(bool(_SHORT.search(text))),
        int(bool(_LONG.search(text))),
    ]
