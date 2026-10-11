"""Prompt format shared by training, generation and evaluation."""

import re

ABSTAIN = "I can't answer this from the provided context."

SYSTEM_PROMPT = (
    "You answer questions using only the provided context. "
    "Do not add facts that are not stated in the context. "
    f'If the context does not contain the answer, reply exactly: "{ABSTAIN}" '
    "Keep answers short: one to three sentences."
)

# The base model never saw ABSTAIN, so the detector also accepts common
# paraphrases. Otherwise the baseline would look worse than it is.
_ABSTAIN_PATTERNS = [
    r"can(?:'|no)?t answer",
    r"cannot answer",
    r"unable to answer",
    r"(?:context|text|passage|document)s? (?:does not|doesn't|do not|don't) (?:contain|provide|mention|say|include|specify|state)",
    r"not (?:mentioned|stated|specified|provided|included) in the (?:context|text|passage|document)",
    r"no information (?:about|on|regarding)",
    r"there is no (?:information|mention)",
]
_ABSTAIN_RE = re.compile("|".join(_ABSTAIN_PATTERNS), re.IGNORECASE)


def user_message(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}"


def prompt_messages(context: str, question: str, system_in_user: bool = False) -> list[dict]:
    """Chat messages up to (not including) the assistant turn."""
    user = user_message(context, question)
    if system_in_user:
        return [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user}"}]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def is_abstention(text: str) -> bool:
    return bool(_ABSTAIN_RE.search(text.replace("’", "'")))
