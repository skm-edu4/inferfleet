from __future__ import annotations

import hashlib
import random

_OPENERS = (
    "Thanks for reaching out — happy to help with that.",
    "Good question. Here is the short version.",
    "I looked into this for you.",
    "Here is what I found.",
)

_FACTS = (
    "Your account settings page lists every option we just discussed, and changes apply immediately.",
    "If the issue started after a recent update, a full sign-out and sign-in usually clears the cached state.",
    "Our documentation covers this in detail, and the steps are identical for every billing tier.",
    "You can retry this safely — idempotency keys mean a duplicate request will never double-charge you.",
    "Most cases like this resolve within one business day once the request reaches the queue.",
    "The API returns a cursor with every page of results, so you can keep paging without duplicates.",
    "Rate limits reset on a fixed 60 second window, so waiting a moment is enough to recover.",
    "Support can escalate this to a human agent if the automated path does not fit your case.",
)

_CLOSERS = (
    "Let me know if you would like the step-by-step walkthrough.",
    "I hope that clears it up!",
    "Anything else you want me to dig into?",
    "Happy to go deeper on any of those points.",
)

FAIL_MARKER = "__FAIL__"
CRASH_MARKER = "__CRASH__"


def _rng(prompt: str) -> random.Random:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def plan(prompt: str, max_tokens: int) -> list[str]:
    """Deterministic mock completion, tokenized as whitespace words.

    Completion length is intentionally ragged: each prompt stops at a
    deterministic fraction (40–100%) of ``max_tokens``, mimicking variable
    end-of-sample times. Uniform lengths would let a whole batch retire on the
    same step, which hides the entire point of continuous batching — ragged
    finishes are what free slots mid-run so waiting jobs can be admitted.
    """
    rng = _rng(prompt)
    words = prompt.split()
    topic = " ".join(words[:6]) if words else "your question"
    lines = [
        f"{rng.choice(_OPENERS)} Regarding \"{topic}\":",
        rng.choice(_FACTS),
        rng.choice(_FACTS),
        rng.choice(_CLOSERS),
    ]
    tokens: list[str] = []
    for line in lines:
        tokens.extend(line.split())
    target = max(4, int(max_tokens * (0.4 + 0.6 * rng.random())))
    if len(tokens) > target:
        tokens = tokens[:target]
        tokens[-1] = tokens[-1] + " ..."
    return tokens
