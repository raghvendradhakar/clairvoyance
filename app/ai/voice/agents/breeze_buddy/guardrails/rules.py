"""Regex checks run before the judge. A match blocks with no judge call."""

import re
from typing import Optional

# reason codes (shared with the judge's decision tool)
PROMPT_INJECTION = "prompt_injection"
SECRET = "secret"
PAYMENT_CARD = "payment_card"

# DOTALL: a newline between the words must not get past the pattern.
_INJECTION = [
    re.compile(p, re.IGNORECASE | re.DOTALL)
    for p in (
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|"
        r"earlier|system|original)\b.{0,20}\b(instructions?|prompts?|rules)\b",
        r"\b(reveal|show|print|repeat|dump)\b.{0,40}\b(system|hidden|internal)\s+"
        r"(prompt|instructions?|message)\b",
        r"\byou are now\b.{0,20}\b(dan|jailbroken|unrestricted)\b",
        r"\b(developer|jailbreak|god)\s+mode\b",
        r"<\|?(system|im_start)\|?>|\[(system|developer)\]",
    )
]

# The lookbehinds skip URL slugs (/products/sk-ii-...) and keep the scan
# linear on long input.
_SECRETS = [
    re.compile(p)
    for p in (
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"(?<![/\w-])sk-(?:proj-)?[A-Za-z0-9_-]{20,}",
        r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{20,}",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bAIza[0-9A-Za-z_-]{35}\b",
        r"\bgh[pousr]_[A-Za-z0-9]{30,}",
        r"(?<![\w-])xox[abprs]-[A-Za-z0-9-]{20,}",
        r"(?<![\w-])eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    )
]

# 13-19 digits, plain or in groups of 4; checked with network prefix + Luhn.
_CARD = re.compile(r"(?<!\d)(?:\d{13,19}|\d{4}([ -])\d{4}\1\d{4}\1\d{1,7})(?!\d)")
_NETWORK_PREFIX = re.compile(r"^(4|5[1-5]|2[2-7]|3[47]|60|65|508)")
# RuPay 81/82 must be 16 digits; Shopify product ids share the prefix.
_RUPAY_16 = re.compile(r"^8[12]\d{14}$")


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def has_card_number(text: str) -> bool:
    for match in _CARD.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if (
            13 <= len(digits) <= 19
            and len(set(digits)) > 1
            and (_NETWORK_PREFIX.match(digits) or _RUPAY_16.match(digits))
            and _luhn(digits)
        ):
            return True
    return False


def _first_match(text: str, patterns) -> bool:
    return any(p.search(text) for p in patterns)


def has_secret(text: str) -> bool:
    return _first_match(text, _SECRETS)


def check_input(text: str) -> Optional[str]:
    """Reason code when the shopper's message must be blocked, else None."""
    if _first_match(text, _INJECTION):
        return PROMPT_INJECTION
    if _first_match(text, _SECRETS):
        return SECRET
    return None


def check_output(text: str) -> Optional[str]:
    """Reason code when assistant text must be blocked, else None."""
    if _first_match(text, _SECRETS):
        return SECRET
    if has_card_number(text):
        return PAYMENT_CARD
    return None
