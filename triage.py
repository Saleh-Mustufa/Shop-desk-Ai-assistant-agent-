"""Deterministic, zero-cost triage for the FR-3 fast path.

:func:`classify` decides — in plain Python, with NO model call and NO Agents
SDK import — whether a customer turn is a plain price/stock question about
exactly one catalogue product. Fast-path turns are served by the FastPath
agent (``agents_desk.make_fastpath_agent``, ``tool_use_behavior=
"stop_on_first_tool"``); every other turn goes to the Desk agent's ordinary
tool loop.

A turn takes the fast path only when ALL of these hold:

(a) it is short: at most :data:`MAX_CHARS` characters (stripped) and at most
    :data:`MAX_SENTENCES` sentences;
(b) it contains no order/intent word (:data:`ORDER_WORDS`, case-insensitive);
(c) it carries price/stock intent (:data:`PRICE_STOCK_PATTERN`,
    case-insensitive) — OR it names a valid product code directly
    (:data:`SKU_PATTERN` validated against ``catalogue.get_product``), because
    a bare product code is itself a price/stock ask;
(d) it resolves to EXACTLY ONE catalogue product: either a valid SKU from the
    text, or a fuzzy ``catalogue.product_by_name`` match on the text with
    intent words and question filler words stripped (with simple
    singularization, so plural forms like "TVs" still match). Ambiguity — two
    different products matched, e.g. "kettle and blender" — or no match at all
    routes to the Desk instead, with a reason that says which.

The product matching here only ever extends the matching in
``catalogue.product_by_name`` (filler stripping + singularization in this
module); ``catalogue.py`` itself is never changed by the fast path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import catalogue

__all__ = [
    "FastPathDecision",
    "MAX_CHARS",
    "MAX_SENTENCES",
    "ORDER_WORDS",
    "SKU_PATTERN",
    "classify",
]

MAX_CHARS = 160
MAX_SENTENCES = 3

# (b) Order/intent words: any of these means the turn is more than a plain
# price/stock lookup (buying, negotiating, logistics, after-sales) — Desk it is.
ORDER_WORDS: frozenset[str] = frozenset(
    {
        "order",
        "buy",
        "purchase",
        "confirm",
        "checkout",
        "deliver",
        "delivery",
        "ship",
        "discount",
        "quote",
        "negotiate",
        "cheaper",
        "warranty",
        "return",
        "refund",
    }
)

# (c) Price/stock intent: multi-word phrases checked on the joined text, plus
# single words checked per token (canonical list per FR-3, plus the plural
# forms "prices"/"costs").
PRICE_STOCK_PHRASES: tuple[str, ...] = ("how much", "much is", "in stock", "selling for")
PRICE_STOCK_WORDS: frozenset[str] = frozenset(
    {"price", "prices", "cost", "costs", "stock", "available", "availability", "selling"}
)

# (d) Direct product-code pattern, validated against the catalogue before use.
SKU_PATTERN = re.compile(r"\b[A-Z]{2,4}-\d{2,3}\b")

_PRICE_STOCK_RE = re.compile(
    r"\b(?:" + "|".join(PRICE_STOCK_PHRASES) + r"|" + "|".join(sorted(PRICE_STOCK_WORDS)) + r")\b"
)

_QUESTION_FILLERS: frozenset[str] = frozenset(
    """
    a all am an and any are as at be been being both but by can could did do does
    doing for from had has have having he her hers him his how i if in into is it
    its just know let many me might much must my no not of on one only or our ours
    out please really say show so some still such tell than that the their them
    then there these they this those to today too under until up was we were what
    when where which who whom whose will with would yes yet you your yours
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


@dataclass
class FastPathDecision:
    """The triage verdict for one customer turn (FR-3 routing input)."""

    is_fast_path: bool
    sku: str | None
    product_name: str | None
    reason: str


def _tokens(text: str) -> list[str]:
    """Lower-case word tokens; possessive "'s" is dropped ("what's" -> "what")."""
    return [token.split("'")[0] for token in _TOKEN_RE.findall(text.lower())]


def _sentence_count(text: str) -> int:
    """Count non-empty segments after splitting on '.', '!', '?'."""
    return len([segment for segment in _SENTENCE_SPLIT_RE.split(text) if segment.strip()])


def _singularize(word: str) -> str:
    """Very small plural stripper so "TVs"/"kettles" match catalogue names.

    Deliberately conservative (catalogue.py is untouched); wrong guesses simply
    fail to match, which routes the turn to the Desk — never invents a product.
    """
    if len(word) < 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _order_words_in(tokens: list[str]) -> list[str]:
    """(b) Order/intent words present (plural forms folded to the singular)."""
    found = []
    for token in tokens:
        if token in ORDER_WORDS or _singularize(token) in ORDER_WORDS:
            if token not in found:
                found.append(token)
    return found


def _has_price_stock_intent(text: str) -> bool:
    """(c) True when a price/stock phrase or word occurs in the text."""
    return _PRICE_STOCK_RE.search(text.lower()) is not None


def _meaningful_tokens(tokens: list[str]) -> list[str]:
    """Tokens worth product-matching: not stopwords, not bare numbers.

    Very short tokens (<3 chars, e.g. digits from "1.7L") are skipped so
    catalogue numerals like "30L"/"43-inch" cannot mass-match; plural tokens
    keep their original length check and are singularized when matched.
    """
    meaningful = []
    for token in tokens:
        if token in ORDER_WORDS or token in PRICE_STOCK_WORDS:
            continue  # intent words are stripped before matching (FR-3 d)
        if token in _QUESTION_FILLERS or token.isdigit() or len(token) < 3:
            continue
        meaningful.append(token)
    return meaningful


def _matched_products(text: str) -> list[dict]:
    """All distinct catalogue products the non-stopword text fuzzy-matches.

    Matching candidates: the stripped remainder as one phrase, then each
    remaining token singularized. Uses catalogue.product_by_name only; the
    result may be empty (no match) or hold several entries (ambiguous).
    """
    remainder = _meaningful_tokens(_tokens(text))
    if not remainder:
        return []
    candidates: list[str] = [
        " ".join(_singularize(token) for token in remainder),
        *(_singularize(token) for token in remainder),
    ]
    products_by_sku: dict[str, dict] = {}
    for candidate in candidates:
        if not candidate.strip():
            continue
        product = catalogue.product_by_name(candidate)
        if product is not None:
            products_by_sku[str(product.get("sku", ""))] = product
    return list(products_by_sku.values())


def _fast(sku: str | None, product_name: str | None, reason: str) -> FastPathDecision:
    return FastPathDecision(True, sku, product_name, reason)


def _desk(reason: str) -> FastPathDecision:
    return FastPathDecision(False, None, None, reason)


def classify(text: str) -> FastPathDecision:
    """Route one customer turn: fast path, or the Desk's ordinary loop.

    Zero cost: pure Python over the text and the catalogue, no model call
    (FR-3). The returned reason always explains the verdict.
    """
    stripped = (text or "").strip()
    if not stripped:
        return _desk("not fast path: the message is empty")

    # (a) Length and sentence budget.
    if len(stripped) > MAX_CHARS or _sentence_count(stripped) > MAX_SENTENCES:
        return _desk(
            f"not fast path: {len(stripped)} chars / {_sentence_count(stripped)} sentences "
            f"exceeds the fast-path budget ({MAX_CHARS} chars / {MAX_SENTENCES} sentences)"
        )

    tokens = _tokens(stripped)

    # (b) No order/intent words.
    order_hits = _order_words_in(tokens)
    if order_hits:
        return _desk(
            "not fast path: contains order/intent word(s) " + ", ".join(repr(w) for w in order_hits)
        )

    # (d-alt) Direct SKU route: a bare valid product code is itself a price/stock
    # ask, so it does not need (c)'s wording. Multiple distinct valid codes are
    # ambiguous; codes not in the catalogue fall through to the name route.
    sku_matches = SKU_PATTERN.findall(stripped)
    if sku_matches:
        skus_by_name: dict[str, str] = {}
        for code in dict.fromkeys(sku_matches):
            product = catalogue.get_product(code)
            if product is not None:
                skus_by_name[str(product.get("sku", ""))] = str(product.get("name", ""))
        if len(skus_by_name) > 1:
            return _desk(
                "not fast path: names multiple catalogue products ("
                + ", ".join(sorted(skus_by_name)) + ")"
            )
        if len(skus_by_name) == 1:
            sku, name = next(iter(skus_by_name.items()))
            return _fast(sku, name, f"fast path: product code {sku} named directly")

    # (c) Price/stock intent is required for the name route.
    if not _has_price_stock_intent(stripped):
        return _desk("not fast path: no price/stock intent detected")

    # (d) Exactly one catalogue product via fuzzy name matching.
    matches = _matched_products(stripped)
    if len(matches) > 1:
        names = ", ".join(f"{p.get('name', '?')} ({p.get('sku', '?')})" for p in matches)
        return _desk(f"not fast path: matches multiple catalogue products: {names}")
    if not matches:
        return _desk("not fast path: no catalogue product matches the question")
    product = matches[0]
    sku = str(product.get("sku", ""))
    name = str(product.get("name", ""))
    return _fast(sku, name, f"fast path: price/stock question about exactly one product: {name} ({sku})")
