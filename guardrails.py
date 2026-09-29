"""Catalogue output guardrail for Shop Desk (FR-6): a data check, not a vibe check.

Runs on the desk agent's FINISHED answer before it reaches the customer. Every
SKU and every currency amount in the answer must be justified by
``catalogue.json`` **as loaded this run** — the guardrail reads the catalogue
fresh through the catalogue accessor on every call, and that accessor's cache
is invalidated by ``catalogue.set_catalogue_path`` /
``catalogue.reset_catalogue_cache``, so mutating the catalogue file and
resetting the cache changes the verdict.

Accepted outputs:
- ``str`` — normal customer replies (SKU, currency-amount and availability checks).
- ``orders.Order`` — structured confirmations (full ``orders.validate_order``).

Currency amounts (fix round 2, controller ruling): a sentence boundary is
``!``, ``?``, ``;`` or a period followed by whitespace/end — never the period
inside a product name like ``Electric kettle 1.7L``. Aggregate candidates are
computed over a WHOLE-TEXT window: products named anywhere (SKU tokens,
canonical and case-sensitive, plus products whose catalogue NAME appears
anywhere) and quantity tokens 1..20 anywhere. Every candidate remains an
arithmetic combination of catalogue unit prices (unit price, unit price ×
quantity, sums, pairwise sums); attribution across products is NOT checked —
that is the accepted letter of FR-6 (existence, not attribution). The wider
window is therefore more permissive, but nothing not derived from the
catalogue is ever accepted.

The guardrail NEVER raises. On an unexpected internal exception it logs and
returns ``tripwire_triggered=False`` with ``reason="guardrail_error"`` — a
broken guardrail must not kill the app. The data guarantee itself is enforced
by tests over the normal paths.

On a tripwire, ``output_info`` carries a machine-readable ``reason`` plus the
offending SKUs/amounts/problems and the ready-made ``polite_refusal`` message
so the caller can compose the polite customer reply.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agents import Agent, GuardrailFunctionOutput, RunContextWrapper

import catalogue
import orders
from context import ShopContext

logger = logging.getLogger("shopdesk.guardrails")

POLITE_REFUSAL = (
    "I'm sorry, I can't quote that — let me re-check our catalogue and get right back to you."
)

# Product codes like KTL-01, FAN-22, TV-43S (optional trailing letter), so the
# catalogue's own TV-43S is visible and invented/lowercase spellings such as
# "TV-55S" or "fan-22" enter the unknown-SKU check (matched EXACTLY,
# case-sensitive, against the catalogue's canonical SKU).
SKU_PATTERN = re.compile(r"\b[A-Za-z]{2,4}-\d{2,3}[A-Za-z]?\b")
# Currency amounts: captures the figure after PKR / Rs. / pkr markers.
AMOUNT_PATTERN = re.compile(r"(?:PKR|Rs\.?|pkr)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)
# Phrases that imply an item is available for sale. Word boundaries on both
# sides so "available" inside "unavailable" does NOT read as availability.
AVAILABILITY_PATTERN = re.compile(
    r"\b(in stock|available|can (be )?order|we (have|sell))\b", re.IGNORECASE
)
# Small integer tokens (bounded: only 1..20 are used as quantity candidates).
QTY_TOKEN_PATTERN = re.compile(r"\b(\d{1,2})\b")
# Sentence boundaries: !, ? and ; (and line breaks) ALWAYS end a sentence; a
# period ends one ONLY when followed by whitespace or the end of the text. A
# period glued to the next character ("1.7L", "e.g.") is NOT a boundary, so a
# product name like "Electric kettle 1.7L (KTL-01)" stays in the same sentence
# as the price that follows it.
_SENTENCE_SPLIT = re.compile(r"[!?;\n]|\.(?=\s|$)")

# Whole-text aggregate window (fix round 2): name words shorter than this are
# never used to map a product onto the text, and pairwise sums are only built
# while at most this many SKUs are mentioned (bounded, deterministic).
_NAME_WORD_MIN_LEN = 3
_MAX_PAIRWISE_SKUS = 8

AMOUNT_TOLERANCE = 0.01
MAX_QTY_TOKEN = 20


def _catalogue_prices() -> list[float]:
    """All catalogue unit prices, read fresh this call."""
    prices: list[float] = []
    for product in catalogue.load_catalogue().get("products", []):
        if isinstance(product, dict):
            try:
                prices.append(float(product.get("price", 0)))
            except (TypeError, ValueError):
                continue
    return prices


def _sentences(text: str) -> list[str]:
    """Split into sentences; protect ``Rs.`` so its period isn't a sentence end."""
    return [s for s in _SENTENCE_SPLIT.split(text.replace("Rs.", "Rs")) if s.strip()]


def _qty_tokens(text: str) -> list[int]:
    """Small integers 1..20 found in the text (bounded, deterministic)."""
    tokens: list[int] = []
    for raw in QTY_TOKEN_PATTERN.findall(text):
        value = int(raw)
        if 1 <= value <= MAX_QTY_TOKEN:
            tokens.append(value)
    return tokens


def _name_mentioned_skus(text: str) -> list[str]:
    """Canonical SKUs whose catalogue NAME appears anywhere in the text.

    Name-based mapping for the whole-text aggregate window, in the spirit of
    ``catalogue.product_by_name`` but applied catalogue-name-word -> text with
    exact/substring (word-start) matching only. Calling ``product_by_name``
    per text token is deliberately avoided: it misses plural forms
    (``product_by_name("kettles")`` returns None) and its fallback substring
    direction is dangerously fuzzy for short tokens. Only alphabetic name
    words of length >= ``_NAME_WORD_MIN_LEN`` match (so "1.7L", "3-in-1" and
    "43-inch" never map anything), case-insensitively, at a word start (so the
    name word "kettle" matches the text word "kettles"). The returned SKUs are
    the catalogue's canonical spellings.
    """
    lowered = text.casefold()
    skus: list[str] = []
    for product in catalogue.load_catalogue().get("products", []):
        if not isinstance(product, dict):
            continue
        sku = str(product.get("sku", ""))
        if not sku or sku in skus:
            continue
        for word in str(product.get("name", "")).split():
            if len(word) < _NAME_WORD_MIN_LEN or not word.isalpha():
                continue
            if re.search(rf"\b{re.escape(word.casefold())}", lowered):
                skus.append(sku)
                break
    return skus


def _aggregate_candidates(prices: list[float], qty_tokens: list[int]) -> set[float]:
    """Arithmetic combinations of catalogue unit prices (whole-text window).

    Every candidate is built ONLY from the unit prices of products named in
    the answer and the small integer quantities found in it: each unit price,
    each unit price × quantity, the sum of unit prices, that sum × quantity,
    and (while at most ``_MAX_PAIRWISE_SKUS`` SKUs are mentioned) pairwise sums
    of distinct SKUs' price×quantity terms — so a mixed quote such as
    "2 kettles and 1 iron come to PKR 12,000" is justified. Nothing here can
    admit a number that is not derived from catalogue prices.
    """
    candidates: set[float] = set()
    quantities = [1] + sorted(set(qty_tokens))
    # Per-SKU terms: unit price times each quantity seen anywhere in the text.
    terms: list[list[float]] = [[round(p * q, 2) for q in quantities] for p in prices]
    for sku_terms in terms:
        candidates.update(sku_terms)  # p_i × q (q=1 gives the unit price itself)
    if prices:
        base_sum = round(sum(prices), 2)
        candidates.add(base_sum)  # Σ p_i
        for qty in quantities:
            candidates.add(round(base_sum * qty, 2))  # Σ (p_i × q)
    if 2 <= len(prices) <= _MAX_PAIRWISE_SKUS:
        for i in range(len(prices)):
            for j in range(i + 1, len(prices)):
                for a in terms[i]:
                    for b in terms[j]:
                        candidates.add(round(a + b, 2))  # pairwise sums, distinct SKUs
    return candidates


def _known_product(sku: str) -> dict[str, Any] | None:
    """Resolve a SKU-token match to its product, canonically and case-sensitively.

    Exact, case-sensitive existence: "fan-22" resolves via the case-insensitive
    accessor but is not the catalogue's canonical "FAN-22", so it is NOT a
    known product (and lands in the unknown-SKU offences).
    """
    product = catalogue.get_product(sku)
    if product is None or str(product.get("sku", "")) != sku:
        return None
    return product


def _check_text(text: str) -> dict[str, list[Any]]:
    """Scan a finished answer; return the offences found, by kind."""
    unknown_skus: list[str] = []
    bad_amounts: list[float] = []
    out_of_stock_skus: list[str] = []

    catalogue_prices = _catalogue_prices()

    # Whole-text window (fix round 2, controller ruling): SKUs and quantity
    # tokens are collected across the ENTIRE answer so a bulk quote in its own
    # sentence is still justified by the catalogue figures named elsewhere.
    working = text.replace("Rs.", "Rs")

    mentioned: dict[str, dict[str, Any]] = {}
    for sku in SKU_PATTERN.findall(working):
        product = _known_product(sku)
        if product is not None:
            mentioned[str(product.get("sku", sku))] = product
    for sku in _name_mentioned_skus(working):
        product = catalogue.get_product(sku)
        if product is not None:
            mentioned[str(product.get("sku", sku))] = product

    # Every amount candidate is an arithmetic combination of catalogue unit
    # prices: global catalogue prices, or aggregates over the products named
    # in the text. Attribution across products is not checked — existence,
    # not attribution, is the accepted letter of FR-6.
    prices = [float(p.get("price", 0)) for p in mentioned.values()]
    candidates = set(catalogue_prices) | _aggregate_candidates(
        prices, _qty_tokens(working)
    )

    for sentence in _sentences(working):
        # (a) SKU existence, sentence-scoped and case-sensitive: every match
        # in this sentence must be the catalogue's canonical spelling.
        for sku in SKU_PATTERN.findall(sentence):
            if _known_product(sku) is None and sku not in unknown_skus:
                unknown_skus.append(sku)

        # (c) Out-of-stock items must never be offered for sale: availability
        # language in the SAME sentence as a zero-stock SKU token is a
        # tripwire. Deliberately sentence-scoped to SKU tokens (not the
        # name-mapped whole-text window): a name alone plus "we have"
        # elsewhere must not read as selling a zero-stock item.
        if AVAILABILITY_PATTERN.search(sentence):
            for sku in SKU_PATTERN.findall(sentence):
                product = _known_product(sku)
                if product is None:
                    continue  # unknown SKUs are reported separately above
                try:
                    stock = int(product.get("stock", 0))
                except (TypeError, ValueError):
                    stock = 0
                if stock <= 0 and sku not in out_of_stock_skus:
                    out_of_stock_skus.append(sku)

        # (b) Every currency amount must be a legitimate catalogue figure: a
        # catalogue unit price, or a whole-text aggregate built from the unit
        # prices of the products named in the answer and its quantity tokens.
        for match in AMOUNT_PATTERN.finditer(sentence):
            raw = match.group(1).replace(",", "")
            try:
                amount = float(raw)
            except ValueError:
                continue
            if not any(
                abs(amount - candidate) <= AMOUNT_TOLERANCE + 1e-9
                for candidate in candidates
            ):
                bad_amounts.append(amount)

    return {
        "unknown_skus": unknown_skus,
        "bad_amounts": bad_amounts,
        "out_of_stock_skus": out_of_stock_skus,
    }


def _info(checked: str, reason: str, **details: Any) -> dict[str, Any]:
    """Build the output_info dict, always carrying the polite refusal text."""
    info: dict[str, Any] = {
        "guardrail": "catalogue_output",
        "checked": checked,
        "reason": reason,
        "polite_refusal": POLITE_REFUSAL,
    }
    info.update(details)
    return info


def catalogue_output_guardrail(
    ctx: RunContextWrapper[ShopContext], agent: Agent, output: Any
) -> GuardrailFunctionOutput:
    """FR-6 output guardrail: check the finished answer against the catalogue.

    Accepts ``str`` replies and :class:`orders.Order` confirmations. The
    context wrapper is part of the SDK contract but the data check does not
    depend on its contents — the catalogue is the truth source.
    """
    try:
        if isinstance(output, orders.Order):
            problems = orders.validate_order(output)
            if problems:
                return GuardrailFunctionOutput(
                    output_info=_info("order", "order_invalid", problems=problems),
                    tripwire_triggered=True,
                )
            return GuardrailFunctionOutput(
                output_info=_info("order", "ok"), tripwire_triggered=False
            )

        if isinstance(output, str):
            offences = _check_text(output)
            reasons: list[str] = []
            if offences["unknown_skus"]:
                reasons.append("unknown_sku")
            if offences["bad_amounts"]:
                reasons.append("unverifiable_amount")
            if offences["out_of_stock_skus"]:
                reasons.append("out_of_stock_sale")
            if reasons:
                return GuardrailFunctionOutput(
                    output_info=_info("text", "+".join(reasons), **offences),
                    tripwire_triggered=True,
                )
            return GuardrailFunctionOutput(
                output_info=_info("text", "ok"), tripwire_triggered=False
            )

        logger.warning(
            "catalogue_output_guardrail: unsupported output type %s; passing through",
            type(output).__name__,
        )
        return GuardrailFunctionOutput(
            output_info=_info(type(output).__name__, "unsupported_output_type"),
            tripwire_triggered=False,
        )
    except Exception as exc:  # a broken guardrail must not kill the app
        logger.exception("catalogue_output_guardrail errored; passing through")
        return GuardrailFunctionOutput(
            output_info=_info("unknown", "guardrail_error", error=str(exc)),
            tripwire_triggered=False,
        )
