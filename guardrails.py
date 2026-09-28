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

# Product codes like KTL-01, FAN-22, TV-43S.
SKU_PATTERN = re.compile(r"\b[A-Z]{2,4}-\d{2,3}\b")
# Currency amounts: captures the figure after PKR / Rs. / pkr markers.
AMOUNT_PATTERN = re.compile(r"(?:PKR|Rs\.?|pkr)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)
# Phrases that imply an item is available for sale.
AVAILABILITY_PATTERN = re.compile(r"(in stock|available|can (be )?order|we (have|sell))", re.IGNORECASE)
# Small integer tokens (bounded: only 1..20 are used as quantity candidates).
QTY_TOKEN_PATTERN = re.compile(r"\b(\d{1,2})\b")
_SENTENCE_SPLIT = re.compile(r"[.!?;\n]+")

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


def _qty_tokens(sentence: str) -> list[int]:
    """Small integers 1..20 found in the sentence (bounded, deterministic)."""
    tokens: list[int] = []
    for raw in QTY_TOKEN_PATTERN.findall(sentence):
        value = int(raw)
        if 1 <= value <= MAX_QTY_TOKEN:
            tokens.append(value)
    return tokens


def _check_text(text: str) -> dict[str, list[Any]]:
    """Scan a finished answer; return the offences found, by kind."""
    unknown_skus: list[str] = []
    bad_amounts: list[float] = []
    out_of_stock_skus: list[str] = []

    catalogue_prices = _catalogue_prices()

    for sentence in _sentences(text):
        mentioned: dict[str, dict[str, Any]] = {}
        for sku in SKU_PATTERN.findall(sentence):
            product = catalogue.get_product(sku)
            if product is None:
                if sku not in unknown_skus:
                    unknown_skus.append(sku)
            else:
                mentioned[str(product.get("sku", sku))] = product

        # Out-of-stock items must never be offered for sale: availability
        # language in the SAME sentence as a zero-stock SKU is a tripwire.
        if AVAILABILITY_PATTERN.search(sentence):
            for sku, product in mentioned.items():
                try:
                    stock = int(product.get("stock", 0))
                except (TypeError, ValueError):
                    stock = 0
                if stock <= 0 and sku not in out_of_stock_skus:
                    out_of_stock_skus.append(sku)

        # Every currency amount must be a legitimate catalogue figure: a
        # catalogue unit price, OR a sentence-scoped aggregate — the sum of
        # the sentence's mentioned SKU prices, optionally multiplied by the
        # small integer quantity tokens found in that sentence.
        prices = [float(p.get("price", 0)) for p in mentioned.values()]
        candidates = set(catalogue_prices)
        if prices:
            line_sum = round(sum(prices), 2)
            candidates.add(line_sum)
            for qty in _qty_tokens(sentence):
                candidates.add(round(line_sum * qty, 2))
                for price in prices:
                    candidates.add(round(price * qty, 2))

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
