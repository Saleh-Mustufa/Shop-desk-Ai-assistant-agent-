"""Dynamic per-turn instruction builders for Shop Desk (FR-4).

The desk agent's system prompt is rebuilt every turn from the ShopContext
fields (shop name, currency, tier-aware tool mentions) plus the simulated
clock: inside shop hours it promises same-day delivery; outside hours it says
the shop is closed and gives the opening time. FR-10 adds one escalation
paragraph: when to use the escalate_to_human tool and which typed reason
codes exist.

FR-2: context is read to build the prompt, but the customer_id is NEVER
included in any prompt text.

``desk_instructions`` is a callable usable directly as
``Agent(instructions=desk_instructions)`` — the SDK calls it as
``instructions(run_context_wrapper, agent)``. Tests can freeze the clock via
:func:`set_clock_provider`; the default is ``datetime.now``.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Callable

from agents import RunContextWrapper

from context import ShopContext

OPEN_HOUR = 9
CLOSE_HOUR = 21

_clock_provider: Callable[[], _dt.datetime] = _dt.datetime.now


def set_clock_provider(provider: Callable[[], _dt.datetime]) -> None:
    """Override the clock source (used by tests to freeze time)."""
    global _clock_provider
    _clock_provider = provider


def get_clock_provider() -> Callable[[], _dt.datetime]:
    """Return the current clock source."""
    return _clock_provider


def _hours_line(now: _dt.datetime) -> str:
    """The per-turn delivery promise, driven by the clock (FR-4)."""
    if OPEN_HOUR <= now.hour < CLOSE_HOUR:
        return "Same-day delivery is available for orders placed now."
    return (
        f"The shop is closed right now (opens at {OPEN_HOUR}:00). "
        "Same-day delivery is not available; orders placed now are processed at opening."
    )


def _tier_line(tier: str) -> str:
    """Tier-aware tool mention (FR-7 interplay with the prompt)."""
    if tier == "regular":
        return (
            "This customer is a regular customer: the loyalty_benefit tool is "
            "available and describes their 5% loyalty discount — mention it when relevant."
        )
    return (
        "This customer is a walk-in customer: the loyalty discount applies only "
        "to regular customers, so do not promise any discount."
    )


def build_desk_prompt(ctx: ShopContext, now: _dt.datetime) -> str:
    """Build the full desk system prompt for one turn. Printable with no model call."""
    shop = getattr(ctx, "shop", "our shop") or "our shop"
    currency = getattr(ctx, "currency", "") or ""
    tier = getattr(ctx, "tier", "walk_in") or "walk_in"

    return (
        f"You are the customer-facing shop assistant for {shop}. "
        f"Quote all prices in {currency}.\n\n"
        f"{_hours_line(now)}\n\n"
        f"{_tier_line(tier)}\n\n"
        "Catalogue tools you may use: lookup_product (by exact product code), "
        "check_stock_by_name (by product name) and list_catalogue (the full list).\n\n"
        "Rules:\n"
        "- Answer ONLY from catalogue tool outputs, never from memory.\n"
        "- Never state a price or SKU you did not get from a tool during this "
        "conversation, and never invent products.\n"
        "- Ask before building an order: confirm each item, quantity and price "
        "with the customer first.\n"
        "- Only when the customer clearly confirms the full order, call the "
        "finalize-order flow so the order-taking step takes over.\n"
        "- Be warm and concise; write in customer-ready sentences.\n\n"
        "If the customer asks for a human, or their request is outside what "
        "you can help with (e.g. complaints about third-party services, legal "
        "questions, anything you cannot resolve), use the escalate_to_human "
        "tool and pass the matching reason (out_of_scope, order_problem, "
        "customer_request, policy, repeated_failure). Tell the customer a "
        "human colleague is taking over."
    )


def build_fastpath_prompt(ctx: ShopContext, now: _dt.datetime) -> str:
    """Concise prompt for the later fast-path catalogue agent."""
    shop = getattr(ctx, "shop", "our shop") or "our shop"
    currency = getattr(ctx, "currency", "") or ""
    return (
        f"You are the fast-path catalogue helper for {shop}. Prices are in {currency}. "
        "For the customer's product question, call lookup_product when they give a "
        "product code, or check_stock_by_name when they describe a product by name. "
        "Answer in one or two short sentences using only the tool output; if the "
        "product is not found, say so politely and suggest they name another product "
        "from the catalogue. Never invent prices, SKUs or stock levels."
    )


def desk_instructions(context: RunContextWrapper, agent: Any) -> str:
    """Dynamic instructions callable: ``Agent(instructions=desk_instructions)``."""
    return build_desk_prompt(context.context, get_clock_provider()())
