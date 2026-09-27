"""Customer-facing catalogue tools for Shop Desk (FR-1 tool surface, FR-7 gating).

Every tool takes ``ctx: RunContextWrapper[ShopContext]`` as its FIRST
parameter; the OpenAI Agents SDK dependency-injects the wrapper and it never
appears in the generated JSON schema (FR-2 acceptance).

Gating (FR-7):
- ``loyalty_benefit`` is tier-gated: only offered to ``tier == "regular"``
  via ``function_tool(is_enabled=...)``.
- ``holiday_bundles`` is statically disabled (off-season): never offered.

Robustness (NFR-4): tools NEVER raise into the runner — every body catches
``Exception`` and returns a polite, useful sentence, logging at warning level.
Tool logic lives in plain ``_*_impl`` helper functions so tests stay
synchronous; the ``@function_tool`` wrappers delegate to them.
"""

from __future__ import annotations

import logging
from typing import Any

from agents import RunContextWrapper, function_tool

import catalogue
from context import ShopContext

logger = logging.getLogger("shopdesk.tools")

SORRY_MESSAGE = (
    "Sorry, I had trouble checking the catalogue just now — please try again."
)


def _price_text(ctx: RunContextWrapper[ShopContext], price: Any) -> str:
    """Format a price using the conversation currency from the context."""
    currency = getattr(ctx.context, "currency", None) or None
    return catalogue.format_price(price, currency)


def _describe_product(ctx: RunContextWrapper[ShopContext], product: dict[str, Any]) -> str:
    """Customer-ready sentence with price and stock for a resolved product."""
    sku = str(product.get("sku", "unknown"))
    name = str(product.get("name", sku))
    price_text = _price_text(ctx, product.get("price"))
    try:
        stock = int(product.get("stock", 0))
    except (TypeError, ValueError):
        stock = 0
    if stock <= 0:
        return (
            f"The {name} ({sku}) is out of stock at the moment. I can suggest "
            "an alternative from our catalogue if you tell me what you need it for."
        )
    return (
        f"The {name} ({sku}) costs {price_text} and we have {stock} in stock."
    )


def _lookup_product_impl(ctx: RunContextWrapper[ShopContext], sku: str) -> str:
    """Plain implementation of lookup_product (kept sync for tests)."""
    try:
        wanted = str(sku or "").strip()
        if not wanted:
            return (
                "Could you share the product code you'd like me to look up? "
                "I can also check by product name."
            )
        product = catalogue.get_product(wanted)
        if product is None:
            return (
                f"I couldn't find product code {wanted} in our catalogue. "
                "Please double-check the code, or tell me the product name "
                "and I'll look it up for you."
            )
        return _describe_product(ctx, product)
    except Exception:
        logger.warning("lookup_product failed for sku=%r", sku, exc_info=True)
        return SORRY_MESSAGE


def _check_stock_by_name_impl(ctx: RunContextWrapper[ShopContext], name: str) -> str:
    """Plain implementation of check_stock_by_name (kept sync for tests)."""
    try:
        wanted = str(name or "").strip()
        if not wanted:
            return (
                "Which product would you like me to check? "
                "Just name any item from our catalogue."
            )
        product = catalogue.product_by_name(wanted)
        if product is None:
            return (
                f'I couldn\'t find anything matching "{wanted}" in our catalogue. '
                "Could you name a product from our catalogue list? I'll gladly "
                "check its price and stock for you."
            )
        return _describe_product(ctx, product)
    except Exception:
        logger.warning("check_stock_by_name failed for name=%r", name, exc_info=True)
        return SORRY_MESSAGE


def _list_catalogue_impl(ctx: RunContextWrapper[ShopContext]) -> str:
    """Plain implementation of list_catalogue (kept sync for tests)."""
    try:
        products = [
            p
            for p in catalogue.load_catalogue().get("products", [])
            if isinstance(p, dict)
        ]
        if not products:
            return "Our catalogue is empty right now — please check back soon."
        lines = [
            f"- {p.get('sku', '?')} — {p.get('name', '?')} ({_price_text(ctx, p.get('price'))})"
            for p in products
        ]
        return "Here's our current catalogue:\n" + "\n".join(lines)
    except Exception:
        logger.warning("list_catalogue failed", exc_info=True)
        return SORRY_MESSAGE


def _loyalty_benefit_impl(ctx: RunContextWrapper[ShopContext]) -> str:
    """Plain implementation of loyalty_benefit (kept sync for tests)."""
    try:
        return (
            "As one of our regular customers you get a 5% loyalty discount, "
            "applied automatically when your order is finalised — no code needed."
        )
    except Exception:
        logger.warning("loyalty_benefit failed", exc_info=True)
        return SORRY_MESSAGE


def _holiday_bundles_impl(ctx: RunContextWrapper[ShopContext]) -> str:
    """Plain implementation of holiday_bundles (kept sync for tests)."""
    try:
        return (
            "We don't have any holiday bundle offers running right now — "
            "our regular catalogue has plenty of great products, though."
        )
    except Exception:
        logger.warning("holiday_bundles failed", exc_info=True)
        return SORRY_MESSAGE


def _loyalty_enabled(
    ctx: RunContextWrapper[ShopContext], agent: Any
) -> bool:
    """FR-7: the loyalty perk is offered only to regular-tier customers."""
    try:
        return getattr(ctx.context, "tier", "walk_in") == "regular"
    except Exception:
        logger.warning("loyalty gate failed; defaulting to disabled", exc_info=True)
        return False


def _holiday_disabled(ctx: RunContextWrapper[ShopContext], agent: Any) -> bool:
    """FR-7: seasonal tool is statically disabled and invisible in every schema."""
    return False


@function_tool(is_enabled=_loyalty_enabled)
def loyalty_benefit(ctx: RunContextWrapper[ShopContext]) -> str:
    """Describe the regular-customer loyalty discount. Only offered to regular-tier customers."""
    return _loyalty_benefit_impl(ctx)


@function_tool(is_enabled=_holiday_disabled)
def holiday_bundles(ctx: RunContextWrapper[ShopContext]) -> str:
    """Mention seasonal holiday bundle offers. Statically disabled (off-season)."""
    return _holiday_bundles_impl(ctx)


@function_tool
def lookup_product(ctx: RunContextWrapper[ShopContext], sku: str) -> str:
    """Look up one product by its exact product code (SKU) and report its price and stock."""
    return _lookup_product_impl(ctx, sku)


@function_tool
def check_stock_by_name(ctx: RunContextWrapper[ShopContext], name: str) -> str:
    """Check price and stock for a product the customer describes by name (fuzzy match)."""
    return _check_stock_by_name_impl(ctx, name)


@function_tool
def list_catalogue(ctx: RunContextWrapper[ShopContext]) -> str:
    """List every product in the shop catalogue with its code, name and price."""
    return _list_catalogue_impl(ctx)
