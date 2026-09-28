"""Typed orders and Python-side order truth for Shop Desk (FR-5).

The order-taker agent may emit a structured :class:`Order`, but the numbers in
it are NEVER trusted: totals are recomputed here in Python, line prices come
from the catalogue (see :func:`build_order`), and every figure on a model-
produced order is re-checked against ``catalogue.json`` by
:func:`validate_order`. A total mismatch is reported, never silently accepted.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel

import catalogue

logger = logging.getLogger("shopdesk.orders")

TOTAL_TOLERANCE = 0.01

OrderStatus = Literal["draft", "confirmed", "escalated"]


class LineItem(BaseModel):
    """One order line: SKU, quantity, and the per-unit price as charged."""

    sku: str
    qty: int
    unit_price: float


class Order(BaseModel):
    """A typed order whose numbers are always re-verified in Python (FR-5)."""

    order_id: str
    status: OrderStatus
    items: list[LineItem]
    total: float


def _catalogue_number(product: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Read a numeric field off a catalogue product, tolerating bad data."""
    try:
        return float(product.get(key, default))
    except (TypeError, ValueError):
        return default


def recompute_total(order: Order) -> float:
    """Recompute the order total in Python — never trust the model's number."""
    return round(sum(item.qty * item.unit_price for item in order.items), 2)


def check_total(order: Order) -> str | None:
    """Return ``None`` when the stated total matches the recomputed one.

    Within :data:`TOTAL_TOLERANCE` (0.01) the order is consistent. Otherwise a
    mismatch description naming BOTH totals is returned — a mismatch is
    reported, never silently accepted.
    """
    recomputed = recompute_total(order)
    if abs(recomputed - order.total) <= TOTAL_TOLERANCE + 1e-9:
        return None
    return (
        f"Order total mismatch: the items sum to {recomputed:.2f} "
        f"but the order states {order.total:.2f}."
    )


def validate_order(order: Order) -> list[str]:
    """Check every line against the catalogue; return the problems found.

    Flags: unknown SKUs, unit prices differing from the catalogue price by
    more than 0.01, non-positive quantities, requested quantity exceeding
    catalogue stock (out-of-stock items are never sold), and a stated total
    that does not match the recomputed one. An empty list means valid.
    """
    problems: list[str] = []
    for item in order.items:
        product = catalogue.get_product(item.sku)
        if product is None:
            problems.append(f"Unknown SKU '{item.sku}' — not in the catalogue.")
            continue
        catalogue_price = _catalogue_number(product, "price")
        if abs(item.unit_price - catalogue_price) > TOTAL_TOLERANCE + 1e-9:
            problems.append(
                f"Unit price {item.unit_price:.2f} for '{item.sku}' does not match "
                f"the catalogue price {catalogue_price:.2f}."
            )
        if item.qty <= 0:
            problems.append(
                f"Quantity for '{item.sku}' must be at least 1 (got {item.qty})."
            )
        stock = int(_catalogue_number(product, "stock"))
        if item.qty > stock:
            problems.append(
                f"Requested {item.qty} of '{item.sku}' but only {stock} are in "
                "stock — out-of-stock items are never sold."
            )
    mismatch = check_total(order)
    if mismatch is not None:
        problems.append(mismatch)
    return problems


def build_order(
    items: list[tuple[str, int]],
    order_id: str,
    status: OrderStatus = "draft",
) -> Order:
    """Build an :class:`Order` with prices taken FROM THE CATALOGUE.

    Python-side truth: the model never sets prices and never computes the
    total. Raises ``ValueError`` with a one-sentence message if any SKU is
    unknown or any quantity is not positive. Called by trusted Python, not by
    a tool.
    """
    line_items: list[LineItem] = []
    for sku, qty in items:
        product = catalogue.get_product(sku)
        if product is None:
            raise ValueError(
                f"Cannot build order '{order_id}': SKU '{sku}' is not in the catalogue."
            )
        if qty <= 0:
            raise ValueError(
                f"Cannot build order '{order_id}': quantity for '{sku}' must be "
                f"a positive whole number (got {qty})."
            )
        line_items.append(
            LineItem(
                sku=str(product.get("sku", sku)),
                qty=int(qty),
                unit_price=_catalogue_number(product, "price"),
            )
        )
    total = round(sum(line.qty * line.unit_price for line in line_items), 2)
    logger.info(
        "Built order %s (status=%s) with %d line(s), total %.2f",
        order_id,
        status,
        len(line_items),
        total,
    )
    return Order(order_id=order_id, status=status, items=line_items, total=total)
