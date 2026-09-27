"""Per-conversation shop context for Shop Desk (FR-2).

The ShopContext instance is created once per conversation and handed to every
run via ``RunContextWrapper[ShopContext]``. Tools read it through the wrapper;
it is NEVER injected into prompt text (no customer identifiers in prompts).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ShopContext:
    """Context passed to every agent run and read by the shop tools."""

    shop: str
    currency: str
    customer_id: str
    tier: str = "walk_in"  # "walk_in" or "regular"
