"""Catalogue accessor for Shop Desk (FR-1 data source).

Loads the repo-root ``catalogue.json`` with a per-path cache. Tests can point
the accessor at a fixture file with :func:`set_catalogue_path` and clear the
cache with :func:`reset_catalogue_cache`.

All lookups are case-insensitive and tolerant of minor whitespace. Errors are
raised as :class:`CatalogueError` with a clear one-sentence message; tool
bodies in ``tools.py`` catch these and turn them into polite sentences.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("shopdesk.catalogue")

DEFAULT_CATALOGUE_PATH = Path(__file__).resolve().parent / "catalogue.json"


class CatalogueError(RuntimeError):
    """Raised when the catalogue file is missing or malformed."""


_path: Path | None = None
_cache: dict[str, Any] | None = None


def set_catalogue_path(path: Path | str) -> None:
    """Point the accessor at a catalogue file (used by tests) and clear cache."""
    global _path, _cache
    _path = Path(path)
    _cache = None


def reset_catalogue_cache() -> None:
    """Drop the cached catalogue so the next load re-reads the file."""
    global _cache
    _cache = None


def _current_path() -> Path:
    return _path if _path is not None else DEFAULT_CATALOGUE_PATH


def _normalise(value: Any) -> str:
    """Lower-case and collapse whitespace so lookups tolerate small typos."""
    return " ".join(str(value).split()).casefold()


def load_catalogue() -> dict[str, Any]:
    """Return the full parsed catalogue (cached per path).

    Raises:
        CatalogueError: with a one-sentence message if the file is missing,
            unreadable or malformed.
    """
    global _cache
    if _cache is not None:
        return _cache

    path = _current_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CatalogueError(
            f"Catalogue file '{path.name}' was not found; expected it in the project root."
        ) from exc
    except OSError as exc:
        raise CatalogueError(
            f"Catalogue file '{path.name}' could not be read from disk."
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CatalogueError(
            f"Catalogue file '{path.name}' is not valid JSON."
        ) from exc

    if not isinstance(data, dict) or not isinstance(data.get("products"), list):
        raise CatalogueError(
            f"Catalogue file '{path.name}' is malformed: expected an object with a 'products' list."
        ) from None

    _cache = data
    return _cache


def get_product(sku: str) -> dict[str, Any] | None:
    """Return the product with the given SKU (case-insensitive) or None."""
    wanted = _normalise(sku)
    if not wanted:
        return None
    for product in load_catalogue().get("products", []):
        if isinstance(product, dict) and _normalise(product.get("sku", "")) == wanted:
            return product
    return None


def product_by_name(name: str) -> dict[str, Any] | None:
    """Fuzzy-resolve a product by (partial) name, case-insensitive.

    Matches an exact normalised name first, then a substring match in either
    direction, then a SKU match as a convenience. Returns None when nothing
    matches — callers must never invent a product.
    """
    wanted = _normalise(name)
    if not wanted:
        return None
    products = [
        p for p in load_catalogue().get("products", []) if isinstance(p, dict)
    ]
    for product in products:
        if _normalise(product.get("name", "")) == wanted:
            return product
    for product in products:
        product_name = _normalise(product.get("name", ""))
        if wanted in product_name or product_name in wanted:
            return product
    return get_product(name)


def format_price(price: Any, currency: str | None = None) -> str:
    """Format a price in whole currency units, e.g. ``"PKR 4,200"``.

    ``currency`` defaults to the catalogue's own ``currency`` field.
    """
    if currency is None:
        currency = str(load_catalogue().get("currency", "") or "")
    try:
        amount = float(price)
    except (TypeError, ValueError):
        return f"{currency} {price}".strip()
    if amount.is_integer():
        return f"{currency} {int(amount):,}"
    return f"{currency} {amount:,.2f}"
