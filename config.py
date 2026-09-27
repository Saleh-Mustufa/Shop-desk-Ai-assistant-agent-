"""Shop-wide configuration: .env loading and startup validation.

Serves NFR-1 (secrets only in .env, fail fast with one sentence) and FR-1's
global-default level (SHOP_DEFAULT_PROFILE). This module never mentions model
names — all model identity and model logic lives in model_config.py.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent

# Idempotent: loads .env once at import; existing process env wins (dotenv
# does not override already-set variables by default).
load_dotenv(_ROOT / ".env")

_DEFAULT_PROFILE = "fast"


def get_gemini_api_key() -> str:
    """Return the Gemini API key read from the environment/.env.

    Fails fast with exactly one sentence and no stack-trace noise when the key
    is missing (NFR-1): SystemExit prints only its message.
    """
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise SystemExit(
            "GEMINI_API_KEY is not set: add it to the .env file in the project root before running Shop Desk."
        )
    return key


def shop_default_profile() -> str:
    """Global-default profile (FR-1 level 1): env SHOP_DEFAULT_PROFILE, default 'fast'."""
    value = (os.environ.get("SHOP_DEFAULT_PROFILE") or "").strip()
    return value or _DEFAULT_PROFILE


def priority_model() -> str | None:
    """Top-priority model name (FR-1): env PRIORITY_MODEL, default None."""
    value = (os.environ.get("PRIORITY_MODEL") or "").strip()
    return value or None
