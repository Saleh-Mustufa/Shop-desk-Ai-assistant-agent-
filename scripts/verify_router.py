"""Live §7 verification: registry vs the real Gemini OpenAI-compatible endpoint.

Reads GEMINI_API_KEY from .env via config (the key is never printed). For every
available registry model it makes one tiny chat completion to confirm the model
ID is live (a model answering 404 / "no longer available" prints FAIL and the
script continues), then proves the routed path end-to-end with a single
RoutedModel.get_response call, printing which candidate served it. Per-model
checks are paced ~4s apart to respect the 5-RPM tier.

Usage:
    python scripts/verify_router.py          # full live verification
    python scripts/verify_router.py --list   # registry table + live model list (re-verification aid)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from agents import ModelSettings, ModelTracing  # noqa: E402
from openai import AsyncOpenAI, OpenAI  # noqa: E402

import model_config  # noqa: E402  — model names come from MODEL_REGISTRY, never hardcoded here
from model_config import MODEL_REGISTRY, get_routed_model  # noqa: E402

CHECK_PACING_SECONDS = 4.0


def _short(exc: BaseException) -> str:
    """One-line, bounded exception description."""
    return " ".join(str(exc).split())[:160] or type(exc).__name__


def print_registry() -> None:
    print("=== §7 registry (model_config.MODEL_REGISTRY) ===")
    for entry in MODEL_REGISTRY.values():
        state = "available" if entry.available else "UNAVAILABLE (kept for provenance)"
        print(
            f"  {entry.name:26s} tier={entry.tier:7s} rpm={entry.rpm:<3d} "
            f"tpm={entry.tpm} rpd={entry.rpd} {state}"
        )


def live_model_ids(client: OpenAI) -> set[str]:
    """Model ids served by the live endpoint, normalized (leading 'models/' stripped)."""
    try:
        return {model.id.split("/")[-1] for model in client.models.list()}
    except Exception as exc:  # noqa: BLE001 — verification must continue
        print(f"  live model list unavailable: {_short(exc)}")
        return set()


def check_models(client: OpenAI) -> dict[str, bool]:
    """One tiny chat completion per available model; FAIL lines never abort."""
    available = [entry for entry in MODEL_REGISTRY.values() if entry.available]
    print(
        f"\n=== per-model live checks ({len(available)} available, "
        f"paced {CHECK_PACING_SECONDS:.0f}s for the 5-RPM tier) ==="
    )
    results: dict[str, bool] = {}
    for position, entry in enumerate(available):
        if position:
            time.sleep(CHECK_PACING_SECONDS)
        try:
            client.chat.completions.create(
                model=entry.name,
                messages=[{"role": "user", "content": "Reply with the single word OK."}],
                max_tokens=5,
            )
            print(f"  {entry.name:26s} OK")
            results[entry.name] = True
        except Exception as exc:  # noqa: BLE001 — a failing model must not stop verification
            print(f"  {entry.name:26s} FAIL ({_short(exc)})")
            results[entry.name] = False
    return results


async def routed_call() -> tuple[str, str]:
    """One end-to-end RoutedModel.get_response through the resolved chain."""
    client = AsyncOpenAI(
        base_url=model_config.GEMINI_OPENAI_BASE_URL, api_key=config.get_gemini_api_key()
    )
    routed = get_routed_model(client=client)
    try:
        response = await routed.get_response(
            None,
            "Reply with the single word OK.",
            ModelSettings(),
            [],
            None,
            [],
            ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
    finally:
        await routed.close()
    text = ""
    for item in response.output:
        for piece in getattr(item, "content", None) or []:
            text += getattr(piece, "text", "") or ""
    return routed.active_model_name, text.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the registry table and the live model list, without chat calls",
    )
    args = parser.parse_args()

    print_registry()
    client = OpenAI(
        base_url=model_config.GEMINI_OPENAI_BASE_URL, api_key=config.get_gemini_api_key()
    )

    if args.list:
        live = live_model_ids(client)
        if live:
            print(f"\n=== live endpoint model list ({len(live)} ids) ===")
            for name in MODEL_REGISTRY:
                print(f"  {name:26s} {'in live list' if name in live else 'NOT in live list'}")
            unknown = sorted(live - set(MODEL_REGISTRY))
            if unknown:
                print(f"  live ids not in registry: {', '.join(unknown)}")
        return 0

    results = check_models(client)

    print("\n=== end-to-end routed call (RoutedModel.get_response) ===")
    served_by = "<none>"
    try:
        served_by, text = asyncio.run(routed_call())
        print(f"  served by: {served_by}")
        print(f"  reply: {text[:80]!r}")
    except Exception as exc:  # noqa: BLE001 — report, then summarize
        print(f"  routed call FAILED: {_short(exc)}")
        served_by = None

    ok = sorted(name for name, passed in results.items() if passed)
    failed = sorted(name for name, passed in results.items() if not passed)
    available_count = sum(1 for entry in MODEL_REGISTRY.values() if entry.available)
    print("\n=== summary ===")
    print(f"  registry: {len(MODEL_REGISTRY)} models, {available_count} available")
    print(f"  live checks: {len(ok)} OK, {len(failed)} FAIL" + (f": {', '.join(failed)}" if failed else ""))
    print(f"  routed call: {'served by ' + served_by if served_by else 'FAILED'}")
    return 1 if (failed or served_by is None) else 0


if __name__ == "__main__":
    raise SystemExit(main())
