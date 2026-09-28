"""LIVE FR-3 verification: the one-model-call fast path vs the Desk's ordinary loop.

Reads GEMINI_API_KEY from .env via config (the key is NEVER printed). Each
scenario is routed by the deterministic, zero-cost triage (triage.classify)
and then run live:

  1. "What does the kettle cost?"  -> fast path -> FastPath agent.
     Asserts EXACTLY 1 model call (len(result.raw_responses)) and that the
     run's final output IS the lookup tool's fixed template sentence.
  2. "How much is the pedestal fan and do you have it in stock?"
     -> fast path -> 1 model call, out-of-stock wording in the final output.
  3. "Can I order two kettles and a blender?" -> triage sends it to the Desk
     agent (order word) -> normal loop with max_turns=6; prints the model-call
     count (>1 proves the ordinary agent loop) and the reply.

All scenarios are served by the 15-RPM lite tier, so ~5s pacing between
scenarios is safely within limits. Ends with the FR-3 statement of what the
fast path gives up and a PASS/FAIL summary; exits 1 on any failure.

Usage:
    python scripts/verify_fastpath.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402  (loads .env; the key is never printed)
from agents import Runner, RunContextWrapper, set_tracing_disabled  # noqa: E402

import agents_desk  # noqa: E402
import triage  # noqa: E402
import tools  # noqa: E402
from context import ShopContext  # noqa: E402

PACE_SECONDS = 5.0

SCENARIO_1 = "What does the kettle cost?"
SCENARIO_2 = "How much is the pedestal fan and do you have it in stock?"
SCENARIO_3 = "Can I order two kettles and a blender?"

FR3_TRADEOFF = (
    "FR-3 trade-off: the fast path answers with ONE fixed template sentence "
    "produced by the lookup tool (price, stock and SKU straight from "
    "catalogue.json this run). It gives up all model phrasing and nuance — no "
    "rephrasing, no comparison across products, no upsell — and no handling of "
    "compound questions beyond the routed product; anything richer goes to the "
    "Desk agent's ordinary loop."
)


def _make_context() -> ShopContext:
    return ShopContext(shop="Al-Noor Electronics", currency="PKR", customer_id="CUST-LIVE", tier="walk_in")


def _tool_sentences(
    ctx_wrapper: RunContextWrapper[ShopContext], sku: str, name: str
) -> tuple[str, str]:
    """The fixed template sentences the two lookup tools produce for one product.

    These are plain catalogue reads (the impl helpers never raise, NFR-4); the
    scenarios turn a divergence between them into a recorded failure instead
    of crashing, so the PASS/FAIL summary always prints.
    """
    return (
        tools._lookup_product_impl(ctx_wrapper, sku),
        tools._check_stock_by_name_impl(ctx_wrapper, name),
    )


def _print_triage(text: str) -> triage.FastPathDecision:
    decision = triage.classify(text)
    route = "fast path" if decision.is_fast_path else "desk (ordinary loop)"
    product = f"sku={decision.sku} product={decision.product_name!r}" if decision.sku else "no product"
    print(f"  triage       : {route} | {product}")
    print(f"                 reason: {decision.reason}")
    return decision


def _check(failures: list[str], label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"  check: {label} {status}"
    if detail:
        line += f" ({detail})"
    print(line)
    if not ok:
        failures.append(label)


async def scenario_1(ctx: ShopContext, wrapper: RunContextWrapper[ShopContext]) -> list[str]:
    failures: list[str] = []
    print("\n--- scenario 1: plain price question -> fast path ---")
    print(f"  customer     : {SCENARIO_1!r}")
    decision = _print_triage(SCENARIO_1)
    agent = agents_desk.get_fastpath_agent()
    print(f"  agent        : {agent.name} (profile fast -> {agent.model.active_model_name})")
    try:
        result = await Runner.run(agent, SCENARIO_1, context=ctx, max_turns=agents_desk.FASTPATH_MAX_TURNS)
    except Exception as exc:  # noqa: BLE001 — report the failure, keep verifying
        print(f"  run FAILED   : {type(exc).__name__}: {exc}")
        return [f"scenario 1 run failed: {exc}"]
    calls = len(result.raw_responses)
    final = result.final_output
    by_sku, by_name = _tool_sentences(wrapper, "KTL-01", "kettle")
    expected = by_sku
    print(f"  model calls  : {calls}   (len(result.raw_responses))")
    print(f"  final answer : {final!r}")
    print(f"  tool template: {expected!r}")
    print(f"  active model : {agent.model.active_model_name}")
    _check(failures, "triage routed to the fast path", decision.is_fast_path and decision.sku == "KTL-01")
    _check(failures, "lookup tools share one template for the product", by_sku == by_name)
    _check(failures, "exactly 1 model call", calls == 1, f"observed {calls}")
    _check(failures, "final output IS the tool's template sentence", final == expected)
    return failures


async def scenario_2(ctx: ShopContext, wrapper: RunContextWrapper[ShopContext]) -> list[str]:
    failures: list[str] = []
    print("\n--- scenario 2: price + stock of an out-of-stock product -> fast path ---")
    print(f"  customer     : {SCENARIO_2!r}")
    decision = _print_triage(SCENARIO_2)
    agent = agents_desk.get_fastpath_agent()
    print(f"  agent        : {agent.name} (profile fast -> {agent.model.active_model_name})")
    try:
        result = await Runner.run(agent, SCENARIO_2, context=ctx, max_turns=agents_desk.FASTPATH_MAX_TURNS)
    except Exception as exc:  # noqa: BLE001 — report the failure, keep verifying
        print(f"  run FAILED   : {type(exc).__name__}: {exc}")
        return [f"scenario 2 run failed: {exc}"]
    calls = len(result.raw_responses)
    final = str(result.final_output)
    by_sku, by_name = _tool_sentences(wrapper, "FAN-22", "pedestal fan")
    expected = by_sku
    print(f"  model calls  : {calls}   (len(result.raw_responses))")
    print(f"  final answer : {final!r}")
    print(f"  tool template: {expected!r}")
    print(f"  active model : {agent.model.active_model_name}")
    _check(failures, "triage routed to the fast path", decision.is_fast_path and decision.sku == "FAN-22")
    _check(failures, "lookup tools share one template for the product", by_sku == by_name)
    _check(failures, "exactly 1 model call", calls == 1, f"observed {calls}")
    _check(failures, "out-of-stock wording, no availability claim", "out of stock" in final.lower())
    _check(failures, "final output IS the tool's template sentence", final == expected)
    return failures


async def scenario_3(ctx: ShopContext) -> list[str]:
    failures: list[str] = []
    print("\n--- scenario 3: order intent -> Desk agent's ordinary loop ---")
    print(f"  customer     : {SCENARIO_3!r}")
    decision = _print_triage(SCENARIO_3)
    agent = agents_desk.get_desk_agent()
    print(f"  agent        : {agent.name} (profile fast -> {agent.model.active_model_name}, max_turns=6)")
    try:
        result = await Runner.run(agent, SCENARIO_3, context=ctx, max_turns=6)
    except Exception as exc:  # noqa: BLE001 — report the failure, keep verifying
        print(f"  run FAILED   : {type(exc).__name__}: {exc}")
        return [f"scenario 3 run failed: {exc}"]
    calls = len(result.raw_responses)
    final = str(result.final_output)
    print(f"  model calls  : {calls}   (len(result.raw_responses))")
    print(f"  reply        : {final!r}")
    print(f"  active model : {agent.model.active_model_name}")
    _check(failures, "triage routed to the desk", not decision.is_fast_path)
    _check(failures, "ordinary loop used (>1 model call)", calls > 1, f"observed {calls}")
    _check(failures, "desk produced a customer reply", bool(final.strip()))
    return failures


async def run_all() -> tuple[list[str], list[str], list[str]]:
    """All scenarios in one event loop (singleton agents reuse their client)."""
    ctx = _make_context()
    wrapper = RunContextWrapper(context=ctx)
    failures_1 = await scenario_1(ctx, wrapper)
    await asyncio.sleep(PACE_SECONDS)
    failures_2 = await scenario_2(ctx, wrapper)
    await asyncio.sleep(PACE_SECONDS)
    failures_3 = await scenario_3(ctx)
    return failures_1, failures_2, failures_3


def main() -> int:
    config.get_gemini_api_key()  # fail fast with one sentence; never printed
    set_tracing_disabled(True)  # FR-13 wires tracing in the lifecycle task
    print("=== FR-3 live verification: fast path (1 model call) vs Desk loop ===")
    failures_1, failures_2, failures_3 = asyncio.run(run_all())

    print("\n=== what the fast path gives up ===")
    print(f"  {FR3_TRADEOFF}")

    print("\n=== summary ===")
    for scenario, failed in (
        ("scenario 1 (kettle price -> fast path, 1 call)", failures_1),
        ("scenario 2 (fan stock+price -> fast path, 1 call)", failures_2),
        ("scenario 3 (order intent -> desk, ordinary loop)", failures_3),
    ):
        print(f"  {scenario}: {'FAIL' if failed else 'PASS'}")
    total_failures = len(failures_1) + len(failures_2) + len(failures_3)
    if total_failures:
        print(f"  FR-3: FAIL ({total_failures} failed check(s))")
        return 1
    print("  FR-3: PASS — plain price/stock questions cost exactly one model call; "
          "ordinary questions keep the normal loop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
