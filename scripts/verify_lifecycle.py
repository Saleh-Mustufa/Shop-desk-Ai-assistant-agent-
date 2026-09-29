"""LIVE FR-11 + FR-13 verification: the cost line story and one-conversation
tracing, end to end against the real Gemini endpoint.

Reads GEMINI_API_KEY from .env via config (the key is NEVER printed). Runs ONE
conversation — a single ``conversation_trace`` scope (FR-13) — with three
turns paced ~5s apart (15-RPM lite tier is safely within limits):

  1. fast-path turn : triage.classify("What does the kettle cost?") routes to
     the FastPath agent -> exactly ONE model call; the reply is the lookup
     tool's fixed template sentence.
  2. desk turn      : "Can you tell me about the microwave and the blender?"
     via the Desk agent's ordinary loop -> several model calls incl. tools.
  3. reasoning turn : "How much would 6 kettles cost?" via the Desk agent ->
     the get_price_quote tool runs the PricingSpecialist as a nested agent
     run (the ContextVar accounting path), landing a "reasoning" record in
     the SAME ledger.

Then it prints and checks the FR-11 deliverables: ledger.cost_line() (must
distinguish fast-path / desk / reasoning turns with real usage tokens) and
the FR-13 deliverables: exactly ONE trace file (traces/<trace_id>.json) whose
model spans carry the chain entries that actually served each profile — the
fast profile's flash-lite entry AND the reasoning profile's entry — with the
MOST EXPENSIVE turn (the model span with the highest total_tokens) called out.

Ends with a PASS/FAIL summary; exits 1 on any failure.

Usage:
    python scripts/verify_lifecycle.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402  (loads .env; the key is never printed)
from agents import RunContextWrapper  # noqa: E402

import agents_desk  # noqa: E402
import tools  # noqa: E402
import tracing_setup  # noqa: E402
import triage  # noqa: E402
from context import ShopContext  # noqa: E402
from runner_cost import (  # noqa: E402
    ConversationBudget,
    ConversationLedger,
    run_desk_turn,
)

PACE_SECONDS = 5.0

FAST_TURN = "What does the kettle cost?"
DESK_TURN = "Can you tell me about the microwave and the blender?"
REASONING_TURN = "How much would 6 kettles cost?"


def _check(failures: list[str], label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"  check: {label} {status}"
    if detail:
        line += f" ({detail})"
    print(line)
    if not ok:
        failures.append(label)


def _model_spans(payload: dict) -> list[dict]:
    """The trace's model spans (the SDK also aggregates usage onto task/turn
    spans; those are excluded so tokens are never double counted)."""
    return [span for span in payload.get("spans", []) if isinstance(span.get("model"), str)]


async def run_conversation() -> tuple[list[str], ConversationLedger, ConversationBudget, dict]:
    """The whole conversation inside ONE trace (FR-13). Returns the ledger,
    budget and the trace summary (empty dict when the trace never completed)."""
    failures: list[str] = []
    ledger = ConversationLedger()
    budget = ConversationBudget()
    ctx = ShopContext(shop="Al-Noor Electronics", currency="PKR", customer_id="CUST-LIVE", tier="walk_in")
    wrapper = RunContextWrapper(context=ctx)

    fast_agent = agents_desk.get_fastpath_agent()
    desk_agent = agents_desk.get_desk_agent()
    pricing = agents_desk.get_pricing_specialist()  # resolved at run time by the quote tool

    session_id = f"verify-lifecycle-{time.strftime('%Y%m%d-%H%M%S')}"
    summary: dict = {}
    with tracing_setup.conversation_trace(session_id) as live_trace:
        # --- turn 1: the fast path (exactly one model call) -----------------
        print(f"\n--- turn 1 (fast path): {FAST_TURN!r} ---")
        decision = triage.classify(FAST_TURN)
        print(f"  triage: {'fast path' if decision.is_fast_path else 'desk'} | {decision.reason}")
        output_1 = await run_desk_turn(fast_agent, FAST_TURN, ctx, ledger, budget)
        print(f"  reply : {output_1!r}")
        expected_by_sku = tools._lookup_product_impl(wrapper, "KTL-01")
        expected_by_name = tools._check_stock_by_name_impl(wrapper, "kettle")
        _check(
            failures,
            "turn 1 triaged to the fast path (kettle)",
            decision.is_fast_path and decision.sku == "KTL-01",
        )
        _check(
            failures,
            "turn 1 reply is the lookup tool's template sentence",
            output_1 in (expected_by_sku, expected_by_name),
        )
        _check(
            failures,
            "turn 1 cost exactly ONE model call of kind 'fast'",
            len(ledger.records) == 1 and ledger.records[0].kind == "fast",
            f"records so far: {[(r.kind, r.model_name, r.total_tokens) for r in ledger.records]}",
        )
        _check(
            failures,
            "turn 1 tokens are the run's real usage (non-zero)",
            ledger.records and ledger.records[0].total_tokens > 0,
        )

        await asyncio.sleep(PACE_SECONDS)

        # --- turn 2: the Desk's ordinary loop (several calls incl. tools) ---
        print(f"\n--- turn 2 (desk loop): {DESK_TURN!r} ---")
        output_2 = await run_desk_turn(desk_agent, DESK_TURN, ctx, ledger, budget)
        print(f"  reply : {output_2!r}")
        desk_calls = len(ledger.records) - 1
        _check(
            failures,
            "turn 2 used the ordinary loop (more than one model call)",
            desk_calls >= 2,
            f"desk calls: {desk_calls}",
        )
        _check(
            failures,
            "turn 2 reply is non-empty and customer-ready",
            isinstance(output_2, str) and bool(output_2.strip()),
        )

        await asyncio.sleep(PACE_SECONDS)

        # --- turn 3: the nested reasoning (quote) call -----------------------
        print(f"\n--- turn 3 (nested quote): {REASONING_TURN!r} ---")
        output_3 = await run_desk_turn(desk_agent, REASONING_TURN, ctx, ledger, budget)
        print(f"  reply : {output_3!r}")
        reasoning_records = [record for record in ledger.records if record.kind == "reasoning"]
        _check(
            failures,
            "turn 3 triggered the nested PricingSpecialist quote (kind 'reasoning')",
            bool(reasoning_records),
            f"records: {[(r.kind, r.model_name, r.total_tokens) for r in ledger.records]}",
        )
        expected_quote = 6 * 4200  # catalogue truth for 6 x KTL-01
        quoted = f"{expected_quote:,}" in str(output_3) or str(expected_quote) in str(output_3)
        _check(
            failures,
            "turn 3 quote matches catalogue truth (6 x 4200)",
            quoted,
            f"reply: {output_3!r}",
        )

        trace_id = live_trace.trace_id
        group_id = live_trace.group_id

    summary = tracing_setup.get_trace_processor().last_trace_summary() or {}
    if summary.get("trace_id") != trace_id:
        failures.append("the conversation's trace summary is missing")
    return failures, ledger, budget, summary


def main() -> int:
    config.get_gemini_api_key()  # fail fast with one sentence; never printed
    print("=== FR-11 + FR-13 live verification: cost line + one conversation = one trace ===")
    processor = tracing_setup.setup_tracing()
    mode = (
        "jsonl + OpenAI upload (OPENAI_API_KEY set)"
        if os.environ.get("OPENAI_API_KEY", "").strip()
        else "jsonl only (no OPENAI_API_KEY; default OpenAI exporter replaced)"
    )
    print(f"  tracing mode : {mode}")
    print(f"  output dir   : {processor.output_dir.resolve()}")

    failures, ledger, budget, summary = asyncio.run(run_conversation())

    # --- the FR-11 cost line -------------------------------------------------
    print("\n=== FR-11 cost line (per-conversation, real usage from the runs) ===")
    print(f"  {ledger.cost_line()}")

    # --- the FR-13 trace file ------------------------------------------------
    print("\n=== FR-13 trace (one conversation = one trace) ===")
    if not summary:
        print("  no trace summary available; cannot inspect the trace file")
        print("\n=== summary ===")
        print(f"  FR-11/FR-13: FAIL ({len(failures)} failed check(s))")
        return 1
    trace_path = Path(summary["file"])
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    print(f"  trace file    : {trace_path}")
    print(f"  trace id      : {payload['trace_id']}")
    print(f"  workflow name : {payload['workflow_name']}")
    print(f"  group id      : {payload['group_id']}")
    print(f"  spans         : {payload['n_spans']}")
    for span in payload["spans"]:
        if isinstance(span.get("model"), str):
            usage = span.get("usage") or {}
            print(
                f"    model span  : model={span['model']} "
                f"tokens={usage.get('input_tokens', 0)}/{usage.get('output_tokens', 0)}"
                f"/{usage.get('total_tokens', 0)} (in/out/total)"
            )

    model_spans = _model_spans(payload)
    if model_spans:
        expensive = max(model_spans, key=lambda span: int((span.get("usage") or {}).get("total_tokens", 0) or 0))
        usage = expensive.get("usage") or {}
        print(
            "  MOST EXPENSIVE TURN: "
            f"model={expensive['model']} total_tokens={usage.get('total_tokens', 0)} "
            f"(started {expensive.get('started_at')})"
        )

    # which chain entry served each profile (from RoutedModel.active_model_name)
    fast_agent = agents_desk.get_fastpath_agent()
    desk_agent = agents_desk.get_desk_agent()
    pricing = agents_desk.get_pricing_specialist()
    fast_model = fast_agent.model.active_model_name
    desk_model = desk_agent.model.active_model_name
    reasoning_model = pricing.model.active_model_name
    print(f"  fast-path turn served by : {fast_model} (profile fast, chain head)")
    print(f"  desk turns served by     : {desk_model} (profile fast, chain head)")
    print(f"  nested quote served by   : {reasoning_model} (profile reasoning, chain head)")

    trace_models = {span["model"] for span in model_spans}
    trace_total = sum(int((span.get("usage") or {}).get("total_tokens", 0) or 0) for span in model_spans)
    _check(
        failures,
        "exactly one trace file was written for the conversation",
        trace_path.exists() and list(trace_path.parent.glob(f"{payload['trace_id']}.json")) == [trace_path],
    )
    _check(
        failures,
        "the trace carries the whole conversation (workflow name, group id, spans)",
        payload["workflow_name"] == tracing_setup.WORKFLOW_NAME
        and str(payload["group_id"]).startswith("verify-lifecycle-")
        and payload["n_spans"] > 0,
    )
    _check(
        failures,
        "trace spans include BOTH the fast-profile model and the reasoning-profile model",
        fast_model in trace_models and reasoning_model in trace_models,
        f"models in trace: {sorted(trace_models)}",
    )
    _check(
        failures,
        "trace model-span tokens match the ledger's real usage (FR-11 numbers from the run)",
        trace_total == ledger.total_tokens,
        f"trace: {trace_total} vs ledger: {ledger.total_tokens}",
    )
    _check(
        failures,
        "cost line distinguishes fast-path / desk / reasoning turns",
        "fast-path: 1" in ledger.cost_line()
        and "desk: 0" not in ledger.cost_line()
        and "reasoning: 0" not in ledger.cost_line(),
        ledger.cost_line(),
    )

    print("\n=== summary ===")
    for scenario, failed in (
        ("turn 1 (fast path, one model call)", [f for f in failures if f.startswith("turn 1")]),
        ("turn 2 (desk ordinary loop)", [f for f in failures if f.startswith("turn 2")]),
        ("turn 3 (nested reasoning quote)", [f for f in failures if f.startswith("turn 3")]),
        ("FR-11 cost line", [f for f in failures if "cost line" in f or "tokens" in f]),
        ("FR-13 one conversation = one trace", [f for f in failures if "trace" in f]),
    ):
        print(f"  {scenario}: {'FAIL' if failed else 'PASS'}")
    if failures:
        print(f"  FR-11/FR-13: FAIL ({len(failures)} failed check(s))")
        return 1
    print(
        "  FR-11/FR-13: PASS — the ledger's cost line distinguishes turn kinds with "
        "real usage, and the whole conversation is one trace persisted at:"
    )
    print(f"  {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
