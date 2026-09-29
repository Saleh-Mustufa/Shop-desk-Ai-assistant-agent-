"""LIVE FR-12 demo: a >= 10-turn Chainlit-equivalent conversation against the
real Gemini endpoint, driven through the SAME ``DeskSession`` the UI uses.

Reads GEMINI_API_KEY from .env via config (the key is NEVER printed). Runs
THIRTEEN scripted turns paced ~5s apart (15-RPM lite tier is safely within
limits), each through ``DeskSession.handle_user_message`` — the exact code
path ``app.py`` relays to Chainlit:

   1. greeting                          -> Desk
   2. kettle price                      -> fast path (one model call)
   3. TV price                          -> fast path
   4. blender stock                     -> fast path
   5. "I'll take 2 kettles"             -> basket KTL-01 x2 (Python parse)
   6. "And a blender as well."          -> basket BLD-07 x1
   7. "How much would that cost altogether?" -> Desk (nested pricing quote)
   8. "What's your delivery like?"      -> Desk (FR-4 clock wording)
   9. pedestal fan stock                -> fast path (0 stock: not sold)
  10. "forget the fan, add 2 irons"     -> basket IRN-12 x2
  11. "confirm the order"               -> Python finalize, ZERO model calls
                                           (2x KTL-01 + 1x BLD-07 + 2x IRN-12
                                            = PKR 22,500 recomputed in Python)
  12. "What did I just order?"          -> Desk, answers from the session
                                           recap (the FR-12 turn-11 memory)
  13. complaint + "speak to a human"    -> Desk -> FR-10 escalation handoff
                                           (typed EscalationReason)

Mid-demo a SECOND DeskSession is created to show session isolation: its
basket and history are empty and never touched by the demo conversation.

Then it prints the FR-12/FR-11/FR-13 evidence: per-turn model-call counts and
active models, the conversation cost line, the ONE trace file for the whole
conversation, and the typed escalation reason. Ends with a PASS/FAIL summary
asserting: >= 10 turns, order confirmed with the correct recomputed total,
the escalation fired, and one trace file written. Exits 1 on any failure.

Usage:
    python scripts/demo_conversation.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402  (loads .env; the key is never printed)

import agents_desk  # noqa: E402
import catalogue  # noqa: E402
import orders  # noqa: E402
import tracing_setup  # noqa: E402
from context import ShopContext  # noqa: E402
from session_store import DeskSession  # noqa: E402

PACE_SECONDS = 5.0

# (text, label) — labels are for the transcript; the DeskSession decides the
# routing itself (triage + confirmation), exactly as it does for the UI.
TURNS: list[tuple[str, str]] = [
    ("Hello! I'm looking for a new kettle for my kitchen.", "greeting (Desk)"),
    ("How much is the Electric kettle 1.7L?", "kettle price (fast path)"),
    ("How much is the 43-inch LED smart TV?", "TV price (fast path)"),
    ("Is the Blender 3-in-1 in stock?", "blender stock (fast path)"),
    ("I'll take 2 kettles please.", "basket: 2x KTL-01"),
    ("And a blender as well.", "basket: +1x BLD-07"),
    ("How much would that cost altogether?", "basket quote (Desk + pricing)"),
    ("What's your delivery like?", "delivery (FR-4 clock wording)"),
    ("Do you have the pedestal fan in stock?", "fan stock (0: not sold)"),
    ("OK, forget the fan. Add 2 irons please.", "basket: +2x IRN-12"),
    ("Great — confirm the order.", "confirm (Python, zero model calls)"),
    ("What did I just order?", "turn-11+ memory (Desk, from the recap)"),
    (
        "I want to file a complaint about my internet provider and speak to a human.",
        "escalation handoff (FR-10, typed reason)",
    ),
]


def _check(failures: list[str], label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"  check: {label} {status}"
    if detail:
        line += f" ({detail})"
    print(line)
    if not ok:
        failures.append(label)


async def run_demo() -> tuple[list[str], DeskSession, dict, dict[int, str]]:
    """The whole conversation inside ONE trace (FR-13)."""
    failures: list[str] = []
    session_id = f"demo-{time.strftime('%Y%m%d-%H%M%S')}"
    data = catalogue.load_catalogue()
    customer = ShopContext(
        shop=str(data.get("shop") or "Al-Noor Electronics"),
        currency=str(data.get("currency") or "PKR"),
        customer_id=f"guest-{session_id[:8]}",
        tier="walk_in",
    )
    session = DeskSession(session_id, customer)

    # FR-12 isolation evidence: a second session mid-demo shares nothing.
    second = DeskSession(f"{session_id}-second", customer)
    print(f"  session      : {session.session_id}")
    print(f"  customer     : {customer.customer_id} (tier={customer.tier})")
    print(f"  isolation    : second session created now, basket={second.basket} history={len(second.history)} msg(s)")

    recaps: dict[int, str] = {}
    with tracing_setup.conversation_trace(session_id) as live_trace:
        for turn_no, (text, label) in enumerate(TURNS, start=1):
            print(f"\n--- turn {turn_no} [{label}] ---")
            print(f"  customer: {text}")
            before = len(session.ledger.records)
            try:
                reply = await session.handle_user_message(text)
            except Exception as exc:  # noqa: BLE001 — the demo must not crash mid-run
                failures.append(f"turn {turn_no} raised {type(exc).__name__}: {exc}")
                print(f"  ERROR  : {type(exc).__name__}: {exc}")
                continue
            print(f"  route  : {session.last_route}")
            print(f"  reply  : {reply}")
            added = session.ledger.records[before:]
            print(
                f"  models : {len(added)} model call(s) -> "
                + (
                    ", ".join(f"{r.agent_name}/{r.model_name} ({r.kind}, {r.total_tokens} tok)" for r in added)
                    if added
                    else "none (Python truth)"
                )
            )
            if session.last_recap:
                recaps[turn_no] = session.last_recap
                print(f"  recap  : {session.last_recap}")
            if turn_no == 5:
                # Mid-conversation isolation proof: the second session is
                # still empty after five real turns of the first one.
                print(
                    f"  isolation: second session after 5 turns -> basket={second.basket}, "
                    f"history={len(second.history)} msg(s)"
                )
            if turn_no < len(TURNS):
                await asyncio.sleep(PACE_SECONDS)
        trace_id = live_trace.trace_id
        group_id = live_trace.group_id

    summary = tracing_setup.get_trace_processor().last_trace_summary() or {}
    if summary.get("trace_id") != trace_id:
        failures.append("the conversation's trace summary is missing")
    _check(
        failures,
        "second DeskSession stayed isolated (empty basket/history)",
        second.basket == {} and len(second.history) == 0,
    )
    return failures, session, summary, recaps


def main() -> int:
    config.get_gemini_api_key()  # fail fast with one sentence; never printed
    print("=== FR-12 live demo: >= 10-turn conversation, one order, one escalation ===")
    processor = tracing_setup.setup_tracing()
    print(f"  tracing mode : {processor.output_dir.resolve()}")

    failures, session, summary, recaps = asyncio.run(run_demo())

    # --- per-turn model-call counts + the FR-11 cost line -------------------
    print("\n=== FR-11 cost line (per-conversation, real usage) ===")
    for record in session.ledger.records:
        print(
            f"  call {record.seq:>2}: {record.agent_name:<17} {record.model_name:<26} "
            f"kind={record.kind:<9} tokens={record.input_tokens}/{record.output_tokens}/{record.total_tokens}"
        )
    print(f"  {session.cost_line()}")

    # --- FR-12 evidence ------------------------------------------------------
    print("\n=== FR-12 evidence ===")
    turns = session.turn_counter
    print(f"  turns handled          : {turns}")
    print(f"  history after trimming : {len(session.history)} message(s) (cap 12, order context protected)")
    memory_turn = max(recaps) if recaps else 0
    print(f"  last basket recap used : turn {memory_turn}: {recaps.get(memory_turn, '-')}")
    print(
        "  trim rule              : keep last 12 messages; drop oldest tool-free Q&A first "
        "(re-fetchable from the catalogue); pending-order messages and the last 2 exchanges never dropped"
    )

    # --- the typed escalation reason (FR-10) --------------------------------
    print("\n=== FR-10 escalation ===")
    reason = agents_desk.last_escalation_reason()
    if reason is not None:
        print(f"  typed reason : {reason.reason}")
        print(f"  details      : {reason.details}")
    else:
        print("  typed reason : <none captured>")

    # --- FR-13 trace ----------------------------------------------------------
    print("\n=== FR-13 trace (one conversation = one trace) ===")
    trace_path: Path | None = None
    if summary:
        trace_path = Path(summary["file"])
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        print(f"  trace file    : {trace_path}")
        print(f"  trace id      : {payload['trace_id']}")
        print(f"  group id      : {payload['group_id']}")
        print(f"  spans         : {payload['n_spans']}")
    else:
        print("  no trace summary available")

    # --- PASS/FAIL summary ----------------------------------------------------
    print("\n=== PASS/FAIL summary ===")
    order = session.draft_order
    data = catalogue.load_catalogue()
    prices = {str(p.get("sku")): float(p.get("price", 0)) for p in data.get("products", []) if isinstance(p, dict)}
    expected_total = 2 * prices["KTL-01"] + 1 * prices["BLD-07"] + 2 * prices["IRN-12"]

    _check(failures, "at least 10 turns handled", turns >= 10, f"turns: {turns}")
    _check(
        failures,
        "order confirmed with the Python-recomputed total (2x KTL-01 + 1x BLD-07 + 2x IRN-12)",
        order is not None
        and order.status == "confirmed"
        and orders.recompute_total(order) == order.total == expected_total,
        f"order={order.order_id if order else None} status={order.status if order else None} total={order.total if order else None} (expected {expected_total})",
    )
    _check(
        failures,
        "escalation fired with a typed reason",
        reason is not None and reason.reason in {
            "out_of_scope",
            "order_problem",
            "customer_request",
            "policy",
            "repeated_failure",
        },
        f"reason={getattr(reason, 'reason', None)}",
    )
    _check(
        failures,
        "one trace file written for the conversation (span floor: >= 2 x turns)",
        bool(summary)
        and trace_path is not None
        and trace_path.exists()
        and str(summary.get("group_id", "")).startswith("demo-")
        and int(summary.get("n_spans", 0) or 0) >= 2 * turns,
        f"group_id={summary.get('group_id')} n_spans={summary.get('n_spans')} "
        f"(floor {2 * turns}; a near-empty trace file must FAIL)",
    )
    _check(
        failures,
        "turn 11+ still remembers the order (session recap carried the basket)",
        memory_turn >= 11 and "KTL-01" in recaps.get(memory_turn, ""),
        f"recap at turn {memory_turn}",
    )

    if failures:
        print(f"  FR-12 demo: FAIL ({len(failures)} failed check(s))")
        return 1
    print(
        "  FR-12 demo: PASS — "
        f"{turns} turns, order {order.order_id} confirmed at {order.total:.0f} PKR, "
        f"escalation typed as '{reason.reason}', one trace at:"
    )
    print(f"  {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
