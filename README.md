# Shop Desk — AI customer assistant for Al-Noor Electronics

OpenAI Agents SDK (`openai-agents==0.22.3`) + Gemini (via its OpenAI-compatible
endpoint, behind the model router in `model_config.py`) + Chainlit UI.
Requirements and traceability live in `SPEC.md`; the architecture plan in
`tasks/plan.md`.

## Quickstart

```bash
pip install -r requirements.txt          # pinned, verified versions
# put GEMINI_API_KEY in .env (never committed; startup fails fast without it)

chainlit run app.py -w                    # the UI (http://localhost:8000)
python scripts/demo_conversation.py       # FR-12 live demo (>=10 turns, order, escalation)
python -m pytest tests/ -v                # offline tests (no API key needed)
python scripts/verify_router.py --list    # re-verify the model registry
```

## The UI and per-session state (FR-12)

`app.py` is a thin Chainlit layer over `session_store.DeskSession` — the SAME
class `scripts/demo_conversation.py` drives, so the demo exercises the exact
code path the UI uses. Each browser session gets its own `DeskSession`
(stored in Chainlit's `cl.user_session`), which owns:

- the message **history**,
- the **basket** (SKU -> quantity, accumulated only from explicit customer
  statements parsed deterministically in Python — see `parse_basket_statement`),
- the **confirmed order** (a typed pydantic `Order`, finalized in Python with
  catalogue prices and a Python-recomputed total — zero model calls on the
  confirmation turn),
- the **cost ledger** and the **model-call budget** (FR-11).

Session isolation: two browser windows create two sessions — two baskets,
two histories, two ledgers. Nothing session-scoped is shared.

### History trimming rule (FR-12)

> Keep at most **12 messages** (6 exchanges) per session. When the history
> grows past that, drop the **oldest messages first**, but never drop (a) any
> message of the *current pending-order discussion* (every message appended
> while the basket is non-empty, up to the confirmation — or an explicit
> basket clear — that empties it) and never drop (b) the **last 2 exchanges**.
> If the protections alone exceed the cap, the history keeps growing —
> protections win.

**What the model actually sees.** The trimmed history is not bookkeeping:
it IS the Desk agent's per-turn input. Desk turns send the session's trimmed
history to the runner as role/content items (`{"role": "user"|"assistant",
"content": str}`) with the current user message last and a short
`[Session note]` basket recap prepended to *that* message's content — so a
dropped turn is one the model literally no longer sees, while the order
discussion and the last 2 exchanges are always still in its input. Fast-path
turns deliberately stay single-shot (raw question only, no history — a
self-contained one-call catalogue lookup).

**Justification.** Plain Q&A is re-derivable from the catalogue cheaply (every
turn's tools re-fetch catalogue truth), so the oldest tool-free Q&A is the
safest thing to lose. Order context is NOT re-derivable from the catalogue
(the basket lives in session state), so it is protected while the order is
pending. On top of the protected history, the basket recap is re-stated on
the current turn of every desk call — that is what makes turn 11 remember the
order under discussion even in a fresh context window.

The recap is conversation INPUT, not prompt text, and carries SKUs and
quantities only: the `ShopContext` (customer id, tier) still travels
exclusively as the run context and is read by tools (FR-2).

### Basket rule (and its limits)

Basket lines are added/removed by deterministic regex against the catalogue:
`N <name>`, `N of <name>`, `<name> x N`, `N x <name/SKU>` (digits and
`a/an/one..twelve`), plus `remove/forget <name>` and `clear the basket`.
Known limits (demo-grade parser, documented in `session_store.py`): a
quantity + product *question* ("how much for 2 kettles?") also updates the
basket; a removal drops the product's whole basket line; repeated statements
accumulate. The Python-rendered confirmation (exact lines + recomputed total)
is the human-visible safety net.

### Confirmation flow (FR-5)

When the message matches confirm / yes place / place the order / checkout /
finalize AND the basket is non-empty, `DeskSession` calls
`agents_desk.finalize_order` directly — pure Python: catalogue unit prices,
recomputed total, full `orders.validate_order` — with ZERO model calls. The
confirmation message is rendered by Python from the validated Order, so it
contains only catalogue-true figures. On problems (e.g. over-stock) the
problems are reported politely, the order stays `draft` and the basket is
kept — never silently accepted.

## Cost line and tracing (FR-11, FR-13)

Every turn runs through `runner_cost.run_desk_turn`, which records real usage
into the session's ledger (`session.cost_line()`) and enforces the 40
model-call conversation ceiling (`SHOP_TURN_CEILING`). The UI logs the cost
line at conversation end and every 10 turns. One conversation is one trace:
`tracing_setup.conversation_trace` scopes it, and the local
`JsonlTraceProcessor` persists `traces/<trace_id>.json` (one file per trace).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — (required) | Gemini key in `.env`; NFR-1 fail fast |
| `SHOP_DEFAULT_PROFILE` | `fast` | FR-1 global default profile |
| `PRIORITY_MODEL` | unset | Model tried first in every chain |
| `SHOP_TURN_CEILING` | `40` | Per-conversation model-call ceiling |
| `SHOP_DEFAULT_TIER` | `walk_in` | UI demo default customer tier (`walk_in`/`regular`, FR-7) |
| `OPENAI_API_KEY` | unset | Optional: traces also upload to OpenAI's platform |

## Live verification scripts (`scripts/`)

`verify_router.py`, `verify_shopcore.py`, `verify_fastpath.py`,
`verify_truth.py`, `verify_agents.py`, `verify_lifecycle.py` and
`demo_conversation.py` each demonstrate one FR against the real endpoint;
`demo_conversation.py` is the FR-12 end-to-end demo (13 scripted turns: fast
path, basket, Python confirmation at PKR 22,500, turn-11 memory, typed
escalation — with PASS/FAIL checks and exit code).
