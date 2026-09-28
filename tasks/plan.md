# Implementation Plan: Shop Desk

## Overview

Implement the approved spec (`SPEC.md`): an OpenAI Agents SDK + Gemini + Chainlit customer assistant for
Al-Noor Electronics, organized as five modules (`router` → `shopcore` → `agents` → `lifecycle` → `ui`) executed
via subagent-driven development (max 2 concurrent subagents, per goal §4/§12).

## Architecture decisions

- **Model router (`model_config.py`) is the only model-aware file.** It implements the §7 registry (tiers, env-tunable
  RPM/TPM/RPD), `PRIORITY_MODEL`, profiles (`fast`/`reasoning`), and a `RoutedModel` implementing the SDK `Model`
  protocol over Gemini's OpenAI-compatible endpoint, with automatic switching (429/RESOURCE_EXHAUSTED → backoff for
  RPM/TPM, cooldown-to-reset for RPD) and switch logging. FR-1's three levels all resolve through it:
  global default profile (env), agent-level override (pricing specialist), run-level override (`RunConfig.model` on
  the re-quote path).
- **Fast path (FR-3):** deterministic triage in Python (length, order-intent keywords, single-product fuzzy match)
  routes plain price/stock questions to a dedicated agent with `tool_use_behavior="stop_on_first_tool"` — exactly one
  model call; the lookup tool's output is the answer verbatim. Everything else takes the normal Desk loop.
- **Truth (FR-6):** one catalogue accessor; the output guardrail re-reads the catalogue at check time and validates
  every price/SKU in the finished answer; the order flow re-checks stock in Python before confirming.
- **Orders (FR-5):** on confirmed intent, an order-taker agent with `output_type=Order` produces the typed object;
  Python recomputes the total and compares; mismatch → reported to the customer, order stays draft.
- **Escalation (FR-10):** `handoff()` with typed `EscalationReason` (Literal) + custom input filter that removes
  tool-call/tool-output items, keeps user turns, desk replies and the order summary.
- **Lifecycle (FR-11/13):** `RunHooks` record per-turn (turn#, model, prompt/completion/total tokens) from run
  context into a conversation ledger; custom runner prints the cost line and enforces the 40-call ceiling (raise in
  hook → caught → polite close). One `trace(...)` per conversation; custom `TracingProcessor` writes one JSONL per
  trace; OpenAI platform upload activates if `OPENAI_API_KEY` is set.
- **UI (FR-12):** thin Chainlit layer over a `DeskSession` orchestrator (the same class the demo script drives);
  per-session state via Chainlit's `user_session`; history trimmed to the last 12 messages, oldest tool-free Q&A
  dropped first (re-fetchable from the catalogue), order context never dropped.

## Task list

## Task 1: Scaffold + §7 model router
- Acceptance: `model_config.py` holds the full registry (15-RPM tier: gemini-3.5-flash-lite, gemini-3.1-flash-lite;
  5-RPM tier: gemini-3.8-flash, gemini-3.7-flash, gemini-3.6-flash, gemini-3.5-flash, gemini-3-flash-preview,
  gemini-2.5-flash[unavailable]); env-tunable RPM/TPM/RPD; `PRIORITY_MODEL`; profiles fast/reasoning;
  `RoutedModel(Model)` over Gemini OpenAI-compat endpoint with automatic switching (exponential backoff RPM/TPM,
  cooldown-to-reset RPD) and switch logging (model out, model in, reason); `resolve()` serves all three FR-1 levels;
  no model name outside this file (enforced by test); `config.py` fails fast with one sentence on missing key;
  `catalogue.json` matches the PDF sample.
- Verify: `python -m pytest tests/test_router.py -v` (offline, fake models) + `python scripts/verify_router.py` (live).
- Files: `model_config.py`, `config.py`, `catalogue.json`, `requirements.txt`, `tests/test_router.py`,
  `scripts/verify_router.py`.

## Task 2: Shop core (FR-2, FR-4, FR-7, FR-1 tool surface)
- Acceptance: `ShopContext` dataclass (shop, currency, customer_id, tier); tools read context via wrapper and their
  generated schemas contain no wrapper param; `lookup_product` returns customer-ready sentences, never raises;
  tier-gated tool offered only to `regular`; seasonal tool statically disabled and invisible in every schema;
  dynamic instructions rebuilt per turn from context + clock: outside shop hours no same-day promise + opening time;
  resolved prompt printable before any model call; no customer id in any prompt text.
- Verify: `python -m pytest tests/test_shopcore.py -v`.
- Files: `context.py`, `catalogue.py`, `tools.py`, `prompts.py`, `tests/test_shopcore.py`.

## Task 3: Fast path + agent graph core (FR-3)
- Acceptance: deterministic triage routes plain price/stock questions to the fast-path agent
  (`tool_use_behavior="stop_on_first_tool"`); fast-path turn costs exactly one model call (trace evidence); ordinary
  questions keep the normal loop; what the fast path loses is stated in the script output.
- Verify: `python -m pytest tests/ -v` + `python scripts/verify_fastpath.py` (live).
- Files: `triage.py`, `agents_desk.py`, `scripts/verify_fastpath.py`.

## Task 4: Orders + catalogue guardrail (FR-5, FR-6)
- Acceptance: `LineItem`/`Order` pydantic models; `recompute_total` recomputes in Python and reports mismatches
  (never silent); output guardrail scans finished answers: every price/SKU must exist in catalogue.json this run;
  out-of-stock items never sold; tripwire result convertible to a polite message; mutation test: editing a catalogue
  price makes a previously passing answer fail.
- Verify: `python -m pytest tests/test_orders_guardrail.py -v`.
- Files: `orders.py`, `guardrails.py`, `tests/test_orders_guardrail.py`.

## Task 5: Agent graph completion + runner/ceiling (FR-5, FR-6 wiring, FR-8)
- Acceptance: guardrail attached as output guardrail on customer-facing agents; confirmation flow runs the order
  taker (`output_type=Order`) and reports recomputed totals; pricing specialist exposed via `as_tool` returning a
  number; per-conversation ceiling of 40 model calls raises in hooks, is caught by the custom runner, ends politely.
- Verify: `python -m pytest tests/test_runner_cost.py -v` + live re-quote path check.
- Files: `agents_desk.py`, `runner_cost.py`, `tests/test_runner_cost.py`.

## Task 6: Escalation handoff (FR-10)
- Acceptance: `handoff()` to escalation agent with typed `EscalationReason` (Literal reason + details); custom input
  filter removes tool-call/tool-output items, keeps user turns, desk replies, order summary; before/after history
  shown; escalation agent answers sensibly without tool noise.
- Verify: `python -m pytest tests/test_handoff.py -v` + `python scripts/verify_agents.py` (live; also covers FR-8/9).
- Files: `agents_desk.py`, `tests/test_handoff.py`, `scripts/verify_agents.py`.

## Task 7: Cost line + tracing (FR-11, FR-13)
- Acceptance: `RunHooks` ledger records per-turn model + real usage from run context; cost line printed per
  conversation, distinguishing fast-path from reasoning turns; one `trace()` per conversation; custom processor
  persists one JSONL per trace; OpenAI upload if `OPENAI_API_KEY` set.
- Verify: `python -m pytest tests/test_runner_cost.py -v` + `python scripts/verify_lifecycle.py` (live).
- Files: `runner_cost.py`, `tracing_setup.py`, `scripts/verify_lifecycle.py`.

## Task 8: Chainlit UI + sessions + demo (FR-12)
- Acceptance: Chainlit app drives `DeskSession` orchestrator; history per session, trimmed to 12 messages, oldest
  tool-free Q&A dropped first, order context never dropped (rule stated in code + README); two sessions never share
  a basket; demo script runs a ≥10-turn conversation with one order + one escalation.
- Verify: `python -m pytest tests/test_session.py -v` + headless Chainlit boot + `python scripts/demo_conversation.py`.
- Files: `app.py`, `session_store.py`, `scripts/demo_conversation.py`, `README.md`.

## Task 9: Whole-project verification + delivery
- Acceptance: every FR verified one-by-one (scripts + tests), FR-by-FR final report with file:line and demo commands,
  token cost of the verification run reported; git history clean (spec-first, conventional commits); all pushed.
- Verify: run every `scripts/verify_*.py` + `python -m pytest tests/ -v`.
- Files: report only (final message), possible small fixes.

## Checkpoints
- After Tasks 1–2: pytest green, router switching proven offline, registry live-verified.
- After Tasks 3–5: fast path 1 call; guardrail mutation fails a previously passing answer; order mismatch caught.
- After Tasks 6–8: demo conversation ≥10 turns with one order + one escalation; cost line + single trace.

## Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Gemini 429s during verification (5–15 RPM tiers) | High | Router auto-switching is built in; verification scripts pace calls; lite tier (15 RPM) is the default |
| Hooks raising may be swallowed by SDK | Medium | Turn ceiling verified in Task 5 with a live test; fallback: ceiling checked in the custom runner before each turn |
| `stop_on_first_tool` fallback when model replies without a tool call | Low | Fast-path prompt mandates the tool call; non-tool reply is still a valid one-call answer (clarifying question) |
| Chainlit session isolation assumptions | Medium | Demo script drives `DeskSession` directly; UI layer verified by headless boot + two-session test |
| Push auth unavailable | Low | Keep granular local commits; push retried per milestone; report at end |

## Open questions

None — see SPEC.md §Open questions.
