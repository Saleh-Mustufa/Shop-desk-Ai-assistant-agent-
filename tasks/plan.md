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

### Phase 1 — Foundation
- [ ] Task 1: Scaffold + §7 model router (`model_config.py`, `config.py`, `catalogue.json`, `requirements.txt`,
      `tests/test_router.py`, `scripts/verify_router.py`)
- [ ] Task 2: Shop core — ShopContext, catalogue accessor, tools (incl. FR-7 gating), dynamic prompts
      (`context.py`, `catalogue.py`, `tools.py`, `prompts.py`, `tests/test_shopcore.py`)

### Checkpoint: foundation
- [ ] pytest green; router switching proven offline; registry verified against live endpoint

### Phase 2 — Agents and truth (T3 ∥ T4 in parallel — disjoint files)
- [ ] Task 3: Fast path + agent graph assembly (`triage.py`, `agents_desk.py` core, `scripts/verify_fastpath.py`)
- [ ] Task 4: Orders + catalogue guardrail (`orders.py`, `guardrails.py`, `tests/test_orders_guardrail.py`)

### Checkpoint: core
- [ ] Fast path = 1 model call live; guardrail mutation test fails a previously passing answer offline

### Phase 3 — Completion
- [ ] Task 5: Agent graph completion — wire guardrail + order taker + pricing-as-tool + turn ceiling into the Desk
      run (`agents_desk.py`, `runner_cost.py`, `tests/test_runner_cost.py`)
- [ ] Task 6: Escalation handoff — typed reason + filtered history + before/after evidence
      (`handoff` wiring in `agents_desk.py`, `tests/test_handoff.py`, `scripts/verify_agents.py`)  (T6 ∥ T7)
- [ ] Task 7: Cost line + tracing — run hooks ledger, per-conversation cost line, trace-per-conversation JSONL
      (`runner_cost.py` hooks, `tracing_setup.py`, `scripts/verify_lifecycle.py`)
- [ ] Task 8: Chainlit UI + sessions + trimming + demo scripts
      (`app.py`, `session_store.py`, `scripts/demo_conversation.py`, `README.md`)

### Checkpoint: complete
- [ ] All FR verifications green; demo conversation ≥10 turns with one order + one escalation

### Phase 4 — Verification & delivery
- [ ] Task 9: Whole-project verification pass (every FR one-by-one), final report, final push

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
