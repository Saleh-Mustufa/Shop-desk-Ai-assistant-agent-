# Task List — Shop Desk (spec: SPEC.md; plan: tasks/plan.md)

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
