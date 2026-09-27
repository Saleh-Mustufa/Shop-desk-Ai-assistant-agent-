# Spec: Shop Desk — AI Customer Assistant for Al-Noor Electronics

Status: **approved baseline** (P0). Source of truth order: /goal prompt → this spec → `shop-desk-project-guide.pdf`.
Every deviation from the PDF is listed in §9. Tasks live in `tasks/plan.md` + `tasks/todo.md`.

---

## Objective

Build a customer-facing assistant for **Al-Noor Electronics** that:

1. answers stock and price questions **only** from `catalogue.json` (never invents a price or SKU),
2. builds a **typed pydantic Order** when the customer confirms, with the total recomputed in Python,
3. hands **genuinely stuck** conversations to a human via a typed escalation handoff with a filtered history,
4. does all of this **cheaply**: plain questions are served by the cheapest capable model, and every model
   override is a deliberate, documented decision.

Theme: *what each turn costs and how a long conversation stays affordable.*

**Stack:** OpenAI Agents SDK (Python, `openai-agents==0.22.3`) + Gemini models (via Gemini's
OpenAI-compatible endpoint) + Chainlit UI.

### Capability map (scope check)

One product, five modules — the 13 FRs are coupled through a shared router, shared run context and one
conversation lifecycle, so this is a single spec, not per-module specs.

| Module id | Responsibility | Depends on |
|---|---|---|
| `router` | §7 model registry, profiles, automatic limit-switching, model resolution for all three FR-1 levels | — |
| `shopcore` | catalogue access, ShopContext, tools, dynamic prompts | router |
| `agents` | fast path, Desk, order taker, pricing specialist, escalation; guardrail; orders | router, shopcore |
| `lifecycle` | custom runner, run hooks, cost line, tracing | router, agents |
| `ui` | Chainlit app, per-session state, history trimming, demo scripts | agents, lifecycle |

Build order: `router` → `shopcore` → `agents` → `lifecycle` → `ui`.

---

## Requirements traceability (FR-1 … FR-13, NFR-1 … NFR-5)

| Req | Requirement (one line) | Realization |
|---|---|---|
| FR-1 | Catalogue only via tools; 3 model-config levels | `tools.py` (only catalogue reader); all levels resolve through `model_config.py` (see §4 below) |
| FR-2 | ShopContext to every run, read by tools, never in prompt text | `context.py`; `function_tool(...)` with explicit params — wrapper never appears in tool schemas; `prompts.py` builds text from ShopContext *fields*, never the id (grep-enforced) |
| FR-3 | Fast path: plain price/stock question = one model call | Deterministic triage (`triage.py`) routes plain lookups to a fast-path agent with `tool_use_behavior="stop_on_first_tool"` — the lookup tool's output **is** the answer; the model never sees the result |
| FR-4 | Dynamic per-turn instructions from context + clock | `prompts.py`: outside shop hours no same-day delivery promise + opening time stated; resolved prompt printable before any model call |
| FR-5 | Confirmed order = typed pydantic Order; total recomputed | `orders.py`: `LineItem`/`Order`; order-taker agent emits structured `Order`; Python recomputes total, mismatch reported, never silently accepted |
| FR-6 | Output guardrail vs catalogue.json; refusal → polite message | `guardrails.py`: data check — every price/SKU in the final answer must exist in `catalogue.json` **this run**; out-of-stock items never sold; tripwire → polite message |
| FR-7 | Statically disabled seasonal tool + tier-gated tool | `tools.py`: `holiday_bundles` with `is_enabled=False` (invisible in every schema); `loyalty_benefit` offered only when `ctx.context.tier == "regular"` |
| FR-8 | Pricing specialist as tool returning a number; turn ceiling | `pricing_specialist.as_tool()` → returns a figure; per-conversation model-call ceiling (40) raised in run hooks, caught by custom runner, ends politely |
| FR-9 | Pricing & Escalation cloned from one base | `agents_desk.py`: `SpecialistBase.clone()`; `is`-comparisons shown in verification; clones never restate inherited model |
| FR-10 | Escalation handoff: typed reason + filtered history | `handoff()` with `input_type=EscalationReason` (Literal reason codes); custom input filter removes tool-call noise, keeps user turns + desk replies; before/after shown |
| FR-11 | Per-conversation cost line from run context | `runner_cost.py`: `RunHooks` record per-turn model + usage (real usage, not estimates); cost line distinguishes fast-path from reasoning turns |
| FR-12 | Chainlit ≥10 turns, per-session history, justified trimming | `app.py` + `session_store.py`: per-session basket/history (Chainlit sessions are isolated), trim rule: keep last 12 messages, drop oldest tool-free Q&A first, never drop active order context |
| FR-13 | One conversation = one trace | `tracing_setup.py`: `trace(workflow_name="shop-desk-conversation", trace_id=<per conversation>, group_id=<session>)`; custom `TracingProcessor` persists each trace as one JSONL file; if `OPENAI_API_KEY` is set the same trace also uploads to OpenAI's platform |
| NFR-1 | Secrets only in `.env`, gitignored, fail fast with one sentence | `.env` (never committed), `config.py` validation at startup |
| NFR-2 | Cheap model default; every override justified | Fast profile default (Desk, fast path, escalation); reasoning profile only for pricing + re-quote — justifications in §5 |
| NFR-3 | Truth from catalogue only | FR-6 guardrail + order flow stock checks + tools read catalogue this run |
| NFR-4 | Tools never raise into the runner | All tool bodies catch exceptions and return a sentence; SDK `failure_error_function` as the outer net |
| NFR-5 | Spec precedes code in git history | This commit (`P0: specification — no code`) precedes all `feat:` commits |

---

## The three configuration levels (FR-1) — all through the router

All model identity and model logic lives **only** in `model_config.py` (§7 of the goal):

1. **Global default (profile `fast`)** — env `SHOP_DEFAULT_PROFILE` (default `fast`) resolved by the router to the
   cheap model chain. Serves fast path, Desk, escalation. Delete the global default and exactly one behaviour
   changes: which model answers plain questions (router falls through to the next chain entry — logged).
2. **Agent-level override (profile `reasoning`)** — the Pricing specialist's model field resolves to the reasoning
   profile via the same router. Justified in §5.
3. **Run-level override** — the re-quote path calls `Runner.run(..., run_config=RunConfig(model=router_model("reasoning")))`
   for that single run only.

**Router mechanics (goal §7):**

- **Registry:** 15-RPM tier: `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`; 5-RPM tier: `gemini-3.8-flash`,
  `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3-flash-preview` (live-corrected from
  `gemini-3-flash`), `gemini-2.5-flash` (kept for provenance; **live verification showed it is closed to new keys**,
  so it is marked unavailable and excluded from live chains).
- **Limits:** RPM/TPM/RPD per model, configurable via env (`RPM_*`, `TPM_*`, `RPD_*`), with in-memory sliding-window
  counters; RPD exhaustion marks a model out until the daily reset.
- **Priority model:** top-level `priority_model` field, settable via `PRIORITY_MODEL` env; the fallback chain starts
  with it, then walks the registry in configured order.
- **Automatic switching:** on 429 / `RESOURCE_EXHAUSTED` / quota errors the router switches to the next chain entry —
  exponential backoff for RPM/TPM, cooldown-until-reset for RPD. Every switch is logged: model out, model in, reason.
- **Wiring:** Gemini via its OpenAI-compatible endpoint
  (`AsyncOpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=GEMINI_API_KEY)`)
  wrapped in a custom `RoutedModel` implementing the SDK `Model` protocol (`get_response`) and a
  `ModelProvider`. This is the SDK's documented custom-provider integration point. (`LitellmModel` exists as an
  alternative; the compat-endpoint route avoids the extra dependency and was smoke-tested with tool calls.)

---

## Model assignments (NFR-2 justifications)

| Agent | Profile | Why |
|---|---|---|
| Fast-path agent | fast (cheapest tier) | One-call lookup answers; no reasoning needed |
| Desk agent | fast (cheap tier) | Conversation, tool orchestration; reasoning model would multiply cost per turn |
| Order taker | fast | Transcribes an already-negotiated basket into a typed object; totals are recomputed in Python anyway |
| Pricing specialist | **reasoning** (agent-level override) | Bespoke/bulk quotes need careful arithmetic; FR-8 explicitly makes it a specialist |
| Escalation agent | fast (inherits base) | Empathy + info gathering, no arithmetic |
| Re-quote path | **reasoning** (run-level override) | Single contested-quote rerun; costs one run, not the conversation |

---

## Commands

```bash
# Env setup
pip install -r requirements.txt

# Run the UI
chainlit run app.py -w

# Unit tests (no API key needed)
python -m pytest tests/ -v

# Live verifications (need GEMINI_API_KEY in .env)
python scripts/verify_router.py        # §7 registry vs live endpoint + switching
python scripts/verify_shopcore.py      # FR-1, FR-2, FR-4, FR-7 evidence
python scripts/verify_fastpath.py      # FR-3 evidence (1-call vs N-call traces)
python scripts/verify_truth.py         # FR-5, FR-6 evidence incl. catalogue-mutation failure
python scripts/verify_agents.py        # FR-8, FR-9, FR-10 evidence (clone `is` checks, handoff before/after)
python scripts/verify_lifecycle.py     # FR-11, FR-13 evidence (cost line, one trace)
python scripts/demo_conversation.py    # FR-12: full ≥10-turn conversation, order, escalation

# Registry re-verification (after Google adds/removes models)
python scripts/verify_router.py --list
```

---

## Project structure

```
catalogue.json          → the single source of product truth (never read outside tools/guardrail)
model_config.py         → §7 router: registry, profiles, RoutedModel, switching — ALL model logic
config.py               → .env loading + startup validation (fails with one clear sentence)
context.py              → ShopContext dataclass (FR-2)
catalogue.py            → read-only catalogue accessor (tools + guardrail use this)
tools.py                → lookup_product, stock tools, tier-gated tool, disabled seasonal tool (FR-1/2/7)
prompts.py              → dynamic per-turn instruction builders (FR-4), printable
orders.py               → LineItem/Order pydantic models + Python total recompute (FR-5)
guardrails.py           → catalogue output guardrail (FR-6)
triage.py               → deterministic fast-path / confirmation routing (FR-3)
agents_desk.py          → base specialist + clones, Desk, fast-path agent, order taker, handoff (FR-3/8/9/10)
runner_cost.py          → custom runner + run hooks + turn ceiling + cost line (FR-8/11)
tracing_setup.py        → trace-per-conversation + local JSONL trace processor (FR-13)
session_store.py        → per-session state: history, basket, cost ledger, trimming rule (FR-12)
app.py                  → Chainlit UI (FR-12)
scripts/                → live verification + demo scripts
tests/                  → offline pytest units (router, guardrail, orders, prompts, trimming)
```

---

## Code style

Type hints everywhere; docstrings on public functions stating the requirement they serve; `logging` (no prints in
library code — prints only in `scripts/` and the cost line output). Tool bodies return sentences, never raise.

```python
@function_tool
def lookup_product(ctx: RunContextWrapper[ShopContext], sku: str) -> str:
    """Look up price and stock for one SKU from the catalogue.

    Returns a customer-ready sentence; never raises (NFR-4).
    """
    try:
        item = catalogue().get(sku)
        if item is None:
            return f"I couldn't find product code {sku!r} in our catalogue."
        ...
    except Exception as exc:  # noqa: BLE001 — tools must not raise into the runner (NFR-4)
        return "Sorry, I had trouble checking the catalogue just now. Please try again."
```

Conventional commits (`spec:`, `feat:`, `fix:`, `docs:`, `chore:`); one logical change per commit; the P0 commit is
exactly `P0: specification — no code` (goal §5).

---

## Testing strategy

- **Offline pytest** (`tests/`, no key, CI-safe): guardrail scanner (price/SKU extraction vs catalogue, mutation
  test with a temp catalogue), order total recompute + mismatch reporting, prompt resolution at two simulated hours,
  tool schemas (no `ShopContext` wrapper param; seasonal tool absent; tier gating), router state machine (simulated
  429/RPM/TPM/RPD with fake models; switch logging; priority_model first), session trimming rule, cost-line
  aggregation from fake usage.
- **Live verification scripts** (`scripts/`, real key): each FR demonstrated on demand, traces captured as evidence.
- **Chainlit** verified by headless boot + the demo script exercising the same `DeskSession` orchestrator the UI uses.

---

## Boundaries

**Always**
- Run `python -m pytest tests/ -v` before every commit.
- Read product truth from `catalogue.json` this run — tools and guardrail only.
- Return sentences from tools; log every router switch (model out, model in, reason).
- Grep for the API key before every push (`git grep` on the committed tree).

**Ask first**
- Adding a dependency.
- Changing `catalogue.json` schema.
- Any deviation not already recorded in §9.

**Never**
- Commit secrets; model name or model logic outside `model_config.py`.
- A tool that raises into the runner.
- A price/stock figure/SKU shown to a customer that didn't come from the catalogue this run.
- More than 2 concurrent subagents. Cutting FR-3 or FR-6. A commit mixing spec and implementation.

---

## Success criteria (Definition of Done)

1. `git log` shows `P0: specification — no code` before the first code commit.
2. FR-1: point at all three override levels; deleting the global default changes exactly one behaviour.
3. FR-2: tool schema has no wrapper param; `grep -ri "customer_id" prompts` finds nothing.
4. FR-3: fast-path trace shows 1 model call; order trace shows more; can name what is lost (fixed template — no
   phrasing, nuance or upsell).
5. FR-4: same question at two simulated hours yields two different delivery promises; resolved prompt printable.
6. FR-5: planted total mismatch is caught and reported, never accepted.
7. FR-6: editing a catalogue price makes a previously passing answer fail; out-of-stock never sold; refusal is polite.
8. FR-7: same question as `walk_in` vs `regular` → different tool sets; seasonal tool in no schema.
9. FR-8: Desk's reply contains the specialist's number in the Desk's wording; ceiling = 40 model calls/conversation;
   hitting it yields a polite close, not a traceback.
10. FR-9: `is` comparisons show what clones share and don't; no clone restates its inherited model.
11. FR-10: escalation reason is a typed value; before/after handoff history shown; escalation agent answers without
    tool noise.
12. FR-11: cost line printed per conversation from real usage; fast-path vs reasoning turns distinguished.
13. FR-12: ≥10-turn conversation works; turn 11 remembers the order; two sessions never share a basket; trim rule
    stated (keep 12, drop oldest tool-free Q&A first — re-fetchable from catalogue; never drop order context).
14. FR-13: one conversation = one trace; fast-path and reasoning turns identifiable; most expensive turn identifiable.
15. Final report: FR-by-FR table (requirement → file:line → demo), routing behaviour observed, total token cost.

---

## Deviations from the PDF (deliberate, per goal §2/§5)

1. **Phase-0 artifacts:** the PDF's `constitution.md` + PDF-style `spec.md`/`plan.md`/`tasks.md` are replaced by this
   spec package (`SPEC.md`, `tasks/plan.md`, `tasks/todo.md`) committed as `P0: specification — no code` — explicitly
   authorized by goal §5. The PDF's spirit (spec precedes code in git history) is preserved.
2. **Model IDs (live-corrected, goal §7.5):** `gemini-3-flash` is served only as `gemini-3-flash-preview`;
   `gemini-2.5-flash` exists in listings but is closed to new API keys (live smoke test) → registry keeps it with
   `available=False`; live chains exclude it.
3. **FR-3 mechanism:** realized with the SDK's `tool_use_behavior="stop_on_first_tool"` on a dedicated fast-path
   agent, fronted by a deterministic (zero-cost) triage function. One model call per fast-path turn; the tool's
   output is the customer-facing answer.
4. **FR-13 "under your own key":** no OpenAI platform key is provided; only a Gemini key exists. Traces are enabled
   and grouped exactly as required (one conversation = one trace) and persisted locally as one JSONL file per trace
   by a custom `TracingProcessor`. If `OPENAI_API_KEY` is set, the same trace also uploads to OpenAI's platform
   (wiring present and documented). This is "tracing under your own infrastructure" — recorded here per goal §2.
5. **FR-9 model note:** the pricing clone overrides the model (reasoning profile) — explicitly permitted by the PDF
   ("differing only in instructions and model settings"); the escalation clone overrides instructions only and
   inherits the base model. Neither clone restates the inherited model.
6. **Work directly on `main`:** the goal §8 mandates `main` as the delivery branch (subagent-driven-development's
   worktree default is overridden by explicit user instruction).

## Open questions

None blocking — all decisions resolvable from the goal prompt and this spec (goal §12).
