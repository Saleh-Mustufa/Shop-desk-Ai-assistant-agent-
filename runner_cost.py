"""Custom turn runner, per-conversation call ceiling and cost ledger (FR-8, FR-11).

This module is the ONLY path the session layer uses to run a customer turn.
It wraps ``Runner.run`` with run hooks that (a) enforce the per-conversation
ceiling of model calls and (b) record real usage numbers for FR-11's cost
reporting.

FR-11 — "numbers come from run context", precisely: token counts are read
from the ``ModelResponse.usage`` object the SDK hands to the ``on_llm_end``
run hook DURING the real run (never estimated and never reconstructed after
the fact), and model identity comes from ``agent.model.active_model_name`` —
the router model's own record of which chain candidate actually served the
call, read in the same hook. A turn that never reaches ``on_llm_end`` (for
example one aborted by the ceiling) contributes no record, so the ledger
only ever contains numbers the run itself reported.

Per-conversation ceiling (FR-11): 40 model calls per conversation by default
(overridable via the ``SHOP_TURN_CEILING`` env var). Justification: a
10-turn conversation rarely exceeds ~3 model calls per turn (the fast path
costs exactly 1; a desk turn is a handful of loop iterations), so 40 leaves
headroom while staying a hard cost stop. ``on_llm_start`` registers each
upcoming call and raises :class:`TurnBudgetExceeded` when the ceiling would
be exceeded — the SDK awaits run hooks inline, so the exception aborts the
run before the model call happens, and :func:`run_desk_turn` turns it into
the polite close.

Nested agent-tool runs count too (FR-8 ceiling, FR-11 completeness): when
the Desk calls ``get_price_quote``, the pricing specialist runs as a nested
``Runner.run`` inside the tool body. :func:`run_desk_turn` scopes the
conversation's ``(ledger, budget)`` in a module-level ``ContextVar`` for the
duration of the run; the tool body reads it and attaches
``ShopDeskHooks(ledger, budget)`` to the nested run, so a nested LLM call
registers against the SAME ceiling and lands in the SAME ledger exactly like
an outer call (every LLM start registers exactly once — nested or not — so
there is no double counting and no re-entrancy hazard). Without the scope
(direct tool tests) the nested run simply runs without hooks.

Cost figures are tokens and model-call counts only — never currency amounts
(prices belong to the catalogue, not to the cost report).
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass

from agents import (
    Agent,
    AgentsException,
    ModelBehaviorError,
    OutputGuardrailTripwireTriggered,
    RunConfig,
    RunContextWrapper,
    RunHooks,
    Runner,
    UserError,
)
from agents.items import TResponseInputItem

import guardrails
import model_config
from context import ShopContext

__all__ = [
    "BUDGET_CLOSE",
    "DEFAULT_MAX_MODEL_CALLS",
    "GENERIC_SORRY",
    "SYSTEM_BUSY",
    "ConversationBudget",
    "ConversationLedger",
    "ShopDeskHooks",
    "TurnBudgetExceeded",
    "TurnRecord",
    "conversation_accounting",
    "run_desk_turn",
    "turn_kind",
]

logger = logging.getLogger("shopdesk.runner")

BUDGET_CLOSE = (
    "I'm sorry — this conversation has reached its length limit for today. "
    "Please start a new chat; your order details are safe."
)
SYSTEM_BUSY = "We're briefly out of capacity — please try again in a moment."
GENERIC_SORRY = "Something went wrong on our side — please try again in a moment."

DEFAULT_MAX_MODEL_CALLS = 40
_CEILING_ENV = "SHOP_TURN_CEILING"


def turn_kind(agent_name: str) -> str:
    """Classify a model call by the agent that made it (FR-11 ledger)."""
    if agent_name == "FastPath":
        return "fast"
    if agent_name == "PricingSpecialist":
        return "reasoning"
    return "desk"


@dataclass
class TurnRecord:
    """One model call, recorded by the run hooks with the run's real numbers."""

    seq: int
    agent_name: str
    model_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    kind: str  # "fast" | "reasoning" | "desk" (see :func:`turn_kind`)


class TurnBudgetExceeded(UserError):
    """Raised when a conversation would exceed its model-call ceiling (one sentence)."""


def _env_ceiling() -> int:
    """Per-conversation ceiling default: env SHOP_TURN_CEILING, else 40."""
    raw = (os.environ.get(_CEILING_ENV) or "").strip()
    if not raw:
        return DEFAULT_MAX_MODEL_CALLS
    try:
        value = int(raw)
    except ValueError:
        logger.warning("ignoring non-integer %s=%r", _CEILING_ENV, raw)
        return DEFAULT_MAX_MODEL_CALLS
    if value <= 0:
        logger.warning("ignoring non-positive %s=%r", _CEILING_ENV, raw)
        return DEFAULT_MAX_MODEL_CALLS
    return value


class ConversationBudget:
    """Hard per-conversation ceiling on model calls (FR-11).

    ``register()`` is called by the run hooks immediately BEFORE each model
    call; it increments the counter and raises :class:`TurnBudgetExceeded`
    when the ceiling would be exceeded, so the run aborts before the call.
    """

    def __init__(self, max_model_calls: int | None = None) -> None:
        self.max_model_calls = (
            max_model_calls if max_model_calls is not None else _env_ceiling()
        )
        self.used = 0

    def register(self) -> None:
        """Count one upcoming model call; raise when the ceiling would be exceeded."""
        if self.used + 1 > self.max_model_calls:
            raise TurnBudgetExceeded(
                f"This conversation reached its limit of {self.max_model_calls} "
                "model calls; please start a new chat."
            )
        self.used += 1


def _model_name(agent: Agent) -> str:
    """The model that served the call: the router's active candidate when available."""
    model = getattr(agent, "model", None)
    if model is None:
        return "unknown"
    active = getattr(model, "active_model_name", None)
    if isinstance(active, str) and active:
        return active
    return str(model)


class ShopDeskHooks(RunHooks):
    """Run hooks wiring the ledger and the ceiling into every model call.

    ``on_llm_start`` registers the upcoming call in the budget — raising there
    aborts the run before the model call happens (the SDK awaits run hooks
    inline and propagates the exception). ``on_llm_end`` appends a
    :class:`TurnRecord` with the usage the SDK reports for that response and
    the agent's own model identity — real numbers, never estimates.
    """

    def __init__(self, ledger: "ConversationLedger", budget: ConversationBudget) -> None:
        self.ledger = ledger
        self.budget = budget

    async def on_llm_start(
        self,
        context: RunContextWrapper[ShopContext],
        agent: Agent[ShopContext],
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        # Raise BEFORE the model call so the ceiling is a hard stop (FR-11).
        self.budget.register()

    async def on_llm_end(
        self,
        context: RunContextWrapper[ShopContext],
        agent: Agent[ShopContext],
        response: object,
    ) -> None:
        usage = getattr(response, "usage", None)
        agent_name = str(getattr(agent, "name", "unknown") or "unknown")
        self.ledger.append(
            TurnRecord(
                seq=0,  # stamped by ConversationLedger.append
                agent_name=agent_name,
                model_name=_model_name(agent),
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
                kind=turn_kind(agent_name),
            )
        )


# Scoped conversation accounting (FR-8 ceiling, FR-11 completeness): set by
# run_desk_turn for the duration of one customer turn. Agent-tool wrappers
# (agents_desk.get_price_quote) read it and attach the SAME ledger/budget to
# nested agent runs, so nested LLM calls register against the same ceiling
# and land in the same ledger. None outside a run_desk_turn scope.
_CONVERSATION_ACCOUNTING: ContextVar[tuple[ConversationLedger, ConversationBudget] | None] = (
    ContextVar("shopdesk_conversation_accounting", default=None)
)


def conversation_accounting() -> tuple[ConversationLedger, ConversationBudget] | None:
    """The ``(ledger, budget)`` scoped to the current turn, or ``None``."""
    return _CONVERSATION_ACCOUNTING.get()


class ConversationLedger:
    """Per-conversation record of every model call plus aggregate cost lines."""

    def __init__(self) -> None:
        self.records: list[TurnRecord] = []
        self.guardrailed = False

    def append(self, record: TurnRecord) -> TurnRecord:
        """Stamp the sequence number and file one turn record."""
        record.seq = len(self.records) + 1
        self.records.append(record)
        return record

    def mark_guardrailed(self) -> None:
        """Flag that this conversation's latest run ended in a guardrail refusal."""
        self.guardrailed = True

    @property
    def total_input_tokens(self) -> int:
        return sum(record.input_tokens for record in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(record.output_tokens for record in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(record.total_tokens for record in self.records)

    def cost_line(self) -> str:
        """One-line summary: turns by kind, token totals, per-model call counts.

        Tokens and model counts only — no price figures (catalogue prices are
        the customer's business; the cost line is the operator's).
        """
        kinds = {"fast": 0, "desk": 0, "reasoning": 0}
        models: dict[str, int] = {}
        for record in self.records:
            kinds[record.kind] = kinds.get(record.kind, 0) + 1
            models[record.model_name] = models.get(record.model_name, 0) + 1
        if models:
            model_bits = ", ".join(
                f"{name} x{count}"
                for name, count in sorted(models.items(), key=lambda item: (-item[1], item[0]))
            )
        else:
            model_bits = "none"
        return (
            f"Cost line — turns: {len(self.records)} "
            f"(fast-path: {kinds['fast']}, desk: {kinds['desk']}, "
            f"reasoning: {kinds['reasoning']}) | "
            f"tokens: {self.total_input_tokens} in / "
            f"{self.total_output_tokens} out / {self.total_tokens} total | "
            f"models: {model_bits}"
        )


async def run_desk_turn(
    agent: Agent[ShopContext],
    user_input: str,
    ctx: ShopContext,
    ledger: ConversationLedger,
    budget: ConversationBudget,
    *,
    run_override_profile: str | None = None,
    max_turns: int = 8,
) -> object:
    """Run ONE customer turn through the SDK with ledger/ceiling hooks (FR-8, FR-11).

    ``run_override_profile`` is the run-level FR-1 override (the re-quote
    path): when set, the run gets a ``RunConfig`` whose model is a fresh
    router model for that profile, overriding every agent's own model for
    this run only. When ``None`` (the default), the agents' own models serve.

    Returns the run's raw ``final_output``: a ``str`` for ordinary answers,
    or an ``orders.Order`` when the order-taker agent produced one — callers
    normalize (the session layer decides what to do with each).

    Error mapping (the customer never sees a traceback):
    - ceiling exceeded -> :data:`BUDGET_CLOSE`
    - output guardrail tripwire -> :data:`guardrails.POLITE_REFUSAL`
      (machine-readable ``output_info`` logged; ledger marked guardrailed)
    - router chain exhausted -> :data:`SYSTEM_BUSY`
    - other Agents SDK errors -> :data:`GENERIC_SORRY`
    """
    hooks = ShopDeskHooks(ledger=ledger, budget=budget)
    run_config: RunConfig | None = None
    if run_override_profile is not None:
        run_config = RunConfig(model=model_config.get_routed_model(run_override_profile))
    # Scope the conversation accounting so agent-tool bodies (get_price_quote)
    # can attach the SAME ledger/budget to their nested agent runs (FR-8/FR-11).
    token = _CONVERSATION_ACCOUNTING.set((ledger, budget))
    try:
        try:
            result = await Runner.run(
                agent,
                user_input,
                context=ctx,
                max_turns=max_turns,
                hooks=hooks,
                run_config=run_config,
            )
        except TurnBudgetExceeded:
            logger.info("turn aborted: per-conversation model-call ceiling reached")
            return BUDGET_CLOSE
        except OutputGuardrailTripwireTriggered as exc:
            output_info = getattr(getattr(exc, "guardrail_result", None), "output", None)
            info = getattr(output_info, "output_info", None)
            logger.warning("output guardrail refused the finished answer: %s", info)
            ledger.mark_guardrailed()
            return guardrails.POLITE_REFUSAL
        except model_config.RoutedModelExhaustedError:
            logger.error("every candidate model failed for this turn")
            return SYSTEM_BUSY
        except ModelBehaviorError:
            logger.error("model behaved unexpectedly; ending the turn politely", exc_info=True)
            return GENERIC_SORRY
        except AgentsException:
            logger.error("agent run failed; ending the turn politely", exc_info=True)
            return GENERIC_SORRY
        return result.final_output
    finally:
        _CONVERSATION_ACCOUNTING.reset(token)
