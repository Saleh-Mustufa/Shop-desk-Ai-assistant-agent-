"""Agent graph core for Shop Desk (FR-3; later tasks extend it additively).

Agents defined here, with their model profile and NFR-2 justification
(spec section "Model assignments"):

- **FastPath** (:func:`make_fastpath_agent`) — profile ``fast`` (cheapest
  tier). Serves plain price/stock questions routed by ``triage.classify``.
  Built with ``tool_use_behavior="stop_on_first_tool"``: the FIRST tool call's
  output becomes the run's final output and no further model call happens, so
  a fast-path turn costs exactly ONE model call (FR-3 trace evidence).
  Justification: a one-call catalogue lookup needs no reasoning.

- **ShopDesk** (:func:`make_desk_agent`) — profile ``fast`` (cheap tier).
  The general customer-facing assistant: conversational tool orchestration
  with the ordinary agent loop (tools run, results go back to the model).
  Justification: a reasoning model would multiply cost on every turn of every
  conversation; the fast tier is capable for catalogue-grounded chat.

Models come ONLY from ``model_config.get_routed_model`` — no model names are
written in this file (NFR-2 / §7 rule). Agents are built lazily and cached as
module singletons (:func:`get_fastpath_agent` / :func:`get_desk_agent`), so
importing this module stays cheap; :func:`reset_agents` clears the cache for
tests.

Extension points for later tasks (deliberately left out, not stubbed):
- FR-8/FR-9: ``SpecialistBase`` + ``clone()`` on this module → the pricing
  specialist (profile "reasoning", agent-level override) and the escalation
  agent (inherits the base model; overrides instructions only).
- FR-10: handoff from Desk to the escalation agent with a typed
  ``EscalationReason`` input and a filtered history.
- FR-5: the order-taker agent emitting the typed pydantic Order, reached when
  the customer confirms.
- FR-6: catalogue output-guardrail wiring on the Desk agent.
"""

from __future__ import annotations

from agents import Agent, RunContextWrapper

import model_config
import prompts
from context import ShopContext
from tools import (
    check_stock_by_name,
    holiday_bundles,
    list_catalogue,
    loyalty_benefit,
    lookup_product,
)

__all__ = [
    "DESK_NAME",
    "FASTPATH_NAME",
    "get_desk_agent",
    "get_fastpath_agent",
    "make_desk_agent",
    "make_fastpath_agent",
    "reset_agents",
]

FASTPATH_NAME = "FastPath"
DESK_NAME = "ShopDesk"

# Fast-path runs end on the first tool call, so two turns are ample (FR-3).
FASTPATH_MAX_TURNS = 2

_fastpath_agent: Agent[ShopContext] | None = None
_desk_agent: Agent[ShopContext] | None = None


def _fastpath_instructions(
    context: RunContextWrapper[ShopContext], agent: Agent[ShopContext]
) -> str:
    """Dynamic-instructions callable wrapping prompts.build_fastpath_prompt (FR-4)."""
    return prompts.build_fastpath_prompt(context.context, prompts.get_clock_provider()())


def make_fastpath_agent() -> Agent[ShopContext]:
    """The FR-3 fast-path agent: one model call, the tool's answer is final.

    ``stop_on_first_tool`` means the first lookup tool's output becomes the
    run's final output — the model never sees the tool result and never
    rephrases it (that is exactly what the fast path gives up).
    """
    return Agent[ShopContext](
        name=FASTPATH_NAME,
        instructions=_fastpath_instructions,
        tools=[lookup_product, check_stock_by_name],
        model=model_config.get_routed_model("fast"),
        tool_use_behavior="stop_on_first_tool",
    )


def make_desk_agent() -> Agent[ShopContext]:
    """The ShopDesk assistant with the ordinary loop over the full tool surface.

    Instructions are ``prompts.desk_instructions`` — the dynamic per-turn
    callable itself, so every turn rebuilds the prompt from the ShopContext
    and the clock (FR-4). Later tasks add handoffs, the output guardrail and
    the order flow on top of this agent (see module docstring).
    """
    return Agent[ShopContext](
        name=DESK_NAME,
        instructions=prompts.desk_instructions,
        tools=[lookup_product, check_stock_by_name, list_catalogue, loyalty_benefit, holiday_bundles],
        model=model_config.get_routed_model("fast"),
    )


def get_fastpath_agent() -> Agent[ShopContext]:
    """Cached FastPath singleton (cheap import; built on first use)."""
    global _fastpath_agent
    if _fastpath_agent is None:
        _fastpath_agent = make_fastpath_agent()
    return _fastpath_agent


def get_desk_agent() -> Agent[ShopContext]:
    """Cached ShopDesk singleton (cheap import; built on first use)."""
    global _desk_agent
    if _desk_agent is None:
        _desk_agent = make_desk_agent()
    return _desk_agent


def reset_agents() -> None:
    """Drop the cached singletons (test helper; next accessor rebuilds)."""
    global _fastpath_agent, _desk_agent
    _fastpath_agent = None
    _desk_agent = None
