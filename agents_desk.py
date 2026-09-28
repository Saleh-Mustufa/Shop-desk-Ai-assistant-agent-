"""Agent graph core for Shop Desk (FR-3, FR-5, FR-6 wiring, FR-8/FR-9).

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

- **SpecialistBase** (:func:`get_specialist_base`) — profile ``fast``. The
  FR-9 base agent that specialists are derived from with ``Agent.clone``;
  never run on its own.

- **PricingSpecialist** (:func:`get_pricing_specialist`) — profile
  ``reasoning`` via an AGENT-LEVEL override (FR-9: the only clone allowed to
  restate a model, because the override IS its NFR-2 justification). Exposed
  to the Desk agent through the ``get_price_quote`` function tool (FR-8) and
  answers with a bare float (``output_type=float``). Justification: the
  specialist multiplies quantity by catalogue unit price on the re-quote
  path, where one wrong digit means a guardrail refusal and a whole retry
  turn; it is invoked rarely (only when the Desk re-quotes), so the more
  expensive profile buys arithmetic reliability exactly where it pays.

  FR-8 letter compliance: ``Agent.as_tool`` would also "expose the specialist
  as a tool", but a nested ``as_tool`` run cannot carry the conversation's
  run hooks, so its LLM call would bypass the FR-11 ceiling and cost ledger.
  This module therefore exposes the specialist through an equivalent custom
  ``@function_tool`` wrapper (:func:`get_price_quote`) that runs the SAME
  specialist agent as a nested ``Runner.run`` and attaches the conversation's
  ``ShopDeskHooks`` when turn accounting is in scope — a custom function tool
  IS "exposed as a tool", and this one returns the specialist's number while
  keeping the nested call inside the conversation's accounting.

- **OrderTaker** (:func:`get_order_taker_agent`) — profile ``fast``. Emits
  the typed pydantic :class:`orders.Order` (``output_type=Order``) when the
  customer confirms (FR-5). Justification: structured transcription of an
  already-confirmed basket; Python truth (:func:`finalize_order`) rebuilds
  the numbers from the catalogue afterwards, so no reasoning tier is needed.

Models come ONLY from ``model_config.get_routed_model`` — no model names are
written in this file (NFR-2 / §7 rule). Agents are built lazily and cached as
module singletons (:func:`get_fastpath_agent` / :func:`get_desk_agent` /
:func:`get_specialist_base` / :func:`get_pricing_specialist` /
:func:`get_order_taker_agent`), so importing this module stays cheap;
:func:`reset_agents` clears the cache for tests.

FR-6 wiring: both customer-facing agents (FastPath and ShopDesk) carry the
catalogue output guardrail as an output guardrail, so every finished answer
is checked against the catalogue as loaded this run before it reaches the
customer. The SDK runs ``OutputGuardrail`` wrapper objects (its decorator
just builds one around the function), so the guardrail function from
``guardrails.py`` is wrapped here once per agent — the check itself is that
module's.

FR-5 Python truth: :func:`finalize_order` — NOT the model — builds the order
with catalogue prices and validates it; the session layer calls it on
confirmation.

Extension points for later tasks (deliberately left out, not stubbed):
- FR-9: the escalation agent (inherits the base model; overrides instructions
  only) and FR-10: the Desk → escalation handoff with a typed
  ``EscalationReason`` input and a filtered history.
"""

from __future__ import annotations

import logging

from agents import Agent, OutputGuardrail, RunContextWrapper, Runner, function_tool

import model_config
import orders
import prompts
import runner_cost
from context import ShopContext
from guardrails import catalogue_output_guardrail
from orders import Order
from tools import (
    SORRY_MESSAGE,
    check_stock_by_name,
    holiday_bundles,
    list_catalogue,
    loyalty_benefit,
    lookup_product,
)

logger = logging.getLogger("shopdesk.agents")

__all__ = [
    "DESK_NAME",
    "FASTPATH_NAME",
    "ORDER_TAKER_NAME",
    "PRICING_SPECIALIST_NAME",
    "QUOTE_TOOL_NAME",
    "SPECIALIST_BASE_NAME",
    "finalize_order",
    "get_desk_agent",
    "get_fastpath_agent",
    "get_order_taker_agent",
    "get_pricing_specialist",
    "get_price_quote",
    "get_specialist_base",
    "make_desk_agent",
    "make_fastpath_agent",
    "make_order_taker_agent",
    "make_pricing_specialist",
    "make_specialist_base",
    "reset_agents",
]

FASTPATH_NAME = "FastPath"
DESK_NAME = "ShopDesk"
SPECIALIST_BASE_NAME = "SpecialistBase"
PRICING_SPECIALIST_NAME = "PricingSpecialist"
ORDER_TAKER_NAME = "OrderTaker"

# FR-8: the pricing specialist exposed to the Desk agent as a tool.
QUOTE_TOOL_NAME = "get_price_quote"

# Fast-path runs end on the first tool call, so two turns are ample (FR-3).
FASTPATH_MAX_TURNS = 2

# The nested pricing-specialist run inside get_price_quote: one lookup turn
# plus the bare-number answer turn, with headroom.
PRICING_TOOL_MAX_TURNS = 4

SPECIALIST_BASE_INSTRUCTIONS = (
    "You are a shop specialist. Complete exactly the task you are given, "
    "using only the outputs of the tools available to you; never invent "
    "products, prices or stock levels."
)

PRICING_INSTRUCTIONS = (
    "You are the pricing specialist. Use the lookup_product tool on the "
    "product code (SKU) the caller provides, read the catalogue unit price "
    "from its output, and compute the requested quantity multiplied by that "
    "unit price yourself. Reply with ONLY the resulting number in the "
    "required response format — a bare number, no words, no currency symbol "
    "and no explanation. Never apply a discount: discounts are promised "
    "verbally and applied at checkout, never baked into a quoted price."
)

ORDER_TAKER_INSTRUCTIONS = (
    "You are the order taker. The user message contains the confirmed basket "
    "as SKU/quantity lines, and may also contain an order reference. Convert "
    "it into the order structure: one line per SKU with its quantity, "
    "unit_price taken ONLY from lookup_product tool outputs for that SKU "
    "(never guessed), status 'confirmed', and total equal to the sum of "
    "quantity times unit price. Use the order id supplied in the message when "
    "one is present, otherwise use ORD-UNSET. Never invent prices, SKUs or "
    "stock levels."
)

_fastpath_agent: Agent[ShopContext] | None = None
_desk_agent: Agent[ShopContext] | None = None
_specialist_base: Agent[ShopContext] | None = None
_pricing_specialist: Agent[ShopContext] | None = None
_order_taker_agent: Agent[ShopContext] | None = None


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

    FR-6 wiring: the catalogue output guardrail runs on that final answer too
    (it is a catalogue sentence and passes; the wiring is the guarantee).
    """
    return Agent[ShopContext](
        name=FASTPATH_NAME,
        instructions=_fastpath_instructions,
        tools=[lookup_product, check_stock_by_name],
        model=model_config.get_routed_model("fast"),
        tool_use_behavior="stop_on_first_tool",
        output_guardrails=[OutputGuardrail(guardrail_function=catalogue_output_guardrail)],
    )


def make_desk_agent() -> Agent[ShopContext]:
    """The ShopDesk assistant with the ordinary loop over the full tool surface.

    Instructions are ``prompts.desk_instructions`` — the dynamic per-turn
    callable itself, so every turn rebuilds the prompt from the ShopContext
    and the clock (FR-4).

    FR-6 wiring: the catalogue output guardrail is attached here as an output
    guardrail, so every finished answer is checked against the catalogue
    before it reaches the customer.

    FR-8 wiring: the pricing specialist (:func:`get_pricing_specialist`) is
    exposed as the ``get_price_quote`` function tool, alongside the five shop
    tools. The wrapper runs the specialist as a nested agent run WITH the
    conversation's ledger/budget hooks when turn accounting is in scope (see
    :func:`_get_price_quote_impl`).
    """
    return Agent[ShopContext](
        name=DESK_NAME,
        instructions=prompts.desk_instructions,
        tools=[
            lookup_product,
            check_stock_by_name,
            list_catalogue,
            loyalty_benefit,
            holiday_bundles,
            get_price_quote,
        ],
        model=model_config.get_routed_model("fast"),
        output_guardrails=[OutputGuardrail(guardrail_function=catalogue_output_guardrail)],
    )


async def _get_price_quote_impl(
    ctx: RunContextWrapper[ShopContext], product: str, qty: int
) -> str:
    """Plain implementation of get_price_quote (kept async; tests call it directly).

    Runs the PricingSpecialist agent as a nested ``Runner.run``. When
    conversation accounting is in scope (``runner_cost.conversation_accounting``,
    set by ``runner_cost.run_desk_turn``), the nested run carries
    ``ShopDeskHooks(ledger, budget)`` so the nested LLM call registers against
    the SAME ceiling and lands in the SAME ledger as an outer call (each LLM
    start registers exactly once — no double counting, no re-entrancy
    hazard). Without the scope the nested run simply has no hooks. The
    specialist's number is catalogue-derived (qty x catalogue unit price; no
    discounted numbers, per the controller ruling) and is returned as one
    sentence the Desk can quote. Mirrors tools.py's never-raise convention.
    """
    wanted = str(product or "").strip()
    try:
        specialist = get_pricing_specialist()
        accounting = runner_cost.conversation_accounting()
        hooks = runner_cost.ShopDeskHooks(*accounting) if accounting is not None else None
        result = await Runner.run(
            specialist,
            f"Product code (SKU): {wanted}\nQuantity: {int(qty)}",
            context=ctx.context,
            hooks=hooks,
            max_turns=PRICING_TOOL_MAX_TURNS,
        )
    except runner_cost.TurnBudgetExceeded:
        logger.warning("price quote aborted: conversation model-call ceiling reached")
        return runner_cost.BUDGET_CLOSE
    except model_config.RoutedModelExhaustedError:
        logger.warning("price quote failed: every candidate model failed")
        return runner_cost.SYSTEM_BUSY
    except Exception:
        logger.warning("get_price_quote failed for product=%r qty=%r", product, qty, exc_info=True)
        return SORRY_MESSAGE
    number = result.final_output
    if isinstance(number, float) and number.is_integer():
        shown = str(int(number))
    else:
        shown = str(number)
    return f"The price quote for {int(qty)} x {wanted} is {shown}"


@function_tool
async def get_price_quote(
    ctx: RunContextWrapper[ShopContext], product: str, qty: int
) -> str:
    """Get a price quote (a single number, PKR) for a product and quantity from our pricing specialist."""
    return await _get_price_quote_impl(ctx, product, qty)


def make_specialist_base() -> Agent[ShopContext]:
    """The FR-9 base agent specialists are cloned from (never run directly).

    Generic specialist instructions only; concrete specialists override
    instructions and (where justified) the model.
    """
    return Agent[ShopContext](
        name=SPECIALIST_BASE_NAME,
        instructions=SPECIALIST_BASE_INSTRUCTIONS,
        model=model_config.get_routed_model("fast"),
    )


def make_pricing_specialist() -> Agent[ShopContext]:
    """The FR-8/FR-9 pricing specialist: a SpecialistBase clone.

    Clone semantics (SDK ``Agent.clone`` = ``dataclasses.replace``, a shallow
    copy): this clone explicitly overrides name, instructions, output_type,
    model (the justified agent-level ``reasoning`` override) and tools; every
    field it does NOT pass is shared with the base by reference. It never
    restates the base's inherited model — the model override below IS the
    NFR-2 justification (see module docstring).
    """
    return get_specialist_base().clone(
        name=PRICING_SPECIALIST_NAME,
        instructions=PRICING_INSTRUCTIONS,
        output_type=float,
        model=model_config.get_routed_model("reasoning"),
        tools=[lookup_product],
    )


def make_order_taker_agent() -> Agent[ShopContext]:
    """The FR-5 order-taker agent: transcribes the confirmed basket to an Order.

    The model's own numbers are never trusted: the session layer calls
    :func:`finalize_order`, which rebuilds the order from the catalogue in
    Python and re-validates it.
    """
    return Agent[ShopContext](
        name=ORDER_TAKER_NAME,
        instructions=ORDER_TAKER_INSTRUCTIONS,
        tools=[lookup_product],
        model=model_config.get_routed_model("fast"),
        output_type=Order,
    )


def finalize_order(
    items: list[tuple[str, int]],
    order_id: str,
    status: orders.OrderStatus = "confirmed",
) -> tuple[Order, list[str]]:
    """Python-side order truth for the confirmation flow (FR-5).

    Calls ``orders.build_order`` (unit prices come FROM THE CATALOGUE — the
    model never sets prices) and then ``orders.validate_order``. Returns
    ``(order, problems)``; an empty ``problems`` list means the order is
    valid. This is what the session layer calls on confirmation.
    """
    order = orders.build_order(items, order_id, status=status)
    return order, orders.validate_order(order)


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


def get_specialist_base() -> Agent[ShopContext]:
    """Cached SpecialistBase singleton (cheap import; built on first use)."""
    global _specialist_base
    if _specialist_base is None:
        _specialist_base = make_specialist_base()
    return _specialist_base


def get_pricing_specialist() -> Agent[ShopContext]:
    """Cached PricingSpecialist singleton (cheap import; built on first use)."""
    global _pricing_specialist
    if _pricing_specialist is None:
        _pricing_specialist = make_pricing_specialist()
    return _pricing_specialist


def get_order_taker_agent() -> Agent[ShopContext]:
    """Cached OrderTaker singleton (cheap import; built on first use)."""
    global _order_taker_agent
    if _order_taker_agent is None:
        _order_taker_agent = make_order_taker_agent()
    return _order_taker_agent


def reset_agents() -> None:
    """Drop the cached singletons (test helper; next accessor rebuilds)."""
    global _fastpath_agent, _desk_agent, _specialist_base, _pricing_specialist
    global _order_taker_agent
    _fastpath_agent = None
    _desk_agent = None
    _specialist_base = None
    _pricing_specialist = None
    _order_taker_agent = None
