"""Agent graph core for Shop Desk (FR-3, FR-5, FR-6 wiring, FR-8/FR-9, FR-10).

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

- **HumanEscalation** (:func:`get_escalation_agent`) — profile ``fast``
  inherited from :func:`get_specialist_base` (FR-9: the clone overrides name,
  instructions and ``handoff_description`` ONLY, so it never restates the
  model). It is NOT exposed as a tool and never runs on its own: the Desk
  reaches it only through the FR-10 handoff (:func:`make_escalation_handoff`)
  when the customer asks for a human, the request is out of scope, or the
  desk is genuinely stuck. The handoff carries a typed
  :class:`EscalationReason` (validated pydantic payload, FR-10) and transfers
  a filtered history (:func:`shop_handoff_filter`) with the desk's
  tool-call noise removed. Justification: a human takeover is a short
  conversational turn — no tools, no reasoning tier, the cheapest profile.

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

FR-10 wiring: the Desk agent carries one handoff —
:func:`make_escalation_handoff` to :func:`get_escalation_agent` — whose tool
call arguments are validated into the typed :class:`EscalationReason`
(Literal reason code + free-text details) and logged by
:func:`_log_escalation_reason` (exposed for evidence via
:func:`last_escalation_reason`). The handoff's ``input_filter``
(:func:`shop_handoff_filter`) strips the desk's tool-call/tool-output items
and empty system fragments from the transcript the escalation agent
receives, and records a compact (before, after) pair in
:data:`CAPTURED_HANDOFF_FILTERS` — FR-10's before/after demo surface.
"""

from __future__ import annotations

import logging
from typing import Literal

from agents import (
    Agent,
    Handoff,
    HandoffCallItem,
    HandoffInputData,
    HandoffOutputItem,
    MCPApprovalRequestItem,
    MCPApprovalResponseItem,
    MCPListToolsItem,
    MessageOutputItem,
    OutputGuardrail,
    ReasoningItem,
    RunContextWrapper,
    Runner,
    RunItem,
    ToolApprovalItem,
    ToolCallItem,
    ToolCallOutputItem,
    ToolSearchCallItem,
    ToolSearchOutputItem,
    function_tool,
    handoff,
)
from agents.extensions import handoff_filters
from agents.items import TResponseInputItem
from pydantic import BaseModel

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

# FR-10: the escalation reason arrives through its own logger so the typed
# value is demonstrably recorded when the handoff fires.
escalation_logger = logging.getLogger("shopdesk.escalation")

__all__ = [
    "CAPTURED_HANDOFF_FILTERS",
    "DESK_NAME",
    "ESCALATE_TOOL_NAME",
    "ESCALATION_INSTRUCTIONS",
    "ESCALATION_NAME",
    "FASTPATH_NAME",
    "ORDER_TAKER_NAME",
    "PRICING_SPECIALIST_NAME",
    "QUOTE_TOOL_NAME",
    "SPECIALIST_BASE_NAME",
    "EscalationReason",
    "clear_captured_handoff_filters",
    "finalize_order",
    "get_desk_agent",
    "get_escalation_agent",
    "get_fastpath_agent",
    "get_order_taker_agent",
    "get_pricing_specialist",
    "get_price_quote",
    "get_specialist_base",
    "last_escalation_reason",
    "make_desk_agent",
    "make_escalation_agent",
    "make_escalation_handoff",
    "make_fastpath_agent",
    "make_order_taker_agent",
    "make_pricing_specialist",
    "make_specialist_base",
    "reset_agents",
    "shop_handoff_filter",
]

FASTPATH_NAME = "FastPath"
DESK_NAME = "ShopDesk"
SPECIALIST_BASE_NAME = "SpecialistBase"
PRICING_SPECIALIST_NAME = "PricingSpecialist"
ORDER_TAKER_NAME = "OrderTaker"
ESCALATION_NAME = "HumanEscalation"

# FR-8: the pricing specialist exposed to the Desk agent as a tool.
QUOTE_TOOL_NAME = "get_price_quote"

# FR-10: the Desk -> escalation handoff tool name.
ESCALATE_TOOL_NAME = "escalate_to_human"

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

ESCALATION_INSTRUCTIONS = (
    "You are a human support agent taking over from the automated desk. You "
    "receive the customer conversation without the desk's tool-call noise. "
    "Greet the customer, acknowledge the reason for the escalation, and ask "
    "for whatever detail you still need (for example an order id or a "
    "contact preference). You cannot look up stock or prices — never quote "
    "figures; the desk's answers are in the conversation history if you need "
    "them. Keep it short and human."
)

_fastpath_agent: Agent[ShopContext] | None = None
_desk_agent: Agent[ShopContext] | None = None
_specialist_base: Agent[ShopContext] | None = None
_pricing_specialist: Agent[ShopContext] | None = None
_order_taker_agent: Agent[ShopContext] | None = None
_escalation_agent: Agent[ShopContext] | None = None


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

    FR-10 wiring: the desk carries exactly one handoff —
    :func:`make_escalation_handoff` to the HumanEscalation agent — so the
    desk can transfer the conversation (with a typed
    :class:`EscalationReason`) when the customer asks for a human, the
    request is out of scope, or it is genuinely stuck. The desk keeps its
    tools and its guardrail.
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
        handoffs=[make_escalation_handoff()],
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


# ---------------------------------------------------------------------------
# FR-10: the escalation handoff (typed reason + filtered history)
# ---------------------------------------------------------------------------


class EscalationReason(BaseModel):
    """The typed payload the Desk passes when it hands off to a human (FR-10).

    ``reason`` is a closed set of Literal codes (machine-readable, never a
    free-form sentence); ``details`` is the free text for the human agent.
    """

    reason: Literal[
        "out_of_scope",
        "order_problem",
        "customer_request",
        "policy",
        "repeated_failure",
    ]
    details: str


def make_escalation_agent() -> Agent[ShopContext]:
    """The FR-9/FR-10 escalation agent: a SpecialistBase clone.

    Clone semantics (SDK ``Agent.clone`` = ``dataclasses.replace``): this
    clone overrides name, handoff_description and instructions ONLY (FR-9) —
    it inherits the base model (fast profile) WITHOUT restating it and
    shares the base's (empty) tools list by reference. It has no tools, no
    guardrail and no output schema: a human takeover is a plain
    conversational turn.
    """
    return get_specialist_base().clone(
        name=ESCALATION_NAME,
        handoff_description=(
            "Hands the conversation to a human agent when the desk is "
            "genuinely stuck"
        ),
        instructions=ESCALATION_INSTRUCTIONS,
    )


def get_escalation_agent() -> Agent[ShopContext]:
    """Cached HumanEscalation singleton (cheap import; built on first use)."""
    global _escalation_agent
    if _escalation_agent is None:
        _escalation_agent = make_escalation_agent()
    return _escalation_agent


_last_escalation_reason: EscalationReason | None = None


def _log_escalation_reason(
    ctx: RunContextWrapper[ShopContext], reason: EscalationReason
) -> None:
    """The handoff's ``on_handoff`` callback (FR-10): record the typed reason.

    The SDK validates the model's tool-call arguments into
    :class:`EscalationReason` BEFORE this runs, so what arrives here is the
    structured object — the log line shows the Literal code and the details,
    which is FR-10's "typed reason" evidence.
    """
    global _last_escalation_reason
    _last_escalation_reason = reason
    escalation_logger.info(
        "escalation reason=%s details=%s", reason.reason, reason.details
    )


def last_escalation_reason() -> EscalationReason | None:
    """The most recent :class:`EscalationReason` a handoff carried, or None.

    Live-verification helper (FR-10): the typed value is returned as the
    structured object, not a sentence. :func:`reset_agents` clears it.
    """
    return _last_escalation_reason


# --- FR-10 handoff input filter --------------------------------------------
#
# The escalation agent receives the customer conversation WITHOUT the desk's
# tool-call noise. Built on the approach of the installed
# ``agents.extensions.handoff_filters.remove_all_tools`` (which drops every
# tool-ish RunItem class and every tool-ish raw input-item type), plus a
# strip of empty system fragments.
#
# WHAT IS REMOVED (from every section the SDK hands the filter):
# - tool-call items and tool outputs from the desk's history:
#   ToolCallItem / ToolCallOutputItem (function calls and their results) and
#   their raw ``function_call`` / ``function_call_output`` equivalents in
#   ``input_history``, plus tool-search, hosted-tool, reasoning and MCP
#   items — the same class set remove_all_tools uses;
# - empty system fragments (system messages with blank content);
# - the handoff tool call itself (HandoffCallItem) and its transfer output
#   (HandoffOutputItem) are ALSO removed from what the escalation agent
#   receives. The takeover is communicated by the SDK's agent switch and the
#   typed EscalationReason is delivered through the on_handoff callback
#   (logged, FR-10 evidence); keeping the call pair would reintroduce exactly
#   the tool noise FR-10 strips, and keeping the call without its output
#   would be malformed model input.
#
# WHAT IS KEPT: user messages, the desk's assistant text replies (so the
# escalation agent sees the conversation and any order summary the desk
# stated in prose), and every non-tool system message.
#
# The filter NEVER raises: on any structure surprise it falls back to
# ``remove_all_tools`` (or returns the input unchanged), logging a warning.

CAPTURED_HANDOFF_FILTERS: list[tuple[str, str]] = []
"""Compact ``(before, after)`` history reprs from every filter run (FR-10).

The demo surface for the before/after handoff history: each entry is
``(before, after)`` where both are one-line, greppable reprs of the input
items before and after filtering. Bounded to the last 10 entries;
:func:`clear_captured_handoff_filters` empties it (test helper).
"""

_CAPTURE_LIMIT = 10
_CAPTURE_SNIPPET = 60
_CAPTURE_MAX_CHARS = 1200

_TOOL_RUN_ITEM_TYPES: tuple[type[RunItem], ...] = (
    ToolCallItem,
    ToolCallOutputItem,
    HandoffCallItem,
    HandoffOutputItem,
    ToolSearchCallItem,
    ToolSearchOutputItem,
    ReasoningItem,
    MCPListToolsItem,
    MCPApprovalRequestItem,
    MCPApprovalResponseItem,
    ToolApprovalItem,
)

# Raw input-item ``type`` values that are tool calls/outputs (the same set
# remove_all_tools strips from ``input_history``).
_TOOL_INPUT_TYPES = frozenset(
    {
        "function_call",
        "function_call_output",
        "computer_call",
        "computer_call_output",
        "file_search_call",
        "tool_search_call",
        "tool_search_output",
        "web_search_call",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
        "mcp_approval_response",
        "reasoning",
        "code_interpreter_call",
        "image_generation_call",
        "local_shell_call",
        "local_shell_call_output",
        "shell_call",
        "shell_call_output",
        "apply_patch_call",
        "apply_patch_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "hosted_tool_call",
        "program",
        "program_output",
    }
)


def _snippet(text: object, limit: int = _CAPTURE_SNIPPET) -> str:
    """One-line, bounded description of a piece of item content."""
    value = " ".join(str(text).split())
    if len(value) > limit:
        return value[: limit - 1] + "…"
    return value


def _message_text(raw_item: object) -> str:
    """Concatenated text parts of a ResponseOutputMessage-ish raw item."""
    bits = [
        part.text
        for part in (getattr(raw_item, "content", None) or ())
        if isinstance(getattr(part, "text", None), str)
    ]
    return " ".join(bits)


def _item_repr(item: object) -> str:
    """One compact, greppable repr for one history/run item (evidence)."""
    if isinstance(item, dict):
        role = item.get("role")
        if role:
            return "{role=" + str(role) + ", " + _snippet(item.get("content")) + "}"
        return "{type=" + str(item.get("type")) + ", " + _snippet(item.get("output")) + "}"
    if isinstance(item, str):
        return "str(" + _snippet(item) + ")"
    if isinstance(item, MessageOutputItem):
        return "assistant(" + _snippet(_message_text(item.raw_item)) + ")"
    if isinstance(item, ToolCallItem):
        return (
            "ToolCallItem(" + _snippet(item.tool_name) + " args="
            + _snippet(getattr(item.raw_item, "arguments", None)) + ")"
        )
    if isinstance(item, ToolCallOutputItem):
        return "ToolCallOutputItem(output=" + _snippet(item.output) + ")"
    if isinstance(item, HandoffCallItem):
        return (
            "HandoffCallItem(" + _snippet(getattr(item.raw_item, "name", None)) + " args="
            + _snippet(getattr(item.raw_item, "arguments", None)) + ")"
        )
    if isinstance(item, HandoffOutputItem):
        raw = item.raw_item
        output = raw.get("output") if isinstance(raw, dict) else raw
        return "HandoffOutputItem(output=" + _snippet(output) + ")"
    return type(item).__name__ + "(" + _snippet(repr(item)) + ")"


def _sections_repr(input_data: HandoffInputData) -> str:
    """Compact one-line repr of every section of a HandoffInputData."""
    try:
        parts: list[str] = []
        history = input_data.input_history
        if isinstance(history, str):
            parts.append("history=[" + _item_repr(history) + "]")
        else:
            parts.append("history=[" + ", ".join(_item_repr(item) for item in history) + "]")
        parts.append(
            "pre=[" + ", ".join(_item_repr(item) for item in input_data.pre_handoff_items) + "]"
        )
        parts.append(
            "new=[" + ", ".join(_item_repr(item) for item in input_data.new_items) + "]"
        )
        if input_data.input_items is not None:
            parts.append(
                "input=[" + ", ".join(_item_repr(item) for item in input_data.input_items) + "]"
            )
        text = " | ".join(parts)
        if len(text) > _CAPTURE_MAX_CHARS:
            text = text[: _CAPTURE_MAX_CHARS - 1] + "…"
        return text
    except Exception:
        return repr(input_data)


def _record_capture(before: str, after: str) -> None:
    """Append one (before, after) pair, keeping only the last 10 entries."""
    CAPTURED_HANDOFF_FILTERS.append((before, after))
    del CAPTURED_HANDOFF_FILTERS[:-_CAPTURE_LIMIT]


def clear_captured_handoff_filters() -> None:
    """Empty :data:`CAPTURED_HANDOFF_FILTERS` (test/script helper)."""
    CAPTURED_HANDOFF_FILTERS.clear()


def _is_empty_system_fragment(item: object) -> bool:
    """A system message whose content is blank (or missing) — dead weight."""
    if not isinstance(item, dict) or item.get("role") != "system":
        return False
    content = item.get("content")
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, (list, tuple)):
        return len(content) == 0
    return False


def _strip_tool_run_items(items: tuple[RunItem, ...]) -> tuple[RunItem, ...]:
    """Drop every tool-ish RunItem (the remove_all_tools class set)."""
    return tuple(item for item in items if not isinstance(item, _TOOL_RUN_ITEM_TYPES))


def _strip_tool_input_items(
    items: tuple[TResponseInputItem, ...],
) -> tuple[TResponseInputItem, ...]:
    """Drop tool-ish raw input items and empty system fragments from history."""
    return tuple(
        item
        for item in items
        if not (
            isinstance(item, dict)
            and (
                item.get("type") in _TOOL_INPUT_TYPES
                or _is_empty_system_fragment(item)
            )
        )
    )


def shop_handoff_filter(input_data: HandoffInputData) -> HandoffInputData:
    """The FR-10 Desk -> escalation input filter (see module block above).

    Removes the desk's tool-call/tool-output noise (and empty system
    fragments) from every section of ``input_data`` and records a compact
    ``(before, after)`` pair in :data:`CAPTURED_HANDOFF_FILTERS`. Never
    raises: on a structure surprise it falls back to
    ``handoff_filters.remove_all_tools`` (or returns the input unchanged)
    with a logged warning.
    """
    before = _sections_repr(input_data)
    try:
        history = input_data.input_history
        filtered_history = (
            _strip_tool_input_items(history) if isinstance(history, tuple) else history
        )
        existing_input_items = input_data.input_items
        filtered_input_items = (
            _strip_tool_run_items(existing_input_items)
            if existing_input_items is not None
            else None
        )
        result = input_data.clone(
            input_history=filtered_history,
            pre_handoff_items=_strip_tool_run_items(input_data.pre_handoff_items),
            new_items=_strip_tool_run_items(input_data.new_items),
            input_items=filtered_input_items,
        )
    except Exception:
        logger.warning(
            "handoff input filter hit an unexpected structure; falling back to "
            "remove_all_tools",
            exc_info=True,
        )
        try:
            result = handoff_filters.remove_all_tools(input_data)
        except Exception:
            logger.warning(
                "remove_all_tools fallback failed; returning handoff input unchanged",
                exc_info=True,
            )
            result = input_data
    _record_capture(before, _sections_repr(result))
    return result


def make_escalation_handoff() -> Handoff[ShopContext, Agent[ShopContext]]:
    """The FR-10 Desk -> HumanEscalation handoff.

    ``handoff()`` validates the model's tool-call arguments into the typed
    :class:`EscalationReason` (``input_type``), reports it through
    :func:`_log_escalation_reason`, and filters the transferred history with
    :func:`shop_handoff_filter`. The tool is named ``escalate_to_human`` so
    the desk's prompt and the model's choice are unambiguous.
    """
    return handoff(
        get_escalation_agent(),
        tool_name_override=ESCALATE_TOOL_NAME,
        tool_description_override=(
            "Transfer this conversation to a human agent. Use when genuinely "
            "stuck, when the customer asks for a human, or when the request "
            "is out of scope. Pass the typed reason."
        ),
        on_handoff=_log_escalation_reason,
        input_type=EscalationReason,
        input_filter=shop_handoff_filter,
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
    global _order_taker_agent, _escalation_agent, _last_escalation_reason
    _fastpath_agent = None
    _desk_agent = None
    _specialist_base = None
    _pricing_specialist = None
    _order_taker_agent = None
    _escalation_agent = None
    _last_escalation_reason = None
