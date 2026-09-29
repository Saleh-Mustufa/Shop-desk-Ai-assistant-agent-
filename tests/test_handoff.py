"""Offline tests for FR-10 (escalation handoff: typed reason + filtered history)
and the FR-9 clone semantics of the escalation agent.

No network and no API key: model calls go through fake SDK Models injected via
``model_config.RoutedModel(delegate_factory=...)`` (same pattern as
tests/test_runner_cost.py and tests/test_triage.py). The E2E test drives the
REAL ``Runner.run`` loop: the desk fake emits one ``escalate_to_human`` tool
call, the SDK executes the handoff (typed EscalationReason + input filter),
and a second fake serves the HumanEscalation reply.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

import pytest
from agents import (
    Agent,
    AgentOutputSchemaBase,
    HandoffInputData,
    Model,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    RunContextWrapper,
    Runner,
    Tool,
    Usage,
    set_tracing_disabled,
)
from agents.items import (
    HandoffCallItem,
    HandoffOutputItem,
    MessageOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
)
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from pydantic import ValidationError

import agents_desk
import model_config
import prompts
from context import ShopContext
from runner_cost import ConversationBudget, ConversationLedger, ShopDeskHooks

ESC_LOGGER = "shopdesk.escalation"

# Greppable markers the filter's capture reprs use for tool-ish items.
_TOOL_MARKERS = (
    "ToolCallItem(",
    "ToolCallOutputItem(",
    "HandoffCallItem(",
    "HandoffOutputItem(",
    "type=function_call",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Keep tests independent of the developer's .env (profile overrides)."""
    monkeypatch.delenv("PRIORITY_MODEL", raising=False)
    monkeypatch.delenv("SHOP_TURN_CEILING", raising=False)


@pytest.fixture(autouse=True)
def _fresh_agents_and_capture():
    """Fresh singletons and an empty capture list for every test."""
    agents_desk.reset_agents()
    agents_desk.clear_captured_handoff_filters()
    yield
    agents_desk.reset_agents()
    agents_desk.clear_captured_handoff_filters()


def make_ctx(tier: str = "walk_in") -> ShopContext:
    return ShopContext(shop="Al-Noor Electronics", currency="PKR", customer_id="CUST-7", tier=tier)


# ---------------------------------------------------------------------------
# Fake SDK models (test_runner_cost.py / test_triage.py pattern)
# ---------------------------------------------------------------------------


class _ScriptedModel(Model):
    """Scripted Model: each call pops one ModelResponse and remembers the input."""

    def __init__(self, script: list[ModelResponse]) -> None:
        self._script: deque[ModelResponse] = deque(script)
        self.calls = 0
        self.last_input: list[TResponseInputItem] | None = None

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Any],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any = None,
    ) -> ModelResponse:
        self.calls += 1
        self.last_input = list(input) if isinstance(input, list) else None
        return self._script.popleft()

    def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        raise NotImplementedError("fake models do not stream")


def _text_response(text: str) -> ModelResponse:
    message = ResponseOutputMessage(
        id="msg_scripted",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
    )
    return ModelResponse(
        output=[message],
        usage=Usage(requests=1, input_tokens=1, output_tokens=1, total_tokens=2),
        response_id="resp_scripted",
    )


def _tool_call_response(name: str, arguments: str) -> ModelResponse:
    call = ResponseFunctionToolCall(
        arguments=arguments,
        call_id="call_1",
        name=name,
        type="function_call",
        id="fc_1",
        status="completed",
    )
    return ModelResponse(
        output=[call],
        usage=Usage(requests=1, input_tokens=1, output_tokens=1, total_tokens=2),
        response_id="resp_tool",
    )


def _routed_fake(fake: Model) -> model_config.RoutedModel:
    """A fast-profile RoutedModel whose every candidate serves the same fake."""
    return model_config.RoutedModel(profile="fast", delegate_factory=lambda _name: fake)


# ---------------------------------------------------------------------------
# (a) FR-9: the escalation agent is a SpecialistBase clone that inherits
# ---------------------------------------------------------------------------


def test_escalation_agent_is_a_base_clone_inheriting_the_model():
    base = agents_desk.get_specialist_base()
    escalation = agents_desk.get_escalation_agent()

    assert escalation is not base  # a distinct agent object
    assert escalation.name == agents_desk.ESCALATION_NAME == "HumanEscalation"
    # FR-9: the model is inherited (the SAME object), never restated.
    assert escalation.model is base.model
    assert isinstance(escalation.model, model_config.RoutedModel)
    assert list(escalation.model.chain) == model_config.resolve_chain("fast")
    assert escalation.instructions is not base.instructions
    assert escalation.instructions == agents_desk.ESCALATION_INSTRUCTIONS
    # Shared by reference (shallow clone), and the base has no tools at all.
    assert escalation.tools is base.tools
    assert escalation.tools == []
    assert escalation.output_type is base.output_type is None
    assert escalation.handoff_description == (
        "Hands the conversation to a human agent when the desk is genuinely stuck"
    )


def test_escalation_agent_singleton_cached_and_reset():
    first = agents_desk.get_escalation_agent()
    assert agents_desk.get_escalation_agent() is first
    agents_desk.reset_agents()
    assert agents_desk.get_escalation_agent() is not first


def test_escalation_instructions_are_human_and_model_free():
    text = agents_desk.ESCALATION_INSTRUCTIONS
    lowered = text.lower()
    assert "human support agent" in lowered
    assert "without the desk's tool-call noise" in lowered
    assert "never quote" in lowered
    # NFR-2: instructions never name a registry model.
    for name in model_config.MODEL_REGISTRY:
        assert name.lower() not in lowered


# ---------------------------------------------------------------------------
# (b) the desk agent carries the escalation handoff
# ---------------------------------------------------------------------------


def test_desk_agent_carries_the_escalation_handoff():
    desk = agents_desk.get_desk_agent()
    assert len(desk.handoffs) == 1
    esc_handoff = desk.handoffs[0]
    assert esc_handoff.agent_name == "HumanEscalation"
    assert esc_handoff.tool_name == agents_desk.ESCALATE_TOOL_NAME == "escalate_to_human"
    assert "human agent" in esc_handoff.tool_description.lower()
    # The handoff is not an ordinary tool: the desk's tool surface is unchanged.
    assert "escalate_to_human" not in {tool.name for tool in desk.tools}


def test_handoff_invocation_returns_the_escalation_agent_and_parses_the_reason():
    desk = agents_desk.get_desk_agent()
    esc_handoff = desk.handoffs[0]
    payload = json.dumps({"reason": "policy", "details": "warranty question"})
    target = asyncio.run(
        esc_handoff.on_invoke_handoff(RunContextWrapper(context=make_ctx()), payload)
    )
    assert target is agents_desk.get_escalation_agent()
    assert target.name == "HumanEscalation"
    # The typed payload really was parsed and recorded by on_handoff.
    reason = agents_desk.last_escalation_reason()
    assert isinstance(reason, agents_desk.EscalationReason)
    assert reason.reason == "policy"
    assert reason.details == "warranty question"


def test_handoff_tool_schema_is_the_typed_reason():
    schema = agents_desk.get_desk_agent().handoffs[0].input_json_schema
    assert set(schema.get("properties", {})) == {"reason", "details"}
    assert set(schema.get("required", [])) == {"reason", "details"}
    assert schema.get("additionalProperties") is False  # strict schema mode
    # The Literal reason codes arrive as a closed enum on the tool schema.
    assert schema["properties"]["reason"].get("enum") == [
        "out_of_scope",
        "order_problem",
        "customer_request",
        "policy",
        "repeated_failure",
    ]


# ---------------------------------------------------------------------------
# (c) the typed reason arrives as a structured object and is logged
# ---------------------------------------------------------------------------


def test_escalation_reason_rejects_unknown_codes():
    with pytest.raises(ValidationError):
        agents_desk.EscalationReason(reason="angry_customer", details="x")


def test_on_handoff_logs_the_typed_reason(caplog):
    assert agents_desk.last_escalation_reason() is None
    reason = agents_desk.EscalationReason(
        reason="order_problem", details="Order ORD-9 arrived with a broken kettle."
    )
    with caplog.at_level(logging.INFO, logger=ESC_LOGGER):
        agents_desk._log_escalation_reason(  # noqa: SLF001 — the callback itself
            RunContextWrapper(context=make_ctx()), reason
        )
    assert "escalation reason=order_problem" in caplog.text
    assert "Order ORD-9 arrived with a broken kettle." in caplog.text
    # The reason arrives as the TYPED object, not a sentence.
    typed = agents_desk.last_escalation_reason()
    assert isinstance(typed, agents_desk.EscalationReason)
    assert typed is reason
    assert typed.reason == "order_problem"  # a Literal code, never free text


# ---------------------------------------------------------------------------
# (d) the input filter: tool noise removed, conversation kept, capture filled
# ---------------------------------------------------------------------------


def _assistant_message(text: str, id: str = "msg_1") -> ResponseOutputMessage:  # noqa: A002
    return ResponseOutputMessage(
        id=id,
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
    )


def _tool_call(name: str, arguments: str, call_id: str = "call_1") -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        arguments=arguments,
        call_id=call_id,
        name=name,
        type="function_call",
        id="fc_" + call_id,
        status="completed",
    )


def _mixed_handoff_input(owner: Agent) -> HandoffInputData:
    """A HandoffInputData mixing conversation turns with tool-call noise."""
    user_dict = {
        "role": "user",
        "content": "My blender arrived broken and I want to speak to a person.",
    }
    empty_system = {"role": "system", "content": "   "}
    stale_call = {
        "type": "function_call",
        "call_id": "call_0",
        "name": "lookup_product",
        "arguments": "{}",
    }
    stale_output = {"type": "function_call_output", "call_id": "call_0", "output": "ok"}

    pre_text = MessageOutputItem(
        agent=owner, raw_item=_assistant_message("Hello! How can I help?", id="msg_pre")
    )
    tool_call_item = ToolCallItem(
        agent=owner, raw_item=_tool_call("lookup_product", '{"product": "BLD-07"}')
    )
    tool_output_item = ToolCallOutputItem(
        agent=owner,
        raw_item={"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        output="ok",
    )
    reply_item = MessageOutputItem(
        agent=owner,
        raw_item=_assistant_message(
            "I'm sorry to hear that — let me bring in a colleague.", id="msg_reply"
        ),
    )
    handoff_call_item = HandoffCallItem(
        agent=owner,
        raw_item=_tool_call(
            "escalate_to_human",
            json.dumps({"reason": "order_problem", "details": "broken delivery"}),
            call_id="call_h",
        ),
    )
    handoff_output_item = HandoffOutputItem(
        agent=owner,
        raw_item={
            "type": "function_call_output",
            "call_id": "call_h",
            "output": '{"assistant": "HumanEscalation"}',
        },
        source_agent=owner,
        target_agent=owner,
    )
    return HandoffInputData(
        input_history=(user_dict, empty_system, stale_call, stale_output),
        pre_handoff_items=(pre_text,),
        new_items=(tool_call_item, tool_output_item, reply_item, handoff_call_item, handoff_output_item),
    )


def test_filter_removes_tool_items_and_keeps_the_conversation():
    owner = Agent(name="ShopDesk")
    data = _mixed_handoff_input(owner)
    filtered = agents_desk.shop_handoff_filter(data)

    # input_history: the user message survives; the empty system fragment and
    # the stale function_call/function_call_output dicts are gone.
    assert filtered.input_history == (
        {"role": "user", "content": "My blender arrived broken and I want to speak to a person."},
    )
    # pre_handoff_items: the desk's text reply survives.
    assert filtered.pre_handoff_items == (data.pre_handoff_items[0],)
    # new_items: ONLY the assistant text reply survives — the desk's shop tool
    # call+output AND the handoff call+output pair are gone.
    assert filtered.new_items == (data.new_items[2],)
    assert not any(
        isinstance(item, (ToolCallItem, ToolCallOutputItem, HandoffCallItem, HandoffOutputItem))
        for item in filtered.pre_handoff_items
    )
    assert not any(
        isinstance(item, (ToolCallItem, ToolCallOutputItem, HandoffCallItem, HandoffOutputItem))
        for item in filtered.new_items
    )
    # The kept conversation items are intact.
    assert filtered.pre_handoff_items[0].raw_item.content[0].text == "Hello! How can I help?"
    assert (
        filtered.new_items[0].raw_item.content[0].text
        == "I'm sorry to hear that — let me bring in a colleague."
    )


def test_filter_capture_records_a_before_after_pair():
    owner = Agent(name="ShopDesk")
    agents_desk.shop_handoff_filter(_mixed_handoff_input(owner))
    assert len(agents_desk.CAPTURED_HANDOFF_FILTERS) == 1
    before, after = agents_desk.CAPTURED_HANDOFF_FILTERS[-1]
    # BEFORE shows the tool noise (shop tools and the handoff call itself)...
    assert any(marker in before for marker in _TOOL_MARKERS)
    assert "ToolCallItem(lookup_product" in before
    assert "HandoffCallItem(escalate_to_human" in before
    assert "{role=user" in before
    # ...AFTER is the clean conversation: user + assistant text, no tool items.
    assert not any(marker in after for marker in _TOOL_MARKERS)
    assert "{role=user" in after
    assert "assistant(" in after


def test_filter_capture_is_bounded_and_clearable():
    owner = Agent(name="ShopDesk")
    for _ in range(13):
        agents_desk.shop_handoff_filter(_mixed_handoff_input(owner))
    assert len(agents_desk.CAPTURED_HANDOFF_FILTERS) == 10  # bounded to the last 10
    agents_desk.clear_captured_handoff_filters()
    assert agents_desk.CAPTURED_HANDOFF_FILTERS == []


def test_filter_never_raises_on_degenerate_input():
    degenerate = HandoffInputData(input_history=(), pre_handoff_items=(), new_items=())
    filtered = agents_desk.shop_handoff_filter(degenerate)
    assert filtered.input_history == ()
    assert filtered.pre_handoff_items == ()
    assert filtered.new_items == ()
    assert len(agents_desk.CAPTURED_HANDOFF_FILTERS) == 1

    # A string history (fresh string run input) passes through unchanged.
    string_history = HandoffInputData(
        input_history="hello there", pre_handoff_items=(), new_items=()
    )
    filtered_str = agents_desk.shop_handoff_filter(string_history)
    assert filtered_str.input_history == "hello there"


def test_desk_prompt_carries_escalation_guidance():
    prompt = prompts.build_desk_prompt(make_ctx(), datetime.datetime(2026, 9, 28, 12, 0))
    assert "escalate_to_human" in prompt
    assert "human colleague is taking over" in prompt
    for code in (
        "out_of_scope",
        "order_problem",
        "customer_request",
        "policy",
        "repeated_failure",
    ):
        assert code in prompt


# ---------------------------------------------------------------------------
# (e) E2E offline handoff: desk escalates, escalation answers, filter ran
# ---------------------------------------------------------------------------


ESCALATION_REPLY = (
    "Hello, I'm a human colleague taking over from our assistant. I'm sorry "
    "about the trouble with your electricity provider — could you share your "
    "account number and how best to reach you?"
)


def test_e2e_desk_hands_off_to_escalation_through_runner(monkeypatch, caplog):
    """The REAL Runner.run loop: desk calls escalate_to_human with a typed
    payload, the SDK switches to HumanEscalation, the filtered history reaches
    the escalation model, and the run's last agent is HumanEscalation."""
    desk_fake = _ScriptedModel(
        [
            _tool_call_response(
                "escalate_to_human",
                json.dumps(
                    {
                        "reason": "out_of_scope",
                        "details": "Customer wants to file a formal complaint "
                        "about their electricity provider and asked for a human.",
                    }
                ),
            )
        ]
    )
    escalation_fake = _ScriptedModel([_text_response(ESCALATION_REPLY)])

    desk_routed = _routed_fake(desk_fake)
    # The escalation's own fast-profile router — mirroring the inherited base
    # model (a distinct instance with the fast chain, never the desk's model).
    escalation_routed = _routed_fake(escalation_fake)
    escalation_agent = agents_desk.make_escalation_agent().clone(model=escalation_routed)
    monkeypatch.setattr(agents_desk, "get_escalation_agent", lambda: escalation_agent)
    desk = agents_desk.make_desk_agent().clone(model=desk_routed)

    ledger = ConversationLedger()
    budget = ConversationBudget(max_model_calls=10)
    set_tracing_disabled(True)
    try:
        with caplog.at_level(logging.INFO, logger=ESC_LOGGER):
            result = asyncio.run(
                Runner.run(
                    desk,
                    "I want to file a formal complaint about my electricity provider.",
                    context=make_ctx(),
                    max_turns=6,
                    hooks=ShopDeskHooks(ledger, budget),
                )
            )
    finally:
        set_tracing_disabled(False)

    # The handoff fired: the run's LAST agent is the escalation agent and the
    # final output is the escalation's reply.
    assert result.last_agent is escalation_agent
    assert result.last_agent.name == "HumanEscalation"
    assert result.final_output == ESCALATION_REPLY

    # The typed reason arrived through on_handoff (log + structured value).
    assert "escalation reason=out_of_scope" in caplog.text
    assert "electricity provider" in caplog.text
    reason = agents_desk.last_escalation_reason()
    assert isinstance(reason, agents_desk.EscalationReason)
    assert reason.reason == "out_of_scope"

    # Accounting: exactly two model calls — desk + escalation — and the
    # escalation's call went through a fast-profile chain (inherits base).
    assert [record.agent_name for record in ledger.records] == ["ShopDesk", "HumanEscalation"]
    assert budget.used == 2
    assert ledger.records[1].model_name == escalation_routed.active_model_name
    assert list(escalation_agent.model.chain) == model_config.resolve_chain("fast")

    # The escalation model received a FILTERED transcript: the user message
    # survived; no tool-call/tool-output items of any kind are present.
    assert escalation_fake.last_input is not None
    assert escalation_fake.last_input != []
    types = [
        item.get("type") for item in escalation_fake.last_input if isinstance(item, dict)
    ]
    assert "function_call" not in types
    assert "function_call_output" not in types
    roles = [
        item.get("role") for item in escalation_fake.last_input if isinstance(item, dict)
    ]
    assert "user" in roles

    # The live filter path recorded its (before, after) pair: the BEFORE shows
    # the handoff tool call itself, the AFTER is clean.
    assert len(agents_desk.CAPTURED_HANDOFF_FILTERS) >= 1
    before, after = agents_desk.CAPTURED_HANDOFF_FILTERS[-1]
    assert "HandoffCallItem(escalate_to_human" in before
    assert not any(marker in after for marker in _TOOL_MARKERS)
