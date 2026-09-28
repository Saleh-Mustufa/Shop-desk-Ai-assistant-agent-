"""Offline tests for FR-8 (specialist graph), FR-6 wiring, FR-9 (clone identity),
FR-5 wiring (order taker + finalize_order) and FR-11 (runner, ceiling, ledger).

No network and no API key: model calls go through fake SDK Models injected via
``model_config.RoutedModel(delegate_factory=...)`` (same pattern as
tests/test_router.py), the catalogue accessor is pointed at a temp fixture
holding exactly the repo ``catalogue.json`` content (same pattern as
tests/test_orders_guardrail.py), and the run-level-override test monkeypatches
``Runner.run`` to capture kwargs. The clone-identity assertions pin the REAL
``Agent.clone`` semantics of the installed SDK (0.22.3): ``clone`` is
``dataclasses.replace`` — a shallow copy where list attributes are shared by
reference unless explicitly overridden (see tests below for both cases).
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from agents import (
    Agent,
    AgentOutputSchemaBase,
    Handoff,
    Model,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    RunConfig,
    RunContextWrapper,
    Tool,
    Usage,
    UserError,
    set_tracing_disabled,
)
from agents.items import TResponseInputItem
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

import agents_desk
import catalogue
import guardrails
import model_config
import orders
import runner_cost
from context import ShopContext
from runner_cost import (
    BUDGET_CLOSE,
    ConversationBudget,
    ConversationLedger,
    ShopDeskHooks,
    TurnBudgetExceeded,
    TurnRecord,
    run_desk_turn,
    turn_kind,
)
from tools import lookup_product

REPO_ROOT = Path(__file__).resolve().parents[1]

# Exactly the repo catalogue.json content.
FIXTURE = {
    "shop": "Al-Noor Electronics",
    "currency": "PKR",
    "products": [
        {"sku": "KTL-01", "name": "Electric kettle 1.7L", "price": 4200, "stock": 12},
        {"sku": "FAN-22", "name": "Pedestal fan", "price": 9800, "stock": 0},
        {"sku": "TV-43S", "name": "43-inch LED smart TV", "price": 74500, "stock": 5},
        {"sku": "MIC-30", "name": "Microwave oven 30L", "price": 23500, "stock": 8},
        {"sku": "IRN-12", "name": "Steam iron", "price": 3600, "stock": 20},
        {"sku": "BLD-07", "name": "Blender 3-in-1", "price": 6900, "stock": 3},
    ],
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Keep tests independent of the developer's .env (profile/ceiling overrides)."""
    monkeypatch.delenv("PRIORITY_MODEL", raising=False)
    monkeypatch.delenv("SHOP_TURN_CEILING", raising=False)


@pytest.fixture()
def fixture_catalogue(tmp_path):
    """Point the catalogue accessor at a temp copy; restore afterwards."""
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps(FIXTURE, indent=2), encoding="utf-8")
    catalogue.set_catalogue_path(path)
    catalogue.reset_catalogue_cache()
    yield path
    catalogue.set_catalogue_path(REPO_ROOT / "catalogue.json")
    catalogue.reset_catalogue_cache()


def make_ctx(tier: str = "walk_in") -> ShopContext:
    return ShopContext(shop="Al-Noor Electronics", currency="PKR", customer_id="CUST-7", tier=tier)


class _TextFakeModel(Model):
    """Fake SDK Model that always answers with one plain assistant message.

    Same scripting shape as tests/test_router.py's FakeModel, but it always
    emits the same text (a desk turn needs exactly one such call here).
    """

    def __init__(self, text: str, usage: Usage | None = None) -> None:
        self.text = text
        self.usage = usage or Usage(requests=1, input_tokens=1, output_tokens=1, total_tokens=2)
        self.calls = 0

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any = None,
    ) -> ModelResponse:
        self.calls += 1
        message = ResponseOutputMessage(
            id=f"msg_{self.calls}",
            type="message",
            role="assistant",
            status="completed",
            content=[ResponseOutputText(text=self.text, type="output_text", annotations=[])],
        )
        return ModelResponse(
            output=[message],
            usage=self.usage,
            response_id=f"resp_{self.calls}",
        )

    def stream_response(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Any]:
        raise NotImplementedError("fake models do not stream")


def _routed_fake(fake: Model) -> model_config.RoutedModel:
    """A RoutedModel whose every chain candidate serves the same fake model."""
    return model_config.RoutedModel(profile="fast", delegate_factory=lambda _name: fake)


class _ScriptedModel(Model):
    """Scripted Model: each call pops one ModelResponse (test_router.py pattern)."""

    def __init__(self, script: list[ModelResponse]) -> None:
        self._script: deque[ModelResponse] = deque(script)
        self.calls = 0

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any = None,
    ) -> ModelResponse:
        self.calls += 1
        return self._script.popleft()

    def stream_response(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Any]:
        raise NotImplementedError("fake models do not stream")


def _text_response(text: str, usage: Usage | None = None) -> ModelResponse:
    message = ResponseOutputMessage(
        id="msg_scripted",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
    )
    return ModelResponse(
        output=[message],
        usage=usage or Usage(requests=1, input_tokens=1, output_tokens=1, total_tokens=2),
        response_id="resp_scripted",
    )


def _tool_call_response(name: str, arguments: str, usage: Usage | None = None) -> ModelResponse:
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
        usage=usage or Usage(requests=1, input_tokens=1, output_tokens=1, total_tokens=2),
        response_id="resp_tool",
    )


def _fake_pricing_specialist(monkeypatch, fake: Model) -> Agent:
    """Point get_pricing_specialist() at a fake-model clone for this test.

    The quote tool resolves the specialist at RUN time through the module
    function, so monkeypatching keeps the nested run offline.
    """
    pricing = agents_desk.make_pricing_specialist().clone(model=_routed_fake(fake))
    monkeypatch.setattr(agents_desk, "get_pricing_specialist", lambda: pricing)
    return pricing


# ---------------------------------------------------------------------------
# (a) FR-9: clone identity, pinned against the REAL installed clone semantics
# ---------------------------------------------------------------------------


def test_fr9_pricing_specialist_is_a_clone_with_explicit_overrides():
    agents_desk.reset_agents()
    base = agents_desk.get_specialist_base()
    pricing = agents_desk.get_pricing_specialist()

    assert base.name == "SpecialistBase"
    assert pricing is not base  # a distinct agent object
    assert pricing.name == "PricingSpecialist"
    assert pricing.instructions != base.instructions  # specialist instructions
    assert pricing.output_type is float  # bare-number contract (FR-8)
    assert base.output_type is None

    # The pricing clone overrides the model explicitly (the ONE allowed
    # agent-level restatement, NFR-2 justified in the agents_desk docstring).
    assert pricing.model is not base.model
    assert isinstance(pricing.model, model_config.RoutedModel)
    assert list(pricing.model.chain) == model_config.resolve_chain("reasoning")
    assert list(base.model.chain) == model_config.resolve_chain("fast")


def test_fr9_clone_sharing_semantics_pinned_to_installed_sdk():
    """Pin the REAL dataclasses.replace sharing, both directions.

    Installed Agent.clone() (0.22.3) is dataclasses.replace: a shallow copy.
    A field the clone does NOT pass arrives as the original's own object, so
    list attributes like ``tools`` are shared by reference; a field the clone
    DOES pass is used exactly as given (a fresh list shares nothing).
    """
    agents_desk.reset_agents()
    base = agents_desk.get_specialist_base()
    pricing = agents_desk.get_pricing_specialist()

    # The pricing clone PASSES tools=[lookup_product] -> its own fresh list,
    # holding the same tool objects the module exports.
    assert pricing.tools is not base.tools
    assert len(pricing.tools) == 1
    assert pricing.tools[0] is lookup_product

    # An escalation-style clone that does NOT pass tools shares the base's
    # list object and inherits the base model WITHOUT restating it (FR-9).
    escalation = base.clone(name="EscalationSpecialist", instructions="Handle escalations.")
    assert escalation.tools is base.tools
    assert escalation.model is base.model
    assert escalation.output_type is base.output_type
    assert escalation.handoffs is base.handoffs  # same list object (shallow copy)


def test_fr9_no_clone_instructions_restate_a_model_name():
    """NFR-2/FR-9: agent instructions never name a registry model."""
    texts = [
        agents_desk.get_specialist_base().instructions,
        agents_desk.get_pricing_specialist().instructions,
        agents_desk.get_order_taker_agent().instructions,
    ]
    for text in texts:
        lowered = text.lower()
        for name in model_config.MODEL_REGISTRY:
            assert name.lower() not in lowered


def test_specialist_singletons_cached_and_reset():
    agents_desk.reset_agents()
    base = agents_desk.get_specialist_base()
    pricing = agents_desk.get_pricing_specialist()
    order_taker = agents_desk.get_order_taker_agent()
    assert agents_desk.get_specialist_base() is base
    assert agents_desk.get_pricing_specialist() is pricing
    assert agents_desk.get_order_taker_agent() is order_taker
    agents_desk.reset_agents()
    assert agents_desk.get_specialist_base() is not base
    assert agents_desk.get_pricing_specialist() is not pricing
    assert agents_desk.get_order_taker_agent() is not order_taker


# ---------------------------------------------------------------------------
# (b) FR-8: the pricing specialist exposed to the Desk as get_price_quote
# ---------------------------------------------------------------------------


def test_desk_agent_exposes_get_price_quote_tool():
    desk = agents_desk.get_desk_agent()
    names = [tool.name for tool in desk.tools]
    # The five shop tools are kept, and the quote tool is appended.
    assert names[:5] == [
        "lookup_product",
        "check_stock_by_name",
        "list_catalogue",
        "loyalty_benefit",
        "holiday_bundles",
    ]
    assert "get_price_quote" in names
    quote_tool = next(tool for tool in desk.tools if tool.name == "get_price_quote")
    assert "pricing specialist" in quote_tool.description
    # The tool name matches the module constant, and the Desk passes a
    # product/SKU plus a quantity (minimal params per the fix-round ruling).
    assert quote_tool.name == agents_desk.QUOTE_TOOL_NAME
    schema = quote_tool.params_json_schema
    assert set(schema.get("properties", {})) == {"product", "qty"}


def test_pricing_specialist_profile_is_reasoning():
    pricing = agents_desk.get_pricing_specialist()
    assert list(pricing.model.chain) == model_config.resolve_chain("reasoning")


# ---------------------------------------------------------------------------
# (c) FR-6: catalogue output guardrail wired on both customer-facing agents
# ---------------------------------------------------------------------------


def test_output_guardrail_wired_on_desk_and_fastpath():
    """FR-6: the SDK runs OutputGuardrail wrappers, so assert on the wrapped
    check function (the wrapper is built around guardrails' function)."""
    for agent in (agents_desk.get_desk_agent(), agents_desk.get_fastpath_agent()):
        functions = [
            wrapper.guardrail_function for wrapper in agent.output_guardrails
        ]
        assert guardrails.catalogue_output_guardrail in functions


def test_order_taker_agent_shape():
    agent = agents_desk.get_order_taker_agent()
    assert agent.name == "OrderTaker"
    assert agent.output_type is orders.Order
    assert [tool.name for tool in agent.tools] == ["lookup_product"]
    assert isinstance(agent.model, model_config.RoutedModel)
    assert list(agent.model.chain) == model_config.resolve_chain("fast")


# ---------------------------------------------------------------------------
# FR-5 wiring: finalize_order is Python-side truth (catalogue prices only)
# ---------------------------------------------------------------------------


def test_finalize_order_builds_from_catalogue_prices(fixture_catalogue):
    order, problems = agents_desk.finalize_order([("KTL-01", 2)], "O-77")
    assert problems == []
    assert order.order_id == "O-77"
    assert order.status == "confirmed"
    assert order.items[0].sku == "KTL-01"
    assert order.items[0].qty == 2
    assert order.items[0].unit_price == 4200.0  # FROM the catalogue
    assert order.total == 8400.0
    assert orders.recompute_total(order) == order.total


def test_finalize_order_reports_problems_never_raises_on_bad_data(fixture_catalogue):
    order, problems = agents_desk.finalize_order([("TV-43S", 99)], "O-78")
    assert order.total == 99 * 74500.0
    assert problems and "stock" in problems[-1]


def test_finalize_order_rejects_unknown_sku_with_valueerror(fixture_catalogue):
    with pytest.raises(ValueError):
        agents_desk.finalize_order([("ZZZ-99", 1)], "O-79")


# ---------------------------------------------------------------------------
# (d) FR-11: the per-conversation ceiling
# ---------------------------------------------------------------------------


def test_budget_register_raises_on_n_plus_one():
    budget = ConversationBudget(max_model_calls=2)
    budget.register()
    budget.register()
    with pytest.raises(TurnBudgetExceeded) as excinfo:
        budget.register()
    message = str(excinfo.value)
    assert "2" in message  # names the ceiling
    assert message.strip().endswith(".") and "\n" not in message  # one sentence


def test_budget_default_from_env_or_40(monkeypatch):
    assert runner_cost.DEFAULT_MAX_MODEL_CALLS == 40
    assert ConversationBudget().max_model_calls == 40  # env unset -> 40
    monkeypatch.setenv("SHOP_TURN_CEILING", "7")
    assert ConversationBudget().max_model_calls == 7
    monkeypatch.setenv("SHOP_TURN_CEILING", "not-a-number")
    assert ConversationBudget().max_model_calls == 40  # bad env -> default
    monkeypatch.setenv("SHOP_TURN_CEILING", "0")
    assert ConversationBudget().max_model_calls == 40  # non-positive -> default


def test_turn_budget_exceeded_is_a_user_error():
    assert issubclass(TurnBudgetExceeded, UserError)


def test_third_turn_returns_budget_close():
    """Budget of 2 across 3 sequential turns: the third gets the polite close."""
    fake = _TextFakeModel(
        "The Electric kettle 1.7L (KTL-01) costs PKR 4,200 and we have 12 in stock."
    )
    agent = agents_desk.make_desk_agent().clone(model=_routed_fake(fake))
    ledger = ConversationLedger()
    budget = ConversationBudget(max_model_calls=2)
    ctx = make_ctx()
    set_tracing_disabled(True)
    try:
        first = asyncio.run(run_desk_turn(agent, "hello", ctx, ledger, budget))
        second = asyncio.run(run_desk_turn(agent, "hello again", ctx, ledger, budget))
        third = asyncio.run(run_desk_turn(agent, "one more", ctx, ledger, budget))
    finally:
        set_tracing_disabled(False)
    assert first == fake.text and second == fake.text
    assert third == BUDGET_CLOSE
    assert budget.used == 2  # the third call was refused BEFORE the model ran
    assert len(ledger.records) == 2  # only the two completed calls are recorded


def test_hooks_on_llm_start_enforces_budget_directly():
    ledger = ConversationLedger()
    hooks = ShopDeskHooks(ledger=ledger, budget=ConversationBudget(max_model_calls=1))
    agent = Agent(name="ShopDesk")
    wrapper = RunContextWrapper(context=make_ctx())
    asyncio.run(hooks.on_llm_start(wrapper, agent, None, []))
    with pytest.raises(TurnBudgetExceeded):
        asyncio.run(hooks.on_llm_start(wrapper, agent, None, []))


# ---------------------------------------------------------------------------
# (e) FR-11: hooks record real usage numbers from the run context
# ---------------------------------------------------------------------------


def test_hooks_record_one_turn_record_with_real_numbers():
    fake = _TextFakeModel(
        "ok", usage=Usage(requests=1, input_tokens=11, output_tokens=7, total_tokens=18)
    )
    routed = _routed_fake(fake)
    agent = Agent(name="ShopDesk", model=routed)
    ledger = ConversationLedger()
    budget = ConversationBudget(max_model_calls=5)
    set_tracing_disabled(True)
    try:
        output = asyncio.run(
            run_desk_turn(agent, "hello", make_ctx(), ledger, budget)
        )
    finally:
        set_tracing_disabled(False)
    assert output == "ok"  # (h) str outputs come back as str
    assert fake.calls == 1
    assert budget.used == 1
    assert len(ledger.records) == 1
    record = ledger.records[0]
    assert record.seq == 1
    assert record.agent_name == "ShopDesk"
    assert record.model_name == routed.active_model_name  # the router's own record
    assert record.input_tokens == 11
    assert record.output_tokens == 7
    assert record.total_tokens == 18
    assert record.kind == "desk"


def test_turn_kind_mapping():
    assert turn_kind("FastPath") == "fast"
    assert turn_kind("PricingSpecialist") == "reasoning"
    for name in ("ShopDesk", "OrderTaker", "SpecialistBase", "AnythingElse"):
        assert turn_kind(name) == "desk"


def test_hooks_on_llm_end_records_reasoning_kind_without_a_run():
    ledger = ConversationLedger()
    hooks = ShopDeskHooks(ledger=ledger, budget=ConversationBudget(max_model_calls=5))
    agent = Agent(
        name="PricingSpecialist", model=model_config.get_routed_model("reasoning")
    )
    response = ModelResponse(
        output=[],
        usage=Usage(requests=1, input_tokens=3, output_tokens=4, total_tokens=7),
        response_id="resp_1",
    )
    asyncio.run(hooks.on_llm_end(RunContextWrapper(context=make_ctx()), agent, response))
    record = ledger.records[0]
    assert record.agent_name == "PricingSpecialist"
    assert record.kind == "reasoning"
    assert record.model_name == agent.model.active_model_name
    assert (record.input_tokens, record.output_tokens, record.total_tokens) == (3, 4, 7)


def test_hooks_survive_missing_usage_and_model():
    """Defensive: a zero usage object and no model on the agent must not raise."""
    ledger = ConversationLedger()
    hooks = ShopDeskHooks(ledger=ledger, budget=ConversationBudget(max_model_calls=5))
    agent = Agent(name="ShopDesk", model=None)
    response = ModelResponse(output=[], usage=Usage(), response_id="resp_1")
    asyncio.run(hooks.on_llm_end(RunContextWrapper(context=make_ctx()), agent, response))
    record = ledger.records[0]
    assert record.model_name == "unknown"  # agent.model is None here
    assert (record.input_tokens, record.output_tokens, record.total_tokens) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Cost line format (FR-11)
# ---------------------------------------------------------------------------


def test_cost_line_format_and_totals():
    ledger = ConversationLedger()
    for _ in range(7):
        ledger.append(TurnRecord(0, "FastPath", "m-lite", 100, 20, 120, "fast"))
    for _ in range(4):
        ledger.append(TurnRecord(0, "ShopDesk", "m-lite", 50, 10, 60, "desk"))
    ledger.append(TurnRecord(0, "PricingSpecialist", "m-big", 200, 50, 250, "reasoning"))
    line = ledger.cost_line()
    assert line == (
        "Cost line — turns: 12 (fast-path: 7, desk: 4, reasoning: 1) | "
        "tokens: 1100 in / 230 out / 1330 total | models: m-lite x11, m-big x1"
    )
    # Tokens and model counts only — never price figures.
    assert "PKR" not in line and "₨" not in line


def test_cost_line_on_empty_ledger():
    assert ConversationLedger().cost_line() == (
        "Cost line — turns: 0 (fast-path: 0, desk: 0, reasoning: 0) | "
        "tokens: 0 in / 0 out / 0 total | models: none"
    )


def test_ledger_marks_guardrailed():
    ledger = ConversationLedger()
    assert ledger.guardrailed is False
    ledger.mark_guardrailed()
    assert ledger.guardrailed is True


# ---------------------------------------------------------------------------
# (f) The guardrail tripwire path through the custom runner
# ---------------------------------------------------------------------------


def test_guardrail_tripwire_returns_polite_refusal_and_marks_ledger(fixture_catalogue):
    fake = _TextFakeModel("The Steam iron (IRN-12) costs PKR 9,999 and we have 20 in stock.")
    agent = agents_desk.make_desk_agent().clone(model=_routed_fake(fake))
    ledger = ConversationLedger()
    budget = ConversationBudget()
    set_tracing_disabled(True)
    try:
        output = asyncio.run(
            run_desk_turn(agent, "how much is the iron?", make_ctx(), ledger, budget)
        )
    finally:
        set_tracing_disabled(False)
    assert output == guardrails.POLITE_REFUSAL
    assert ledger.guardrailed is True
    # The offending call still happened and is on the books (FR-11 honesty).
    assert len(ledger.records) == 1


# ---------------------------------------------------------------------------
# Fix round 1: nested quote calls are accounted (FR-8 ceiling, FR-11 ledger)
# ---------------------------------------------------------------------------


def test_nested_quote_call_is_recorded_in_the_same_ledger(monkeypatch, fixture_catalogue):
    """(a) success path: a desk turn calling get_price_quote records BOTH the
    desk call and the nested pricing call (kinds desk + reasoning, correct
    model names) against the SAME ledger and budget."""
    desk_usage = Usage(requests=1, input_tokens=30, output_tokens=10, total_tokens=40)
    desk_fake = _ScriptedModel(
        [_tool_call_response("get_price_quote", '{"product": "KTL-01", "qty": 4}', desk_usage)]
    )
    desk_routed = _routed_fake(desk_fake)
    # output_type=float: the SDK wraps scalar outputs, so the specialist's
    # payload is the wrapper object and final_output unwraps to the float.
    pricing_fake = _TextFakeModel(
        '{"response": 16800}',
        usage=Usage(requests=1, input_tokens=5, output_tokens=2, total_tokens=7),
    )
    pricing = _fake_pricing_specialist(monkeypatch, pricing_fake)
    # stop_on_first_tool: the quote tool's sentence IS the final answer, so
    # the turn is exactly one desk call + one nested pricing call.
    agent = agents_desk.make_desk_agent().clone(
        model=desk_routed, tool_use_behavior="stop_on_first_tool"
    )
    ledger = ConversationLedger()
    budget = ConversationBudget(max_model_calls=10)
    set_tracing_disabled(True)
    try:
        output = asyncio.run(
            run_desk_turn(agent, "quote 4 kettles", make_ctx(), ledger, budget)
        )
    finally:
        set_tracing_disabled(False)
    assert output == "The price quote for 4 x KTL-01 is 16800"
    assert [record.kind for record in ledger.records] == ["desk", "reasoning"]
    assert [record.agent_name for record in ledger.records] == ["ShopDesk", "PricingSpecialist"]
    assert ledger.records[0].model_name == desk_routed.active_model_name
    assert ledger.records[1].model_name == pricing.model.active_model_name
    assert (ledger.records[1].input_tokens, ledger.records[1].output_tokens) == (5, 2)
    assert budget.used == 2  # outer desk call + nested pricing call, same budget


def test_turn_ceiling_bounds_the_nested_quote_call(monkeypatch, fixture_catalogue):
    """(a) budget=2: the outer desk call (1) plus the nested pricing call (2)
    exhaust the budget, so the Desk's follow-up call trips and BUDGET_CLOSE
    surfaces. Without nested accounting the follow-up would have been call #2
    and succeeded — the nested call is what the ceiling now bounds."""
    desk_fake = _ScriptedModel(
        [_tool_call_response("get_price_quote", '{"product": "KTL-01", "qty": 4}')]
    )
    pricing_fake = _TextFakeModel('{"response": 16800}')
    _fake_pricing_specialist(monkeypatch, pricing_fake)
    agent = agents_desk.make_desk_agent().clone(model=_routed_fake(desk_fake))
    ledger = ConversationLedger()
    budget = ConversationBudget(max_model_calls=2)
    set_tracing_disabled(True)
    try:
        output = asyncio.run(
            run_desk_turn(agent, "quote 4 kettles", make_ctx(), ledger, budget)
        )
    finally:
        set_tracing_disabled(False)
    assert output == BUDGET_CLOSE
    # Exactly the two completed calls are on the books: desk + nested pricing.
    assert [record.kind for record in ledger.records] == ["desk", "reasoning"]
    assert [record.agent_name for record in ledger.records] == ["ShopDesk", "PricingSpecialist"]
    assert budget.used == 2
    assert desk_fake.calls == 1  # the follow-up desk call aborted before running
    assert pricing_fake.calls == 1


def test_quote_tool_works_without_conversation_accounting(monkeypatch):
    """(b) outside run_desk_turn the tool still runs the nested specialist —
    just without hooks, and no accounting objects exist to touch."""
    assert runner_cost.conversation_accounting() is None
    fake = _TextFakeModel('{"response": 8400}')
    _fake_pricing_specialist(monkeypatch, fake)
    wrapper = RunContextWrapper(context=make_ctx())
    output = asyncio.run(agents_desk._get_price_quote_impl(wrapper, "KTL-01", 2))  # noqa: SLF001
    assert output == "The price quote for 2 x KTL-01 is 8400"
    assert fake.calls == 1
    assert runner_cost.conversation_accounting() is None  # nothing leaked


def test_quote_tool_surfaces_budget_close_when_nested_run_trips(monkeypatch):
    """Belt-and-suspenders: when the nested run itself trips the ceiling, the
    tool returns BUDGET_CLOSE (never raises into the desk runner), and the
    budget is not double-registered."""
    fake = _TextFakeModel('{"response": 16800}')
    _fake_pricing_specialist(monkeypatch, fake)
    ledger = ConversationLedger()
    budget = ConversationBudget(max_model_calls=1)
    budget.register()  # the conversation's one allowed call is already spent

    async def scenario():
        token = runner_cost._CONVERSATION_ACCOUNTING.set((ledger, budget))  # noqa: SLF001
        try:
            return await agents_desk._get_price_quote_impl(  # noqa: SLF001
                RunContextWrapper(context=make_ctx()), "KTL-01", 4
            )
        finally:
            runner_cost._CONVERSATION_ACCOUNTING.reset(token)  # noqa: SLF001

    output = asyncio.run(scenario())
    assert output == BUDGET_CLOSE
    assert budget.used == 1  # the nested register raised BEFORE counting
    assert ledger.records == []  # the nested call never completed


# ---------------------------------------------------------------------------
# (g) Run-level FR-1 override: RunConfig carries a fresh router model
# ---------------------------------------------------------------------------


def _capture_runner(monkeypatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake_run(starting_agent, input, **kwargs):
        captured["agent"] = starting_agent
        captured["input"] = input
        captured["kwargs"] = kwargs
        return SimpleNamespace(final_output="ok", new_items=[], raw_responses=[])

    monkeypatch.setattr(runner_cost.Runner, "run", fake_run)
    return captured


def test_run_level_override_passes_reasoning_run_config(monkeypatch):
    captured = _capture_runner(monkeypatch)
    agent = agents_desk.get_desk_agent()
    output = asyncio.run(
        run_desk_turn(
            agent,
            "re-quote the kettle",
            make_ctx(),
            ConversationLedger(),
            ConversationBudget(),
            run_override_profile="reasoning",
        )
    )
    assert output == "ok"
    run_config = captured["kwargs"]["run_config"]
    assert isinstance(run_config, RunConfig)
    assert isinstance(run_config.model, model_config.RoutedModel)
    assert list(run_config.model.chain) == model_config.resolve_chain("reasoning")
    # The agent's own model is untouched by the run-level override.
    assert captured["agent"] is agent
    assert captured["kwargs"]["max_turns"] == 8
    assert isinstance(captured["kwargs"]["hooks"], ShopDeskHooks)


def test_no_override_leaves_run_config_none(monkeypatch):
    captured = _capture_runner(monkeypatch)
    agent = agents_desk.get_desk_agent()
    asyncio.run(
        run_desk_turn(agent, "hello", make_ctx(), ConversationLedger(), ConversationBudget())
    )
    assert captured["kwargs"]["run_config"] is None  # agent's own model serves


# ---------------------------------------------------------------------------
# (h) Final-output normalization: raw pass-through, callers normalize
# ---------------------------------------------------------------------------


def test_run_desk_turn_returns_str_for_str_outputs():
    fake = _TextFakeModel("The Electric kettle 1.7L (KTL-01) costs PKR 4,200 and we have 12 in stock.")
    agent = agents_desk.make_desk_agent().clone(model=_routed_fake(fake))
    ledger = ConversationLedger()
    set_tracing_disabled(True)
    try:
        output = asyncio.run(
            run_desk_turn(agent, "kettle?", make_ctx(), ledger, ConversationBudget())
        )
    finally:
        set_tracing_disabled(False)
    assert isinstance(output, str)
    assert output == fake.text


def test_run_desk_turn_returns_order_unchanged(monkeypatch):
    order = orders.build_order([("KTL-01", 2)], "O-9", status="confirmed")
    async def fake_run(starting_agent, input, **kwargs):
        return SimpleNamespace(final_output=order, new_items=[], raw_responses=[])

    monkeypatch.setattr(runner_cost.Runner, "run", fake_run)
    output = asyncio.run(
        run_desk_turn(
            agents_desk.get_order_taker_agent(),
            "KTL-01 x2",
            make_ctx(),
            ConversationLedger(),
            ConversationBudget(),
        )
    )
    assert output is order  # returned raw, not strified; session layer decides


def test_hooks_type_is_a_run_hooks_subclass():
    # RunHooks is a subscripted generic alias; check against its origin.
    from agents.lifecycle import RunHooksBase

    assert issubclass(ShopDeskHooks, RunHooksBase)
