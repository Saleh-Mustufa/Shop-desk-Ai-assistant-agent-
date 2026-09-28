"""Offline tests for the FR-3 fast-path triage (triage.classify) and the
agent graph core (agents_desk).

No network and no model calls: classify() is pure Python over the text and
the repo-root catalogue.json, and agent construction only resolves model
chains through model_config (the Gemini client is built lazily on first use).
"""

from __future__ import annotations

import asyncio
import datetime
import json
from collections.abc import AsyncIterator
from pathlib import Path
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
    RunContextWrapper,
    Runner,
    Tool,
    set_tracing_disabled,
)
from agents.items import TResponseInputItem
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall

import agents_desk
import catalogue
import model_config
import prompts
import tools
import triage
from context import ShopContext

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_ctx(tier: str = "walk_in") -> ShopContext:
    return ShopContext(shop="Al-Noor Electronics", currency="PKR", customer_id="CUST-7", tier=tier)


def wrapper(tier: str = "walk_in") -> RunContextWrapper[ShopContext]:
    return RunContextWrapper(context=make_ctx(tier=tier))


# ---------------------------------------------------------------------------
# triage.classify — the FR-3 fast-path scenarios required by the plan
# ---------------------------------------------------------------------------


def test_kettle_price_question_is_fast_path():
    decision = triage.classify("What does the kettle cost?")
    assert decision.is_fast_path is True
    assert decision.sku == "KTL-01"
    assert decision.product_name == "Electric kettle 1.7L"


def test_blender_how_much_is_fast_path():
    decision = triage.classify("How much is the blender?")
    assert decision.is_fast_path is True
    assert decision.sku == "BLD-07"
    assert decision.product_name == "Blender 3-in-1"


def test_order_word_goes_to_desk():
    decision = triage.classify("Can I order two kettles?")
    assert decision.is_fast_path is False
    assert decision.sku is None and decision.product_name is None
    assert "order" in decision.reason


def test_two_products_is_ambiguous_and_goes_to_desk():
    decision = triage.classify("What's the price of the kettle and the fan?")
    assert decision.is_fast_path is False
    assert "multiple catalogue products" in decision.reason


def test_plural_tvs_in_stock_is_fast_path():
    # "TVs" must fuzzy-match "43-inch LED smart TV" via singularization.
    decision = triage.classify("Do you have TVs in stock?")
    assert decision.is_fast_path is True
    assert decision.sku == "TV-43S"
    assert decision.product_name == "43-inch LED smart TV"


def test_greeting_is_not_fast_path():
    decision = triage.classify("hello")
    assert decision.is_fast_path is False
    assert decision.sku is None


def test_bare_sku_is_fast_path():
    decision = triage.classify("FAN-22?")
    assert decision.is_fast_path is True
    assert decision.sku == "FAN-22"
    assert decision.product_name == "Pedestal fan"


def test_unknown_product_price_goes_to_desk():
    decision = triage.classify("What's the price of a juicer?")
    assert decision.is_fast_path is False
    assert "no catalogue product matches" in decision.reason


# ---------------------------------------------------------------------------
# triage.classify — additional guardrails of the fast-path contract
# ---------------------------------------------------------------------------


def test_long_and_many_sentence_messages_go_to_desk():
    long_text = "What does the kettle cost? " * 6  # >160 chars, 6 sentences
    decision = triage.classify(long_text)
    assert decision.is_fast_path is False
    assert "budget" in decision.reason


def test_compound_stock_question_stays_fast_path_within_budget():
    decision = triage.classify("How much is the pedestal fan and do you have it in stock?")
    assert decision.is_fast_path is True
    assert decision.sku == "FAN-22"


def test_two_skus_are_ambiguous():
    decision = triage.classify("KTL-01 and FAN-22?")
    assert decision.is_fast_path is False
    assert "multiple" in decision.reason


def test_unknown_sku_falls_through_to_desk():
    decision = triage.classify("What's the price of ZZZ-99?")
    assert decision.is_fast_path is False


def test_refund_word_goes_to_desk_even_with_a_sku():
    decision = triage.classify("I want a refund for KTL-01")
    assert decision.is_fast_path is False
    assert "refund" in decision.reason


def test_decision_includes_reason_always():
    for text in ("hello", "FAN-22?", "How much is the blender?", "Can I order two kettles?"):
        assert triage.classify(text).reason


# ---------------------------------------------------------------------------
# agents_desk — FastPath and ShopDesk agents (FR-3, NFR-2)
# ---------------------------------------------------------------------------


def _tool_names(agent: Agent) -> set[str]:
    return {tool.name for tool in agent.tools}


def test_fastpath_agent_shape():
    agent = agents_desk.get_fastpath_agent()
    assert agent.name == "FastPath"
    assert agent.tool_use_behavior == "stop_on_first_tool"
    assert agent.reset_tool_choice is True  # SDK default, deliberately untouched
    assert callable(agent.instructions)
    assert _tool_names(agent) == {"lookup_product", "check_stock_by_name"}
    assert isinstance(agent.model, model_config.RoutedModel)  # only source of models
    assert agent.model.chain  # resolved fast chain is non-empty


def test_fastpath_instructions_wrap_build_fastpath_prompt():
    prompts.set_clock_provider(lambda: datetime.datetime(2026, 9, 28, 12, 0))
    try:
        agent = agents_desk.get_fastpath_agent()
        text = agent.instructions(wrapper(), agent)  # type: ignore[operator]
    finally:
        prompts.set_clock_provider(datetime.datetime.now)
    assert isinstance(text, str) and text
    assert "lookup_product" in text and "check_stock_by_name" in text
    assert "customer_id" not in text and "CUST-7" not in text


def test_desk_agent_shape():
    agent = agents_desk.get_desk_agent()
    assert agent.name == "ShopDesk"
    assert agent.tool_use_behavior == "run_llm_again"  # ordinary loop (FR-3 contrast)
    assert agent.instructions is prompts.desk_instructions  # the callable itself
    # The five shop tools plus the FR-8 pricing-specialist quote tool.
    assert _tool_names(agent) == {
        "lookup_product",
        "check_stock_by_name",
        "list_catalogue",
        "loyalty_benefit",
        "holiday_bundles",
        "get_price_quote",
    }
    assert isinstance(agent.model, model_config.RoutedModel)


def test_both_agents_use_the_fast_profile_router():
    for agent in (agents_desk.get_fastpath_agent(), agents_desk.get_desk_agent()):
        assert agent.model.chain[0] in model_config.MODEL_REGISTRY  # registry names only
        assert model_config.MODEL_REGISTRY[agent.model.chain[0]].rpm == 15  # cheapest tier first


def test_singletons_are_cached_and_resettable():
    agents_desk.reset_agents()
    first = agents_desk.get_fastpath_agent()
    assert agents_desk.get_fastpath_agent() is first
    agents_desk.reset_agents()
    assert agents_desk.get_fastpath_agent() is not first
    desk_first = agents_desk.get_desk_agent()
    agents_desk.reset_agents()
    assert agents_desk.get_desk_agent() is not desk_first


@pytest.mark.parametrize("factory", [agents_desk.make_fastpath_agent, agents_desk.make_desk_agent])
def test_factories_build_fresh_agents(factory):
    assert factory() is not factory()


# ---------------------------------------------------------------------------
# Review fix round 1: -ing/-ed/-s inflections cannot bypass the order guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Is shipping available for the fan?",  # review finding 1, case 1
        "How much is shipping for the kettle?",  # review finding 1, case 2
        "Are delivery options available for the kettle?",  # review finding 1, case 3
        "Can I get a refund on the fan?",
        "The kettle was discounted, how much is it now?",
        "Two kettles were delivered yesterday, what did they cost?",
    ],
)
def test_inflected_order_words_go_to_desk(text):
    decision = triage.classify(text)
    assert decision.is_fast_path is False
    assert decision.sku is None and decision.product_name is None


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("shipping", "ship"),
        ("buying", "buy"),
        ("delivered", "deliver"),
        ("negotiating", "negotiate"),
        ("discounted", "discount"),
        ("confirmed", "confirm"),
        ("returning", "return"),
        ("purchased", "purchase"),
        ("refunded", "refund"),
        ("quoted", "quote"),
        ("orders", "order"),
        ("refunds", "refund"),
        ("deliveries", "delivery"),
    ],
)
def test_order_word_inflections_fold_to_stems(word, stem):
    assert triage._fold_to_order_stem(word) == stem


@pytest.mark.parametrize(
    "word", ["electric", "blender", "kettle", "hello", "led", "does", "fanned", "evening"]
)
def test_inflection_folding_does_not_touch_normal_words(word):
    assert triage._fold_to_order_stem(word) is None


@pytest.mark.parametrize(
    ("text", "fast", "sku"),
    [
        ("What does the kettle cost?", True, "KTL-01"),
        ("How much is the blender?", True, "BLD-07"),
        ("Can I order two kettles?", False, None),
        ("What's the price of the kettle and the fan?", False, None),
        ("Do you have TVs in stock?", True, "TV-43S"),
        ("hello", False, None),
        ("FAN-22?", True, "FAN-22"),
        ("What's the price of a juicer?", False, None),
    ],
)
def test_brief_required_scenarios_unchanged_after_fix(text, fast, sku):
    """The 8 scenarios the task brief mandates behave exactly as specified."""
    decision = triage.classify(text)
    assert decision.is_fast_path is fast
    assert decision.sku == sku


# ---------------------------------------------------------------------------
# Review fix round 1: the one-model-call guarantee, pinned offline
# ---------------------------------------------------------------------------


class _ToolCallFakeModel(Model):
    """Fake SDK Model that always answers with one lookup tool call.

    Same scripting shape as tests/test_router.py's FakeModel, but it emits a
    single ResponseFunctionToolCall so Runner executes the REAL lookup tool
    against the fixture catalogue — no network, no API key.
    """

    def __init__(self, tool_name: str, arguments: str) -> None:
        self._tool_name = tool_name
        self._arguments = arguments
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
        call = ResponseFunctionToolCall(
            arguments=self._arguments,
            call_id="call_1",
            name=self._tool_name,
            type="function_call",
            id="fc_1",
            status="completed",
        )
        return ModelResponse(
            output=[call],
            usage=Usage(requests=1, input_tokens=1, output_tokens=1, total_tokens=3),
            response_id="resp_1",
        )

    def stream_response(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Any]:
        raise NotImplementedError("fake models do not stream")


def test_fastpath_costs_exactly_one_model_call(tmp_path):
    """FR-3 pinned offline (review fix round 1): through Runner with a fake
    model, the fast-path agent makes EXACTLY one model call and the tool's
    output IS the run's final output (stop_on_first_tool)."""
    path = tmp_path / "catalogue.json"
    path.write_text(
        json.dumps(
            {
                "shop": "Test Mart",
                "currency": "PKR",
                "products": [
                    {"sku": "KTL-01", "name": "Electric kettle 1.7L", "price": 4200, "stock": 12}
                ],
            }
        ),
        encoding="utf-8",
    )
    catalogue.set_catalogue_path(path)
    try:
        fake = _ToolCallFakeModel("check_stock_by_name", '{"name": "kettle"}')
        routed = model_config.RoutedModel(profile="fast", delegate_factory=lambda _name: fake)
        agent = agents_desk.make_fastpath_agent().clone(model=routed)
        assert agent.tool_use_behavior == "stop_on_first_tool"
        ctx = make_ctx()
        set_tracing_disabled(True)
        try:
            result = asyncio.run(
                Runner.run(
                    agent,
                    "What does the kettle cost?",
                    context=ctx,
                    max_turns=agents_desk.FASTPATH_MAX_TURNS,
                )
            )
        finally:
            set_tracing_disabled(False)
        expected = tools._check_stock_by_name_impl(RunContextWrapper(context=ctx), "kettle")
        assert fake.calls == 1  # the fake would serve any second model call
        assert len(result.raw_responses) == 1  # trace evidence (FR-3)
        assert result.final_output == expected  # the tool's output IS the answer
        assert "12 in stock" in str(result.final_output)
    finally:
        catalogue.set_catalogue_path(catalogue.DEFAULT_CATALOGUE_PATH)
        catalogue.reset_catalogue_cache()
