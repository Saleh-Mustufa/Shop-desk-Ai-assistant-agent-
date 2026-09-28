"""Offline tests for the FR-3 fast-path triage (triage.classify) and the
agent graph core (agents_desk).

No network and no model calls: classify() is pure Python over the text and
the repo-root catalogue.json, and agent construction only resolves model
chains through model_config (the Gemini client is built lazily on first use).
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest
from agents import Agent, RunContextWrapper

import agents_desk
import model_config
import prompts
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
    assert _tool_names(agent) == {
        "lookup_product",
        "check_stock_by_name",
        "list_catalogue",
        "loyalty_benefit",
        "holiday_bundles",
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
