"""Offline tests for the Shop Desk core: context, catalogue, tools, prompts.

No network, no model calls. Tool schemas are checked directly from the
generated ``params_json_schema``, and offered-tools enumeration uses the SDK's
supported ``Agent.get_all_tools(run_context)`` path (which is where
``is_enabled`` filtering is applied in openai-agents 0.22.3), driven with
``asyncio.run`` so no pytest-asyncio plugin is needed.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest
from agents import Agent, RunContextWrapper

import catalogue
import prompts
import tools
from context import ShopContext

REPO_ROOT = Path(__file__).resolve().parents[1]

FIXTURE = {
    "shop": "Test Mart",
    "currency": "PKR",
    "products": [
        {"sku": "KTL-01", "name": "Electric kettle 1.7L", "price": 4200, "stock": 12},
        {"sku": "FAN-22", "name": "Pedestal fan", "price": 9800, "stock": 0},
        {"sku": "TV-43S", "name": "43-inch LED smart TV", "price": 74500, "stock": 5},
    ],
}

ALL_TOOLS = [
    tools.lookup_product,
    tools.check_stock_by_name,
    tools.list_catalogue,
    tools.loyalty_benefit,
    tools.holiday_bundles,
]


@pytest.fixture()
def fixture_catalogue(tmp_path):
    """Point the catalogue accessor at a temp fixture; restore afterwards."""
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps(FIXTURE), encoding="utf-8")
    catalogue.set_catalogue_path(path)
    yield path
    catalogue.set_catalogue_path(REPO_ROOT / "catalogue.json")
    catalogue.reset_catalogue_cache()


def make_ctx(tier: str = "walk_in", customer_id: str = "CUST-42") -> ShopContext:
    return ShopContext(
        shop="Test Mart",
        currency="PKR",
        customer_id=customer_id,
        tier=tier,
    )


def wrapper(tier: str = "walk_in") -> RunContextWrapper[ShopContext]:
    return RunContextWrapper(context=make_ctx(tier=tier))


# ---------------------------------------------------------------------------
# (a) Tool schemas contain business params only — no wrapper/context leakage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_name", "expected_params"),
    [
        ("lookup_product", {"sku"}),
        ("check_stock_by_name", {"name"}),
        ("list_catalogue", set()),
        ("loyalty_benefit", set()),
        ("holiday_bundles", set()),
    ],
)
def test_tool_schemas_contain_business_params_only(tool_name, expected_params):
    function_tool = getattr(tools, tool_name)
    properties = set(function_tool.params_json_schema.get("properties", {}))
    assert properties == expected_params
    for banned in ("ctx", "context", "wrapper", "ShopContext"):
        assert banned not in properties
        assert banned not in json.dumps(function_tool.params_json_schema)


def test_tool_schemas_are_strict_objects():
    for function_tool in ALL_TOOLS:
        schema = function_tool.params_json_schema
        assert schema["type"] == "object"
        assert schema.get("additionalProperties") is False
        assert set(schema.get("required", [])) == set(
            schema.get("properties", {})
        )


# ---------------------------------------------------------------------------
# (b) Tier gating flags
# ---------------------------------------------------------------------------


def test_loyalty_benefit_gate_matches_tier():
    agent = Agent(name="gate-test", instructions="test", tools=ALL_TOOLS)
    assert tools.loyalty_benefit.is_enabled(wrapper("regular"), agent) is True
    assert tools.loyalty_benefit.is_enabled(wrapper("walk_in"), agent) is False


def test_holiday_bundles_gate_always_false():
    agent = Agent(name="gate-test", instructions="test", tools=ALL_TOOLS)
    for tier in ("regular", "walk_in"):
        assert tools.holiday_bundles.is_enabled(wrapper(tier), agent) is False


# ---------------------------------------------------------------------------
# (c) Offered-tools enumeration: seasonal tool invisible, loyalty tier-gated
# ---------------------------------------------------------------------------


def test_offered_tools_filtering_for_both_tiers():
    agent = Agent(name="desk", instructions="test", tools=ALL_TOOLS)
    for tier, loyalty_expected in (("regular", True), ("walk_in", False)):
        run_context = RunContextWrapper(context=make_ctx(tier=tier))
        offered = {t.name for t in asyncio.run(agent.get_all_tools(run_context))}
        # The seasonal tool must be invisible in every schema, for every tier.
        assert "holiday_bundles" not in offered
        assert loyalty_expected == ("loyalty_benefit" in offered)
        assert {"lookup_product", "check_stock_by_name", "list_catalogue"} <= offered


# ---------------------------------------------------------------------------
# (d) Prompts: clock-driven promises, shop name, no customer identifiers
# ---------------------------------------------------------------------------


def test_day_prompt_promises_same_day_delivery():
    prompt = prompts.build_desk_prompt(make_ctx(), datetime(2026, 9, 28, 12, 0))
    assert "Same-day delivery is available" in prompt
    assert "Test Mart" in prompt
    assert "PKR" in prompt


def test_night_prompt_says_closed_with_opening_time():
    prompt = prompts.build_desk_prompt(make_ctx(), datetime(2026, 9, 28, 22, 0))
    assert "Same-day delivery is not available" in prompt
    assert "opens at 9:00" in prompt
    assert "Test Mart" in prompt


def test_boundary_hours_use_correct_branch():
    opening = prompts.build_desk_prompt(make_ctx(), datetime(2026, 9, 28, 9, 0))
    closing = prompts.build_desk_prompt(make_ctx(), datetime(2026, 9, 28, 21, 0))
    assert "Same-day delivery is available" in opening
    assert "Same-day delivery is not available" in closing


@pytest.mark.parametrize("tier", ["walk_in", "regular"])
def test_prompt_never_contains_customer_identifiers(tier):
    ctx = make_ctx(tier=tier, customer_id="SECRET-CUST-42")
    for hour in (12, 22):
        prompt = prompts.build_desk_prompt(ctx, datetime(2026, 9, 28, hour, 0))
        assert "customer_id" not in prompt
        assert "SECRET-CUST-42" not in prompt


def test_prompts_differ_by_tier():
    day = datetime(2026, 9, 28, 12, 0)
    regular = prompts.build_desk_prompt(make_ctx(tier="regular"), day)
    walk_in = prompts.build_desk_prompt(make_ctx(tier="walk_in"), day)
    assert regular != walk_in
    assert "loyalty_benefit" in regular
    assert "loyalty_benefit" not in walk_in


def test_desk_instructions_callable_uses_context_and_clock():
    prompts.set_clock_provider(lambda: datetime(2026, 9, 28, 12, 0))
    try:
        text = prompts.desk_instructions(wrapper("regular"), None)
    finally:
        prompts.set_clock_provider(datetime.now)
    assert isinstance(text, str) and text
    assert "Same-day delivery is available" in text
    assert "Test Mart" in text
    assert "customer_id" not in text


def test_fastpath_prompt_names_the_lookup_tools():
    prompt = prompts.build_fastpath_prompt(make_ctx(), datetime(2026, 9, 28, 12, 0))
    assert "lookup_product" in prompt
    assert "check_stock_by_name" in prompt
    assert "Test Mart" in prompt
    assert "customer_id" not in prompt


# ---------------------------------------------------------------------------
# (e) Tool bodies: customer-ready sentences, never raise
# ---------------------------------------------------------------------------


def test_lookup_product_known_sku(fixture_catalogue):
    out = tools._lookup_product_impl(wrapper(), "KTL-01")
    assert "Electric kettle 1.7L" in out
    assert "4,200" in out  # formatted price
    assert "12 in stock" in out


def test_lookup_product_is_case_insensitive(fixture_catalogue):
    out = tools._lookup_product_impl(wrapper(), "ktl-01")
    assert "12 in stock" in out


def test_lookup_product_unknown_sku_is_polite_and_invents_nothing(fixture_catalogue):
    out = tools._lookup_product_impl(wrapper(), "ZZZ-99")
    assert "couldn't find" in out
    for real_sku in ("KTL-01", "FAN-22", "TV-43S"):
        assert real_sku not in out  # no invented/offered substitutes


def test_lookup_product_out_of_stock_no_order_suggestion(fixture_catalogue):
    out = tools._lookup_product_impl(wrapper(), "FAN-22")
    lowered = out.lower()
    assert "out of stock" in lowered
    assert "in stock" not in lowered  # never claims availability
    assert "order" not in lowered  # never suggests ordering it
    assert "alternative" in lowered


def test_check_stock_by_name_fuzzy_match(fixture_catalogue):
    out = tools._check_stock_by_name_impl(wrapper(), "kettle")
    assert "Electric kettle 1.7L" in out
    assert "4,200" in out
    assert "12 in stock" in out


def test_check_stock_by_name_unknown_is_polite(fixture_catalogue):
    out = tools._check_stock_by_name_impl(wrapper(), "hoverboard")
    assert "couldn't find" in out
    assert "KTL-01" not in out  # lists nothing invented


def test_list_catalogue_lists_only_fixture_items(fixture_catalogue):
    out = tools._list_catalogue_impl(wrapper())
    for sku, name, price in (
        ("KTL-01", "Electric kettle 1.7L", "4,200"),
        ("FAN-22", "Pedestal fan", "9,800"),
        ("TV-43S", "43-inch LED smart TV", "74,500"),
    ):
        assert sku in out and name in out and price in out
    assert "MIC-30" not in out  # nothing outside the fixture file leaks in


def test_loyalty_benefit_describes_discount_without_recomputing(fixture_catalogue):
    out = tools._loyalty_benefit_impl(wrapper("regular"))
    assert "5%" in out
    assert "PKR" not in out  # no recomputed price


@pytest.mark.parametrize(
    ("patch_target", "invoke"),
    [
        ("get_product", lambda: tools._lookup_product_impl(wrapper(), "KTL-01")),
        (
            "product_by_name",
            lambda: tools._check_stock_by_name_impl(wrapper(), "kettle"),
        ),
        ("load_catalogue", lambda: tools._list_catalogue_impl(wrapper())),
    ],
    ids=["lookup", "by_name", "list"],
)
def test_failing_accessor_returns_polite_sentence_not_raise(
    fixture_catalogue, monkeypatch, patch_target, invoke
):
    def boom(*args, **kwargs):
        raise RuntimeError("simulated catalogue failure")

    monkeypatch.setattr(catalogue, patch_target, boom)
    out = invoke()  # must not raise
    assert out == tools.SORRY_MESSAGE
    assert out.startswith("Sorry, I had trouble")


# ---------------------------------------------------------------------------
# Catalogue accessor behaviour
# ---------------------------------------------------------------------------


def test_default_catalogue_path_is_repo_root():
    assert catalogue.DEFAULT_CATALOGUE_PATH == REPO_ROOT / "catalogue.json"


def test_missing_catalogue_raises_one_clear_sentence(tmp_path):
    catalogue.set_catalogue_path(tmp_path / "missing.json")
    try:
        with pytest.raises(RuntimeError) as excinfo:
            catalogue.load_catalogue()
        message = str(excinfo.value)
        assert "catalogue" in message.lower()
        assert message.strip().endswith(".")
        assert message.count(". ") == 0  # a single, clear sentence
    finally:
        catalogue.set_catalogue_path(REPO_ROOT / "catalogue.json")
        catalogue.reset_catalogue_cache()


def test_malformed_catalogue_raises_clear_sentence(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ definitely not json", encoding="utf-8")
    catalogue.set_catalogue_path(bad)
    try:
        with pytest.raises(RuntimeError):
            catalogue.load_catalogue()
    finally:
        catalogue.set_catalogue_path(REPO_ROOT / "catalogue.json")
        catalogue.reset_catalogue_cache()


def test_format_price(fixture_catalogue):
    assert catalogue.format_price(4200) == "PKR 4,200"
    assert catalogue.format_price(74500, "PKR") == "PKR 74,500"
    assert catalogue.format_price(99.5) == "PKR 99.50"


def test_product_by_name_tolerates_whitespace_and_case(fixture_catalogue):
    product = catalogue.product_by_name("  PEDESTAL   fan ")
    assert product is not None and product["sku"] == "FAN-22"


def test_shop_context_dataclass_shape():
    ctx = ShopContext(shop="S", currency="PKR", customer_id="C1")
    assert ctx.tier == "walk_in"
    ctx2 = ShopContext(shop="S", currency="PKR", customer_id="C2", tier="regular")
    assert ctx2.tier == "regular"
