"""Offline tests for FR-5 (typed orders) and FR-6 (catalogue output guardrail).

No network, no model calls, no SDK run loop. The guardrail is driven directly
with a ``RunContextWrapper[ShopContext]`` built the SDK way. The catalogue
accessor is pointed at a temp fixture holding exactly the repo
``catalogue.json`` content, so the mutation test is fully self-contained.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agents import Agent, GuardrailFunctionOutput, RunContextWrapper

import catalogue
import orders
from context import ShopContext
from guardrails import POLITE_REFUSAL, catalogue_output_guardrail

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

CATALOGUE_ANSWER = (
    "The Electric kettle 1.7L (KTL-01) costs PKR 4,200 and we have 12 in stock."
)

AGENT = Agent(name="guardrail-test")


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


@pytest.fixture()
def run_wrapper() -> RunContextWrapper[ShopContext]:
    return RunContextWrapper(
        context=ShopContext(
            shop="Al-Noor Electronics",
            currency="PKR",
            customer_id="CUST-42",
            tier="regular",
        )
    )


# ---------------------------------------------------------------------------
# (a) Total recompute: Python truth, mismatches reported never accepted
# ---------------------------------------------------------------------------


def test_recompute_total_sums_qty_times_unit_price():
    order = orders.Order(
        order_id="O-1",
        status="draft",
        items=[
            orders.LineItem(sku="KTL-01", qty=2, unit_price=4200.0),
            orders.LineItem(sku="IRN-12", qty=1, unit_price=3600.0),
        ],
        total=12000.0,
    )
    assert orders.recompute_total(order) == 12000.0


def test_check_total_returns_none_for_consistent_order():
    order = orders.build_order([("KTL-01", 2)], "O-2")
    assert order.total == 8400.0
    assert orders.check_total(order) is None


def test_check_total_reports_planted_mismatch_with_both_totals():
    order = orders.Order(
        order_id="O-3",
        status="draft",
        items=[orders.LineItem(sku="KTL-01", qty=2, unit_price=4200.0)],
        total=8401.0,  # planted: items actually sum to 8400
    )
    message = orders.check_total(order)
    assert message is not None  # never silently accepted
    assert "8400" in message and "8401" in message  # both totals named


# ---------------------------------------------------------------------------
# (b) validate_order flags
# ---------------------------------------------------------------------------


def test_validate_order_flags_unknown_sku(fixture_catalogue):
    order = orders.Order(
        order_id="O-4",
        status="draft",
        items=[orders.LineItem(sku="XYZ-99", qty=1, unit_price=100.0)],
        total=100.0,
    )
    problems = orders.validate_order(order)
    assert problems and any("XYZ-99" in p for p in problems)


def test_validate_order_flags_unit_price_off_catalogue(fixture_catalogue):
    order = orders.Order(
        order_id="O-5",
        status="draft",
        items=[orders.LineItem(sku="KTL-01", qty=1, unit_price=5000.0)],
        total=5000.0,
    )
    problems = orders.validate_order(order)
    assert problems and any("catalogue price" in p for p in problems)


def test_validate_order_flags_any_qty_of_zero_stock_fan22(fixture_catalogue):
    order = orders.Order(
        order_id="O-6",
        status="draft",
        items=[orders.LineItem(sku="FAN-22", qty=1, unit_price=9800.0)],
        total=9800.0,
    )
    problems = orders.validate_order(order)
    assert problems and any("in stock" in p for p in problems)


def test_validate_order_flags_qty_over_stock_for_bld07(fixture_catalogue):
    # BLD-07 has stock 3: ordering 4 is flagged, ordering 2 is not.
    over = orders.Order(
        order_id="O-7",
        status="draft",
        items=[orders.LineItem(sku="BLD-07", qty=4, unit_price=6900.0)],
        total=27600.0,
    )
    assert any("in stock" in p for p in orders.validate_order(over))

    within = orders.Order(
        order_id="O-8",
        status="draft",
        items=[orders.LineItem(sku="BLD-07", qty=2, unit_price=6900.0)],
        total=13800.0,
    )
    assert [p for p in orders.validate_order(within) if "in stock" in p] == []


def test_validate_order_flags_zero_qty(fixture_catalogue):
    order = orders.Order(
        order_id="O-9",
        status="draft",
        items=[orders.LineItem(sku="KTL-01", qty=0, unit_price=4200.0)],
        total=0.0,
    )
    problems = orders.validate_order(order)
    assert problems and any("at least 1" in p for p in problems)


def test_validate_order_valid_order_has_no_problems(fixture_catalogue):
    order = orders.build_order([("KTL-01", 2), ("IRN-12", 3)], "O-10")
    assert orders.validate_order(order) == []


# ---------------------------------------------------------------------------
# (c) build_order prices from the catalogue — the model never sets prices
# ---------------------------------------------------------------------------


def test_build_order_prices_from_catalogue_ignoring_model_prices(fixture_catalogue):
    order = orders.build_order([("KTL-01", 2)], "O-11")
    assert order.items[0].unit_price == 4200.0  # catalogue price, not a model guess
    assert order.items[0].sku == "KTL-01"
    assert order.total == 8400.0  # computed here, never stated by the model
    assert order.status == "draft"


def test_build_order_raises_one_sentence_for_unknown_sku(fixture_catalogue):
    with pytest.raises(ValueError) as excinfo:
        orders.build_order([("NOPE-99", 1)], "O-12")
    message = str(excinfo.value)
    assert "NOPE-99" in message
    assert message.strip().endswith(".") and ". " not in message


def test_build_order_raises_for_non_positive_qty(fixture_catalogue):
    with pytest.raises(ValueError) as excinfo:
        orders.build_order([("KTL-01", 0)], "O-13")
    assert "KTL-01" in str(excinfo.value)


# ---------------------------------------------------------------------------
# (d) Guardrail text path: a truthful catalogue answer passes
# ---------------------------------------------------------------------------


def test_guardrail_passes_truthful_catalogue_answer(fixture_catalogue, run_wrapper):
    result = catalogue_output_guardrail(run_wrapper, AGENT, CATALOGUE_ANSWER)
    assert isinstance(result, GuardrailFunctionOutput)
    assert result.tripwire_triggered is False
    assert result.output_info["reason"] == "ok"


# ---------------------------------------------------------------------------
# (e) Mutation proof (FR-6 done-when): re-pricing the catalogue this run
#     makes the previously passing answer fail
# ---------------------------------------------------------------------------


def test_catalogue_mutation_flips_a_passing_answer_to_failure(
    fixture_catalogue, run_wrapper
):
    assert (
        catalogue_output_guardrail(run_wrapper, AGENT, CATALOGUE_ANSWER).tripwire_triggered
        is False
    )

    mutated = json.loads(json.dumps(FIXTURE))
    mutated["products"][0]["price"] = 4300  # kettle re-priced this run
    fixture_catalogue.write_text(json.dumps(mutated, indent=2), encoding="utf-8")
    catalogue.reset_catalogue_cache()

    result = catalogue_output_guardrail(run_wrapper, AGENT, CATALOGUE_ANSWER)
    assert result.tripwire_triggered is True
    assert "unverifiable_amount" in result.output_info["reason"]
    assert 4200.0 in result.output_info["bad_amounts"]  # the stale figure is named


# ---------------------------------------------------------------------------
# (f) Guardrail text path: trips and passes per catalogue data
# ---------------------------------------------------------------------------


def test_guardrail_trips_on_invented_sku(fixture_catalogue, run_wrapper):
    result = catalogue_output_guardrail(
        run_wrapper, AGENT, "We can order the XYZ-99 unit for you next week."
    )
    assert result.tripwire_triggered is True
    assert "XYZ-99" in result.output_info["unknown_skus"]
    assert "unknown_sku" in result.output_info["reason"]


def test_guardrail_trips_on_unverifiable_amount(fixture_catalogue, run_wrapper):
    result = catalogue_output_guardrail(
        run_wrapper, AGENT, "That air fryer costs PKR 9,999 today."
    )
    assert result.tripwire_triggered is True
    assert 9999.0 in result.output_info["bad_amounts"]
    assert "unverifiable_amount" in result.output_info["reason"]


def test_guardrail_allows_sentence_scoped_aggregate(fixture_catalogue, run_wrapper):
    # Unit 4,200 x qty 2 = 8,400, justified within the same sentence.
    result = catalogue_output_guardrail(
        run_wrapper, AGENT, "2 kettles (KTL-01) come to PKR 8,400 in total."
    )
    assert result.tripwire_triggered is False
    assert result.output_info["reason"] == "ok"


def test_guardrail_trips_when_out_of_stock_item_offered_for_sale(
    fixture_catalogue, run_wrapper
):
    result = catalogue_output_guardrail(
        run_wrapper,
        AGENT,
        "Good news — the pedestal fan (FAN-22) is in stock and we can order it today.",
    )
    assert result.tripwire_triggered is True
    assert result.output_info["out_of_stock_skus"] == ["FAN-22"]
    assert "out_of_stock_sale" in result.output_info["reason"]


def test_guardrail_passes_neutral_out_of_stock_mention(fixture_catalogue, run_wrapper):
    result = catalogue_output_guardrail(
        run_wrapper, AGENT, "The pedestal fan (FAN-22) is currently out of stock."
    )
    assert result.tripwire_triggered is False
    assert result.output_info["reason"] == "ok"


def test_guardrail_passes_trailing_letter_sku_with_real_price(
    fixture_catalogue, run_wrapper
):
    # TV-43S (optional trailing letter in the SKU pattern) with its real price.
    result = catalogue_output_guardrail(
        run_wrapper, AGENT, "The 43-inch LED smart TV (TV-43S) costs PKR 74,500."
    )
    assert result.tripwire_triggered is False
    assert result.output_info["reason"] == "ok"


def test_guardrail_trips_on_invented_trailing_letter_sku(
    fixture_catalogue, run_wrapper
):
    result = catalogue_output_guardrail(
        run_wrapper, AGENT, "The TV-55S is a great deal for you."
    )
    assert result.tripwire_triggered is True
    assert "TV-55S" in result.output_info["unknown_skus"]
    assert "unknown_sku" in result.output_info["reason"]


def test_guardrail_trips_on_lowercase_sku_spelling(fixture_catalogue, run_wrapper):
    # Matches must exist EXACTLY (case-sensitive): "fan-22" is not the
    # catalogue's canonical "FAN-22", so it trips as an unknown SKU.
    result = catalogue_output_guardrail(
        run_wrapper, AGENT, "The fan-22 is ready for pickup."
    )
    assert result.tripwire_triggered is True
    assert "fan-22" in result.output_info["unknown_skus"]
    assert "unknown_sku" in result.output_info["reason"]


def test_guardrail_passes_unavailable_out_of_stock_item(fixture_catalogue, run_wrapper):
    # "unavailable" must NOT read as availability (word-boundary fix).
    result = catalogue_output_guardrail(
        run_wrapper, AGENT, "The pedestal fan (FAN-22) is currently unavailable."
    )
    assert result.tripwire_triggered is False
    assert result.output_info["reason"] == "ok"


# ---------------------------------------------------------------------------
# (g) Guardrail Order path
# ---------------------------------------------------------------------------


def test_guardrail_passes_valid_order(fixture_catalogue, run_wrapper):
    order = orders.build_order([("KTL-01", 2), ("BLD-07", 1)], "O-14")
    result = catalogue_output_guardrail(run_wrapper, AGENT, order)
    assert result.tripwire_triggered is False
    assert result.output_info["reason"] == "ok"


def test_guardrail_trips_on_order_with_wrong_total(fixture_catalogue, run_wrapper):
    order = orders.build_order([("KTL-01", 2)], "O-15")
    tampered = order.model_copy(update={"total": order.total + 1.0})
    result = catalogue_output_guardrail(run_wrapper, AGENT, tampered)
    assert result.tripwire_triggered is True
    assert result.output_info["reason"] == "order_invalid"
    problems = result.output_info["problems"]
    assert problems and any("mismatch" in p.lower() for p in problems)


# ---------------------------------------------------------------------------
# (h) The guardrail never raises — garbage in, GuardrailFunctionOutput out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "garbage",
    [None, b"\x00binary", 3.14, {"not": "an order"}],
    ids=["none", "bytes", "float", "dict"],
)
def test_guardrail_never_raises_on_garbage_output(
    fixture_catalogue, run_wrapper, garbage
):
    result = catalogue_output_guardrail(run_wrapper, AGENT, garbage)
    assert isinstance(result, GuardrailFunctionOutput)
    assert result.tripwire_triggered is False


def test_guardrail_survives_internal_error_without_raising(
    fixture_catalogue, run_wrapper, monkeypatch
):
    def boom():
        raise RuntimeError("simulated catalogue failure")

    monkeypatch.setattr(catalogue, "load_catalogue", boom)
    result = catalogue_output_guardrail(run_wrapper, AGENT, CATALOGUE_ANSWER)
    assert isinstance(result, GuardrailFunctionOutput)
    assert result.tripwire_triggered is False  # a broken guardrail must not kill the app
    assert result.output_info["reason"] == "guardrail_error"


def test_tripwire_output_info_is_convertible_to_polite_message(
    fixture_catalogue, run_wrapper
):
    result = catalogue_output_guardrail(
        run_wrapper, AGENT, "That gadget costs PKR 9,999."
    )
    assert result.tripwire_triggered is True
    assert result.output_info["polite_refusal"] == POLITE_REFUSAL
    assert POLITE_REFUSAL  # non-empty, ready for the customer
    assert result.output_info["reason"]  # machine-readable reason present
