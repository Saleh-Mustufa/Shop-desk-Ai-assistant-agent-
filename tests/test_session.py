"""Offline tests for FR-12 (DeskSession: per-session state, history trimming,
confirmation flow, basket parsing, session isolation, cost line, fast/desk
routing and the turn-11 order memory).

No network and no API key: model calls go through fake SDK Models injected via
``model_config.RoutedModel(delegate_factory=...)`` (same pattern as
tests/test_handoff.py / tests/test_runner_cost.py), the catalogue accessor is
pointed at a temp fixture holding exactly the repo ``catalogue.json`` content
(same pattern as tests/test_orders_guardrail.py), and the real ``Runner.run``
loop executes inside ``DeskSession.handle_user_message`` — the exact code path
the Chainlit UI and the demo script drive.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from agents import (
    AgentOutputSchemaBase,
    Model,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    Tool,
    Usage,
    set_tracing_disabled,
)
from agents.items import TResponseInputItem
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

import agents_desk
import model_config
from context import ShopContext
from orders import recompute_total
from runner_cost import ConversationBudget, ConversationLedger
from session_store import (
    CONFIRM_PATTERN,
    LAST_EXCHANGES_KEPT,
    MAX_HISTORY_MESSAGES,
    DeskSession,
    DeskSessionRegistry,
    parse_basket_statement,
)

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
    """Keep tests independent of the developer's .env overrides."""
    monkeypatch.delenv("PRIORITY_MODEL", raising=False)
    monkeypatch.delenv("SHOP_TURN_CEILING", raising=False)
    monkeypatch.delenv("SHOP_DEFAULT_TIER", raising=False)


@pytest.fixture(autouse=True)
def _unlimited_router(monkeypatch):
    """Lift the router's real sliding-window limits for offline runs.

    The fake delegates share the router's module-level per-model usage
    counters (keyed by registry model name); the trimming tests fire dozens
    of model calls in seconds, which would trip the 15-RPM lite tier and
    starve every later test with SYSTEM_BUSY turns.
    """
    unlimited = {
        name: model_config.ModelEntry(
            name=name, rpm=0, tpm=None, rpd=None, tier=entry.tier, available=entry.available
        )
        for name, entry in model_config.MODEL_REGISTRY.items()
    }
    monkeypatch.setattr(model_config, "MODEL_REGISTRY", unlimited)
    model_config.reset_state()
    yield
    model_config.reset_state()


@pytest.fixture(autouse=True)
def fixture_catalogue(tmp_path):
    """Point the catalogue accessor at a temp copy; restore afterwards."""
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps(FIXTURE, indent=2), encoding="utf-8")
    import catalogue

    catalogue.set_catalogue_path(path)
    catalogue.reset_catalogue_cache()
    yield path
    catalogue.set_catalogue_path(REPO_ROOT / "catalogue.json")
    catalogue.reset_catalogue_cache()


@pytest.fixture(autouse=True)
def _no_tracing_and_fresh_registry():
    set_tracing_disabled(True)
    DeskSessionRegistry.clear()
    yield
    DeskSessionRegistry.clear()
    set_tracing_disabled(False)


# ---------------------------------------------------------------------------
# Fake models (per-agent RoutedModel delegate pattern from test_handoff.py)
# ---------------------------------------------------------------------------


class _TextModel(Model):
    """Fake SDK Model: always the same text; records every call and input."""

    def __init__(self, text: str, usage: Usage | None = None) -> None:
        self.text = text
        self.usage = usage or Usage(requests=1, input_tokens=2, output_tokens=3, total_tokens=5)
        self.calls = 0
        self.inputs: list[Any] = []

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],  # noqa: A002
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
        self.inputs.append(input)
        message = ResponseOutputMessage(
            id=f"msg_{self.calls}",
            type="message",
            role="assistant",
            status="completed",
            content=[ResponseOutputText(text=self.text, type="output_text", annotations=[])],
        )
        return ModelResponse(
            output=[message], usage=self.usage, response_id=f"resp_{self.calls}"
        )

    def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        raise NotImplementedError("fake models do not stream")


def _routed_fake(fake: Model) -> model_config.RoutedModel:
    """A fast-profile RoutedModel whose every candidate serves the same fake."""
    return model_config.RoutedModel(profile="fast", delegate_factory=lambda _name: fake)


FAST_REPLY = "The Electric kettle 1.7L (KTL-01) costs PKR 4,200 and we have 12 in stock."
DESK_REPLY = "Noted — anything else I can help with?"


@pytest.fixture()
def fake_agents(monkeypatch):
    """Route the session's agents to fake models (offline, per-agent)."""
    fake_fast = _TextModel(FAST_REPLY)
    fake_desk = _TextModel(DESK_REPLY)
    fast_agent = agents_desk.make_fastpath_agent().clone(model=_routed_fake(fake_fast))
    desk_agent = agents_desk.make_desk_agent().clone(model=_routed_fake(fake_desk))
    monkeypatch.setattr(agents_desk, "get_fastpath_agent", lambda: fast_agent)
    monkeypatch.setattr(agents_desk, "get_desk_agent", lambda: desk_agent)
    return fake_fast, fake_desk


def make_session(
    session_id: str = "S-1",
    tier: str = "walk_in",
    ledger: ConversationLedger | None = None,
    budget: ConversationBudget | None = None,
) -> DeskSession:
    customer = ShopContext(
        shop="Al-Noor Electronics", currency="PKR", customer_id="CUST-7", tier=tier
    )
    return DeskSession(session_id, customer, ledger=ledger, budget=budget)


def _turn(session: DeskSession, text: str) -> str:
    return asyncio.run(session.handle_user_message(text))


def _input_text(raw: Any) -> str:
    """Flatten whatever the fake model received into one searchable string."""
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, default=str)


# ---------------------------------------------------------------------------
# (0) Trim rule constants + confirmation pattern are exposed
# ---------------------------------------------------------------------------


def test_trim_rule_constants_are_exposed():
    assert MAX_HISTORY_MESSAGES == 12  # 6 exchanges, per the FR-12 decision
    assert LAST_EXCHANGES_KEPT == 2


def test_confirm_pattern_matches_the_confirmation_phrases():
    for text in (
        "confirm",
        "Confirm the order",
        "yes place it",
        "place the order now",
        "checkout please",
        "let's finalize",
    ):
        assert CONFIRM_PATTERN.search(text), text
    assert not CONFIRM_PATTERN.search("what time do you open?")


# ---------------------------------------------------------------------------
# (a) History trimming: 30 messages -> 12 kept, oldest dropped first,
#     pending-order messages retained
# ---------------------------------------------------------------------------


def test_trim_keeps_twelve_drops_oldest_and_retains_pending_order(fake_agents):
    session = make_session()
    for i in range(11):  # turns 1-11: plain Q&A (22 messages)
        _turn(session, f"filler question {i}")
    _turn(session, "I'll take 2 kettles")  # turn 12: basket becomes pending
    for i in range(3):  # turns 13-15
        _turn(session, f"more chat {i}")

    # 15 turns = 30 messages -> trimmed to the cap.
    assert len(session.history) == MAX_HISTORY_MESSAGES == 12
    # Oldest dropped FIRST: the first kept message is turn 10's user message.
    assert session.history[0] == {"role": "user", "content": "filler question 9"}
    # The pending-order discussion (from the basket mention on) is retained.
    contents = [message["content"] for message in session.history]
    assert "I'll take 2 kettles" in contents
    flags = session.order_context_flags()
    assert len(flags) == 12
    assert all(flags[-8:])  # basket mention + everything after is flagged
    assert not any(flags[:4])  # the trimmed remainder starts as plain Q&A
    assert session.basket == {"KTL-01": 2}


def test_trim_without_order_context_keeps_exactly_the_cap(fake_agents):
    session = make_session()
    for i in range(16):  # 16 turns = 32 plain messages
        _turn(session, f"plain question {i}")
    assert len(session.history) == MAX_HISTORY_MESSAGES
    assert session.history[0]["content"] == "plain question 10"  # oldest dropped
    assert all(not flag for flag in session.order_context_flags())


def test_pending_order_protection_wins_over_the_cap(fake_agents):
    session = make_session()
    _turn(session, "I'll take 1 kettle")  # pending from turn 1 on
    for i in range(9):
        _turn(session, f"still deciding {i}")
    # Every message is pending-order context: protections win, the history
    # keeps growing (the cap is soft while an order is pending). The basket
    # itself is never lost — the recap re-injects it every turn regardless.
    assert len(session.history) == 20
    assert all(session.order_context_flags())
    assert session.basket == {"KTL-01": 1}


# ---------------------------------------------------------------------------
# (b) Confirmation flow: Python truth, zero model calls, never silent
# ---------------------------------------------------------------------------


def test_confirmation_finalizes_in_python_with_recomputed_total(fake_agents):
    fake_fast, fake_desk = fake_agents
    session = make_session()
    _turn(session, "I'll take 2 kettles")
    fast_before, desk_before = fake_fast.calls, fake_desk.calls

    reply = _turn(session, "Great — confirm the order.")

    # Zero model calls: the confirmation path is pure Python (FR-5).
    assert (fake_fast.calls, fake_desk.calls) == (fast_before, desk_before)
    # Python-rendered summary from the validated Order (catalogue-true).
    assert "confirmed" in reply.lower()
    assert "KTL-01" in reply and "Electric kettle 1.7L" in reply
    assert "8,400" in reply  # 2 x 4,200 recomputed in Python
    assert "Total" in reply
    order = session.draft_order
    assert order is not None and order.status == "confirmed"
    assert recompute_total(order) == order.total == 8400.0
    assert order.order_id == "ORD-0001"
    assert session.basket == {}  # consumed by the confirmation
    assert session.history[-1]["role"] == "assistant"


def test_confirmation_problems_reported_politely_order_stays_draft(fake_agents):
    fake_fast, fake_desk = fake_agents
    session = make_session()
    _turn(session, "I'll take 1 fan")  # FAN-22 has 0 stock
    fast_before, desk_before = fake_fast.calls, fake_desk.calls

    reply = _turn(session, "confirm the order")

    # Zero model calls; the problems are reported, never silently accepted.
    assert (fake_fast.calls, fake_desk.calls) == (fast_before, desk_before)
    assert "sorry" in reply.lower()
    assert "FAN-22" in reply
    assert "in stock" in reply  # validate_order's problem sentence
    assert session.draft_order is not None
    assert session.draft_order.status == "draft"  # NOT accepted
    assert session.basket == {"FAN-22": 1}  # basket kept for a corrected retry


def test_confirm_with_empty_basket_routes_to_the_desk(fake_agents):
    fake_fast, fake_desk = fake_agents
    session = make_session()
    reply = _turn(session, "confirm please")
    assert fake_desk.calls == 1 and fake_fast.calls == 0  # desk turn, no finalize
    assert reply == DESK_REPLY
    assert session.draft_order is None
    assert session.basket == {}


# ---------------------------------------------------------------------------
# (c) Basket accumulation: deterministic statements, unknowns ignored
# ---------------------------------------------------------------------------


def test_parse_basket_statement_patterns():
    assert parse_basket_statement("I'll take 2 kettles") == ({"KTL-01": 2}, [], False)
    assert parse_basket_statement("add 3 blenders") == ({"BLD-07": 3}, [], False)
    assert parse_basket_statement("and a blender as well") == ({"BLD-07": 1}, [], False)
    assert parse_basket_statement("2x KTL-01") == ({"KTL-01": 2}, [], False)
    assert parse_basket_statement("KTL-01 x 2") == ({"KTL-01": 2}, [], False)
    assert parse_basket_statement("3 of the steam irons") == ({"IRN-12": 3}, [], False)
    assert parse_basket_statement("one pedestal fan please") == ({"FAN-22": 1}, [], False)
    assert parse_basket_statement("2 kettles and 3 blenders") == (
        {"KTL-01": 2, "BLD-07": 3},
        [],
        False,
    )
    # Unknown products and plain questions parse to nothing.
    assert parse_basket_statement("I'll take 2 unicorns and 4 dragons") == ({}, [], False)
    assert parse_basket_statement("what time do you open?") == ({}, [], False)


def test_session_basket_accumulates_and_ignores_unknown_products(fake_agents):
    session = make_session()
    _turn(session, "I'll take 2 kettles")
    assert session.basket == {"KTL-01": 2}
    _turn(session, "add 3 blenders")
    assert session.basket == {"KTL-01": 2, "BLD-07": 3}
    _turn(session, "I'll take 2 unicorns please")  # ignored safely
    assert session.basket == {"KTL-01": 2, "BLD-07": 3}


def test_basket_removal_and_clear_all(fake_agents):
    session = make_session()
    _turn(session, "I'll take 2 kettles")
    _turn(session, "and a blender")
    assert session.basket == {"KTL-01": 2, "BLD-07": 1}
    _turn(session, "ok forget the blender")
    assert session.basket == {"KTL-01": 2}
    _turn(session, "please clear the basket")
    assert session.basket == {}


# ---------------------------------------------------------------------------
# (d) Isolation: two sessions never share a basket / history / ledger
# ---------------------------------------------------------------------------


def test_two_sessions_never_share_state(fake_agents):
    s1 = DeskSessionRegistry.create(
        "sess-A", ShopContext(shop="Al-Noor Electronics", currency="PKR", customer_id="A-1")
    )
    s2 = DeskSessionRegistry.create(
        "sess-B", ShopContext(shop="Al-Noor Electronics", currency="PKR", customer_id="B-1")
    )
    assert DeskSessionRegistry.get("sess-A") is s1
    assert DeskSessionRegistry.get("sess-B") is s2

    _turn(s1, "I'll take 2 kettles")
    assert s1.basket == {"KTL-01": 2}
    assert s2.basket == {}  # the second session's basket is untouched
    assert s2.history == []
    assert len(s1.ledger.records) == 1 and len(s2.ledger.records) == 0

    # Mutating one session never touches the other.
    s1.basket["BLD-07"] = 1
    s1.history.append({"role": "user", "content": "s1 only"})
    assert s2.basket == {}
    assert s2.history == []

    _turn(s2, "How much is the kettle?")  # fast path on s2's own ledger
    assert len(s2.ledger.records) == 1 and len(s1.ledger.records) == 1
    assert s1.cost_line() != "" and "fast-path: 1" in s2.cost_line()


# ---------------------------------------------------------------------------
# (e) Cost line exposes kinds after fake-model turns
# ---------------------------------------------------------------------------


def test_cost_line_exposes_kinds_and_budget(fake_agents):
    session = make_session()
    _turn(session, "How much is the kettle?")  # fast path: one call
    _turn(session, "Tell me about your shop.")  # desk loop: one call
    line = session.cost_line()
    assert "fast-path: 1" in line and "desk: 1" in line
    assert session.ledger.records[0].kind == "fast"
    assert session.ledger.records[1].kind == "desk"
    assert session.ledger.records[0].total_tokens == 5  # real usage from the fake
    assert session.budget.used == 2


# ---------------------------------------------------------------------------
# (f) Fast path vs desk routing inside handle_user_message
# ---------------------------------------------------------------------------


def test_routing_fastpath_vs_desk(fake_agents):
    fake_fast, fake_desk = fake_agents
    session = make_session()

    _turn(session, "How much is the kettle?")
    assert fake_fast.calls == 1 and fake_desk.calls == 0
    assert session.last_route is not None and session.last_route.startswith("fast-path")

    _turn(session, "What's your delivery like?")  # order word -> desk
    assert fake_desk.calls == 1 and fake_fast.calls == 1
    assert session.last_route is not None and session.last_route.startswith("desk")


# ---------------------------------------------------------------------------
# (g) FR-12's turn-11 memory: the recap carries the basket past the trim
# ---------------------------------------------------------------------------


def test_turn_eleven_still_remembers_the_basket_via_the_recap(fake_agents):
    fake_fast, fake_desk = fake_agents
    session = make_session()
    _turn(session, "I'll take 2 kettles")  # turn 1 mentions the kettles
    for i in range(9):  # turns 2-10
        _turn(session, f"chatter {i}")
    reply = _turn(session, "so what was the total for those kettles again?")  # turn 11

    assert session.turn_counter == 11
    assert reply == DESK_REPLY
    # The desk input carried the basket recap regardless of history trimming:
    # the recap mechanism, not the history, carries the basket.
    text = _input_text(fake_desk.inputs[-1])
    assert "Session note" in text
    assert "KTL-01" in text
    assert "2x KTL-01" in text
    assert "those kettles" in text  # the customer's own words travelled too
    # And the basket itself is intact (the trim never touches session state).
    assert session.basket == {"KTL-01": 2}
