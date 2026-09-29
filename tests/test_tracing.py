"""Offline tests for tracing (FR-13) and the run_desk_turn catch-all (NFR-4).

No network and no API-key assumptions: the autouse fixture deletes
``OPENAI_API_KEY`` so ``setup_tracing()`` always runs in jsonl-only mode (the
SDK's default OpenAI exporter is REPLACED, never installed), and model calls
go through fake SDK Models injected via ``model_config.RoutedModel(
delegate_factory=...)`` — the same pattern as tests/test_runner_cost.py.

For trace content the fake models record their own generation span (exactly
what the installed ``OpenAIChatCompletionsModel`` does), so traces contain
model spans carrying the fake's model name and real usage token counts.

Global tracing state is restored after every test: processors are reset via
``set_trace_processors([])`` and the singleton processor's in-memory state is
fresh per test (the singleton is recreated only by ``setup_tracing``), so no
state leaks across the suite.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
    Tool,
    Usage,
    set_trace_processors,
    set_tracing_disabled,
)
from agents.items import TResponseInputItem
from agents.tracing import generation_span
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

import agents_desk
import model_config
import runner_cost
import tracing_setup
from context import ShopContext
from runner_cost import (
    ConversationBudget,
    ConversationLedger,
    run_desk_turn,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Exactly the repo catalogue.json content (same fixture pattern as
# tests/test_runner_cost.py; needed for the desk output guardrail).
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
def _tracing_hygiene(monkeypatch):
    """Keep tests offline (no key) and never leak global tracing state.

    Mirrors tests/test_router.py's autouse hygiene: env is cleaned, tracing is
    enabled with an EMPTY processor set, and after the test the processor set
    is cleared again so other modules never see this module's processors.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PRIORITY_MODEL", raising=False)
    monkeypatch.delenv("SHOP_TURN_CEILING", raising=False)
    model_config.reset_state()
    set_tracing_disabled(False)  # these tests need spans; normalize the flag
    set_trace_processors([])
    yield
    set_trace_processors([])
    model_config.reset_state()


@pytest.fixture()
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


def make_ctx(tier: str = "walk_in") -> ShopContext:
    return ShopContext(shop="Al-Noor Electronics", currency="PKR", customer_id="CUST-7", tier=tier)


def _usage_dict(usage: Usage) -> dict[str, int]:
    """The span-usage shape the installed SDK writes for model calls."""
    return {
        "requests": int(usage.requests or 0),
        "input_tokens": int(usage.input_tokens or 0),
        "output_tokens": int(usage.output_tokens or 0),
        "total_tokens": int(usage.total_tokens or 0),
    }


class _SpanningFakeModel(Model):
    """Fake SDK Model that answers with one message AND records a generation span.

    Mirrors the installed OpenAIChatCompletionsModel, which wraps each call in
    ``generation_span(model=..., usage=...)``: this is what puts model-name and
    usage data into the trace for the processor to persist offline.
    """

    def __init__(
        self,
        text: str,
        model_name: str,
        usage: Usage | None = None,
        usages: list[Usage] | None = None,
    ) -> None:
        self.text = text
        self.model_name = model_name
        self._usages = list(usages) if usages is not None else [usage or Usage()]
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
        usage = self._usages[min(self.calls, len(self._usages)) - 1]
        with generation_span(model=self.model_name, usage=_usage_dict(usage)):
            pass  # span start/end brackets the (instant) fake model call
        message = ResponseOutputMessage(
            id=f"msg_{self.calls}",
            type="message",
            role="assistant",
            status="completed",
            content=[ResponseOutputText(text=self.text, type="output_text", annotations=[])],
        )
        return ModelResponse(
            output=[message],
            usage=usage,
            response_id=f"resp_{self.calls}",
        )

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("fake models do not stream")


class _BoomModel(Model):
    """Fake SDK Model whose get_response raises a plain (non-SDK) RuntimeError."""

    def __init__(self) -> None:
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
        raise RuntimeError("connector exploded (non-SDK error)")

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("fake models do not stream")


class _TextFakeModel(Model):
    """Plain fake model (no spans): one message answer, per test_runner_cost."""

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
        return ModelResponse(output=[message], usage=self.usage, response_id=f"resp_{self.calls}")

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("fake models do not stream")


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


class _ScriptedModel(Model):
    """Scripted Model: each call pops one ModelResponse (test_runner_cost pattern)."""

    def __init__(self, script: list[ModelResponse]) -> None:
        from collections import deque

        self._script = deque(script)
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

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("fake models do not stream")


def _routed_fake(fake: Model) -> model_config.RoutedModel:
    """A RoutedModel whose every chain candidate serves the same fake model."""
    return model_config.RoutedModel(profile="fast", delegate_factory=lambda _name: fake)


# ---------------------------------------------------------------------------
# (a) setup_tracing is idempotent
# ---------------------------------------------------------------------------


def test_setup_tracing_is_idempotent():
    first = tracing_setup.setup_tracing()
    second = tracing_setup.setup_tracing()  # second call must not stack processors
    assert first is second  # one singleton processor
    installed = tracing_setup.installed_processors()
    jsonl = [p for p in installed if isinstance(p, tracing_setup.JsonlTraceProcessor)]
    assert len(jsonl) == 1  # exactly one JsonlTraceProcessor installed
    assert jsonl[0] is first
    assert installed == jsonl  # no OPENAI_API_KEY -> nothing else is installed
    # The default OpenAI exporter/batch processor was REPLACED, not kept.
    from agents.tracing.processors import BatchTraceProcessor

    assert not [p for p in installed if isinstance(p, BatchTraceProcessor)]


def test_setup_tracing_appends_openai_exporter_only_with_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    from agents.tracing.processors import BatchTraceProcessor

    processor = tracing_setup.setup_tracing()
    installed = tracing_setup.installed_processors()
    assert installed[0] is processor
    assert len(installed) == 2  # jsonl + one upload processor
    assert isinstance(installed[1], BatchTraceProcessor)


# ---------------------------------------------------------------------------
# (b) one conversation = one trace, persisted as one JSON file
# ---------------------------------------------------------------------------


def test_one_conversation_is_one_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(tracing_setup, "TRACES_DIR", tmp_path)
    processor = tracing_setup.setup_tracing()

    fast_fake = _SpanningFakeModel(
        "The Electric kettle 1.7L (KTL-01) costs PKR 4,200 and we have 12 in stock.",
        model_name="fake-lite",
        usage=Usage(requests=1, input_tokens=11, output_tokens=7, total_tokens=18),
    )
    desk_fake = _SpanningFakeModel(
        "Blenders are on aisle 3.",
        model_name="fake-desk",
        usage=Usage(requests=1, input_tokens=20, output_tokens=10, total_tokens=30),
    )
    fast_agent = Agent(name="FastPath", model=_routed_fake(fast_fake))
    desk_agent = Agent(name="ShopDesk", model=_routed_fake(desk_fake))
    ledger = ConversationLedger()
    budget = ConversationBudget(max_model_calls=10)
    ctx = make_ctx()

    with tracing_setup.conversation_trace("sess-1"):
        first = asyncio.run(run_desk_turn(fast_agent, "What does the kettle cost?", ctx, ledger, budget))
        second = asyncio.run(run_desk_turn(desk_agent, "tell me about blenders", ctx, ledger, budget))

    assert first == fast_fake.text and second == desk_fake.text
    summary = processor.last_trace_summary()
    assert summary is not None
    assert summary["group_id"] == "sess-1"
    assert summary["workflow_name"] == tracing_setup.WORKFLOW_NAME == "shop-desk-conversation"
    assert summary["n_spans"] > 0
    assert summary["total_tokens"] == 18 + 30  # both model spans' tokens

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1  # exactly ONE trace file for the conversation
    assert files[0].name == f"{summary['trace_id']}.json"
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["trace_id"] == summary["trace_id"]
    assert payload["workflow_name"] == "shop-desk-conversation"
    assert payload["group_id"] == "sess-1"
    assert payload["metadata"] == {"session_id": "sess-1"}
    assert payload["n_spans"] == len(payload["spans"]) == summary["n_spans"]
    model_names = {span["model"] for span in payload["spans"] if "model" in span}
    assert {"fake-lite", "fake-desk"} <= model_names  # both turns' models present


def test_conversation_trace_id_shapes(monkeypatch):
    """Valid ids pass through; anything else maps deterministically; None -> SDK."""
    valid = "trace_" + "0123abcd" * 4
    with tracing_setup.conversation_trace("s1", valid) as t1:
        assert t1.trace_id == valid
    with tracing_setup.conversation_trace("s2", "my-conversation-42") as t2:
        assert tracing_setup._TRACE_ID_RE.match(t2.trace_id)  # noqa: SLF001 — pinned shape
        again = tracing_setup.conversation_trace("s2", "my-conversation-42")
        assert again.trace_id == t2.trace_id  # deterministic derivation
    with tracing_setup.conversation_trace("s3") as t3:
        assert tracing_setup._TRACE_ID_RE.match(t3.trace_id)  # noqa: SLF001 — SDK-generated


# ---------------------------------------------------------------------------
# (c) the most expensive turn is identifiable from the trace
# ---------------------------------------------------------------------------


def test_most_expensive_turn_identifiable_by_usage(monkeypatch, tmp_path):
    monkeypatch.setattr(tracing_setup, "TRACES_DIR", tmp_path)
    processor = tracing_setup.setup_tracing()

    fake = _SpanningFakeModel(
        "ok",
        model_name="fake-usage",
        usages=[
            Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
            Usage(requests=1, input_tokens=40, output_tokens=20, total_tokens=60),
        ],
    )
    agent = Agent(name="ShopDesk", model=_routed_fake(fake))
    ledger = ConversationLedger()
    budget = ConversationBudget(max_model_calls=10)

    with tracing_setup.conversation_trace("sess-cost"):
        asyncio.run(run_desk_turn(agent, "first", make_ctx(), ledger, budget))
        asyncio.run(run_desk_turn(agent, "second", make_ctx(), ledger, budget))

    payload = json.loads(
        (tmp_path / f"{processor.last_trace_summary()['trace_id']}.json").read_text(encoding="utf-8")
    )
    model_spans = [span for span in payload["spans"] if "model" in span]
    totals = sorted(span["usage"]["total_tokens"] for span in model_spans)
    assert totals == [15, 60]  # two model spans, distinct usage per turn
    expensive = max(model_spans, key=lambda span: span["usage"]["total_tokens"])
    # The most expensive span is exactly the second turn's usage (matched by values).
    assert expensive["usage"] == {
        "input_tokens": 40,
        "output_tokens": 20,
        "total_tokens": 60,
    }
    # ...and the ledger (whose records ARE the turns) agrees.
    assert [record.total_tokens for record in ledger.records] == [15, 60]
    assert ledger.records[-1].total_tokens == expensive["usage"]["total_tokens"]


# ---------------------------------------------------------------------------
# (d) the processor never raises into the app
# ---------------------------------------------------------------------------


def test_processor_never_raises_on_garbage(tmp_path, caplog):
    processor = tracing_setup.JsonlTraceProcessor(output_dir=tmp_path)

    class _HostileSpanData:
        @property
        def usage(self) -> dict[str, int]:
            raise RuntimeError("boom")

    class _HostileTrace:
        @property
        def trace_id(self) -> str:
            raise RuntimeError("boom")

    garbage_spans = [
        SimpleNamespace(),  # no fields at all
        SimpleNamespace(trace_id=123),  # wrong type -> unattributable, dropped
        SimpleNamespace(trace_id="trace_" + "a" * 32, span_data=_HostileSpanData()),
        SimpleNamespace(
            trace_id="trace_" + "a" * 32,
            span_data=SimpleNamespace(usage="junk", model=5, type=None),
            started_at=1,
        ),
    ]
    with caplog.at_level(logging.ERROR, logger="shopdesk.tracing"):
        for span in garbage_spans:
            processor.on_span_end(span)
        processor.on_trace_end(_HostileTrace())  # raising trace_id
        processor.on_trace_end(
            SimpleNamespace(trace_id="trace_" + "b" * 32, name=7, group_id=None, metadata=None)
        )
    # Hostile inputs were logged, never raised.
    error_records = [
        record for record in caplog.records
        if record.name == "shopdesk.tracing" and record.levelno >= logging.ERROR
    ]
    assert error_records, "hostile inputs must be logged, not raised"
    # A garbage-but-string trace still produced a file with defensive placeholders.
    payload = json.loads((tmp_path / ("trace_" + "b" * 32 + ".json")).read_text(encoding="utf-8"))
    assert payload["workflow_name"] == 7  # stored as-is: extraction never crashes
    assert payload["n_spans"] == 0


# ---------------------------------------------------------------------------
# (e) run_desk_turn catch-all: non-SDK errors end the turn politely (NFR-4)
# ---------------------------------------------------------------------------


def test_cancelled_error_is_not_an_exception():
    """Pinned: except Exception must not swallow task cancellation in 3.14."""
    assert not issubclass(asyncio.CancelledError, Exception)


def test_catch_all_maps_non_sdk_runner_error_to_generic_sorry(monkeypatch):
    """A non-SDK error raised out of Runner.run hits the final except Exception."""

    async def boom(starting_agent, input, **kwargs):
        raise RuntimeError("connector exploded")

    monkeypatch.setattr(runner_cost.Runner, "run", boom)
    output = asyncio.run(
        run_desk_turn(
            Agent(name="ShopDesk"),
            "hello",
            make_ctx(),
            ConversationLedger(),
            ConversationBudget(),
        )
    )
    assert output == runner_cost.GENERIC_SORRY


def test_fake_model_runtime_error_never_raises_out():
    """End-to-end: a fake model raising a plain RuntimeError -> GENERIC_SORRY."""
    boom = _BoomModel()
    agent = Agent(name="ShopDesk", model=boom)  # raw model: no RoutedModel switching
    ledger = ConversationLedger()
    budget = ConversationBudget()
    output = asyncio.run(run_desk_turn(agent, "hello", make_ctx(), ledger, budget))
    assert output == runner_cost.GENERIC_SORRY
    assert boom.calls == 1
    assert budget.used == 1  # registered before the call...
    assert ledger.records == []  # ...but the call never completed, so no record


# ---------------------------------------------------------------------------
# (f) cost line distinguishes fast-path from reasoning turns (FR-11)
# ---------------------------------------------------------------------------


def test_cost_line_distinguishes_kinds_via_conversation_accounting(monkeypatch, fixture_catalogue):
    """One FastPath-kind turn + one desk turn with a nested PricingSpecialist
    quote (the ContextVar accounting path) -> cost_line() shows all three kinds."""
    fast_fake = _TextFakeModel(
        "The Electric kettle 1.7L (KTL-01) costs PKR 4,200 and we have 12 in stock."
    )
    fast_agent = Agent(name="FastPath", model=_routed_fake(fast_fake))

    desk_routed = _routed_fake(
        _ScriptedModel(
            [_tool_call_response("get_price_quote", '{"product": "KTL-01", "qty": 4}')]
        )
    )
    pricing_fake = _TextFakeModel(
        '{"response": 16800}',
        usage=Usage(requests=1, input_tokens=5, output_tokens=2, total_tokens=7),
    )
    pricing = agents_desk.make_pricing_specialist().clone(
        model=model_config.RoutedModel(
            profile="reasoning", delegate_factory=lambda _name: pricing_fake
        )
    )
    monkeypatch.setattr(agents_desk, "get_pricing_specialist", lambda: pricing)

    desk_agent = agents_desk.make_desk_agent().clone(
        model=desk_routed, tool_use_behavior="stop_on_first_tool"
    )
    ledger = ConversationLedger()
    budget = ConversationBudget(max_model_calls=10)

    first = asyncio.run(
        run_desk_turn(fast_agent, "What does the kettle cost?", make_ctx(), ledger, budget)
    )
    second = asyncio.run(
        run_desk_turn(desk_agent, "quote 4 kettles", make_ctx(), ledger, budget)
    )
    assert first == fast_fake.text
    assert second == "The price quote for 4 x KTL-01 is 16800"

    kinds = [record.kind for record in ledger.records]
    assert kinds == ["fast", "desk", "reasoning"]
    line = ledger.cost_line()
    assert "fast-path: 1" in line
    assert "desk: 1" in line
    assert "reasoning: 1" in line
    assert "turns: 3" in line
    assert pricing.model.active_model_name in line  # the reasoning chain entry
