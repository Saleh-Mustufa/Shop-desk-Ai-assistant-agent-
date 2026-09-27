"""Offline tests for the §7 router (no network, no API key needed).

Covers: registry tiers/limits, resolve_chain precedence for the three FR-1
levels, PRIORITY_MODEL prepending, availability filtering, the switching state
machine with injected fake models (429 backoff, RPD cooldown, local windows),
switch logging, the exhausted-chain error, the env-tunable limits, config
fail-fast, and the no-model-name-outside-model_config.py rule.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import openai
import pytest
from agents import (
    AgentOutputSchemaBase,
    Handoff,
    Model,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    Tool,
    Usage,
    UserError,
)
from agents.items import TResponseInputItem, TResponseStreamEvent
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

import config
import model_config

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers: env hygiene, fakes, response builders
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env_and_state(monkeypatch):
    """Keep tests independent of the developer's .env and of router state."""
    monkeypatch.delenv("PRIORITY_MODEL", raising=False)
    monkeypatch.delenv("SHOP_DEFAULT_PROFILE", raising=False)
    model_config.reset_state()
    yield
    model_config.reset_state()


def _rate_limit_error(message: str = "Resource exhausted: quota exceeded. Please retry in 20s."):
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(429, request=request, json={"error": {"message": message}})
    return openai.RateLimitError(message, response=response, body=None)


def _model_response(text: str = "ok") -> ModelResponse:
    message = ResponseOutputMessage(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
    )
    return ModelResponse(
        output=[message],
        usage=Usage(requests=1, input_tokens=1, output_tokens=1, total_tokens=3),
        response_id="resp_1",
    )


class FakeModel(Model):
    """Scripted Model: each call pops one item (response) or raises it."""

    def __init__(self, script: list[ModelResponse | BaseException]) -> None:
        self._script: deque[ModelResponse | BaseException] = deque(script)
        self.calls = 0
        self.inputs: list[str | list[TResponseInputItem]] = []

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
        prompt=None,
    ) -> ModelResponse:
        self.calls += 1
        self.inputs.append(input)
        item = self._script[0] if len(self._script) == 1 else self._script.popleft()
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, ModelResponse)
        return item

    def stream_response(
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
        prompt=None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        raise NotImplementedError("fake models do not stream")


def _fake_routed_model(script_for_first, **kwargs) -> tuple[model_config.RoutedModel, dict[str, FakeModel]]:
    """RoutedModel over the fast chain where candidate 1 follows `script_for_first`."""
    chain = model_config.resolve_chain("fast")
    fakes: dict[str, FakeModel] = {}
    for position, name in enumerate(chain):
        fakes[name] = FakeModel(script_for_first() if position == 0 else [_model_response()])
    return model_config.RoutedModel("fast", delegate_factory=lambda name: fakes[name], **kwargs), fakes


def _get_response(routed: model_config.RoutedModel, prompt: str = "hello router"):
    return asyncio.run(
        routed.get_response(
            None,
            prompt,
            ModelSettings(),
            [],
            None,
            [],
            ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
    )


# ---------------------------------------------------------------------------
# (a) Registry tiers and limits
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_names_exact(self):
        assert set(model_config.MODEL_REGISTRY) == {
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
        }

    def test_tiers_and_rpm(self):
        lite = {n: e for n, e in model_config.MODEL_REGISTRY.items() if e.tier == "lite15"}
        flash = {n: e for n, e in model_config.MODEL_REGISTRY.items() if e.tier == "flash5"}
        assert set(lite) == {"gemini-3.5-flash-lite", "gemini-3.1-flash-lite"}
        assert set(flash) == {
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
        }
        assert all(e.rpm == 15 for e in lite.values())
        assert all(e.rpm == 5 for e in flash.values())

    def test_limit_fields_present_and_env_overridable_shape(self):
        for entry in model_config.MODEL_REGISTRY.values():
            assert isinstance(entry.rpm, int) and entry.rpm > 0
            assert entry.tpm is None or isinstance(entry.tpm, int)
            assert entry.rpd is None or isinstance(entry.rpd, int)

    def test_gemini_2_5_flash_unavailable_for_provenance(self):
        entry = model_config.MODEL_REGISTRY["gemini-2.5-flash"]
        assert entry.available is False
        assert all(e.available for n, e in model_config.MODEL_REGISTRY.items() if n != "gemini-2.5-flash")

    def test_env_overrides_limits(self, monkeypatch):
        monkeypatch.setenv("RPM_GEMINI_3_8_FLASH", "9")
        monkeypatch.setenv("TPM_GEMINI_3_8_FLASH", "123")
        monkeypatch.setenv("RPD_GEMINI_3_8_FLASH", "45")
        try:
            importlib.reload(model_config)
            entry = model_config.MODEL_REGISTRY["gemini-3.8-flash"]
            assert (entry.rpm, entry.tpm, entry.rpd) == (9, 123, 45)
        finally:
            monkeypatch.delenv("RPM_GEMINI_3_8_FLASH", raising=False)
            monkeypatch.delenv("TPM_GEMINI_3_8_FLASH", raising=False)
            monkeypatch.delenv("RPD_GEMINI_3_8_FLASH", raising=False)
            importlib.reload(model_config)


# ---------------------------------------------------------------------------
# (b) resolve_chain precedence — the three FR-1 levels
# ---------------------------------------------------------------------------


class TestResolveChain:
    def test_profile_chain_default_fast(self):
        chain = model_config.resolve_chain()
        assert chain[0] == "gemini-3.5-flash-lite"
        assert chain[1] == "gemini-3.1-flash-lite"
        assert chain[2] == "gemini-3.8-flash"  # flash5 tier follows in registry order

    def test_reasoning_chain_strongest_first(self):
        chain = model_config.resolve_chain("reasoning")
        assert chain[0] == "gemini-3.8-flash"
        assert chain[-2:] == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]  # lite as last resort

    def test_agent_override_wins_over_profile(self):
        assert model_config.resolve_chain(agent_override="gemini-3.6-flash") == ["gemini-3.6-flash"]
        assert model_config.resolve_chain(agent_override="reasoning") == list(
            model_config.resolve_chain("reasoning")
        )

    def test_run_override_wins_over_agent_override(self):
        chain = model_config.resolve_chain(
            agent_override="gemini-3.6-flash", run_override="gemini-3.8-flash"
        )
        assert chain == ["gemini-3.8-flash"]
        chain = model_config.resolve_chain(
            agent_override="gemini-3.6-flash", run_override="reasoning"
        )
        assert chain == list(model_config.resolve_chain("reasoning"))

    def test_unknown_names_raise_user_error(self):
        with pytest.raises(UserError) as exc:
            model_config.resolve_chain(profile="no-such-profile")
        assert "\n" not in str(exc.value)
        with pytest.raises(UserError):
            model_config.resolve_chain(run_override="gemini-nope")
        with pytest.raises(UserError):
            model_config.resolve_chain(agent_override="gemini-nope")

    def test_unavailable_override_raises_user_error(self):
        with pytest.raises(UserError) as exc:
            model_config.resolve_chain(run_override="gemini-2.5-flash")
        assert "unavailable" in str(exc.value)


# ---------------------------------------------------------------------------
# (c) PRIORITY_MODEL prepends
# ---------------------------------------------------------------------------


class TestPriorityModel:
    def test_priority_model_prepends_to_profile_chain(self, monkeypatch):
        base = [m for m in model_config.PROFILES["fast"] if model_config.MODEL_REGISTRY[m].available]
        monkeypatch.setenv("PRIORITY_MODEL", "gemini-3.6-flash")
        chain = model_config.resolve_chain("fast")
        assert chain == ["gemini-3.6-flash"] + [m for m in base if m != "gemini-3.6-flash"]

    def test_priority_model_prepends_to_override_chain(self, monkeypatch):
        monkeypatch.setenv("PRIORITY_MODEL", "gemini-3.7-flash")
        assert model_config.resolve_chain(run_override="gemini-3.8-flash") == [
            "gemini-3.7-flash",
            "gemini-3.8-flash",
        ]

    def test_priority_model_ignored_when_unavailable_or_unknown(self, monkeypatch):
        base = model_config.resolve_chain("fast")
        monkeypatch.setenv("PRIORITY_MODEL", "gemini-2.5-flash")
        assert model_config.resolve_chain("fast") == base
        monkeypatch.setenv("PRIORITY_MODEL", "gemini-nope")
        assert model_config.resolve_chain("fast") == base

    def test_no_duplicate_when_priority_equals_first_candidate(self, monkeypatch):
        monkeypatch.setenv("PRIORITY_MODEL", "gemini-3.5-flash-lite")
        chain = model_config.resolve_chain("fast")
        assert chain.count("gemini-3.5-flash-lite") == 1


# ---------------------------------------------------------------------------
# (d) Chains skip unavailable models
# ---------------------------------------------------------------------------


class TestAvailabilityFiltering:
    def test_no_chain_contains_unavailable_model(self):
        unavailable = {n for n, e in model_config.MODEL_REGISTRY.items() if not e.available}
        assert unavailable == {"gemini-2.5-flash"}
        for chain in (
            model_config.resolve_chain("fast"),
            model_config.resolve_chain("reasoning"),
            model_config.resolve_chain(agent_override="reasoning"),
            model_config.resolve_chain(run_override="fast"),
        ):
            assert not (set(chain) & unavailable)

    def test_routed_model_chain_matches_resolve_chain(self):
        routed = model_config.get_routed_model("fast")
        assert routed.chain == tuple(model_config.resolve_chain("fast"))
        assert routed.active_model_name == model_config.resolve_chain("fast")[0]


# ---------------------------------------------------------------------------
# (e)+(f) Switching state machine with injected fake models
# ---------------------------------------------------------------------------


class TestSwitching:
    def test_429_on_first_candidate_switches_to_second(
        self, caplog, monkeypatch
    ):
        monkeypatch.setattr(model_config, "BACKOFF_BASE_SECONDS", 0.0)
        routed, fakes = _fake_routed_model(
            lambda: [_rate_limit_error()]  # repeat-last semantics: always raise
        )
        chain = routed.chain
        with caplog.at_level(logging.WARNING, logger="shopdesk.router"):
            response = _get_response(routed, "hello router")
        # Same request retried on candidate 2 and its response returned unchanged.
        assert response is fakes[chain[1]]._script[0]
        assert all(i == "hello router" for i in fakes[chain[0]].inputs)
        assert fakes[chain[1]].inputs == ["hello router"]
        # Max 3 attempts on the same model, then switch.
        assert fakes[chain[0]].calls == model_config.MAX_ATTEMPTS_PER_MODEL
        assert fakes[chain[1]].calls == 1
        assert routed.active_model_name == chain[1]
        # Switch logged: model out, model in, reason.
        assert "switching out" in caplog.text
        assert chain[0] in caplog.text and chain[1] in caplog.text

    def test_backoff_doubles_from_base_two_seconds(self, monkeypatch):
        delays: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            delays.append(seconds)

        monkeypatch.setattr(model_config.asyncio, "sleep", fake_sleep)
        # Raise twice, then succeed on the 3rd attempt -> stay on candidate 1.
        routed, fakes = _fake_routed_model(
            lambda: [_rate_limit_error(), _rate_limit_error(), _model_response()]
        )
        response = _get_response(routed)
        assert delays == [2.0, 4.0]
        assert routed.active_model_name == routed.chain[0]
        assert response.usage.total_tokens == 3

    def test_all_candidates_fail_raises_one_sentence_user_error(self, monkeypatch):
        monkeypatch.setattr(model_config, "BACKOFF_BASE_SECONDS", 0.0)
        chain = model_config.resolve_chain("fast")
        fakes = {name: FakeModel([_rate_limit_error()]) for name in chain}
        routed = model_config.RoutedModel("fast", delegate_factory=lambda name: fakes[name])
        with pytest.raises(model_config.RoutedModelExhaustedError) as exc:
            _get_response(routed)
        assert isinstance(exc.value, UserError)
        message = str(exc.value)
        assert "\n" not in message  # one sentence, no stack-trace noise
        assert message.rstrip().endswith(".")
        assert "failed" in message
        assert all(fakes[name].calls == model_config.MAX_ATTEMPTS_PER_MODEL for name in chain)

    def test_rpd_quota_marks_cooldown_until_reset(self, caplog):
        per_day = _rate_limit_error(
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier: quota exceeded per day"
        )
        chain = model_config.resolve_chain("fast")
        fakes1 = {
            name: FakeModel([per_day]) if position == 0 else FakeModel([_model_response()])
            for position, name in enumerate(chain)
        }
        routed1 = model_config.RoutedModel("fast", delegate_factory=lambda name: fakes1[name])
        with caplog.at_level(logging.WARNING, logger="shopdesk.router"):
            _get_response(routed1)
            # Candidate 1 is now cooling down until the daily reset.
            assert chain[0] in model_config._cooldown_until
            assert "model out" in caplog.text and chain[0] in caplog.text
            assert fakes1[chain[0]].calls == 1  # RPD marks out on the first 429, no retries
            # A fresh routed model (active index 0) skips the cooled-down candidate
            # without attempting it and is served by candidate 2.
            fakes2 = {name: FakeModel([_model_response()]) for name in chain}
            routed2 = model_config.RoutedModel("fast", delegate_factory=lambda name: fakes2[name])
            _get_response(routed2)
            assert fakes2[chain[0]].calls == 0
        assert "cooldown until" in caplog.text
        assert routed2.active_model_name == chain[1]
        model_config.reset_state()
        assert chain[0] not in model_config._cooldown_until

    def test_timeout_switches_immediately_without_backoff(self, monkeypatch):
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        timeout_error = openai.APITimeoutError(request=request)
        routed, fakes = _fake_routed_model(lambda: [timeout_error])
        _get_response(routed)
        # Immediate switch: exactly one attempt on candidate 1, no backoff retries.
        assert fakes[routed.chain[0]].calls == 1
        assert routed.active_model_name == routed.chain[1]

    def test_local_rpm_window_full_skips_candidate(self):
        chain = model_config.resolve_chain("fast")
        for _ in range(model_config.MODEL_REGISTRY[chain[0]].rpm):
            model_config._record_success(chain[0], _model_response())
        routed = model_config.RoutedModel(
            "fast", delegate_factory=lambda name: FakeModel([_model_response()])
        )
        _get_response(routed)
        # Candidate 1 is locally exhausted -> candidate 2 serves without attempts on 1.
        assert routed.active_model_name == chain[1]

    def test_window_elapsed_makes_model_usable_again(self):
        """Read-time pruning: expired window entries must not block the model forever."""
        chain = model_config.resolve_chain("fast")
        name = chain[0]
        entry = model_config.MODEL_REGISTRY[name]
        stale = time.monotonic() - (model_config.RPM_WINDOW_SECONDS + 1)
        model_config._usage_rpm[name] = deque([stale] * entry.rpm)
        # Sanity: with only stale entries the model would have been skipped forever
        # before read-time pruning (a skipped model never records new successes).
        routed = model_config.RoutedModel(
            "fast", delegate_factory=lambda n: FakeModel([_model_response()])
        )
        _get_response(routed)
        # The expired window was pruned at read time -> candidate 1 serves again.
        assert routed.active_model_name == name
        # Stale entries were dropped on read; only the fresh success remains.
        remaining = model_config._usage_rpm[name]
        assert len(remaining) == 1
        assert time.monotonic() - remaining[0] < model_config.RPM_WINDOW_SECONDS

    def test_expired_rpd_cooldown_is_cleared_on_read(self):
        """An RPD cooldown whose reset instant has passed is removed, not just ignored."""
        chain = model_config.resolve_chain("fast")
        name = chain[0]
        model_config._cooldown_until[name] = datetime.now(timezone.utc) - timedelta(seconds=1)
        routed = model_config.RoutedModel(
            "fast", delegate_factory=lambda n: FakeModel([_model_response()])
        )
        _get_response(routed)
        assert routed.active_model_name == name
        assert name not in model_config._cooldown_until

    def test_reset_state_clears_counters_and_cooldowns(self):
        chain = model_config.resolve_chain("fast")
        model_config._record_success(chain[0], _model_response())
        model_config._cooldown_until[chain[0]] = model_config._next_pacific_midnight()
        model_config.reset_state()
        assert not model_config._usage_rpm and not model_config._usage_tpm and not model_config._usage_rpd
        assert not model_config._cooldown_until


# ---------------------------------------------------------------------------
# RoutedModelProvider (SDK integration point)
# ---------------------------------------------------------------------------


class TestProvider:
    def test_provider_resolves_profiles_and_models(self):
        provider = model_config.RoutedModelProvider()
        by_profile = provider.get_model("reasoning")
        assert by_profile.chain == tuple(model_config.resolve_chain("reasoning"))
        by_model = provider.get_model("gemini-3.6-flash")
        assert by_model.chain == ("gemini-3.6-flash",)
        assert provider.get_model("reasoning") is by_profile  # cached

    def test_provider_rejects_unknown_and_empty(self):
        provider = model_config.RoutedModelProvider()
        with pytest.raises(UserError):
            provider.get_model("gemini-nope")
        with pytest.raises(UserError):
            provider.get_model(None)


# ---------------------------------------------------------------------------
# config.py fail-fast (NFR-1)
# ---------------------------------------------------------------------------


class TestConfig:
    def test_missing_key_fails_fast_one_sentence(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(SystemExit) as exc:
            config.get_gemini_api_key()
        message = str(exc.value)
        assert "\n" not in message and "GEMINI_API_KEY" in message

    def test_present_key_returned_without_echo(self):
        # The key itself is never printed anywhere; we only assert it resolves.
        assert isinstance(config.get_gemini_api_key(), str)
        assert config.get_gemini_api_key() != ""

    def test_shop_default_profile_default_fast(self, monkeypatch):
        monkeypatch.delenv("SHOP_DEFAULT_PROFILE", raising=False)
        assert config.shop_default_profile() == "fast"
        monkeypatch.setenv("SHOP_DEFAULT_PROFILE", "reasoning")
        assert config.shop_default_profile() == "reasoning"

    def test_priority_model_default_none(self, monkeypatch):
        monkeypatch.delenv("PRIORITY_MODEL", raising=False)
        assert config.priority_model() is None


# ---------------------------------------------------------------------------
# (g) No model name outside model_config.py
# ---------------------------------------------------------------------------


def _scanned_python_files():
    """All repo *.py except model_config.py, tests/, scripts/, .agents/ and hidden dirs."""
    for path in sorted(_REPO_ROOT.rglob("*.py")):
        parts = path.relative_to(_REPO_ROOT).parts
        if any(part.startswith(".") for part in parts):
            continue
        if any(part in {"tests", "scripts", "tasks", "__pycache__"} for part in parts):
            continue
        if len(parts) == 1 and parts[0] == "model_config.py":
            continue
        yield path


def test_no_model_names_outside_router():
    """Ruling 2: model identity may only appear in model_config.py."""
    for path in _scanned_python_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name in model_config.MODEL_REGISTRY:
            assert name not in text, (
                f"{path} mentions registry model name '{name}'; "
                "model identity may only appear in model_config.py "
                "(scripts must import MODEL_REGISTRY instead)."
            )
