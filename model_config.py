"""The §7 model router: registry, profiles, FR-1 resolution, and RoutedModel.

ALL model identity and model logic lives ONLY in this file — no other project
file may contain a registry model name (enforced by tests/test_router.py).

Serves:
- FR-1: all three configuration levels resolve in one place via resolve_chain()
  (run-level override > agent-level override > global profile chain, with the
  PRIORITY_MODEL env prepended to every resolved chain).
- Goal §7 automatic switching: on 429 / RESOURCE_EXHAUSTED / quota errors the
  router switches to the next chain entry — exponential backoff for RPM/TPM
  limits, cooldown-until-daily-reset for RPD exhaustion, immediate switch on
  timeouts/connection errors. Every switch is logged (model out, model in,
  reason) to logger "shopdesk.router".
- Spec "Router mechanics": Gemini via its OpenAI-compatible endpoint, wrapped
  in a custom RoutedModel implementing the SDK Model protocol.

Scope note (controller ruling): automatic switching (backoff, RPD cooldown,
capacity skipping) lives on the non-streaming `get_response` path ONLY.
`stream_response` delegates to the active candidate with no switching — the
app runs non-streaming `Runner.run`, so there is no parity requirement.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Callable
from zoneinfo import ZoneInfo

import openai
from agents import (
    AgentOutputSchemaBase,
    Handoff,
    Model,
    ModelProvider,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    Tool,
    UserError,
)
from agents.items import TResponseInputItem, TResponseStreamEvent
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.retry import ModelRetryAdvice, ModelRetryAdviceRequest
from openai import AsyncOpenAI
from openai.types.responses import ResponsePromptParam

import config

__all__ = [
    "GEMINI_OPENAI_BASE_URL",
    "MODEL_REGISTRY",
    "ModelEntry",
    "PROFILES",
    "RoutedModel",
    "RoutedModelExhaustedError",
    "RoutedModelProvider",
    "get_routed_model",
    "priority_model",
    "resolve_chain",
    "reset_state",
]

logger = logging.getLogger("shopdesk.router")

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Switching behaviour (goal §7): base 2s doubling backoff, max 3 attempts on
# the same model, then switch. Module-level constants so tests can shrink them.
BACKOFF_BASE_SECONDS = 2.0
MAX_ATTEMPTS_PER_MODEL = 3

# Sliding-window sizes for the in-memory usage counters.
RPM_WINDOW_SECONDS = 60.0
TPM_WINDOW_SECONDS = 60.0
RPD_WINDOW_SECONDS = 24 * 60 * 60.0

# Gemini daily quotas reset at midnight Pacific (goal §7).
PACIFIC_TZ = "America/Los_Angeles"

_TIER_LITE = "lite15"
_TIER_FLASH = "flash5"


@dataclass(frozen=True)
class ModelEntry:
    """One registry model: its limits, tier, and live availability."""

    name: str
    rpm: int
    tpm: int | None
    rpd: int | None
    tier: str  # "lite15" (15 RPM) | "flash5" (5 RPM)
    available: bool = True


def _env_suffix(name: str) -> str:
    """Registry model name -> env suffix, e.g. gemini-3.5-flash-lite -> GEMINI_3_5_FLASH_LITE."""
    return name.upper().replace(".", "_").replace("-", "_")


def _env_int(prefix: str, name: str) -> int | None:
    """Read a per-model limit override from the env (RPM_/TPM_/RPD_ + name)."""
    raw = (os.environ.get(f"{prefix}_{_env_suffix(name)}") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("ignoring non-integer env %s_%s=%r", prefix, _env_suffix(name), raw)
        return None


def _limit(prefix: str, name: str, default: int | None) -> int | None:
    """Env override for one limit; 0 disables the limit (becomes None)."""
    value = _env_int(prefix, name)
    if value is None:
        return default
    return None if value == 0 else value


def _build_registry() -> dict[str, ModelEntry]:
    """Registry in configured order: lite15 tier first, then the flash5 tier.

    gemini-2.5-flash stays for provenance but ships available=False (live
    verification showed it is closed to new API keys) so chains skip it.
    TPM/RPD defaults are the documented free-tier figures; every limit is
    overridable via RPM_/TPM_/RPD_ env vars.
    """
    specs: list[tuple[str, int, str, int | None, int | None, bool]] = [
        ("gemini-3.5-flash-lite", 15, _TIER_LITE, 250_000, 1_000, True),
        ("gemini-3.1-flash-lite", 15, _TIER_LITE, 250_000, 1_000, True),
        ("gemini-3.8-flash", 5, _TIER_FLASH, 250_000, 250, True),
        ("gemini-3.7-flash", 5, _TIER_FLASH, 250_000, 250, True),
        ("gemini-3.6-flash", 5, _TIER_FLASH, 250_000, 250, True),
        ("gemini-3.5-flash", 5, _TIER_FLASH, 250_000, 250, True),
        ("gemini-3-flash-preview", 5, _TIER_FLASH, 250_000, 250, True),
        ("gemini-2.5-flash", 5, _TIER_FLASH, 250_000, 250, False),
    ]
    registry: dict[str, ModelEntry] = {}
    for name, rpm, tier, tpm, rpd, available in specs:
        registry[name] = ModelEntry(
            name=name,
            rpm=_limit("RPM", name, rpm) or rpm,
            tpm=_limit("TPM", name, tpm),
            rpd=_limit("RPD", name, rpd),
            tier=tier,
            available=available,
        )
    return registry


MODEL_REGISTRY: dict[str, ModelEntry] = _build_registry()

# Configured chains (registry order). resolve_chain() filters entries whose
# registry entry is unavailable, so gemini-2.5-flash never reaches a chain.
PROFILES: dict[str, tuple[str, ...]] = {
    "fast": (
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
    ),
    "reasoning": (
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
    ),
}


def priority_model() -> str | None:
    """Top-priority model name (FR-1): env PRIORITY_MODEL via config; None when unset."""
    return config.priority_model()


def _resolve_level(value: str | None, level: str) -> list[str] | None:
    """Resolve one FR-1 level: a profile name or a single registry model name.

    Returns None when the level is unset; raises agents.UserError (one
    sentence) for unknown or unavailable names.
    """
    if value is None:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate in PROFILES:
        return [name for name in PROFILES[candidate] if MODEL_REGISTRY[name].available]
    entry = MODEL_REGISTRY.get(candidate)
    if entry is None:
        known = ", ".join(list(PROFILES) + list(MODEL_REGISTRY))
        raise UserError(
            f"Unknown {level} '{candidate}': it is neither a configured profile nor a registry model (known: {known})."
        )
    if not entry.available:
        raise UserError(
            f"{level.capitalize()} '{candidate}' is marked unavailable in the registry; choose another model or profile."
        )
    return [entry.name]


def resolve_chain(
    profile: str = "fast",
    agent_override: str | None = None,
    run_override: str | None = None,
) -> list[str]:
    """Resolve the three FR-1 levels into one ordered candidate chain.

    Precedence: run_override (highest) > agent_override > profile chain. Every
    level resolves through the registry/profiles; unknown names raise
    agents.UserError with a one-sentence message. When PRIORITY_MODEL is set
    and is an available registry model it moves to the front of every resolved
    chain (tried first, listed once).
    """
    chain = _resolve_level(run_override, "run-level override")
    if chain is None:
        chain = _resolve_level(agent_override, "agent-level override")
    if chain is None:
        chain = _resolve_level(profile, "profile")
    if not chain:
        raise UserError(
            f"No available model matches the request for profile '{profile}': every candidate is marked unavailable."
        )
    pm = priority_model()
    if pm:
        entry = MODEL_REGISTRY.get(pm)
        if entry is not None and entry.available:
            # Move the priority model to the front so it is tried first; it
            # appears exactly once even when the profile chain already had it.
            chain = [pm, *[name for name in chain if name != pm]]
        else:
            logger.warning("PRIORITY_MODEL '%s' ignored: not an available registry model.", pm)
    return chain


# --- Sliding-window usage counters and RPD cooldowns (process-global, keyed by
# model name: limits are per API key, shared by every RoutedModel instance). ---

_usage_rpm: dict[str, deque[float]] = {}
_usage_tpm: dict[str, deque[tuple[float, int]]] = {}
_usage_rpd: dict[str, deque[float]] = {}
_cooldown_until: dict[str, datetime] = {}


def reset_state() -> None:
    """Clear all sliding-window counters and RPD cooldowns (test helper)."""
    _usage_rpm.clear()
    _usage_tpm.clear()
    _usage_rpd.clear()
    _cooldown_until.clear()


def _prune(name: str, now: float) -> None:
    """Drop usage entries older than each sliding window."""
    for store, window in (
        (_usage_rpm, RPM_WINDOW_SECONDS),
        (_usage_tpm, TPM_WINDOW_SECONDS),
        (_usage_rpd, RPD_WINDOW_SECONDS),
    ):
        entries = store.get(name)
        if not entries:
            continue
        while entries:
            oldest = entries[0]
            stamp = oldest[0] if isinstance(oldest, tuple) else oldest
            if now - stamp <= window:
                break
            entries.popleft()


def _record_success(name: str, response: object) -> None:
    """Record one successful call in the rpm/tpm/rpd sliding windows.

    Token totals come from the response usage when available; counting calls
    is acceptable for RPM (goal §7).
    """
    now = time.monotonic()
    _usage_rpm.setdefault(name, deque()).append(now)
    tokens = 0
    usage = getattr(response, "usage", None)
    if usage is not None:
        try:
            tokens = int(getattr(usage, "total_tokens", 0) or 0)
        except (TypeError, ValueError):
            tokens = 0
    _usage_tpm.setdefault(name, deque()).append((now, tokens))
    _usage_rpd.setdefault(name, deque()).append(now)
    _prune(name, now)


def _has_local_capacity(name: str) -> tuple[bool, str]:
    """True when the model is within its locally observed limits.

    Prunes expired entries at read time: a model that was skipped never
    records new successes, so without read-time pruning its windows would
    never free up (worst case: RPD-exhausted stays dead after the reset).
    """
    _prune(name, time.monotonic())
    entry = MODEL_REGISTRY[name]
    rpm_calls = _usage_rpm.get(name)
    if entry.rpm and rpm_calls and len(rpm_calls) >= entry.rpm:
        return False, f"local sliding-window limit reached ({entry.rpm} requests/min)"
    tpm_calls = _usage_tpm.get(name)
    if entry.tpm and tpm_calls and sum(tokens for _, tokens in tpm_calls) >= entry.tpm:
        return False, f"local sliding-window limit reached ({entry.tpm} tokens/min)"
    rpd_calls = _usage_rpd.get(name)
    if entry.rpd and rpd_calls and len(rpd_calls) >= entry.rpd:
        return False, f"local sliding-window limit reached ({entry.rpd} requests/day)"
    return True, ""


def _next_pacific_midnight(now: datetime | None = None) -> datetime:
    """Next 00:00 America/Los_Angeles (Gemini's daily quota reset), tz-aware."""
    now = now or datetime.now(timezone.utc)
    try:
        pacific = ZoneInfo(PACIFIC_TZ)
    except Exception:  # pragma: no cover — tzdata missing; fixed PST is the safe (later) bound
        pacific = timezone(timedelta(hours=-8), name="PST")
    local = now.astimezone(pacific)
    return local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)


def _short_reason(exc: BaseException) -> str:
    """One-line, bounded description of an exception for logs/error messages."""
    text = " ".join(str(exc).split())
    if len(text) > 200:
        text = text[:197] + "..."
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


class _ErrorKind(Enum):
    """How the router reacts to a failed candidate call."""

    BACKOFF = "rpm/tpm rate limit"  # exponential backoff on the same model, then switch
    RPD = "daily quota (rpd)"  # mark out until the daily reset, then switch
    SWITCH = "no cooldown"  # immediate switch, no cooldown


class RoutedModelExhaustedError(UserError):
    """Raised when every candidate in the resolved chain failed (one sentence)."""


class RoutedModel(Model):
    """SDK Model over Gemini's OpenAI-compatible endpoint with §7 switching.

    Composes one OpenAIChatCompletionsModel per chain candidate, all sharing a
    single AsyncOpenAI client built against GEMINI_OPENAI_BASE_URL with the key
    from config.get_gemini_api_key(). get_response delegates to the active
    candidate; on a limit error it switches to the next candidate and retries
    the same request there. For tests, `delegate_factory` injects fake models.
    """

    def __init__(
        self,
        profile: str = "fast",
        agent_override: str | None = None,
        run_override: str | None = None,
        *,
        delegate_factory: Callable[[str], Model] | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._profile = profile
        self._chain: list[str] = resolve_chain(profile, agent_override, run_override)
        self._client = client
        self._delegate_factory = delegate_factory
        self._delegates: dict[str, Model] = {}
        if delegate_factory is not None:
            self._delegates = {name: delegate_factory(name) for name in self._chain}
        self._active_index = 0

    # -- introspection ----------------------------------------------------

    @property
    def chain(self) -> tuple[str, ...]:
        """The resolved candidate chain, in switching order."""
        return tuple(self._chain)

    @property
    def active_model_name(self) -> str:
        """The registry name of the candidate currently serving requests."""
        return self._chain[self._active_index]

    # -- delegation plumbing ----------------------------------------------

    def _delegate_for(self, name: str) -> Model:
        """Return (building lazily) the per-candidate delegate model."""
        delegate = self._delegates.get(name)
        if delegate is None:
            if self._client is None:
                self._client = AsyncOpenAI(
                    base_url=GEMINI_OPENAI_BASE_URL, api_key=config.get_gemini_api_key()
                )
            if self._delegate_factory is None:
                factory_client = self._client
                self._delegate_factory = (
                    lambda model_name: OpenAIChatCompletionsModel(
                        model=model_name, openai_client=factory_client
                    )
                )
            delegate = self._delegate_factory(name)
            self._delegates[name] = delegate
        return delegate

    def _skip_reason(self, name: str) -> str | None:
        """Why a candidate must not be attempted now (cooldown / local limits)."""
        until = _cooldown_until.get(name)
        if until is not None:
            if datetime.now(timezone.utc) < until:
                return f"daily quota (rpd) cooldown until {until.isoformat()}"
            # Cooldown has passed (daily reset) — clear it so the model can serve again.
            _cooldown_until.pop(name, None)
        has_capacity, reason = _has_local_capacity(name)
        return None if has_capacity else reason

    def _log_switch_out(self, out: str, into: str | None, reason: str) -> None:
        """Log one switch (goal §7): model out, model in, reason."""
        if into is None:
            logger.warning("switching out %s → <no next candidate>: %s", out, reason)
        else:
            logger.warning("switching out %s → %s: %s", out, into, reason)

    @staticmethod
    def _classify(exc: BaseException) -> _ErrorKind:
        """Classify a failed candidate call (goal §7 error taxonomy)."""
        if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
            return _ErrorKind.SWITCH
        is_429 = isinstance(exc, openai.RateLimitError) or (
            isinstance(exc, openai.APIStatusError)
            and getattr(exc, "status_code", None) == 429
        )
        if is_429:
            message = str(exc).lower()
            quota_like = "resource_exhausted" in message or "quota" in message
            rpd_like = quota_like and any(
                marker in message for marker in ("perday", "per day", "daily", "rpd")
            )
            return _ErrorKind.RPD if rpd_like else _ErrorKind.BACKOFF
        return _ErrorKind.SWITCH

    # -- SDK Model protocol ------------------------------------------------

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
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        """Serve one request through the chain, switching on limit errors (§7)."""
        order = self._chain[self._active_index :] + self._chain[: self._active_index]
        last_error: BaseException | None = None
        last_candidate: str | None = None
        for position, name in enumerate(order):
            into = order[position + 1] if position + 1 < len(order) else None
            skip_reason = self._skip_reason(name)
            if skip_reason is not None:
                last_candidate = name
                self._log_switch_out(name, into, skip_reason)
                continue
            delegate = self._delegate_for(name)
            attempts = 0
            while True:
                attempts += 1
                try:
                    response = await delegate.get_response(
                        system_instructions,
                        input,
                        model_settings,
                        tools,
                        output_schema,
                        handoffs,
                        tracing,
                        previous_response_id=previous_response_id,
                        conversation_id=conversation_id,
                        prompt=prompt,
                    )
                    _record_success(name, response)
                    self._active_index = self._chain.index(name)
                    return response
                except Exception as exc:  # noqa: BLE001 — switching must never crash the run (§7)
                    last_error = exc
                    last_candidate = name
                    kind = self._classify(exc)
                    if kind is _ErrorKind.RPD:
                        until = _next_pacific_midnight()
                        _cooldown_until[name] = until
                        logger.warning("model out %s until %s", name, until.isoformat())
                        self._log_switch_out(name, into, "daily quota (rpd) exhausted")
                        break
                    if kind is _ErrorKind.SWITCH:
                        self._log_switch_out(name, into, _short_reason(exc))
                        break
                    if attempts >= MAX_ATTEMPTS_PER_MODEL:
                        self._log_switch_out(
                            name, into, f"rate limited after {attempts} attempts (rpm/tpm)"
                        )
                        break
                    await asyncio.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempts - 1)))
        attempted = last_candidate is not None
        raise RoutedModelExhaustedError(_exhausted_message(self._chain, last_candidate, last_error, attempted))

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
        prompt: ResponsePromptParam | None = None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        """Stream via the active candidate only — NO automatic switching.

        Switching, backoff and capacity/cooldown checks are deliberately a
        non-streaming `get_response` carve-in (controller ruling); the app
        uses non-streaming Runner.run, so the streaming path needs none.
        """
        return self._delegate_for(self.active_model_name).stream_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )

    def get_retry_advice(self, request: ModelRetryAdviceRequest) -> ModelRetryAdvice | None:
        """Delegate retry guidance to the active candidate when one is built."""
        delegate = self._delegates.get(self.active_model_name)
        if delegate is not None:
            return delegate.get_retry_advice(request)
        return None

    async def close(self) -> None:
        """Release delegates and the shared client (SDK Model protocol)."""
        for delegate in self._delegates.values():
            closer = getattr(delegate, "close", None)
            if closer is not None:
                result = closer()
                if asyncio.iscoroutine(result):
                    await result
        self._delegates.clear()
        if self._client is not None:
            await self._client.close()
            self._client = None


def _exhausted_message(
    chain: list[str], last_candidate: str | None, last_error: BaseException | None, attempted: bool
) -> str:
    """One-sentence summary for the exhausted-chain failure (§7)."""
    names = ", ".join(chain)
    if attempted and last_candidate is not None and last_error is not None:
        return (
            f"All {len(chain)} candidate models ({names}) failed; the last failure was on "
            f"{last_candidate}: {_short_reason(last_error)}."
        )
    return (
        f"All {len(chain)} candidate models ({names}) were skipped by cooldown or local limits, "
        "so no model is available to serve the request."
    )


def get_routed_model(
    profile: str = "fast",
    agent_override: str | None = None,
    run_override: str | None = None,
    client: AsyncOpenAI | None = None,
) -> RoutedModel:
    """Build a RoutedModel from the resolved FR-1 chain (router public API)."""
    return RoutedModel(profile, agent_override, run_override, client=client)


class RoutedModelProvider(ModelProvider):
    """SDK ModelProvider integration point: names resolve through the router.

    get_model() accepts either a profile name ("fast"/"reasoning") or a single
    registry model name; anything else raises agents.UserError via
    resolve_chain. Instances are cached per name.
    """

    def __init__(self) -> None:
        self._cache: dict[str, RoutedModel] = {}

    def get_model(self, model_name: str | None) -> RoutedModel:
        key = (model_name or "").strip()
        if not key:
            raise UserError("RoutedModelProvider.get_model needs a profile or model name; none was given.")
        routed = self._cache.get(key)
        if routed is None:
            if key in PROFILES:
                routed = RoutedModel(profile=key)
            else:
                routed = RoutedModel(run_override=key)
            self._cache[key] = routed
        return routed
