"""Tracing setup for Shop Desk (FR-13) — one conversation is ONE trace, persisted
as ONE JSON file, with an optional upload to OpenAI's platform.

What this module provides:

- :func:`setup_tracing` — idempotent global setup. It REPLACES the SDK's
  processor set with exactly one :class:`JsonlTraceProcessor` (this module's
  singleton), so without ``OPENAI_API_KEY`` the SDK's default
  ``BackendSpanExporter`` is never installed and never attempts an upload —
  no 401 noise in the logs. When ``OPENAI_API_KEY`` IS set (checked at call
  time), a ``BatchTraceProcessor(BackendSpanExporter())`` is appended so the
  same traces also upload to the OpenAI platform under the user's own key.
- :class:`JsonlTraceProcessor` — buffers finished spans per trace; when a
  trace ends it REWRITES ``traces/<trace_id>.json`` with the WHOLE accumulated
  buffer for that trace id (merge semantics, see the class docstring), so the
  file holds the trace id, workflow name, group id, metadata and one record
  per span (name/type, started_at/ended_at, parent_id, and for model spans
  the model name plus usage token counts, extracted defensively — usage may
  be a dict or an object depending on the span data type, and may be missing
  entirely). The processor NEVER raises into the app: every I/O step is
  wrapped and failures are logged to ``shopdesk.tracing``.
- :func:`conversation_trace` — the FR-13 context manager: ONE
  ``agents.trace(workflow_name="shop-desk-conversation", group_id=session_id)``
  spanning a whole conversation (every turn's ``Runner.run``, including the
  nested pricing-specialist run inside ``get_price_quote``, lands in that
  same trace because the SDK attaches spans to the current trace).

The trace id: the SDK generates ``trace_<32 hex>`` ids when none is given
(``tracing.util.gen_trace_id``) and does not validate custom ones, but the
OpenAI upload path expects that exact shape. When a conversation id is
supplied and is not already in that shape, a valid trace id is derived from
it deterministically (SHA-256, first 32 hex chars) so the same conversation
id always maps to the same trace id; ``None`` lets the SDK generate one.

Output: ``traces/`` (module constant :data:`TRACES_DIR`, created lazily at
write time; already listed in ``.gitignore``). One file per trace, JSON
formatted — the per-span records inside are the "JSONL" line items. Token
figures in traces are tokens only, never currency amounts (same rule as the
FR-11 cost ledger).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from agents.tracing import Trace, TracingProcessor, set_trace_processors, trace
from agents.tracing.processors import BackendSpanExporter, BatchTraceProcessor
from agents.tracing.setup import get_trace_provider

__all__ = [
    "TRACES_DIR",
    "WORKFLOW_NAME",
    "MAX_COMPLETED_TRACE_BUFFERS",
    "JsonlTraceProcessor",
    "conversation_trace",
    "get_trace_processor",
    "installed_processors",
    "setup_tracing",
]

logger = logging.getLogger("shopdesk.tracing")

# Trace files land here (created lazily on first write). Already covered by
# .gitignore ("traces/").
TRACES_DIR = Path("traces")

# One whole conversation = one trace, under this workflow name (FR-13).
WORKFLOW_NAME = "shop-desk-conversation"

# The SDK's own trace-id shape (gen_trace_id: "trace_" + uuid4().hex).
_TRACE_ID_RE = re.compile(r"^trace_[0-9a-f]{32}$")

# Memory bound for merge semantics: completed traces' span buffers are kept
# only for the last 4 completed trace ids (LRU-evicted after each write), so
# the UI's per-turn scopes can share one id and still accumulate without the
# memory growing with the number of conversations.
MAX_COMPLETED_TRACE_BUFFERS = 4

_PROCESSOR: "JsonlTraceProcessor | None" = None
_PROCESSOR_LOCK = threading.Lock()


def _derive_trace_id(conversation_id: str) -> str:
    """A valid ``trace_<32 hex>`` id, deterministically derived from one id."""
    digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:32]
    return f"trace_{digest}"


def _usage_totals(span_data: Any) -> dict[str, int] | None:
    """Token counts from a span's usage, defensively.

    Model spans carry usage either as a dict (``GenerationSpanData`` /
    ``ResponseSpanData`` in the installed SDK) or as an object with token
    attributes; anything missing or malformed counts as zero. Returns ``None``
    when the span has no usage at all (non-model spans).
    """
    usage = getattr(span_data, "usage", None)
    if usage is None:
        return None

    def token(name: str) -> int:
        try:
            value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        except Exception:  # noqa: BLE001 — a hostile usage object must not break tracing
            return 0
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    return {
        "input_tokens": token("input_tokens"),
        "output_tokens": token("output_tokens"),
        "total_tokens": token("total_tokens"),
    }


def _span_record(span: Any) -> dict[str, Any] | None:
    """One serializable record for a finished span, or ``None`` to drop it.

    Defensive by contract: spans come from the SDK (or tests), so every field
    is read with ``getattr`` and nothing here is allowed to raise — a span
    that cannot be attributed to a trace is dropped, a span with partial
    fields is recorded with ``None`` placeholders.
    """
    try:
        trace_id = getattr(span, "trace_id", None)
        if not isinstance(trace_id, str) or not trace_id:
            return None  # unattributable — cannot file it under any trace
        span_data = getattr(span, "span_data", None)
        span_type = getattr(span_data, "type", None)
        name = getattr(span_data, "name", None)
        record: dict[str, Any] = {
            "name": name if isinstance(name, str) and name else str(span_type or "span"),
            "type": str(span_type) if span_type is not None else None,
            "span_id": getattr(span, "span_id", None),
            "trace_id": trace_id,
            "parent_id": getattr(span, "parent_id", None),
            "started_at": getattr(span, "started_at", None),
            "ended_at": getattr(span, "ended_at", None),
        }
        model = getattr(span_data, "model", None)
        if isinstance(model, str) and model:
            record["model"] = model  # model spans only (GenerationSpanData et al.)
        usage = _usage_totals(span_data)
        if usage is not None:
            record["usage"] = usage
        return record
    except Exception:  # noqa: BLE001 — never raise into the app (NFR-4)
        logger.exception("tracing: failed to record a finished span; the span is dropped")
        return None


class JsonlTraceProcessor(TracingProcessor):
    """Buffers finished spans per trace; on trace end rewrites one JSON file.

    MERGE SEMANTICS — one trace file per conversation trace id; repeated
    scopes with the same id merge into it: the in-memory buffer for a trace
    id is NOT popped on trace end. Every trace end REWRITES
    ``traces/<trace_id>.json`` with the WHOLE accumulated buffer for that id,
    in span-arrival order. A trace opened once and closed once (the demo
    script's single scope) behaves exactly as before; multiple scopes that
    share one trace id — the Chainlit app opens one scope per turn with the
    session-derived id — produce ONE file containing ALL of the
    conversation's spans, so the browser path satisfies FR-13 too (fast-path
    and reasoning turns are all in the file).

    Memory stays bounded: after writing, only the last
    :data:`MAX_COMPLETED_TRACE_BUFFERS` (4) COMPLETED trace ids keep their
    buffers (LRU eviction, oldest first); a conversation resumed after its
    buffer was evicted starts a fresh buffer and rewrites the file with the
    new scope's spans only — the accepted trade for bounded memory. In-flight
    traces are never evicted by completion (they are dropped by
    :meth:`shutdown`).

    Completed traces are summarized in memory (trace_id, group_id, n_spans,
    total_tokens — model-span tokens only, so the SDK's task/turn usage
    aggregates never double count) — :meth:`last_trace_summary` exposes the
    most recent one for programmatic assertions and for the session layer's
    FR-13 proof line. With repeated scopes one trace id yields one summary
    per scope end, each reflecting the file as accumulated so far.
    """

    def __init__(self, output_dir: str | Path | None = None) -> None:
        # output_dir=None resolves the module-level TRACES_DIR at WRITE time,
        # so tests can repoint the destination by monkeypatching the constant.
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._lock = threading.Lock()
        self._spans_by_trace: dict[str, list[dict[str, Any]]] = {}
        self._summaries: list[dict[str, Any]] = []
        # Completed trace ids, most recent last — the LRU order for buffer
        # eviction after writing.
        self._completed_order: list[str] = []

    # -- configuration -----------------------------------------------------

    @property
    def output_dir(self) -> Path:
        """Where trace files are written (module TRACES_DIR when not overridden)."""
        return self._output_dir if self._output_dir is not None else TRACES_DIR

    # -- TracingProcessor interface (every hook is raise-free by contract) --

    def on_trace_start(self, trace: Trace) -> None:
        """Nothing to buffer at start; spans arrive via on_span_end."""

    def on_span_start(self, span: Any) -> None:
        """Nothing to buffer at span start; only finished spans are recorded."""

    def on_span_end(self, span: Any) -> None:
        """File one finished span under its trace; never raises."""
        record = _span_record(span)
        if record is None:
            return
        try:
            with self._lock:
                self._spans_by_trace.setdefault(record["trace_id"], []).append(record)
        except Exception:  # noqa: BLE001
            logger.exception("tracing: failed to buffer a finished span; the span is dropped")

    def on_trace_end(self, trace: Trace) -> None:
        """Persist the trace as one JSON file (accumulate-and-rewrite); never raises.

        The in-memory buffer for the trace id is NOT popped: the file is
        rewritten with the WHOLE buffer for that id, so repeated scopes
        sharing one trace id (the UI's per-turn scopes) merge into ONE file
        holding every span of the conversation, in arrival order. After
        writing, completed buffers are LRU-evicted down to
        :data:`MAX_COMPLETED_TRACE_BUFFERS` (the file always exists before
        its buffer may be dropped).
        """
        trace_id = "?"  # resolved inside the try; keeps the except handler safe
        try:
            candidate = getattr(trace, "trace_id", None)
            if isinstance(candidate, str) and candidate:
                trace_id = candidate
            with self._lock:
                spans = list(self._spans_by_trace.get(trace_id, []))
                if trace_id in self._completed_order:
                    self._completed_order.remove(trace_id)
                self._completed_order.append(trace_id)
            payload = {
                "trace_id": trace_id,
                "workflow_name": getattr(trace, "name", None),
                "group_id": getattr(trace, "group_id", None),
                "metadata": getattr(trace, "metadata", None),
                "n_spans": len(spans),
                "spans": spans,
            }
            out_dir = self.output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{trace_id}.json"
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            # Token totals come from MODEL spans only: the SDK also aggregates
            # usage onto task/turn spans, and summing those would double count.
            # A model span is any span carrying a model name (generation spans
            # on the chat-completions path).
            total_tokens = sum(
                int(span.get("usage", {}).get("total_tokens", 0) or 0)
                for span in spans
                if isinstance(span.get("model"), str) and isinstance(span.get("usage"), dict)
            )
            summary = {
                "trace_id": trace_id,
                "group_id": payload["group_id"],
                "workflow_name": payload["workflow_name"],
                "n_spans": len(spans),
                "total_tokens": total_tokens,
                "file": str(path),
            }
            with self._lock:
                self._summaries.append(summary)
                # LRU-evict the OLDEST completed trace buffers AFTER writing,
                # so the file is always on disk before its buffer may go.
                while len(self._completed_order) > MAX_COMPLETED_TRACE_BUFFERS:
                    oldest = self._completed_order.pop(0)
                    self._spans_by_trace.pop(oldest, None)
            logger.info(
                "trace persisted: %s (%d spans, %d buffer(s) retained)",
                path,
                len(spans),
                len(self._spans_by_trace),
            )
        except Exception:  # noqa: BLE001 — a broken trace write must never break a turn
            logger.exception("tracing: failed to persist trace %r", trace_id)

    def force_flush(self) -> None:
        """Everything is written synchronously at trace end; nothing to flush."""

    def shutdown(self) -> None:
        """Drop all in-memory span buffers (clean close; summaries stay).

        Completed buffers linger up to :data:`MAX_COMPLETED_TRACE_BUFFERS`
        for merge semantics; shutdown clears them and any never-finished
        traces alike (files for completed traces are already on disk).
        """
        try:
            with self._lock:
                self._spans_by_trace.clear()
                self._completed_order.clear()
        except Exception:  # noqa: BLE001
            logger.exception("tracing: processor shutdown failed to clear buffers")

    # -- programmatic access (tests, session layer) -------------------------

    def last_trace_summary(self) -> dict[str, Any] | None:
        """The summary of the most recently completed trace, or ``None``."""
        with self._lock:
            return dict(self._summaries[-1]) if self._summaries else None

    def trace_summaries(self) -> list[dict[str, Any]]:
        """Copies of every completed trace summary, oldest first."""
        with self._lock:
            return [dict(summary) for summary in self._summaries]


def get_trace_processor() -> JsonlTraceProcessor:
    """The module's singleton JsonlTraceProcessor (created on first use)."""
    global _PROCESSOR
    if _PROCESSOR is None:
        with _PROCESSOR_LOCK:
            if _PROCESSOR is None:
                _PROCESSOR = JsonlTraceProcessor()
    return _PROCESSOR


def installed_processors() -> list[TracingProcessor]:
    """The currently installed SDK trace processors (best effort enumeration).

    The SDK keeps them in the default provider's multi-processor
    (``provider._multi_processor._processors``); there is no public accessor
    in the installed version, so this reads the tuple defensively.
    """
    try:
        multi = getattr(get_trace_provider(), "_multi_processor", None)
        return list(getattr(multi, "_processors", ()) or ())
    except Exception:  # noqa: BLE001 — enumeration must never break the app
        return []


def setup_tracing() -> JsonlTraceProcessor:
    """Idempotent global tracing setup (FR-13). Returns the processor.

    REPLACES the SDK's processor set with the singleton
    :class:`JsonlTraceProcessor` — the default ``BatchTraceProcessor`` (which
    uploads to OpenAI and logs 401 errors without a key) is removed. When
    ``OPENAI_API_KEY`` is set in the environment (checked at call time), a
    fresh ``BatchTraceProcessor(BackendSpanExporter())`` is appended so traces
    ALSO upload to the OpenAI platform under the user's own key.
    """
    processor = get_trace_processor()
    processors: list[TracingProcessor] = [processor]
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if api_key:
        processors.append(BatchTraceProcessor(BackendSpanExporter()))
        mode = (
            "jsonl file tracing + OpenAI platform upload "
            "(OPENAI_API_KEY is set; uploading under your own key)"
        )
    else:
        mode = (
            "jsonl file tracing only (no OPENAI_API_KEY; the SDK's default "
            "OpenAI exporter is replaced, so there are no upload attempts)"
        )
    set_trace_processors(processors)
    logger.info("Shop Desk tracing active: %s", mode)
    return processor


def conversation_trace(session_id: str, conversation_id: str | None = None) -> Trace:
    """ONE trace for a whole conversation (FR-13); use as a context manager.

    ``with conversation_trace(session_id):`` spans every turn of the
    conversation — each ``run_desk_turn`` (and the nested pricing-specialist
    run inside ``get_price_quote``) attaches its spans to the current trace,
    so one conversation produces exactly one trace file.

    ``group_id`` is the session id; ``metadata`` carries it too. When
    ``conversation_id`` is given it seeds the trace id: already-valid
    ``trace_<32 hex>`` ids are used as-is, anything else maps deterministically
    to one (same conversation id -> same trace id); ``None`` lets the SDK
    generate a fresh id.
    """
    session_id = str(session_id)
    trace_id: str | None = None
    if conversation_id is not None:
        conversation_id = str(conversation_id)
        if _TRACE_ID_RE.match(conversation_id):
            trace_id = conversation_id
        else:
            trace_id = _derive_trace_id(conversation_id)
    return trace(
        workflow_name=WORKFLOW_NAME,
        trace_id=trace_id,
        group_id=session_id,
        metadata={"session_id": session_id},
    )
