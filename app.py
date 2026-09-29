"""Chainlit UI for Shop Desk (FR-12) — a THIN layer over DeskSession.

Every piece of conversation logic lives in ``session_store.DeskSession``; this
module only wires Chainlit's session lifecycle to it:

- ``@cl.on_chat_start``: validate the configuration (NFR-1 fail fast with one
  sentence), install tracing (FR-13, idempotent), build the per-session
  ``ShopContext`` (FR-2) and a fresh ``DeskSession``, and store it in
  Chainlit's own ``cl.user_session``.
- ``@cl.on_message``: run the turn through ``session.handle_user_message``
  inside a ``conversation_trace`` scope (FR-13) and send the reply.
- ``@cl.on_chat_end``: log the FR-11 cost line (also logged every 10 turns —
  never spammed into the UI).

FR-12 session isolation: the DeskSession is stored under ``cl.user_session``,
which Chainlit keys by the websocket session id — two browser windows create
two sessions with two baskets, two histories and two ledgers (the same
isolation the offline test and the demo script demonstrate through
``DeskSessionRegistry``/plain instances).

FR-13 note: a Chainlit handler runs in its own asyncio task, so a trace
opened in one handler cannot span the next handler's task (ContextVar
isolation). Each message therefore opens a ``conversation_trace`` scoped to
that turn, with a conversation id DERIVED from the session id — every turn of
one conversation shares ONE stable trace id, and the
``JsonlTraceProcessor``'s accumulate-and-rewrite semantics MERGE those
per-turn scopes into ONE conversation trace file
(``traces/<trace_id>.json``): after N turns the file carries ALL N turns'
spans in arrival order, so the browser path satisfies FR-13's "one
conversation = one trace" (open the file and the fast-path turns, the desk
turns and the most expensive turn are all there).
``scripts/demo_conversation.py`` wraps a whole conversation in a single
trace scope in one task and remains the strict single-scope reference
demonstration. No customer identifier ever enters any prompt or input text
(FR-2).
"""

from __future__ import annotations

import logging
import os

import chainlit as cl

import catalogue
import config
import session_store
import tracing_setup
from context import ShopContext
from session_store import DeskSession

logger = logging.getLogger("shopdesk.app")

_SESSION_KEY = "desk_session"
_CONFIG_ERROR_KEY = "config_error"

_VALID_TIERS = ("walk_in", "regular")


def default_tier() -> str:
    """The demo default tier: env SHOP_DEFAULT_TIER, else 'walk_in' (FR-2/FR-7)."""
    raw = (os.environ.get("SHOP_DEFAULT_TIER") or "").strip().lower()
    if raw in _VALID_TIERS:
        return raw
    if raw:
        logger.warning(
            "ignoring unsupported SHOP_DEFAULT_TIER=%r (valid: %s); using walk_in",
            raw,
            ", ".join(_VALID_TIERS),
        )
    return "walk_in"


def build_customer(session_id: str) -> ShopContext:
    """The per-session run context (FR-2): travels as context, never as text."""
    data = catalogue.load_catalogue()
    return ShopContext(
        shop=str(data.get("shop") or "Al-Noor Electronics"),
        currency=str(data.get("currency") or "PKR"),
        customer_id=f"guest-{session_id[:8]}",
        tier=default_tier(),
    )


@cl.on_chat_start
async def on_chat_start() -> None:
    """New Chainlit session -> one fresh DeskSession (FR-12 isolation)."""
    try:
        config.get_gemini_api_key()  # NFR-1: fail fast with one sentence
        config_error = None
    except SystemExit as exc:
        config_error = str(exc)
    if config_error is not None:
        cl.user_session.set(_CONFIG_ERROR_KEY, config_error)
        await cl.Message(content=config_error).send()
        return

    tracing_setup.setup_tracing()  # FR-13, idempotent
    session_id = cl.context.session.id
    session = DeskSession(session_id, build_customer(session_id))
    cl.user_session.set(_SESSION_KEY, session)
    logger.info(
        "session %s started (customer_id=%s tier=%s)",
        session.session_id,
        session.customer.customer_id,
        session.customer.tier,
    )
    data = catalogue.load_catalogue()
    shop = str(data.get("shop") or "our shop")
    await cl.Message(
        content=(
            f"Welcome to {shop}! I can check prices and stock from our "
            "catalogue, keep a basket for you and confirm your order. Try: "
            "'How much is the kettle?' or 'I'll take 2 kettles'."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """One customer turn: DeskSession handles everything; this only relays."""
    config_error = cl.user_session.get(_CONFIG_ERROR_KEY)
    if config_error:
        await cl.Message(content=str(config_error)).send()
        return
    session: DeskSession | None = cl.user_session.get(_SESSION_KEY)
    if session is None:
        await cl.Message(
            content="Please reload the chat window to start a new session."
        ).send()
        return

    # FR-13: this per-turn scope shares the conversation's stable trace id,
    # so the processor MERGES it into the conversation's one trace file
    # (accumulate-and-rewrite — see the module docstring).
    async with tracing_setup.conversation_trace(
        session.session_id, conversation_id=session.session_id
    ):
        reply = await session.handle_user_message(message.content or "")
    await cl.Message(content=reply).send()

    if session.turn_counter % 10 == 0:
        # FR-11: the cost line is an operator surface — logged, not spammed
        # into the customer's chat.
        logger.info(
            "session %s cost line after %d turns: %s",
            session.session_id,
            session.turn_counter,
            session.cost_line(),
        )


@cl.on_chat_end
async def on_chat_end() -> None:
    """Log the conversation's FR-11 cost line at its end."""
    session: DeskSession | None = cl.user_session.get(_SESSION_KEY)
    if session is not None:
        logger.info(
            "session %s ended after %d turns | %s",
            session.session_id,
            session.turn_counter,
            session.cost_line(),
        )
