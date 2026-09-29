"""Per-session conversation state for Shop Desk (FR-12).

:class:`DeskSession` owns ALL conversation state and logic for one customer
session: the message history, the basket, the confirmed order, the cost
ledger and the model-call budget. ``app.py`` is a THIN Chainlit layer over
this class, and ``scripts/demo_conversation.py`` drives the SAME class
directly — the demo exercises the exact code path the UI uses.

STATE (one DeskSession per session; Chainlit sessions are isolated, and two
instances never share a basket, history or ledger):

- ``history``: ``list[dict]`` of ``{"role": "user" | "assistant", "content":
  str}`` — the FR-12 per-session history, trimmed by the rule below;
- ``basket``: ``dict[sku, qty]`` accumulated during THIS session;
- ``draft_order``: the last finalized :class:`orders.Order` (``confirmed``
  when accepted, ``draft`` when problems were reported — never silently
  accepted, FR-5);
- ``ledger``/``budget``: the FR-11 cost ledger and the FR-8/FR-11 model-call
  ceiling, shared by every turn of the session.

THE TRIMMING RULE (FR-12, stated here and in the README):

    Keep at most :data:`MAX_HISTORY_MESSAGES` (12) messages = 6 exchanges.
    When the history grows past that, drop the OLDEST messages first, but
    never drop (a) any message of the CURRENT pending-order discussion —
    every message appended while the basket is non-empty, up to the order
    confirmation (or an explicit basket clear / full removal) that empties
    it — and never drop (b) the last :data:`LAST_EXCHANGES_KEPT` (2)
    exchanges. If the protections alone exceed the cap the history keeps
    growing (protections win; the cap is soft in that case).

    The trimmed history has a REAL consumer: the desk model's per-turn input
    IS the trimmed history (see :meth:`DeskSession._desk_input`), so what
    trimming throws away first is literally what the model stops seeing —
    the oldest tool-free Q&A.

    Justification: plain Q&A is re-derivable from the catalogue cheaply —
    every turn's tools re-fetch catalogue truth anyway — so the oldest
    tool-free Q&A is the safest thing to lose. Order context is NOT
    re-derivable from the catalogue (the basket lives in session state), so
    it is protected while the order is pending. The basket itself is ALSO
    re-stated on the current turn every desk turn (see below), so even a
    dropped order message would not lose the basket — the protection is
    belt-and-braces on top of the recap.

BASKET MEMORY, HISTORY AND FR-2: the desk agent's per-turn INPUT is the
session's TRIMMED HISTORY, handed to the runner as role/content items
(``{"role": "user" | "assistant", "content": str}`` — the SDK's accepted
input shape), with the CURRENT user message last and a short session-side
recap PREPENDED TO THAT MESSAGE'S content (``[Session note] Current basket
(pending): 2x KTL-01 (Electric kettle 1.7L).``). The recap is conversation
INPUT, not the system prompt, and it carries product SKUs/quantities only —
the :class:`ShopContext` itself (customer_id, tier) still travels
exclusively as the run context and is read by tools via the wrapper; no
customer identifier is ever placed in prompt/input text (FR-2). The FAST
PATH deliberately stays single-shot: raw text only, no history and no recap
— a self-contained one-call catalogue lookup (FR-3), which is exactly the
turn kind that needs no dialogue memory.

BASKET RULE (deterministic, zero-cost): the basket is updated in Python from
explicit user statements parsed by :func:`parse_basket_statement` — patterns
``N <name>``, ``N of <name>``, ``<name> x N``, ``N x <name/SKU>`` (and plain
``remove/forget <name>`` / ``clear the basket`` reversals) matched against
the catalogue via the product's full name, its SKU and its long name words.
Quantity words: digits and ``a/an/one..twelve``; repeated statements
ACCUMULATE. Known limits (documented, demo-grade parser): a quantity +
product QUESTION ("how much for 2 kettles?") also updates the basket; a
removal statement drops the product's WHOLE basket line (the quantity in a
removal is ignored); name words like "led" could in principle over-match.
The desk conversation and the Python-rendered confirmation (exact lines +
recomputed total) are the human-visible safety net.

CONFIRMATION FLOW (FR-5, Python truth): when the user's message matches
:data:`CONFIRM_PATTERN` (confirm / yes place / place the order / checkout /
finalize) AND the basket is non-empty, the session calls
``agents_desk.finalize_order`` DIRECTLY — pure Python: catalogue prices,
recomputed total, ``orders.validate_order`` — with ZERO model calls. The
customer-facing summary is rendered BY PYTHON from the validated Order, so
it contains only catalogue-true figures (FR-6's order path). On problems
(e.g. over-stock) the problems are reported politely, the order is stored as
``draft`` and the basket is kept — never silently accepted. The OrderTaker
agent (``agents_desk.get_order_taker_agent``) exists for the FR-5
structured-output demonstration in scripts/tests; the session's confirmation
path does not run it (zero model calls — cheaper AND safer, and the model's
numbers are never trusted anyway).

ROUTING: every other turn goes through ``triage.classify`` — plain
price/stock questions about one product run the FastPath agent (one model
call, FR-3); everything else runs the Desk agent (ordinary loop, FR-10
escalation handoff included). Results are normalized: the customer never
sees anything but a sentence (NFR-4).
"""

from __future__ import annotations

import logging
import re
from typing import Any

import agents_desk
import catalogue
import orders
import runner_cost
import triage
from context import ShopContext
from orders import Order
from runner_cost import ConversationBudget, ConversationLedger, run_desk_turn

logger = logging.getLogger("shopdesk.session")

__all__ = [
    "CONFIRM_PATTERN",
    "LAST_EXCHANGES_KEPT",
    "MAX_HISTORY_MESSAGES",
    "DeskSession",
    "DeskSessionRegistry",
    "parse_basket_statement",
]

# ---------------------------------------------------------------------------
# FR-12 trimming rule (stated in the module docstring and the README)
# ---------------------------------------------------------------------------

MAX_HISTORY_MESSAGES = 12
"""FR-12 cap: keep at most 12 messages (6 exchanges) per session."""

LAST_EXCHANGES_KEPT = 2
"""Never drop the last 2 exchanges (4 messages), whatever the trim needs."""

# ---------------------------------------------------------------------------
# Confirmation detection (FR-5) — basket non-empty AND one of these phrases.
# Limit: a bare substring like "before I confirm..." also matches; accepted
# for demo-grade determinism (the desk conversation clarifies in practice).
# ---------------------------------------------------------------------------

CONFIRM_PATTERN = re.compile(
    r"\b(?:confirm|yes place|place the order|checkout|finalize|finalise)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Deterministic basket parsing (see module docstring for the rule + limits)
# ---------------------------------------------------------------------------

NUMBER_WORDS: dict[str, int] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

# A quantity token: 1-2 digits (not glued to letters or a decimal point via
# the lookbehind, so "1.7L" never yields a quantity — but deliberately with
# NO trailing word boundary so "2x KTL-01" still parses) or a number word
# (longer words first so regex alternation cannot shadow "ten" with "t").
_QTY_ALT = (
    r"(?:(?<![\w.])\d{1,2}|\b(?:"
    + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
    + r")\b)"
)

_REMOVE_VERBS = r"(?:remove|delete|drop|forget)"
_CLEAR_ALL_RE = re.compile(
    r"(?:\b(?:clear|empty|reset)\b[^.?!]{0,30}\bbasket\b|\bstart\s+over\b)",
    re.IGNORECASE,
)

_NAME_WORD_MIN_LEN = 3


def _qty_value(raw: str) -> int:
    """Parse one quantity token ('3', 'two', 'a') into a positive int."""
    text = raw.strip().casefold()
    if text.isdigit():
        return int(text)
    return NUMBER_WORDS.get(text, 0)


def _normalise(value: Any) -> str:
    """Lower-case and collapse whitespace (same tolerance as catalogue.py)."""
    return " ".join(str(value or "").split()).casefold()


def _product_forms(product: dict[str, Any]) -> list[tuple[str, bool]]:
    """Matchable ``(form, allow_plural)`` pairs for one catalogue product.

    Most specific first: the full normalised name, then the SKU, then the
    catalogue name's longer alphabetic words (so "2 kettles" and "2 irons"
    resolve while short/risky tokens like "tv" stay out — documented limit).
    """
    name = str(product.get("name", "") or "")
    forms: list[tuple[str, bool]] = []
    full = _normalise(name)
    if full:
        forms.append((full, False))
    sku = _normalise(product.get("sku", ""))
    if sku:
        forms.append((sku, True))
    words = sorted(
        {
            word.casefold()
            for word in name.split()
            if word.isalpha() and len(word) >= _NAME_WORD_MIN_LEN
        },
        key=len,
        reverse=True,
    )
    known = {form for form, _ in forms}
    forms.extend((word, True) for word in words if word not in known)
    return forms


def parse_basket_statement(
    text: str,
) -> tuple[dict[str, int], list[str], bool]:
    """Deterministically parse ONE user message for basket changes.

    Returns ``(additions, removals, clear_all)``:

    - ``additions``: ``{canonical_sku: qty}`` gathered from the patterns
      ``N <name>``, ``N of <name>``, ``<name> x N`` and ``N x <name/SKU>``
      (quantity words per :data:`NUMBER_WORDS` plus 1-2 digit numbers);
    - ``removals``: SKUs named after a remove/forget/delete/drop verb (the
      whole basket line is removed; the statement's quantity is ignored);
    - ``clear_all``: True for "clear/empty/reset the basket" / "start over".

    A product's first (most specific) matching pattern wins; several
    DIFFERENT products may all match one statement ("2 kettles and 3
    blenders" adds both). Nothing here ever invents a product: only
    catalogue-resolved forms match, and unknown text yields empty results.
    """
    text = str(text or "")
    additions: dict[str, int] = {}
    removals: list[str] = []

    for product in catalogue.load_catalogue().get("products", []):
        if not isinstance(product, dict):
            continue
        sku = str(product.get("sku", "")).strip()
        if not sku or sku in additions or sku in removals:
            continue
        removed = False
        for form, allow_plural in _product_forms(product):
            suffix = r"(?:es|s)?" if allow_plural else ""
            remove_re = re.compile(
                rf"\b{_REMOVE_VERBS}\s+(?:the\s+|my\s+|that\s+|this\s+)?{re.escape(form)}{suffix}\b",
                re.IGNORECASE,
            )
            if remove_re.search(text):
                removals.append(sku)
                removed = True
                break
        if removed:
            continue
        for form, allow_plural in _product_forms(product):
            suffix = r"(?:es|s)?" if allow_plural else ""
            patterns = (
                # "2 kettles" / "3 of the blenders" — quantity first.
                re.compile(
                    rf"(?<![\w.])(?P<qty>{_QTY_ALT})\s+"
                    rf"(?:of\s+(?:the\s+|these\s+|those\s+)?)?{re.escape(form)}{suffix}\b",
                    re.IGNORECASE,
                ),
                # "KTL-01 x 2" / "kettles x 2" — form first.
                re.compile(
                    rf"\b{re.escape(form)}{suffix}\s*[x×]\s*(?P<qty>{_QTY_ALT})\b",
                    re.IGNORECASE,
                ),
                # "2x KTL-01" — quantity, an x/×, then the form.
                re.compile(
                    rf"(?<![\w.])(?P<qty>{_QTY_ALT})\s*[x×]\s*{re.escape(form)}\b",
                    re.IGNORECASE,
                ),
            )
            match = None
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    break
            if match is None:
                continue
            qty = _qty_value(match.group("qty"))
            if qty > 0:
                additions[sku] = qty
                break
    return additions, removals, bool(_CLEAR_ALL_RE.search(text))


# ---------------------------------------------------------------------------
# DeskSession (FR-12)
# ---------------------------------------------------------------------------


class DeskSession:
    """All conversation state and per-turn logic for one customer session.

    One instance per Chainlit session (``app.py`` stores it in
    ``cl.user_session``); demo scripts and tests instantiate it directly.
    Instances share NOTHING mutable — two sessions never share a basket,
    a history or a ledger (FR-12 isolation).
    """

    def __init__(
        self,
        session_id: str,
        customer: ShopContext,
        *,
        ledger: ConversationLedger | None = None,
        budget: ConversationBudget | None = None,
    ) -> None:
        self.session_id = str(session_id)
        self.customer = customer
        self.history: list[dict[str, str]] = []
        self.basket: dict[str, int] = {}
        self.draft_order: Order | None = None
        self.ledger = ledger if ledger is not None else ConversationLedger()
        self.budget = budget if budget is not None else ConversationBudget()
        self.turn_counter = 0
        # Per-session confirmation sequence for order ids (ORD-0001, ...);
        # incremented per finalize attempt so ids are never reused.
        self._confirm_sequence = 0
        # FR-12 bookkeeping: order-context flag per history entry (parallel
        # list, kept in sync by _append_message/_trim_history). True = the
        # message belongs to the current pending-order discussion.
        self._order_flags: list[bool] = []
        # Evidence for the UI/demo: how the last turn was routed and which
        # session-side recap (if any) prefixed the desk input.
        self.last_route: str | None = None
        self.last_recap: str | None = None

    # -- introspection (UI/demo/tests) ------------------------------------

    def cost_line(self) -> str:
        """FR-11 cost line for this conversation (delegates to the ledger)."""
        return self.ledger.cost_line()

    def order_context_flags(self) -> list[bool]:
        """Copies of the per-message order-context flags (trim evidence)."""
        return list(self._order_flags)

    def basket_recap(self) -> str:
        """The short session-side recap prefixed to desk inputs.

        Pending basket first, then the already-confirmed order (so the desk
        remembers a confirmed order too — turn-11 memory). SKUs and
        quantities only; never a customer identifier (FR-2).
        """
        parts: list[str] = []
        if self.basket:
            items = ", ".join(
                f"{qty}x {sku} ({self._product_name(sku)})"
                for sku, qty in sorted(self.basket.items())
            )
            parts.append(f"Current basket (pending): {items}.")
        if self.draft_order is not None and self.draft_order.status == "confirmed":
            items = ", ".join(
                f"{item.qty}x {item.sku} ({self._product_name(item.sku)})"
                for item in self.draft_order.items
            )
            parts.append(
                f"Order already confirmed this session: "
                f"{self.draft_order.order_id} — {items}."
            )
        return " ".join(parts)

    def _desk_input(self) -> list[dict[str, str]]:
        """The desk run input: the TRIMMED history + the recap on the current turn.

        FR-12's "history is carried per session and trimmed past a turn
        count" is realized HERE: the model input is the session's trimmed
        history as SDK role/content items (``{"role": "user"|"assistant",
        "content": str}`` — the shape the runner passes straight through to
        ``Runner.run``, which accepts ``str | list[TResponseInputItem]``),
        with the current user message LAST. The basket recap is prepended to
        THAT message's content as a clearly marked ``[Session note]`` — one
        item, so the model always knows what is conversation memory and what
        is the customer speaking now. The recap is conversation input, not
        the system prompt, and carries SKUs/quantities only (FR-2). The fast
        path does NOT go through this method: it stays single-shot by design.
        """
        items = [dict(message) for message in self.history]  # copies: no aliasing
        recap = self.basket_recap()
        if recap:
            note = f"[Session note] {recap}"
            if items and items[-1].get("role") == "user":
                items[-1] = {
                    "role": "user",
                    "content": f"{note}\n\n{items[-1]['content']}",
                }
            else:  # defensive: the current turn must never go missing
                items.append({"role": "user", "content": note})
        return items

    def _product_name(self, sku: str) -> str:
        product = catalogue.get_product(sku)
        return str(product.get("name", sku)) if product else sku

    # -- history bookkeeping ------------------------------------------------

    def _append_message(self, role: str, content: str, *, order_context: bool) -> None:
        self.history.append({"role": role, "content": content})
        self._order_flags.append(order_context)

    def _trim_history(self) -> None:
        """FR-12 trim: oldest first, order context and last exchanges kept.

        The rule is stated in the module docstring and the README. When the
        protected messages alone exceed MAX_HISTORY_MESSAGES the history
        grows past the cap (protections win) — logged once per trim.
        """
        total = len(self.history)
        if total <= MAX_HISTORY_MESSAGES:
            return
        must_drop = total - MAX_HISTORY_MESSAGES
        protected_tail = min(LAST_EXCHANGES_KEPT * 2, total)
        drop: set[int] = set()
        for index in range(total - protected_tail):  # oldest first
            if len(drop) >= must_drop:
                break
            if not self._order_flags[index]:
                drop.add(index)
        if not drop:
            logger.debug(
                "session %s: %d messages all protected; history exceeds the cap",
                self.session_id,
                total,
            )
            return
        self.history = [m for i, m in enumerate(self.history) if i not in drop]
        self._order_flags = [f for i, f in enumerate(self._order_flags) if i not in drop]
        logger.info(
            "session %s: trimmed %d oldest message(s); history now %d (cap %d)",
            self.session_id,
            len(drop),
            len(self.history),
            MAX_HISTORY_MESSAGES,
        )

    # -- confirmation flow (FR-5, Python truth, zero model calls) ----------

    def _finalize_basket(self) -> str:
        """Finalize the current basket in Python; render the customer reply."""
        self._confirm_sequence += 1
        order_id = f"ORD-{self._confirm_sequence:04d}"
        items = sorted(self.basket.items())
        try:
            order, problems = agents_desk.finalize_order(
                items, order_id, status="confirmed"
            )
        except ValueError as exc:  # build_order refused (bad SKU/qty)
            logger.warning("finalize refused for basket %s: %s", items, exc)
            return (
                "I'm sorry — I couldn't confirm that order: "
                f"{exc} Your basket is unchanged; please adjust it and "
                "confirm again."
            )
        if problems:
            # Never silently accepted (FR-5): the order is stored as DRAFT,
            # the basket is kept and every problem is reported politely.
            self.draft_order = order.model_copy(update={"status": "draft"})
            joined = " ".join(problems)
            return (
                "I'm sorry — I couldn't confirm that order yet. "
                f"{joined} Your basket is unchanged; please adjust it and "
                "confirm again."
            )
        # Accepted: the order is already status="confirmed" from
        # finalize_order (catalogue prices, recomputed total). The basket is
        # consumed, the pending-order protection lifts (the recap carries
        # the confirmed order from here on) and the summary is rendered BY
        # PYTHON from the validated Order — catalogue-true figures only.
        self.basket.clear()
        self._order_flags = [False] * len(self._order_flags)
        self.draft_order = order
        return self._render_confirmation(order)

    def _render_confirmation(self, order: Order) -> str:
        """Python-rendered order summary: line items + recomputed total."""
        currency = self.customer.currency
        lines = [
            "Your order is confirmed — thank you!",
            f"Order {order.order_id} ({order.status}):",
        ]
        for item in order.items:
            line_total = item.qty * item.unit_price
            lines.append(
                f"- {item.qty} x {self._product_name(item.sku)} ({item.sku}) @ "
                f"{catalogue.format_price(item.unit_price, currency)} = "
                f"{catalogue.format_price(line_total, currency)}"
            )
        lines.append(f"Total: {catalogue.format_price(orders.recompute_total(order), currency)}")
        return "\n".join(lines)

    # -- the per-turn flow (FR-12) ------------------------------------------

    async def handle_user_message(self, text: str) -> str:
        """One customer turn: confirm-check -> basket parse -> route -> reply.

        Desk turns send the TRIMMED history (recap on the current turn, see
        :meth:`_desk_input`) to the desk agent; fast-path turns stay
        single-shot; confirmations run pure Python with zero model calls
        (FR-5). Returns the assistant reply (always a customer-ready
        sentence); the reply is also appended to the (trimmed) per-session
        history.
        """
        text = str(text or "").strip()
        self.turn_counter += 1

        # 1. Basket parse FIRST (deterministic Python): the just-parsed items
        #    are visible in this turn's recap, and the message that fills the
        #    basket is flagged as order context.
        additions, removals, clear_all = parse_basket_statement(text)
        if clear_all:
            self.basket.clear()
            logger.info("session %s: basket cleared by customer", self.session_id)
        for sku in removals:
            if self.basket.pop(sku, None) is not None:
                logger.info("session %s: basket line removed: %s", self.session_id, sku)
        for sku, qty in additions.items():
            if sku in removals:  # a removal statement wins for that product
                continue
            self.basket[sku] = self.basket.get(sku, 0) + qty
            logger.info(
                "session %s: basket updated: %s x%d", self.session_id, sku, qty
            )
        if not self.basket:
            # The order discussion is over (confirmed, explicitly cleared, or
            # fully removed): its messages stop being trim-protected, so dead
            # order discussions cannot pin the history past the cap (FR-12).
            self._order_flags = [False] * len(self._order_flags)

        if not text:
            reply = (
                "It looks like your message came through empty — "
                "could you type your question again?"
            )
            self._append_message("user", "", order_context=bool(self.basket))
            self._append_message("assistant", reply, order_context=bool(self.basket))
            self._trim_history()
            self.last_route = "empty-input"
            self.last_recap = None
            return reply

        self._append_message("user", text, order_context=bool(self.basket))

        # 2. Confirmation detection: Python truth, ZERO model calls (FR-5).
        if self.basket and CONFIRM_PATTERN.search(text):
            self.last_route = "confirmation (zero model calls)"
            self.last_recap = None
            reply = self._finalize_basket()
            self._append_message("assistant", reply, order_context=bool(self.basket))
            self._trim_history()
            return reply

        # 3. Route: fast path (one model call) or the Desk's ordinary loop.
        decision = triage.classify(text)
        if decision.is_fast_path:
            # Single-shot by design: raw text only, no history, no recap —
            # the fast path is a self-contained one-call lookup (FR-3).
            self.last_route = f"fast-path ({decision.reason})"
            self.last_recap = None
            model_input = text
            agent = agents_desk.get_fastpath_agent()
            max_turns = agents_desk.FASTPATH_MAX_TURNS
        else:
            # FR-12: trim FIRST so the desk model sees exactly the trimmed
            # history; the just-appended user message sits inside the
            # protected last exchange and always survives the trim.
            self._trim_history()
            recap = self.basket_recap()
            self.last_recap = recap or None
            self.last_route = f"desk ({decision.reason})"
            # The recap is conversation INPUT (not the system prompt) and
            # carries SKUs/quantities only; the ShopContext still travels as
            # the run context (FR-2). The rest of the input is the trimmed
            # history itself (see _desk_input).
            model_input = self._desk_input()
            agent = agents_desk.get_desk_agent()
            max_turns = 8

        result = await run_desk_turn(
            agent,
            model_input,
            self.customer,
            self.ledger,
            self.budget,
            max_turns=max_turns,
        )

        # 5. Normalize: the customer only ever sees a sentence (NFR-4).
        if isinstance(result, Order):
            reply = self._render_confirmation(result)
        else:
            reply = str(result) if result is not None else runner_cost.GENERIC_SORRY
            if not reply.strip():
                reply = runner_cost.GENERIC_SORRY

        self._append_message("assistant", reply, order_context=bool(self.basket))
        self._trim_history()
        return reply


class DeskSessionRegistry:
    """Module-level registry of DeskSessions by id (tests/demo ONLY).

    The Chainlit app does NOT use this — it stores the session in Chainlit's
    own ``cl.user_session`` (keyed by the websocket session id), which is
    what makes two browser windows isolated. The registry exists so tests
    and demo scripts can create/lookup several sessions by id — e.g. to
    demonstrate that two sessions never share a basket (FR-12).
    """

    _sessions: dict[str, DeskSession] = {}

    @classmethod
    def create(cls, session_id: str, customer: ShopContext) -> DeskSession:
        """Create and register a fresh DeskSession (replaces any old one)."""
        session = DeskSession(session_id, customer)
        cls._sessions[str(session_id)] = session
        return session

    @classmethod
    def get(cls, session_id: str) -> DeskSession | None:
        """The registered session for ``session_id``, or None."""
        return cls._sessions.get(str(session_id))

    @classmethod
    def remove(cls, session_id: str) -> None:
        """Drop one session (test/demo cleanup)."""
        cls._sessions.pop(str(session_id), None)

    @classmethod
    def clear(cls) -> None:
        """Drop every session (test/demo cleanup)."""
        cls._sessions.clear()
