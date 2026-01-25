"""The wire protocol between the Go control plane and a Python worker.

This module is a direct transcription of PROTOCOL.md and is deliberately the only
place in the worker that knows field names. The protocol is frozen and nothing is
generated from a schema, so the defence against drift is a schema declared here and
enforced on every encode: a mistyped field name must fail in a unit test rather
than produce a line the Go side silently drops.

Decoding is asymmetric to encoding on purpose. Outbound messages are validated
strictly because we author them; inbound messages tolerate unknown fields and skip
unknown types, because PROTOCOL.md requires a newer control plane to be able to
talk to an older worker without taking it down.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

# Message type names. Spelled once so a caller cannot invent a near-miss.
TYPE_REGISTER = "register"
TYPE_HEARTBEAT = "heartbeat"
TYPE_FIRST_TOKEN = "first_token"
TYPE_TOKEN = "token"
TYPE_COMPLETED = "completed"
TYPE_FAILED = "failed"
TYPE_PREEMPTED = "preempted"
TYPE_GROW_REQUEST = "grow_request"
TYPE_REJECTED = "rejected"
TYPE_DISPATCH = "dispatch"
TYPE_CANCEL = "cancel"
TYPE_GRANT = "grant"
TYPE_DRAIN = "drain"

# stop_reason is a closed set in PROTOCOL.md. The control plane's metrics bucket on
# it, so an unlisted value is a silent metrics hole rather than a loud error.
STOP_EOS = "eos"
STOP_LENGTH = "length"
STOP_CANCELLED = "cancelled"
STOP_REASONS = frozenset({STOP_EOS, STOP_LENGTH, STOP_CANCELLED})

# The two preemption causes call for opposite operator responses — add workers versus
# fix the estimator — which is the entire reason the field exists, so an unlisted
# value here is worse than useless and is refused on encode.
PREEMPT_CACHE_EXHAUSTED = "cache_exhausted"
PREEMPT_RESERVATION_EXCEEDED = "reservation_exceeded"
PREEMPT_REASONS = frozenset({PREEMPT_CACHE_EXHAUSTED, PREEMPT_RESERVATION_EXCEEDED})

# The only reason PROTOCOL.md defines for `rejected`. Keeping the set closed means a
# second overload cause has to be added to the protocol before it can be reported,
# rather than appearing as an unbucketed string in the control plane's metrics.
REJECT_BACKPRESSURE = "backpressure"
REJECT_REASONS = frozenset({REJECT_BACKPRESSURE})

DEFAULT_HEARTBEAT_INTERVAL_MS = 500


class ProtocolError(Exception):
    """An outbound message violates PROTOCOL.md."""


# Per-type field schema for OUTBOUND validation: field name -> accepted types.
# Every field is required; the protocol has no optional fields, and an omitted
# field reads as a zero on the Go side rather than as missing, which is worse than
# an error.
_OUTBOUND_SCHEMA: dict[str, dict[str, tuple[type, ...]]] = {
    TYPE_REGISTER: {
        "worker_id": (str,),
        "model_id": (str,),
        "total_cache_blocks": (int,),
        "tokens_per_block": (int,),
        "max_batch_size": (int,),
        "device": (str,),
    },
    TYPE_HEARTBEAT: {
        "worker_id": (str,),
        "used_cache_blocks": (int,),
        "active_sequences": (int,),
        "tokens_per_second": (int, float),
        "queue_depth": (int,),
    },
    TYPE_FIRST_TOKEN: {
        "request_id": (str,),
        "latency_ms": (int, float),
    },
    TYPE_TOKEN: {
        "request_id": (str,),
        "token": (str,),
        "generated": (int,),
        "blocks": (int,),
    },
    TYPE_COMPLETED: {
        "request_id": (str,),
        "generated": (int,),
        "stop_reason": (str,),
    },
    TYPE_FAILED: {
        "request_id": (str,),
        "error": (str,),
        "retryable": (bool,),
    },
    TYPE_PREEMPTED: {
        "request_id": (str,),
        "generated": (int,),
        "reason": (str,),
    },
    TYPE_GROW_REQUEST: {
        "request_id": (str,),
        "want_blocks": (int,),
        # Sent alongside the want so the control plane can see that the worker really
        # does have room, and decide without a round trip to ask.
        "free_blocks": (int,),
    },
    TYPE_REJECTED: {
        "request_id": (str,),
        "reason": (str,),
    },
}


def encode_payload(msg_type: str, payload: Mapping[str, Any]) -> str:
    """Validate a worker->control-plane message and return its wire line.

    The returned string includes the terminating newline, because the framing is
    part of the message and splitting them invites a caller that forgets.
    """
    schema = _OUTBOUND_SCHEMA.get(msg_type)
    if schema is None:
        raise ProtocolError(f"not a worker->control-plane message type: {msg_type!r}")

    unknown = sorted(set(payload) - set(schema))
    if unknown:
        raise ProtocolError(f"{msg_type}: unknown field(s) {unknown}; allowed {sorted(schema)}")
    missing = sorted(set(schema) - set(payload))
    if missing:
        raise ProtocolError(f"{msg_type}: missing field(s) {missing}")

    for name, accepted in schema.items():
        value = payload[name]
        # bool is a subclass of int in Python; an accidental True in a count field
        # encodes as `true` and the Go side fails to unmarshal it into an int.
        if isinstance(value, bool) and bool not in accepted:
            raise ProtocolError(f"{msg_type}.{name}: bool is not valid here")
        if not isinstance(value, accepted):
            names = "|".join(t.__name__ for t in accepted)
            raise ProtocolError(
                f"{msg_type}.{name}: expected {names}, got {type(value).__name__}"
            )

    if msg_type == TYPE_COMPLETED and payload["stop_reason"] not in STOP_REASONS:
        raise ProtocolError(
            f"completed.stop_reason: {payload['stop_reason']!r} not in {sorted(STOP_REASONS)}"
        )
    if msg_type == TYPE_PREEMPTED and payload["reason"] not in PREEMPT_REASONS:
        raise ProtocolError(
            f"preempted.reason: {payload['reason']!r} not in {sorted(PREEMPT_REASONS)}"
        )
    if msg_type == TYPE_REJECTED and payload["reason"] not in REJECT_REASONS:
        raise ProtocolError(
            f"rejected.reason: {payload['reason']!r} not in {sorted(REJECT_REASONS)}"
        )

    body = {"type": msg_type}
    body.update(payload)
    return json.dumps(body, separators=(",", ":"), sort_keys=True) + "\n"


@dataclass(frozen=True)
class Register:
    worker_id: str
    model_id: str
    total_cache_blocks: int
    tokens_per_block: int
    max_batch_size: int
    device: str

    def encode(self) -> str:
        return encode_payload(
            TYPE_REGISTER,
            {
                "worker_id": self.worker_id,
                "model_id": self.model_id,
                "total_cache_blocks": self.total_cache_blocks,
                "tokens_per_block": self.tokens_per_block,
                "max_batch_size": self.max_batch_size,
                "device": self.device,
            },
        )


@dataclass(frozen=True)
class Heartbeat:
    worker_id: str
    used_cache_blocks: int
    active_sequences: int
    tokens_per_second: float
    queue_depth: int

    def encode(self) -> str:
        return encode_payload(
            TYPE_HEARTBEAT,
            {
                "worker_id": self.worker_id,
                "used_cache_blocks": self.used_cache_blocks,
                "active_sequences": self.active_sequences,
                "tokens_per_second": self.tokens_per_second,
                "queue_depth": self.queue_depth,
            },
        )


@dataclass(frozen=True)
class FirstToken:
    request_id: str
    latency_ms: float

    def encode(self) -> str:
        return encode_payload(
            TYPE_FIRST_TOKEN,
            {"request_id": self.request_id, "latency_ms": self.latency_ms},
        )


@dataclass(frozen=True)
class Token:
    request_id: str
    token: str
    generated: int
    blocks: int

    def encode(self) -> str:
        return encode_payload(
            TYPE_TOKEN,
            {
                "request_id": self.request_id,
                "token": self.token,
                "generated": self.generated,
                "blocks": self.blocks,
            },
        )


@dataclass(frozen=True)
class Completed:
    request_id: str
    generated: int
    stop_reason: str

    def encode(self) -> str:
        return encode_payload(
            TYPE_COMPLETED,
            {
                "request_id": self.request_id,
                "generated": self.generated,
                "stop_reason": self.stop_reason,
            },
        )


@dataclass(frozen=True)
class Failed:
    request_id: str
    error: str
    retryable: bool

    def encode(self) -> str:
        return encode_payload(
            TYPE_FAILED,
            {
                "request_id": self.request_id,
                "error": self.error,
                "retryable": self.retryable,
            },
        )


@dataclass(frozen=True)
class Preempted:
    request_id: str
    generated: int
    reason: str

    def encode(self) -> str:
        return encode_payload(
            TYPE_PREEMPTED,
            {
                "request_id": self.request_id,
                "generated": self.generated,
                "reason": self.reason,
            },
        )


@dataclass(frozen=True)
class GrowRequest:
    request_id: str
    want_blocks: int
    free_blocks: int

    def encode(self) -> str:
        return encode_payload(
            TYPE_GROW_REQUEST,
            {
                "request_id": self.request_id,
                "want_blocks": self.want_blocks,
                "free_blocks": self.free_blocks,
            },
        )


@dataclass(frozen=True)
class Rejected:
    request_id: str
    reason: str = REJECT_BACKPRESSURE

    def encode(self) -> str:
        return encode_payload(
            TYPE_REJECTED, {"request_id": self.request_id, "reason": self.reason}
        )


@dataclass(frozen=True)
class DispatchItem:
    request_id: str
    prompt: str
    max_output_tokens: int
    reserved_blocks: int


@dataclass(frozen=True)
class Dispatch:
    batch: tuple[DispatchItem, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Cancel:
    request_id: str


@dataclass(frozen=True)
class Grant:
    request_id: str

    # Additional blocks ON TOP OF the original reservation, not a new total: the
    # control plane answers the `want_blocks` it was asked for, and reading this as a
    # replacement cap would shrink a sequence that had already grown legitimately.
    extra_blocks: int


@dataclass(frozen=True)
class Drain:
    pass


# Inbound messages are what a worker must parse. Worker->control-plane types are
# also decodable so a test harness standing in for the control plane can use this
# module rather than reimplementing half of it.
InboundMessage = Dispatch | Cancel | Grant | Drain
OutboundMessage = (
    Register | Heartbeat | FirstToken | Token | Completed | Failed | Preempted | GrowRequest | Rejected
)
Message = (
    Register
    | Heartbeat
    | FirstToken
    | Token
    | Completed
    | Failed
    | Preempted
    | GrowRequest
    | Rejected
    | Dispatch
    | Cancel
    | Grant
    | Drain
)


def _req_str(obj: Mapping[str, Any], key: str) -> str:
    value = obj.get(key, "")
    return value if isinstance(value, str) else str(value)


def _req_int(obj: Mapping[str, Any], key: str) -> int:
    value = obj.get(key, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _req_float(obj: Mapping[str, Any], key: str) -> float:
    value = obj.get(key, 0.0)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _decode_object(obj: Mapping[str, Any]) -> Message | None:
    """Build a message from an already-parsed JSON object, or None to skip it."""
    msg_type = obj.get("type")
    if not isinstance(msg_type, str):
        logger.warning("protocol: message with no string 'type' field, skipping")
        return None

    if msg_type == TYPE_DISPATCH:
        raw_batch = obj.get("batch")
        items: list[DispatchItem] = []
        if isinstance(raw_batch, Sequence) and not isinstance(raw_batch, (str, bytes)):
            for entry in raw_batch:
                if not isinstance(entry, Mapping):
                    logger.warning("protocol: dispatch batch entry is not an object, skipping")
                    continue
                items.append(
                    DispatchItem(
                        request_id=_req_str(entry, "request_id"),
                        prompt=_req_str(entry, "prompt"),
                        max_output_tokens=_req_int(entry, "max_output_tokens"),
                        reserved_blocks=_req_int(entry, "reserved_blocks"),
                    )
                )
        return Dispatch(batch=tuple(items))
    if msg_type == TYPE_CANCEL:
        return Cancel(request_id=_req_str(obj, "request_id"))
    if msg_type == TYPE_GRANT:
        return Grant(
            request_id=_req_str(obj, "request_id"),
            extra_blocks=_req_int(obj, "extra_blocks"),
        )
    if msg_type == TYPE_DRAIN:
        return Drain()
    if msg_type == TYPE_REGISTER:
        return Register(
            worker_id=_req_str(obj, "worker_id"),
            model_id=_req_str(obj, "model_id"),
            total_cache_blocks=_req_int(obj, "total_cache_blocks"),
            tokens_per_block=_req_int(obj, "tokens_per_block"),
            max_batch_size=_req_int(obj, "max_batch_size"),
            device=_req_str(obj, "device"),
        )
    if msg_type == TYPE_HEARTBEAT:
        return Heartbeat(
            worker_id=_req_str(obj, "worker_id"),
            used_cache_blocks=_req_int(obj, "used_cache_blocks"),
            active_sequences=_req_int(obj, "active_sequences"),
            tokens_per_second=_req_float(obj, "tokens_per_second"),
            queue_depth=_req_int(obj, "queue_depth"),
        )
    if msg_type == TYPE_FIRST_TOKEN:
        return FirstToken(
            request_id=_req_str(obj, "request_id"),
            latency_ms=_req_float(obj, "latency_ms"),
        )
    if msg_type == TYPE_TOKEN:
        return Token(
            request_id=_req_str(obj, "request_id"),
            token=_req_str(obj, "token"),
            generated=_req_int(obj, "generated"),
            blocks=_req_int(obj, "blocks"),
        )
    if msg_type == TYPE_COMPLETED:
        return Completed(
            request_id=_req_str(obj, "request_id"),
            generated=_req_int(obj, "generated"),
            stop_reason=_req_str(obj, "stop_reason"),
        )
    if msg_type == TYPE_FAILED:
        retryable = obj.get("retryable", False)
        return Failed(
            request_id=_req_str(obj, "request_id"),
            error=_req_str(obj, "error"),
            retryable=bool(retryable),
        )
    if msg_type == TYPE_PREEMPTED:
        return Preempted(
            request_id=_req_str(obj, "request_id"),
            generated=_req_int(obj, "generated"),
            reason=_req_str(obj, "reason"),
        )
    if msg_type == TYPE_GROW_REQUEST:
        return GrowRequest(
            request_id=_req_str(obj, "request_id"),
            want_blocks=_req_int(obj, "want_blocks"),
            free_blocks=_req_int(obj, "free_blocks"),
        )
    if msg_type == TYPE_REJECTED:
        return Rejected(
            request_id=_req_str(obj, "request_id"),
            reason=_req_str(obj, "reason"),
        )

    # Required by PROTOCOL.md: log and carry on. A newer control plane adding a
    # message type must not be able to kill an older worker.
    logger.info("protocol: unknown message type %r, skipping", msg_type)
    return None


def decode_line(line: str) -> Message | None:
    """Decode one wire line. Returns None for anything that must be skipped.

    Malformed JSON is a skip rather than a raise for the same reason an unknown
    type is: the connection carries every worker's work, and tearing it down over
    one unreadable line loses in-flight sequences that were otherwise fine.
    """
    text = line.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("protocol: malformed JSON line skipped: %s", exc)
        return None
    if not isinstance(obj, Mapping):
        logger.warning("protocol: top-level JSON value is not an object, skipping")
        return None
    return _decode_object(obj)


class FramedConnection:
    """Newline-delimited JSON over a socket, with its own read buffer.

    The buffer is not optional: a TCP read can return half a line or three lines,
    and a reader that assumes one read is one message works in tests and corrupts
    the stream under load.
    """

    def __init__(self, sock: socket.socket, read_chunk: int = 65536) -> None:
        self._sock = sock
        self._read_chunk = read_chunk
        self._buf = bytearray()
        self._closed = False
        # Sends happen from the engine loop and potentially from a heartbeat timer,
        # and a half-written line is unrecoverable for the peer.
        self._send_lock = threading.Lock()

    def send_line(self, line: str) -> None:
        data = line.encode("utf-8")
        with self._send_lock:
            if self._closed:
                raise ConnectionError("connection closed")
            self._sock.sendall(data)

    def send(self, msg: OutboundMessage) -> None:
        self.send_line(msg.encode())

    def messages(self) -> Iterator[Message]:
        """Yield decoded messages until the peer closes.

        Skipped lines simply do not yield; the caller never sees them, which is
        what makes "unknown types are not fatal" true at the call site too.
        """
        while True:
            line = self.read_line()
            if line is None:
                return
            msg = decode_line(line)
            if msg is not None:
                yield msg

    def read_line(self) -> str | None:
        """Return the next complete line without its newline, or None at EOF."""
        while True:
            idx = self._buf.find(b"\n")
            if idx >= 0:
                raw = bytes(self._buf[:idx])
                del self._buf[: idx + 1]
                return raw.decode("utf-8", errors="replace")
            try:
                chunk = self._sock.recv(self._read_chunk)
            except (OSError, ValueError):
                return None
            if not chunk:
                # A trailing fragment with no newline is an incomplete message and
                # is dropped rather than guessed at.
                if self._buf:
                    logger.warning("protocol: discarding %d trailing bytes at EOF", len(self._buf))
                    self._buf.clear()
                return None
            self._buf.extend(chunk)

    def close(self) -> None:
        with self._send_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
