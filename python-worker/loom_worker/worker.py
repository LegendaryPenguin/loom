"""The worker process: one socket to the control plane, one engine, one model.

The loop is single-threaded by design. A reader thread does nothing but decode lines
onto a bounded queue; every decision — admit, cancel, preempt, retire, heartbeat — is
made on the step loop's own thread. Cache accounting is the worker's authoritative
report to the control plane (PROTOCOL.md), and the cheapest way to keep an
authoritative number correct is to have exactly one thread allowed to change it.
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable

from . import protocol as proto
from .engine import (
    AdmitResult,
    CompletedEvent,
    FailedEvent,
    FirstTokenEvent,
    GrowRequestEvent,
    InferenceEngine,
    PreemptedEvent,
    SequenceSpec,
    TokenEvent,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    worker_id: str
    heartbeat_interval_ms: int = proto.DEFAULT_HEARTBEAT_INTERVAL_MS

    # Bounded, because PROTOCOL.md makes backpressure explicit: a full dispatch
    # channel is rejected rather than buffered. Buffering here would hide queue depth
    # from the one component that needs to see it to make a scheduling decision.
    inbound_capacity: int = 64

    # How much of the engine's modelled step cost to actually spend. 1.0 makes the
    # simulated engine behave like a device of that speed, which is what the load
    # experiments need; 0.0 runs flat out, which is what tests want.
    realtime_factor: float = 1.0

    # Idle sleep when there is nothing to do, so an empty worker is not a spin loop.
    idle_sleep_s: float = 0.002

    # Throughput smoothing for the heartbeat's tokens_per_second. Decayed rather than
    # cumulative: the autoscaler needs the worker's current speed, and a lifetime
    # average of a worker that has been up for an hour responds to nothing.
    throughput_decay: float = 0.3


class Worker:
    """Drives an engine against a control-plane connection."""

    def __init__(
        self,
        engine: InferenceEngine,
        config: WorkerConfig,
        conn: proto.FramedConnection,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._engine = engine
        self._config = config
        self._conn = conn
        self._clock = clock
        self._sleep = sleep
        self._inbound: queue.Queue[proto.Message] = queue.Queue(maxsize=config.inbound_capacity)
        self._stop = threading.Event()
        self._peer_gone = threading.Event()
        self._draining = False
        self._tokens_per_second = 0.0
        self._reader: threading.Thread | None = None
        self._exit_reason = "not_started"

    @property
    def draining(self) -> bool:
        return self._draining

    @property
    def exit_reason(self) -> str:
        return self._exit_reason

    @property
    def tokens_per_second(self) -> float:
        return self._tokens_per_second

    # Lifecycle.

    def register(self) -> None:
        """Announce real measured capacity.

        total_cache_blocks comes from the engine's own cache, not from a configured
        guess that happens to be passed through: PROTOCOL.md puts the measurement on
        the worker because a control plane that assumes capacity eventually assumes
        wrong and causes an out-of-memory kill mid-generation.
        """
        cache = self._engine.cache
        self._conn.send(
            proto.Register(
                worker_id=self._config.worker_id,
                model_id=self._engine.model_id,
                total_cache_blocks=cache.total(),
                tokens_per_block=cache.tokens_per_block,
                max_batch_size=self._engine.max_batch_size,
                device=self._engine.device,
            )
        )

    def run(self) -> str:
        """Register, then loop until drained, stopped, or the peer disappears."""
        self.register()
        self._reader = threading.Thread(target=self._read_loop, name="loom-worker-reader", daemon=True)
        self._reader.start()
        try:
            self._loop()
        finally:
            self._stop.set()
            self._conn.close()
        return self._exit_reason

    def stop(self) -> None:
        self._stop.set()

    def _read_loop(self) -> None:
        try:
            for msg in self._conn.messages():
                if self._stop.is_set():
                    break
                self._offer(msg)
        except Exception:  # a dead socket is a normal end, not an incident
            logger.debug("worker: reader stopped", exc_info=True)
        finally:
            self._peer_gone.set()

    def _offer(self, msg: proto.Message) -> None:
        try:
            self._inbound.put_nowait(msg)
        except queue.Full:
            # Explicit backpressure, reported as `rejected` rather than `failed`:
            # overload means the fleet is too small while a fault means this worker is
            # sick, and reporting both the same way makes them uncountable.
            if isinstance(msg, proto.Dispatch):
                for item in msg.batch:
                    self._send_rejected(item.request_id)
            else:
                logger.warning("worker: dropped %s, inbound channel full", type(msg).__name__)

    def _loop(self) -> None:
        heartbeat_interval = self._config.heartbeat_interval_ms / 1000.0
        next_heartbeat = self._clock()
        last_step_end = self._clock()

        while not self._stop.is_set():
            self._drain_inbound()

            now = self._clock()
            if now >= next_heartbeat:
                if not self._send_heartbeat():
                    self._exit_reason = "peer_gone"
                    return
                # Fixed cadence from the scheduled time, not from now: drifting the
                # cadence by however long the last step took is how a busy worker
                # misses three heartbeats and gets declared dead while working fine.
                next_heartbeat += heartbeat_interval
                if next_heartbeat <= now:
                    next_heartbeat = now + heartbeat_interval

            if self._draining and not self._engine.has_work():
                self._exit_reason = "drained"
                return

            if self._peer_gone.is_set() and self._inbound.empty():
                self._exit_reason = "peer_gone"
                return

            if not self._engine.has_work():
                self._sleep(min(self._config.idle_sleep_s, max(0.0, next_heartbeat - self._clock())))
                continue

            result = self._engine.step()
            if not self._emit(result.events):
                self._exit_reason = "peer_gone"
                return

            tokens = sum(1 for e in result.events if isinstance(e, TokenEvent))
            step_end = self._clock()
            self._update_throughput(tokens, step_end - last_step_end)
            last_step_end = step_end

            pace_s = (result.duration_ms / 1000.0) * self._config.realtime_factor
            if pace_s > 0:
                self._sleep(pace_s)

        self._exit_reason = "stopped"

    # Inbound handling.

    def _drain_inbound(self) -> None:
        while True:
            try:
                msg = self._inbound.get_nowait()
            except queue.Empty:
                return
            self._handle(msg)

    def _handle(self, msg: proto.Message) -> None:
        if isinstance(msg, proto.Dispatch):
            self._handle_dispatch(msg)
        elif isinstance(msg, proto.Cancel):
            self._handle_cancel(msg)
        elif isinstance(msg, proto.Grant):
            self._handle_grant(msg)
        elif isinstance(msg, proto.Drain):
            # Keep heartbeating, accept nothing new, exit when the running set empties.
            # Draining rather than killing is what stops scale-down from failing work
            # that was nearly done.
            logger.info("worker: draining with %d active", self._engine.active_sequences())
            self._draining = True
        else:
            # Worker->control-plane types arriving inbound are a control plane bug or
            # a loopback test; neither is worth dying over.
            logger.info("worker: ignoring inbound %s", type(msg).__name__)

    def _handle_dispatch(self, msg: proto.Dispatch) -> None:
        for item in msg.batch:
            if self._draining:
                self._send_failed(item.request_id, "worker draining", True)
                continue
            spec = SequenceSpec(
                request_id=item.request_id,
                prompt=item.prompt,
                max_output_tokens=max(1, item.max_output_tokens),
                reserved_blocks=item.reserved_blocks,
            )
            outcome = self._engine.admit(spec)
            if outcome is AdmitResult.ADMITTED:
                continue
            if outcome is AdmitResult.DUPLICATE:
                # At-least-once dispatch: a redelivery of something we hold, or
                # already finished, must be a no-op. Starting a second copy would
                # double the request's cache footprint and stream its tokens twice.
                logger.info("worker: duplicate dispatch for %s ignored", item.request_id)
                continue
            if outcome is AdmitResult.REJECTED_TOO_LARGE:
                # Retryable=false: the prompt does not fit the reservation the control
                # plane computed, and it will not fit it anywhere else either. Saying
                # retryable here turns one bad request into an infinite loop.
                self._send_failed(
                    item.request_id,
                    f"prompt needs more than the reserved {item.reserved_blocks} blocks",
                    False,
                )
                continue
            # Batch full and cache exhausted are both "this worker has no room",
            # which is overload rather than a fault, so they take the same path as a
            # full inbound channel.
            logger.info(
                "worker: rejecting %s, %s",
                item.request_id,
                "batch full" if outcome is AdmitResult.REJECTED_BATCH_FULL else "cache exhausted",
            )
            self._send_rejected(item.request_id)

    def _handle_cancel(self, msg: proto.Cancel) -> None:
        events = self._engine.cancel(msg.request_id)
        if not events:
            # Nothing held: either it already completed (and we have reported it) or we
            # never had it. Reporting a second completion would double-count in the
            # control plane's metrics, which is worse than the silence.
            logger.info("worker: cancel for unknown request %s", msg.request_id)
            return
        self._emit(events)

    def _handle_grant(self, msg: proto.Grant) -> None:
        if self._engine.grant(msg.request_id, msg.extra_blocks):
            logger.info("worker: granted %s %d extra blocks", msg.request_id, msg.extra_blocks)
            return
        # A grant for a sequence that has already gone, or an empty one. Not an error:
        # the grant and the sequence's end race, and the loser is simply discarded.
        logger.info(
            "worker: grant of %d blocks for %s had no effect",
            msg.extra_blocks,
            msg.request_id,
        )

    # Outbound.

    def _emit(self, events: list[object]) -> bool:
        """Translate engine events to wire messages. False means the peer is gone."""
        for event in events:
            try:
                if isinstance(event, FirstTokenEvent):
                    self._conn.send(proto.FirstToken(event.request_id, event.latency_ms))
                elif isinstance(event, TokenEvent):
                    self._conn.send(
                        proto.Token(event.request_id, event.token, event.generated, event.blocks)
                    )
                elif isinstance(event, CompletedEvent):
                    self._conn.send(
                        proto.Completed(event.request_id, event.generated, event.stop_reason)
                    )
                elif isinstance(event, FailedEvent):
                    self._conn.send(
                        proto.Failed(event.request_id, event.error, event.retryable)
                    )
                elif isinstance(event, PreemptedEvent):
                    logger.info(
                        "worker: preempted %s after %d tokens (%s)",
                        event.request_id,
                        event.generated,
                        event.reason,
                    )
                    self._conn.send(
                        proto.Preempted(event.request_id, event.generated, event.reason)
                    )
                elif isinstance(event, GrowRequestEvent):
                    logger.info(
                        "worker: asking for %d more blocks for %s (%d free)",
                        event.want_blocks,
                        event.request_id,
                        event.free_blocks,
                    )
                    self._conn.send(
                        proto.GrowRequest(
                            event.request_id, event.want_blocks, event.free_blocks
                        )
                    )
                else:
                    logger.warning("worker: unhandled engine event %r", event)
            except (ConnectionError, OSError):
                self._peer_gone.set()
                return False
        return True

    def _send_failed(self, request_id: str, error: str, retryable: bool) -> None:
        try:
            self._conn.send(proto.Failed(request_id, error, retryable))
        except (ConnectionError, OSError):
            self._peer_gone.set()

    def _send_rejected(self, request_id: str) -> None:
        try:
            self._conn.send(proto.Rejected(request_id, proto.REJECT_BACKPRESSURE))
        except (ConnectionError, OSError):
            self._peer_gone.set()

    def _send_heartbeat(self) -> bool:
        cache = self._engine.cache
        try:
            self._conn.send(
                proto.Heartbeat(
                    worker_id=self._config.worker_id,
                    used_cache_blocks=cache.used(),
                    active_sequences=self._engine.active_sequences(),
                    tokens_per_second=round(self._tokens_per_second, 3),
                    queue_depth=self._inbound.qsize(),
                )
            )
            return True
        except (ConnectionError, OSError):
            self._peer_gone.set()
            return False

    def _update_throughput(self, tokens: int, elapsed_s: float) -> None:
        if elapsed_s <= 0:
            return
        instant = tokens / elapsed_s
        a = self._config.throughput_decay
        self._tokens_per_second = (1.0 - a) * self._tokens_per_second + a * instant


def connect(addr: str, timeout_s: float = 10.0) -> proto.FramedConnection:
    """Open a connection to `host:port`."""
    host, _, port = addr.rpartition(":")
    if not host or not port:
        raise ValueError(f"expected host:port, got {addr!r}")
    sock = socket.create_connection((host, int(port)), timeout=timeout_s)
    # Blocking reads with no timeout once connected: liveness is the heartbeat's job,
    # and a read timeout here would only produce spurious disconnects on an idle link.
    sock.settimeout(None)
    # Per-decode-step messages are small and latency-sensitive; coalescing them costs
    # inter-token latency the control plane would read as the worker being slow.
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return proto.FramedConnection(sock)
