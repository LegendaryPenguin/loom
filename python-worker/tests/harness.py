"""Shared test support: import path fix plus a fake control plane.

The fake control plane speaks the protocol over a real socket rather than being a
stub handed to the worker. A stub would let the worker's framing, its registration
order and its backpressure behaviour all be wrong without any test noticing, and
those are exactly the parts that only fail against a real byte stream.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping

# Importing loom_worker from a tests directory that unittest discovery has made the
# top-level dir requires the package root on sys.path.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from loom_worker import protocol as proto  # noqa: E402


def encode_inbound(msg_type: str, **fields: Any) -> str:
    """Encode a control-plane->worker message.

    These live in the test harness rather than in loom_worker.protocol because the
    worker never sends them, and an encoder the production code does not use is an
    encoder nothing keeps honest.
    """
    body: dict[str, Any] = {"type": msg_type}
    body.update(fields)
    return json.dumps(body) + "\n"


def dispatch_line(items: Iterable[Mapping[str, Any]]) -> str:
    return encode_inbound(proto.TYPE_DISPATCH, batch=list(items))


def dispatch_item(
    request_id: str,
    prompt: str = "the quick brown fox jumps over the lazy dog",
    max_output_tokens: int = 8,
    reserved_blocks: int = 64,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "prompt": prompt,
        "max_output_tokens": max_output_tokens,
        "reserved_blocks": reserved_blocks,
    }


def cancel_line(request_id: str) -> str:
    return encode_inbound(proto.TYPE_CANCEL, request_id=request_id)


def grant_line(request_id: str, extra_blocks: int) -> str:
    return encode_inbound(
        proto.TYPE_GRANT, request_id=request_id, extra_blocks=extra_blocks
    )


def drain_line() -> str:
    return encode_inbound(proto.TYPE_DRAIN)


def start_worker(engine: Any, control_plane: "FakeControlPlane", **config: Any) -> tuple[Any, threading.Thread]:
    """Run a worker against the fake control plane on its own thread."""
    from loom_worker.worker import Worker, WorkerConfig, connect

    defaults: dict[str, Any] = {
        "worker_id": "w-test",
        "heartbeat_interval_ms": 50,
        # A fraction of the engine's modelled cost: fast enough for a test, slow
        # enough that a sequence is still running when the test cancels it.
        "realtime_factor": 0.25,
    }
    defaults.update(config)
    worker = Worker(engine, WorkerConfig(**defaults), connect(control_plane.addr))
    thread = threading.Thread(target=worker.run, name="loom-worker-under-test", daemon=True)
    thread.start()
    control_plane.wait_connected()
    return worker, thread


class FakeControlPlane:
    """Accepts one worker connection, records every message, and can talk back."""

    def __init__(self, timeout_s: float = 10.0, auto_grant: bool = False) -> None:
        # auto_grant answers every grow_request with exactly what was asked for, which
        # is the "the estimate was low but there is room" case the protocol added
        # grow_request/grant for. Off by default so a test can also exercise silence.
        self._auto_grant = auto_grant
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self._timeout_s = timeout_s
        self._messages: list[proto.Message] = []
        self._lock = threading.Condition()
        self._conn: proto.FramedConnection | None = None
        self._accepted = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._serve, name="fake-control-plane", daemon=True)
        self._thread.start()

    @property
    def addr(self) -> str:
        host, port = self._listener.getsockname()
        return f"{host}:{port}"

    def _serve(self) -> None:
        try:
            self._listener.settimeout(self._timeout_s)
            sock, _ = self._listener.accept()
        except OSError:
            return
        conn = proto.FramedConnection(sock)
        self._conn = conn
        self._accepted.set()
        try:
            for msg in conn.messages():
                with self._lock:
                    self._messages.append(msg)
                    self._lock.notify_all()
                # Replying outside the lock: send_line can block, and holding the lock
                # across it would stall every wait_for in the test.
                if self._auto_grant and isinstance(msg, proto.GrowRequest):
                    conn.send_line(grant_line(msg.request_id, msg.want_blocks))
        except Exception:
            pass
        finally:
            with self._lock:
                self._closed = True
                self._lock.notify_all()

    def wait_connected(self, timeout_s: float = 5.0) -> None:
        if not self._accepted.wait(timeout_s):
            raise AssertionError("worker never connected")

    def send_line(self, line: str) -> None:
        self.wait_connected()
        assert self._conn is not None
        self._conn.send_line(line)

    def messages(self) -> list[proto.Message]:
        with self._lock:
            return list(self._messages)

    def of_type(self, cls: type) -> list[Any]:
        return [m for m in self.messages() if isinstance(m, cls)]

    def wait_for(
        self, predicate: Callable[[list[proto.Message]], bool], timeout_s: float = 5.0
    ) -> list[proto.Message]:
        deadline = time.monotonic() + timeout_s
        with self._lock:
            while not predicate(list(self._messages)):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"timed out waiting; saw {[type(m).__name__ for m in self._messages]}"
                    )
                self._lock.wait(remaining)
            return list(self._messages)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
        try:
            self._listener.close()
        except OSError:
            pass
