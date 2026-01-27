"""`python -m loom_worker` — run one worker against a control plane."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import time

from .engine import build_engine
from .worker import Worker, WorkerConfig, connect


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m loom_worker",
        description="A Loom model worker: one model, one control-plane connection.",
    )
    p.add_argument("--addr", required=True, help="control plane address as host:port")
    p.add_argument("--model", required=True, dest="model", help="model id this worker serves")
    p.add_argument(
        "--simulated",
        action="store_true",
        help="use the deterministic simulated engine instead of a real model",
    )
    p.add_argument(
        "--blocks",
        type=int,
        default=None,
        help="KV-cache blocks to report; omit to measure from the loaded model",
    )
    p.add_argument("--tokens-per-block", type=int, default=16, help="cache block granularity")
    p.add_argument("--max-batch-size", type=int, default=32, help="sequences per forward pass")
    p.add_argument("--seed", type=int, default=0, help="seed for the simulated engine")
    p.add_argument(
        "--worker-id",
        default=None,
        help="identity reported at registration; defaults to host-pid",
    )
    p.add_argument(
        "--heartbeat-ms", type=int, default=500, help="heartbeat interval in milliseconds"
    )
    p.add_argument(
        "--grant-timeout-ms",
        type=float,
        default=1000.0,
        help=(
            "how long a sequence waits for a grant of blocks beyond its reservation "
            "before being preempted as reservation_exceeded"
        ),
    )
    p.add_argument(
        "--realtime-factor",
        type=float,
        default=1.0,
        help="fraction of the engine's modelled step cost to actually spend; 0 runs flat out",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("loom_worker")

    engine, fallback_reason = build_engine(
        model_id=args.model,
        simulated=args.simulated,
        total_blocks=args.blocks,
        tokens_per_block=args.tokens_per_block,
        max_batch_size=args.max_batch_size,
        seed=args.seed,
        grant_timeout_ms=args.grant_timeout_ms,
    )
    if fallback_reason is not None:
        # Loud, because which engine produced a measurement is part of the measurement.
        log.warning("real engine unavailable, running simulated: %s", fallback_reason)

    worker_id = args.worker_id or f"{socket.gethostname()}-{os.getpid()}"
    log.info(
        "worker %s serving %s on %s with %d blocks of %d tokens",
        worker_id,
        engine.model_id,
        engine.device,
        engine.cache.total(),
        engine.cache.tokens_per_block,
    )

    try:
        conn = connect(args.addr)
    except OSError as exc:
        log.error("cannot connect to %s: %s", args.addr, exc)
        return 1

    worker = Worker(
        engine,
        WorkerConfig(
            worker_id=worker_id,
            heartbeat_interval_ms=args.heartbeat_ms,
            realtime_factor=args.realtime_factor,
        ),
        conn,
    )
    started = time.monotonic()
    try:
        reason = worker.run()
    except KeyboardInterrupt:
        worker.stop()
        reason = "interrupted"
    log.info("worker %s exiting after %.1fs: %s", worker_id, time.monotonic() - started, reason)
    # A drained worker exited on instruction and succeeded; losing the peer did not.
    return 0 if reason in ("drained", "stopped", "interrupted") else 1


if __name__ == "__main__":
    sys.exit(main())
