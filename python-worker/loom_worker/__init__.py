"""Loom model workers: the Python half of the inference serving platform.

The control plane decides what runs; this package runs it. The split is along the
line where the problem changes — resource accounting and concurrency on one side, a
model runtime on the other — and the only thing shared across it is PROTOCOL.md.
"""

from __future__ import annotations

from .engine import (
    AdmitResult,
    EngineUnavailable,
    GrowRequestEvent,
    InferenceEngine,
    SequenceSpec,
    SimulatedEngine,
    TransformersEngine,
    build_engine,
    transformers_unavailable_reason,
)
from .kvcache import CachePolicy, PagedKVCache
from .worker import Worker, WorkerConfig, connect

__all__ = [
    "AdmitResult",
    "CachePolicy",
    "EngineUnavailable",
    "GrowRequestEvent",
    "InferenceEngine",
    "PagedKVCache",
    "SequenceSpec",
    "SimulatedEngine",
    "TransformersEngine",
    "Worker",
    "WorkerConfig",
    "build_engine",
    "connect",
    "transformers_unavailable_reason",
]
