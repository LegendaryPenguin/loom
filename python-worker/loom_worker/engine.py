"""Inference engines: the thing that actually advances sequences.

Two implementations share one base class, and the base class is where continuous
batching lives. That placement is the point of the file: the batching rules — admit
mid-flight, retire on completion, grow the cache per step, preempt when it cannot
grow — are properties of the serving loop, not of the model runtime, so both the
simulated engine used for load experiments and the real transformers engine get
exactly the same semantics. A subclass only has to know how to produce one token.

`step()` advances a SET of sequences by one position. A step function that advances
one sequence cannot express continuous batching at all: the only loop you can build
on top of it runs sequences to completion in turn, which is static batching with
extra steps.
"""

from __future__ import annotations

import logging
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .kvcache import CachePolicy, PagedKVCache
from .protocol import (
    PREEMPT_CACHE_EXHAUSTED,
    PREEMPT_RESERVATION_EXCEEDED,
    STOP_CANCELLED,
    STOP_EOS,
    STOP_LENGTH,
)

logger = logging.getLogger(__name__)


class EngineUnavailable(RuntimeError):
    """A requested engine cannot run in this process (missing runtime, bad model)."""


@dataclass(frozen=True)
class SequenceSpec:
    """One dispatched request, as the engine needs it."""

    request_id: str
    prompt: str
    max_output_tokens: int

    # The control plane's allocation. PROTOCOL.md: the worker must not exceed it
    # without asking — and `grow_request` is how it asks. The cap starts here and is
    # raised by a `grant`; see _ensure_blocks and BaseEngine.grant.
    reserved_blocks: int


class AdmitResult(Enum):
    ADMITTED = "admitted"
    DUPLICATE = "duplicate"  # at-least-once redelivery of something we hold or finished
    REJECTED_CAPACITY = "rejected_capacity"
    REJECTED_BATCH_FULL = "rejected_batch_full"
    REJECTED_TOO_LARGE = "rejected_too_large"  # prompt will not fit the reservation


# Engine-level events. Intentionally not the protocol dataclasses: the engine knows
# nothing about the wire, and keeping the translation in the worker is what lets the
# engine be driven directly by a test with no socket in sight.
@dataclass(frozen=True)
class FirstTokenEvent:
    request_id: str
    latency_ms: float


@dataclass(frozen=True)
class TokenEvent:
    request_id: str
    token: str
    generated: int
    blocks: int


@dataclass(frozen=True)
class CompletedEvent:
    request_id: str
    generated: int
    stop_reason: str


@dataclass(frozen=True)
class FailedEvent:
    request_id: str
    error: str
    retryable: bool


@dataclass(frozen=True)
class PreemptedEvent:
    request_id: str
    generated: int

    # One of protocol.PREEMPT_REASONS. Set from what actually blocked the sequence,
    # not from which branch of the code noticed: the control plane routes on it.
    reason: str


@dataclass(frozen=True)
class GrowRequestEvent:
    """The sequence needs more than its cap allows and the worker has the room."""

    request_id: str
    want_blocks: int
    free_blocks: int


EngineEvent = (
    FirstTokenEvent | TokenEvent | CompletedEvent | FailedEvent | PreemptedEvent | GrowRequestEvent
)


@dataclass
class StepResult:
    """What one forward pass produced, plus how long that pass costs.

    duration_ms is the engine's own cost for the step: measured for a real model,
    modelled for the simulated one. The worker uses it to pace, and the load
    experiments depend on it being a function of the seed and nothing else.
    """

    events: list[EngineEvent] = field(default_factory=list)
    duration_ms: float = 0.0
    batch_size: int = 0


class _Phase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"

    # Asked for more cache than its reservation allows and waiting for a `grant`. The
    # sequence keeps its blocks while it waits — dropping them would make the grant
    # useless by the time it arrived — but produces nothing, so the wait is bounded.
    WAITING_GRANT = "waiting_grant"


@dataclass
class _Sequence:
    spec: SequenceSpec
    prompt_tokens: int
    admit_order: int
    phase: _Phase = _Phase.PREFILL
    generated: int = 0

    # The live block cap: the reservation, plus whatever the control plane has
    # granted on top of it.
    cap_blocks: int = 0

    # Set while WAITING_GRANT, so the timeout is measured from the ask.
    waiting_since: float | None = None
    resume_phase: _Phase = _Phase.DECODE
    # Engine-specific per-sequence state: the RNG for the simulated engine, the
    # past_key_values handle for the real one.
    ext: dict[str, Any] = field(default_factory=dict)

    @property
    def context_tokens(self) -> int:
        return self.prompt_tokens + self.generated


class InferenceEngine(ABC):
    """What the worker needs from whatever is producing tokens."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def device(self) -> str: ...

    @property
    @abstractmethod
    def cache(self) -> PagedKVCache: ...

    @property
    @abstractmethod
    def max_batch_size(self) -> int: ...

    @abstractmethod
    def admit(self, spec: SequenceSpec) -> AdmitResult: ...

    @abstractmethod
    def cancel(self, request_id: str) -> list[EngineEvent]: ...

    @abstractmethod
    def grant(self, request_id: str, extra_blocks: int) -> bool: ...

    @abstractmethod
    def step(self) -> StepResult: ...

    @abstractmethod
    def holds(self, request_id: str) -> bool: ...

    @abstractmethod
    def active_sequences(self) -> int: ...

    @abstractmethod
    def has_work(self) -> bool: ...


class BaseEngine(InferenceEngine):
    """Continuous batching, cache accounting and preemption, independent of runtime."""

    def __init__(
        self,
        model_id: str,
        policy: CachePolicy,
        max_batch_size: int,
        device: str = "cpu",
        enforce_reservation: bool = True,
        grant_timeout_ms: float = 1000.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._model_id = model_id
        self._cache = PagedKVCache(policy)
        self._max_batch_size = max(1, max_batch_size)
        self._device = device
        self._enforce_reservation = enforce_reservation
        # How long a sequence waits for a `grant` before being preempted. Bounded
        # because a sequence waiting on a grant holds its blocks and produces nothing:
        # a control plane that never answers would otherwise park cache forever. The
        # default is twice the default heartbeat interval, so the control plane has
        # seen at least one heartbeat's worth of fresh state before the wait expires.
        self._grant_timeout_ms = grant_timeout_ms
        self._clock = clock
        self._active: dict[str, _Sequence] = {}
        # Ids that have ever been admitted, so a redelivery arriving after the
        # sequence finished is recognised as a duplicate rather than run a second
        # time. PROTOCOL.md guarantees only at-least-once.
        self._seen: set[str] = set()
        self._admit_counter = 0
        self._tokens_emitted = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> str:
        return self._device

    @property
    def cache(self) -> PagedKVCache:
        return self._cache

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def tokens_emitted(self) -> int:
        return self._tokens_emitted

    def holds(self, request_id: str) -> bool:
        return request_id in self._active

    def seen(self, request_id: str) -> bool:
        return request_id in self._seen

    def active_sequences(self) -> int:
        return len(self._active)

    def has_work(self) -> bool:
        return bool(self._active)

    def active_ids(self) -> tuple[str, ...]:
        return tuple(self._active)

    def generated(self, request_id: str) -> int:
        seq = self._active.get(request_id)
        return 0 if seq is None else seq.generated

    def admit(self, spec: SequenceSpec) -> AdmitResult:
        """Add a sequence to the running set, to be prefilled on the next step.

        Admission into the live set rather than into a staging queue is what makes a
        request admitted mid-flight start producing on the very next step instead of
        waiting for the current batch to drain.
        """
        if spec.request_id in self._seen:
            return AdmitResult.DUPLICATE
        if len(self._active) >= self._max_batch_size:
            return AdmitResult.REJECTED_BATCH_FULL

        prompt_tokens = max(1, self.count_prompt_tokens(spec.prompt))
        # Only what the prompt needs, not the declared worst case: taking the peak up
        # front is the contiguous policy's cost, and paging exists to avoid it.
        initial = max(1, self._cache.blocks_for(prompt_tokens))
        if self._enforce_reservation and initial > spec.reserved_blocks:
            return AdmitResult.REJECTED_TOO_LARGE
        if not self._cache.allocate(spec.request_id, initial):
            return AdmitResult.REJECTED_CAPACITY

        self._admit_counter += 1
        seq = _Sequence(
            spec=spec,
            prompt_tokens=prompt_tokens,
            admit_order=self._admit_counter,
            cap_blocks=spec.reserved_blocks,
        )
        self.on_admit(seq)
        self._active[spec.request_id] = seq
        self._seen.add(spec.request_id)
        return AdmitResult.ADMITTED

    def cancel(self, request_id: str) -> list[EngineEvent]:
        """Drop a sequence and free its blocks now.

        Now, not at natural completion: PROTOCOL.md calls prompt release the whole
        point of cancelling, and a cancelled sequence whose memory is reclaimed late
        is indistinguishable from not cancelling at all.
        """
        seq = self._active.pop(request_id, None)
        if seq is None:
            return []
        generated = seq.generated
        self.on_retire(seq)
        self._cache.release(request_id)
        return [CompletedEvent(request_id, generated, STOP_CANCELLED)]

    def step(self) -> StepResult:
        result = StepResult(batch_size=len(self._active))
        if not self._active:
            return result

        cost_ms = self.step_overhead_ms(len(self._active))
        # Snapshot the ids: preemption can evict a sequence other than the one being
        # advanced, so the live dict changes underneath this loop.
        for request_id in list(self._active):
            seq = self._active.get(request_id)
            if seq is None:
                continue
            try:
                cost_ms += self._advance(seq, result)
            except Exception as exc:  # a runtime failure is one request's problem
                logger.exception("engine: sequence %s failed", request_id)
                self._retire(seq)
                result.events.append(FailedEvent(request_id, f"{type(exc).__name__}: {exc}", True))

        result.duration_ms = cost_ms
        return result

    def _advance(self, seq: _Sequence, result: StepResult) -> float:
        """Advance one sequence by one position, returning its cost in ms."""
        request_id = seq.spec.request_id

        # Grow before producing: the position about to be written needs a home, and
        # discovering that after computing the token means either dropping a token
        # silently or writing outside the accounting.
        if not self._ensure_blocks(seq, seq.context_tokens + 1, result):
            return 0.0

        if seq.phase is _Phase.PREFILL:
            token, cost_ms, is_eos = self.prefill(seq)
            seq.phase = _Phase.DECODE
            seq.generated = 1
            self._tokens_emitted += 1
            # first_token is a separate message because time-to-first-token is
            # dominated by prefill while the rest is decode throughput; PROTOCOL.md
            # keeps them apart so a latency regression names its own cause.
            result.events.append(FirstTokenEvent(request_id, round(cost_ms, 4)))
            result.events.append(
                TokenEvent(request_id, token, seq.generated, self._cache.blocks(request_id))
            )
        else:
            token, cost_ms, is_eos = self.decode(seq)
            seq.generated += 1
            self._tokens_emitted += 1
            result.events.append(
                TokenEvent(request_id, token, seq.generated, self._cache.blocks(request_id))
            )

        if is_eos:
            self._finish(seq, STOP_EOS, result)
        elif seq.generated >= seq.spec.max_output_tokens:
            self._finish(seq, STOP_LENGTH, result)
        return cost_ms

    def _ensure_blocks(self, seq: _Sequence, target_tokens: int, result: StepResult) -> bool:
        """Make room for `target_tokens` positions, asking or preempting as needed.

        Returns False when the sequence must not be advanced this step: either it is
        waiting for a grant, or it was preempted.
        """
        request_id = seq.spec.request_id
        need = self._cache.blocks_for(target_tokens)
        held = self._cache.blocks(request_id)
        if need <= held:
            return True

        if self._enforce_reservation and need > seq.cap_blocks:
            return self._request_grant(seq, need - held, result)

        while self._cache.blocks(request_id) < need:
            if self._cache.grow(request_id):
                continue
            victim = self._choose_victim(exclude=request_id)
            if victim is None:
                # Nothing left to evict but this sequence, so this sequence is the
                # one that goes. Reported, never silently truncated: a system that
                # quietly preempts looks healthy from outside while doing a large
                # fraction of its work twice.
                self._preempt(seq, PREEMPT_CACHE_EXHAUSTED, result)
                return False
            self._preempt(victim, PREEMPT_CACHE_EXHAUSTED, result)
        return True

    def _request_grant(self, seq: _Sequence, want: int, result: StepResult) -> bool:
        """Ask the control plane to raise this sequence's cap, or give up on it.

        Asking rather than preempting is the point of `grow_request`: preempting a
        sequence the worker has free blocks to serve turns one low estimate into
        repeated work and tells the operator nothing about why.
        """
        free = self._cache.free()
        if free < want:
            # The device is the binding constraint, not the reservation, and no grant
            # can conjure blocks. The reason names what actually blocked it, because
            # the two reasons send the operator in opposite directions.
            self._preempt(seq, PREEMPT_CACHE_EXHAUSTED, result)
            return False

        if seq.phase is not _Phase.WAITING_GRANT:
            seq.resume_phase = seq.phase
            seq.phase = _Phase.WAITING_GRANT
            seq.waiting_since = self._clock()
            # Asked once per wait, not once per step: re-asking every step would turn
            # one slow answer into a message flood at exactly the moment the control
            # plane is busy.
            result.events.append(
                GrowRequestEvent(seq.spec.request_id, want_blocks=want, free_blocks=free)
            )
            return False

        waited_ms = (self._clock() - (seq.waiting_since or 0.0)) * 1000.0
        if waited_ms >= self._grant_timeout_ms:
            # No answer in time. The estimator was wrong and the control plane either
            # declined or cannot answer; either way the sequence has to go back so its
            # blocks serve someone who fits.
            self._preempt(seq, PREEMPT_RESERVATION_EXCEEDED, result)
        return False

    def grant(self, request_id: str, extra_blocks: int) -> bool:
        """Raise a sequence's cap by `extra_blocks` and let it resume.

        Additive to the existing cap rather than a replacement, matching the protocol's
        `extra_blocks`. Applied even to a sequence that is not currently waiting: a
        grant that arrives just after the ask was superseded is still permission, and
        discarding it would strand the sequence at the next boundary.
        """
        if extra_blocks <= 0:
            return False
        seq = self._active.get(request_id)
        if seq is None:
            # The sequence finished, was cancelled, or was preempted while the grant
            # was in flight. Nothing to raise, and nothing to complain about.
            return False
        seq.cap_blocks += extra_blocks
        if seq.phase is _Phase.WAITING_GRANT:
            seq.phase = seq.resume_phase
            seq.waiting_since = None
        return True

    def waiting_for_grant(self, request_id: str) -> bool:
        seq = self._active.get(request_id)
        return seq is not None and seq.phase is _Phase.WAITING_GRANT

    def cap_blocks(self, request_id: str) -> int:
        seq = self._active.get(request_id)
        return 0 if seq is None else seq.cap_blocks

    def _choose_victim(self, exclude: str) -> _Sequence | None:
        """Pick the sequence to evict: biggest holding first, newest to break ties.

        Biggest holding frees the most memory per eviction, so one preemption is
        more likely to be enough. Newest-first among equals preserves the progress of
        older sequences, which is both less wasted work and better head-of-line
        behaviour.
        """
        candidates = [s for rid, s in self._active.items() if rid != exclude]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda s: (self._cache.blocks(s.spec.request_id), s.admit_order),
        )

    def _preempt(self, seq: _Sequence, reason: str, result: StepResult) -> None:
        self._retire(seq)
        result.events.append(PreemptedEvent(seq.spec.request_id, seq.generated, reason))

    def _finish(self, seq: _Sequence, stop_reason: str, result: StepResult) -> None:
        generated = seq.generated
        self._retire(seq)
        result.events.append(CompletedEvent(seq.spec.request_id, generated, stop_reason))

    def _retire(self, seq: _Sequence) -> None:
        self._active.pop(seq.spec.request_id, None)
        self.on_retire(seq)
        self._cache.release(seq.spec.request_id)

    # Hooks for a runtime.

    @abstractmethod
    def count_prompt_tokens(self, prompt: str) -> int: ...

    @abstractmethod
    def prefill(self, seq: _Sequence) -> tuple[str, float, bool]:
        """Process the whole prompt and return (first token, cost ms, is_eos)."""

    @abstractmethod
    def decode(self, seq: _Sequence) -> tuple[str, float, bool]:
        """Produce one more token and return (token, cost ms, is_eos)."""

    def step_overhead_ms(self, batch_size: int) -> float:
        """Fixed cost of a forward pass, independent of per-sequence work."""
        return 0.0

    def on_admit(self, seq: _Sequence) -> None:
        """Set up per-sequence runtime state."""

    def on_retire(self, seq: _Sequence) -> None:
        """Drop per-sequence runtime state, including any real KV tensors."""


# A small closed vocabulary. Fixed rather than generated so a recorded token stream
# from one version of this file is readable in the next.
_SIM_VOCAB: tuple[str, ...] = (
    " the", " a", " model", " token", " cache", " block", " worker", " queue",
    " latency", " batch", " prompt", " decode", " prefill", " scheduler",
    " memory", " request", " and", " of", " to", " is", " not", " that",
    ".", ",", "\n",
)


class SimulatedEngine(BaseEngine):
    """Deterministic, seeded, no model.

    This is the engine the load experiments use, and the reason is not convenience:
    real inference on a CPU produces a few tokens per second, which cannot drive a
    scheduler into the overload region the scheduler exists to handle. The cost model
    is the part that has to be right — prefill roughly linear in prompt tokens,
    decode a per-step cost that grows mildly with context length, because that growth
    is what makes a long-running sequence progressively more expensive to keep and is
    therefore what the batching policy is trading against.

    Determinism is per sequence, not per engine: each sequence draws from an RNG
    seeded by (seed, request_id), so its token stream and its timings are identical
    no matter what else was interleaved with it. A single engine-wide RNG would make
    every stream depend on arrival order, and then a scheduling change would look
    like a model change.
    """

    def __init__(
        self,
        model_id: str = "simulated",
        policy: CachePolicy | None = None,
        max_batch_size: int = 32,
        seed: int = 0,
        prefill_ms_per_token: float = 0.25,
        prefill_base_ms: float = 2.0,
        decode_ms_per_token: float = 4.0,
        # Decode cost per step rises with context length; this is the slope per 1000
        # tokens of context, expressed as a fraction of the base per-step cost.
        context_cost_per_1k: float = 0.35,
        step_overhead_base_ms: float = 0.5,
        jitter: float = 0.05,
        chars_per_token: int = 4,
        eos_bias: float = 0.6,
        enforce_reservation: bool = True,
        grant_timeout_ms: float = 1000.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(
            model_id=model_id,
            policy=policy or CachePolicy(tokens_per_block=16, total_blocks=2048),
            max_batch_size=max_batch_size,
            device="simulated",
            enforce_reservation=enforce_reservation,
            grant_timeout_ms=grant_timeout_ms,
            clock=clock,
        )
        self._seed = seed
        self._prefill_ms_per_token = prefill_ms_per_token
        self._prefill_base_ms = prefill_base_ms
        self._decode_ms_per_token = decode_ms_per_token
        self._context_cost_per_1k = context_cost_per_1k
        self._step_overhead_base_ms = step_overhead_base_ms
        self._jitter = jitter
        self._chars_per_token = max(1, chars_per_token)
        self._eos_bias = eos_bias

    @property
    def seed(self) -> int:
        return self._seed

    def count_prompt_tokens(self, prompt: str) -> int:
        # A fixed characters-per-token ratio. Crude, but the experiments care that
        # prompt cost scales with prompt size and is reproducible, not that it
        # matches any particular tokenizer.
        return max(1, math.ceil(len(prompt) / self._chars_per_token))

    def on_admit(self, seq: _Sequence) -> None:
        rng = random.Random(f"{self._seed}:{seq.spec.request_id}")
        # The length the sequence would stop at on its own. Drawn once at admission
        # so it cannot change under re-dispatch, and biased toward the full budget so
        # the batch contains a mix of short and long sequences — a uniform draw makes
        # every sequence mid-length and hides the head-of-line problem that
        # continuous batching exists to solve.
        budget = max(1, seq.spec.max_output_tokens)
        if rng.random() < self._eos_bias:
            target = min(budget, 1 + int(rng.random() * budget))
        else:
            target = budget
        seq.ext["rng"] = rng
        seq.ext["target_len"] = target
        # A sequence that runs to its full budget stops because the budget ran out,
        # not because the model emitted end-of-sequence. Reporting eos there would
        # make the two stop reasons indistinguishable, and the control plane uses them
        # to tell a satisfied request from a truncated one.
        seq.ext["natural_stop"] = target < budget

    def on_retire(self, seq: _Sequence) -> None:
        seq.ext.pop("rng", None)

    def _jittered(self, rng: random.Random, base_ms: float) -> float:
        if self._jitter <= 0.0:
            return base_ms
        return base_ms * (1.0 + rng.uniform(-self._jitter, self._jitter))

    def _next_token(self, seq: _Sequence) -> tuple[str, bool]:
        rng: random.Random = seq.ext["rng"]
        token = _SIM_VOCAB[rng.randrange(len(_SIM_VOCAB))]
        is_eos = bool(seq.ext["natural_stop"]) and (seq.generated + 1) >= int(seq.ext["target_len"])
        return token, is_eos

    def prefill(self, seq: _Sequence) -> tuple[str, float, bool]:
        rng: random.Random = seq.ext["rng"]
        cost = self._prefill_base_ms + self._prefill_ms_per_token * seq.prompt_tokens
        token, is_eos = self._next_token(seq)
        return token, self._jittered(rng, cost), is_eos

    def decode(self, seq: _Sequence) -> tuple[str, float, bool]:
        rng: random.Random = seq.ext["rng"]
        growth = 1.0 + self._context_cost_per_1k * (seq.context_tokens / 1000.0)
        cost = self._decode_ms_per_token * growth
        token, is_eos = self._next_token(seq)
        return token, self._jittered(rng, cost), is_eos

    def step_overhead_ms(self, batch_size: int) -> float:
        return self._step_overhead_base_ms


# torch and transformers are optional and must stay optional: importing this module
# on a machine without them has to succeed, because the simulated engine is what the
# load experiments run and requiring a multi-hundred-megabyte dependency to run them
# would be absurd.
TRANSFORMERS_AVAILABLE = False
TRANSFORMERS_IMPORT_ERROR: str | None = None
try:  # pragma: no cover - availability depends on the environment
    import torch as _torch
    from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
    from transformers import AutoTokenizer as _AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except Exception as _exc:  # ImportError, but a broken install raises other things too
    _torch = None  # type: ignore[assignment]
    _AutoModelForCausalLM = None  # type: ignore[assignment]
    _AutoTokenizer = None  # type: ignore[assignment]
    TRANSFORMERS_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"


def transformers_unavailable_reason() -> str | None:
    """Why the real engine cannot run here, or None if it can."""
    return None if TRANSFORMERS_AVAILABLE else (TRANSFORMERS_IMPORT_ERROR or "torch/transformers not installed")


class TransformersEngine(BaseEngine):
    """A real model with genuine incremental decoding and a real KV cache.

    Its job is to prove the end-to-end path works against an actual runtime, so the
    decoding here is real: one forward pass per sequence per step, feeding back a
    single token and the previous `past_key_values` rather than re-running the prompt.
    Re-running the prompt each step would produce identical output while making the
    cost quadratic and the KV accounting fictional.

    Per-sequence KV state rather than one padded batch tensor: joining sequences into
    one tensor means the batch is only as flexible as its padding allows, and a
    sequence admitted mid-flight would force a reallocation of the whole batch. The
    cost is more Python-level forward passes, which does not matter because this
    engine is for proving correctness and the simulated one carries the load.
    """

    def __init__(
        self,
        model_id: str = "sshleifer/tiny-gpt2",
        policy: CachePolicy | None = None,
        max_batch_size: int = 8,
        device: str = "cpu",
        enforce_reservation: bool = True,
        local_files_only: bool = False,
        tokens_per_block: int = 16,
        cache_budget_bytes: int = 1 << 30,
        grant_timeout_ms: float = 1000.0,
    ) -> None:
        reason = transformers_unavailable_reason()
        if reason is not None:
            raise EngineUnavailable(reason)
        try:
            self._tokenizer = _AutoTokenizer.from_pretrained(
                model_id, local_files_only=local_files_only
            )
            self._model = _AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=local_files_only
            )
        except Exception as exc:
            # A missing download or an incompatible checkpoint is the same kind of
            # problem as a missing library from the worker's point of view: run
            # simulated instead of dying.
            raise EngineUnavailable(f"cannot load {model_id}: {type(exc).__name__}: {exc}") from exc
        self._model.eval()
        self._model.to(device)
        self._eos_id = getattr(self._tokenizer, "eos_token_id", None)
        self._bytes_per_token = self.measure_bytes_per_token()
        if policy is None:
            # The capacity reported at registration is derived from the loaded model's
            # real per-token cache footprint, not configured. PROTOCOL.md puts the
            # measurement here because only this process can see both numbers.
            policy = CachePolicy(
                tokens_per_block=tokens_per_block,
                total_blocks=max(
                    1,
                    int(cache_budget_bytes // max(1, self._bytes_per_token * tokens_per_block)),
                ),
            )
        super().__init__(
            model_id=model_id,
            policy=policy,
            max_batch_size=max_batch_size,
            device=device,
            enforce_reservation=enforce_reservation,
            grant_timeout_ms=grant_timeout_ms,
        )

    def measure_bytes_per_token(self) -> int:
        """KV bytes one sequence position costs: 2 tensors x layers x hidden x dtype."""
        cfg = self._model.config
        layers = int(getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", 1) or 1)
        hidden = int(getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", 1) or 1)
        heads = int(getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", 1) or 1)
        kv_heads = int(getattr(cfg, "num_key_value_heads", heads) or heads)
        head_dim = max(1, hidden // max(1, heads))
        dtype_size = int(getattr(self._model.dtype, "itemsize", 4) or 4)
        return 2 * layers * kv_heads * head_dim * dtype_size

    def count_prompt_tokens(self, prompt: str) -> int:
        ids = self._tokenizer.encode(prompt or " ")
        return max(1, len(ids))

    def on_retire(self, seq: _Sequence) -> None:
        # Dropping the tensors here is what makes the block accounting mean anything:
        # released blocks that still have live tensors behind them are a leak the
        # cache cannot see.
        seq.ext.pop("past", None)
        seq.ext.pop("last_id", None)

    def _forward(self, input_ids: Any, past: Any) -> tuple[int, Any]:
        with _torch.no_grad():
            out = self._model(input_ids=input_ids, past_key_values=past, use_cache=True)
        next_id = int(_torch.argmax(out.logits[0, -1, :]).item())
        return next_id, out.past_key_values

    def prefill(self, seq: _Sequence) -> tuple[str, float, bool]:
        started = time.perf_counter()
        ids = self._tokenizer.encode(seq.spec.prompt or " ")
        if not ids:
            ids = [self._eos_id if self._eos_id is not None else 0]
        input_ids = _torch.tensor([ids], dtype=_torch.long, device=self._device)
        next_id, past = self._forward(input_ids, None)
        seq.ext["past"] = past
        seq.ext["last_id"] = next_id
        cost_ms = (time.perf_counter() - started) * 1000.0
        return self._tokenizer.decode([next_id]), cost_ms, next_id == self._eos_id

    def decode(self, seq: _Sequence) -> tuple[str, float, bool]:
        started = time.perf_counter()
        last_id = int(seq.ext["last_id"])
        input_ids = _torch.tensor([[last_id]], dtype=_torch.long, device=self._device)
        next_id, past = self._forward(input_ids, seq.ext["past"])
        seq.ext["past"] = past
        seq.ext["last_id"] = next_id
        cost_ms = (time.perf_counter() - started) * 1000.0
        return self._tokenizer.decode([next_id]), cost_ms, next_id == self._eos_id


DEFAULT_SIMULATED_BLOCKS = 2048


def build_engine(
    model_id: str,
    simulated: bool,
    total_blocks: int | None = None,
    tokens_per_block: int = 16,
    max_batch_size: int = 32,
    seed: int = 0,
    grant_timeout_ms: float = 1000.0,
) -> tuple[InferenceEngine, str | None]:
    """Build the requested engine, falling back to simulated with a reason.

    Returning the reason rather than logging and swallowing it matters because
    "running simulated" is a fact about the measurements, and a run whose numbers came
    from the simulator while the operator believed otherwise is worse than a run that
    refused to start.

    total_blocks of None means "measure it": the real engine derives capacity from the
    loaded model's per-token footprint, which is the arrangement PROTOCOL.md requires.
    """

    def simulated_engine() -> SimulatedEngine:
        return SimulatedEngine(
            model_id=model_id,
            policy=CachePolicy(
                tokens_per_block=tokens_per_block,
                total_blocks=DEFAULT_SIMULATED_BLOCKS if total_blocks is None else total_blocks,
            ),
            max_batch_size=max_batch_size,
            seed=seed,
            grant_timeout_ms=grant_timeout_ms,
        )

    if simulated:
        return simulated_engine(), None
    try:
        policy = (
            None
            if total_blocks is None
            else CachePolicy(tokens_per_block=tokens_per_block, total_blocks=total_blocks)
        )
        return (
            TransformersEngine(
                model_id=model_id,
                policy=policy,
                max_batch_size=max_batch_size,
                tokens_per_block=tokens_per_block,
                grant_timeout_ms=grant_timeout_ms,
            ),
            None,
        )
    except EngineUnavailable as exc:
        return simulated_engine(), str(exc)
