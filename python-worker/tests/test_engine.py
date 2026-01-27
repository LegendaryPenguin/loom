"""Engine tests: determinism, continuous batching, preemption, idempotent admission.

The continuous-batching test is the load-bearing one. Static batching and continuous
batching are indistinguishable on throughput for a uniform workload; the difference
only shows up as whether a sequence admitted while a longer one is running can make
progress before that longer one finishes. So that is asserted directly rather than
inferred from a rate.
"""

from __future__ import annotations

import unittest

import harness  # noqa: F401

from loom_worker.engine import (
    AdmitResult,
    CompletedEvent,
    FirstTokenEvent,
    PreemptedEvent,
    SequenceSpec,
    SimulatedEngine,
    StepResult,
    TokenEvent,
)
from loom_worker.engine import GrowRequestEvent
from loom_worker.kvcache import CachePolicy
from loom_worker.protocol import (
    PREEMPT_CACHE_EXHAUSTED,
    PREEMPT_RESERVATION_EXCEEDED,
    STOP_CANCELLED,
    STOP_EOS,
    STOP_LENGTH,
)

PROMPT = "the quick brown fox jumps over the lazy dog, repeatedly and at some length"


def spec(request_id: str, max_output: int = 8, prompt: str = PROMPT, reserved: int = 64) -> SequenceSpec:
    return SequenceSpec(
        request_id=request_id,
        prompt=prompt,
        max_output_tokens=max_output,
        reserved_blocks=reserved,
    )


def run_to_quiescence(engine: SimulatedEngine, max_steps: int = 5000) -> list[StepResult]:
    results: list[StepResult] = []
    steps = 0
    while engine.has_work():
        steps += 1
        if steps > max_steps:
            raise AssertionError("engine did not quiesce")
        results.append(engine.step())
    return results


def token_streams(results: list[StepResult]) -> dict[str, list[str]]:
    streams: dict[str, list[str]] = {}
    for result in results:
        for event in result.events:
            if isinstance(event, TokenEvent):
                streams.setdefault(event.request_id, []).append(event.token)
    return streams


class DeterminismTest(unittest.TestCase):
    def _trace(self, seed: int) -> tuple[dict[str, list[str]], list[float], list[float]]:
        engine = SimulatedEngine(
            policy=CachePolicy(tokens_per_block=16, total_blocks=512),
            max_batch_size=8,
            seed=seed,
        )
        for i in range(5):
            self.assertIs(
                engine.admit(spec(f"r-{i}", max_output=12 + 3 * i, prompt=PROMPT * (i + 1))),
                AdmitResult.ADMITTED,
            )
        results = run_to_quiescence(engine)
        durations = [r.duration_ms for r in results]
        first_tokens = [
            e.latency_ms for r in results for e in r.events if isinstance(e, FirstTokenEvent)
        ]
        return token_streams(results), durations, first_tokens

    def test_same_seed_produces_identical_streams_and_timings(self) -> None:
        a_streams, a_durations, a_first = self._trace(1234)
        b_streams, b_durations, b_first = self._trace(1234)
        self.assertEqual(a_streams, b_streams)
        # Byte-identical, not approximately equal: the load experiments compare runs
        # across code changes, and a tolerance would hide the drift they exist to catch.
        self.assertEqual(
            ["".join(v) for v in a_streams.values()], ["".join(v) for v in b_streams.values()]
        )
        self.assertEqual(a_durations, b_durations)
        self.assertEqual(a_first, b_first)
        self.assertGreater(len(a_durations), 5)

    def test_a_different_seed_produces_a_different_stream(self) -> None:
        a_streams, _, _ = self._trace(1234)
        b_streams, _, _ = self._trace(5678)
        self.assertNotEqual(a_streams, b_streams)

    def test_a_sequence_stream_does_not_depend_on_what_it_was_batched_with(self) -> None:
        # Per-sequence RNGs rather than one engine RNG: otherwise a scheduling change
        # alters every token stream and looks like a model change.
        def stream_of_r1(companions: int) -> list[str]:
            engine = SimulatedEngine(
                policy=CachePolicy(tokens_per_block=16, total_blocks=512),
                max_batch_size=8,
                seed=99,
            )
            engine.admit(spec("r-1", max_output=10))
            for i in range(companions):
                engine.admit(spec(f"other-{i}", max_output=20, prompt=PROMPT * 2))
            return token_streams(run_to_quiescence(engine))["r-1"]

        self.assertEqual(stream_of_r1(0), stream_of_r1(3))

    def test_prefill_cost_scales_with_prompt_length(self) -> None:
        engine = SimulatedEngine(
            policy=CachePolicy(tokens_per_block=16, total_blocks=512), seed=7, jitter=0.0
        )
        engine.admit(spec("short", max_output=1, prompt="tiny"))
        engine.admit(spec("long", max_output=1, prompt=PROMPT * 40))
        first = {
            e.request_id: e.latency_ms
            for e in engine.step().events
            if isinstance(e, FirstTokenEvent)
        }
        self.assertGreater(first["long"], 10 * first["short"])

    def test_decode_cost_grows_with_context_length(self) -> None:
        engine = SimulatedEngine(
            policy=CachePolicy(tokens_per_block=16, total_blocks=4096),
            seed=3,
            jitter=0.0,
            eos_bias=0.0,
            step_overhead_base_ms=0.0,
        )
        engine.admit(spec("r-1", max_output=400, prompt=PROMPT))
        results = run_to_quiescence(engine)
        early = results[5].duration_ms
        late = results[-5].duration_ms
        self.assertGreater(late, early, "a longer context must cost more per step")


class ContinuousBatchingTest(unittest.TestCase):
    def test_a_sequence_admitted_mid_flight_produces_before_the_long_one_finishes(self) -> None:
        engine = SimulatedEngine(
            policy=CachePolicy(tokens_per_block=16, total_blocks=1024),
            max_batch_size=8,
            seed=11,
            eos_bias=0.0,  # every sequence runs to its full budget, so lengths are known
        )
        self.assertIs(engine.admit(spec("long", max_output=200)), AdmitResult.ADMITTED)

        long_tokens_before_admit = 0
        for _ in range(10):
            long_tokens_before_admit += sum(
                1 for e in engine.step().events if isinstance(e, TokenEvent)
            )
        self.assertEqual(long_tokens_before_admit, 10)
        self.assertTrue(engine.holds("long"))

        # Admitted while "long" is mid-decode, with no gap in its progress.
        self.assertIs(engine.admit(spec("late", max_output=5)), AdmitResult.ADMITTED)

        late_first_step: int | None = None
        late_completed_step: int | None = None
        long_completed_step: int | None = None
        long_tokens_during = 0
        for step_index in range(1, 400):
            result = engine.step()
            for event in result.events:
                if isinstance(event, FirstTokenEvent) and event.request_id == "late":
                    late_first_step = step_index
                if isinstance(event, TokenEvent) and event.request_id == "long":
                    long_tokens_during += 1
                if isinstance(event, CompletedEvent):
                    if event.request_id == "late":
                        late_completed_step = step_index
                    if event.request_id == "long":
                        long_completed_step = step_index
            if long_completed_step is not None:
                break

        self.assertEqual(late_first_step, 1, "the late sequence prefills on the very next step")
        self.assertIsNotNone(late_completed_step)
        self.assertIsNotNone(long_completed_step)
        assert late_completed_step is not None and long_completed_step is not None
        # The property that distinguishes continuous from static batching: the short
        # sequence finishes while the long one is still running, rather than waiting
        # for the batch it joined to drain.
        self.assertLess(late_completed_step, long_completed_step)
        # And the long sequence kept making progress every step meanwhile, so the
        # admission did not stall the batch.
        self.assertEqual(long_tokens_during, long_completed_step)

    def test_a_retired_sequence_frees_room_for_the_next_admission(self) -> None:
        engine = SimulatedEngine(
            policy=CachePolicy(tokens_per_block=16, total_blocks=1024),
            max_batch_size=2,
            seed=5,
            eos_bias=0.0,
        )
        engine.admit(spec("a", max_output=3))
        engine.admit(spec("b", max_output=50))
        self.assertIs(engine.admit(spec("c", max_output=3)), AdmitResult.REJECTED_BATCH_FULL)
        for _ in range(4):
            engine.step()
        self.assertFalse(engine.holds("a"))
        self.assertIs(engine.admit(spec("c", max_output=3)), AdmitResult.ADMITTED)
        self.assertTrue(engine.holds("b"), "the long sequence was never interrupted")

    def test_stop_reasons_distinguish_eos_from_length(self) -> None:
        engine = SimulatedEngine(
            policy=CachePolicy(tokens_per_block=16, total_blocks=512), seed=42
        )
        engine.admit(spec("budgeted", max_output=6, reserved=64))
        completed = [
            e for r in run_to_quiescence(engine) for e in r.events if isinstance(e, CompletedEvent)
        ]
        self.assertEqual(len(completed), 1)
        self.assertIn(completed[0].stop_reason, (STOP_EOS, STOP_LENGTH))
        self.assertLessEqual(completed[0].generated, 6)

    def test_length_stop_is_reported_when_the_budget_is_the_binding_limit(self) -> None:
        engine = SimulatedEngine(
            policy=CachePolicy(tokens_per_block=16, total_blocks=512), seed=42, eos_bias=0.0
        )
        engine.admit(spec("budgeted", max_output=6))
        completed = [
            e for r in run_to_quiescence(engine) for e in r.events if isinstance(e, CompletedEvent)
        ]
        self.assertEqual(completed[0].stop_reason, STOP_LENGTH)
        self.assertEqual(completed[0].generated, 6)

    def test_token_events_count_up_from_one_and_report_blocks(self) -> None:
        engine = SimulatedEngine(
            policy=CachePolicy(tokens_per_block=4, total_blocks=512), seed=8, eos_bias=0.0
        )
        engine.admit(spec("r-1", max_output=20))
        tokens = [
            e for r in run_to_quiescence(engine) for e in r.events if isinstance(e, TokenEvent)
        ]
        self.assertEqual([e.generated for e in tokens], list(range(1, 21)))
        self.assertEqual(sorted(e.blocks for e in tokens), [e.blocks for e in tokens])
        self.assertGreater(tokens[-1].blocks, tokens[0].blocks, "the cache grew as it decoded")


class CacheExhaustionTest(unittest.TestCase):
    def test_exhaustion_mid_decode_preempts_rather_than_crashing(self) -> None:
        # Three blocks of four tokens: both sequences fit their prompts, neither can
        # grow twice, so growth mid-decode must evict somebody.
        engine = SimulatedEngine(
            policy=CachePolicy(tokens_per_block=4, total_blocks=3),
            max_batch_size=4,
            seed=17,
            eos_bias=0.0,
        )
        short_prompt = "abcdefghijklmnop"  # 16 chars -> 4 tokens -> exactly one block
        self.assertIs(
            engine.admit(spec("a", max_output=50, prompt=short_prompt)), AdmitResult.ADMITTED
        )
        self.assertIs(
            engine.admit(spec("b", max_output=50, prompt=short_prompt)), AdmitResult.ADMITTED
        )
        self.assertEqual(engine.cache.used(), 2)

        preemptions: list[PreemptedEvent] = []
        tokens = 0
        for _ in range(40):
            result = engine.step()
            for event in result.events:
                if isinstance(event, PreemptedEvent):
                    preemptions.append(event)
                if isinstance(event, TokenEvent):
                    tokens += 1
            engine.cache.check_invariant()

        self.assertTrue(preemptions, "exhaustion must be reported, not silently absorbed")
        self.assertTrue(all(p.reason == PREEMPT_CACHE_EXHAUSTED for p in preemptions))
        self.assertTrue(
            all(p.generated > 0 for p in preemptions),
            "a preemption reports the work the control plane will have to redo",
        )
        self.assertGreater(tokens, 0, "the survivor kept running")
        self.assertLessEqual(engine.cache.used(), engine.cache.total())

    def test_a_preempted_sequence_is_not_also_reported_completed(self) -> None:
        engine = SimulatedEngine(
            policy=CachePolicy(tokens_per_block=4, total_blocks=2),
            max_batch_size=4,
            seed=2,
            eos_bias=0.0,
        )
        engine.admit(spec("a", max_output=50, prompt="abcdefghijklmnop"))
        engine.admit(spec("b", max_output=50, prompt="abcdefghijklmnop"))
        events = [e for r in [engine.step() for _ in range(6)] for e in r.events]
        preempted = {e.request_id for e in events if isinstance(e, PreemptedEvent)}
        completed = {e.request_id for e in events if isinstance(e, CompletedEvent)}
        self.assertTrue(preempted)
        self.assertFalse(preempted & completed, "one terminal event per sequence per attempt")

    def test_over_reservation_with_no_free_blocks_is_cache_exhausted_not_reservation(self) -> None:
        # Both limits are violated at once, and the reason has to name the one that
        # actually blocked it: no grant can conjure a block that does not exist, so
        # this is "add workers", not "fix the estimator".
        engine = SimulatedEngine(
            policy=CachePolicy(tokens_per_block=4, total_blocks=1), seed=1, eos_bias=0.0
        )
        engine.admit(spec("tight", max_output=50, prompt="abcdefghijklmnop", reserved=1))
        events = [e for r in [engine.step() for _ in range(3)] for e in r.events]
        preempted = [e for e in events if isinstance(e, PreemptedEvent)]
        self.assertEqual(len(preempted), 1)
        self.assertEqual(preempted[0].reason, PREEMPT_CACHE_EXHAUSTED)
        self.assertEqual([e for e in events if isinstance(e, GrowRequestEvent)], [])

    def test_a_prompt_larger_than_its_reservation_is_refused_at_admission(self) -> None:
        engine = SimulatedEngine(
            policy=CachePolicy(tokens_per_block=4, total_blocks=64), seed=1
        )
        self.assertIs(
            engine.admit(spec("huge", prompt=PROMPT * 10, reserved=1)),
            AdmitResult.REJECTED_TOO_LARGE,
        )
        self.assertEqual(engine.cache.used(), 0, "a refused admission holds nothing")

    def test_admission_is_refused_when_the_cache_is_full(self) -> None:
        engine = SimulatedEngine(
            policy=CachePolicy(tokens_per_block=4, total_blocks=1), max_batch_size=8, seed=1
        )
        self.assertIs(
            engine.admit(spec("a", prompt="abcdefghijklmnop")), AdmitResult.ADMITTED
        )
        self.assertIs(
            engine.admit(spec("b", prompt="abcdefghijklmnop")), AdmitResult.REJECTED_CAPACITY
        )
        self.assertEqual(engine.cache.sequences(), 1)


class GrowRequestTest(unittest.TestCase):
    """Crossing the reservation asks first; preemption is the fallback, not the rule."""

    def make_engine(self, grant_timeout_ms: float = 1000.0, total_blocks: int = 64) -> SimulatedEngine:
        return SimulatedEngine(
            policy=CachePolicy(tokens_per_block=4, total_blocks=total_blocks),
            max_batch_size=4,
            seed=3,
            eos_bias=0.0,
            grant_timeout_ms=grant_timeout_ms,
        )

    # A 16-character prompt is 4 tokens, which is exactly one block at this block size,
    # so the sequence starts holding precisely its reservation of 1.
    PROMPT_ONE_BLOCK = "abcdefghijklmnop"

    def test_crossing_the_reservation_asks_instead_of_preempting(self) -> None:
        engine = self.make_engine()
        engine.admit(spec("tight", max_output=12, prompt=self.PROMPT_ONE_BLOCK, reserved=1))
        result = engine.step()

        asks = [e for e in result.events if isinstance(e, GrowRequestEvent)]
        self.assertEqual(len(asks), 1)
        self.assertEqual(asks[0].request_id, "tight")
        self.assertEqual(asks[0].want_blocks, 1, "it asks for what it needs, not for its peak")
        self.assertEqual(asks[0].free_blocks, 63, "and reports the room it has")
        # Nothing preempted and nothing produced: the sequence is parked, not evicted,
        # because dropping its blocks would make the grant useless when it arrived.
        self.assertEqual([e for e in result.events if isinstance(e, PreemptedEvent)], [])
        self.assertEqual([e for e in result.events if isinstance(e, TokenEvent)], [])
        self.assertTrue(engine.holds("tight"))
        self.assertTrue(engine.waiting_for_grant("tight"))
        self.assertEqual(engine.cache.blocks("tight"), 1)

    def test_it_asks_once_per_wait_rather_than_once_per_step(self) -> None:
        engine = self.make_engine()
        engine.admit(spec("tight", max_output=12, prompt=self.PROMPT_ONE_BLOCK, reserved=1))
        events = [e for r in [engine.step() for _ in range(5)] for e in r.events]
        self.assertEqual(len([e for e in events if isinstance(e, GrowRequestEvent)]), 1)

    def test_a_grant_lets_the_sequence_exceed_its_reservation_and_finish(self) -> None:
        engine = self.make_engine()
        engine.admit(spec("tight", max_output=12, prompt=self.PROMPT_ONE_BLOCK, reserved=1))
        engine.step()
        self.assertTrue(engine.waiting_for_grant("tight"))

        self.assertTrue(engine.grant("tight", 4))
        self.assertFalse(engine.waiting_for_grant("tight"))
        self.assertEqual(engine.cap_blocks("tight"), 5, "additive to the reservation")

        results = run_to_quiescence(engine)
        events = [e for r in results for e in r.events]
        tokens = [e for e in events if isinstance(e, TokenEvent)]
        completed = [e for e in events if isinstance(e, CompletedEvent)]
        self.assertEqual([e for e in events if isinstance(e, PreemptedEvent)], [])
        self.assertEqual([t.generated for t in tokens], list(range(1, 13)))
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].generated, 12)
        self.assertEqual(completed[0].stop_reason, STOP_LENGTH)
        # It really did exceed the original reservation of 1 block.
        self.assertGreater(max(t.blocks for t in tokens), 1)
        self.assertEqual(engine.cache.used(), 0)
        engine.cache.check_invariant()

    def test_no_grant_within_the_timeout_preempts_as_reservation_exceeded(self) -> None:
        # Timeout of zero: the ask goes out on the first step and the wait has already
        # expired by the second, which makes the no-answer path deterministic.
        engine = self.make_engine(grant_timeout_ms=0.0)
        engine.admit(spec("tight", max_output=12, prompt=self.PROMPT_ONE_BLOCK, reserved=1))

        first = engine.step()
        self.assertEqual(len([e for e in first.events if isinstance(e, GrowRequestEvent)]), 1)
        second = engine.step()
        preempted = [e for e in second.events if isinstance(e, PreemptedEvent)]
        self.assertEqual(len(preempted), 1)
        self.assertEqual(preempted[0].request_id, "tight")
        # The estimator was wrong, and the worker had room, so this is not exhaustion.
        self.assertEqual(preempted[0].reason, PREEMPT_RESERVATION_EXCEEDED)
        self.assertFalse(engine.holds("tight"))
        self.assertEqual(engine.cache.used(), 0, "a preempted sequence releases its blocks")
        self.assertEqual(
            [e for e in second.events if isinstance(e, CompletedEvent)],
            [],
            "preempted and completed are different outcomes",
        )

    def test_a_late_grant_still_raises_the_cap_of_a_running_sequence(self) -> None:
        engine = self.make_engine()
        engine.admit(spec("r-1", max_output=12, prompt=self.PROMPT_ONE_BLOCK, reserved=8))
        engine.step()
        self.assertFalse(engine.waiting_for_grant("r-1"), "it is within its reservation")
        self.assertTrue(engine.grant("r-1", 2), "permission is still permission")
        self.assertEqual(engine.cap_blocks("r-1"), 10)

    def test_a_grant_for_a_sequence_that_is_gone_is_ignored(self) -> None:
        engine = self.make_engine()
        self.assertFalse(engine.grant("never-existed", 4))
        engine.admit(spec("r-1", max_output=2, prompt=self.PROMPT_ONE_BLOCK, reserved=8))
        run_to_quiescence(engine)
        # The grant and the sequence's end race; the loser is discarded quietly.
        self.assertFalse(engine.grant("r-1", 4))

    def test_an_empty_grant_is_refused(self) -> None:
        engine = self.make_engine()
        engine.admit(spec("r-1", max_output=12, prompt=self.PROMPT_ONE_BLOCK, reserved=1))
        self.assertFalse(engine.grant("r-1", 0))
        self.assertFalse(engine.grant("r-1", -3))
        self.assertEqual(engine.cap_blocks("r-1"), 1)

    def test_a_sequence_waiting_for_a_grant_does_not_stall_the_others(self) -> None:
        engine = self.make_engine()
        engine.admit(spec("waiting", max_output=12, prompt=self.PROMPT_ONE_BLOCK, reserved=1))
        engine.admit(spec("running", max_output=12, prompt=self.PROMPT_ONE_BLOCK, reserved=32))

        produced = 0
        for _ in range(6):
            for event in engine.step().events:
                if isinstance(event, TokenEvent):
                    self.assertEqual(event.request_id, "running")
                    produced += 1
        self.assertEqual(produced, 6, "the batch keeps moving while one member waits")
        self.assertTrue(engine.waiting_for_grant("waiting"))


class IdempotenceAndCancelTest(unittest.TestCase):
    def test_duplicate_dispatch_is_idempotent(self) -> None:
        engine = SimulatedEngine(
            policy=CachePolicy(tokens_per_block=16, total_blocks=512), seed=4, eos_bias=0.0
        )
        self.assertIs(engine.admit(spec("r-1", max_output=20)), AdmitResult.ADMITTED)
        used_after_first = engine.cache.used()
        for _ in range(3):
            self.assertIs(engine.admit(spec("r-1", max_output=20)), AdmitResult.DUPLICATE)
        self.assertEqual(engine.active_sequences(), 1)
        self.assertEqual(engine.cache.used(), used_after_first)

        engine.step()
        engine.step()
        # A redelivery mid-flight must not reset progress either.
        self.assertIs(engine.admit(spec("r-1", max_output=20)), AdmitResult.DUPLICATE)
        self.assertEqual(engine.generated("r-1"), 2)

        results = run_to_quiescence(engine)
        completed = [e for r in results for e in r.events if isinstance(e, CompletedEvent)]
        self.assertEqual(len(completed), 1, "one dispatch, one completion, however many copies")

    def test_duplicate_dispatch_after_completion_is_still_a_no_op(self) -> None:
        engine = SimulatedEngine(
            policy=CachePolicy(tokens_per_block=16, total_blocks=512), seed=4, eos_bias=0.0
        )
        engine.admit(spec("r-1", max_output=2))
        run_to_quiescence(engine)
        self.assertIs(engine.admit(spec("r-1", max_output=2)), AdmitResult.DUPLICATE)
        self.assertFalse(engine.has_work())

    def test_cancel_releases_blocks_before_natural_completion(self) -> None:
        engine = SimulatedEngine(
            policy=CachePolicy(tokens_per_block=4, total_blocks=512),
            seed=6,
            eos_bias=0.0,
        )
        engine.admit(spec("doomed", max_output=500))
        engine.admit(spec("survivor", max_output=500))
        for _ in range(20):
            engine.step()
        before = engine.cache.used()
        doomed_blocks = engine.cache.blocks("doomed")
        self.assertGreater(doomed_blocks, 1, "it has grown past its prompt")

        events = engine.cancel("doomed")
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], CompletedEvent)
        self.assertEqual(events[0].stop_reason, STOP_CANCELLED)
        self.assertGreater(events[0].generated, 0)

        self.assertEqual(engine.cache.used(), before - doomed_blocks)
        self.assertEqual(engine.cache.blocks("doomed"), 0)
        self.assertTrue(engine.holds("survivor"), "cancelling one must not disturb the other")
        engine.cache.check_invariant()

        # Still far from its natural end, which is the point of asserting the drop now.
        self.assertLess(engine.generated("survivor"), 500)

    def test_cancelling_an_unknown_sequence_yields_no_events(self) -> None:
        engine = SimulatedEngine(policy=CachePolicy(tokens_per_block=4, total_blocks=8), seed=0)
        self.assertEqual(engine.cancel("never-heard-of-it"), [])

    def test_step_on_an_empty_engine_is_harmless(self) -> None:
        engine = SimulatedEngine(policy=CachePolicy(tokens_per_block=4, total_blocks=8), seed=0)
        result = engine.step()
        self.assertEqual(result.events, [])
        self.assertEqual(result.batch_size, 0)


if __name__ == "__main__":
    unittest.main()
