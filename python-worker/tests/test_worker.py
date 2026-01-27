"""Worker tests over a real socket: registration, idempotence, cancel, drain.

Every one of these drives the worker through the wire rather than calling its
methods. The behaviours being checked — a duplicate dispatch, a cancel racing a
decode step, a drain — are all about what arrives when, and a direct method call
decides the ordering the test was supposed to be checking.
"""

from __future__ import annotations

import logging
import unittest

import harness
from harness import FakeControlPlane, cancel_line, dispatch_item, dispatch_line, drain_line

from loom_worker import protocol as proto
from loom_worker.engine import SimulatedEngine
from loom_worker.kvcache import CachePolicy

LONG_PROMPT = "the quick brown fox jumps over the lazy dog" * 2


class WorkerTestCase(unittest.TestCase):
    def make_engine(self, **kwargs: object) -> SimulatedEngine:
        params: dict[str, object] = {
            "model_id": "tiny-gpt2",
            "policy": CachePolicy(tokens_per_block=16, total_blocks=256),
            "max_batch_size": 8,
            "seed": 1234,
            "eos_bias": 0.0,  # deterministic lengths: every sequence runs its budget
        }
        params.update(kwargs)
        return SimulatedEngine(**params)  # type: ignore[arg-type]

    def start(self, engine: SimulatedEngine, **config: object) -> None:
        self.cp = FakeControlPlane()
        self.addCleanup(self.cp.close)
        self.worker, self.thread = harness.start_worker(engine, self.cp, **config)
        self.addCleanup(self.worker.stop)

    def completed_for(self, request_id: str) -> list[proto.Completed]:
        return [m for m in self.cp.of_type(proto.Completed) if m.request_id == request_id]


class RegistrationTest(WorkerTestCase):
    def test_registers_with_its_own_measured_capacity(self) -> None:
        engine = self.make_engine(policy=CachePolicy(tokens_per_block=32, total_blocks=777))
        self.start(engine)
        msgs = self.cp.wait_for(lambda m: len(m) >= 1)
        first = msgs[0]
        self.assertIsInstance(first, proto.Register)
        assert isinstance(first, proto.Register)
        self.assertEqual(first.worker_id, "w-test")
        self.assertEqual(first.model_id, "tiny-gpt2")
        self.assertEqual(first.total_cache_blocks, 777)
        self.assertEqual(first.tokens_per_block, 32)
        self.assertEqual(first.max_batch_size, 8)
        self.assertEqual(first.device, "simulated")

    def test_heartbeats_report_live_cache_usage(self) -> None:
        self.start(self.make_engine())
        self.cp.send_line(
            dispatch_line([dispatch_item("r-1", prompt=LONG_PROMPT, max_output_tokens=400)])
        )
        beats = self.cp.wait_for(
            lambda m: any(
                isinstance(x, proto.Heartbeat) and x.used_cache_blocks > 0 and x.active_sequences == 1
                for x in m
            )
        )
        live = [
            b for b in beats if isinstance(b, proto.Heartbeat) and b.used_cache_blocks > 0
        ][0]
        self.assertEqual(live.worker_id, "w-test")
        self.assertEqual(live.active_sequences, 1)
        self.assertEqual(live.queue_depth, 0, "a worker that buffers hides depth from the scheduler")
        self.assertGreaterEqual(live.tokens_per_second, 0.0)


class IdempotenceTest(WorkerTestCase):
    def test_duplicate_dispatch_is_a_no_op(self) -> None:
        self.start(self.make_engine())
        item = dispatch_item("r-dup", prompt=LONG_PROMPT, max_output_tokens=6)
        # At-least-once delivery: the same batch arriving twice, which the control
        # plane cannot rule out over a socket that can drop an acknowledgement.
        self.cp.send_line(dispatch_line([item]))
        self.cp.send_line(dispatch_line([item]))
        self.cp.send_line(dispatch_line([item]))

        self.cp.wait_for(lambda m: any(isinstance(x, proto.Completed) for x in m))
        # Give any second copy a chance to show up before asserting it did not.
        self.cp.send_line(dispatch_line([dispatch_item("r-sentinel", max_output_tokens=2)]))
        self.cp.wait_for(
            lambda m: any(
                isinstance(x, proto.Completed) and x.request_id == "r-sentinel" for x in m
            )
        )

        tokens = [m for m in self.cp.of_type(proto.Token) if m.request_id == "r-dup"]
        self.assertEqual([t.generated for t in tokens], [1, 2, 3, 4, 5, 6])
        self.assertEqual(
            len([m for m in self.cp.of_type(proto.FirstToken) if m.request_id == "r-dup"]), 1
        )
        self.assertEqual(len(self.completed_for("r-dup")), 1)
        self.assertEqual(self.cp.of_type(proto.Failed), [], "a duplicate is not a failure")


class CancelTest(WorkerTestCase):
    def test_cancel_frees_blocks_before_the_sequence_would_have_finished(self) -> None:
        engine = self.make_engine()
        self.start(engine)
        # A budget far larger than the test's lifetime, so any observed release is a
        # consequence of the cancel and not of natural completion.
        self.cp.send_line(
            dispatch_line(
                [dispatch_item("r-long", prompt=LONG_PROMPT, max_output_tokens=100_000)]
            )
        )
        self.cp.wait_for(
            lambda m: sum(
                1 for x in m if isinstance(x, proto.Token) and x.request_id == "r-long"
            )
            >= 3
        )
        self.assertGreater(engine.cache.used(), 0)

        self.cp.send_line(cancel_line("r-long"))
        msgs = self.cp.wait_for(
            lambda m: any(
                isinstance(x, proto.Completed) and x.request_id == "r-long" for x in m
            )
        )
        completed = [
            x for x in msgs if isinstance(x, proto.Completed) and x.request_id == "r-long"
        ][0]
        self.assertEqual(completed.stop_reason, proto.STOP_CANCELLED)
        self.assertGreater(completed.generated, 0)
        self.assertLess(completed.generated, 100_000, "it was cancelled, not finished")

        self.assertEqual(engine.cache.used(), 0, "blocks are released promptly, not at completion")
        self.assertEqual(engine.active_sequences(), 0)
        # And the release is visible to the control plane on the next heartbeat, which
        # is the only way admission learns the capacity came back.
        self.cp.wait_for(
            lambda m: any(
                isinstance(x, proto.Heartbeat) and x.used_cache_blocks == 0 and x.active_sequences == 0
                for x in m[m.index(completed) :]
            )
        )

    def test_cancel_for_an_unknown_request_is_ignored(self) -> None:
        self.start(self.make_engine())
        self.cp.send_line(cancel_line("r-never-existed"))
        self.cp.send_line(dispatch_line([dispatch_item("r-after", max_output_tokens=2)]))
        # The worker is still working, and it did not invent a second completion for a
        # request it never had.
        self.cp.wait_for(
            lambda m: any(isinstance(x, proto.Completed) and x.request_id == "r-after" for x in m)
        )
        self.assertEqual([c.request_id for c in self.cp.of_type(proto.Completed)], ["r-after"])


class GrowRequestTest(WorkerTestCase):
    """The grow_request / grant round trip, and both preemption reasons, over the wire."""

    # 16 characters is 4 tokens, exactly one block at the block size used here, so a
    # reservation of one block is satisfied by the prompt and crossed by the output.
    ONE_BLOCK_PROMPT = "abcdefghijklmnop"

    def small_block_engine(self, **kwargs: object) -> SimulatedEngine:
        return self.make_engine(policy=CachePolicy(tokens_per_block=4, total_blocks=64), **kwargs)

    def test_grow_request_and_grant_let_a_sequence_exceed_its_reservation_and_finish(self) -> None:
        self.cp = FakeControlPlane(auto_grant=True)
        self.addCleanup(self.cp.close)
        self.worker, self.thread = harness.start_worker(self.small_block_engine(), self.cp)
        self.addCleanup(self.worker.stop)

        self.cp.send_line(
            dispatch_line(
                [
                    dispatch_item(
                        "r-tight",
                        prompt=self.ONE_BLOCK_PROMPT,
                        max_output_tokens=12,
                        reserved_blocks=1,
                    )
                ]
            )
        )
        msgs = self.cp.wait_for(
            lambda m: any(
                isinstance(x, proto.Completed) and x.request_id == "r-tight" for x in m
            ),
            timeout_s=15.0,
        )

        asks = [m for m in msgs if isinstance(m, proto.GrowRequest)]
        self.assertTrue(asks, "it must ask before exceeding the reservation")
        self.assertEqual(asks[0].request_id, "r-tight")
        self.assertEqual(asks[0].want_blocks, 1)
        self.assertGreater(asks[0].free_blocks, 0, "it only asks when it has the room")

        # The grant arrived, so the sequence ran to its full budget rather than being
        # preempted on a worker that had capacity to serve it.
        self.assertEqual(self.cp.of_type(proto.Preempted), [])
        completed = self.completed_for("r-tight")
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].generated, 12)
        self.assertEqual(completed[0].stop_reason, proto.STOP_LENGTH)
        tokens = [m for m in msgs if isinstance(m, proto.Token) and m.request_id == "r-tight"]
        self.assertEqual([t.generated for t in tokens], list(range(1, 13)))
        self.assertGreater(max(t.blocks for t in tokens), 1, "it really did exceed one block")

    def test_a_grow_request_that_is_never_granted_preempts_as_reservation_exceeded(self) -> None:
        # The fake control plane records the ask and says nothing, which is the case the
        # timeout exists for.
        self.start(self.small_block_engine(grant_timeout_ms=20.0))
        self.cp.send_line(
            dispatch_line(
                [
                    dispatch_item(
                        "r-tight",
                        prompt=self.ONE_BLOCK_PROMPT,
                        max_output_tokens=12,
                        reserved_blocks=1,
                    )
                ]
            )
        )
        msgs = self.cp.wait_for(
            lambda m: any(isinstance(x, proto.Preempted) for x in m), timeout_s=15.0
        )
        self.assertTrue([m for m in msgs if isinstance(m, proto.GrowRequest)], "it asked first")

        preempted = [m for m in msgs if isinstance(m, proto.Preempted)][0]
        self.assertEqual(preempted.request_id, "r-tight")
        self.assertEqual(preempted.reason, proto.PREEMPT_RESERVATION_EXCEEDED)
        self.assertEqual(self.completed_for("r-tight"), [], "preempted is not completed")

    def test_a_grant_arriving_after_the_sequence_is_gone_is_harmless(self) -> None:
        self.start(self.small_block_engine())
        self.cp.send_line(
            dispatch_line([dispatch_item("r-done", max_output_tokens=3, reserved_blocks=32)])
        )
        self.cp.wait_for(
            lambda m: any(isinstance(x, proto.Completed) and x.request_id == "r-done" for x in m)
        )
        self.cp.send_line(harness.grant_line("r-done", 4))
        # Still alive and serving afterwards.
        self.cp.send_line(
            dispatch_line([dispatch_item("r-next", max_output_tokens=3, reserved_blocks=32)])
        )
        self.cp.wait_for(
            lambda m: any(isinstance(x, proto.Completed) and x.request_id == "r-next" for x in m)
        )

    def test_genuine_exhaustion_preempts_as_cache_exhausted(self) -> None:
        # Three blocks between two sequences that each need two: the device is the
        # binding constraint, so no grant could help and none is asked for.
        self.start(
            self.make_engine(policy=CachePolicy(tokens_per_block=4, total_blocks=3)),
            heartbeat_interval_ms=20,
        )
        self.cp.send_line(
            dispatch_line(
                [
                    dispatch_item(
                        "r-a",
                        prompt=self.ONE_BLOCK_PROMPT,
                        max_output_tokens=200,
                        reserved_blocks=64,
                    ),
                    dispatch_item(
                        "r-b",
                        prompt=self.ONE_BLOCK_PROMPT,
                        max_output_tokens=200,
                        reserved_blocks=64,
                    ),
                ]
            )
        )
        msgs = self.cp.wait_for(
            lambda m: any(isinstance(x, proto.Preempted) for x in m), timeout_s=15.0
        )
        preempted = [m for m in msgs if isinstance(m, proto.Preempted)]
        self.assertTrue(all(p.reason == proto.PREEMPT_CACHE_EXHAUSTED for p in preempted))
        self.assertEqual(self.cp.of_type(proto.GrowRequest), [], "a grant cannot create blocks")


class BackpressureTest(WorkerTestCase):
    def test_a_full_inbound_channel_replies_rejected_not_failed(self) -> None:
        # The queue is filled directly, before the step loop is running to drain it:
        # racing the reader thread against the loop would make this test flaky about
        # the one thing it is asserting.
        cp = FakeControlPlane()
        self.addCleanup(cp.close)
        from loom_worker.worker import Worker, WorkerConfig, connect

        conn = connect(cp.addr)
        self.addCleanup(conn.close)
        worker = Worker(
            self.make_engine(), WorkerConfig(worker_id="w-full", inbound_capacity=1), conn
        )
        self.addCleanup(worker.stop)
        cp.wait_connected()

        accepted = dispatch_line([dispatch_item("r-accepted")])
        overflow = proto.decode_line(dispatch_line([dispatch_item("r-overflow")]))
        assert overflow is not None
        worker._offer(proto.decode_line(accepted))  # fills the single slot
        worker._offer(overflow)  # no room, so this one is refused

        msgs = cp.wait_for(lambda m: any(isinstance(x, proto.Rejected) for x in m))
        rejected = [m for m in msgs if isinstance(m, proto.Rejected)]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].request_id, "r-overflow")
        self.assertEqual(rejected[0].reason, proto.REJECT_BACKPRESSURE)
        # Overload must not look like a sick worker: one means the fleet is too small,
        # the other means this host needs attention.
        self.assertEqual(cp.of_type(proto.Failed), [])

    def test_a_full_batch_replies_rejected_not_failed(self) -> None:
        self.start(self.make_engine(max_batch_size=1))
        self.cp.send_line(
            dispatch_line(
                [
                    dispatch_item("r-1", prompt=LONG_PROMPT, max_output_tokens=400),
                    dispatch_item("r-2", prompt=LONG_PROMPT, max_output_tokens=400),
                ]
            )
        )
        msgs = self.cp.wait_for(lambda m: any(isinstance(x, proto.Rejected) for x in m))
        rejected = [m for m in msgs if isinstance(m, proto.Rejected)][0]
        self.assertEqual(rejected.request_id, "r-2")
        self.assertEqual(rejected.reason, proto.REJECT_BACKPRESSURE)
        self.assertEqual(self.cp.of_type(proto.Failed), [])

    def test_a_cache_too_full_to_admit_replies_rejected(self) -> None:
        self.start(self.make_engine(policy=CachePolicy(tokens_per_block=4, total_blocks=1)))
        self.cp.send_line(
            dispatch_line(
                [
                    dispatch_item("r-1", prompt="abcdefghijklmnop", max_output_tokens=400),
                    dispatch_item("r-2", prompt="abcdefghijklmnop", max_output_tokens=400),
                ]
            )
        )
        msgs = self.cp.wait_for(lambda m: any(isinstance(x, proto.Rejected) for x in m))
        self.assertEqual([m.request_id for m in msgs if isinstance(m, proto.Rejected)], ["r-2"])


class DrainTest(WorkerTestCase):
    def test_drain_finishes_in_flight_work_rejects_new_work_then_exits(self) -> None:
        # A short heartbeat interval and enough in-flight work that the drain spans
        # several of them, so "keeps heartbeating while draining" is actually tested
        # rather than being true by the drain finishing first.
        self.start(self.make_engine(), heartbeat_interval_ms=5)
        self.cp.send_line(
            dispatch_line(
                [
                    dispatch_item("r-1", prompt=LONG_PROMPT, max_output_tokens=120),
                    dispatch_item("r-2", prompt=LONG_PROMPT, max_output_tokens=100),
                ]
            )
        )
        # Wait until both are genuinely in flight, so the drain is draining something.
        self.cp.wait_for(
            lambda m: {x.request_id for x in m if isinstance(x, proto.FirstToken)} == {"r-1", "r-2"}
        )

        self.cp.send_line(drain_line())
        self.cp.send_line(dispatch_line([dispatch_item("r-late", max_output_tokens=4)]))

        msgs = self.cp.wait_for(
            lambda m: any(isinstance(x, proto.Failed) and x.request_id == "r-late" for x in m)
        )
        rejected = [x for x in msgs if isinstance(x, proto.Failed)][0]
        self.assertTrue(rejected.retryable, "the request is fine; this worker is leaving")
        self.assertIn("draining", rejected.error)

        self.thread.join(timeout=10.0)
        self.assertFalse(self.thread.is_alive(), "a drained worker exits")
        self.assertEqual(self.worker.exit_reason, "drained")

        # In-flight work completed rather than being failed or truncated.
        self.assertEqual(len(self.completed_for("r-1")), 1)
        self.assertEqual(len(self.completed_for("r-2")), 1)
        self.assertEqual(self.completed_for("r-1")[0].generated, 120)
        self.assertEqual(self.completed_for("r-2")[0].generated, 100)
        self.assertNotIn("r-late", {t.request_id for t in self.cp.of_type(proto.Token)})

        # It kept heartbeating while draining: PROTOCOL.md requires it, because a
        # silent drain looks exactly like a dead worker and gets reaped three misses
        # later, which would fail the very work the drain is protecting.
        final = self.cp.messages()
        after_drain = final[final.index(rejected) :]
        self.assertTrue([m for m in after_drain if isinstance(m, proto.Heartbeat)])

    def test_drain_with_nothing_in_flight_exits_immediately(self) -> None:
        self.start(self.make_engine())
        self.cp.wait_for(lambda m: any(isinstance(x, proto.Heartbeat) for x in m))
        self.cp.send_line(drain_line())
        self.thread.join(timeout=10.0)
        self.assertEqual(self.worker.exit_reason, "drained")


class ToleranceTest(WorkerTestCase):
    def test_an_unknown_message_type_does_not_take_the_worker_down(self) -> None:
        # The skip is logged by the worker's reader thread, so it is quieted here
        # rather than captured: assertLogs across threads would race the step loop.
        quiet = logging.getLogger("loom_worker.protocol")
        previous = quiet.level
        quiet.setLevel(logging.ERROR)
        self.addCleanup(quiet.setLevel, previous)

        self.start(self.make_engine())
        self.cp.send_line('{"type":"reshard","target":"w-2"}\n')
        self.cp.send_line("this line is not json at all\n")
        self.cp.send_line(dispatch_line([dispatch_item("r-after-junk", max_output_tokens=3)]))
        self.cp.wait_for(
            lambda m: any(
                isinstance(x, proto.Completed) and x.request_id == "r-after-junk" for x in m
            )
        )

    def test_a_prompt_too_large_for_its_reservation_fails_unretryably(self) -> None:
        self.start(self.make_engine())
        self.cp.send_line(
            dispatch_line(
                [dispatch_item("r-big", prompt="x" * 4096, max_output_tokens=4, reserved_blocks=1)]
            )
        )
        msgs = self.cp.wait_for(lambda m: any(isinstance(x, proto.Failed) for x in m))
        failure = [x for x in msgs if isinstance(x, proto.Failed)][0]
        self.assertEqual(failure.request_id, "r-big")
        # Not retryable: it will fail the same way on every worker, and claiming
        # otherwise turns one bad request into an infinite retry loop.
        self.assertFalse(failure.retryable)


if __name__ == "__main__":
    unittest.main()
