"""One request, end to end over a socket, with the whole message sequence asserted.

The individual behaviours are covered elsewhere; what this test adds is the ORDER.
register before anything else, first_token before any token, tokens numbered without
a gap, completed last and only once. Those are the parts the control plane's own
state machine depends on, and none of them is visible from inside the worker.
"""

from __future__ import annotations

import unittest

import harness
from harness import FakeControlPlane, dispatch_item, dispatch_line

from loom_worker import protocol as proto
from loom_worker.engine import SimulatedEngine
from loom_worker.kvcache import CachePolicy

PROMPT = "the quick brown fox jumps over the lazy dog, several times over, at length"
MAX_OUTPUT = 9


class EndToEndTest(unittest.TestCase):
    def test_full_message_sequence_for_one_request(self) -> None:
        engine = SimulatedEngine(
            model_id="tiny-gpt2",
            policy=CachePolicy(tokens_per_block=4, total_blocks=128),
            max_batch_size=4,
            seed=2024,
            eos_bias=0.0,  # the budget is the binding limit, so the length is known
        )
        cp = FakeControlPlane()
        self.addCleanup(cp.close)
        worker, thread = harness.start_worker(engine, cp, worker_id="w-e2e")
        self.addCleanup(worker.stop)

        # Nothing is dispatched until the worker has registered, matching the control
        # plane's own rule.
        cp.wait_for(lambda m: bool(m) and isinstance(m[0], proto.Register))
        cp.send_line(
            dispatch_line(
                [
                    dispatch_item(
                        "r-7",
                        prompt=PROMPT,
                        max_output_tokens=MAX_OUTPUT,
                        reserved_blocks=32,
                    )
                ]
            )
        )
        msgs = cp.wait_for(
            lambda m: any(isinstance(x, proto.Completed) and x.request_id == "r-7" for x in m),
            timeout_s=15.0,
        )

        register = msgs[0]
        assert isinstance(register, proto.Register)
        self.assertEqual(register.worker_id, "w-e2e")
        self.assertEqual(register.model_id, "tiny-gpt2")
        self.assertEqual(register.tokens_per_block, 4)
        self.assertEqual(register.total_cache_blocks, 128)

        self.assertTrue(
            any(isinstance(m, proto.Heartbeat) for m in msgs), "the worker must heartbeat"
        )

        for_request = [
            m
            for m in msgs
            if isinstance(m, (proto.FirstToken, proto.Token, proto.Completed, proto.Failed, proto.Preempted))
            and m.request_id == "r-7"
        ]
        self.assertEqual(
            [type(m).__name__ for m in for_request],
            ["FirstToken"] + ["Token"] * MAX_OUTPUT + ["Completed"],
        )

        first = for_request[0]
        assert isinstance(first, proto.FirstToken)
        self.assertGreater(first.latency_ms, 0.0, "time to first token is prefill, and prefill costs")

        tokens = [m for m in for_request if isinstance(m, proto.Token)]
        self.assertEqual([t.generated for t in tokens], list(range(1, MAX_OUTPUT + 1)))
        self.assertTrue(all(t.token for t in tokens), "every token carries text")
        # The prompt is 19 tokens at 4 tokens per block, so the sequence starts on 5
        # blocks and takes more as it decodes past each boundary.
        self.assertGreaterEqual(tokens[0].blocks, 5)
        self.assertGreater(tokens[-1].blocks, tokens[0].blocks)
        self.assertEqual(sorted(t.blocks for t in tokens), [t.blocks for t in tokens])

        completed = for_request[-1]
        assert isinstance(completed, proto.Completed)
        self.assertEqual(completed.generated, MAX_OUTPUT)
        self.assertEqual(completed.stop_reason, proto.STOP_LENGTH)

        # Completion released the blocks, and the next heartbeat says so: the report is
        # authoritative, so a worker that completed without releasing would convince the
        # control plane that capacity had permanently vanished.
        self.assertEqual(engine.cache.used(), 0)
        cp.wait_for(
            lambda m: any(
                isinstance(x, proto.Heartbeat)
                and x.used_cache_blocks == 0
                and x.active_sequences == 0
                for x in m[m.index(completed) :]
            ),
            timeout_s=10.0,
        )

        self.assertEqual(cp.of_type(proto.Failed), [])
        self.assertEqual(cp.of_type(proto.Preempted), [])

    def test_two_requests_interleave_on_one_connection(self) -> None:
        engine = SimulatedEngine(
            policy=CachePolicy(tokens_per_block=16, total_blocks=256),
            max_batch_size=4,
            seed=99,
            eos_bias=0.0,
        )
        cp = FakeControlPlane()
        self.addCleanup(cp.close)
        worker, _ = harness.start_worker(engine, cp)
        self.addCleanup(worker.stop)

        cp.send_line(
            dispatch_line(
                [
                    dispatch_item("r-a", prompt=PROMPT, max_output_tokens=20),
                    dispatch_item("r-b", prompt=PROMPT, max_output_tokens=4),
                ]
            )
        )
        msgs = cp.wait_for(
            lambda m: {
                x.request_id for x in m if isinstance(x, proto.Completed)
            } == {"r-a", "r-b"},
            timeout_s=15.0,
        )

        order = [m.request_id for m in msgs if isinstance(m, proto.Completed)]
        # The short request finishes first even though both were dispatched together,
        # which is only possible if the batch is continuous rather than static.
        self.assertEqual(order, ["r-b", "r-a"])

        a_tokens = [m for m in msgs if isinstance(m, proto.Token) and m.request_id == "r-a"]
        b_tokens = [m for m in msgs if isinstance(m, proto.Token) and m.request_id == "r-b"]
        self.assertEqual(len(a_tokens), 20)
        self.assertEqual(len(b_tokens), 4)
        # Interleaved on the wire, one position per sequence per step.
        b_last_index = msgs.index(b_tokens[-1])
        a_after_b = [m for m in msgs[b_last_index:] if isinstance(m, proto.Token) and m.request_id == "r-a"]
        self.assertTrue(a_after_b, "the long request kept producing after the short one ended")


if __name__ == "__main__":
    unittest.main()
