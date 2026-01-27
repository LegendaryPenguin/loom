"""The real-model path, skipped cleanly when torch/transformers are absent.

Skipped rather than failed, and with the reason printed: the simulated engine is what
the load experiments run, so a machine without a multi-hundred-megabyte ML stack is a
supported configuration, not a broken one. What this test adds when it does run is the
only proof that the continuous-batching loop drives a real runtime — the simulated
engine shares the loop but not the tensors.
"""

from __future__ import annotations

import unittest

import harness  # noqa: F401

from loom_worker.engine import (
    CompletedEvent,
    EngineUnavailable,
    SequenceSpec,
    TokenEvent,
    TransformersEngine,
    transformers_unavailable_reason,
)
from loom_worker.kvcache import CachePolicy

UNAVAILABLE = transformers_unavailable_reason()
MODEL_ID = "sshleifer/tiny-gpt2"


class ImportSafetyTest(unittest.TestCase):
    def test_importing_the_module_never_requires_torch(self) -> None:
        # The import at the top of this file already proves it; the assertion is that
        # the module reports its own availability instead of raising at import time.
        import loom_worker.engine as engine_module

        self.assertIsInstance(engine_module.TRANSFORMERS_AVAILABLE, bool)
        if not engine_module.TRANSFORMERS_AVAILABLE:
            self.assertIsNotNone(transformers_unavailable_reason())

    def test_constructing_without_the_runtime_raises_a_typed_error(self) -> None:
        if UNAVAILABLE is None:
            self.skipTest("torch/transformers are installed, so this path is unreachable")
        with self.assertRaises(EngineUnavailable):
            TransformersEngine(model_id=MODEL_ID)

    def test_build_engine_falls_back_with_a_reason(self) -> None:
        from loom_worker.engine import SimulatedEngine, build_engine

        engine, reason = build_engine(model_id="nonexistent/model-that-is-not-real", simulated=False)
        if UNAVAILABLE is None and reason is None:
            self.skipTest("the runtime loaded a model under that name, which is unexpected but fine")
        self.assertIsInstance(engine, SimulatedEngine)
        self.assertIsInstance(reason, str)
        self.assertTrue(reason)


@unittest.skipIf(UNAVAILABLE is not None, f"real engine unavailable: {UNAVAILABLE}")
class TransformersEngineTest(unittest.TestCase):
    engine: TransformersEngine

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.engine = TransformersEngine(
                model_id=MODEL_ID,
                policy=CachePolicy(tokens_per_block=16, total_blocks=256),
                max_batch_size=4,
            )
        except EngineUnavailable as exc:
            # A missing download is an environment fact, not a defect in the worker.
            raise unittest.SkipTest(f"cannot load {MODEL_ID}: {exc}") from exc

    def test_produces_real_tokens_with_incremental_decoding(self) -> None:
        spec = SequenceSpec(
            request_id="r-real",
            prompt="The capital of France is",
            max_output_tokens=5,
            reserved_blocks=32,
        )
        from loom_worker.engine import AdmitResult

        self.assertIs(self.engine.admit(spec), AdmitResult.ADMITTED)
        tokens: list[TokenEvent] = []
        completed: list[CompletedEvent] = []
        for _ in range(20):
            if not self.engine.has_work():
                break
            for event in self.engine.step().events:
                if isinstance(event, TokenEvent):
                    tokens.append(event)
                elif isinstance(event, CompletedEvent):
                    completed.append(event)

        self.assertTrue(tokens)
        self.assertEqual([t.generated for t in tokens], list(range(1, len(tokens) + 1)))
        self.assertEqual(len(completed), 1)
        self.assertEqual(self.engine.cache.used(), 0, "completion released the blocks")
        self.engine.cache.check_invariant()

    def test_capacity_is_derived_from_the_model_rather_than_configured(self) -> None:
        measured = TransformersEngine(
            model_id=MODEL_ID, max_batch_size=2, tokens_per_block=16, cache_budget_bytes=1 << 20
        )
        self.assertGreater(measured.measure_bytes_per_token(), 0)
        self.assertGreater(measured.cache.total(), 0)

    def test_two_sequences_advance_together(self) -> None:
        from loom_worker.engine import AdmitResult

        for request_id, budget in (("r-short", 2), ("r-long", 6)):
            self.assertIs(
                self.engine.admit(
                    SequenceSpec(
                        request_id=request_id,
                        prompt="Loom serves tokens",
                        max_output_tokens=budget,
                        reserved_blocks=32,
                    )
                ),
                AdmitResult.ADMITTED,
            )
        order: list[str] = []
        for _ in range(20):
            if not self.engine.has_work():
                break
            for event in self.engine.step().events:
                if isinstance(event, CompletedEvent):
                    order.append(event.request_id)
        self.assertEqual(order, ["r-short", "r-long"])


if __name__ == "__main__":
    unittest.main()
