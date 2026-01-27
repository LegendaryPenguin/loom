"""KV-cache accounting tests.

This number is reported to the control plane as authoritative and overwrites its
estimate, so these tests are about exactness rather than plausibility: every path
that touches the cache has to leave `used + free == total` true.
"""

from __future__ import annotations

import random
import unittest

import harness  # noqa: F401

from loom_worker.kvcache import CachePolicy, PagedKVCache


class BlocksForTest(unittest.TestCase):
    def test_ceiling_division(self) -> None:
        policy = CachePolicy(tokens_per_block=16, total_blocks=10)
        self.assertEqual(policy.blocks_for(0), 0)
        self.assertEqual(policy.blocks_for(1), 1)
        self.assertEqual(policy.blocks_for(16), 1)
        self.assertEqual(policy.blocks_for(17), 2)
        self.assertEqual(policy.blocks_for(32), 2)


class AccountingTest(unittest.TestCase):
    def test_exact_under_a_long_randomized_sequence(self) -> None:
        cache = PagedKVCache(CachePolicy(tokens_per_block=8, total_blocks=97))
        rng = random.Random(20240917)
        live: dict[str, int] = {}
        next_id = 0

        for _ in range(20000):
            action = rng.random()
            if action < 0.4 or not live:
                request_id = f"r-{next_id}"
                next_id += 1
                want = rng.randint(1, 12)
                if cache.allocate(request_id, want):
                    live[request_id] = want
            elif action < 0.75:
                request_id = rng.choice(list(live))
                want = rng.randint(1, 3)
                if cache.grow(request_id, want):
                    live[request_id] += want
            else:
                request_id = rng.choice(list(live))
                freed = cache.release(request_id)
                self.assertEqual(freed, live.pop(request_id))

            cache.check_invariant()
            self.assertEqual(cache.used(), sum(live.values()))
            self.assertEqual(cache.free(), cache.total() - cache.used())
            self.assertEqual(cache.sequences(), len(live))

        for request_id in list(live):
            cache.release(request_id)
        self.assertEqual(cache.used(), 0)
        self.assertEqual(cache.free(), cache.total())

    def test_double_release_is_a_no_op(self) -> None:
        # Preemption, cancellation and natural completion race; double release must
        # not corrupt the accounting.
        cache = PagedKVCache(CachePolicy(tokens_per_block=4, total_blocks=8))
        cache.allocate("r-1", 3)
        self.assertEqual(cache.release("r-1"), 3)
        self.assertEqual(cache.release("r-1"), 0)
        self.assertEqual(cache.release("never-existed"), 0)
        cache.check_invariant()
        self.assertEqual(cache.used(), 0)


class CapacityTest(unittest.TestCase):
    def test_allocation_at_capacity_fails_with_nothing_partially_held(self) -> None:
        cache = PagedKVCache(CachePolicy(tokens_per_block=16, total_blocks=10))
        self.assertTrue(cache.allocate("r-1", 7))
        self.assertFalse(cache.allocate("r-2", 4), "must refuse rather than overcommit")
        self.assertEqual(cache.blocks("r-2"), 0, "a refused allocation holds nothing")
        self.assertEqual(cache.sequences(), 1)
        self.assertEqual(cache.used(), 7)
        self.assertEqual(cache.free(), 3)
        cache.check_invariant()
        # The capacity that was free is still usable by someone who fits.
        self.assertTrue(cache.allocate("r-3", 3))
        self.assertEqual(cache.free(), 0)

    def test_allocation_of_exactly_the_remainder_succeeds(self) -> None:
        cache = PagedKVCache(CachePolicy(tokens_per_block=16, total_blocks=5))
        self.assertTrue(cache.allocate("r-1", 2))
        self.assertTrue(cache.allocate("r-2", 3))
        self.assertEqual(cache.free(), 0)
        self.assertFalse(cache.allocate("r-3", 1))

    def test_nonpositive_and_duplicate_allocations_are_refused(self) -> None:
        cache = PagedKVCache(CachePolicy(tokens_per_block=16, total_blocks=4))
        self.assertFalse(cache.allocate("r-1", 0))
        self.assertFalse(cache.allocate("r-1", -1))
        self.assertTrue(cache.allocate("r-1", 1))
        # Topping up under one id would make release free an amount nobody asked for,
        # and at-least-once dispatch makes the duplicate call routine.
        self.assertFalse(cache.allocate("r-1", 1))
        self.assertEqual(cache.blocks("r-1"), 1)

    def test_grow_is_all_or_nothing_at_capacity(self) -> None:
        cache = PagedKVCache(CachePolicy(tokens_per_block=4, total_blocks=4))
        cache.allocate("r-1", 3)
        self.assertFalse(cache.grow("r-1", 2), "two blocks are not available")
        self.assertEqual(cache.blocks("r-1"), 3, "the partial block was not taken")
        self.assertTrue(cache.grow("r-1", 1))
        self.assertFalse(cache.grow("r-1", 1))
        cache.check_invariant()

    def test_grow_on_an_unknown_sequence_fails(self) -> None:
        cache = PagedKVCache(CachePolicy(tokens_per_block=4, total_blocks=4))
        self.assertFalse(cache.grow("r-nope"))


class GrowthTest(unittest.TestCase):
    def test_grow_across_a_block_boundary(self) -> None:
        policy = CachePolicy(tokens_per_block=4, total_blocks=16)
        cache = PagedKVCache(policy)
        tokens = 4
        self.assertTrue(cache.allocate("r-1", policy.blocks_for(tokens)))
        self.assertEqual(cache.blocks("r-1"), 1)

        # Token 5 crosses into a second block: the step before it needs no growth, the
        # step at the boundary does, and that asymmetry is the whole reason Grow exists.
        tokens += 1
        self.assertEqual(policy.blocks_for(tokens), 2)
        self.assertTrue(cache.grow("r-1"))
        self.assertEqual(cache.blocks("r-1"), 2)
        self.assertEqual(cache.used(), 2)

        for _ in range(3):
            tokens += 1
            self.assertEqual(policy.blocks_for(tokens), 2, "mid-block tokens are free")
        tokens += 1
        self.assertEqual(policy.blocks_for(tokens), 3)
        cache.check_invariant()

    def test_release_makes_blocks_reusable(self) -> None:
        cache = PagedKVCache(CachePolicy(tokens_per_block=16, total_blocks=6))
        self.assertTrue(cache.allocate("big", 6))
        self.assertFalse(cache.allocate("waiting", 6))
        cache.release("big")
        # Immediately reusable by any other sequence, with no placement constraint:
        # the property the contiguous policy cannot offer.
        self.assertTrue(cache.allocate("waiting", 6))
        self.assertEqual(cache.used(), 6)
        self.assertEqual(cache.fragmentation(), 0)
        cache.check_invariant()

    def test_freed_blocks_are_reused_in_full_regardless_of_shape(self) -> None:
        cache = PagedKVCache(CachePolicy(tokens_per_block=16, total_blocks=8))
        for i in range(8):
            self.assertTrue(cache.allocate(f"r-{i}", 1))
        # Free an interleaved, non-adjacent set, then take all of it as one allocation.
        for i in (1, 3, 5, 7):
            cache.release(f"r-{i}")
        self.assertEqual(cache.free(), 4)
        self.assertTrue(cache.allocate("scattered", 4))
        self.assertEqual(cache.free(), 0)
        cache.check_invariant()


class DegenerateTest(unittest.TestCase):
    def test_zero_capacity_cache_refuses_everything(self) -> None:
        cache = PagedKVCache(CachePolicy(tokens_per_block=16, total_blocks=0))
        self.assertFalse(cache.allocate("r-1", 1))
        self.assertEqual(cache.total(), 0)
        cache.check_invariant()

    def test_negative_total_is_clamped(self) -> None:
        cache = PagedKVCache(CachePolicy(tokens_per_block=16, total_blocks=-5))
        self.assertEqual(cache.total(), 0)
        cache.check_invariant()


if __name__ == "__main__":
    unittest.main()
