"""Block-based KV-cache accounting on the worker side.

This mirrors the Go control plane's PagedCache semantics deliberately, including
the refusals: allocating for an id that already holds blocks fails rather than
topping up, and a failed allocation leaves nothing behind. The two halves of the
system reconcile these numbers on every heartbeat (PROTOCOL.md: the worker's
`used_cache_blocks` overwrites the control plane's estimate), so if the semantics
diverge the reconciliation turns a modelling difference into a phantom capacity
change.

This is the authoritative number. Everything downstream of a wrong value here is
an overcommit, so the invariant `used + free == total` is checked rather than
assumed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


class CacheError(Exception):
    """The cache's accounting invariant was violated, which is a bug, not a state."""


@dataclass(frozen=True)
class CachePolicy:
    """How the cache is divided. Mirrors the Go CachePolicy."""

    tokens_per_block: int
    total_blocks: int

    def blocks_for(self, tokens: int) -> int:
        """Blocks a sequence of `tokens` positions occupies.

        Ceiling division, which is where internal fragmentation comes from: one
        token into a new block holds the whole block.
        """
        if self.tokens_per_block <= 0:
            return 0
        if tokens <= 0:
            return 0
        return (tokens + self.tokens_per_block - 1) // self.tokens_per_block


class PagedKVCache:
    """Fixed-size blocks, no adjacency requirement, so any free block fits anyone.

    Fragmentation is identically zero and that is the whole argument for paging:
    there is no placement decision, so no free block is ever unusable, a sequence
    can start with only what its prompt needs, and a finished sequence's blocks are
    immediately available to every other sequence.
    """

    def __init__(self, policy: CachePolicy) -> None:
        total = max(0, policy.total_blocks)
        self._policy = policy
        self._total = total
        self._lock = threading.Lock()
        # A stack, matching the Go side: most-recently-freed first keeps the live
        # block-metadata working set small, and blocks are interchangeable so there
        # is no placement quality to lose.
        self._free: list[int] = list(range(total - 1, -1, -1))
        self._held: dict[str, list[int]] = {}
        self._used = 0

    @property
    def policy(self) -> CachePolicy:
        return self._policy

    @property
    def tokens_per_block(self) -> int:
        return self._policy.tokens_per_block

    def blocks_for(self, tokens: int) -> int:
        return self._policy.blocks_for(tokens)

    def allocate(self, request_id: str, blocks: int) -> bool:
        """Reserve `blocks` for a sequence, all or nothing.

        An id that already holds blocks is refused rather than topped up: merging
        two allocations under one id would make release() free an amount nobody
        asked for, and at-least-once dispatch makes duplicate allocate calls for
        one request a routine event rather than a bug.
        """
        if blocks <= 0:
            return False
        with self._lock:
            if request_id in self._held:
                return False
            if len(self._free) < blocks:
                # Nothing is taken before the capacity check passes, so a refused
                # allocation cannot leak a partial hold.
                return False
            taken = self._free[len(self._free) - blocks :]
            del self._free[len(self._free) - blocks :]
            self._held[request_id] = taken
            self._used += blocks
            return True

    def grow(self, request_id: str, blocks: int = 1) -> bool:
        """Extend a running sequence, all or nothing.

        False means the sequence must be preempted or the step must fail; it cannot
        simply continue, because the positions it is about to write have nowhere to
        live.
        """
        if blocks <= 0:
            return False
        with self._lock:
            held = self._held.get(request_id)
            if held is None:
                return False
            if len(self._free) < blocks:
                return False
            taken = self._free[len(self._free) - blocks :]
            del self._free[len(self._free) - blocks :]
            held.extend(taken)
            self._used += blocks
            return True

    def release(self, request_id: str) -> int:
        """Free everything held by a sequence and return how many blocks came back.

        Releasing an unknown id is a no-op rather than an error: preemption,
        cancellation and natural completion all race, and double release must not
        corrupt the accounting.
        """
        with self._lock:
            held = self._held.pop(request_id, None)
            if held is None:
                return 0
            self._free.extend(held)
            self._used -= len(held)
            return len(held)

    def used(self) -> int:
        with self._lock:
            return self._used

    def free(self) -> int:
        with self._lock:
            return len(self._free)

    def total(self) -> int:
        return self._total

    def fragmentation(self) -> int:
        """Always zero under paging, which is precisely the property being claimed."""
        return 0

    def blocks(self, request_id: str) -> int:
        with self._lock:
            held = self._held.get(request_id)
            return 0 if held is None else len(held)

    def sequences(self) -> int:
        """Sequences currently holding blocks: the concurrency actually sustained."""
        with self._lock:
            return len(self._held)

    def holders(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._held)

    def check_invariant(self) -> None:
        """Raise if the accounting has drifted. Called from tests and after release
        paths, because a drifted cache reports capacity that does not exist and the
        control plane trusts the report."""
        with self._lock:
            if self._used + len(self._free) != self._total:
                raise CacheError(
                    f"used({self._used}) + free({len(self._free)}) != total({self._total})"
                )
            if self._used != sum(len(v) for v in self._held.values()):
                raise CacheError("used does not match the sum of per-sequence holdings")
            if len(set(self._free)) != len(self._free):
                raise CacheError("a block index appears twice in the free list")
