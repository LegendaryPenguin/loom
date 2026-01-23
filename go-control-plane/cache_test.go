package loom

import (
	"fmt"
	"math/rand"
	"testing"
)

func testPolicy(tokensPerBlock, total int) CachePolicy {
	return CachePolicy{TokensPerBlock: tokensPerBlock, TotalBlocks: total}
}

func checkInvariant(t *testing.T, c KVCache, label string) {
	t.Helper()
	if c.Used()+c.Free() != c.Total() {
		t.Fatalf("%s: Used(%d)+Free(%d) != Total(%d)", label, c.Used(), c.Free(), c.Total())
	}
	if c.Used() < 0 || c.Free() < 0 {
		t.Fatalf("%s: negative accounting used=%d free=%d", label, c.Used(), c.Free())
	}
}

func TestPagedCacheAllocateGrowReleaseAccounting(t *testing.T) {
	c := NewPagedCache(testPolicy(16, 64))
	checkInvariant(t, c, "fresh")
	if c.Free() != 64 {
		t.Fatalf("fresh cache free = %d, want 64", c.Free())
	}

	if !c.Allocate("a", 10) {
		t.Fatal("allocate a failed on an empty cache")
	}
	if c.Used() != 10 || c.Free() != 54 {
		t.Fatalf("after allocate: used=%d free=%d, want 10/54", c.Used(), c.Free())
	}
	checkInvariant(t, c, "after allocate")

	for i := 0; i < 5; i++ {
		if !c.Grow("a") {
			t.Fatalf("grow %d failed with %d blocks free", i, c.Free())
		}
	}
	if c.Blocks("a") != 15 || c.Used() != 15 {
		t.Fatalf("after 5 grows: blocks=%d used=%d, want 15/15", c.Blocks("a"), c.Used())
	}
	checkInvariant(t, c, "after grows")

	c.Release("a")
	if c.Used() != 0 || c.Free() != 64 {
		t.Fatalf("after release: used=%d free=%d, want 0/64", c.Used(), c.Free())
	}
	if c.Blocks("a") != 0 {
		t.Fatalf("released sequence still holds %d blocks", c.Blocks("a"))
	}

	// Double release must not credit the free list twice, which would let the cache
	// hand out blocks that do not exist.
	c.Release("a")
	checkInvariant(t, c, "after double release")
	if c.Free() != 64 {
		t.Fatalf("double release inflated free to %d", c.Free())
	}

	if c.Grow("gone") {
		t.Fatal("grew a sequence that holds nothing")
	}
	if c.Fragmentation() != 0 {
		t.Fatalf("paged fragmentation = %d, want 0", c.Fragmentation())
	}
}

func TestPagedCacheFailsCleanlyAtCapacity(t *testing.T) {
	c := NewPagedCache(testPolicy(16, 10))
	if !c.Allocate("a", 7) {
		t.Fatal("allocate a failed")
	}

	// 7 held, 3 free, asking for 4: the interesting case, because a buggy allocator
	// takes the 3 it can and leaves a sequence that cannot run holding memory.
	if c.Allocate("b", 4) {
		t.Fatal("allocate succeeded with insufficient free blocks")
	}
	if c.Blocks("b") != 0 {
		t.Fatalf("failed allocation left %d blocks held", c.Blocks("b"))
	}
	if c.Free() != 3 || c.Used() != 7 {
		t.Fatalf("failed allocation disturbed accounting: used=%d free=%d", c.Used(), c.Free())
	}
	checkInvariant(t, c, "after failed allocate")

	if !c.Allocate("b", 3) {
		t.Fatal("exact-fit allocation failed")
	}
	if c.Free() != 0 {
		t.Fatalf("free = %d after exact fill, want 0", c.Free())
	}
	if c.Grow("a") {
		t.Fatal("grew a sequence with a full cache")
	}
	if c.Allocate("a", 1) {
		t.Fatal("re-allocating an existing sequence succeeded; Release would free the wrong amount")
	}
	checkInvariant(t, c, "full")
}

func TestPagedCacheBlocksReusableAfterRelease(t *testing.T) {
	c := NewPagedCache(testPolicy(16, 8))
	if !c.Allocate("a", 8) {
		t.Fatal("allocate a failed")
	}
	if c.Allocate("b", 1) {
		t.Fatal("allocate b succeeded on a full cache")
	}
	c.Release("a")
	// The point of paging: a's blocks are usable by anyone, in any shape, not only
	// by a sequence that happens to fit the hole a left.
	for i := 0; i < 8; i++ {
		id := RequestID(fmt.Sprintf("r-%d", i))
		if !c.Allocate(id, 1) {
			t.Fatalf("reuse allocation %d failed; free=%d", i, c.Free())
		}
	}
	if c.Used() != 8 || c.Sequences() != 8 {
		t.Fatalf("after reuse: used=%d sequences=%d, want 8/8", c.Used(), c.Sequences())
	}
	checkInvariant(t, c, "after reuse")
}

func TestPagedCacheInvariantUnderRandomizedOps(t *testing.T) {
	const total = 97 // deliberately not a round number or a multiple of any request size
	c := NewPagedCache(testPolicy(16, total))
	rng := rand.New(rand.NewSource(20240917))

	held := make(map[RequestID]int)
	for step := 0; step < 20000; step++ {
		id := RequestID(fmt.Sprintf("r-%d", rng.Intn(24)))
		switch rng.Intn(3) {
		case 0:
			n := 1 + rng.Intn(12)
			if c.Allocate(id, n) {
				if _, dup := held[id]; dup {
					t.Fatalf("step %d: allocated over a live sequence", step)
				}
				held[id] = n
			} else if _, live := held[id]; !live && c.Free() >= n {
				t.Fatalf("step %d: allocate of %d refused with %d free", step, n, c.Free())
			}
		case 1:
			if c.Grow(id) {
				held[id]++
			} else if _, live := held[id]; live && c.Free() > 0 {
				t.Fatalf("step %d: grow refused with %d free", step, c.Free())
			}
		default:
			c.Release(id)
			delete(held, id)
		}

		checkInvariant(t, c, fmt.Sprintf("step %d", step))
		sum := 0
		for k, n := range held {
			if got := c.Blocks(k); got != n {
				t.Fatalf("step %d: %s holds %d, bookkeeping says %d", step, k, got, n)
			}
			sum += n
		}
		if sum != c.Used() {
			t.Fatalf("step %d: per-sequence sum %d != Used %d", step, sum, c.Used())
		}
		if c.Fragmentation() != 0 {
			t.Fatalf("step %d: paged fragmentation became %d", step, c.Fragmentation())
		}
	}

	for id := range held {
		c.Release(id)
	}
	if c.Used() != 0 || c.Free() != total {
		t.Fatalf("after draining: used=%d free=%d, want 0/%d", c.Used(), c.Free(), total)
	}
}

func TestContiguousCacheFragmentationBlocksAViableAllocation(t *testing.T) {
	// Twenty blocks, five four-block sequences, then free every other one. Eight
	// blocks are free and the cache still cannot satisfy a six-block request,
	// because free capacity under this policy is position-dependent.
	c := NewContiguousCache(testPolicy(16, 20), BestFit)
	ids := []RequestID{"a", "b", "c", "d", "e"}
	for _, id := range ids {
		if !c.Allocate(id, 4) {
			t.Fatalf("allocate %s failed; free=%d", id, c.Free())
		}
	}
	if c.Free() != 0 || c.Fragmentation() != 0 {
		t.Fatalf("full cache: free=%d frag=%d, want 0/0", c.Free(), c.Fragmentation())
	}

	c.Release("b")
	c.Release("d")

	if c.Free() != 8 {
		t.Fatalf("free = %d after releasing two 4-block runs, want 8", c.Free())
	}
	if got := c.LargestFreeRun(); got != 4 {
		t.Fatalf("largest free run = %d, want 4", got)
	}
	if got := c.Fragmentation(); got != 4 {
		t.Fatalf("fragmentation = %d, want 4 (8 free, 4 reachable)", got)
	}
	t.Logf("contiguous: %d free blocks, largest run %d, fragmentation %d",
		c.Free(), c.LargestFreeRun(), c.Fragmentation())

	if c.Allocate("f", 6) {
		t.Fatal("6-block allocation succeeded with no 6-block run")
	}
	if c.Blocks("f") != 0 {
		t.Fatalf("failed contiguous allocation left %d blocks held", c.Blocks("f"))
	}
	checkInvariant(t, c, "fragmented")

	// Same free capacity, no placement constraint, and it fits. The 4 blocks the
	// contiguous policy owns and cannot spend are the cost of the constraint.
	paged := NewPagedCache(testPolicy(16, 20))
	for _, id := range []RequestID{"a", "c", "e"} {
		if !paged.Allocate(id, 4) {
			t.Fatalf("paged setup allocate %s failed", id)
		}
	}
	if !paged.Allocate("f", 6) {
		t.Fatal("paged cache with 8 free blocks refused a 6-block allocation")
	}
}

func TestContiguousCacheGrowsIntoAdjacentFreeSpace(t *testing.T) {
	// A fair contiguous allocator can extend a run upward when the next block is
	// still free. The reason the policy must reserve its worst case anyway is that
	// it cannot rely on that: the same grow fails the moment a neighbour exists.
	c := NewContiguousCache(testPolicy(16, 10), FirstFit)
	if !c.AllocateWithInitial("a", 2, 2) {
		t.Fatal("allocate a failed")
	}
	if !c.Grow("a") {
		t.Fatal("grow into free adjacent space failed")
	}
	if c.Blocks("a") != 3 || c.Used() != 3 {
		t.Fatalf("after grow: blocks=%d used=%d, want 3/3", c.Blocks("a"), c.Used())
	}
	checkInvariant(t, c, "after adjacent grow")

	if !c.Allocate("b", 7) {
		t.Fatal("allocate b failed")
	}
	if c.Grow("a") {
		t.Fatal("grew a run into a block held by another sequence")
	}
	if c.Blocks("b") != 7 {
		t.Fatalf("b holds %d blocks, want 7", c.Blocks("b"))
	}
	checkInvariant(t, c, "boxed in")

	// Headroom inside an existing worst-case reservation is always growable, which
	// is the only growth this policy can promise.
	c2 := NewContiguousCache(testPolicy(16, 10), BestFit)
	if !c2.AllocateWithInitial("x", 6, 1) {
		t.Fatal("reserve-peak allocate failed")
	}
	if !c2.Allocate("y", 4) {
		t.Fatal("allocate y failed")
	}
	for i := 0; i < 5; i++ {
		if !c2.Grow("x") {
			t.Fatalf("grow %d inside the reservation failed", i)
		}
	}
	if c2.Grow("x") {
		t.Fatal("grew past the reservation with no adjacent free block")
	}
	if c2.Used() != 10 || c2.Free() != 0 {
		t.Fatalf("used=%d free=%d, want 10/0", c2.Used(), c2.Free())
	}
}

func TestContiguousCacheInvariantUnderRandomizedOps(t *testing.T) {
	const total = 64
	c := NewContiguousCache(testPolicy(16, total), BestFit)
	rng := rand.New(rand.NewSource(991))
	live := make(map[RequestID]bool)

	for step := 0; step < 8000; step++ {
		id := RequestID(fmt.Sprintf("r-%d", rng.Intn(16)))
		switch rng.Intn(3) {
		case 0:
			if c.Allocate(id, 1+rng.Intn(9)) {
				live[id] = true
			}
		case 1:
			c.Grow(id)
		default:
			c.Release(id)
			delete(live, id)
		}
		checkInvariant(t, c, fmt.Sprintf("step %d", step))
		if frag := c.Fragmentation(); frag < 0 || frag > c.Free() {
			t.Fatalf("step %d: fragmentation %d out of range (free %d)", step, frag, c.Free())
		}
		if run := c.LargestFreeRun(); run > c.Free() {
			t.Fatalf("step %d: largest run %d exceeds free %d", step, run, c.Free())
		}
	}
}

// generation is one unit of the comparison workload: what the client declared and
// what it actually produced. The gap between the two is the entire subject.
type generation struct {
	promptTokens int
	maxTokens    int
	actualTokens int
}

// comparisonWorkload is a realistic mix: most generations stop early, a minority
// run long, and every one of them declares the same large ceiling because clients
// declare a ceiling they might need rather than the length they will use.
func comparisonWorkload(n int) []generation {
	const (
		prompt    = 48
		declared  = 512
		shortGen  = 32
		longGen   = 480
		longEvery = 5
	)
	out := make([]generation, 0, n)
	for i := 0; i < n; i++ {
		actual := shortGen
		if i%longEvery == longEvery-1 {
			actual = longGen
		}
		out = append(out, generation{promptTokens: prompt, maxTokens: declared, actualTokens: actual})
	}
	return out
}

// sustainPaged admits sequences until the cache cannot take another, growing each
// one to the length it actually reaches. The reservation tracks reality.
func sustainPaged(policy CachePolicy, work []generation) (concurrent int, c *PagedCache) {
	c = NewPagedCache(policy)
	for i, g := range work {
		id := RequestID(fmt.Sprintf("p-%d", i))
		if !c.Allocate(id, policy.BlocksFor(g.promptTokens)) {
			return i, c
		}
		target := policy.BlocksFor(g.promptTokens + g.actualTokens)
		for c.Blocks(id) < target {
			if !c.Grow(id) {
				// Growth failure mid-decode means this sequence cannot be sustained; it
				// would be preempted, so it does not count toward sustained concurrency.
				c.Release(id)
				return i, c
			}
		}
	}
	return len(work), c
}

// sustainContiguous admits the same sequences, reserving each one's declared
// worst case because the policy has no way to grow into space that may be taken.
func sustainContiguous(policy CachePolicy, work []generation, fit FitPolicy) (concurrent int, c *ContiguousCache) {
	c = NewContiguousCache(policy, fit)
	for i, g := range work {
		id := RequestID(fmt.Sprintf("c-%d", i))
		peak := policy.BlocksFor(g.promptTokens + g.maxTokens)
		initial := policy.BlocksFor(g.promptTokens)
		if !c.AllocateWithInitial(id, peak, initial) {
			return i, c
		}
		target := policy.BlocksFor(g.promptTokens + g.actualTokens)
		for c.Blocks(id) < target {
			if !c.Grow(id) {
				c.Release(id)
				return i, c
			}
		}
	}
	return len(work), c
}

func TestPagedSustainsStrictlyMoreConcurrencyThanContiguous(t *testing.T) {
	policy := testPolicy(16, 256)
	work := comparisonWorkload(64)

	paged, pc := sustainPaged(policy, work)
	best, bc := sustainContiguous(policy, work, BestFit)
	first, fc := sustainContiguous(policy, work, FirstFit)

	t.Logf("cache: %d blocks of %d tokens (%d tokens of KV capacity)",
		policy.TotalBlocks, policy.TokensPerBlock, policy.TotalBlocks*policy.TokensPerBlock)
	t.Logf("workload: 48-token prompt, 512-token declared ceiling, 4-in-5 stop at 32 tokens, 1-in-5 runs to 480")
	t.Logf("paged:            %2d concurrent sequences, used %3d, free %3d, fragmentation %d",
		paged, pc.Used(), pc.Free(), pc.Fragmentation())
	t.Logf("contiguous/best:  %2d concurrent sequences, used %3d, free %3d, fragmentation %d (largest run %d)",
		best, bc.Used(), bc.Free(), bc.Fragmentation(), bc.LargestFreeRun())
	t.Logf("contiguous/first: %2d concurrent sequences, used %3d, free %3d, fragmentation %d (largest run %d)",
		first, fc.Used(), fc.Free(), fc.Fragmentation(), fc.LargestFreeRun())
	t.Logf("paged advantage: %.2fx over best-fit contiguous", float64(paged)/float64(best))

	if paged <= best {
		t.Fatalf("paged sustained %d concurrent, best-fit contiguous %d; paged must be strictly higher", paged, best)
	}
	if paged <= first {
		t.Fatalf("paged sustained %d concurrent, first-fit contiguous %d; paged must be strictly higher", paged, first)
	}
	checkInvariant(t, pc, "paged final")
	checkInvariant(t, bc, "contiguous final")

	// The paged cache stopped because the next sequence genuinely did not fit, not
	// because the harness gave up: the comparison is between two exhausted caches.
	next := work[paged]
	if need := policy.BlocksFor(next.promptTokens + next.actualTokens); pc.Free() >= need {
		t.Fatalf("paged stopped with %d free but the next sequence needed only %d", pc.Free(), need)
	}

	// Churn is where the second cost appears. Filling a cache from empty produces no
	// holes; a running system is constantly releasing sequences in the middle of the
	// address space, and only one of these two policies has to care.
	for i := 0; i < best; i += 2 {
		bc.Release(RequestID(fmt.Sprintf("c-%d", i)))
	}
	for i := 0; i < paged; i += 2 {
		pc.Release(RequestID(fmt.Sprintf("p-%d", i)))
	}
	t.Logf("after releasing every other sequence: contiguous free %d, largest run %d, fragmentation %d; paged free %d, fragmentation %d",
		bc.Free(), bc.LargestFreeRun(), bc.Fragmentation(), pc.Free(), pc.Fragmentation())
	if bc.Fragmentation() <= 0 {
		t.Fatalf("contiguous fragmentation %d after churn; expected unusable free capacity", bc.Fragmentation())
	}
}

func TestContiguousWastesCapacityOnOverReservation(t *testing.T) {
	// The same workload, measured as capacity rather than concurrency: how much of
	// each cache's committed memory corresponds to tokens that exist.
	policy := testPolicy(16, 256)
	work := comparisonWorkload(64)

	paged, pc := sustainPaged(policy, work)
	contig, cc := sustainContiguous(policy, work, BestFit)

	liveTokens := func(n int) int {
		total := 0
		for _, g := range work[:n] {
			total += g.promptTokens + g.actualTokens
		}
		return total
	}
	pagedUseful := float64(liveTokens(paged)) / float64(pc.Used()*policy.TokensPerBlock)
	contigUseful := float64(liveTokens(contig)) / float64(cc.Used()*policy.TokensPerBlock)
	t.Logf("useful fraction of committed cache: paged %.2f, contiguous %.2f", pagedUseful, contigUseful)

	if pagedUseful <= contigUseful {
		t.Fatalf("paged useful fraction %.3f did not beat contiguous %.3f", pagedUseful, contigUseful)
	}
}
