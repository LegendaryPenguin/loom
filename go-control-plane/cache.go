package loom

import (
	"sync"
)

// Both caches are checked against the contract at compile time, so a change to
// KVCache fails the build here rather than at the first call site that cares.
var (
	_ KVCache = (*PagedCache)(nil)
	_ KVCache = (*ContiguousCache)(nil)
)

// PagedCache is the block-paged KV cache: a sequence holds a set of block
// indexes that need not be adjacent.
//
// Everything interesting about this type follows from one property: because a
// sequence never needs its blocks to be next to each other, any free block is
// usable by any sequence. That is why Fragmentation is identically zero, why a
// sequence can start with only the blocks its prompt needs and take more as it
// decodes, and why the blocks a finished sequence returns are immediately
// available to every other sequence rather than only to one that happens to fit
// the hole. The reservation tracks the sequence's actual length instead of its
// declared worst case, and the declared worst case is usually wrong by an order
// of magnitude.
//
// It is also what makes preemption a recomputation rather than a loss: the victim's
// blocks go back to the general pool while its generated tokens stay on the control
// plane's state, so resuming it costs one prefill over prompt-plus-generated-tokens
// rather than generating those tokens a second time.
type PagedCache struct {
	mu     sync.Mutex
	policy CachePolicy

	// free is a stack of unallocated block indexes. A stack rather than a queue
	// deliberately: reusing the most recently freed block keeps the working set of
	// block metadata small, and since blocks are interchangeable there is no
	// placement decision to get right.
	free []int

	held map[RequestID][]int

	total int
	used  int
}

// NewPagedCache builds a paged cache over policy.TotalBlocks blocks.
func NewPagedCache(policy CachePolicy) *PagedCache {
	total := policy.TotalBlocks
	if total < 0 {
		total = 0
	}
	c := &PagedCache{
		policy: policy,
		free:   make([]int, 0, total),
		held:   make(map[RequestID][]int, 16),
		total:  total,
	}
	// Descending so that the first allocations hand out low block indexes, which
	// makes test failures and traces easier to read. It has no effect on behaviour.
	for i := total - 1; i >= 0; i-- {
		c.free = append(c.free, i)
	}
	return c
}

// Allocate takes blocks for id, all or nothing.
//
// A sequence already holding blocks is refused rather than topped up: silently
// merging two allocations under one id would make Release free an amount the
// caller never asked for, and at-least-once dispatch means duplicate allocate
// calls for the same request do happen.
func (c *PagedCache) Allocate(id RequestID, blocks int) bool {
	if blocks <= 0 {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, exists := c.held[id]; exists {
		return false
	}
	if len(c.free) < blocks {
		return false
	}
	taken := make([]int, blocks)
	copy(taken, c.free[len(c.free)-blocks:])
	c.free = c.free[:len(c.free)-blocks]
	c.held[id] = taken
	c.used += blocks
	return true
}

// Grow takes one more block for a running sequence, which is what a decode step
// needs when it crosses a block boundary.
func (c *PagedCache) Grow(id RequestID) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	cur, exists := c.held[id]
	if !exists {
		return false
	}
	if len(c.free) == 0 {
		return false
	}
	b := c.free[len(c.free)-1]
	c.free = c.free[:len(c.free)-1]
	c.held[id] = append(cur, b)
	c.used++
	return true
}

// Release returns every block held by id. Releasing an unknown id is a no-op
// rather than an error: preemption and worker loss both race with natural
// completion, and double-release must not corrupt the accounting.
func (c *PagedCache) Release(id RequestID) {
	c.mu.Lock()
	defer c.mu.Unlock()
	blocks, exists := c.held[id]
	if !exists {
		return
	}
	delete(c.held, id)
	c.free = append(c.free, blocks...)
	c.used -= len(blocks)
}

func (c *PagedCache) Used() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.used
}

func (c *PagedCache) Free() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.free)
}

func (c *PagedCache) Total() int { return c.total }

// Fragmentation is zero, and the reason it can be is the entire argument for
// paging: there is no placement constraint, so no free block is ever unusable.
func (c *PagedCache) Fragmentation() int { return 0 }

// Blocks reports how many blocks id currently holds, for assertions and for the
// control plane's reconciliation against a worker report.
func (c *PagedCache) Blocks(id RequestID) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.held[id])
}

// Sequences is the number of sequences currently holding blocks, which is the
// concurrency the cache is actually sustaining.
func (c *PagedCache) Sequences() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.held)
}

// FitPolicy selects which free run a contiguous allocation takes.
type FitPolicy uint8

const (
	// BestFit takes the smallest free run that fits. This is the default, and the
	// reason is specific to this workload rather than general folklore: every
	// reservation here is a worst-case reservation and therefore large, so the
	// scarce resource is large runs. Best-fit spends small holes first and leaves
	// the large runs intact for the requests that cannot use anything else.
	BestFit FitPolicy = iota

	// FirstFit takes the lowest-addressed run that fits. Cheaper per allocation and
	// it clusters allocations at the low end, which keeps one large run at the top
	// for longer; it pays for that by chopping up runs it passes over.
	FirstFit
)

func (f FitPolicy) String() string {
	if f == FirstFit {
		return "first-fit"
	}
	return "best-fit"
}

type contiguousExtent struct {
	start int

	// reserved is the run length the sequence took at admission, which for this
	// policy is its worst case. It is unavailable to anyone else for the
	// sequence's whole lifetime whether or not it is ever written.
	reserved int

	// used is how much of the reservation the sequence has actually reached. It
	// exists only so Grow can tell "there is headroom inside the reservation"
	// from "this sequence needs memory it never reserved".
	used int
}

// ContiguousCache is the single-run KV cache that the paged one is measured
// against.
//
// Two consequences of requiring one unbroken run, and both are real costs rather
// than artifacts of a bad implementation. First, a sequence cannot grow into
// space another sequence may have taken in the meantime, so it has to reserve its
// worst case — max output tokens — at admission, and a request that declares 2048
// and stops at 30 holds the full 2048 the whole time. Second, free capacity
// becomes position-dependent: blocks can be free and still unusable because they
// are not adjacent to each other.
//
// This is implemented honestly, not as a straw man. It uses best-fit by default
// (see FitPolicy for why that is the right choice for worst-case reservations),
// and Grow will extend a run into the immediately following block when that block
// happens to be free, which is the best a contiguous allocator can do. It loses
// the comparison anyway, which is the point.
type ContiguousCache struct {
	mu     sync.Mutex
	policy CachePolicy
	fit    FitPolicy

	occupied []bool
	runs     map[RequestID]contiguousExtent

	total int
	used  int
}

// NewContiguousCache builds a contiguous cache using the given fit policy.
func NewContiguousCache(policy CachePolicy, fit FitPolicy) *ContiguousCache {
	total := policy.TotalBlocks
	if total < 0 {
		total = 0
	}
	return &ContiguousCache{
		policy:   policy,
		fit:      fit,
		occupied: make([]bool, total),
		runs:     make(map[RequestID]contiguousExtent, 16),
		total:    total,
	}
}

// Fit reports the placement policy in use, so a benchmark can state which
// variant produced its numbers.
func (c *ContiguousCache) Fit() FitPolicy { return c.fit }

// Allocate reserves one unbroken run of the requested length. The caller is
// expected to pass the sequence's PEAK block count, because this policy has no
// way to honour a later request for more.
func (c *ContiguousCache) Allocate(id RequestID, blocks int) bool {
	if blocks <= 0 {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.allocateLocked(id, blocks, blocks)
}

func (c *ContiguousCache) allocateLocked(id RequestID, blocks, initial int) bool {
	if _, exists := c.runs[id]; exists {
		return false
	}
	start, ok := c.findRunLocked(blocks)
	if !ok {
		// Nothing is marked before the search succeeds, so a failed allocation
		// leaves no partial hold behind.
		return false
	}
	for i := start; i < start+blocks; i++ {
		c.occupied[i] = true
	}
	c.runs[id] = contiguousExtent{start: start, reserved: blocks, used: initial}
	c.used += blocks
	return true
}

// AllocateWithInitial reserves the worst case but records only initial blocks as
// touched, so a test or simulator can exercise Grow against a real reservation.
// The capacity cost is the reservation either way, which is the whole problem.
func (c *ContiguousCache) AllocateWithInitial(id RequestID, peak, initial int) bool {
	if initial <= 0 || initial > peak {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.allocateLocked(id, peak, initial)
}

// findRunLocked returns the start index of a free run of length blocks.
func (c *ContiguousCache) findRunLocked(blocks int) (int, bool) {
	bestStart, bestLen := -1, 0
	i := 0
	for i < c.total {
		if c.occupied[i] {
			i++
			continue
		}
		start := i
		for i < c.total && !c.occupied[i] {
			i++
		}
		runLen := i - start
		if runLen < blocks {
			continue
		}
		if c.fit == FirstFit {
			return start, true
		}
		if bestStart < 0 || runLen < bestLen {
			bestStart, bestLen = start, runLen
		}
	}
	return bestStart, bestStart >= 0
}

// Grow extends a sequence by one block.
//
// Two chances, in order: consume headroom already inside the reservation, or
// extend the run into the next block if it is still free. The second case is why
// this is not a straw man — a contiguous allocator genuinely can grow upward when
// nobody took the space — and its unreliability is why the policy has to reserve
// the peak in the first place.
func (c *ContiguousCache) Grow(id RequestID) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, exists := c.runs[id]
	if !exists {
		return false
	}
	if e.used < e.reserved {
		e.used++
		c.runs[id] = e
		return true
	}
	next := e.start + e.reserved
	if next >= c.total || c.occupied[next] {
		return false
	}
	c.occupied[next] = true
	e.reserved++
	e.used++
	c.runs[id] = e
	c.used++
	return true
}

func (c *ContiguousCache) Release(id RequestID) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, exists := c.runs[id]
	if !exists {
		return
	}
	delete(c.runs, id)
	for i := e.start; i < e.start+e.reserved; i++ {
		c.occupied[i] = false
	}
	c.used -= e.reserved
}

// Used counts reserved blocks, not touched blocks. Reserved is the honest number
// because a reserved block cannot be given to another sequence, so it is spent
// from the system's point of view no matter how empty it is.
func (c *ContiguousCache) Used() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.used
}

func (c *ContiguousCache) Free() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.total - c.used
}

func (c *ContiguousCache) Total() int { return c.total }

// Fragmentation is free capacity that is not reachable as a single run: the free
// block count minus the largest free run. It is the capacity the policy owns and
// cannot spend, and it is exactly the quantity that is structurally absent from
// PagedCache.
func (c *ContiguousCache) Fragmentation() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	free := c.total - c.used
	if free <= 0 {
		return 0
	}
	return free - c.largestFreeRunLocked()
}

// LargestFreeRun is the biggest allocation this cache could currently satisfy.
func (c *ContiguousCache) LargestFreeRun() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.largestFreeRunLocked()
}

func (c *ContiguousCache) largestFreeRunLocked() int {
	best, run := 0, 0
	for i := 0; i < c.total; i++ {
		if c.occupied[i] {
			run = 0
			continue
		}
		run++
		if run > best {
			best = run
		}
	}
	return best
}

// Blocks reports the blocks reserved for id.
func (c *ContiguousCache) Blocks(id RequestID) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.runs[id].reserved
}

// Sequences is the concurrency the cache is currently sustaining.
func (c *ContiguousCache) Sequences() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.runs)
}
