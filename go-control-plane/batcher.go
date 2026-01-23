package loom

import (
	"sync"
	"time"
)

var (
	_ Batcher = (*StaticBatcher)(nil)
	_ Batcher = (*ContinuousBatcher)(nil)
)

// VictimPolicy chooses which running sequence is preempted when the cache cannot
// grow one.
type VictimPolicy uint8

const (
	// PreemptNewest evicts the most recently started sequence. This is the default.
	//
	// The reasoning is about total wasted work, not fairness. Preemption frees the
	// victim's KV blocks but keeps its generated tokens, so resuming it costs a
	// prefill over prompt-plus-generated-tokens — a cost that grows with the victim's
	// progress. The cheapest victim is therefore the one with the least progress, and
	// in a decode loop that is almost always the newest arrival. It also keeps a
	// system under sustained pressure from thrashing: the
	// older sequences run to completion and release their blocks, which ends the
	// shortage, whereas evicting the oldest-and-largest repeatedly throws away the
	// most work and leaves the shortage in place.
	//
	// The cost is honest and worth stating: a steadily loaded system can preempt the
	// same unlucky request more than once, so the preemption counter must be watched
	// per request, not only in aggregate.
	PreemptNewest VictimPolicy = iota

	// PreemptLongestRemaining evicts the sequence with the most generation left.
	// It frees the largest future reservation and so is the least likely to need a
	// second preemption, at the price of charging the re-prefill to whichever
	// sequence has the furthest still to go — the one whose remaining decode the
	// extra prefill delays for longest.
	PreemptLongestRemaining
)

func (p VictimPolicy) String() string {
	if p == PreemptLongestRemaining {
		return "longest-remaining"
	}
	return "newest"
}

// batcherDone reports whether a sequence has left the running set. Both
// conditions are needed: the worker reports completion by phase, but a sequence
// that reached its declared ceiling is finished whether or not that report has
// arrived yet, and leaving it in the running set would hold a batch slot for a
// sequence that will never produce another token.
func batcherDone(st *RequestState) bool {
	if st == nil {
		return true
	}
	if st.Phase == PhaseDone {
		return true
	}
	return st.MaxOutputTokens > 0 && st.GeneratedTokens >= st.MaxOutputTokens
}

// batchFit decides whether a request can be taken into the running set, and
// reserves its blocks when it can.
//
// Returns the blocks reserved. Zero means it did not fit, and nothing was
// reserved — a partial reservation here would be a leak, since the caller has no
// handle on a sequence it did not admit.
func batchFit(cache KVCache, est *CostEstimator, mode ReservationMode, w *WorkerState, st *RequestState) int {
	need := 1
	if est != nil {
		// EstimateResumed rather than Estimate: a sequence that was preempted keeps the
		// tokens it generated, so its footprint on re-admission starts at
		// prompt-plus-generated rather than at the prompt. Reserving the original
		// expected footprint for it guarantees a growth failure on its first step back.
		need = est.ReserveBlocks(est.EstimateResumed(st), mode)
		if need < 1 {
			need = 1
		}
	}
	if cache == nil {
		// With no cache handle the worker's reported free-block count is the only
		// capacity signal available. It is the control plane's estimate rather than the
		// truth, which is exactly why a batcher that owns a cache is preferred.
		if w != nil && w.FreeBlocks() < need {
			return 0
		}
		return need
	}
	if !cache.Allocate(st.ID, need) {
		return 0
	}
	return need
}

// StaticBatcher forms a batch, dispatches it, and does not touch it again until
// every member has finished.
//
// This is the baseline, and its pathology is the reason continuous batching
// exists: the batch's slots are owned by the batch, so a 30-token request and a
// 2000-token request admitted together means the 30-token one's slot sits idle for
// 1970 decode steps. The accelerator keeps paying batch-of-N cost to compute one
// sequence, and queued work that would have fitted in the freed slot waits for the
// longest member instead of for capacity.
//
// It is implemented faithfully rather than crippled: it fills to the batch-size
// cap, it honours the wait window, and it respects cache capacity. The pathology
// is structural, not a missing feature.
type StaticBatcher struct {
	mu sync.Mutex

	maxWait time.Duration
	est     *CostEstimator
	mode    ReservationMode
	cache   KVCache

	// pending is the batch being accumulated during the wait window; dispatched is
	// the batch currently held hostage.
	pending     []*RequestState
	dispatched  []*RequestState
	windowStart time.Time
	windowOpen  bool

	batches uint64
}

// NewStaticBatcher builds the baseline batcher. maxWait is the accumulation
// window and is the latency/throughput dial: zero dispatches as soon as anything
// is runnable, a longer window trades first-token latency for fuller batches.
func NewStaticBatcher(maxWait time.Duration, est *CostEstimator, mode ReservationMode, cache KVCache) *StaticBatcher {
	return &StaticBatcher{maxWait: maxWait, est: est, mode: mode, cache: cache}
}

func (b *StaticBatcher) Name() string { return "static" }

func (b *StaticBatcher) MaxWait() time.Duration { return b.maxWait }

// Batches counts dispatches, which is how the hostage effect becomes visible: the
// same offered load produces far fewer, longer-lived batches here than under
// continuous batching.
func (b *StaticBatcher) Batches() uint64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.batches
}

// FormBatch returns the sequences for the next forward pass.
//
// While a dispatched batch has any member still running it is returned unchanged
// and the queue is not consulted at all. That refusal to look at the queue is the
// baseline's defining behaviour.
// Step advances a static batch, which means doing nothing to it.
//
// A static batch is held until every member finishes, so there is no eviction and
// no admission mid-flight, and therefore no cache growth this batcher could fail
// to satisfy. The emptiness is the pathology, not an omission: the slot freed by a
// finished short sequence stays idle until the longest member is done.
func (b *StaticBatcher) Step(running []*RequestState, q Queue, w *WorkerState, now time.Time) (survivors, preempted, retired []*RequestState) {
	return running, nil, nil
}

func (b *StaticBatcher) FormBatch(running []*RequestState, q Queue, w *WorkerState, now time.Time) (batch, retired []*RequestState) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if len(b.dispatched) > 0 {
		live := b.dispatched[:0:0]
		for _, st := range b.dispatched {
			if batcherDone(st) {
				if b.cache != nil {
					b.cache.Release(st.ID)
				}
				st.Phase = PhaseDone
				retired = append(retired, st)
				continue
			}
			live = append(live, st)
		}
		if len(live) > 0 {
			b.dispatched = live
			out := make([]*RequestState, len(live))
			copy(out, live)
			return out, retired
		}
		b.dispatched = nil
	}

	limit := 0
	if w != nil {
		limit = w.MaxBatchSize
	}
	if limit <= 0 {
		limit = 1
	}

	if q != nil {
		for len(b.pending) < limit {
			st := q.Pop()
			if st == nil {
				break
			}
			if batchFit(b.cache, b.est, b.mode, w, st) == 0 {
				// It did not fit, and pushing it back is the only correct move: holding it
				// inside the batcher would make it invisible to deadline expiry and
				// cancellation, both of which only see the queue.
				if !q.Push(st) {
					st.Phase = PhaseDone
				}
				break
			}
			if !b.windowOpen {
				b.windowStart, b.windowOpen = now, true
			}
			st.Phase = PhasePrefill
			b.pending = append(b.pending, st)
		}
	}

	if len(b.pending) == 0 {
		return nil, retired
	}
	// Dispatch when the batch is full or the window has closed. Waiting past a full
	// batch would add latency for no throughput, and waiting past the window would
	// make the dial meaningless.
	if len(b.pending) < limit && b.windowOpen && now.Sub(b.windowStart) < b.maxWait {
		return nil, retired
	}

	b.dispatched = b.pending
	b.pending = nil
	b.windowOpen = false
	b.batches++
	out := make([]*RequestState, len(b.dispatched))
	copy(out, b.dispatched)
	return out, retired
}

// ContinuousBatcher re-forms the running set on every step.
//
// Finished sequences are evicted and their blocks released, then queued requests
// are admitted into whatever that freed, bounded by both the batch-size cap and
// real cache capacity. The property this buys is the one static batching cannot
// have: a request that arrives while a 2000-token generation is running starts and
// finishes during it, instead of after it.
type ContinuousBatcher struct {
	mu sync.Mutex

	maxWait time.Duration
	est     *CostEstimator
	mode    ReservationMode
	cache   KVCache
	victims VictimPolicy

	windowStart time.Time
	windowOpen  bool

	preemptions uint64
	steps       uint64
	admitted    uint64
}

// NewContinuousBatcher builds the continuous batcher. The cache is needed rather
// than optional for Step: growth failure is the event preemption exists to handle,
// and only the cache can report it.
func NewContinuousBatcher(maxWait time.Duration, est *CostEstimator, mode ReservationMode, cache KVCache, victims VictimPolicy) *ContinuousBatcher {
	return &ContinuousBatcher{maxWait: maxWait, est: est, mode: mode, cache: cache, victims: victims}
}

func (b *ContinuousBatcher) Name() string { return "continuous" }

func (b *ContinuousBatcher) MaxWait() time.Duration { return b.maxWait }

// Preemptions is the count of sequences evicted to free cache. It is exported
// because a system that preempts constantly looks healthy from the outside while
// doing a large fraction of its work twice; unreported preemption is the failure
// mode that hides every other one.
func (b *ContinuousBatcher) Preemptions() uint64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.preemptions
}

// Admitted counts sequences taken into the running set mid-flight.
func (b *ContinuousBatcher) Admitted() uint64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.admitted
}

// FormBatch evicts finished sequences and admits queued ones into the freed
// capacity.
//
// retired is the members of running that had already finished and whose cache has
// now been released. They are reported rather than silently dropped because the
// caller, not the batcher, owns the reservation ledger: a driver that diffs only
// the returned batch against what it believed was running cannot tell a retirement
// from a request it should still be tracking.
func (b *ContinuousBatcher) FormBatch(running []*RequestState, q Queue, w *WorkerState, now time.Time) (batch, retired []*RequestState) {
	b.mu.Lock()
	defer b.mu.Unlock()

	batch = make([]*RequestState, 0, len(running)+1)
	for _, st := range running {
		if batcherDone(st) {
			if b.cache != nil {
				b.cache.Release(st.ID)
			}
			st.Phase = PhaseDone
			retired = append(retired, st)
			continue
		}
		batch = append(batch, st)
	}

	limit := 0
	if w != nil {
		limit = w.MaxBatchSize
	}
	if limit <= 0 {
		limit = 1
	}

	admittedNow := 0
	if q != nil {
		for len(batch) < limit {
			st := q.Pop()
			if st == nil {
				break
			}
			if batchFit(b.cache, b.est, b.mode, w, st) == 0 {
				if !q.Push(st) {
					st.Phase = PhaseDone
				}
				break
			}
			if st.StartedAt.IsZero() {
				st.StartedAt = now
			}
			st.Phase = PhasePrefill
			batch = append(batch, st)
			admittedNow++
		}
	}

	// The wait window only applies when there is nothing already running. Once the
	// accelerator is busy, delaying an admission buys no batching benefit — the next
	// step is going to happen regardless — and costs the new request pure latency.
	if len(running) == 0 && admittedNow > 0 && len(batch) < limit && b.maxWait > 0 {
		if !b.windowOpen {
			b.windowStart, b.windowOpen = now, true
		}
		if now.Sub(b.windowStart) < b.maxWait {
			// Put them back rather than holding them inside the batcher. A request parked
			// in the batcher is invisible to deadline expiry and cancellation, which both
			// only see the queue, and a reservation held for a sequence that is not running
			// is capacity another sequence could have used during the window.
			for _, st := range batch {
				st.Phase = PhaseQueued
				if b.cache != nil {
					b.cache.Release(st.ID)
				}
				if q != nil {
					q.Push(st)
				}
			}
			return nil, retired
		}
	}
	b.windowOpen = false
	b.admitted += uint64(admittedNow)
	if len(batch) == 0 {
		return nil, retired
	}
	return batch, retired
}

// Step advances every running sequence by one token, growing the cache where a
// block boundary is crossed and preempting when it cannot.
//
// Preempted sequences have their blocks released, their phase reset to queued, and
// are pushed back onto the queue so they are re-run rather than lost — a
// preemption that silently drops the request turns a memory shortage into a
// failed request, which is a much worse outcome than a slow one. The tokens the
// victim already generated stay on the state: the blocks are what was scarce, and
// re-prefilling prompt-plus-generated-tokens on reschedule is cheaper than
// generating those tokens a second time.
//
// Postcondition: every sequence in running is returned in exactly one of
// survivors, preempted and retired. The three are disjoint and together account
// for the input. retired is the members that had already finished, whose cache has
// been released. A caller that un-bills a reservation per returned entry depends on
// this, and so does correctness: a sequence reported as both preempted and
// surviving is simultaneously in this worker's running set and in the shared
// queue, so the next Pop dispatches a second copy of a request that is still
// generating.
func (b *ContinuousBatcher) Step(running []*RequestState, q Queue, w *WorkerState, now time.Time) (survivors, preempted, retired []*RequestState) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.steps++
	live := make([]*RequestState, 0, len(running))
	for _, st := range running {
		if batcherDone(st) {
			if b.cache != nil {
				b.cache.Release(st.ID)
			}
			st.Phase = PhaseDone
			retired = append(retired, st)
			continue
		}
		live = append(live, st)
	}

	// Sequences preempted earlier in this very step. They hold no blocks and are
	// already back on the queue, so the decode loop must not touch them again:
	// advancing one would put it in survivors while it is also queued, and
	// preempting it again would report it twice. Allocated on the first preemption,
	// since the overwhelming majority of steps have none.
	var gone map[RequestID]bool
	markGone := func(st *RequestState) {
		if gone == nil {
			gone = make(map[RequestID]bool, 2)
		}
		gone[st.ID] = true
	}

	tokensPerBlock := 16
	if b.est != nil && b.est.Policy.TokensPerBlock > 0 {
		tokensPerBlock = b.est.Policy.TokensPerBlock
	}

	for _, st := range live {
		if gone[st.ID] {
			continue
		}
		st.Phase = PhaseDecode
		st.GeneratedTokens++
		if st.FirstToken.IsZero() {
			st.FirstToken = now
		}
		if b.cache == nil {
			continue
		}
		total := st.PromptTokens + st.GeneratedTokens
		want := (total + tokensPerBlock - 1) / tokensPerBlock
		have := heldBlocks(b.cache, st)
		// Growth is driven by the gap between what the sequence now spans and what it
		// actually holds, not by the arithmetic of block boundaries. A sequence admitted
		// on its expected footprint already holds blocks past its current position, and
		// growing it at every boundary regardless would over-reserve exactly the capacity
		// paging exists to recover.
		if want <= have {
			st.HeldBlocks = have
			continue
		}
		if b.cache.Grow(st.ID) {
			st.HeldBlocks = have + 1
			continue
		}
		// Cache exhausted mid-decode. Something must leave; the only question is who,
		// and the victim may be this very sequence.
		victim := b.pickVictimLocked(live, gone, st)
		if victim == nil {
			// No eviction can help, so the step itself fails for this sequence rather than
			// silently continuing with memory it does not hold.
			preempted = append(preempted, b.preemptLocked(st, q, now))
			markGone(st)
			continue
		}
		preempted = append(preempted, b.preemptLocked(victim, q, now))
		markGone(victim)
		if victim == st {
			continue
		}
		// With the victim's blocks returned, the grow that failed is retried once. One
		// retry rather than a loop: if a whole sequence's worth of blocks is still not
		// enough, the shortage is structural and evicting more of the running set would
		// just thrash.
		if b.cache.Grow(st.ID) {
			st.HeldBlocks = have + 1
		} else {
			preempted = append(preempted, b.preemptLocked(st, q, now))
			markGone(st)
		}
	}

	for _, st := range live {
		if gone[st.ID] {
			continue
		}
		survivors = append(survivors, st)
	}
	return survivors, preempted, retired
}

// pickVictimLocked selects the sequence to evict. grower is the sequence that
// needs the memory; it is eligible, because refusing to consider it would evict a
// sequence with more progress to serve one with less. gone names the sequences
// already preempted during this step, which hold nothing and are already queued.
func (b *ContinuousBatcher) pickVictimLocked(live []*RequestState, gone map[RequestID]bool, grower *RequestState) *RequestState {
	var best *RequestState
	for _, st := range live {
		if gone[st.ID] || st.Phase == PhaseQueued || st.Phase == PhaseDone {
			continue
		}
		// Evicting a sequence that holds nothing frees nothing, so it is not a victim.
		if heldBlocks(b.cache, st) == 0 {
			continue
		}
		if best == nil {
			best = st
			continue
		}
		if b.victims == PreemptLongestRemaining {
			if remainingTokens(st) > remainingTokens(best) {
				best = st
			}
			continue
		}
		if st.StartedAt.After(best.StartedAt) {
			best = st
		}
	}
	if best == nil {
		return grower
	}
	return best
}

// blockCounter is the optional per-sequence accounting both cache
// implementations provide. The KVCache contract only exposes aggregate counts,
// which is right for the interface — a scheduler reasons about capacity, not
// placement — but preemption has to know whether evicting a given sequence would
// actually free anything.
type blockCounter interface {
	Blocks(id RequestID) int
}

func heldBlocks(cache KVCache, st *RequestState) int {
	if c, ok := cache.(blockCounter); ok {
		return c.Blocks(st.ID)
	}
	return st.HeldBlocks
}

func remainingTokens(st *RequestState) int {
	r := st.MaxOutputTokens - st.GeneratedTokens
	if r < 0 {
		return 0
	}
	return r
}

// preemptLocked evicts one sequence: its blocks go back to the cache and it goes
// back to the queue.
//
// GeneratedTokens is deliberately left alone. Freeing the blocks is what resolves
// the shortage; the tokens cost nothing to keep, and keeping them means the
// reschedule re-prefills prompt-plus-generated-tokens instead of generating those
// tokens again. That is the recomputation strategy, and it is why the victim's
// re-admission needs a larger reservation than its first admission did — see
// CostEstimator.EstimateResumed.
func (b *ContinuousBatcher) preemptLocked(st *RequestState, q Queue, now time.Time) *RequestState {
	if b.cache != nil {
		b.cache.Release(st.ID)
	}
	st.HeldBlocks = 0
	st.Phase = PhaseQueued
	// A fresh wait starts here. The time this request spent running is not queue
	// wait, and leaving QueuedAt at its original value hands it an aging credit it
	// did not earn and makes it dominate the head-of-line wait the autoscaler reads.
	st.QueuedAt = now
	b.preemptions++
	if q != nil && !q.Push(st) {
		// The queue is full, so the request cannot be retried and must be failed
		// visibly. It is still counted as a preemption: the memory shortage happened
		// whether or not the retry had somewhere to go.
		st.Phase = PhaseDone
	}
	return st
}
