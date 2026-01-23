package loom

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func batchTestEstimator(policy CachePolicy) *CostEstimator {
	// Fraction 1.0 so the batcher tests reserve exactly the declared ceiling, which
	// keeps the arithmetic in the assertions checkable by hand. The expected-length
	// model is exercised in the admission tests.
	est := NewCostEstimator(policy, FractionOfMax{Fraction: 1, Floor: 1})
	return est
}

func queuedRequest(id RequestID, prompt, max int) *RequestState {
	return &RequestState{
		Request:  *testRequest(id, "t1", prompt, max),
		Phase:    PhaseQueued,
		QueuedAt: testNow,
	}
}

func contains(batch []*RequestState, id RequestID) bool {
	for _, st := range batch {
		if st.ID == id {
			return true
		}
	}
	return false
}

// assertStepAccounting checks the postcondition Step promises: every sequence it
// was handed comes back in exactly one of the three result slices.
//
// Both halves matter and for different reasons. A sequence in two slices is the
// accounting bug — a driver that un-bills a reservation per returned entry
// double-releases — and a sequence reported as preempted while also surviving is
// worse than that: it is simultaneously in a worker's running set and in the
// shared queue, so the next Pop hands out a second copy of a request that is
// still running. A sequence in no slice is the silent retirement, where the
// batcher released the cache and the driver never learned it should stop
// tracking the request.
func assertStepAccounting(t *testing.T, label string, running, survivors, preempted, retired []*RequestState) {
	t.Helper()

	seen := make(map[RequestID]string, len(running))
	for _, group := range []struct {
		name string
		sts  []*RequestState
	}{{"survivors", survivors}, {"preempted", preempted}, {"retired", retired}} {
		for _, st := range group.sts {
			if prev, dup := seen[st.ID]; dup {
				t.Fatalf("%s: %s reported in both %s and %s", label, st.ID, prev, group.name)
			}
			seen[st.ID] = group.name
		}
	}
	for _, st := range running {
		if _, ok := seen[st.ID]; !ok {
			t.Fatalf("%s: %s was handed to Step and reported in none of survivors/preempted/retired", label, st.ID)
		}
	}
	if len(seen) != len(running) {
		t.Fatalf("%s: %d sequences reported for %d inputs", label, len(seen), len(running))
	}
}

// TestStepAccountsForEverySequenceItWasHanded drives the interleaving where a
// sequence is preempted as an earlier sequence's victim and its own turn in the
// decode loop comes afterwards.
//
// The two sequences are sized so the older one's grow fails while the newer one
// still holds blocks, which is what makes the newer one the victim at an
// iteration before its own.
func TestStepAccountsForEverySequenceItWasHanded(t *testing.T) {
	policy := testPolicy(16, 3)
	cache := NewPagedCache(policy)
	// A tiny expected fraction so each sequence is admitted on the one or two blocks
	// its prompt plus first token needs, leaving the cache exactly full.
	est := NewCostEstimator(policy, FractionOfMax{Fraction: 0.0001, Floor: 1})
	b := NewContinuousBatcher(0, est, ReserveExpected, cache, PreemptNewest)
	w := testWorker("w1", "m1", 3, 0, 2, 0)

	q := newACBQueue(8)
	// 31 prompt tokens: two blocks cover the first generated token and the second one
	// crosses into a third block that does not exist.
	older := queuedRequest("older", 31, 160)
	// One prompt token, so its own block boundary is 15 tokens away and it is still
	// holding a block when the older sequence's grow fails.
	newer := queuedRequest("newer", 1, 160)
	q.Push(older)
	q.Push(newer)

	batch, _ := b.FormBatch(nil, q, w, testNow)
	if len(batch) != 2 {
		t.Fatalf("initial batch size %d, want 2", len(batch))
	}
	if cache.Free() != 0 {
		t.Fatalf("cache free = %d, want 0 so the next grow fails", cache.Free())
	}
	older.StartedAt = testNow
	newer.StartedAt = testNow.Add(time.Millisecond)

	var preempted, retired []*RequestState
	for step := 0; step < 8 && len(preempted) == 0; step++ {
		in := append([]*RequestState(nil), batch...)
		batch, preempted, retired = b.Step(batch, q, nil, testNow.Add(time.Duration(step)*time.Millisecond))
		assertStepAccounting(t, fmt.Sprintf("step %d", step), in, batch, preempted, retired)
		checkInvariant(t, cache, "victim-before-own-turn step")
	}

	if len(preempted) != 1 || preempted[0].ID != "newer" {
		t.Fatalf("preempted = %v, want exactly the newer sequence", ids(preempted))
	}
	// The whole point of preempting it: the sequence that needed the block got it.
	if len(batch) != 1 || batch[0].ID != "older" {
		t.Fatalf("survivors = %v, want just older", ids(batch))
	}
	if got := cache.Blocks("older"); got != 3 {
		t.Fatalf("survivor holds %d blocks, want 3 after the retry", got)
	}
	// A victim that is also still advancing would be generating tokens on a worker
	// while the queue hands a second copy of it to another worker.
	if cache.Blocks("newer") != 0 {
		t.Fatalf("victim holds %d blocks after preemption", cache.Blocks("newer"))
	}
	if newer.Phase != PhaseQueued {
		t.Fatalf("victim phase = %v, want queued", newer.Phase)
	}
}

// onDemandCache is a KVCache whose Grow serves a sequence holding nothing, by
// allocating it a block on the spot.
//
// That is legal under the KVCache contract and it is what a worker-side allocator
// that pages a sequence back in looks like. It matters here because it is the
// implementation under which a resurrected victim's grow *succeeds*: the victim is
// then advanced, lands in survivors, and is in a running set and the queue at the
// same time. PagedCache happens to hide that by refusing to grow an unheld id, so a
// test written only against PagedCache would see the double-report and never the
// double-dispatch.
type onDemandCache struct {
	held  map[RequestID]int
	total int
	used  int
}

func newOnDemandCache(total int) *onDemandCache {
	return &onDemandCache{held: make(map[RequestID]int), total: total}
}

func (c *onDemandCache) Allocate(id RequestID, blocks int) bool {
	if blocks <= 0 || c.used+blocks > c.total {
		return false
	}
	if _, dup := c.held[id]; dup {
		return false
	}
	c.held[id] = blocks
	c.used += blocks
	return true
}

func (c *onDemandCache) Grow(id RequestID) bool {
	if c.used >= c.total {
		return false
	}
	c.held[id]++
	c.used++
	return true
}

func (c *onDemandCache) Release(id RequestID) {
	c.used -= c.held[id]
	delete(c.held, id)
}

func (c *onDemandCache) Blocks(id RequestID) int { return c.held[id] }
func (c *onDemandCache) Used() int               { return c.used }
func (c *onDemandCache) Free() int               { return c.total - c.used }
func (c *onDemandCache) Total() int              { return c.total }
func (c *onDemandCache) Fragmentation() int      { return 0 }

// TestStepDoesNotAdvanceASequenceItPreempted is the correctness case: a victim
// that is still advanced after its preemption ends up in the worker's running set
// and in the shared queue at once, so the next Pop starts a second copy of a
// request that is still generating tokens.
func TestStepDoesNotAdvanceASequenceItPreempted(t *testing.T) {
	policy := testPolicy(16, 3)
	cache := newOnDemandCache(3)
	est := NewCostEstimator(policy, FractionOfMax{Fraction: 0.0001, Floor: 1})
	b := NewContinuousBatcher(0, est, ReserveExpected, cache, PreemptNewest)
	w := testWorker("w1", "m1", 3, 0, 2, 0)

	q := newACBQueue(8)
	older := queuedRequest("older", 31, 160)
	newer := queuedRequest("newer", 1, 160)
	q.Push(older)
	q.Push(newer)

	batch, _ := b.FormBatch(nil, q, w, testNow)
	if len(batch) != 2 || cache.Free() != 0 {
		t.Fatalf("setup: batch=%v free=%d, want 2 sequences and a full cache", ids(batch), cache.Free())
	}
	older.StartedAt = testNow
	newer.StartedAt = testNow.Add(time.Millisecond)

	var preempted, retired []*RequestState
	for step := 0; step < 8 && len(preempted) == 0; step++ {
		in := append([]*RequestState(nil), batch...)
		batch, preempted, retired = b.Step(batch, q, nil, testNow.Add(time.Duration(step)*time.Millisecond))
		assertStepAccounting(t, fmt.Sprintf("step %d", step), in, batch, preempted, retired)

		queued := make(map[RequestID]bool, q.Len())
		q.Each(func(st *RequestState) bool { queued[st.ID] = true; return true })
		for _, st := range batch {
			if queued[st.ID] {
				t.Fatalf("step %d: %s survived the step that preempted it, so it is running and queued at once",
					step, st.ID)
			}
		}
	}
	if len(preempted) != 1 || preempted[0].ID != "newer" {
		t.Fatalf("preempted = %v, want exactly the newer sequence", ids(preempted))
	}
	// Its tokens are kept — that is the recomputation design — but it must not have
	// been advanced by the step that evicted it, because nothing computed that token.
	if got := preempted[0].GeneratedTokens; got != 1 {
		t.Fatalf("victim generated tokens = %d, want 1: the step that preempted it must not advance it", got)
	}
}

// TestStepAccountingHoldsUnderCachePressure drives the postcondition over a
// workload that preempts repeatedly, and checks the consequence the postcondition
// exists to prevent: no sequence is ever in a worker's running set and in the
// shared queue at the same time.
func TestStepAccountingHoldsUnderCachePressure(t *testing.T) {
	for _, victims := range []VictimPolicy{PreemptNewest, PreemptLongestRemaining} {
		t.Run(victims.String(), func(t *testing.T) {
			rng := rand.New(rand.NewSource(7))
			policy := testPolicy(16, 10)
			cache := NewPagedCache(policy)
			// Reserve barely more than the prompt, so sequences are admitted cheaply and the
			// shortage shows up as a failed grow mid-decode rather than as a refused admission.
			est := NewCostEstimator(policy, FractionOfMax{Fraction: 0.0001, Floor: 1})
			b := NewContinuousBatcher(0, est, ReserveExpected, cache, victims)
			w := testWorker("w1", "m1", 10, 0, 6, 0)

			q := newACBQueue(32)
			for i := 0; i < 24; i++ {
				st := queuedRequest(RequestID(fmt.Sprintf("r-%02d", i)), 1+rng.Intn(24), 32+rng.Intn(160))
				st.StartedAt = testNow.Add(time.Duration(i) * time.Microsecond)
				q.Push(st)
			}

			var batch []*RequestState
			preemptions := 0
			for step := 0; step < 2000; step++ {
				now := testNow.Add(time.Duration(step) * time.Millisecond)

				formed := append([]*RequestState(nil), batch...)
				var retired []*RequestState
				batch, retired = b.FormBatch(batch, q, w, now)
				assertFormBatchAccounting(t, fmt.Sprintf("form %d", step), formed, batch, retired, q)

				in := append([]*RequestState(nil), batch...)
				survivors, preempted, stepRetired := b.Step(batch, q, nil, now)
				assertStepAccounting(t, fmt.Sprintf("step %d", step), in, survivors, preempted, stepRetired)
				preemptions += len(preempted)
				batch = survivors

				queued := make(map[RequestID]bool, q.Len())
				q.Each(func(st *RequestState) bool { queued[st.ID] = true; return true })
				for _, st := range survivors {
					if queued[st.ID] {
						t.Fatalf("step %d: %s is running and queued at once; the next Pop dispatches a second copy of it",
							step, st.ID)
					}
				}
				checkInvariant(t, cache, "pressure step")
			}
			if preemptions == 0 {
				t.Fatal("no preemption in a workload sized to force them; the postcondition is not being exercised")
			}
			t.Logf("%s: %d preemptions over 2000 steps with the accounting intact", victims, preemptions)
		})
	}
}

// assertFormBatchAccounting checks that FormBatch reports the members of the
// running set it dropped, instead of releasing their cache and leaving the caller
// to find out. A sequence it neither returned nor retired, and which is not waiting
// in the queue, is a leaked reservation.
func assertFormBatchAccounting(t *testing.T, label string, running, batch, retired []*RequestState, q Queue) {
	t.Helper()

	accounted := make(map[RequestID]bool, len(batch)+len(retired))
	for _, st := range batch {
		accounted[st.ID] = true
	}
	for _, st := range retired {
		if accounted[st.ID] {
			t.Fatalf("%s: %s reported as both batched and retired", label, st.ID)
		}
		accounted[st.ID] = true
	}
	q.Snapshot().Each(func(st *RequestState) bool {
		accounted[st.ID] = true
		return true
	})
	for _, st := range running {
		if !accounted[st.ID] {
			t.Fatalf("%s: %s left the running set without being reported in batch, retired or the queue", label, st.ID)
		}
	}
}

// TestFormBatchReportsTheSequencesItRetires pins the leak directly: a member of
// the running set that is already finished has its cache released by FormBatch, and
// a caller that only reads the batch cannot tell that from a sequence still
// running.
func TestFormBatchReportsTheSequencesItRetires(t *testing.T) {
	policy := testPolicy(16, 16)
	for _, tc := range []struct {
		name string
		make func(cache KVCache, est *CostEstimator) Batcher
	}{
		{"continuous", func(cache KVCache, est *CostEstimator) Batcher {
			return NewContinuousBatcher(0, est, ReservePeak, cache, PreemptNewest)
		}},
		{"static", func(cache KVCache, est *CostEstimator) Batcher {
			return NewStaticBatcher(0, est, ReservePeak, cache)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cache := NewPagedCache(policy)
			b := tc.make(cache, batchTestEstimator(policy))
			w := testWorker("w1", "m1", 16, 0, 4, 0)

			q := newACBQueue(8)
			// Two tokens of ceiling, so it is finished after two steps and the step after
			// that is the one that retires it.
			done := queuedRequest("done", 16, 2)
			q.Push(done)

			batch, retired := b.FormBatch(nil, q, w, testNow)
			if len(batch) != 1 || len(retired) != 0 {
				t.Fatalf("initial: batch=%v retired=%v", ids(batch), ids(retired))
			}
			done.GeneratedTokens = done.MaxOutputTokens
			if cache.Used() == 0 {
				t.Fatal("admission reserved nothing, so there is no reservation to leak")
			}

			batch, retired = b.FormBatch(batch, q, w, testNow.Add(time.Millisecond))
			if len(batch) != 0 {
				t.Fatalf("batch = %v, want empty once the only member finished", ids(batch))
			}
			if len(retired) != 1 || retired[0].ID != "done" {
				t.Fatalf("retired = %v, want [done]: the cache was released, so the caller must be told",
					ids(retired))
			}
			if cache.Used() != 0 {
				t.Fatalf("cache used = %d after retirement", cache.Used())
			}
		})
	}
}

// TestStepReportsAnAlreadyFinishedMemberAsRetired covers the path F4 makes
// reachable: a preempted sequence keeps its generated tokens, so it can be
// re-admitted already past its ceiling and be retired by the very next Step.
func TestStepReportsAnAlreadyFinishedMemberAsRetired(t *testing.T) {
	policy := testPolicy(16, 16)
	cache := NewPagedCache(policy)
	b := NewContinuousBatcher(0, batchTestEstimator(policy), ReservePeak, cache, PreemptNewest)
	w := testWorker("w1", "m1", 16, 0, 4, 0)

	q := newACBQueue(8)
	st := queuedRequest("resumed", 16, 4)
	q.Push(st)
	batch, _ := b.FormBatch(nil, q, w, testNow)
	st.GeneratedTokens = st.MaxOutputTokens

	survivors, preempted, retired := b.Step(batch, q, nil, testNow.Add(time.Millisecond))
	assertStepAccounting(t, "retire-on-step", batch, survivors, preempted, retired)
	if len(retired) != 1 || retired[0].ID != "resumed" {
		t.Fatalf("retired = %v, want [resumed]", ids(retired))
	}
	if cache.Used() != 0 {
		t.Fatalf("cache used = %d after the sequence was retired", cache.Used())
	}
}

// TestPreemptionKeepsTokensAndRestartsTheWait pins the two halves of the
// recomputation design: the victim's progress survives, and its wait does not.
func TestPreemptionKeepsTokensAndRestartsTheWait(t *testing.T) {
	policy := testPolicy(16, 4)
	cache := NewPagedCache(policy)
	est := NewCostEstimator(policy, FractionOfMax{Fraction: 0.0001, Floor: 1})
	b := NewContinuousBatcher(0, est, ReserveExpected, cache, PreemptNewest)
	w := testWorker("w1", "m1", 4, 0, 2, 0)

	// One block each on admission and two spare, so both run well past their first
	// block boundary before the cache is exhausted and the victim has real progress
	// to lose.
	q := newACBQueue(8)
	older := queuedRequest("older", 1, 200)
	newer := queuedRequest("newer", 1, 200)
	q.Push(older)
	q.Push(newer)
	batch, _ := b.FormBatch(nil, q, w, testNow)
	if len(batch) != 2 {
		t.Fatalf("initial batch size %d, want 2", len(batch))
	}
	older.StartedAt, newer.StartedAt = testNow, testNow.Add(time.Millisecond)

	// Far from the enqueue, so an unreset QueuedAt is unmistakable.
	const ranFor = 90 * time.Second
	var preempted []*RequestState
	preemptedAt := testNow
	for step := 0; step < 128 && len(preempted) == 0; step++ {
		preemptedAt = testNow.Add(ranFor + time.Duration(step)*time.Millisecond)
		batch, preempted, _ = b.Step(batch, q, nil, preemptedAt)
	}
	if len(preempted) != 1 {
		t.Fatalf("preempted = %v, want one victim", ids(preempted))
	}
	victim := preempted[0]
	if victim.GeneratedTokens < 16 {
		t.Fatalf("victim generated %d tokens; the scenario is meant to preempt one with real progress",
			victim.GeneratedTokens)
	}
	// The wait the scheduler and the autoscaler see must start at the preemption, not
	// at the original enqueue: the time the victim spent running is not queue wait.
	if wait := q.OldestWait(preemptedAt); wait != 0 {
		t.Fatalf("head-of-line wait = %v immediately after preemption; QueuedAt was not reset", wait)
	}
	if !victim.QueuedAt.Equal(preemptedAt) {
		t.Fatalf("victim QueuedAt = %v, want the preemption time %v", victim.QueuedAt, preemptedAt)
	}

	// Re-admission must reserve for what the sequence already spans, not for what a
	// fresh request of its shape would need, or its first step back fails to grow.
	resumed := est.EstimateResumed(victim)
	fresh := est.Estimate(&victim.Request)
	if resumed.ExpectedCacheBlocks <= fresh.ExpectedCacheBlocks {
		t.Fatalf("resumed reservation %d blocks is not larger than the original %d",
			resumed.ExpectedCacheBlocks, fresh.ExpectedCacheBlocks)
	}
	if want := policy.BlocksFor(victim.PromptTokens + victim.GeneratedTokens); resumed.ExpectedCacheBlocks < want {
		t.Fatalf("resumed reservation %d blocks does not cover the %d the sequence already spans",
			resumed.ExpectedCacheBlocks, want)
	}
	// The re-prefill is the real cost of the preemption, and it is a prefill over a
	// longer prompt than the client sent.
	if resumed.PrefillTokens != victim.PromptTokens+victim.GeneratedTokens {
		t.Fatalf("resumed prefill = %d tokens, want prompt+generated = %d",
			resumed.PrefillTokens, victim.PromptTokens+victim.GeneratedTokens)
	}
	if minimum := int64(resumed.PrefillTokens) * est.PrefillNanosPerToken; resumed.EstimatedServiceNanos < minimum {
		t.Fatalf("resumed service estimate %dns does not even cover the re-prefill of %d tokens",
			resumed.EstimatedServiceNanos, resumed.PrefillTokens)
	}
	if resumed.PeakCacheBlocks != fresh.PeakCacheBlocks {
		t.Fatalf("peak moved on resume: %d vs %d; the declared ceiling has not changed",
			resumed.PeakCacheBlocks, fresh.PeakCacheBlocks)
	}

	// Re-admission must actually hand it that larger reservation, which is what stops
	// its first step back from failing to grow and being preempted again. Checked on a
	// roomy cache, because the one it was evicted from has no room for its new
	// footprint — which is itself the point: a resumed sequence needs more capacity
	// than it did the first time.
	spans := policy.BlocksFor(victim.PromptTokens + victim.GeneratedTokens)
	roomyPolicy := testPolicy(16, 32)
	roomy := NewPagedCache(roomyPolicy)
	elsewhere := NewContinuousBatcher(0, est, ReserveExpected, roomy, PreemptNewest)
	q2 := newACBQueue(4)
	q2.Push(victim)
	readmitted, _ := elsewhere.FormBatch(nil, q2, testWorker("w2", "m1", 32, 0, 2, 0), preemptedAt.Add(time.Millisecond))
	if !contains(readmitted, victim.ID) {
		t.Fatal("victim was not re-admitted onto a worker with room for it")
	}
	if got := roomy.Blocks(victim.ID); got < spans {
		t.Fatalf("re-admitted victim holds %d blocks for a sequence spanning %d", got, spans)
	}
}

func ids(sts []*RequestState) []RequestID {
	out := make([]RequestID, 0, len(sts))
	for _, st := range sts {
		out = append(out, st.ID)
	}
	return out
}

func TestContinuousBatcherAdmitsIntoFreedCapacityMidFlight(t *testing.T) {
	policy := testPolicy(16, 8)
	cache := NewPagedCache(policy)
	est := batchTestEstimator(policy)
	b := NewContinuousBatcher(0, est, ReservePeak, cache, PreemptNewest)
	w := testWorker("w1", "m1", 8, 0, 2, 0)

	q := newACBQueue(8)
	long := queuedRequest("long", 16, 64)  // 5 blocks
	short := queuedRequest("short", 16, 2) // 2 blocks
	third := queuedRequest("third", 16, 2) // 2 blocks, only fits once short leaves
	q.Push(long)
	q.Push(short)
	q.Push(third)

	var batch []*RequestState
	thirdStarted, longFinished := -1, -1
	for step := 0; step < 200; step++ {
		batch, _ = b.FormBatch(batch, q, w, testNow.Add(time.Duration(step)*time.Millisecond))
		if thirdStarted < 0 && contains(batch, third.ID) {
			thirdStarted = step
		}
		if longFinished < 0 && long.Phase == PhaseDone {
			longFinished = step
		}
		if thirdStarted >= 0 && longFinished >= 0 {
			break
		}
		var preempted []*RequestState
		batch, preempted, _ = b.Step(batch, q, nil, testNow.Add(time.Duration(step)*time.Millisecond))
		if len(preempted) != 0 {
			t.Fatalf("step %d: unexpected preemption %v", step, preempted[0].ID)
		}
		checkInvariant(t, cache, "continuous step")
	}

	if thirdStarted < 0 {
		t.Fatal("the queued request never started")
	}
	if longFinished < 0 {
		t.Fatal("the long request never finished")
	}
	t.Logf("continuous: queued request started at step %d, long generation finished at step %d",
		thirdStarted, longFinished)
	// This is the property static batching cannot have.
	if thirdStarted >= longFinished {
		t.Fatalf("queued request started at step %d but the long one finished at %d; nothing was admitted mid-flight",
			thirdStarted, longFinished)
	}
	if b.Admitted() != 3 {
		t.Fatalf("admitted %d sequences, want 3", b.Admitted())
	}
	if b.Preemptions() != 0 {
		t.Fatalf("%d preemptions in a workload that fits", b.Preemptions())
	}
}

func TestStaticBatcherIsHeldHostageByItsLongestMember(t *testing.T) {
	policy := testPolicy(16, 8)
	cache := NewPagedCache(policy)
	est := batchTestEstimator(policy)
	b := NewStaticBatcher(0, est, ReservePeak, cache)
	w := testWorker("w1", "m1", 8, 0, 2, 0)

	q := newACBQueue(8)
	long := queuedRequest("long", 16, 64)
	short := queuedRequest("short", 16, 2)
	third := queuedRequest("third", 16, 2)
	q.Push(long)
	q.Push(short)
	q.Push(third)

	var batch []*RequestState
	thirdStarted, shortFinished, longFinished := -1, -1, -1
	for step := 0; step < 400; step++ {
		now := testNow.Add(time.Duration(step) * time.Millisecond)
		batch, _ = b.FormBatch(batch, q, w, now)
		if thirdStarted < 0 && contains(batch, third.ID) {
			thirdStarted = step
		}
		if shortFinished < 0 && batcherDone(short) {
			shortFinished = step
		}
		if longFinished < 0 && batcherDone(long) {
			longFinished = step
		}
		// The pathology, asserted while it is happening: the short request is finished,
		// its slot is idle, capacity is free, and the queued request still cannot run.
		if shortFinished >= 0 && longFinished < 0 {
			if contains(batch, third.ID) {
				t.Fatalf("step %d: static batcher admitted into a freed slot", step)
			}
			if third.Phase != PhaseQueued || q.Len() == 0 {
				t.Fatalf("step %d: queued request left the queue before the batch drained", step)
			}
			if cache.Free() < 2 {
				t.Fatalf("step %d: expected free capacity while the batch is held, got %d", step, cache.Free())
			}
		}
		for _, st := range batch {
			st.Phase = PhaseDecode
			st.GeneratedTokens++
		}
		checkInvariant(t, cache, "static step")
		if thirdStarted >= 0 {
			break
		}
	}

	if shortFinished < 0 || longFinished < 0 || thirdStarted < 0 {
		t.Fatalf("workload did not complete: short=%d long=%d third=%d", shortFinished, longFinished, thirdStarted)
	}
	t.Logf("static: short finished at step %d, long at step %d, queued request started at step %d (%d steps of idle slot)",
		shortFinished, longFinished, thirdStarted, thirdStarted-shortFinished)
	if thirdStarted < longFinished {
		t.Fatalf("queued request started at %d, before the long one finished at %d: this is not static batching",
			thirdStarted, longFinished)
	}
	if idle := thirdStarted - shortFinished; idle < 50 {
		t.Fatalf("only %d steps of hostage time; the pathology is not being exercised", idle)
	}
	if b.Batches() != 2 {
		t.Fatalf("%d batches dispatched, want 2", b.Batches())
	}
}

func TestContinuousBatcherPreemptsOnGrowFailureAndCountsIt(t *testing.T) {
	// Five blocks for two sequences that each start with two and will both need a
	// third on the same step. One of them cannot have it.
	policy := testPolicy(16, 5)
	cache := NewPagedCache(policy)
	est := NewCostEstimator(policy, FractionOfMax{Fraction: 0.01, Floor: 1})
	b := NewContinuousBatcher(0, est, ReserveExpected, cache, PreemptNewest)
	w := testWorker("w1", "m1", 5, 0, 2, 0)

	q := newACBQueue(8)
	first := queuedRequest("first", 16, 160)
	second := queuedRequest("second", 16, 160)
	q.Push(first)
	q.Push(second)

	batch, _ := b.FormBatch(nil, q, w, testNow)
	if len(batch) != 2 {
		t.Fatalf("initial batch size %d, want 2", len(batch))
	}
	// Distinct start times so "newest" is well defined.
	first.StartedAt = testNow
	second.StartedAt = testNow.Add(time.Millisecond)
	if cache.Used() != 4 || cache.Free() != 1 {
		t.Fatalf("after admission: used=%d free=%d, want 4/1", cache.Used(), cache.Free())
	}

	var preempted []*RequestState
	for step := 0; step < 64 && len(preempted) == 0; step++ {
		batch, preempted, _ = b.Step(batch, q, nil, testNow.Add(time.Duration(step)*time.Millisecond))
		checkInvariant(t, cache, "preemption step")
	}

	if len(preempted) != 1 {
		t.Fatalf("%d sequences preempted, want 1", len(preempted))
	}
	victim := preempted[0]
	if victim.ID != "second" {
		t.Fatalf("victim = %s, want second under the newest-first policy", victim.ID)
	}
	if victim.Phase != PhaseQueued {
		t.Fatalf("victim phase = %v, want queued so it is re-run rather than lost", victim.Phase)
	}
	if cache.Blocks(victim.ID) != 0 {
		t.Fatalf("victim still holds %d blocks", cache.Blocks(victim.ID))
	}
	if b.Preemptions() != 1 {
		t.Fatalf("preemption count = %d, want 1; an uncounted preemption is an invisible one", b.Preemptions())
	}
	if q.Len() != 1 {
		t.Fatalf("queue length = %d, want 1: the preempted request must be re-queued", q.Len())
	}
	if len(batch) != 1 || batch[0].ID != "first" {
		t.Fatalf("surviving batch = %v, want just first", batch)
	}
	if cache.Blocks("first") != 3 {
		t.Fatalf("survivor holds %d blocks, want 3 (it got the block it needed)", cache.Blocks("first"))
	}
	t.Logf("preempted %s after %d generated tokens to let %s grow",
		victim.ID, victim.GeneratedTokens, batch[0].ID)
}

func TestVictimPolicyChangesWhoIsPreempted(t *testing.T) {
	policy := testPolicy(16, 5)
	cache := NewPagedCache(policy)
	// A tiny fraction so both sequences are admitted on a two-block footprint
	// regardless of their very different ceilings, which is what puts the two victim
	// policies in disagreement.
	est := NewCostEstimator(policy, FractionOfMax{Fraction: 0.0001, Floor: 1})
	// Longest-remaining frees the biggest future reservation at the cost of
	// discarding the most completed work; it should pick the other victim here.
	b := NewContinuousBatcher(0, est, ReserveExpected, cache, PreemptLongestRemaining)
	w := testWorker("w1", "m1", 5, 0, 2, 0)

	q := newACBQueue(8)
	hog := queuedRequest("hog", 16, 4096)
	modest := queuedRequest("modest", 16, 160)
	q.Push(hog)
	q.Push(modest)

	batch, _ := b.FormBatch(nil, q, w, testNow)
	if len(batch) != 2 {
		t.Fatalf("initial batch size %d, want 2", len(batch))
	}
	hog.StartedAt = testNow
	modest.StartedAt = testNow.Add(time.Millisecond)

	var preempted []*RequestState
	for step := 0; step < 64 && len(preempted) == 0; step++ {
		batch, preempted, _ = b.Step(batch, q, nil, testNow.Add(time.Duration(step)*time.Millisecond))
		checkInvariant(t, cache, "victim policy step")
	}
	if len(preempted) != 1 || preempted[0].ID != "hog" {
		t.Fatalf("preempted = %v, want hog under longest-remaining", preempted)
	}
	if len(batch) != 1 || batch[0].ID != "modest" {
		t.Fatalf("survivors = %v, want modest", batch)
	}
	// The survivor's failed grow is retried once with the victim's blocks returned,
	// so the step it was taking actually completes.
	if got := cache.Blocks("modest"); got != 3 {
		t.Fatalf("survivor holds %d blocks, want 3 after the retry", got)
	}
}

func TestPreemptedRequestIsFailedVisiblyWhenTheQueueIsFull(t *testing.T) {
	policy := testPolicy(16, 5)
	cache := NewPagedCache(policy)
	est := NewCostEstimator(policy, FractionOfMax{Fraction: 0.01, Floor: 1})
	b := NewContinuousBatcher(0, est, ReserveExpected, cache, PreemptNewest)
	w := testWorker("w1", "m1", 5, 0, 2, 0)

	q := newACBQueue(2)
	first := queuedRequest("first", 16, 160)
	second := queuedRequest("second", 16, 160)
	q.Push(first)
	q.Push(second)
	batch, _ := b.FormBatch(nil, q, w, testNow)
	first.StartedAt = testNow
	second.StartedAt = testNow.Add(time.Millisecond)

	// Fill the queue so the preempted request has nowhere to go.
	q.Push(queuedRequest("filler-1", 16, 16))
	q.Push(queuedRequest("filler-2", 16, 16))

	var preempted []*RequestState
	for step := 0; step < 64 && len(preempted) == 0; step++ {
		batch, preempted, _ = b.Step(batch, q, nil, testNow.Add(time.Duration(step)*time.Millisecond))
	}
	if len(preempted) != 1 {
		t.Fatalf("%d preempted, want 1", len(preempted))
	}
	if preempted[0].Phase != PhaseDone {
		t.Fatalf("phase = %v, want done: a preemption with nowhere to retry must fail visibly", preempted[0].Phase)
	}
	if b.Preemptions() != 1 {
		t.Fatalf("preemptions = %d, want 1; the shortage happened regardless of the retry", b.Preemptions())
	}
	checkInvariant(t, cache, "queue-full preemption")
}

func TestStaticWaitWindowChangesBatchSize(t *testing.T) {
	policy := testPolicy(16, 256)
	est := batchTestEstimator(policy)
	w := testWorker("w1", "m1", 256, 0, 4, 0)

	// No window: whatever is runnable at the instant of the call is the batch, so a
	// trickle of arrivals produces batches of one.
	immediate := NewStaticBatcher(0, est, ReservePeak, NewPagedCache(policy))
	q := newACBQueue(8)
	q.Push(queuedRequest("a", 16, 16))
	batch, _ := immediate.FormBatch(nil, q, w, testNow)
	if len(batch) != 1 {
		t.Fatalf("zero-window batch size = %d, want 1", len(batch))
	}

	// A 10ms window over the same trickle: three more arrive before it closes, and
	// the batch is four times the size for the price of up to 10ms of latency.
	windowed := NewStaticBatcher(10*time.Millisecond, est, ReservePeak, NewPagedCache(policy))
	q2 := newACBQueue(8)
	q2.Push(queuedRequest("a", 16, 16))
	if got, _ := windowed.FormBatch(nil, q2, w, testNow); got != nil {
		t.Fatalf("batch of %d dispatched inside the wait window", len(got))
	}
	q2.Push(queuedRequest("b", 16, 16))
	if got, _ := windowed.FormBatch(nil, q2, w, testNow.Add(5*time.Millisecond)); got != nil {
		t.Fatalf("batch of %d dispatched inside the wait window", len(got))
	}
	q2.Push(queuedRequest("c", 16, 16))
	q2.Push(queuedRequest("d", 16, 16))
	batch, _ = windowed.FormBatch(nil, q2, w, testNow.Add(11*time.Millisecond))
	if len(batch) != 4 {
		t.Fatalf("windowed batch size = %d, want 4", len(batch))
	}
	t.Logf("static wait window: 0ms -> batch of 1, 10ms -> batch of 4")

	// A full batch dispatches without waiting out the window: holding a batch that
	// cannot grow adds latency for no throughput.
	full := NewStaticBatcher(time.Hour, est, ReservePeak, NewPagedCache(policy))
	q3 := newACBQueue(8)
	for _, id := range []RequestID{"a", "b", "c", "d"} {
		q3.Push(queuedRequest(id, 16, 16))
	}
	if got, _ := full.FormBatch(nil, q3, w, testNow); len(got) != 4 {
		t.Fatalf("full batch held by the window: got %d, want 4", len(got))
	}
}

func TestContinuousWaitWindowAppliesOnlyWhenIdle(t *testing.T) {
	policy := testPolicy(16, 256)
	est := batchTestEstimator(policy)
	w := testWorker("w1", "m1", 256, 0, 4, 0)
	cache := NewPagedCache(policy)
	b := NewContinuousBatcher(10*time.Millisecond, est, ReservePeak, cache, PreemptNewest)

	q := newACBQueue(8)
	q.Push(queuedRequest("a", 16, 64))
	if got, _ := b.FormBatch(nil, q, w, testNow); got != nil {
		t.Fatalf("dispatched %d sequences inside the window from an idle running set", len(got))
	}
	if q.Len() != 1 {
		t.Fatalf("queue length = %d; a held request must stay in the queue where expiry can see it", q.Len())
	}
	if cache.Used() != 0 {
		t.Fatalf("cache used = %d while holding for company; the reservation must not be held", cache.Used())
	}

	q.Push(queuedRequest("b", 16, 64))
	q.Push(queuedRequest("c", 16, 64))
	batch, _ := b.FormBatch(nil, q, w, testNow.Add(11*time.Millisecond))
	if len(batch) != 3 {
		t.Fatalf("batch size after the window closed = %d, want 3", len(batch))
	}

	// Already running: a new arrival is admitted on the next step rather than held,
	// because delaying it buys no batching and costs pure latency.
	q.Push(queuedRequest("d", 16, 64))
	batch, _ = b.FormBatch(batch, q, w, testNow.Add(12*time.Millisecond))
	if len(batch) != 4 || !contains(batch, "d") {
		t.Fatalf("batch = %d sequences, want 4 including the new arrival", len(batch))
	}
	if b.MaxWait() != 10*time.Millisecond {
		t.Fatalf("MaxWait = %v, want 10ms", b.MaxWait())
	}
}

func TestBatcherNamesAndCacheReleaseOnCompletion(t *testing.T) {
	policy := testPolicy(16, 64)
	est := batchTestEstimator(policy)
	cache := NewPagedCache(policy)
	cont := NewContinuousBatcher(0, est, ReservePeak, cache, PreemptNewest)
	stat := NewStaticBatcher(0, est, ReservePeak, NewPagedCache(policy))
	if cont.Name() != "continuous" || stat.Name() != "static" {
		t.Fatalf("names = %q/%q", cont.Name(), stat.Name())
	}

	// Blocks must come back when a sequence finishes; a batcher that evicts without
	// releasing leaks the whole cache over a long run.
	w := testWorker("w1", "m1", 64, 0, 4, 0)
	q := newACBQueue(8)
	q.Push(queuedRequest("a", 16, 1))
	batch, _ := cont.FormBatch(nil, q, w, testNow)
	if cache.Used() == 0 {
		t.Fatal("admission reserved nothing")
	}
	batch, _, _ = cont.Step(batch, q, nil, testNow)
	batch, _ = cont.FormBatch(batch, q, w, testNow.Add(time.Millisecond))
	if len(batch) != 0 {
		t.Fatalf("batch = %d sequences, want 0 after the only one finished", len(batch))
	}
	if cache.Used() != 0 || cache.Free() != 64 {
		t.Fatalf("after completion: used=%d free=%d, want 0/64", cache.Used(), cache.Free())
	}
}
