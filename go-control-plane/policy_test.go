package loom

import (
	"math/rand"
	"sort"
	"strconv"
	"testing"
	"time"
)

// policyTestClock is a settable clock, because every interesting property of
// these policies is a function of wait time and none of them should need a sleep
// to exercise.
type policyTestClock struct{ now time.Time }

func (c *policyTestClock) Now() time.Time          { return c.now }
func (c *policyTestClock) Advance(d time.Duration) { c.now = c.now.Add(d) }

func policyTestReq(id, tenant string, promptTokens, outputTokens, priority int) *RequestState {
	return &RequestState{
		Request: Request{
			ID:              RequestID(id),
			Tenant:          TenantID(tenant),
			Model:           "m",
			Priority:        priority,
			PromptTokens:    promptTokens,
			MaxOutputTokens: outputTokens,
		},
	}
}

func policyTestQueue(capacity int, p SchedulePolicy, clk *policyTestClock) *BoundedQueue {
	q := NewBoundedQueue(capacity, p)
	q.SetClock(clk.Now)
	return q
}

func popIDs(q *BoundedQueue, n int) []RequestID {
	var out []RequestID
	for i := 0; i < n; i++ {
		st := q.Pop()
		if st == nil {
			break
		}
		out = append(out, st.ID)
	}
	return out
}

func TestFIFOStrictArrivalOrder(t *testing.T) {
	clk := &policyTestClock{now: schedTestBase}
	q := policyTestQueue(32, NewFIFOPolicy(), clk)

	// Deliberately decreasing cost, so a cost-sensitive policy would reorder and
	// FIFO must not.
	var want []RequestID
	for i := 0; i < 10; i++ {
		id := "r" + strconv.Itoa(i)
		q.Push(policyTestReq(id, "t"+strconv.Itoa(i%3), 32, 1000-80*i, 0))
		want = append(want, RequestID(id))
		clk.Advance(time.Millisecond)
	}

	got := popIDs(q, 10)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("FIFO order = %v, want %v", got, want)
		}
	}
}

func TestPolicyRespectsPriority(t *testing.T) {
	// Each policy's own ordering must be subordinate to Priority: the cheap,
	// early, or fair-share-favoured request still loses to a higher priority.
	cases := []struct {
		name   string
		policy SchedulePolicy
	}{
		{"fifo", NewFIFOPolicy()},
		{"sjf", NewShortestJobFirstPolicy(0)},
		{"sjf-aged", NewShortestJobFirstPolicy(1)},
		{"fair-share", NewFairSharePolicy()},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			clk := &policyTestClock{now: schedTestBase}
			q := policyTestQueue(16, tc.policy, clk)

			// Arrives first, cheapest, and from a tenant with no history.
			q.Push(policyTestReq("cheap-early", "t1", 8, 8, 0))
			clk.Advance(time.Second)
			q.Push(policyTestReq("expensive-late", "t2", 4096, 4096, 5))
			clk.Advance(time.Second)
			q.Push(policyTestReq("mid", "t3", 64, 64, 1))

			got := popIDs(q, 3)
			want := []RequestID{"expensive-late", "mid", "cheap-early"}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("order = %v, want %v", got, want)
				}
			}
		})
	}
}

// sjfStarvationRun streams short requests past one long request and reports how
// long the long one waited before being selected, and whether it ran at all.
func sjfStarvationRun(t *testing.T, agingRate float64, steps int, step time.Duration) (time.Duration, bool) {
	t.Helper()

	clk := &policyTestClock{now: schedTestBase}
	pol := NewShortestJobFirstPolicy(agingRate)
	q := policyTestQueue(256, pol, clk)

	long := policyTestReq("long", "t1", 512, 5000, 0)
	if !q.Push(long) {
		t.Fatal("pushing the long request failed")
	}
	// A standing backlog of short work. Starvation is a property of a queue that
	// never empties, so the backlog is refilled as it drains.
	const backlog = 4
	shortSeq := 0
	refill := func() {
		for q.Len() < backlog+1 {
			q.Push(policyTestReq("s"+strconv.Itoa(shortSeq), "t1", 32, 32, 0))
			shortSeq++
		}
	}
	refill()

	for i := 0; i < steps; i++ {
		st := q.Pop()
		if st == nil {
			t.Fatalf("step %d: Pop returned nil with %d queued", i, q.Len())
		}
		if st.ID == "long" {
			return clk.now.Sub(st.QueuedAt), true
		}
		clk.Advance(step)
		refill()
	}
	return clk.now.Sub(schedTestBase), false
}

func TestSJFStarvesLongRequestWithoutAging(t *testing.T) {
	// The pathology, asserted rather than described: with aging off, the long
	// request is passed over by every cheaper arrival and its wait is bounded only
	// by how long the test is willing to run.
	const (
		steps     = 600
		step      = 100 * time.Millisecond
		threshold = 30 * time.Second
	)
	wait, ran := sjfStarvationRun(t, 0, steps, step)
	if ran {
		t.Fatalf("pure SJF dispatched the long request after %v; expected starvation", wait)
	}
	if wait < threshold {
		t.Fatalf("long request waited only %v, want at least %v", wait, threshold)
	}
	t.Logf("pure SJF: long request still queued after %v", wait)
}

func TestSJFAgingPreventsStarvation(t *testing.T) {
	// The fix. The rate is the exchange rate between waiting and cost: at 50, one
	// second of waiting cancels 50 seconds of estimated service, so the ~100s long
	// request overtakes the ~0.6s short ones after about two seconds of waiting.
	const (
		steps = 600
		step  = 100 * time.Millisecond
		bound = 5 * time.Second
	)
	wait, ran := sjfStarvationRun(t, 50, steps, step)
	if !ran {
		t.Fatalf("aged SJF never dispatched the long request within %v", wait)
	}
	if wait > bound {
		t.Fatalf("aged SJF dispatched the long request after %v, want within %v", wait, bound)
	}
	t.Logf("aged SJF: long request dispatched after %v", wait)
}

func TestSJFStillPrefersShortWorkWhenNothingIsOld(t *testing.T) {
	// Aging must not degenerate into FIFO: with every request freshly queued, the
	// cheapest still wins, which is where SJF's mean-latency advantage comes from.
	clk := &policyTestClock{now: schedTestBase}
	q := policyTestQueue(16, NewShortestJobFirstPolicy(50), clk)
	q.Push(policyTestReq("long", "t1", 512, 4000, 0))
	q.Push(policyTestReq("short", "t1", 32, 16, 0))

	if st := q.Pop(); st == nil || st.ID != "short" {
		t.Fatalf("Pop() = %v, want short", st)
	}
}

// fairShareRun keeps both tenants permanently backlogged and reports the service
// time each one was granted.
func fairShareRun(t *testing.T, pol *FairSharePolicy, pops int, heavyOutput, lightOutput int) (servedWork map[TenantID]int64, servedCount map[TenantID]int) {
	t.Helper()

	clk := &policyTestClock{now: schedTestBase}
	q := policyTestQueue(512, pol, clk)

	const backlogPerTenant = 6
	seq := 0
	refill := func() {
		for q.TenantCount("heavy") < backlogPerTenant {
			q.Push(policyTestReq("h"+strconv.Itoa(seq), "heavy", 128, heavyOutput, 0))
			seq++
		}
		for q.TenantCount("light") < backlogPerTenant {
			q.Push(policyTestReq("l"+strconv.Itoa(seq), "light", 128, lightOutput, 0))
			seq++
		}
	}
	refill()

	servedWork = map[TenantID]int64{}
	servedCount = map[TenantID]int{}
	cost := DefaultPolicyCostModel()
	for i := 0; i < pops; i++ {
		st := q.Pop()
		if st == nil {
			t.Fatalf("pop %d returned nil with %d queued", i, q.Len())
		}
		servedWork[st.Tenant] += cost.ServiceNanos(st)
		servedCount[st.Tenant]++
		clk.Advance(10 * time.Millisecond)
		refill()
	}
	return servedWork, servedCount
}

func TestFairShareBoundsHeavyTenant(t *testing.T) {
	// The heavy tenant submits requests with ten times the output budget, which is
	// the shape of the attack: one tenant's huge prompts and generations soaking
	// the accelerator. With equal weights, served work should land near 1:1 rather
	// than near the 10:1 the submission mix would give under FIFO.
	pol := NewFairSharePolicy()
	work, count := fairShareRun(t, pol, 200, 2000, 200)

	if count["light"] == 0 {
		t.Fatal("light tenant was never served")
	}
	ratio := float64(work["heavy"]) / float64(work["light"])
	t.Logf("fair share, equal weights: heavy served %d requests / %v, light served %d requests / %v, work ratio %.2f",
		count["heavy"], time.Duration(work["heavy"]), count["light"], time.Duration(work["light"]), ratio)

	if ratio < 0.8 || ratio > 1.25 {
		t.Fatalf("served work ratio heavy:light = %.2f, want near 1.0 (submission ratio was 10)", ratio)
	}
	// Not starved: the lighter tenant, submitting a tenth of the work per request,
	// should be selected far more often than the heavy one.
	if count["light"] <= count["heavy"] {
		t.Fatalf("light served %d times, heavy %d: the lighter tenant is not getting its turns",
			count["light"], count["heavy"])
	}
}

func TestFairShareHonoursWeights(t *testing.T) {
	pol := NewFairSharePolicy()
	pol.SetWeight("heavy", 3)
	pol.SetWeight("light", 1)

	work, count := fairShareRun(t, pol, 240, 400, 400)
	ratio := float64(work["heavy"]) / float64(work["light"])
	t.Logf("fair share, weights 3:1: heavy %d reqs / %v, light %d reqs / %v, work ratio %.2f",
		count["heavy"], time.Duration(work["heavy"]), count["light"], time.Duration(work["light"]), ratio)

	if ratio < 2.4 || ratio > 3.6 {
		t.Fatalf("served work ratio heavy:light = %.2f, want near the 3.0 weight ratio", ratio)
	}
}

func TestFairShareNewTenantIsNotPunishedOrRewarded(t *testing.T) {
	// A tenant arriving mid-stream must not inherit credit for having been idle,
	// which would let it monopolize on arrival.
	clk := &policyTestClock{now: schedTestBase}
	pol := NewFairSharePolicy()
	q := policyTestQueue(64, pol, clk)

	for i := 0; i < 10; i++ {
		q.Push(policyTestReq("a"+strconv.Itoa(i), "incumbent", 128, 256, 0))
	}
	for i := 0; i < 4; i++ {
		if st := q.Pop(); st == nil || st.Tenant != "incumbent" {
			t.Fatalf("expected incumbent to be served while alone, got %v", st)
		}
		clk.Advance(10 * time.Millisecond)
	}

	for i := 0; i < 4; i++ {
		q.Push(policyTestReq("n"+strconv.Itoa(i), "newcomer", 128, 256, 0))
	}
	served := map[TenantID]int{}
	for i := 0; i < 6; i++ {
		st := q.Pop()
		if st == nil {
			t.Fatal("unexpected nil pop")
		}
		served[st.Tenant]++
		clk.Advance(10 * time.Millisecond)
	}
	if served["newcomer"] == 0 {
		t.Fatal("newcomer was starved by the incumbent's accumulated virtual time")
	}
	if served["newcomer"] > 4 || served["incumbent"] == 0 {
		t.Fatalf("newcomer monopolized on arrival: newcomer %d, incumbent %d",
			served["newcomer"], served["incumbent"])
	}
}

// oneWorker is a worker with a settable amount of free cache, used to make a
// request infeasible on purpose.
func oneWorker(totalBlocks, usedBlocks int) []*WorkerState {
	return []*WorkerState{{
		ID:               "w1",
		Model:            "m",
		TotalCacheBlocks: totalBlocks,
		UsedCacheBlocks:  usedBlocks,
		MaxBatchSize:     8,
		LastHeartbeat:    schedTestBase,
	}}
}

func TestHeadOfLineBlockingIsSkippedButBounded(t *testing.T) {
	// Both halves matter, and fixing either one alone is the easy mistake: skip
	// the unschedulable head and smaller work flows but the big request never
	// runs; refuse to skip and one oversized request stalls the whole system.
	for _, tc := range []struct {
		name   string
		policy SchedulePolicy
		grace  time.Duration
	}{
		{"fifo", NewFIFOPolicy(), time.Second},
		{"sjf", NewShortestJobFirstPolicy(0), time.Second},
		{"fair-share", NewFairSharePolicy(), time.Second},
	} {
		t.Run(tc.name, func(t *testing.T) {
			switch p := tc.policy.(type) {
			case *FIFOPolicy:
				p.SkipGrace = tc.grace
			case *ShortestJobFirstPolicy:
				p.SkipGrace = tc.grace
			case *FairSharePolicy:
				p.SkipGrace = tc.grace
			}

			clk := &policyTestClock{now: schedTestBase}
			q := policyTestQueue(32, tc.policy, clk)

			// 64 free blocks at 16 tokens per block holds 1024 tokens; the big
			// request needs 512 blocks, so it fits on an empty worker and not on
			// this one — which is the distinction between "skip for now" and the
			// RejectTooLarge that admission is responsible for.
			workers := oneWorker(1024, 960)
			q.SetWorkerSource(func() []*WorkerState { return workers })

			big := policyTestReq("big", "t1", 4096, 4096, 0)
			if !q.Push(big) {
				t.Fatal("pushing the big request failed")
			}
			for i := 0; i < 4; i++ {
				q.Push(policyTestReq("small"+strconv.Itoa(i), "t1", 64, 64, 0))
			}

			// Part one: the infeasible head does not block the feasible tail.
			for i := 0; i < 3; i++ {
				clk.Advance(100 * time.Millisecond)
				st := q.Pop()
				if st == nil {
					t.Fatalf("pop %d returned nil: the infeasible head blocked feasible work", i)
				}
				if st.ID == "big" {
					t.Fatalf("pop %d dispatched the big request, which does not fit", i)
				}
			}

			// Part two: once the big request has waited out the grace period it stops
			// yielding, so the running set drains instead of smaller work overtaking
			// it forever.
			clk.Advance(2 * time.Second)
			if st := q.Pop(); st != nil {
				t.Fatalf("after the grace period Pop() = %v, want nil so the big request keeps its place", st.ID)
			}

			// Part three: and when capacity appears, it runs — within a bounded
			// number of dispatches, not necessarily the very next one, since a
			// cost-ordered policy may legitimately clear the remaining small work
			// first. The bound is what makes this a starvation test rather than a
			// strict-ordering test.
			workers[0].UsedCacheBlocks = 0
			ran := false
			for i := 0; i <= q.Len(); i++ {
				st := q.Pop()
				if st == nil {
					t.Fatalf("Pop() = nil with %d queued and capacity free", q.Len())
				}
				clk.Advance(100 * time.Millisecond)
				if st.ID == "big" {
					ran = true
					break
				}
			}
			if !ran {
				t.Fatal("the big request never ran even after capacity freed")
			}
		})
	}
}

// TestFeasibilityGatesOnTheReservationActuallyMade pins the agreement between the
// scheduler's feasibility test and the admission controller's reservation mode. A
// policy testing the peak while the system reserves the expected footprint refuses
// to dispatch at the concurrency the rest of the system is planning for.
func TestFeasibilityGatesOnTheReservationActuallyMade(t *testing.T) {
	cachePolicy := CachePolicy{TokensPerBlock: 16, TotalBlocks: 1024}
	// Half of the declared ceiling, which is what the expected footprint is worth
	// here: 64 + 1024 tokens of peak is 68 blocks, 64 + 512 is 36.
	est := NewCostEstimator(cachePolicy, FractionOfMax{Fraction: 0.5, Floor: 1})

	clk := &policyTestClock{now: schedTestBase}
	pol := NewFIFOPolicy()
	q := policyTestQueue(32, pol, clk)
	// 40 free blocks: more than the expected 36, less than the peak 68.
	workers := oneWorker(1024, 984)
	q.SetWorkerSource(func() []*WorkerState { return workers })

	req := policyTestReq("r1", "t1", 64, 1024, 0)
	req.Est = est.Estimate(&req.Request)
	if req.Est.ExpectedCacheBlocks >= workers[0].FreeBlocks() || req.Est.PeakCacheBlocks <= workers[0].FreeBlocks() {
		t.Fatalf("setup: expected=%d peak=%d free=%d, want free strictly between them",
			req.Est.ExpectedCacheBlocks, req.Est.PeakCacheBlocks, workers[0].FreeBlocks())
	}
	q.Push(req)

	if got := pol.Cost.Blocks(req); got != req.Est.PeakCacheBlocks {
		t.Fatalf("default cost model reports %d blocks, want the peak %d", got, req.Est.PeakCacheBlocks)
	}
	if st := q.Pop(); st != nil {
		t.Fatal("peak-mode feasibility dispatched a request that does not fit at its peak")
	}

	pol.SetReservationMode(ReserveExpected, est)
	if got := pol.Cost.Blocks(req); got != req.Est.ExpectedCacheBlocks {
		t.Fatalf("expected-mode cost model reports %d blocks, want the reservation %d",
			got, req.Est.ExpectedCacheBlocks)
	}
	st := q.Pop()
	if st == nil {
		t.Fatal("under expected reservation the request fits, but the feasibility test refused it")
	}
	if st.ID != req.ID {
		t.Fatalf("popped %q", st.ID)
	}

	// A sequence that was preempted is past its original estimate, and the number it
	// needs is what it already spans — not what admission guessed before it ran.
	resumed := policyTestReq("r2", "t1", 64, 1024, 0)
	resumed.Est = est.Estimate(&resumed.Request)
	resumed.GeneratedTokens = 900
	if got, want := pol.Cost.Blocks(resumed), cachePolicy.BlocksFor(64+900); got < want {
		t.Fatalf("resumed sequence gated on %d blocks, want at least the %d it spans", got, want)
	}
}

// TestSchedulerAlignsPolicyFeasibilityWithAdmission checks the wiring, since the
// agreement is worthless if every caller has to remember to configure it.
func TestSchedulerAlignsPolicyFeasibilityWithAdmission(t *testing.T) {
	cachePolicy := CachePolicy{TokensPerBlock: 32, TotalBlocks: 1024}
	est := NewCostEstimator(cachePolicy, FractionOfMax{Fraction: 0.5, Floor: 1})
	pol := NewFIFOPolicy()
	if pol.ReservationMode() != ReservePeak {
		t.Fatalf("a fresh policy gates on %v, want the conservative peak", pol.ReservationMode())
	}

	NewScheduler(SchedulerConfig{
		Queue:     NewBoundedQueue(8, pol),
		Registry:  NewRegistry(RegistryConfig{}),
		Admission: NewQuotaAdmissionController(est, ReserveExpected, nil),
	})
	if pol.ReservationMode() != ReserveExpected {
		t.Fatalf("policy gates on %v while admission reserves %v", pol.ReservationMode(), ReserveExpected)
	}
	if pol.Cost.TokensPerBlock != cachePolicy.TokensPerBlock {
		t.Fatalf("policy block size = %d, admission's = %d; the two are counting different blocks",
			pol.Cost.TokensPerBlock, cachePolicy.TokensPerBlock)
	}
}

// TestGraceStallIsScopedToTheWorkerTheRequestIsWaitingFor pins the convoy fix: an
// overdue request stops the dispatches that would delay it, not every dispatch in
// the fleet.
func TestGraceStallIsScopedToTheWorkerTheRequestIsWaitingFor(t *testing.T) {
	clk := &policyTestClock{now: schedTestBase}
	pol := NewFIFOPolicy()
	pol.SkipGrace = time.Second
	q := policyTestQueue(32, pol, clk)

	// Two interchangeable workers, each with 64 of 1024 blocks free. The big request
	// needs 512 and fits on neither right now, so one of them has to drain for it —
	// but only one.
	workers := []*WorkerState{
		{ID: "w1", Model: "m", TotalCacheBlocks: 1024, UsedCacheBlocks: 960, MaxBatchSize: 8, LastHeartbeat: schedTestBase},
		{ID: "w2", Model: "m", TotalCacheBlocks: 1024, UsedCacheBlocks: 960, MaxBatchSize: 8, LastHeartbeat: schedTestBase},
	}
	q.SetWorkerSource(func() []*WorkerState { return workers })

	big := policyTestReq("big", "t1", 4096, 4096, 0)
	q.Push(big)
	for i := 0; i < 4; i++ {
		q.Push(policyTestReq("small"+strconv.Itoa(i), "t1", 64, 64, 0))
	}

	clk.Advance(2 * time.Second)
	if stalls, _ := pol.GraceStalls(); stalls != 0 {
		t.Fatalf("stall counter = %d before any decision", stalls)
	}

	// The overdue request is waiting on one worker; work that the other worker can
	// take must still flow, or the whole fleet drains to admit one request.
	st := q.Pop()
	if st == nil {
		stalls, who := pol.GraceStalls()
		t.Fatalf("the whole fleet stalled for %q (%d stalls) though the other worker had room", who, stalls)
	}
	if st.ID == "big" {
		t.Fatal("dispatched the big request, which fits on neither worker")
	}
	if stalls, _ := pol.GraceStalls(); stalls != 0 {
		t.Fatalf("stall counter = %d after a dispatch that was not suppressed", stalls)
	}

	// Now make the small work placeable only on the worker the big request is
	// draining: there is no longer a dispatch that does not delay it, so the stall is
	// correct — and it is counted, because an uncounted stall is a throughput loss
	// with nothing to read.
	target := pol.drainTargetFor(big, workers)
	if target == nil {
		t.Fatal("no drain target for a request that fits on an empty worker")
	}
	for _, w := range workers {
		if w != target {
			w.Draining = true
		}
	}
	if st := q.Pop(); st != nil {
		t.Fatalf("Pop() = %v; the overdue request has nowhere else to wait for, so it must keep its place", st.ID)
	}
	stalls, who := pol.GraceStalls()
	if stalls != 1 {
		t.Fatalf("stall counter = %d, want 1", stalls)
	}
	if who != big.ID {
		t.Fatalf("stall attributed to %q, want %q", who, big.ID)
	}

	// The guarantee the grace exists for: the worker it was waiting on drains, and it
	// runs.
	target.UsedCacheBlocks = 0
	ran := false
	for i := 0; i <= q.Len(); i++ {
		st := q.Pop()
		if st == nil {
			t.Fatalf("Pop() = nil with %d queued and capacity free", q.Len())
		}
		if st.ID == big.ID {
			ran = true
			break
		}
	}
	if !ran {
		t.Fatal("the overdue request never ran after the worker it was waiting for drained")
	}
}

func TestBoundedSkipDoesNotScanWholeQueue(t *testing.T) {
	// The skip budget is what keeps one pathological decision from walking a deep
	// queue: past MaxSkips infeasible candidates the policy gives up for this
	// round rather than looking for the one feasible request at the back.
	clk := &policyTestClock{now: schedTestBase}
	pol := NewFIFOPolicy()
	pol.MaxSkips = 3
	pol.SkipGrace = time.Hour
	q := policyTestQueue(64, pol, clk)
	q.SetWorkerSource(func() []*WorkerState { return oneWorker(256, 192) })

	for i := 0; i < 10; i++ {
		q.Push(policyTestReq("big"+strconv.Itoa(i), "t1", 4096, 4096, 0))
	}
	q.Push(policyTestReq("small", "t1", 32, 32, 0))

	if st := q.Pop(); st != nil {
		t.Fatalf("Pop() = %v, want nil: the feasible request is past the skip budget", st.ID)
	}

	pol.MaxSkips = 20
	st := q.Pop()
	if st == nil || st.ID != "small" {
		t.Fatalf("Pop() with a larger skip budget = %v, want small", st)
	}
}

func TestFeasibilityWithNoWorkerStateDoesNotStall(t *testing.T) {
	// An unwired worker source must not read as "no capacity", or a missing
	// wiring step would look exactly like a total stall.
	clk := &policyTestClock{now: schedTestBase}
	q := policyTestQueue(8, NewFIFOPolicy(), clk)
	q.Push(policyTestReq("huge", "t1", 1<<20, 1<<20, 0))
	if st := q.Pop(); st == nil || st.ID != "huge" {
		t.Fatalf("Pop() = %v, want huge", st)
	}
}

func TestFeasibilitySkipsDrainingAndWrongModelWorkers(t *testing.T) {
	clk := &policyTestClock{now: schedTestBase}
	pol := NewFIFOPolicy()
	pol.SkipGrace = time.Hour
	q := policyTestQueue(8, pol, clk)

	workers := []*WorkerState{
		{ID: "draining", Model: "m", TotalCacheBlocks: 1024, MaxBatchSize: 8, Draining: true},
		{ID: "other-model", Model: "other", TotalCacheBlocks: 1024, MaxBatchSize: 8},
		{ID: "full-batch", Model: "m", TotalCacheBlocks: 1024, MaxBatchSize: 2, ActiveSequences: 2},
	}
	q.SetWorkerSource(func() []*WorkerState { return workers })
	q.Push(policyTestReq("r1", "t1", 64, 64, 0))

	if st := q.Pop(); st != nil {
		t.Fatalf("Pop() = %v, want nil: no worker can take it", st.ID)
	}
	workers = append(workers, &WorkerState{ID: "good", Model: "m", TotalCacheBlocks: 1024, MaxBatchSize: 8})
	if st := q.Pop(); st == nil || st.ID != "r1" {
		t.Fatalf("Pop() = %v, want r1", st)
	}
}

// --- deterministic comparison of the three policies ---

type simArrival struct {
	id      RequestID
	tenant  TenantID
	at      time.Duration
	service time.Duration
}

// simWorkload is a fixed seeded arrival sequence: mostly short requests with a
// heavy tail, which is the shape real prompt and generation lengths have and the
// shape that makes the policies differ at all. A uniform workload would make all
// three look identical and prove nothing.
//
// The mean interarrival time is set above the mean service time on purpose. An
// overloaded queue has no steady-state wait distribution — every percentile is
// just a function of how long the simulation ran, and the policies get compared on
// backlog growth rather than on ordering. At roughly 0.7 utilization the queue
// still empties periodically, which is what makes the numbers mean something.
func simWorkload(n int) []simArrival {
	rng := rand.New(rand.NewSource(20240101))
	out := make([]simArrival, 0, n)
	var at time.Duration
	for i := 0; i < n; i++ {
		at += time.Duration(rng.ExpFloat64() * float64(110*time.Millisecond))
		var service time.Duration
		switch {
		case rng.Float64() < 0.85:
			service = time.Duration(5+rng.Intn(25)) * time.Millisecond
		case rng.Float64() < 0.8:
			service = time.Duration(100+rng.Intn(200)) * time.Millisecond
		default:
			service = time.Duration(800+rng.Intn(1200)) * time.Millisecond
		}
		tenant := TenantID("t" + strconv.Itoa(i%3))
		out = append(out, simArrival{
			id:      RequestID("r" + strconv.Itoa(i)),
			tenant:  tenant,
			at:      at,
			service: service,
		})
	}
	return out
}

// runSim plays the workload through one policy on a single worker that runs one
// request at a time, and returns every request's queue wait.
func runSim(t *testing.T, pol SchedulePolicy, setService func(func(*RequestState) int64), w []simArrival) []time.Duration {
	t.Helper()

	service := make(map[RequestID]time.Duration, len(w))
	for _, a := range w {
		service[a.id] = a.service
	}
	setService(func(st *RequestState) int64 { return int64(service[st.ID]) })

	clk := &policyTestClock{now: schedTestBase}
	q := policyTestQueue(len(w)+1, pol, clk)

	next := 0
	admit := func() {
		for next < len(w) && !schedTestBase.Add(w[next].at).After(clk.now) {
			a := w[next]
			st := policyTestReq(string(a.id), string(a.tenant), 64, 64, 0)
			if !q.Push(st) {
				t.Fatalf("sim queue overflowed at %s", a.id)
			}
			next++
		}
	}

	waits := make([]time.Duration, 0, len(w))
	freeAt := schedTestBase
	for len(waits) < len(w) {
		if clk.now.Before(freeAt) {
			clk.now = freeAt
		}
		admit()
		if q.Len() == 0 {
			if next >= len(w) {
				t.Fatalf("sim stalled with %d of %d served", len(waits), len(w))
			}
			clk.now = schedTestBase.Add(w[next].at)
			continue
		}
		st := q.Pop()
		if st == nil {
			t.Fatalf("sim: policy %s returned nil with %d queued", pol.Name(), q.Len())
		}
		waits = append(waits, clk.now.Sub(st.QueuedAt))
		freeAt = clk.now.Add(service[st.ID])
	}
	return waits
}

func quantile(sorted []time.Duration, q float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	i := int(q * float64(len(sorted)-1))
	return sorted[i]
}

func mean(ds []time.Duration) time.Duration {
	if len(ds) == 0 {
		return 0
	}
	var sum time.Duration
	for _, d := range ds {
		sum += d
	}
	return sum / time.Duration(len(ds))
}

func TestPolicyWaitDistributionsOnFixedWorkload(t *testing.T) {
	const n = 600
	w := simWorkload(n)

	fifo := NewFIFOPolicy()
	sjf := NewShortestJobFirstPolicy(0)
	fair := NewFairSharePolicy()
	// Aged SJF is not one of the three required rows; it is here because the whole
	// claim about aging is that it buys back the tail without giving up the mean,
	// and that claim should be measured on the same workload as everything else.
	sjfAged := NewShortestJobFirstPolicy(2)

	type result struct {
		name                string
		mean, p50, p99, max time.Duration
	}
	var results []result
	for _, c := range []struct {
		label      string
		pol        SchedulePolicy
		setService func(func(*RequestState) int64)
	}{
		{"fifo", fifo, func(f func(*RequestState) int64) { fifo.ServiceNanosFn = f }},
		{"sjf", sjf, func(f func(*RequestState) int64) { sjf.ServiceNanosFn = f }},
		{"fair-share", fair, func(f func(*RequestState) int64) { fair.ServiceNanosFn = f }},
		{"sjf-aged", sjfAged, func(f func(*RequestState) int64) { sjfAged.ServiceNanosFn = f }},
	} {
		waits := runSim(t, c.pol, c.setService, w)
		sort.Slice(waits, func(i, j int) bool { return waits[i] < waits[j] })
		results = append(results, result{
			name: c.label,
			mean: mean(waits),
			p50:  quantile(waits, 0.50),
			p99:  quantile(waits, 0.99),
			max:  waits[len(waits)-1],
		})
	}

	for _, r := range results {
		t.Logf("%-11s mean=%-14v p50=%-10v p99=%-10v max=%v", r.name, r.mean, r.p50, r.p99, r.max)
	}

	byName := map[string]result{}
	for _, r := range results {
		byName[r.name] = r
	}

	// SJF wins the body of the distribution. This is the half of the tradeoff that
	// is not in doubt.
	if byName["sjf"].mean >= byName["fifo"].mean {
		t.Errorf("expected SJF mean wait (%v) below FIFO's (%v)", byName["sjf"].mean, byName["fifo"].mean)
	}
	if byName["sjf"].p50 >= byName["fifo"].p50 {
		t.Errorf("expected SJF p50 wait (%v) below FIFO's (%v)", byName["sjf"].p50, byName["fifo"].p50)
	}

	// FIFO wins the tail — but at the worst case, not at p99.
	//
	// The naive version of this assertion, "FIFO has the lower p99", is false on
	// this workload and was measured to be false rather than assumed: SJF's p99 is
	// well under FIFO's, because only about 3% of requests are long ones and SJF
	// deferring them improves the wait of the other 97% enough to pull the whole
	// distribution down, including the 99th percentile. Starvation is a property of
	// a small subpopulation, so it surfaces above roughly the 99.5th percentile.
	// The statement that survives measurement is about the worst case: FIFO's
	// longest wait is bounded by the work queued ahead of a request, while SJF's is
	// bounded by nothing.
	if byName["fifo"].max >= byName["sjf"].max {
		t.Errorf("expected FIFO worst-case wait (%v) below SJF's (%v)", byName["fifo"].max, byName["sjf"].max)
	}
	if byName["sjf"].p99 < byName["fifo"].p99 {
		t.Logf("note: SJF p99 (%v) beats FIFO p99 (%v); SJF's penalty shows up at the max (%v vs %v), not at p99",
			byName["sjf"].p99, byName["fifo"].p99, byName["sjf"].max, byName["fifo"].max)
	}

	// Aging is supposed to buy the worst case back without surrendering the mean.
	if byName["sjf-aged"].max >= byName["sjf"].max {
		t.Errorf("aging did not improve the worst-case wait: aged %v, unaged %v",
			byName["sjf-aged"].max, byName["sjf"].max)
	}
	if byName["sjf-aged"].mean >= byName["fifo"].mean {
		t.Errorf("aged SJF gave up its mean-latency advantage: aged %v, FIFO %v",
			byName["sjf-aged"].mean, byName["fifo"].mean)
	}

	// Fair share is not competing on either number: it reorders across tenants to
	// bound per-tenant consumption, and on a single-tenant-weighted workload like
	// this one that reordering can only cost latency. Logged rather than asserted,
	// because the property it exists for is tested in the fair-share tests.
	t.Logf("fair-share pays %v of mean wait over FIFO for its per-tenant bound",
		byName["fair-share"].mean-byName["fifo"].mean)
}
