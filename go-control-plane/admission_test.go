package loom

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

// acbQueue is a minimal FIFO queue for the admission and batcher tests.
//
// It exists so these tests depend only on the Queue contract rather than on
// another component's implementation: a test that fails because someone else's
// queue changed its ordering has told you nothing about admission.
type acbQueue struct {
	mu    sync.Mutex
	items []*RequestState
	limit int
}

func newACBQueue(limit int) *acbQueue { return &acbQueue{limit: limit} }

func (q *acbQueue) Push(st *RequestState) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) >= q.limit {
		return false
	}
	q.items = append(q.items, st)
	return true
}

func (q *acbQueue) Pop() *RequestState {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) == 0 {
		return nil
	}
	st := q.items[0]
	q.items = q.items[1:]
	return st
}

func (q *acbQueue) Remove(id RequestID) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	for i, st := range q.items {
		if st.ID == id {
			q.items = append(q.items[:i], q.items[i+1:]...)
			return true
		}
	}
	return false
}

func (q *acbQueue) DropExpired(now time.Time) []*RequestState {
	q.mu.Lock()
	defer q.mu.Unlock()
	var dropped []*RequestState
	kept := q.items[:0]
	for _, st := range q.items {
		if !st.Deadline.IsZero() && now.After(st.Deadline) {
			dropped = append(dropped, st)
			continue
		}
		kept = append(kept, st)
	}
	q.items = kept
	return dropped
}

func (q *acbQueue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.items)
}

func (q *acbQueue) Cap() int { return q.limit }

// Snapshot satisfies the read side of Queue. The fake is its own snapshot; these
// tests drive admission, which only ever reads.
func (q *acbQueue) Snapshot() QueueSnapshot { return q }

func (q *acbQueue) Each(fn func(st *RequestState) bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	for _, st := range q.items {
		if !fn(st) {
			return
		}
	}
}

func (q *acbQueue) TenantCount(t TenantID) int {
	q.mu.Lock()
	defer q.mu.Unlock()
	n := 0
	for _, st := range q.items {
		if st.Tenant == t {
			n++
		}
	}
	return n
}

func (q *acbQueue) TenantCounts() map[TenantID]int {
	q.mu.Lock()
	defer q.mu.Unlock()
	out := make(map[TenantID]int)
	for _, st := range q.items {
		out[st.Tenant]++
	}
	return out
}

func (q *acbQueue) OldestWait(now time.Time) time.Duration {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) == 0 {
		return 0
	}
	return now.Sub(q.items[0].QueuedAt)
}

var testNow = time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC)

func testWorker(id WorkerID, model ModelID, total, used, batch, active int) *WorkerState {
	return &WorkerState{
		ID:               id,
		Model:            model,
		TotalCacheBlocks: total,
		UsedCacheBlocks:  used,
		MaxBatchSize:     batch,
		ActiveSequences:  active,
		LastHeartbeat:    testNow,
	}
}

func testRequest(id RequestID, tenant TenantID, prompt, max int) *Request {
	return &Request{
		ID:              id,
		Tenant:          tenant,
		Model:           "m1",
		PromptTokens:    prompt,
		MaxOutputTokens: max,
		ArrivalTime:     testNow,
	}
}

func testController(mode ReservationMode, quotas map[TenantID]TenantQuota) *QuotaAdmissionController {
	est := NewCostEstimator(testPolicy(16, 256), FractionOfMax{Fraction: 0.25, Floor: 16})
	return NewQuotaAdmissionController(est, mode, quotas)
}

func TestAdmitAcceptsWhenCapacityExists(t *testing.T) {
	c := testController(ReservePeak, nil)
	workers := []*WorkerState{testWorker("w1", "m1", 256, 0, 8, 0)}
	q := newACBQueue(4)

	dec, reason, cost := c.Admit(testRequest("r1", "t1", 48, 128), workers, q, testNow)
	if dec != AdmitAccepted || reason != RejectNone {
		t.Fatalf("decision = %v/%v, want accepted/none", dec, reason)
	}
	if cost.PeakCacheBlocks != 11 {
		t.Fatalf("peak blocks = %d, want 11 for 48+128 tokens at 16/block", cost.PeakCacheBlocks)
	}
	if cost.ExpectedCacheBlocks >= cost.PeakCacheBlocks {
		t.Fatalf("expected blocks %d must be below peak %d under a fractional length model",
			cost.ExpectedCacheBlocks, cost.PeakCacheBlocks)
	}
	if cost.EstimatedServiceNanos <= 0 {
		t.Fatal("service estimate must be positive")
	}
}

func TestAdmitQueuesWhenWorkersAreFullButCapable(t *testing.T) {
	c := testController(ReservePeak, nil)
	// Full right now, large enough once something finishes: that is "not now", not "no".
	workers := []*WorkerState{testWorker("w1", "m1", 256, 250, 8, 8)}
	q := newACBQueue(4)

	dec, reason, _ := c.Admit(testRequest("r1", "t1", 48, 128), workers, q, testNow)
	if dec != AdmitQueued || reason != RejectNone {
		t.Fatalf("decision = %v/%v, want queued/none", dec, reason)
	}
}

func TestAdmitRejectModelUnavailable(t *testing.T) {
	c := testController(ReservePeak, nil)
	workers := []*WorkerState{testWorker("w1", "other-model", 256, 0, 8, 0)}

	dec, reason, _ := c.Admit(testRequest("r1", "t1", 48, 128), workers, newACBQueue(4), testNow)
	if dec != AdmitRejected || reason != RejectModelUnavailable {
		t.Fatalf("decision = %v/%v, want rejected/model_unavailable", dec, reason)
	}

	// An empty fleet is the same answer: there is no event pending that would make
	// this servable, so queueing would be waiting for nothing.
	dec, reason, _ = c.Admit(testRequest("r2", "t1", 48, 128), nil, newACBQueue(4), testNow)
	if dec != AdmitRejected || reason != RejectModelUnavailable {
		t.Fatalf("empty fleet: decision = %v/%v, want rejected/model_unavailable", dec, reason)
	}
}

func TestAdmitRejectTooLargeIsPermanentAndNeverQueued(t *testing.T) {
	c := testController(ReservePeak, nil)
	// 64 blocks of capacity, and a request whose peak is 4097 tokens of KV: it does
	// not fit on an empty machine, so no amount of waiting changes the answer.
	workers := []*WorkerState{testWorker("w1", "m1", 64, 0, 8, 0)}
	q := newACBQueue(16)

	req := testRequest("big", "t1", 1, 4096)
	dec, reason, cost := c.Admit(req, workers, q, testNow)
	if dec != AdmitRejected || reason != RejectTooLarge {
		t.Fatalf("decision = %v/%v, want rejected/too_large", dec, reason)
	}
	if cost.PeakCacheBlocks <= workers[0].TotalCacheBlocks {
		t.Fatalf("test is not exercising the case: peak %d fits in %d", cost.PeakCacheBlocks, workers[0].TotalCacheBlocks)
	}
	if q.Len() != 0 {
		t.Fatalf("a too-large request was queued (len=%d); it must be refused permanently", q.Len())
	}

	// An empty queue and an idle fleet do not rescue it either, which is what makes
	// the refusal permanent rather than a capacity complaint in disguise.
	for i := 0; i < 5; i++ {
		workers[0].UsedCacheBlocks = 0
		workers[0].ActiveSequences = 0
		if dec, reason, _ = c.Admit(req, workers, q, testNow.Add(time.Duration(i)*time.Minute)); dec != AdmitRejected || reason != RejectTooLarge {
			t.Fatalf("retry %d: decision = %v/%v, want rejected/too_large", i, dec, reason)
		}
	}
	if q.Len() != 0 {
		t.Fatalf("queue grew to %d across retries", q.Len())
	}
	if n, b, _ := c.TenantUsage("t1"); n != 0 || b != 0 {
		t.Fatalf("refused request left a quota charge: requests=%d blocks=%d", n, b)
	}
}

func TestAdmitTooLargeIsJudgedOnPeakNotExpected(t *testing.T) {
	// Admitted on its expected footprint this request fits, but it can provably
	// never reach its declared ceiling on this hardware. Letting it in means
	// preempting it after it has already produced most of its output.
	c := testController(ReserveExpected, nil)
	workers := []*WorkerState{testWorker("w1", "m1", 40, 0, 8, 0)}

	dec, reason, cost := c.Admit(testRequest("r1", "t1", 16, 1024), workers, newACBQueue(4), testNow)
	if cost.ExpectedCacheBlocks > 40 {
		t.Fatalf("test setup wrong: expected blocks %d already exceeds capacity", cost.ExpectedCacheBlocks)
	}
	if dec != AdmitRejected || reason != RejectTooLarge {
		t.Fatalf("decision = %v/%v, want rejected/too_large (peak %d vs capacity 40)", dec, reason, cost.PeakCacheBlocks)
	}
}

func TestAdmitRejectTenantQuotaOnEachDimension(t *testing.T) {
	workers := []*WorkerState{testWorker("w1", "m1", 4096, 0, 64, 0)}

	t.Run("concurrent requests", func(t *testing.T) {
		c := testController(ReservePeak, map[TenantID]TenantQuota{"t1": {MaxConcurrentRequests: 2}})
		q := newACBQueue(16)
		for i := 0; i < 2; i++ {
			if dec, _, _ := c.Admit(testRequest(RequestID(fmt.Sprintf("r%d", i)), "t1", 48, 128), workers, q, testNow); dec != AdmitAccepted {
				t.Fatalf("request %d: decision = %v, want accepted", i, dec)
			}
		}
		dec, reason, _ := c.Admit(testRequest("r2", "t1", 48, 128), workers, q, testNow)
		if dec != AdmitRejected || reason != RejectTenantQuota {
			t.Fatalf("decision = %v/%v, want rejected/tenant_quota", dec, reason)
		}
		// Quota is a hard bound, not a scheduling preference: the machine is idle and the
		// answer is still no.
		if workers[0].FreeBlocks() != 4096 {
			t.Fatal("test setup wrong: the worker should be idle")
		}
		if q.Len() != 0 {
			t.Fatalf("an over-quota request was queued (len=%d)", q.Len())
		}
		// Another tenant is unaffected, because the ceiling is per tenant.
		if dec, _, _ := c.Admit(testRequest("other", "t2", 48, 128), workers, q, testNow); dec != AdmitAccepted {
			t.Fatalf("second tenant decision = %v, want accepted", dec)
		}
		// And the quota frees up when a request finishes.
		c.Complete("r0")
		if dec, _, _ := c.Admit(testRequest("r3", "t1", 48, 128), workers, q, testNow); dec != AdmitAccepted {
			t.Fatalf("after completion decision = %v, want accepted", dec)
		}
	})

	t.Run("cache blocks", func(t *testing.T) {
		c := testController(ReservePeak, map[TenantID]TenantQuota{"t1": {MaxCacheBlocks: 20}})
		q := newACBQueue(16)
		if dec, _, _ := c.Admit(testRequest("a", "t1", 48, 128), workers, q, testNow); dec != AdmitAccepted {
			t.Fatalf("first: decision = %v, want accepted", dec)
		}
		if _, b, _ := c.TenantUsage("t1"); b != 11 {
			t.Fatalf("tenant blocks = %d, want 11", b)
		}
		dec, reason, _ := c.Admit(testRequest("b", "t1", 48, 128), workers, q, testNow)
		if dec != AdmitRejected || reason != RejectTenantQuota {
			t.Fatalf("decision = %v/%v, want rejected/tenant_quota (22 > 20 blocks)", dec, reason)
		}
	})

	t.Run("tokens per second", func(t *testing.T) {
		c := testController(ReservePeak, map[TenantID]TenantQuota{"t1": {MaxTokensPerSecond: 250}})
		q := newACBQueue(16)
		// The default decode model is 5ms/token, so one request demands ~200 tok/s.
		if dec, _, _ := c.Admit(testRequest("a", "t1", 48, 128), workers, q, testNow); dec != AdmitAccepted {
			t.Fatalf("first: decision = %v, want accepted", dec)
		}
		if _, _, rate := c.TenantUsage("t1"); rate < 100 || rate > 200 {
			t.Fatalf("committed rate = %.1f tok/s, outside the expected range", rate)
		}
		dec, reason, _ := c.Admit(testRequest("b", "t1", 48, 128), workers, q, testNow)
		if dec != AdmitRejected || reason != RejectTenantQuota {
			t.Fatalf("decision = %v/%v, want rejected/tenant_quota", dec, reason)
		}
	})
}

func TestAdmitQuotaIsAHardBoundUnderConcurrentSubmissions(t *testing.T) {
	const quota = 10
	const submitters = 200
	c := testController(ReservePeak, map[TenantID]TenantQuota{"t1": {MaxConcurrentRequests: quota}})
	workers := []*WorkerState{testWorker("w1", "m1", 1<<20, 0, 1<<20, 0)}
	q := newACBQueue(submitters)

	var mu sync.Mutex
	admitted := 0
	var wg sync.WaitGroup
	for i := 0; i < submitters; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			dec, reason, _ := c.Admit(testRequest(RequestID(fmt.Sprintf("r%d", i)), "t1", 48, 128), workers, q, testNow)
			if dec == AdmitRejected && reason != RejectTenantQuota {
				t.Errorf("request %d rejected for %v, want tenant_quota", i, reason)
				return
			}
			if dec != AdmitRejected {
				mu.Lock()
				admitted++
				mu.Unlock()
			}
		}(i)
	}
	wg.Wait()

	if admitted != quota {
		t.Fatalf("%d requests admitted under a quota of %d; the bound is not hard", admitted, quota)
	}
	if n, _, _ := c.TenantUsage("t1"); n != quota {
		t.Fatalf("tenant usage = %d, want %d", n, quota)
	}
}

func TestAdmitRejectDeadlineInfeasibleUpFront(t *testing.T) {
	c := testController(ReservePeak, nil)
	workers := []*WorkerState{testWorker("w1", "m1", 4096, 0, 64, 0)}
	q := newACBQueue(16)

	// 256 output tokens at 5ms each is about 1.3s of decode. A 100ms deadline cannot
	// be met even on a completely idle machine, so running it would spend capacity
	// producing something nobody will use.
	req := testRequest("late", "t1", 48, 256)
	req.Deadline = testNow.Add(100 * time.Millisecond)
	dec, reason, cost := c.Admit(req, workers, q, testNow)
	if dec != AdmitRejected || reason != RejectDeadlineInfeasible {
		t.Fatalf("decision = %v/%v, want rejected/deadline_infeasible (service %v)",
			dec, reason, time.Duration(cost.EstimatedServiceNanos))
	}
	if q.Len() != 0 {
		t.Fatalf("deadline-infeasible request was queued (len=%d)", q.Len())
	}
	if n, b, _ := c.TenantUsage("t1"); n != 0 || b != 0 {
		t.Fatalf("refused request left a quota charge: requests=%d blocks=%d", n, b)
	}

	// A generous deadline on the same request is admitted, which is what makes this
	// a feasibility check rather than a deadline ban.
	req2 := testRequest("ok", "t1", 48, 256)
	req2.Deadline = testNow.Add(30 * time.Second)
	if dec, reason, _ = c.Admit(req2, workers, q, testNow); dec != AdmitAccepted {
		t.Fatalf("feasible deadline: decision = %v/%v, want accepted", dec, reason)
	}
}

func TestAdmitDeadlineAccountsForQueueWait(t *testing.T) {
	c := testController(ReservePeak, nil)
	// Full worker, so the request has to wait behind the queue's head.
	workers := []*WorkerState{testWorker("w1", "m1", 256, 256, 8, 8)}
	q := newACBQueue(16)
	head := &RequestState{Request: *testRequest("head", "t1", 48, 128), QueuedAt: testNow.Add(-5 * time.Second)}
	q.Push(head)

	req := testRequest("r1", "t1", 48, 16)
	req.Deadline = testNow.Add(2 * time.Second)
	dec, reason, _ := c.Admit(req, workers, q, testNow)
	if dec != AdmitRejected || reason != RejectDeadlineInfeasible {
		t.Fatalf("decision = %v/%v, want rejected/deadline_infeasible: the 5s head-of-line wait alone exceeds the deadline",
			dec, reason)
	}
}

func TestAdmitRejectNoCapacityWhenOnlyBigWorkerIsDraining(t *testing.T) {
	c := testController(ReservePeak, nil)
	big := testWorker("big", "m1", 256, 0, 8, 0)
	big.Draining = true
	small := testWorker("small", "m1", 8, 0, 8, 0)
	q := newACBQueue(16)

	// The fleet contains hardware big enough, so this is not RejectTooLarge; but no
	// worker accepting work could ever hold it, so queueing it would be waiting for
	// the fleet to change shape rather than for capacity to free up.
	dec, reason, _ := c.Admit(testRequest("r1", "t1", 48, 256), []*WorkerState{big, small}, q, testNow)
	if dec != AdmitRejected || reason != RejectNoCapacity {
		t.Fatalf("decision = %v/%v, want rejected/no_capacity", dec, reason)
	}
	if q.Len() != 0 {
		t.Fatalf("request was queued (len=%d) when no live worker can ever hold it", q.Len())
	}

	// Stop draining and the same request becomes a queueing problem rather than a
	// refusal, which is the distinction the two reasons exist to draw.
	big.Draining = false
	big.UsedCacheBlocks = 256
	big.ActiveSequences = 8
	if dec, reason, _ = c.Admit(testRequest("r2", "t1", 48, 256), []*WorkerState{big, small}, q, testNow); dec != AdmitQueued {
		t.Fatalf("decision = %v/%v, want queued once the big worker accepts work", dec, reason)
	}
}

func TestAdmitRejectQueueFull(t *testing.T) {
	c := testController(ReservePeak, nil)
	workers := []*WorkerState{testWorker("w1", "m1", 256, 256, 8, 8)}
	q := newACBQueue(1)
	q.Push(&RequestState{Request: *testRequest("head", "t1", 48, 128), QueuedAt: testNow})

	dec, reason, _ := c.Admit(testRequest("r1", "t1", 48, 128), workers, q, testNow)
	if dec != AdmitRejected || reason != RejectQueueFull {
		t.Fatalf("decision = %v/%v, want rejected/queue_full", dec, reason)
	}
	if n, b, _ := c.TenantUsage("t1"); n != 0 || b != 0 {
		t.Fatalf("refused request left a quota charge: requests=%d blocks=%d", n, b)
	}
}

func TestAdmitReservationModeChangesTheAnswer(t *testing.T) {
	// The same request against the same machine, decided differently by the two
	// reservation modes. This is the peak-versus-expected tradeoff made visible:
	// ReservePeak refuses capacity it would probably not have needed, ReserveExpected
	// accepts a request it may later have to preempt.
	req := testRequest("r1", "t1", 48, 512)
	workers := func() []*WorkerState { return []*WorkerState{testWorker("w1", "m1", 36, 20, 8, 1)} }

	peak := testController(ReservePeak, nil)
	decPeak, _, cost := peak.Admit(req, workers(), newACBQueue(8), testNow)

	expected := testController(ReserveExpected, nil)
	decExp, _, _ := expected.Admit(req, workers(), newACBQueue(8), testNow)

	t.Logf("peak %d blocks -> %v; expected %d blocks -> %v (16 free)",
		cost.PeakCacheBlocks, decPeak, cost.ExpectedCacheBlocks, decExp)
	if decPeak != AdmitQueued {
		t.Fatalf("peak mode decision = %v, want queued (needs %d of 16 free)", decPeak, cost.PeakCacheBlocks)
	}
	if decExp != AdmitAccepted {
		t.Fatalf("expected mode decision = %v, want accepted (needs %d of 16 free)", decExp, cost.ExpectedCacheBlocks)
	}
}

func TestOutputLengthModels(t *testing.T) {
	req := testRequest("r1", "t1", 0, 400)

	frac := FractionOfMax{Fraction: 0.25, Floor: 8}
	if got := frac.ExpectedTokens(req); got != 100 {
		t.Fatalf("fraction model = %d, want 100", got)
	}
	// The floor exists because a prediction of zero blocks guarantees a growth
	// failure on the first decode step.
	if got := frac.ExpectedTokens(testRequest("r2", "t1", 0, 4)); got != 4 {
		t.Fatalf("fraction model clamped to %d, want the 4-token ceiling", got)
	}

	// A heavy tail: the mean is 164, the median is 32. A quantile model is explicit
	// about which part of the distribution it is planning for.
	observed := []int{16, 24, 32, 32, 40, 48, 900, 1024}
	median := NewQuantileLengthModel(observed, 0.5)
	if got := median.ExpectedTokens(req); got != 32 {
		t.Fatalf("median model = %d, want 32", got)
	}
	p99 := NewQuantileLengthModel(observed, 0.99)
	if got := p99.ExpectedTokens(req); got != 400 {
		t.Fatalf("p99 model = %d, want 400 (clamped to the declared ceiling)", got)
	}
	if median.Name() == p99.Name() && median.ExpectedTokens(req) == p99.ExpectedTokens(req) {
		t.Fatal("the quantile choice must change the prediction")
	}
}

func TestReconcileReportWinsAndDriftIsSignedBothWays(t *testing.T) {
	c := testController(ReservePeak, nil)
	w := testWorker("w1", "m1", 256, 120, 8, 4)

	// Over-reservation: the control plane thinks 120 blocks are held, the worker
	// measures 90. The report wins and the positive drift says capacity is idle.
	drift := c.Reconcile(w, 90, 3)
	if drift != 30 {
		t.Fatalf("drift = %d, want +30", drift)
	}
	if w.UsedCacheBlocks != 90 || w.ActiveSequences != 3 {
		t.Fatalf("report did not win: used=%d active=%d", w.UsedCacheBlocks, w.ActiveSequences)
	}
	if w.FreeBlocks() != 166 {
		t.Fatalf("free blocks = %d, want 166 after reconciliation", w.FreeBlocks())
	}

	// Under-reservation: the worker is fuller than the estimate, so the next
	// admission would have overcommitted. Negative drift must stay negative.
	drift = c.Reconcile(w, 200, 7)
	if drift != -110 {
		t.Fatalf("drift = %d, want -110", drift)
	}
	if w.UsedCacheBlocks != 200 {
		t.Fatalf("used = %d, want 200", w.UsedCacheBlocks)
	}

	// Agreement produces zero, which is what a healthy estimator looks like.
	if drift = c.Reconcile(w, 200, 7); drift != 0 {
		t.Fatalf("drift = %d, want 0 when estimate and report agree", drift)
	}

	m := c.Metrics()
	if m.Reconciles != 3 {
		t.Fatalf("reconciles = %d, want 3", m.Reconciles)
	}
	if m.ReconcileDriftLast != 0 {
		t.Fatalf("last drift = %d, want 0", m.ReconcileDriftLast)
	}
	if m.DriftSum != -80 {
		t.Fatalf("signed drift sum = %d, want -80 (+30-110+0)", m.DriftSum)
	}
	if m.DriftAbsSum != 140 {
		t.Fatalf("absolute drift sum = %d, want 140", m.DriftAbsSum)
	}
	if m.DriftMaxOver != 30 || m.DriftMaxUnder != -110 {
		t.Fatalf("drift extremes = +%d/%d, want +30/-110", m.DriftMaxOver, m.DriftMaxUnder)
	}
}

func TestReconcileCorrectsAdmissionDecisions(t *testing.T) {
	// Reconciliation is not a metric for its own sake: it changes what is admitted.
	// Before the report the controller believes the worker is nearly full and queues;
	// after the report it knows there is room and accepts.
	c := testController(ReservePeak, nil)
	w := testWorker("w1", "m1", 64, 60, 8, 4)
	q := newACBQueue(8)

	if dec, _, _ := c.Admit(testRequest("r1", "t1", 48, 128), []*WorkerState{w}, q, testNow); dec != AdmitQueued {
		t.Fatalf("pre-reconcile decision = %v, want queued", dec)
	}
	if drift := c.Reconcile(w, 10, 1); drift != 50 {
		t.Fatalf("drift = %d, want +50", drift)
	}
	if dec, _, _ := c.Admit(testRequest("r2", "t1", 48, 128), []*WorkerState{w}, q, testNow); dec != AdmitAccepted {
		t.Fatalf("post-reconcile decision = %v, want accepted", dec)
	}
}

func TestMetricsCountEveryOutcome(t *testing.T) {
	c := testController(ReservePeak, nil)
	workers := []*WorkerState{testWorker("w1", "m1", 64, 0, 8, 0)}

	c.Admit(testRequest("a", "t1", 48, 128), workers, newACBQueue(8), testNow)
	c.Admit(testRequest("b", "t1", 1, 8192), workers, newACBQueue(8), testNow)
	full := newACBQueue(0)
	workers[0].UsedCacheBlocks = 64
	workers[0].ActiveSequences = 8
	c.Admit(testRequest("c", "t1", 48, 128), workers, full, testNow)

	m := c.Metrics()
	if m.Accepted != 1 || m.Rejected != 2 {
		t.Fatalf("accepted=%d rejected=%d, want 1/2", m.Accepted, m.Rejected)
	}
	if m.ByReason[RejectTooLarge] != 1 || m.ByReason[RejectQueueFull] != 1 {
		t.Fatalf("reason counts = %v", m.ByReason)
	}
	// The snapshot is a copy; mutating it must not touch the controller.
	m.ByReason[RejectTooLarge] = 99
	if c.Metrics().ByReason[RejectTooLarge] != 1 {
		t.Fatal("Metrics() leaked its internal reason map")
	}
}

func TestAdmitIsRaceFreeAcrossTenants(t *testing.T) {
	c := testController(ReserveExpected, map[TenantID]TenantQuota{
		"t1": {MaxConcurrentRequests: 5, MaxCacheBlocks: 200},
		"t2": {MaxCacheBlocks: 60},
	})
	q := newACBQueue(1024)

	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			tenant := TenantID("t1")
			if i%2 == 0 {
				tenant = "t2"
			}
			id := RequestID(fmt.Sprintf("%s-%d", tenant, i))
			// Each goroutine carries its own WorkerState. WorkerState is a plain struct with
			// no internal synchronization, so whoever owns it — the registry, in the real
			// system — serializes reads and writes to it; the controller's own state is the
			// part that has to be safe under concurrent admission, and that is what is
			// shared here.
			w := testWorker(WorkerID(fmt.Sprintf("w%d", i)), "m1", 1<<16, 0, 1<<16, 0)
			c.Admit(testRequest(id, tenant, 48, 128), []*WorkerState{w}, q, testNow)
			if i%3 == 0 {
				c.Complete(id)
			}
			c.Reconcile(w, i, i)
		}(i)
	}
	wg.Wait()

	n1, b1, _ := c.TenantUsage("t1")
	if n1 > 5 || b1 > 200 {
		t.Fatalf("t1 exceeded its quota: requests=%d blocks=%d", n1, b1)
	}
	_, b2, _ := c.TenantUsage("t2")
	if b2 > 60 {
		t.Fatalf("t2 exceeded its block quota: %d", b2)
	}
}
