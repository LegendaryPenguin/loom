package loom

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeClient is an in-process worker.
//
// In-process rather than over a socket because what these tests are about is the
// scheduler's accounting across a worker's lifecycle, and a real socket would add
// a second source of flakiness to assertions about capacity. The socket itself is
// covered in workerconn_test.go.
type fakeClient struct {
	id WorkerID

	mu         sync.Mutex
	events     chan WorkerEvent
	dispatched []RequestID
	batches    [][]RequestID
	cancelled  []RequestID
	drains     int
	closed     bool
	failures   map[RequestID]Failure
	dispatchFn func(batch []*RequestState) error
}

func newFakeClient(id WorkerID) *fakeClient {
	return &fakeClient{id: id, events: make(chan WorkerEvent, 256), failures: make(map[RequestID]Failure)}
}

func (f *fakeClient) Dispatch(ctx context.Context, batch []*RequestState) error {
	f.mu.Lock()
	fn := f.dispatchFn
	f.mu.Unlock()
	if fn != nil {
		if err := fn(batch); err != nil {
			return err
		}
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	ids := make([]RequestID, 0, len(batch))
	for _, st := range batch {
		ids = append(ids, st.ID)
		f.dispatched = append(f.dispatched, st.ID)
	}
	f.batches = append(f.batches, ids)
	return nil
}

func (f *fakeClient) Cancel(ctx context.Context, id RequestID) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.cancelled = append(f.cancelled, id)
	return nil
}

func (f *fakeClient) Drain(ctx context.Context) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.drains++
	return nil
}

func (f *fakeClient) Events() <-chan WorkerEvent { return f.events }

func (f *fakeClient) Close() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.closed {
		f.closed = true
		close(f.events)
	}
	return nil
}

func (f *fakeClient) TakeFailure(id RequestID) (Failure, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	fl, ok := f.failures[id]
	delete(f.failures, id)
	return fl, ok
}

func (f *fakeClient) setFailure(fl Failure) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.failures[fl.Request] = fl
}

func (f *fakeClient) emit(ev WorkerEvent) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closed {
		return
	}
	if ev.Worker == "" {
		ev.Worker = f.id
	}
	if ev.Timestamp.IsZero() {
		ev.Timestamp = time.Now()
	}
	f.events <- ev
}

func (f *fakeClient) sawDispatch(id RequestID) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, d := range f.dispatched {
		if d == id {
			return true
		}
	}
	return false
}

func (f *fakeClient) dispatchCount(id RequestID) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	n := 0
	for _, d := range f.dispatched {
		if d == id {
			n++
		}
	}
	return n
}

func (f *fakeClient) cancels() []RequestID {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]RequestID(nil), f.cancelled...)
}

func (f *fakeClient) drainCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.drains
}

// harness is a wired control plane with the pieces a test needs to poke at.
type harness struct {
	t     *testing.T
	sched *Scheduler
	queue *BoundedQueue
	reg   *Registry
	adm   *QuotaAdmissionController
	met   *Metrics
	est   *CostEstimator

	mu     sync.Mutex
	caches map[WorkerID]KVCache
}

type harnessOpts struct {
	queueCapacity int
	blocks        int
	batchSize     int
	quotas        map[TenantID]TenantQuota
	retry         RetryPolicy
	mode          ReservationMode
	batcher       string
	batchWindow   time.Duration
	heartbeat     time.Duration
	hbTimeout     time.Duration
}

func newHarness(t *testing.T, opts harnessOpts) *harness {
	t.Helper()
	if opts.queueCapacity == 0 {
		opts.queueCapacity = 64
	}
	if opts.blocks == 0 {
		opts.blocks = 64
	}
	if opts.batchSize == 0 {
		opts.batchSize = 4
	}
	if opts.retry.MaxAttempts == 0 {
		opts.retry = DefaultRetryPolicy()
	}
	if opts.heartbeat == 0 {
		// Long enough that no test is reaped by accident; the reaping tests set their
		// own timeout explicitly.
		opts.heartbeat = 10 * time.Second
	}

	h := &harness{t: t, caches: make(map[WorkerID]KVCache)}
	h.met = NewMetrics("loom")
	h.reg = NewRegistry(RegistryConfig{HeartbeatInterval: opts.heartbeat, Retry: opts.retry})
	h.queue = NewBoundedQueue(opts.queueCapacity, NewFIFOPolicy())
	h.est = NewCostEstimator(CachePolicy{TokensPerBlock: 16}, FractionOfMax{Fraction: 0.5, Floor: 1})
	h.adm = NewQuotaAdmissionController(h.est, opts.mode, opts.quotas)

	h.sched = NewScheduler(SchedulerConfig{
		Queue: h.queue, Registry: h.reg, Admission: h.adm, Metrics: h.met,
		Tick:             time.Millisecond,
		HeartbeatTimeout: opts.hbTimeout,
		DispatchTimeout:  time.Second,
		NewCache: func(w *WorkerState) KVCache {
			c := NewPagedCache(CachePolicy{TokensPerBlock: 16, TotalBlocks: w.TotalCacheBlocks})
			h.mu.Lock()
			h.caches[w.ID] = c
			h.mu.Unlock()
			return c
		},
		NewBatcher: func(w *WorkerState, cache KVCache) Batcher {
			if opts.batcher == BatcherStatic {
				return NewStaticBatcher(opts.batchWindow, h.est, opts.mode, cache)
			}
			return NewContinuousBatcher(opts.batchWindow, h.est, opts.mode, cache, PreemptNewest)
		},
	})
	h.sched.Start()
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = h.sched.Shutdown(ctx)
	})
	return h
}

func (h *harness) addWorker(id WorkerID, model ModelID, blocks, batchSize int) *fakeClient {
	h.t.Helper()
	fc := newFakeClient(id)
	ws := &WorkerState{ID: id, Model: model, TotalCacheBlocks: blocks, MaxBatchSize: batchSize}
	if err := h.sched.AddWorker(ws, fc); err != nil {
		h.t.Fatalf("add worker %s: %v", id, err)
	}
	go h.sched.ConsumeEvents(id, fc)
	return fc
}

func (h *harness) cache(id WorkerID) KVCache {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.caches[id]
}

func (h *harness) submit(id RequestID, tenant TenantID, model ModelID, prompt int, out int, deadline time.Duration) (*Handle, AdmitDecision, RejectReason) {
	h.t.Helper()
	req := &Request{
		ID: id, Tenant: tenant, Model: model,
		PromptTokens: prompt, MaxOutputTokens: out,
		ArrivalTime: time.Now(),
	}
	if deadline > 0 {
		req.Deadline = time.Now().Add(deadline)
	}
	return h.sched.Submit(req, "prompt text for "+string(id))
}

func waitFor(t *testing.T, within time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(within)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func waitDone(t *testing.T, h *Handle, within time.Duration) Result {
	t.Helper()
	select {
	case <-h.Done():
		return h.Result()
	case <-time.After(within):
		t.Fatalf("request %s never reached a terminal state (status %s)", h.ID(), h.Result().Status)
	}
	return Result{}
}

// metricValue reads one series out of the exposition output, which is the only
// public surface the metrics have — and reading it here also proves the series is
// actually exported rather than only incremented.
func metricValue(t *testing.T, m *Metrics, series string) float64 {
	t.Helper()
	var b strings.Builder
	if err := m.Export(&b); err != nil {
		t.Fatalf("export metrics: %v", err)
	}
	for _, line := range strings.Split(b.String(), "\n") {
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}
		name, value, ok := strings.Cut(line, " ")
		if !ok || name != series {
			continue
		}
		f, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
		if err != nil {
			t.Fatalf("parse %q: %v", line, err)
		}
		return f
	}
	return 0
}

func drainChunks(h *Handle) string {
	var b strings.Builder
	for c := range h.Chunks() {
		b.WriteString(c.Text)
	}
	return b.String()
}

func TestSchedulerRequestFlowsSubmitToComplete(t *testing.T) {
	t.Parallel()
	h := newHarness(t, harnessOpts{})
	fc := h.addWorker("w-1", "m", 64, 4)

	handle, dec, reason := h.submit("r-1", "t-1", "m", 32, 16, 0)
	if handle == nil {
		t.Fatalf("submit refused: %s/%s", dec, reason)
	}
	waitFor(t, 2*time.Second, "dispatch", func() bool { return fc.sawDispatch("r-1") })

	if w, ok := h.reg.WorkerOf("r-1"); !ok || w != "w-1" {
		t.Fatalf("request not assigned to the worker it was dispatched to: %s %v", w, ok)
	}
	if used := h.cache("w-1").Used(); used <= 0 {
		t.Fatal("dispatch reserved no cache blocks")
	}

	text := make(chan string, 1)
	go func() { text <- drainChunks(handle) }()

	fc.emit(WorkerEvent{Kind: EventFirstToken, Request: "r-1", Tokens: 1})
	fc.emit(WorkerEvent{Kind: EventToken, Request: "r-1", Tokens: 1, Blocks: 3, Err: "Hello"})
	fc.emit(WorkerEvent{Kind: EventToken, Request: "r-1", Tokens: 2, Blocks: 4, Err: " world"})
	fc.emit(WorkerEvent{Kind: EventCompleted, Request: "r-1", Tokens: 2, Err: "eos"})

	res := waitDone(t, handle, 2*time.Second)
	if res.Status != StatusCompleted {
		t.Fatalf("status %s (err %q)", res.Status, res.Err)
	}
	if res.StopReason != "eos" || res.GeneratedTokens != 2 {
		t.Fatalf("result %+v", res)
	}
	if res.FirstTokenAt.IsZero() {
		t.Fatal("first token time was never recorded, so time-to-first-token is unmeasurable")
	}
	if got := <-text; got != "Hello world" {
		t.Fatalf("streamed text %q", got)
	}

	waitFor(t, 2*time.Second, "cache release", func() bool { return h.cache("w-1").Used() == 0 })
	if reqs, _, _ := h.adm.TenantUsage("t-1"); reqs != 0 {
		t.Fatalf("tenant still charged for a completed request: %d", reqs)
	}
	if n := metricValue(t, h.met, "loom_requests_completed_total"); n != 1 {
		t.Fatalf("completed counter %v", n)
	}
}

// TestSchedulerKilledWorkerReleasesReservations is the capacity-leak test.
//
// Capacity leaked on worker loss is invisible: every individual request looks
// fine, the accounting simply says the fleet is full while the accelerators sit
// idle. The assertion is therefore not "the requests were retried" but "the system
// still admits afterwards", which is the only thing that distinguishes a released
// reservation from a forgotten one.
func TestSchedulerKilledWorkerReleasesReservationsAndStillAdmits(t *testing.T) {
	t.Parallel()
	quota := map[TenantID]TenantQuota{"t-1": {MaxConcurrentRequests: 2, MaxCacheBlocks: 8}}
	h := newHarness(t, harnessOpts{quotas: quota, mode: ReserveExpected})
	fc1 := h.addWorker("w-1", "m", 32, 4)

	a, _, _ := h.submit("r-a", "t-1", "m", 32, 32, 0)
	b, _, _ := h.submit("r-b", "t-1", "m", 32, 32, 0)
	if a == nil || b == nil {
		t.Fatal("the two requests that fit the quota were refused")
	}
	waitFor(t, 2*time.Second, "both dispatched", func() bool {
		return fc1.sawDispatch("r-a") && fc1.sawDispatch("r-b")
	})
	if _, blocks, _ := h.adm.TenantUsage("t-1"); blocks == 0 {
		t.Fatal("the tenant was never charged, so there is no reservation to leak")
	}

	// A third request must be refused while the quota is genuinely consumed;
	// otherwise the test below proves nothing about release.
	if handle, _, reason := h.submit("r-c", "t-1", "m", 32, 32, 0); handle != nil {
		t.Fatalf("quota did not bind: third request accepted (%s)", reason)
	} else if reason != RejectTenantQuota {
		t.Fatalf("third request refused for the wrong reason: %s", reason)
	}

	// The worker dies with both sequences running.
	_ = fc1.Close()

	waitFor(t, 2*time.Second, "worker forgotten", func() bool {
		_, ok := h.reg.Get("w-1")
		return !ok
	})
	if n := metricValue(t, h.met, "loom_worker_failures_total"); n != 1 {
		t.Fatalf("worker failure counter %v", n)
	}
	if n := metricValue(t, h.met, "loom_reaped_requests_total"); n != 2 {
		t.Fatalf("reaped request counter %v", n)
	}
	waitFor(t, 2*time.Second, "assignments released", func() bool {
		_, ok := h.reg.WorkerOf("r-a")
		return !ok
	})

	// A replacement worker, and the re-queued work must run on it.
	fc2 := h.addWorker("w-2", "m", 32, 4)
	waitFor(t, 3*time.Second, "redispatch after worker loss", func() bool {
		return fc2.sawDispatch("r-a") && fc2.sawDispatch("r-b")
	})

	fc2.emit(WorkerEvent{Kind: EventCompleted, Request: "r-a", Tokens: 5, Err: "eos"})
	fc2.emit(WorkerEvent{Kind: EventCompleted, Request: "r-b", Tokens: 5, Err: "eos"})
	if res := waitDone(t, a, 3*time.Second); res.Status != StatusCompleted {
		t.Fatalf("r-a finished as %s: %q", res.Status, res.Err)
	}
	if res := waitDone(t, b, 3*time.Second); res.Status != StatusCompleted {
		t.Fatalf("r-b finished as %s: %q", res.Status, res.Err)
	}

	// The leak would show up here: every charge refunded means the quota is free
	// again, and a leaked one means this admission is refused forever.
	waitFor(t, 2*time.Second, "quota released", func() bool {
		reqs, blocks, _ := h.adm.TenantUsage("t-1")
		return reqs == 0 && blocks == 0
	})
	later, dec, reason := h.submit("r-d", "t-1", "m", 32, 32, 0)
	if later == nil {
		t.Fatalf("the system stopped admitting after a worker loss: %s/%s", dec, reason)
	}
	waitFor(t, 2*time.Second, "post-loss dispatch", func() bool { return fc2.sawDispatch("r-d") })
	if used := h.cache("w-2").Used(); used <= 0 {
		t.Fatal("post-loss dispatch reserved nothing, so the cache accounting is wrong")
	}
}

func TestSchedulerReapsWorkerThatStopsHeartbeating(t *testing.T) {
	t.Parallel()
	// One attempt, so the reaped request is failed rather than retried and the test
	// asserts on a terminal state instead of a race with a redispatch.
	h := newHarness(t, harnessOpts{
		retry:     RetryPolicy{MaxAttempts: 1},
		hbTimeout: 50 * time.Millisecond,
	})
	fc := h.addWorker("w-1", "m", 32, 2)

	handle, _, _ := h.submit("r-1", "t-1", "m", 32, 16, 0)
	if handle == nil {
		t.Fatal("submit refused")
	}
	waitFor(t, 2*time.Second, "dispatch", func() bool { return fc.sawDispatch("r-1") })

	// Silence. No heartbeat ever arrives, so the protocol's missed-heartbeat rule
	// must declare the worker dead and release what it held.
	res := waitDone(t, handle, 3*time.Second)
	if res.Status != StatusFailed {
		t.Fatalf("status %s", res.Status)
	}
	if n := metricValue(t, h.met, `loom_retries_total{action="none"}`); n != 1 {
		t.Fatalf("classifier verdict not recorded: %v", n)
	}
	waitFor(t, 2*time.Second, "quota released", func() bool {
		reqs, _, _ := h.adm.TenantUsage("t-1")
		return reqs == 0
	})
}

func TestSchedulerPreemptedRequestIsRequeuedAndCompletes(t *testing.T) {
	t.Parallel()
	h := newHarness(t, harnessOpts{})
	fc := h.addWorker("w-1", "m", 64, 4)

	handle, _, _ := h.submit("r-1", "t-1", "m", 32, 64, 0)
	if handle == nil {
		t.Fatal("submit refused")
	}
	waitFor(t, 2*time.Second, "first dispatch", func() bool { return fc.dispatchCount("r-1") == 1 })

	fc.emit(WorkerEvent{Kind: EventFirstToken, Request: "r-1", Tokens: 1})
	fc.emit(WorkerEvent{Kind: EventPreempted, Request: "r-1", Tokens: 31})

	// Re-queued, not failed: a preemption that fails the request turns a memory
	// shortage into an error, which is a far worse outcome than a slow response.
	waitFor(t, 3*time.Second, "redispatch after preemption", func() bool {
		return fc.dispatchCount("r-1") >= 2
	})
	if n := metricValue(t, h.met, "loom_requests_preempted_total"); n != 1 {
		t.Fatalf("preemption counter %v", n)
	}
	if res := handle.Result(); res.Status.Terminal() {
		t.Fatalf("a preempted request must not be terminal: %s", res.Status)
	}

	fc.emit(WorkerEvent{Kind: EventCompleted, Request: "r-1", Tokens: 64, Err: "eos"})
	res := waitDone(t, handle, 3*time.Second)
	if res.Status != StatusCompleted {
		t.Fatalf("status %s: %q", res.Status, res.Err)
	}
	if res.Attempts < 2 {
		t.Fatalf("attempts %d: the retry was not counted", res.Attempts)
	}
}

func TestSchedulerCancellationReleasesCapacityPromptly(t *testing.T) {
	t.Parallel()
	h := newHarness(t, harnessOpts{})
	fc := h.addWorker("w-1", "m", 64, 4)

	handle, _, _ := h.submit("r-1", "t-1", "m", 64, 128, 0)
	if handle == nil {
		t.Fatal("submit refused")
	}
	waitFor(t, 2*time.Second, "dispatch", func() bool { return fc.sawDispatch("r-1") })
	if h.cache("w-1").Used() == 0 {
		t.Fatal("nothing was reserved, so releasing it proves nothing")
	}

	handle.Cancel()

	res := waitDone(t, handle, 2*time.Second)
	if res.Status != StatusCancelled {
		t.Fatalf("status %s", res.Status)
	}
	// Promptly: at cancellation, not at the generation's natural end. The worker has
	// not reported anything yet and the blocks are already back.
	waitFor(t, time.Second, "blocks released at cancel", func() bool { return h.cache("w-1").Used() == 0 })
	if _, ok := h.reg.WorkerOf("r-1"); ok {
		t.Fatal("assignment survived cancellation")
	}
	waitFor(t, time.Second, "cancel sent to worker", func() bool { return len(fc.cancels()) == 1 })
	if reqs, _, _ := h.adm.TenantUsage("t-1"); reqs != 0 {
		t.Fatalf("tenant still charged after cancellation: %d", reqs)
	}

	// The worker's eventual confirmation must be a no-op rather than a second
	// release of the same blocks.
	fc.emit(WorkerEvent{Kind: EventCompleted, Request: "r-1", Tokens: 3, Err: "cancelled"})
	time.Sleep(20 * time.Millisecond)
	if used := h.cache("w-1").Used(); used != 0 {
		t.Fatalf("late confirmation disturbed the accounting: used=%d", used)
	}
	if n := metricValue(t, h.met, "loom_requests_cancelled_total"); n != 1 {
		t.Fatalf("cancellation counted %v times", n)
	}
}

func TestSchedulerDropsDeadlineExpiredRequestsWithoutDispatching(t *testing.T) {
	t.Parallel()
	// A single batch slot, occupied, so the second request has to wait — and its
	// deadline passes while it does.
	h := newHarness(t, harnessOpts{batchSize: 1})
	fc := h.addWorker("w-1", "m", 64, 1)

	first, _, _ := h.submit("r-hog", "t-1", "m", 16, 256, 0)
	if first == nil {
		t.Fatal("first submit refused")
	}
	waitFor(t, 2*time.Second, "hog dispatched", func() bool { return fc.sawDispatch("r-hog") })

	late, dec, reason := h.submit("r-late", "t-1", "m", 16, 2, 80*time.Millisecond)
	if late == nil {
		t.Fatalf("the deadlined request was refused at admission: %s/%s", dec, reason)
	}

	res := waitDone(t, late, 2*time.Second)
	if res.Status != StatusTimedOut {
		t.Fatalf("status %s", res.Status)
	}
	if fc.sawDispatch("r-late") {
		t.Fatal("an expired request was dispatched; serving it burns capacity for a result nobody wants")
	}
	if n := metricValue(t, h.met, "loom_requests_timed_out_total"); n != 1 {
		t.Fatalf("timeout counter %v", n)
	}
}

func TestSchedulerRecordsEstimateVersusActualDrift(t *testing.T) {
	t.Parallel()
	h := newHarness(t, harnessOpts{mode: ReservePeak})
	fc := h.addWorker("w-1", "m", 64, 4)

	handle, _, _ := h.submit("r-1", "t-1", "m", 32, 64, 0)
	if handle == nil {
		t.Fatal("submit refused")
	}
	waitFor(t, 2*time.Second, "dispatch", func() bool { return fc.sawDispatch("r-1") })

	estimate := h.cache("w-1").Used()
	if estimate < 2 {
		t.Fatalf("expected a multi-block reservation to drift against, got %d", estimate)
	}

	// The worker reports holding fewer blocks than the control plane reserved, which
	// is the over-reservation case: capacity sitting idle inside a reservation that
	// will never be used.
	reported := estimate - 1
	h.sched.Heartbeat(Heartbeat{Worker: "w-1", UsedCacheBlocks: reported, ActiveSequences: 1,
		TokensPerSecond: 120, At: time.Now()})

	waitFor(t, 2*time.Second, "reconcile", func() bool { return h.adm.Metrics().Reconciles > 0 })
	am := h.adm.Metrics()
	if am.ReconcileDriftLast != estimate-reported {
		t.Fatalf("drift %d, want %d", am.ReconcileDriftLast, estimate-reported)
	}
	if am.DriftMaxOver != estimate-reported {
		t.Fatalf("over-reservation excursion %d, want %d", am.DriftMaxOver, estimate-reported)
	}
	waitFor(t, 2*time.Second, "drift gauge", func() bool {
		return metricValue(t, h.met, "loom_cache_blocks_drift") == float64(estimate-reported)
	})
}

func TestSchedulerRetryRespectsTheClassifier(t *testing.T) {
	t.Parallel()

	t.Run("permanent failure is not retried", func(t *testing.T) {
		t.Parallel()
		h := newHarness(t, harnessOpts{})
		fc := h.addWorker("w-1", "m", 64, 4)

		handle, _, _ := h.submit("r-1", "t-1", "m", 32, 16, 0)
		if handle == nil {
			t.Fatal("submit refused")
		}
		waitFor(t, 2*time.Second, "dispatch", func() bool { return fc.sawDispatch("r-1") })

		// retryable=false on the wire: this request will fail identically everywhere,
		// so retrying it is pure waste and one bad request becomes a loop.
		fc.setFailure(Failure{Request: "r-1", Worker: "w-1", Transient: false, Err: "prompt exceeds context window"})
		fc.emit(WorkerEvent{Kind: EventFailed, Request: "r-1", Err: "prompt exceeds context window"})

		res := waitDone(t, handle, 2*time.Second)
		if res.Status != StatusFailed {
			t.Fatalf("status %s", res.Status)
		}
		if n := fc.dispatchCount("r-1"); n != 1 {
			t.Fatalf("a permanent failure was retried %d times", n-1)
		}
		if n := metricValue(t, h.met, `loom_retries_total{action="none"}`); n != 1 {
			t.Fatalf("do-not-retry verdict not recorded: %v", n)
		}
		waitFor(t, time.Second, "blocks released", func() bool { return h.cache("w-1").Used() == 0 })
	})

	t.Run("transient failure retries on the same worker", func(t *testing.T) {
		t.Parallel()
		h := newHarness(t, harnessOpts{})
		fc := h.addWorker("w-1", "m", 64, 4)

		handle, _, _ := h.submit("r-1", "t-1", "m", 32, 16, 0)
		if handle == nil {
			t.Fatal("submit refused")
		}
		waitFor(t, 2*time.Second, "dispatch", func() bool { return fc.sawDispatch("r-1") })

		fc.setFailure(Failure{Request: "r-1", Worker: "w-1", Transient: true, Err: "transient device error"})
		fc.emit(WorkerEvent{Kind: EventFailed, Request: "r-1", Err: "transient device error"})

		// The same worker, because it still holds the prompt's KV blocks: re-prefilling
		// elsewhere costs more than the retry saves.
		waitFor(t, 3*time.Second, "retry on the same worker", func() bool { return fc.dispatchCount("r-1") >= 2 })
		if n := metricValue(t, h.met, `loom_retries_total{action="same"}`); n != 1 {
			t.Fatalf("retry-same verdict not recorded: %v", n)
		}

		fc.emit(WorkerEvent{Kind: EventCompleted, Request: "r-1", Tokens: 4, Err: "eos"})
		if res := waitDone(t, handle, 2*time.Second); res.Status != StatusCompleted {
			t.Fatalf("status %s: %q", res.Status, res.Err)
		}
	})

	t.Run("a failure with no retryability report is treated as permanent", func(t *testing.T) {
		t.Parallel()
		h := newHarness(t, harnessOpts{})
		fc := h.addWorker("w-1", "m", 64, 4)

		handle, _, _ := h.submit("r-1", "t-1", "m", 32, 16, 0)
		if handle == nil {
			t.Fatal("submit refused")
		}
		waitFor(t, 2*time.Second, "dispatch", func() bool { return fc.sawDispatch("r-1") })

		// No setFailure: the worker said nothing about retryability. Guessing
		// "transient" is how one poison request becomes a retry storm.
		fc.emit(WorkerEvent{Kind: EventFailed, Request: "r-1", Err: "unqualified failure"})

		res := waitDone(t, handle, 2*time.Second)
		if res.Status != StatusFailed {
			t.Fatalf("status %s", res.Status)
		}
		if n := fc.dispatchCount("r-1"); n != 1 {
			t.Fatalf("an unqualified failure was retried %d times", n-1)
		}
	})

	t.Run("attempt cap stops an endlessly transient request", func(t *testing.T) {
		t.Parallel()
		h := newHarness(t, harnessOpts{retry: RetryPolicy{MaxAttempts: 2}})
		fc := h.addWorker("w-1", "m", 64, 4)

		handle, _, _ := h.submit("r-1", "t-1", "m", 32, 16, 0)
		if handle == nil {
			t.Fatal("submit refused")
		}
		// Fail every attempt the scheduler makes. The cap, not the test, decides how
		// many there are.
		for i := 0; i < 5 && !handle.Result().Status.Terminal(); i++ {
			attempt := i + 1
			waitFor(t, 3*time.Second, fmt.Sprintf("dispatch %d or a terminal state", attempt), func() bool {
				return fc.dispatchCount("r-1") >= attempt || handle.Result().Status.Terminal()
			})
			if handle.Result().Status.Terminal() {
				break
			}
			fc.setFailure(Failure{Request: "r-1", Worker: "w-1", Transient: true, Err: "flaky"})
			fc.emit(WorkerEvent{Kind: EventFailed, Request: "r-1", Err: "flaky"})
		}
		res := waitDone(t, handle, 3*time.Second)
		if res.Status != StatusFailed {
			t.Fatalf("status %s", res.Status)
		}
		if n := fc.dispatchCount("r-1"); n > 2 {
			t.Fatalf("attempt cap of 2 allowed %d dispatches", n)
		}
	})
}

func TestSchedulerGracefulShutdownDrainsInFlight(t *testing.T) {
	t.Parallel()
	h := newHarness(t, harnessOpts{batchSize: 1})
	fc := h.addWorker("w-1", "m", 64, 1)

	running, _, _ := h.submit("r-running", "t-1", "m", 32, 32, 0)
	if running == nil {
		t.Fatal("submit refused")
	}
	waitFor(t, 2*time.Second, "dispatch", func() bool { return fc.sawDispatch("r-running") })

	queued, _, _ := h.submit("r-queued", "t-1", "m", 32, 32, 0)
	if queued == nil {
		t.Fatal("second submit refused")
	}

	done := make(chan error, 1)
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		done <- h.sched.Shutdown(ctx)
	}()

	// Draining rather than killing: the worker is told to finish, and the request
	// that was already running is allowed to.
	waitFor(t, 2*time.Second, "drain sent", func() bool { return fc.drainCount() >= 1 })
	if h.sched.Accepting() {
		t.Fatal("still accepting during shutdown")
	}
	if handle, _, reason := h.submit("r-late", "t-1", "m", 32, 32, 0); handle != nil {
		t.Fatalf("a submission during shutdown was accepted (%s)", reason)
	}

	// Queued work never started, so refusing it costs nothing and holding it would
	// make the shutdown as long as the queue is deep.
	if res := waitDone(t, queued, 2*time.Second); res.Status != StatusFailed {
		t.Fatalf("queued request finished as %s", res.Status)
	}

	fc.emit(WorkerEvent{Kind: EventCompleted, Request: "r-running", Tokens: 9, Err: "eos"})
	if res := waitDone(t, running, 3*time.Second); res.Status != StatusCompleted {
		t.Fatalf("in-flight request finished as %s: %q", res.Status, res.Err)
	}

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("shutdown: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("shutdown did not return after the in-flight work finished")
	}
}

func TestSchedulerQueueFullIsReportedAsSuch(t *testing.T) {
	t.Parallel()
	// The worker is registered but has no scheduler slot, which models a worker that
	// is present and healthy while nothing is draining the queue. Capacity one makes
	// the second submission the one that has to be refused.
	h := newHarness(t, harnessOpts{queueCapacity: 1})
	if err := h.reg.Register(&WorkerState{ID: "w-big", Model: "m", TotalCacheBlocks: 64,
		MaxBatchSize: 1, LastHeartbeat: time.Now()}); err != nil {
		t.Fatalf("register: %v", err)
	}

	first, dec, reason := h.submit("r-1", "t-1", "m", 512, 512, 0)
	if first == nil {
		t.Fatalf("first submit refused: %s/%s", dec, reason)
	}
	second, _, reason := h.submit("r-2", "t-1", "m", 512, 512, 0)
	if second != nil {
		t.Fatal("the second submission fit a queue of capacity one")
	}
	if reason != RejectQueueFull {
		t.Fatalf("refusal reason %s, want queue_full", reason)
	}
	if n := metricValue(t, h.met, `loom_requests_rejected_total{reason="queue_full"}`); n != 1 {
		t.Fatalf("rejection not recorded under its reason: %v", n)
	}
}
