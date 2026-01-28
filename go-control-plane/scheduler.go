package loom

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

// Errors the scheduler returns to a submitter.
var (
	ErrSchedulerStopped  = errors.New("loom: scheduler is not accepting requests")
	ErrDuplicateRequest  = errors.New("loom: request id already in flight")
	ErrNoWorkerForModel  = errors.New("loom: no worker serves the requested model")
	ErrWorkerSlotMissing = errors.New("loom: worker has no scheduler slot")
)

// RequestStatus is the client-visible lifecycle state.
//
// Distinct from Phase, which is about where the work is (queued, prefill,
// decode). A request can be in PhaseQueued because it has not started yet or
// because it was preempted and is waiting to resume, and a client needs to be
// told the difference between "still going" and "over, and here is why".
type RequestStatus uint8

const (
	StatusQueued RequestStatus = iota
	StatusRunning
	StatusCompleted
	StatusFailed
	StatusCancelled
	StatusTimedOut
	StatusRejected
)

func (s RequestStatus) String() string {
	switch s {
	case StatusQueued:
		return "queued"
	case StatusRunning:
		return "running"
	case StatusCompleted:
		return "completed"
	case StatusFailed:
		return "failed"
	case StatusCancelled:
		return "cancelled"
	case StatusTimedOut:
		return "timed_out"
	case StatusRejected:
		return "rejected"
	}
	return "unknown"
}

// Terminal reports whether no further transition is possible.
func (s RequestStatus) Terminal() bool {
	switch s {
	case StatusCompleted, StatusFailed, StatusCancelled, StatusTimedOut, StatusRejected:
		return true
	}
	return false
}

// Chunk is one streamed increment of output.
type Chunk struct {
	Text string

	// Tokens is the running generated count, so a client that dropped a chunk can
	// see that it did.
	Tokens int

	// First marks the first-token notification, which arrives before any text. It
	// exists so time-to-first-token is observable by the client rather than only in
	// the server's histogram.
	First bool
}

// Result is the terminal (or current) view of a request.
type Result struct {
	ID     RequestID
	Tenant TenantID
	Model  ModelID

	Status RequestStatus
	Reason RejectReason

	// StopReason is the worker's own `eos`/`length`/`cancelled`.
	StopReason string
	Err        string

	Phase           Phase
	GeneratedTokens int
	PromptTokens    int
	Worker          WorkerID
	Attempts        int
	HeldBlocks      int

	ArrivalTime  time.Time
	QueuedAt     time.Time
	StartedAt    time.Time
	FirstTokenAt time.Time
	CompletedAt  time.Time

	// DroppedChunks counts output increments discarded because the client was not
	// reading fast enough. Reported rather than hidden: a client that sees a token
	// count larger than the text it received needs to know why, and the control
	// plane must not stall on a slow reader to avoid it.
	DroppedChunks int
}

// Handle is the submitter's view of one request.
type Handle struct {
	id     RequestID
	chunks chan Chunk
	done   chan struct{}
	cancel func()

	mu      sync.Mutex
	res     Result
	dropped int
	closed  bool
}

// ID is the request id.
func (h *Handle) ID() RequestID { return h.id }

// Chunks streams output increments. Closed when the request reaches a terminal
// state, so a range over it terminates without a second signal.
func (h *Handle) Chunks() <-chan Chunk { return h.chunks }

// Done closes when the request is terminal.
func (h *Handle) Done() <-chan struct{} { return h.done }

// Result is a snapshot, safe to call at any time from any goroutine.
func (h *Handle) Result() Result {
	h.mu.Lock()
	defer h.mu.Unlock()
	r := h.res
	r.DroppedChunks = h.dropped
	return r
}

// Cancel abandons the request.
func (h *Handle) Cancel() {
	if h.cancel != nil {
		h.cancel()
	}
}

// push is called only from the scheduler loop, which is what makes the single
// writer discipline on the channel hold.
func (h *Handle) push(c Chunk) {
	h.mu.Lock()
	if h.closed {
		h.mu.Unlock()
		return
	}
	select {
	case h.chunks <- c:
	default:
		h.dropped++
	}
	h.mu.Unlock()
}

func (h *Handle) setStatus(st RequestStatus, mutate func(r *Result)) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.res.Status.Terminal() {
		return
	}
	h.res.Status = st
	if mutate != nil {
		mutate(&h.res)
	}
	if st.Terminal() && !h.closed {
		h.closed = true
		close(h.chunks)
		close(h.done)
	}
}

// SchedulerConfig assembles the scheduler from the components it drives.
//
// The concrete *Registry and *QuotaAdmissionController are required rather than
// their interfaces because the scheduler needs operations the frozen interfaces do
// not expose: the registry's health census and its retry classifier, and the
// admission controller's Reconcile, which is what turns a worker's authoritative
// cache report into the drift number the whole reservation design is judged by.
type SchedulerConfig struct {
	Queue     *BoundedQueue
	Registry  *Registry
	Admission *QuotaAdmissionController
	Metrics   *Metrics

	// Autoscaler is optional; nil turns autoscaling off, which is one of the
	// experiment dimensions.
	Autoscaler Autoscaler

	// NewCache and NewBatcher are per-worker, because cache capacity is a property
	// of the individual worker's device as reported at registration, and a batcher
	// owns the cache it allocates from.
	NewCache   func(w *WorkerState) KVCache
	NewBatcher func(w *WorkerState, cache KVCache) Batcher

	Clock Clock

	// Tick is the scheduling period. Submissions and worker events also wake the
	// loop, so this bounds only the time-driven work: deadline expiry, reaping,
	// gauges and autoscaling.
	Tick time.Duration

	// HeartbeatTimeout is the silence after which a worker is declared dead. Zero
	// takes the registry's configured threshold, so the protocol's three-misses
	// rule is not restated in two places.
	HeartbeatTimeout time.Duration

	DispatchTimeout time.Duration
	CancelTimeout   time.Duration

	// ChunkBuffer bounds a handle's undelivered output.
	ChunkBuffer int

	// RetainCompleted is how long a terminal request stays queryable. Status for a
	// request that has just finished is exactly what a client asks for, and
	// forgetting it the instant it ends makes the endpoint useless.
	RetainCompleted time.Duration

	// OnScaleUp receives scale-up recommendations. The scheduler cannot start a
	// worker — that is the orchestrator's job — so it reports rather than pretends.
	OnScaleUp func(ScaleDecision)

	Logf func(format string, args ...any)
}

func (c SchedulerConfig) withDefaults() SchedulerConfig {
	if c.Clock == nil {
		c.Clock = SystemClock{}
	}
	if c.Tick <= 0 {
		c.Tick = 5 * time.Millisecond
	}
	if c.DispatchTimeout <= 0 {
		c.DispatchTimeout = 2 * time.Second
	}
	if c.CancelTimeout <= 0 {
		c.CancelTimeout = time.Second
	}
	if c.ChunkBuffer <= 0 {
		c.ChunkBuffer = 256
	}
	if c.RetainCompleted <= 0 {
		c.RetainCompleted = 5 * time.Minute
	}
	if c.NewCache == nil {
		c.NewCache = func(w *WorkerState) KVCache {
			return NewPagedCache(CachePolicy{TokensPerBlock: 16, TotalBlocks: w.TotalCacheBlocks})
		}
	}
	if c.NewBatcher == nil {
		c.NewBatcher = func(w *WorkerState, cache KVCache) Batcher {
			return NewContinuousBatcher(0, nil, ReserveExpected, cache, PreemptNewest)
		}
	}
	return c
}

// workerSlot is the scheduler's per-worker state.
//
// The cache lives here rather than globally because KV-cache blocks are not
// fungible across devices: a free block on worker B is of no use to a sequence
// running on worker A, and a single pooled counter would let the scheduler admit
// into capacity that exists on the wrong machine.
type workerSlot struct {
	id     WorkerID
	model  ModelID
	client WorkerClient
	cache  KVCache
	batch  Batcher

	// running is the set the worker is believed to hold. Order is preserved so the
	// batcher sees a stable running set and its victim choice is reproducible.
	running []*RequestState

	drained bool
}

func (s *workerSlot) has(id RequestID) bool {
	for _, st := range s.running {
		if st.ID == id {
			return true
		}
	}
	return false
}

func (s *workerSlot) drop(id RequestID) {
	for i, st := range s.running {
		if st.ID == id {
			s.running = append(s.running[:i], s.running[i+1:]...)
			return
		}
	}
}

type opKind uint8

const (
	opWorkerEvent opKind = iota
	opHeartbeat
	opAddWorker
	opWorkerGone
	opCancel
	opDrainWorker
	opShutdown
)

type schedOp struct {
	kind   opKind
	worker WorkerID
	ev     WorkerEvent
	hb     Heartbeat
	req    RequestID

	state  *WorkerState
	client WorkerClient

	// ack, when non-nil, is closed once the loop has applied the op. Used by the
	// operations whose caller needs the result before continuing — adding a worker,
	// mainly, since dispatching to a half-registered worker is a race.
	ack chan error
}

// Scheduler is the loop that makes the system work.
//
// Every piece of mutable scheduling state — the per-worker running sets, the
// caches, the batchers — is owned by one goroutine, and everything else reaches it
// by sending an operation. That is a deliberate choice over locking each piece:
// the interesting invariants span components (a reservation released in a cache,
// an assignment dropped in the registry, a charge refunded in admission, all for
// one request), and under locks those three steps are three windows in which the
// accounting disagrees with itself.
type Scheduler struct {
	cfg SchedulerConfig

	ops  chan schedOp
	wake chan struct{}
	quit chan struct{}

	started  atomic.Bool
	stopping atomic.Bool
	loopDone chan struct{}
	stopOnce sync.Once

	mu      sync.Mutex
	handles map[RequestID]*Handle
	states  map[RequestID]*RequestState
	prompts map[RequestID]string

	// loop-owned below this line.
	slots     map[WorkerID]*workerSlot
	drift     map[WorkerID]int
	retire    []retiring
	draining  bool
	lastScale ScaleDecision
}

type retiring struct {
	id RequestID
	at time.Time
}

// NewScheduler builds a scheduler. Call Start to run it.
func NewScheduler(cfg SchedulerConfig) *Scheduler {
	cfg = cfg.withDefaults()
	s := &Scheduler{
		cfg:      cfg,
		ops:      make(chan schedOp, 1024),
		wake:     make(chan struct{}, 1),
		quit:     make(chan struct{}),
		loopDone: make(chan struct{}),
		handles:  make(map[RequestID]*Handle),
		states:   make(map[RequestID]*RequestState),
		prompts:  make(map[RequestID]string),
		slots:    make(map[WorkerID]*workerSlot),
		drift:    make(map[WorkerID]int),
	}
	if cfg.Queue != nil && cfg.Registry != nil {
		// The policy's feasibility check needs live worker state, and the queue is
		// what calls the policy. Wiring it here keeps the queue from needing to know
		// about the registry at construction.
		cfg.Queue.SetWorkerSource(func() []*WorkerState {
			return cfg.Registry.Healthy(cfg.Clock.Now())
		})
	}
	if cfg.Queue != nil && cfg.Admission != nil {
		// The policy's feasibility test and the admission controller's reservation have
		// to be the same number, or the scheduler refuses to dispatch at the concurrency
		// admission is reserving for. Taken from the controller rather than configured
		// separately, because two places to set one decision is one place to get it wrong.
		if p, ok := cfg.Queue.Policy().(interface {
			SetReservationMode(ReservationMode, *CostEstimator)
		}); ok {
			p.SetReservationMode(cfg.Admission.Mode(), cfg.Admission.Estimator())
		}
	}
	return s
}

// Start runs the loop. Idempotent.
func (s *Scheduler) Start() {
	if s.started.Swap(true) {
		return
	}
	go s.run()
}

// Prompt returns the stored prompt text for a request, which is what a
// WorkerConn's Prompts hook is wired to.
func (s *Scheduler) Prompt(id RequestID) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.prompts[id]
}

// Lookup returns a submitted request's handle, for a status query.
func (s *Scheduler) Lookup(id RequestID) (*Handle, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	h, ok := s.handles[id]
	return h, ok
}

// Accepting reports whether submissions are still being taken, which is what
// /readyz needs during a graceful shutdown.
func (s *Scheduler) Accepting() bool { return !s.stopping.Load() }

// LastScaleDecision is the most recent non-zero autoscaler recommendation, for an
// operator endpoint and for the experiments. Published through the mutex rather
// than read from the loop's own state, which only the loop may touch.
func (s *Scheduler) LastScaleDecision() ScaleDecision {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastScale
}

// Submit admits a request and queues it.
//
// The admission decision is returned rather than translated, so the caller can
// map it to its own protocol: the gateway turns RejectQueueFull into 429 and
// RejectModelUnavailable into 503, and a bench harness counts them. A nil handle
// means refused.
func (s *Scheduler) Submit(req *Request, prompt string) (*Handle, AdmitDecision, RejectReason) {
	if s.cfg.Metrics != nil {
		s.cfg.Metrics.RequestArrived()
	}
	if s.stopping.Load() {
		// Refusing during drain rather than queueing: the loop is no longer
		// dispatching, so a queued request would wait out the shutdown and then fail.
		s.reject(RejectNoCapacity)
		return nil, AdmitRejected, RejectNoCapacity
	}

	now := s.cfg.Clock.Now()
	if req.ArrivalTime.IsZero() {
		req.ArrivalTime = now
	}

	dec, reason, cost := s.cfg.Admission.Admit(req, s.admissionWorkers(now), s.cfg.Queue, now)
	if dec == AdmitRejected {
		s.reject(reason)
		return nil, dec, reason
	}

	st := &RequestState{Request: *req, Phase: PhaseQueued, Est: cost, QueuedAt: now}
	h := &Handle{
		id:     req.ID,
		chunks: make(chan Chunk, s.cfg.ChunkBuffer),
		done:   make(chan struct{}),
		res: Result{
			ID: req.ID, Tenant: req.Tenant, Model: req.Model,
			Status: StatusQueued, Phase: PhaseQueued,
			PromptTokens: req.PromptTokens, ArrivalTime: req.ArrivalTime, QueuedAt: now,
		},
	}
	h.cancel = func() { s.Cancel(req.ID) }

	s.mu.Lock()
	if _, dup := s.handles[req.ID]; dup {
		s.mu.Unlock()
		s.cfg.Admission.Complete(req.ID)
		s.reject(RejectNone)
		return nil, AdmitRejected, RejectNone
	}
	s.handles[req.ID] = h
	s.states[req.ID] = st
	s.prompts[req.ID] = prompt
	s.mu.Unlock()

	if !s.cfg.Queue.Push(st) {
		// Admission said yes and the queue said no. It can happen: admission's check
		// and this push are not one atomic step, and that is tolerable precisely
		// because the loser is told the truthful reason rather than being dropped.
		s.mu.Lock()
		delete(s.handles, req.ID)
		delete(s.states, req.ID)
		delete(s.prompts, req.ID)
		s.mu.Unlock()
		s.cfg.Admission.Complete(req.ID)
		s.reject(RejectQueueFull)
		return nil, AdmitRejected, RejectQueueFull
	}

	if s.cfg.Metrics != nil {
		s.cfg.Metrics.RequestAccepted()
	}
	if req.Cancelled != nil {
		// One watcher per request rather than scanning every queued request's channel
		// each tick: cancellation has to be prompt to be worth anything, and a scan
		// costs the whole queue depth on every tick to notice a rare event.
		go s.watchCancel(req.ID, req.Cancelled, h.done)
	}
	s.kick()
	return h, dec, reason
}

// admissionWorkers is the fleet view the admission controller needs: heartbeating
// workers including the draining ones.
//
// Not Healthy(), which drops draining workers — admission distinguishes "too large
// for any hardware we have" from "nothing is accepting work right now", and it can
// only tell them apart if it can see hardware that exists but is not taking work.
// Not All() either, since a dead worker's free blocks are not capacity.
func (s *Scheduler) admissionWorkers(now time.Time) []*WorkerState {
	all := s.cfg.Registry.All()
	out := all[:0:0]
	for _, w := range all {
		if state, ok := s.cfg.Registry.Health(w.ID, now); ok && state == HealthHealthy {
			out = append(out, w)
		}
	}
	return out
}

func (s *Scheduler) watchCancel(id RequestID, cancelled <-chan struct{}, done <-chan struct{}) {
	select {
	case <-cancelled:
		s.Cancel(id)
	case <-done:
	case <-s.loopDone:
	}
}

func (s *Scheduler) reject(reason RejectReason) {
	if s.cfg.Metrics != nil {
		s.cfg.Metrics.RequestRejected(reason)
	}
}

// Cancel abandons a request, releasing its reservation now rather than at the
// worker's convenience.
func (s *Scheduler) Cancel(id RequestID) {
	s.post(schedOp{kind: opCancel, req: id})
}

// AddWorker registers a worker and starts consuming its events. It blocks until
// the loop has the worker, because a dispatch to a worker whose slot does not
// exist yet would be dropped.
func (s *Scheduler) AddWorker(ws *WorkerState, client WorkerClient) error {
	if ws == nil || ws.ID == "" {
		return ErrUnknownWorker
	}
	ack := make(chan error, 1)
	copyState := *ws
	s.post(schedOp{kind: opAddWorker, worker: ws.ID, state: &copyState, client: client, ack: ack})
	select {
	case err := <-ack:
		return err
	case <-s.loopDone:
		return ErrSchedulerStopped
	}
}

// WorkerGone reports that a worker's connection ended. Reporting it is much
// better than waiting for the heartbeat timeout to notice: the reservations come
// back immediately instead of after three missed intervals.
func (s *Scheduler) WorkerGone(id WorkerID) {
	s.post(schedOp{kind: opWorkerGone, worker: id})
}

// Heartbeat applies a worker's report, reconciling the control plane's cache
// estimate against it.
func (s *Scheduler) Heartbeat(hb Heartbeat) {
	s.post(schedOp{kind: opHeartbeat, worker: hb.Worker, hb: hb})
}

// DrainWorker asks a worker to stop taking new work and finish what it has.
func (s *Scheduler) DrainWorker(id WorkerID) {
	s.post(schedOp{kind: opDrainWorker, worker: id})
}

// ConsumeEvents funnels a worker client's event channel into the loop. The
// scheduler calls it for workers added through AddWorker; a caller with its own
// client wiring can call it directly.
func (s *Scheduler) ConsumeEvents(id WorkerID, client WorkerClient) {
	for ev := range client.Events() {
		if ev.Worker == "" {
			ev.Worker = id
		}
		s.post(schedOp{kind: opWorkerEvent, worker: ev.Worker, ev: ev})
	}
	// The channel closing is the connection ending, so the capacity the worker held
	// must come back now.
	s.WorkerGone(id)
}

func (s *Scheduler) post(op schedOp) {
	select {
	case s.ops <- op:
	case <-s.loopDone:
		if op.ack != nil {
			op.ack <- ErrSchedulerStopped
		}
	}
}

func (s *Scheduler) kick() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

// Shutdown stops accepting, drains in-flight work, and closes the workers.
//
// Draining rather than killing: the requests already most of the way through a
// generation finish, which is the difference between a deploy that is invisible to
// clients and one that fails a few hundred of them. If ctx expires first, what is
// still running is failed and the loop stops anyway — a shutdown that can be
// blocked indefinitely by one stuck worker is not a shutdown.
func (s *Scheduler) Shutdown(ctx context.Context) error {
	s.stopping.Store(true)
	if !s.started.Load() {
		s.stopOnce.Do(func() { close(s.quit) })
		return nil
	}
	s.post(schedOp{kind: opShutdown})
	select {
	case <-s.loopDone:
		return nil
	case <-ctx.Done():
		s.stopOnce.Do(func() { close(s.quit) })
		<-s.loopDone
		return ctx.Err()
	}
}

func (s *Scheduler) run() {
	ticker := time.NewTicker(s.cfg.Tick)
	defer ticker.Stop()
	defer close(s.loopDone)

	for {
		select {
		case <-s.quit:
			s.teardown(true)
			return
		case op := <-s.ops:
			s.apply(op)
			if s.draining && s.drainComplete() {
				s.teardown(false)
				return
			}
		case <-s.wake:
			s.tick(s.cfg.Clock.Now())
		case <-ticker.C:
			s.tick(s.cfg.Clock.Now())
			if s.draining && s.drainComplete() {
				s.teardown(false)
				return
			}
		}
	}
}

func (s *Scheduler) apply(op schedOp) {
	now := s.cfg.Clock.Now()
	switch op.kind {
	case opWorkerEvent:
		s.onEvent(op.ev, now)
	case opHeartbeat:
		s.onHeartbeat(op.hb, now)
	case opAddWorker:
		err := s.addWorker(op.state, op.client, now)
		if op.ack != nil {
			op.ack <- err
		}
		if err == nil {
			s.kick()
		}
	case opWorkerGone:
		s.loseWorker(op.worker, now, "connection_closed")
	case opCancel:
		s.cancel(op.req, now)
	case opDrainWorker:
		s.drainWorker(op.worker, now)
	case opShutdown:
		s.beginDrain(now)
	}
	if op.ack != nil && op.kind != opAddWorker {
		op.ack <- nil
	}
}

// tick is the time-driven half of the loop.
//
// The order is the policy. Expired requests go before anything else so capacity is
// not spent forming a batch around work nobody wants. Reaping goes before
// dispatch so the blocks a dead worker was holding are available to the batch this
// tick forms — the alternative leaks that capacity for a whole tick, and under
// repeated worker loss the leak is permanent.
func (s *Scheduler) tick(now time.Time) {
	s.purgeRetired(now)
	s.dropExpired(now)
	s.reap(now)
	if !s.draining {
		s.dispatchPass(now)
	}
	s.publish(now)
	s.autoscale(now)
}

func (s *Scheduler) dropExpired(now time.Time) {
	for _, st := range s.cfg.Queue.DropExpired(now) {
		if s.cfg.Metrics != nil {
			s.cfg.Metrics.RequestTimedOut()
		}
		s.finish(st, StatusTimedOut, "", "deadline expired while queued", now)
	}
}

func (s *Scheduler) reap(now time.Time) {
	timeout := s.cfg.HeartbeatTimeout
	if timeout <= 0 {
		timeout = s.cfg.Registry.DeadAfter()
	}
	for _, dead := range s.cfg.Registry.ReapDead(now, timeout) {
		s.releaseLostWorker(dead.ID, dead.Requests, now, "heartbeat_timeout")
	}
}

// releaseLostWorker is the capacity-leak path, and the one place in the scheduler
// where getting it wrong is invisible until the cluster wedges.
//
// Each of the four releases is for a different ledger and none of them implies the
// others: the per-worker cache (the block accounting admission plans against), the
// registry assignment (the mapping reaping itself depends on), the tenant's quota
// charge, and the request's own state. A worker loss that releases three of the
// four leaves capacity that is accounted for but can never be used again.
func (s *Scheduler) releaseLostWorker(id WorkerID, requests []RequestID, now time.Time, why string) {
	slot := s.slots[id]
	if s.cfg.Metrics != nil {
		s.cfg.Metrics.WorkerFailed()
		s.cfg.Metrics.WorkerReaped(len(requests))
	}

	for _, rid := range requests {
		st := s.state(rid)
		if slot != nil && slot.cache != nil {
			slot.cache.Release(rid)
		}
		s.cfg.Registry.Release(rid)
		if slot != nil {
			slot.drop(rid)
		}
		if st == nil {
			continue
		}
		st.HeldBlocks = 0
		st.Worker = ""
		f := Failure{Request: rid, Worker: id, Transient: true, WorkerLost: true, Err: why}
		s.applyRetry(st, f, now)
	}

	if slot != nil {
		// Whatever the worker still held beyond its reported assignments goes too. A
		// cache mirror kept alive for a dead worker is capacity the scheduler believes
		// in and can never dispatch into.
		delete(s.slots, id)
		delete(s.drift, id)
		if slot.client != nil {
			if c, ok := slot.client.(interface{ Close() error }); ok {
				_ = c.Close()
			}
		}
	}
	// Drop the registry entry too, so a restarted worker can register the same id.
	s.cfg.Registry.Forget(id)
}

func (s *Scheduler) loseWorker(id WorkerID, now time.Time, why string) {
	if _, ok := s.slots[id]; !ok {
		if _, known := s.cfg.Registry.Get(id); !known {
			return
		}
	}
	requests := s.cfg.Registry.Deregister(id)
	s.releaseLostWorker(id, requests, now, why)
}

// applyRetry runs the classifier and acts on it. Never a blind retry: a permanent
// failure retried is one bad request turned into a loop, and a retry past the
// deadline is capacity spent on a result nobody will read.
func (s *Scheduler) applyRetry(st *RequestState, f Failure, now time.Time) {
	verdict := s.cfg.Registry.ClassifyRetry(st, f, now)
	if s.cfg.Metrics != nil {
		s.cfg.Metrics.Retry(verdict.Action)
	}
	switch verdict.Action {
	case DoNotRetry:
		if s.cfg.Metrics != nil {
			s.cfg.Metrics.RequestFailed()
		}
		s.finish(st, StatusFailed, "", failureMessage(f, verdict), now)
	case RetrySame:
		if s.redispatchSame(st, f.Worker, now) {
			return
		}
		s.requeue(st, now)
	default:
		s.requeue(st, now)
	}
}

func failureMessage(f Failure, v RetryVerdict) string {
	if f.Err == "" {
		return v.Reason
	}
	return f.Err + " (" + v.Reason + ")"
}

// redispatchSame puts a transiently failed request straight back on the worker
// that failed it, which is worth doing only because that worker still holds the
// prompt's KV blocks — re-prefilling elsewhere costs more than the retry saves.
func (s *Scheduler) redispatchSame(st *RequestState, id WorkerID, now time.Time) bool {
	slot := s.slots[id]
	if slot == nil || slot.drained {
		return false
	}
	ws, ok := s.cfg.Registry.Get(id)
	if !ok || ws.Draining {
		return false
	}
	need := st.HeldBlocks
	if need <= 0 {
		need = st.Est.ExpectedCacheBlocks
	}
	if need <= 0 {
		need = 1
	}
	if slot.cache != nil && !slot.cache.Allocate(st.ID, need) {
		return false
	}
	st.HeldBlocks = need
	st.Worker = id
	st.Attempts++
	if err := s.cfg.Registry.Assign(st.ID, id, need); err != nil {
		if slot.cache != nil {
			slot.cache.Release(st.ID)
		}
		return false
	}
	if err := s.dispatch(slot, []*RequestState{st}, now); err != nil {
		if slot.cache != nil {
			slot.cache.Release(st.ID)
		}
		s.cfg.Registry.Release(st.ID)
		st.HeldBlocks, st.Worker = 0, ""
		return false
	}
	slot.running = append(slot.running, st)
	s.markRunning(st)
	return true
}

func (s *Scheduler) requeue(st *RequestState, now time.Time) {
	st.Phase = PhaseQueued
	st.Worker = ""
	st.HeldBlocks = 0
	st.QueuedAt = now
	if !s.cfg.Queue.Push(st) {
		// Nowhere to put it. Failing is the honest outcome; holding it outside the
		// queue would make it invisible to deadline expiry and cancellation both.
		if s.cfg.Metrics != nil {
			s.cfg.Metrics.RequestFailed()
		}
		s.finish(st, StatusFailed, "", "requeue failed: queue full", now)
		return
	}
	if h := s.handle(st.ID); h != nil {
		h.setStatus(StatusQueued, func(r *Result) {
			r.Phase = PhaseQueued
			r.Worker = ""
			r.Attempts = st.Attempts
			r.QueuedAt = now
		})
	}
	s.kick()
}

// dispatchPass forms and sends a batch per healthy worker.
func (s *Scheduler) dispatchPass(now time.Time) {
	for _, ws := range s.cfg.Registry.Healthy(now) {
		slot := s.slots[ws.ID]
		if slot == nil || slot.drained {
			continue
		}
		batch, retired := slot.batch.FormBatch(slot.running, s.cfg.Queue, ws, now)
		// The batcher has released these sequences' blocks, so the scheduler's mirror of
		// that cache has to stop counting them. Their completion is still the worker's to
		// report — the metrics and the client's result come from the `completed` event —
		// but holding the reservation until that event arrives reserves capacity for a
		// sequence that is already finished.
		for _, st := range retired {
			s.releaseSequence(slot, st)
		}
		if len(batch) == 0 {
			continue
		}

		fresh := make([]*RequestState, 0, len(batch))
		for _, st := range batch {
			if st == nil || slot.has(st.ID) {
				continue
			}
			fresh = append(fresh, st)
		}
		if len(fresh) == 0 {
			continue
		}

		sendable := make([]*RequestState, 0, len(fresh))
		for _, st := range fresh {
			if s.handle(st.ID) == nil {
				// Cancelled or expired while the batcher held it. Dispatching now would
				// start work whose result has nowhere to go, and whose blocks the batcher
				// has already reserved.
				if slot.cache != nil {
					slot.cache.Release(st.ID)
				}
				continue
			}
			blocks := heldBlocks(slot.cache, st)
			if blocks <= 0 {
				blocks = st.Est.ExpectedCacheBlocks
			}
			st.HeldBlocks = blocks
			st.Worker = ws.ID
			if st.StartedAt.IsZero() {
				st.StartedAt = now
			}
			st.Attempts++
			if err := s.cfg.Registry.Assign(st.ID, ws.ID, blocks); err != nil {
				if slot.cache != nil {
					slot.cache.Release(st.ID)
				}
				s.logf("loom: assign %s to %s: %v", st.ID, ws.ID, err)
				s.requeue(st, now)
				continue
			}
			// Rebill before the request runs: the batcher may have reserved more than
			// admission charged, and a quota checked against the charge rather than the
			// reservation bounds the estimate instead of the reality.
			s.cfg.Admission.Rebill(st.ID, blocks)
			if s.cfg.Metrics != nil {
				s.cfg.Metrics.ObserveQueueWait(now.Sub(st.QueuedAt))
			}
			sendable = append(sendable, st)
		}
		if len(sendable) == 0 {
			continue
		}

		if err := s.dispatch(slot, sendable, now); err != nil {
			s.logf("loom: dispatch to %s: %v", ws.ID, err)
			for _, st := range sendable {
				if slot.cache != nil {
					slot.cache.Release(st.ID)
				}
				s.cfg.Registry.Release(st.ID)
				st.HeldBlocks, st.Worker = 0, ""
				// A failed dispatch is a worker problem, not a request problem, so it is
				// classified as transient and the classifier still gets the final say.
				s.applyRetry(st, Failure{Request: st.ID, Worker: ws.ID, Transient: true, Err: err.Error()}, now)
			}
			continue
		}

		slot.running = append(slot.running, sendable...)
		for _, st := range sendable {
			s.markRunning(st)
		}
		if s.cfg.Metrics != nil {
			s.cfg.Metrics.ObserveBatchSize(len(slot.running))
		}
	}
}

func (s *Scheduler) dispatch(slot *workerSlot, batch []*RequestState, now time.Time) error {
	ctx, cancel := context.WithDeadline(context.Background(), now.Add(s.cfg.DispatchTimeout))
	defer cancel()
	return slot.client.Dispatch(ctx, batch)
}

func (s *Scheduler) markRunning(st *RequestState) {
	st.Phase = PhasePrefill
	if h := s.handle(st.ID); h != nil {
		h.setStatus(StatusRunning, func(r *Result) {
			r.Phase = st.Phase
			r.Worker = st.Worker
			r.StartedAt = st.StartedAt
			r.Attempts = st.Attempts
			r.HeldBlocks = st.HeldBlocks
		})
	}
}

func (s *Scheduler) onEvent(ev WorkerEvent, now time.Time) {
	st := s.state(ev.Request)
	if st == nil {
		// A completion for a request the scheduler has already finished — a cancelled
		// one, usually, since PROTOCOL.md has the worker confirm a cancel with
		// `completed`. Ignoring it is correct; re-releasing its blocks would double
		// count and invent free capacity.
		return
	}
	slot := s.slots[ev.Worker]
	switch ev.Kind {
	case EventFirstToken:
		if st.FirstToken.IsZero() {
			st.FirstToken = ev.Timestamp
			if s.cfg.Metrics != nil {
				// Measured from arrival, not from prefill start: queue wait is part of what
				// the client experiences, and the worker's own latency_ms cannot see it.
				s.cfg.Metrics.ObserveTimeToFirstToken(ev.Timestamp.Sub(st.ArrivalTime))
			}
		}
		st.Phase = PhaseDecode
		if st.GeneratedTokens < 1 {
			st.GeneratedTokens = 1
		}
		if h := s.handle(st.ID); h != nil {
			h.setStatus(StatusRunning, func(r *Result) {
				r.FirstTokenAt = st.FirstToken
				r.Phase = st.Phase
			})
			h.push(Chunk{First: true, Tokens: st.GeneratedTokens})
		}

	case EventToken:
		st.Phase = PhaseDecode
		// The worker's count is cumulative and authoritative, but it must never move
		// backwards: a redelivered token message would otherwise un-count progress and
		// make the request look like it had more generation left than it does.
		if ev.Tokens > 0 {
			if ev.Tokens > st.GeneratedTokens {
				st.GeneratedTokens = ev.Tokens
			}
		} else {
			st.GeneratedTokens++
		}
		s.reconcileSequenceBlocks(slot, st, ev.Blocks)
		if h := s.handle(st.ID); h != nil {
			h.setStatus(StatusRunning, func(r *Result) {
				r.GeneratedTokens = st.GeneratedTokens
				r.Phase = st.Phase
				r.HeldBlocks = st.HeldBlocks
			})
			if ev.Err != "" {
				h.push(Chunk{Text: ev.Err, Tokens: st.GeneratedTokens})
			}
		}

	case EventCompleted:
		if ev.Tokens > st.GeneratedTokens {
			st.GeneratedTokens = ev.Tokens
		}
		status := StatusCompleted
		if ev.Err == "cancelled" {
			status = StatusCancelled
		}
		if s.cfg.Metrics != nil {
			if status == StatusCompleted {
				s.cfg.Metrics.RequestCompleted()
			} else {
				s.cfg.Metrics.RequestCancelled()
			}
			s.cfg.Metrics.ObserveTotalLatency(ev.Timestamp.Sub(st.ArrivalTime))
			if !st.StartedAt.IsZero() && st.GeneratedTokens > 0 {
				if d := ev.Timestamp.Sub(st.StartedAt).Seconds(); d > 0 {
					s.cfg.Metrics.ObserveTokensPerSecond(float64(st.GeneratedTokens) / d)
				}
			}
		}
		s.finish(st, status, ev.Err, "", ev.Timestamp)

	case EventFailed:
		f := Failure{Request: st.ID, Worker: ev.Worker, Err: ev.Err}
		if r, ok := slotFailure(slot, st.ID); ok {
			f = r
		}
		// Without the worker's own retryable flag the only safe reading is permanent:
		// guessing "transient" is how one poison request becomes a retry storm.
		s.releaseSequence(slot, st)
		s.applyRetry(st, f, now)

	case EventPreempted:
		if s.cfg.Metrics != nil {
			s.cfg.Metrics.RequestPreempted()
		}
		if ev.Tokens > st.GeneratedTokens {
			st.GeneratedTokens = ev.Tokens
		}
		// The tokens already produced are kept on the state so the metrics can show
		// how much work a preemption threw away, but the request goes back to the
		// queue to be run again rather than being failed.
		s.releaseSequence(slot, st)
		s.requeue(st, now)
	}
}

// slotFailure recovers the worker's retryable flag, which WorkerEvent has no
// field for. A client that cannot report it leaves the failure permanent.
func slotFailure(slot *workerSlot, id RequestID) (Failure, bool) {
	if slot == nil || slot.client == nil {
		return Failure{}, false
	}
	r, ok := slot.client.(interface {
		TakeFailure(RequestID) (Failure, bool)
	})
	if !ok {
		return Failure{}, false
	}
	return r.TakeFailure(id)
}

// reconcileSequenceBlocks follows the worker's per-sequence block report.
//
// The worker is authoritative on its own cache, so the control plane's mirror
// moves to match rather than the reverse. The mirror still has to be grown through
// the cache so the capacity is actually accounted for — simply assigning the
// number would let the scheduler believe in free blocks the worker has used.
func (s *Scheduler) reconcileSequenceBlocks(slot *workerSlot, st *RequestState, reported int) {
	if reported <= 0 || slot == nil {
		return
	}
	if slot.cache != nil {
		for held := heldBlocks(slot.cache, st); held < reported; held++ {
			if !slot.cache.Grow(st.ID) {
				break
			}
		}
		st.HeldBlocks = heldBlocks(slot.cache, st)
	} else {
		st.HeldBlocks = reported
	}
	if st.HeldBlocks > 0 {
		_ = s.cfg.Registry.UpdateAssignment(st.ID, st.HeldBlocks)
		s.cfg.Admission.Rebill(st.ID, st.HeldBlocks)
	}
}

func (s *Scheduler) releaseSequence(slot *workerSlot, st *RequestState) {
	if slot != nil {
		if slot.cache != nil {
			slot.cache.Release(st.ID)
		}
		slot.drop(st.ID)
	}
	s.cfg.Registry.Release(st.ID)
	st.HeldBlocks = 0
}

func (s *Scheduler) onHeartbeat(hb Heartbeat, now time.Time) {
	ws, ok := s.cfg.Registry.Get(hb.Worker)
	if !ok {
		return
	}
	slot := s.slots[hb.Worker]
	if slot != nil && slot.cache != nil {
		// The estimate the report is measured against is the scheduler's own cache
		// mirror, not the previous heartbeat. Comparing two reports would only measure
		// the worker against itself and say nothing about whether the control plane is
		// over- or under-reserving.
		ws.UsedCacheBlocks = slot.cache.Used()
	}
	drift := s.cfg.Admission.Reconcile(ws, hb.UsedCacheBlocks, hb.ActiveSequences)
	s.drift[hb.Worker] = drift
	_ = s.cfg.Registry.Heartbeat(hb.Worker, hb.UsedCacheBlocks, hb.ActiveSequences, hb.TokensPerSecond)
	if s.cfg.Metrics != nil && hb.TokensPerSecond > 0 {
		s.cfg.Metrics.ObserveTokensPerSecond(hb.TokensPerSecond)
	}
}

func (s *Scheduler) addWorker(ws *WorkerState, client WorkerClient, now time.Time) error {
	if ws.LastHeartbeat.IsZero() {
		ws.LastHeartbeat = now
	}
	if err := s.cfg.Registry.Register(ws); err != nil {
		return err
	}
	cache := s.cfg.NewCache(ws)
	slot := &workerSlot{id: ws.ID, model: ws.Model, client: client, cache: cache,
		batch: s.cfg.NewBatcher(ws, cache)}
	s.slots[ws.ID] = slot
	if a, ok := s.cfg.Autoscaler.(*QueueWaitAutoscaler); ok && a != nil {
		// Capacity that appeared without the scaler asking for it still has to count
		// against demand, or the scaler buys the same capacity a second time.
		a.NoteWorkerStarted(now)
	}
	return nil
}

func (s *Scheduler) drainWorker(id WorkerID, now time.Time) {
	slot := s.slots[id]
	if slot == nil {
		return
	}
	slot.drained = true
	_ = s.cfg.Registry.SetDraining(id, true)
	s.sendDrain(slot, now)
}

func (s *Scheduler) sendDrain(slot *workerSlot, now time.Time) {
	if d, ok := slot.client.(interface {
		Drain(context.Context) error
	}); ok {
		ctx, cancel := context.WithDeadline(context.Background(), now.Add(s.cfg.CancelTimeout))
		defer cancel()
		if err := d.Drain(ctx); err != nil {
			s.logf("loom: drain %s: %v", slot.id, err)
		}
	}
}

func (s *Scheduler) cancel(id RequestID, now time.Time) {
	st := s.state(id)
	if st == nil {
		return
	}
	if st.Worker != "" {
		slot := s.slots[st.Worker]
		if slot != nil {
			// Asked and released in the same breath. Waiting for the worker's
			// confirmation before releasing would hold the blocks for the rest of a
			// generation the client has already abandoned, which is the exact thing
			// cancellation exists to avoid; the duplicate `completed` that arrives later
			// finds no state and is ignored.
			s.sendCancel(slot, id, now)
		}
		s.releaseSequence(slot, st)
	} else {
		s.cfg.Queue.Remove(id)
	}
	if s.cfg.Metrics != nil {
		s.cfg.Metrics.RequestCancelled()
	}
	s.finish(st, StatusCancelled, "cancelled", "", now)
}

func (s *Scheduler) sendCancel(slot *workerSlot, id RequestID, now time.Time) {
	deadline := now.Add(s.cfg.CancelTimeout)
	client := slot.client
	// Off the loop: a cancel to a worker that has stopped reading would otherwise
	// hold up every other request's scheduling for the length of the timeout.
	go func() {
		ctx, cancel := context.WithDeadline(context.Background(), deadline)
		defer cancel()
		_ = client.Cancel(ctx, id)
	}()
}

// finish moves a request to a terminal state and releases everything it held.
func (s *Scheduler) finish(st *RequestState, status RequestStatus, stopReason, errMsg string, now time.Time) {
	if st == nil {
		return
	}
	if st.Worker != "" {
		s.releaseSequence(s.slots[st.Worker], st)
	}
	st.Phase = PhaseDone
	st.CompletedAt = now
	s.cfg.Admission.Complete(st.ID)

	h := s.handle(st.ID)
	s.mu.Lock()
	delete(s.states, st.ID)
	delete(s.prompts, st.ID)
	s.mu.Unlock()

	if h != nil {
		h.setStatus(status, func(r *Result) {
			r.Phase = PhaseDone
			r.StopReason = stopReason
			r.Err = errMsg
			r.GeneratedTokens = st.GeneratedTokens
			r.Worker = st.Worker
			r.Attempts = st.Attempts
			r.HeldBlocks = 0
			r.CompletedAt = now
		})
	}
	s.retire = append(s.retire, retiring{id: st.ID, at: now})
}

// purgeRetired forgets handles whose retention window has passed, so a long run
// does not accumulate one handle per request ever served.
func (s *Scheduler) purgeRetired(now time.Time) {
	keep := s.retire[:0]
	var drop []RequestID
	for _, r := range s.retire {
		if now.Sub(r.at) < s.cfg.RetainCompleted {
			keep = append(keep, r)
			continue
		}
		drop = append(drop, r.id)
	}
	s.retire = keep
	if len(drop) == 0 {
		return
	}
	s.mu.Lock()
	for _, id := range drop {
		delete(s.handles, id)
	}
	s.mu.Unlock()
}

func (s *Scheduler) publish(now time.Time) {
	m := s.cfg.Metrics
	if m == nil {
		return
	}
	m.SetQueueDepth(s.cfg.Queue.Len(), s.cfg.Queue.Cap())

	active, used, total := 0, 0, 0
	for _, slot := range s.slots {
		active += len(slot.running)
		if slot.cache != nil {
			used += slot.cache.Used()
			total += slot.cache.Total()
		}
	}
	m.SetActiveSequences(active)
	m.SetCacheBlocks(used, total)

	sum := 0
	for _, d := range s.drift {
		sum += d
	}
	m.SetCacheDrift(sum)

	m.SetWorkersByHealth(s.cfg.Registry.HealthCounts(now))
	draining := 0
	for _, w := range s.cfg.Registry.All() {
		if w.Draining {
			draining++
		}
	}
	m.SetDrainingWorkers(draining)
}

func (s *Scheduler) autoscale(now time.Time) {
	if s.cfg.Autoscaler == nil || s.draining {
		return
	}
	d := s.cfg.Autoscaler.Evaluate(s.cfg.Queue, s.cfg.Registry.All(), now)
	if d.Delta == 0 {
		return
	}
	s.mu.Lock()
	s.lastScale = d
	s.mu.Unlock()

	if d.Delta > 0 {
		// The scheduler cannot start a worker; reporting the recommendation is the
		// honest boundary, and the orchestrator that can is the one that hears it.
		if s.cfg.OnScaleUp != nil {
			s.cfg.OnScaleUp(d)
		}
		return
	}
	target := d.Target
	if target == "" {
		if a, ok := s.cfg.Autoscaler.(*QueueWaitAutoscaler); ok {
			if t, found := a.ScaleDownTarget(s.cfg.Registry.Healthy(now)); found {
				target = t
			}
		}
	}
	if target != "" {
		// Drained, not killed: its in-flight work is still valid, and failing it would
		// make scale-down cost client-visible errors.
		s.drainWorker(target, now)
	}
}

func (s *Scheduler) beginDrain(now time.Time) {
	if s.draining {
		return
	}
	s.draining = true
	s.stopping.Store(true)

	// Queued work never started, so nothing is lost by refusing it now, and holding
	// it would make the shutdown as long as the queue is deep.
	s.failQueued(now, "control plane shutting down")
	for _, slot := range s.slots {
		slot.drained = true
		_ = s.cfg.Registry.SetDraining(slot.id, true)
		s.sendDrain(slot, now)
	}
}

// failQueued empties the queue through Remove rather than Pop.
//
// Pop asks the policy, and the policy can legitimately answer "nothing is
// runnable" — during a drain there are no eligible workers, so every queued
// request is unrunnable and Pop returns nil while the queue is not empty. Draining
// has to remove what is there, not what is schedulable.
func (s *Scheduler) failQueued(now time.Time, why string) {
	var queued []*RequestState
	s.cfg.Queue.Snapshot().Each(func(st *RequestState) bool {
		queued = append(queued, st)
		return true
	})
	for _, st := range queued {
		if !s.cfg.Queue.Remove(st.ID) {
			continue
		}
		if s.cfg.Metrics != nil {
			s.cfg.Metrics.RequestFailed()
		}
		s.finish(st, StatusFailed, "", why, now)
	}
}

func (s *Scheduler) drainComplete() bool {
	for _, slot := range s.slots {
		if len(slot.running) > 0 {
			return false
		}
	}
	return true
}

// teardown closes the workers and fails whatever is still in flight. hard is set
// when the drain deadline expired, which is the only case where a running request
// is failed rather than awaited.
func (s *Scheduler) teardown(hard bool) {
	now := s.cfg.Clock.Now()
	s.stopping.Store(true)
	if hard {
		s.failQueued(now, "control plane shutting down")
	}
	for id, slot := range s.slots {
		for _, st := range append([]*RequestState(nil), slot.running...) {
			s.releaseSequence(slot, st)
			if s.cfg.Metrics != nil {
				s.cfg.Metrics.RequestFailed()
			}
			s.finish(st, StatusFailed, "", "control plane shutting down", now)
		}
		if c, ok := slot.client.(interface{ Close() error }); ok {
			_ = c.Close()
		}
		delete(s.slots, id)
	}
}

func (s *Scheduler) state(id RequestID) *RequestState {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.states[id]
}

func (s *Scheduler) handle(id RequestID) *Handle {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.handles[id]
}

func (s *Scheduler) logf(format string, args ...any) {
	if s.cfg.Logf != nil {
		s.cfg.Logf(format, args...)
	}
}
