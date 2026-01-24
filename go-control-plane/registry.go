package loom

import (
	"errors"
	"sort"
	"sync"
	"time"
)

// Clock is the time source for everything in the registry and the autoscaler.
//
// It exists because the interesting behaviour in both components is defined
// entirely by elapsed time — a missed heartbeat, a cold start, a cooldown — and a
// test that proves that behaviour by sleeping is a test that will eventually fail
// on a loaded CI box for reasons unrelated to the code. Injecting the clock makes
// those tests assertions about the state machine rather than about the scheduler
// of the machine running them.
type Clock interface {
	Now() time.Time
}

// SystemClock is the production Clock.
type SystemClock struct{}

func (SystemClock) Now() time.Time { return time.Now() }

// HealthState is a worker's liveness as the registry sees it.
//
// Three states rather than a boolean, because the two failure modes have
// asymmetric cost. A worker that misses one heartbeat is far more likely to be
// in a garbage-collection pause, a long forward pass, or a momentarily congested
// socket than to be dead; reaping it immediately fails every request it was
// running and throws away a perfectly good accelerator. A worker that really has
// crashed, on the other hand, holds reservations that nothing will ever release.
// Suspect is the state that lets the control plane stop sending new work — which
// is cheap and reversible — without taking the destructive action yet.
type HealthState uint8

const (
	// HealthHealthy means heartbeating within the interval.
	HealthHealthy HealthState = iota

	// HealthSuspect means one interval has passed without a heartbeat. The worker
	// keeps its in-flight requests; it just stops receiving new ones.
	HealthSuspect

	// HealthDead means the dead threshold passed. Its requests are reaped and its
	// reservations released.
	HealthDead
)

func (h HealthState) String() string {
	switch h {
	case HealthHealthy:
		return "healthy"
	case HealthSuspect:
		return "suspect"
	case HealthDead:
		return "dead"
	}
	return "unknown"
}

// RetryAction is the retry classifier's verdict.
type RetryAction uint8

const (
	// RetryElsewhere re-queues the request excluding the worker that failed it.
	RetryElsewhere RetryAction = iota

	// RetrySame re-dispatches to the same worker, which is worth doing when the
	// failure was transient and that worker still holds the request's prefix in
	// cache — re-prefilling elsewhere costs more than the retry saves.
	RetrySame

	// DoNotRetry fails the request to the client.
	DoNotRetry
)

func (a RetryAction) String() string {
	switch a {
	case RetryElsewhere:
		return "elsewhere"
	case RetrySame:
		return "same"
	case DoNotRetry:
		return "none"
	}
	return "unknown"
}

// Failure is what the control plane learns when a request does not complete.
//
// Transient is carried from the worker's `retryable` flag in PROTOCOL.md rather
// than inferred from the error string, because the worker is the only party that
// knows whether the prompt was malformed or the device hiccuped, and a control
// plane that guesses "retryable" turns one poison request into a retry storm.
type Failure struct {
	Request RequestID
	Worker  WorkerID

	// Transient is the worker's own classification. False means this request will
	// fail identically everywhere.
	Transient bool

	// WorkerLost is set when the failure is the worker disappearing rather than
	// the request failing. Reaping sets it; a worker-reported failure does not.
	WorkerLost bool

	Err string
}

// RetryVerdict is the classification plus the reason, so the metrics and the
// operator can see why a request was abandoned instead of only that it was.
type RetryVerdict struct {
	Action RetryAction
	Reason string
}

// RetryPolicy bounds retrying.
//
// Both bounds are necessary and neither is sufficient. The attempt cap stops a
// request that fails transiently forever from being retried forever. The deadline
// headroom stops a request from being retried into a deadline it can no longer
// meet, which would burn a full service time to produce a result nobody wants.
type RetryPolicy struct {
	// MaxAttempts counts total attempts, not retries. One means no retry.
	MaxAttempts int

	// MinDeadlineHeadroom is the time that must remain before the deadline for a
	// retry to be worth starting. Set it to roughly the request's expected service
	// time; retrying with less than that left is spending capacity on a result
	// that will arrive late.
	MinDeadlineHeadroom time.Duration
}

// DefaultRetryPolicy is a conservative starting point: two retries, and a retry
// must have a quarter-second of deadline left to be worth starting.
func DefaultRetryPolicy() RetryPolicy {
	return RetryPolicy{MaxAttempts: 3, MinDeadlineHeadroom: 250 * time.Millisecond}
}

// RegistryConfig configures health detection.
type RegistryConfig struct {
	Clock Clock

	// HeartbeatInterval is the worker's reporting period from PROTOCOL.md.
	HeartbeatInterval time.Duration

	// SuspectAfterMisses and DeadAfterMisses are counted in intervals. The
	// protocol's rule is three consecutive misses to declare death.
	SuspectAfterMisses int
	DeadAfterMisses    int

	// WarmupTimeout is how long a worker that has registered but never heartbeated
	// may stay unreaped.
	//
	// It exists because the dead threshold is three missed heartbeats — 1.5s by
	// default — while loading a model takes tens of seconds. A worker that registers
	// at launch is therefore declared dead and reaped before it has ever served
	// anything, and the only trace is an inflated failure count: a fleet that killed
	// two workers reported four reaped. PROTOCOL.md is self-consistent about this
	// (register on connect, connect when ready), so this is the registry refusing to
	// turn an integration mistake into phantom failures.
	//
	// Bounded rather than "never reap until the first heartbeat", because a loader
	// that hangs forever must still be reaped — an unbounded warm-up would make a
	// stuck worker invisible. The registry cannot know the cold start, so the bound is
	// configuration: set it above the model's load time.
	WarmupTimeout time.Duration

	Retry RetryPolicy
}

func (c RegistryConfig) withDefaults() RegistryConfig {
	if c.Clock == nil {
		c.Clock = SystemClock{}
	}
	if c.HeartbeatInterval <= 0 {
		c.HeartbeatInterval = 500 * time.Millisecond
	}
	if c.SuspectAfterMisses <= 0 {
		c.SuspectAfterMisses = 1
	}
	if c.DeadAfterMisses <= c.SuspectAfterMisses {
		c.DeadAfterMisses = 3
	}
	if c.WarmupTimeout <= 0 {
		// Longer than any plausible model load, and still finite. The cost of being
		// generous here is that a worker which registered and then hung is reaped late;
		// the cost of being stingy is reaping healthy workers mid-cold-start and
		// reporting them as failures.
		c.WarmupTimeout = 2 * time.Minute
	}
	if c.Retry.MaxAttempts <= 0 {
		c.Retry = DefaultRetryPolicy()
	}
	return c
}

// Errors returned by the registry. They are values rather than formatted strings
// so callers can branch on them; a duplicate registration is a routine event
// during a worker restart, not an outage.
var (
	ErrWorkerExists      = errors.New("loom: worker already registered")
	ErrUnknownWorker     = errors.New("loom: unknown worker")
	ErrWorkerDead        = errors.New("loom: worker is dead")
	ErrRequestAssigned   = errors.New("loom: request already assigned")
	ErrUnknownAssignment = errors.New("loom: request not assigned")
)

// DeadWorker is one reaped worker and the capacity that came back with it.
//
// ReleasedBlocks is the part that is easy to omit and expensive to omit. The
// WorkerRegistry interface only returns request ids, which is enough to re-queue
// the work but not enough for whatever holds the reservations to know how much
// came back — so the registry reports the number it tracked, and the capacity
// owner reconciles against it.
// DeadWorker is what a reaped worker was holding when it died.
//
// ReleasedBlocks is the field that matters and the reason this is a struct rather
// than a map of ids: the caller owns the reservation ledger and cannot release
// capacity it is not told about. Reporting only the request ids made the
// capacity-leak-on-worker-loss bug unfixable without reaching past the interface.
// Each worker appears here exactly once; a second reap would double-release.
type DeadWorker struct {
	ID WorkerID

	// Requests were running on the worker and now need retry or failure.
	Requests []RequestID

	// ReleasedBlocks is the cache capacity returning to the pool.
	ReleasedBlocks int
}

type assignment struct {
	worker WorkerID
	blocks int
}

type workerEntry struct {
	state WorkerState

	// requests is the worker's in-flight set. This mapping is the whole reason
	// reaping can work at all: without it a dead worker's requests are
	// unidentifiable and their reservations are unrecoverable.
	requests map[RequestID]int // request -> reserved blocks

	// registeredAt and heardFrom are the warm-up window. Until a worker has sent one
	// heartbeat of its own, its silence is indistinguishable from a model still
	// loading, so it is not reaped for it.
	registeredAt time.Time
	heardFrom    bool

	// reaped records that this worker's requests were already handed out, so a
	// second ReapDead does not hand out the same requests twice and cause a double
	// release of the same reservations.
	reaped bool
}

// Registry is the WorkerRegistry implementation.
//
// One mutex rather than per-worker locks: every interesting operation
// (ReapDead, Healthy, the assignment index) spans workers, so per-worker locking
// would need a second lock over the index anyway and would buy nothing but a
// deadlock ordering problem. Heartbeats are the only high-rate operation and they
// are O(1) under the lock.
type Registry struct {
	cfg RegistryConfig

	mu      sync.Mutex
	workers map[WorkerID]*workerEntry

	// assignments is the reverse index, so a completion does not have to scan
	// every worker to find which one held the request.
	assignments map[RequestID]assignment
}

// NewRegistry builds a Registry. A nil-field config is filled with the
// protocol's defaults.
func NewRegistry(cfg RegistryConfig) *Registry {
	return &Registry{
		cfg:         cfg.withDefaults(),
		workers:     make(map[WorkerID]*workerEntry),
		assignments: make(map[RequestID]assignment),
	}
}

func (r *Registry) suspectAfter() time.Duration {
	return time.Duration(r.cfg.SuspectAfterMisses) * r.cfg.HeartbeatInterval
}

// DeadAfter is the silence that constitutes death. Exported because callers of
// ReapDead need a sensible timeout to pass and should not have to recompute the
// protocol's rule themselves.
func (r *Registry) DeadAfter() time.Duration {
	return time.Duration(r.cfg.DeadAfterMisses) * r.cfg.HeartbeatInterval
}

// Register adds a worker. Registration timestamps the first heartbeat, since a
// worker that has just connected has by definition just reported.
func (r *Registry) Register(w *WorkerState) error {
	if w == nil || w.ID == "" {
		return ErrUnknownWorker
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, ok := r.workers[w.ID]; ok {
		return ErrWorkerExists
	}
	st := *w
	if st.LastHeartbeat.IsZero() {
		st.LastHeartbeat = r.cfg.Clock.Now()
	}
	r.workers[w.ID] = &workerEntry{
		state:        st,
		requests:     make(map[RequestID]int),
		registeredAt: r.cfg.Clock.Now(),
	}
	return nil
}

// Heartbeat applies a worker's report.
//
// used and active overwrite the registry's numbers rather than being merged with
// them: PROTOCOL.md makes the worker authoritative on its own cache, and a
// control plane that prefers its own prediction drifts until it overcommits.
//
// A heartbeat also resurrects a suspect worker, which is the entire point of the
// suspect state. It does not resurrect a reaped one — those requests have already
// been re-queued elsewhere, and accepting the worker back as healthy would let it
// keep running copies of work that is now running somewhere else.
func (r *Registry) Heartbeat(id WorkerID, used, active int, tokensPerSec float64) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	e, ok := r.workers[id]
	if !ok {
		return ErrUnknownWorker
	}
	if e.reaped {
		return ErrWorkerDead
	}
	e.state.UsedCacheBlocks = used
	e.state.ActiveSequences = active
	e.state.RecentTokensPerSecond = tokensPerSec
	e.state.LastHeartbeat = r.cfg.Clock.Now()
	// The warm-up is over at the first report, whatever it says. From here on silence
	// means the worker stopped, not that it never started.
	e.heardFrom = true
	return nil
}

// Deregister removes a worker and returns its in-flight requests so the caller
// can retry or fail them. A clean shutdown should drain first; this is the path
// for a worker that is already gone.
func (r *Registry) Deregister(id WorkerID) []RequestID {
	r.mu.Lock()
	defer r.mu.Unlock()

	e, ok := r.workers[id]
	if !ok {
		return nil
	}
	reqs := r.detachLocked(e)
	delete(r.workers, id)
	return reqs
}

// detachLocked removes every assignment on the worker and returns the request
// ids in a stable order. Sorted because callers re-queue in the returned order
// and a map-random order makes a reaping test unreproducible.
func (r *Registry) detachLocked(e *workerEntry) []RequestID {
	reqs := make([]RequestID, 0, len(e.requests))
	for id := range e.requests {
		reqs = append(reqs, id)
		delete(r.assignments, id)
	}
	sort.Slice(reqs, func(i, j int) bool { return reqs[i] < reqs[j] })
	e.requests = make(map[RequestID]int)
	e.state.ActiveSequences = 0
	e.state.UsedCacheBlocks = 0
	return reqs
}

// Get returns a copy of the worker's state.
//
// A copy, not a pointer into the registry: the alternative hands callers a
// struct that the next heartbeat mutates underneath them, which is a data race
// that only shows up under load.
func (r *Registry) Get(id WorkerID) (*WorkerState, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	e, ok := r.workers[id]
	if !ok {
		return nil, false
	}
	st := e.state
	return &st, true
}

// Healthy returns workers eligible for new work.
//
// Draining and suspect workers are both excluded, for different reasons: a
// draining worker was asked to stop taking work, and a suspect one has not
// answered recently enough to be trusted with more. Both keep their in-flight
// requests.
func (r *Registry) Healthy(now time.Time) []*WorkerState {
	r.mu.Lock()
	defer r.mu.Unlock()

	out := make([]*WorkerState, 0, len(r.workers))
	for _, e := range r.workers {
		if e.state.Draining || r.healthLocked(e, now) != HealthHealthy {
			continue
		}
		st := e.state
		out = append(out, &st)
	}
	sortWorkers(out)
	return out
}

// All returns every known worker, including draining and dead ones, which is
// what the metrics and the reaper need to see.
func (r *Registry) All() []*WorkerState {
	r.mu.Lock()
	defer r.mu.Unlock()

	out := make([]*WorkerState, 0, len(r.workers))
	for _, e := range r.workers {
		st := e.state
		out = append(out, &st)
	}
	sortWorkers(out)
	return out
}

func sortWorkers(ws []*WorkerState) {
	sort.Slice(ws, func(i, j int) bool { return ws[i].ID < ws[j].ID })
}

func (r *Registry) healthLocked(e *workerEntry, now time.Time) HealthState {
	if e.reaped {
		return HealthDead
	}
	silence := now.Sub(e.state.LastHeartbeat)
	switch {
	case silence >= r.DeadAfter():
		return HealthDead
	case silence >= r.suspectAfter():
		return HealthSuspect
	}
	return HealthHealthy
}

// warmingLocked reports whether a worker's silence is still explained by never
// having started.
//
// A worker holding in-flight requests is never warming, whatever its heartbeat
// history says: something dispatched to it, so it was believed to be serving, and
// its reservations have to come back if it is gone. The warm-up protects the case
// the trap actually produces — a worker registered at launch, loading its model,
// holding nothing.
func (r *Registry) warmingLocked(e *workerEntry, now time.Time) bool {
	if e.heardFrom || e.reaped || len(e.requests) > 0 {
		return false
	}
	if e.registeredAt.IsZero() {
		return false
	}
	return now.Sub(e.registeredAt) < r.cfg.WarmupTimeout
}

// Warming lists workers that have registered and not yet reported. A worker that
// stays here is a cold start in progress, or an integration that registers before
// it is ready — which is worth being able to see, since the symptom otherwise is
// requests failing on a worker that never ran any.
func (r *Registry) Warming(now time.Time) []WorkerID {
	r.mu.Lock()
	defer r.mu.Unlock()

	var out []WorkerID
	for id, e := range r.workers {
		if r.warmingLocked(e, now) {
			out = append(out, id)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// Health reports one worker's state in the health machine.
func (r *Registry) Health(id WorkerID, now time.Time) (HealthState, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	e, ok := r.workers[id]
	if !ok {
		return HealthDead, false
	}
	return r.healthLocked(e, now), true
}

// HealthCounts is the workers-by-state gauge the metrics export. Draining
// workers are counted under their liveness state, because a draining worker is
// still a live worker holding real work; the drain flag is reported separately.
func (r *Registry) HealthCounts(now time.Time) map[HealthState]int {
	r.mu.Lock()
	defer r.mu.Unlock()

	counts := map[HealthState]int{HealthHealthy: 0, HealthSuspect: 0, HealthDead: 0}
	for _, e := range r.workers {
		counts[r.healthLocked(e, now)]++
	}
	return counts
}

// SetDraining marks a worker drained or un-drained.
//
// A draining worker keeps its in-flight requests and keeps heartbeating, so it
// leaves Healthy() but stays in All(). Draining rather than killing is what makes
// scale-down free: the requests that were already most of the way done still
// finish.
func (r *Registry) SetDraining(id WorkerID, draining bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	e, ok := r.workers[id]
	if !ok {
		return ErrUnknownWorker
	}
	e.state.Draining = draining
	return nil
}

// Assign records that a request is running on a worker holding the given number
// of cache blocks.
//
// Rejecting an assignment to a draining or dead worker here rather than trusting
// the caller is deliberate: this is the last point at which the mistake is cheap.
func (r *Registry) Assign(req RequestID, id WorkerID, blocks int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	e, ok := r.workers[id]
	if !ok {
		return ErrUnknownWorker
	}
	if e.reaped {
		return ErrWorkerDead
	}
	if cur, ok := r.assignments[req]; ok {
		if cur.worker == id {
			// At-least-once dispatch means a duplicate is expected; PROTOCOL.md
			// requires it be a no-op rather than a second copy.
			return nil
		}
		return ErrRequestAssigned
	}
	e.requests[req] = blocks
	r.assignments[req] = assignment{worker: id, blocks: blocks}
	e.state.ActiveSequences = len(e.requests)
	return nil
}

// UpdateAssignment revises a running request's block count, which decode does
// every time a sequence crosses a block boundary. Keeping this current is what
// makes the number ReapDead reports a real figure rather than the stale estimate
// from dispatch time.
func (r *Registry) UpdateAssignment(req RequestID, blocks int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	a, ok := r.assignments[req]
	if !ok {
		return ErrUnknownAssignment
	}
	a.blocks = blocks
	r.assignments[req] = a
	if e, ok := r.workers[a.worker]; ok {
		e.requests[req] = blocks
	}
	return nil
}

// Release removes an assignment on normal completion, cancellation, or
// preemption, returning the worker it was on and the blocks it held.
func (r *Registry) Release(req RequestID) (WorkerID, int, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	a, ok := r.assignments[req]
	if !ok {
		return "", 0, false
	}
	delete(r.assignments, req)
	if e, ok := r.workers[a.worker]; ok {
		delete(e.requests, req)
		e.state.ActiveSequences = len(e.requests)
	}
	return a.worker, a.blocks, true
}

// RequestsOn lists a worker's in-flight requests.
func (r *Registry) RequestsOn(id WorkerID) []RequestID {
	r.mu.Lock()
	defer r.mu.Unlock()

	e, ok := r.workers[id]
	if !ok {
		return nil
	}
	out := make([]RequestID, 0, len(e.requests))
	for req := range e.requests {
		out = append(out, req)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// WorkerOf reports which worker holds a request.
func (r *Registry) WorkerOf(req RequestID) (WorkerID, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	a, ok := r.assignments[req]
	return a.worker, ok
}

// ReapDead marks expired workers dead and returns what was running on them.
//
// The returned requests are the only record that work existed; the blocks they
// held are the only record that capacity existed. A worker loss that returns the
// first and drops the second leaks that capacity permanently, and the leak is
// invisible — every individual request looks fine — until the accounting says the
// cluster is full while the accelerators sit idle. Use ReapDead when the
// caller owns the reservations.
//
// timeout of zero means use the registry's configured dead threshold.
//
// Each dead worker is reported exactly once. Reporting it again on the next tick
// would hand the same request ids out twice, and the second release of the same
// reservation corrupts the accounting in the opposite direction — phantom free
// capacity, then an out-of-memory kill mid-generation.
func (r *Registry) ReapDead(now time.Time, timeout time.Duration) []DeadWorker {
	if timeout <= 0 {
		timeout = r.DeadAfter()
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	var out []DeadWorker
	for id, e := range r.workers {
		if e.reaped {
			continue
		}
		if now.Sub(e.state.LastHeartbeat) < timeout {
			continue
		}
		if r.warmingLocked(e, now) {
			// It has never heartbeated, so this silence is a cold start until the warm-up
			// window says otherwise. Reaping here is what turns a slow model load into a
			// phantom worker failure.
			continue
		}
		released := 0
		for _, blocks := range e.requests {
			released += blocks
		}
		d := DeadWorker{ID: id, Requests: r.detachLocked(e), ReleasedBlocks: released}
		e.reaped = true
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Forget removes reaped workers from the registry.
//
// Separate from reaping so the metrics can report a dead worker for at least one
// scrape. A worker that vanishes between scrapes is a worker whose failure never
// appears in any graph.
func (r *Registry) Forget(ids ...WorkerID) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, id := range ids {
		if e, ok := r.workers[id]; ok && e.reaped {
			delete(r.workers, id)
		}
	}
}

// ClassifyRetry decides what to do about a failed request.
//
// The order of the checks is the policy. Permanent failures are refused before
// anything else, because retrying one is pure waste regardless of how much
// deadline or attempt budget is left. The deadline is checked before the attempt
// cap so that a request which can no longer be served in time is reported as a
// deadline problem rather than as an exhausted-retries problem — the two call for
// different operator responses.
//
// st may be nil when the request state has already been discarded, in which case
// only the failure's own classification can be applied.
func (r *Registry) ClassifyRetry(st *RequestState, f Failure, now time.Time) RetryVerdict {
	if !f.Transient && !f.WorkerLost {
		return RetryVerdict{DoNotRetry, "permanent_failure"}
	}
	if st == nil {
		return RetryVerdict{RetryElsewhere, "no_request_state"}
	}
	if !st.Deadline.IsZero() {
		if !now.Before(st.Deadline) {
			return RetryVerdict{DoNotRetry, "deadline_passed"}
		}
		if st.Deadline.Sub(now) < r.cfg.Retry.MinDeadlineHeadroom {
			return RetryVerdict{DoNotRetry, "deadline_insufficient_headroom"}
		}
	}
	if st.Attempts >= r.cfg.Retry.MaxAttempts {
		return RetryVerdict{DoNotRetry, "attempt_cap"}
	}
	if f.WorkerLost {
		return RetryVerdict{RetryElsewhere, "worker_lost"}
	}

	// A transient failure on a worker that is still taking work is worth retrying
	// in place: that worker still holds the prompt's KV blocks, so the retry skips
	// a prefill that any other worker would have to redo.
	worker := f.Worker
	if worker == "" {
		worker = st.Worker
	}
	r.mu.Lock()
	e, ok := r.workers[worker]
	eligible := ok && !e.state.Draining && r.healthLocked(e, now) == HealthHealthy
	r.mu.Unlock()

	if eligible {
		return RetryVerdict{RetrySame, "transient_worker_healthy"}
	}
	return RetryVerdict{RetryElsewhere, "transient_worker_ineligible"}
}

// RetryPolicy exposes the configured bounds, so the gateway can report them
// rather than duplicating the constants.
func (r *Registry) RetryPolicy() RetryPolicy { return r.cfg.Retry }

// Compile-time proof that the registry still satisfies the frozen contract. A
// contract change in another file should break here rather than at some call site
// three packages away.
var _ WorkerRegistry = (*Registry)(nil)
