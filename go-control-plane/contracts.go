package loom

import (
	"context"
	"time"
)

// CachePolicy describes how KV-cache memory is divided.
//
// Paging the cache is the single most consequential design choice in the system.
// A contiguous-allocation cache must reserve a request's WORST CASE up front,
// because its memory has to be one unbricked run — so a request that might
// generate 2048 tokens holds 2048 tokens' worth of memory from the moment it
// starts, even if it stops after 30. Fixed-size blocks let the reservation grow
// as the sequence actually does, and the freed blocks of a finished sequence are
// immediately reusable by any other. The difference in achievable concurrency is
// large, and bench/ measures it rather than asserting it.
type CachePolicy struct {
	// TokensPerBlock is the block granularity. Smaller blocks waste less on the
	// partial final block of each sequence but cost more bookkeeping per step.
	TokensPerBlock int

	// TotalBlocks is the capacity being divided.
	TotalBlocks int

	// Contiguous selects the naive policy, kept so the paged one has something to
	// be measured against.
	Contiguous bool
}

// BlocksFor is how many blocks a sequence of the given length occupies. The
// ceiling division is where internal fragmentation comes from: a sequence one
// token into a new block holds the whole block.
func (p CachePolicy) BlocksFor(tokens int) int {
	if p.TokensPerBlock <= 0 {
		return 0
	}
	return (tokens + p.TokensPerBlock - 1) / p.TokensPerBlock
}

// KVCache accounts for cache blocks on one worker.
//
// It is an interface so the paged and contiguous policies are interchangeable and
// directly comparable, and so the simulator can use the same accounting the real
// worker does. Accounting that only exists in the simulator proves nothing about
// the system.
type KVCache interface {
	// Allocate reserves blocks for a sequence, returning false when capacity is
	// exhausted. Must not partially allocate: a half-reserved sequence cannot run
	// and its blocks would leak.
	Allocate(id RequestID, blocks int) bool

	// Grow extends a running sequence by one block, which is what a decode step
	// needs when it crosses a block boundary. Returning false means the sequence
	// must be preempted or the step must fail — it cannot simply continue.
	Grow(id RequestID) bool

	// Release frees everything held by a sequence.
	Release(id RequestID)

	Used() int
	Free() int
	Total() int

	// Fragmentation reports capacity that exists but is unusable under the current
	// policy. Always zero for the paged policy, which is precisely the point being
	// demonstrated.
	Fragmentation() int
}

// QueueSnapshot is the read side of a queue.
//
// Separated from Queue because a scheduling policy needs to INSPECT the queue to
// make a decision and must not be able to mutate it. Handing a policy the full
// Queue would let it pop, which is the scheduler's job, and a policy that
// accidentally removed a request would hand the same work out twice.
type QueueSnapshot interface {
	// Each visits queued requests in arrival order until fn returns false.
	Each(fn func(st *RequestState) bool)

	// TenantCount is the number queued for one tenant, maintained incrementally
	// because the fair-share policy consults it on every decision and an O(n)
	// recount per decision is O(n^2) per drain.
	TenantCount(t TenantID) int

	TenantCounts() map[TenantID]int

	Len() int
}

// Queue is the bounded request queue.
//
// Bounded is not a detail. An unbounded queue converts overload into unbounded
// latency, which is strictly worse than refusal: the client has already given up
// by the time the work completes, so the capacity spent on it is pure waste and
// it delays requests that would still have been useful. Every queue in this
// system has a limit and a documented behaviour at that limit.
type Queue interface {
	// Push enqueues or reports that it cannot. Never blocks.
	Push(st *RequestState) bool

	// Pop returns the next request the policy selects, or nil when empty.
	Pop() *RequestState

	// Remove drops a request by id, for cancellation and deadline expiry.
	Remove(id RequestID) bool

	// DropExpired removes requests whose deadlines have passed and returns them,
	// so the caller can report them as failed rather than silently losing them.
	DropExpired(now time.Time) []*RequestState

	Len() int
	Cap() int

	// OldestWait is the head-of-line wait time, which is the autoscaler's primary
	// signal — more informative than queue depth, since depth says nothing about
	// how fast the queue is draining.
	OldestWait(now time.Time) time.Duration

	// Snapshot returns the read-only view a policy is given.
	Snapshot() QueueSnapshot
}

// SchedulePolicy picks which queued request runs next.
//
// Implementations exist to be compared. FIFO is the correctness baseline and is
// also genuinely fair in arrival order; shortest-job-first improves mean latency
// and starves long requests; fair-share bounds per-tenant consumption at some
// cost to throughput. The experiments in bench/ run the same workload through
// each so the tradeoff is a measurement rather than an opinion.
type SchedulePolicy interface {
	Name() string

	// Next selects from the queue given current worker state. Returning nil means
	// nothing is runnable right now, which is different from the queue being
	// empty — a request may be queued but unable to fit.
	//
	// Pure selection: Next does NOT remove the request. Pop owns removal, so a
	// caller may peek without committing. A caller that invokes Next directly must
	// Remove the id before dispatching or it will hand the same request out twice.
	Next(q QueueSnapshot, workers []*WorkerState, now time.Time) *RequestState
}

// AdmissionController decides whether a request enters the system at all.
type AdmissionController interface {
	// Admit evaluates a request against queue depth, tenant quota, and whether it
	// could ever fit. Returns the decision, the reason on refusal, and the cost
	// estimate it computed, so the caller can log why rather than just what.
	//
	// `workers` MUST include draining workers, and must be every worker that is
	// currently heartbeating. This is load-bearing and easy to get wrong: the
	// implementation derives the fleet's maximum capacity from this slice to decide
	// whether a request could EVER fit, and filters draining workers itself when
	// deciding whether it fits right now. Passing an already-filtered healthy-only
	// set makes RejectNoCapacity unreachable and silently reports capacity
	// exhaustion as RejectModelUnavailable instead.
	Admit(req *Request, workers []*WorkerState, q Queue, now time.Time) (AdmitDecision, RejectReason, EstimatedCost)

	// Complete releases what Admit charged. Admit has to charge the tenant's quota
	// at admission time for the bound to be hard, which means something must give
	// it back; without this hook a tenant's quota is consumed permanently by every
	// request it ever completes.
	Complete(id RequestID)

	// Rebill corrects the charge for a sequence admitted on its expected footprint
	// that then outgrew it. The alternative, leaving the original estimate in
	// place, means the quota drifts from reality in whichever direction the
	// estimate was wrong.
	Rebill(id RequestID, blocks int)
}

// Batcher assembles a batch from runnable requests.
//
// Continuous batching is the behaviour that matters: rather than running a batch
// to completion and then forming the next, a finished sequence is evicted from
// the running set and a queued one admitted into the freed capacity on the next
// step. A static batch is held hostage by its longest member — with a 30-token
// and a 2000-token request in one batch, the accelerator spends most of its time
// computing one sequence at batch-of-eight cost.
type Batcher interface {
	Name() string

	// FormBatch returns the sequences for the next forward pass, respecting the
	// worker's batch-size cap and cache capacity, plus the members of running it
	// retired — already-finished sequences whose cache it released.
	//
	// retired is reported rather than simply dropped because the caller owns the
	// reservation ledger and cannot release what it is not told about. A caller that
	// diffed the batch against its own previous running set would work, but it would
	// be reconstructing information the batcher already had, and every caller would
	// have to do it.
	FormBatch(running []*RequestState, q Queue, w *WorkerState, now time.Time) (batch, retired []*RequestState)

	// MaxWait is how long the batcher will hold a request hoping for company. Zero
	// means dispatch immediately, which minimizes latency at low load and wastes
	// throughput at high load; the window is the whole latency-throughput dial.
	MaxWait() time.Duration

	// Step advances the running set by one decode step and reports which sequences
	// survived, which were preempted, and which it retired.
	//
	// This exists because FormBatch is called *between* steps and so has no way to
	// report a failure that happens *during* one. A sequence whose cache cannot
	// grow mid-decode must be preempted, and a contract without this method leaves
	// that path implementable only outside the interface — meaning a driver written
	// against Batcher alone would never handle it at all.
	//
	// Implementations MUST return every sequence in running in exactly one of the
	// three slices. The caller un-bills a reservation per returned entry, so a
	// sequence in two slices double-releases and one in none leaks; and a sequence
	// reported as both preempted and surviving is in a running set and in the queue
	// at the same time, which hands a second copy of a live request to the next
	// worker that pops.
	Step(running []*RequestState, q Queue, w *WorkerState, now time.Time) (survivors, preempted, retired []*RequestState)
}

// WorkerRegistry tracks workers and their health.
type WorkerRegistry interface {
	Register(w *WorkerState) error
	Heartbeat(id WorkerID, used, active int, tokensPerSec float64) error

	// Deregister removes a worker. Its in-flight requests become the caller's
	// problem to retry or fail.
	Deregister(id WorkerID) []RequestID

	Get(id WorkerID) (*WorkerState, bool)

	// Healthy returns workers eligible for new work: heartbeating and not draining.
	Healthy(now time.Time) []*WorkerState

	// All includes unhealthy and draining workers, for metrics and for deciding
	// what to reap.
	All() []*WorkerState

	// ReapDead marks workers whose heartbeats expired and returns what was lost
	// with them.
	//
	// Returning the released block count, not just the request ids, is the point:
	// the caller owns the reservation ledger and cannot release what it is not
	// told about. An earlier shape returned only the ids, which made the capacity
	// leak on worker loss impossible to fix without reaching past the interface.
	// Each worker is reported exactly once; a second reap would double-release.
	ReapDead(now time.Time, timeout time.Duration) []DeadWorker
}

// ScaleDecision is what the autoscaler recommends.
type ScaleDecision struct {
	Delta  int    // positive to add workers, negative to remove
	Reason string // why, for the metrics and the operator

	// Target names the worker to drain on a scale-down. Carried here rather than
	// left to the caller because the autoscaler chose it from the state it just
	// examined; a caller re-deriving "least loaded" from newer state can pick a
	// different worker than the one the decision was reasoned about.
	Target WorkerID
}

// Autoscaler decides the worker count.
//
// Scaling on queue wait rather than utilization is deliberate: a half-utilized
// system with a growing queue is misconfigured, and utilization alone cannot see
// that.
//
// But queue wait alone is NOT sufficient, and the benchmarks disproved the
// stronger claim this comment used to make. It asserted that "a fully utilized
// system with an empty queue is working perfectly and needs nothing." Under
// continuous batching that is false. With 96 concurrent slots across the fleet,
// the workers behave like a processor-sharing server: requests are admitted
// immediately and simply run slower, so at 90% offered load head-of-line wait
// measures 2.8ms against a queue depth of roughly zero while per-request
// throughput is collapsing. The queue only forms once slots are exhausted, which
// makes it a cliff rather than a gradient — the signal arrives after the
// degradation it was supposed to predict.
//
// A correct signal needs a slowdown term alongside queue wait: inter-token
// latency, or tokens-per-second per sequence against its uncontended rate.
// QueueWaitAutoscaler now carries the second of those — see SlowdownScaleUp — and
// treats it as additive: queue wait still fires on its own, and either signal is
// enough to buy capacity. The term's own weakness is named where it is
// implemented: the uncontended baseline is learned as a high-water mark, so a
// fleet that has been saturated since startup learns its degraded rate as normal
// unless the baseline is configured.
type Autoscaler interface {
	// Evaluate is called periodically. Implementations must apply hysteresis —
	// separate scale-up and scale-down thresholds plus a cooldown — or the worker
	// count oscillates around any single threshold, and each oscillation costs a
	// full cold start.
	Evaluate(q Queue, workers []*WorkerState, now time.Time) ScaleDecision
}

// WorkerClient is the control plane's handle on a worker process.
type WorkerClient interface {
	// Dispatch sends a batch. The context carries the deadline, so a worker that
	// stops responding does not pin the dispatcher indefinitely.
	Dispatch(ctx context.Context, batch []*RequestState) error

	// Cancel asks the worker to abandon a sequence and free its blocks. Best
	// effort: the control plane must not assume success, since a worker that has
	// died will never answer, which is exactly when cancellation matters most.
	Cancel(ctx context.Context, id RequestID) error

	// Events streams progress — first token, each token, completion, failure — so
	// the control plane learns the actual cache growth rather than its estimate.
	Events() <-chan WorkerEvent
}

// WorkerEventKind distinguishes the notifications a worker sends.
type WorkerEventKind uint8

const (
	EventFirstToken WorkerEventKind = iota
	EventToken
	EventCompleted
	EventFailed
	EventPreempted
)

// WorkerEvent is one notification from a worker.
type WorkerEvent struct {
	Kind    WorkerEventKind
	Worker  WorkerID
	Request RequestID
	Tokens  int
	Blocks  int

	// Err is the failure message on EventFailed.
	Err string

	// Retryable is the single most consequential bit in this struct. It comes from
	// the worker's `failed` message and decides whether the request is retried
	// elsewhere or abandoned. A control plane that has to guess will guess
	// "retryable", which turns one poison request into a cluster-wide retry storm —
	// so it is carried explicitly rather than inferred, and an absent value must be
	// treated as permanent.
	Retryable bool

	// Text carries the generated token on EventToken and the stop reason on
	// EventCompleted. One ordered channel rather than several, because token order
	// is part of the result and two channels cannot preserve it.
	Text string

	// Reason distinguishes cache_exhausted from reservation_exceeded on
	// EventPreempted. Those call for opposite operator responses — add workers
	// versus fix the estimator — so an undifferentiated preemption is unactionable.
	Reason string

	Timestamp time.Time
}
