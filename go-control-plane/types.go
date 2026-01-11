// Package loom is the control plane for an inference serving platform.
//
// The engineering problem here is not the model. It is that inference requests
// have wildly variable cost, that the memory a request will need grows as it
// runs, and that the resource it competes for cannot be oversubscribed the way
// CPU can. A scheduler that treats requests as interchangeable units of work —
// which is what a conventional queue does — will either leave the accelerator
// idle or run it out of memory mid-generation.
package loom

import (
	"time"
)

// RequestID identifies one inference request for its whole lifetime, including
// across a retry on a different worker.
type RequestID string

// WorkerID identifies a model worker process.
type WorkerID string

// ModelID names a model. A worker serves exactly one at a time, because loading
// a second one into the same accelerator is what the memory accounting exists to
// prevent.
type ModelID string

// TenantID is the unit of fairness and quota.
type TenantID string

// Phase distinguishes the two halves of an LLM request, which behave so
// differently that scheduling them identically is the most common mistake in a
// naive serving system.
//
// Prefill processes the whole prompt in one pass: compute-bound, roughly linear
// in prompt length, and it produces the initial KV cache. Decode then generates
// one token at a time: memory-bandwidth-bound, and each step grows the KV cache
// by one position per sequence. A long prompt is expensive once; a long
// generation is expensive forever, and it holds memory the entire time.
type Phase uint8

const (
	PhaseQueued Phase = iota
	PhasePrefill
	PhaseDecode
	PhaseDone
)

func (p Phase) String() string {
	switch p {
	case PhaseQueued:
		return "queued"
	case PhasePrefill:
		return "prefill"
	case PhaseDecode:
		return "decode"
	case PhaseDone:
		return "done"
	}
	return "unknown"
}

// Request is one inference request as the control plane sees it.
type Request struct {
	ID       RequestID
	Tenant   TenantID
	Model    ModelID
	Priority int // higher runs sooner, within fairness limits

	// Prompt is the text to run. The control plane never inspects it — scheduling
	// decisions are driven entirely by PromptTokens — but it has to be carried
	// because the worker needs it, and routing it around the scheduler out of band
	// means a dispatched request can arrive with an empty prompt and no component
	// is obviously at fault.
	Prompt string

	// PromptTokens is known at admission. This is what makes cost estimation
	// possible at all: the prefill cost and the initial cache footprint are both
	// functions of it.
	PromptTokens int

	// MaxOutputTokens is the client's declared ceiling. It is an upper bound, not
	// a prediction — most requests stop earlier — so using it directly as the
	// memory reservation is safe but wastes capacity, and using the mean is
	// efficient but can run the accelerator out of memory. The admission
	// controller's handling of this gap is the interesting part of the design.
	MaxOutputTokens int

	ArrivalTime time.Time

	// Deadline, when non-zero, is when the result stops being useful. A request
	// whose deadline has passed while queued should be dropped rather than run:
	// serving it consumes capacity to produce something nobody wants.
	Deadline time.Time

	// Cancelled is closed by the caller to abandon the request. The scheduler
	// must release its reservation promptly rather than at natural completion,
	// or a client that gave up still ties up memory.
	Cancelled <-chan struct{}
}

// EstimatedCost is what the admission controller computes before deciding.
type EstimatedCost struct {
	// PrefillTokens is PromptTokens; named separately because the cost model
	// weights prefill and decode differently.
	PrefillTokens int

	// PeakCacheBlocks is the KV-cache block count the request would hold at its
	// worst, if it generates MaxOutputTokens.
	PeakCacheBlocks int

	// ExpectedCacheBlocks is the block count at the expected output length,
	// which is what the scheduler actually plans around.
	ExpectedCacheBlocks int

	// EstimatedServiceNanos is the predicted total occupancy, used by
	// shortest-job-first and by deadline feasibility checks.
	EstimatedServiceNanos int64
}

// RequestState is the control plane's live view of one in-flight request.
type RequestState struct {
	Request

	Phase Phase

	// GeneratedTokens advances during decode. Together with PromptTokens it gives
	// the request's current cache footprint, which is the number that actually
	// constrains admission of the next request.
	GeneratedTokens int

	Worker WorkerID

	// HeldBlocks is the cache blocks currently allocated. Tracked separately from
	// the estimate because the estimate is a guess and this is the truth.
	HeldBlocks int

	// Est is the cost the admission controller computed, carried forward rather
	// than recomputed.
	//
	// Admission estimates the cost to decide whether to accept; the scheduler needs
	// the same number to order work and to check feasibility. Recomputing it
	// independently lets the two disagree about the same request, and a scheduler
	// that thinks a request is cheaper than admission did will overcommit. One
	// computation, carried.
	Est EstimatedCost

	// QueuedAt is when this request most recently started waiting for a slot. It is
	// reset on every re-queue, including preemption, so it measures the current wait
	// rather than the lifetime. Scheduling and the autoscaler read it; anything
	// reported to a client measures from Request.ArrivalTime instead, which never
	// moves.
	QueuedAt    time.Time
	StartedAt   time.Time
	FirstToken  time.Time
	CompletedAt time.Time

	Attempts int
}

// WorkerState is what the registry knows about a worker.
type WorkerState struct {
	ID    WorkerID
	Model ModelID

	// TotalCacheBlocks is the worker's KV-cache capacity in blocks. This is the
	// real constraint on concurrency: not a request count, not a memory
	// percentage, but how many fixed-size cache blocks exist.
	TotalCacheBlocks int
	UsedCacheBlocks  int

	// MaxBatchSize caps sequences in one forward pass, independent of memory.
	MaxBatchSize int

	ActiveSequences int

	// RecentTokensPerSecond is a decayed throughput measure, used by the
	// autoscaler and for placement among workers of differing speed.
	RecentTokensPerSecond float64

	LastHeartbeat time.Time

	// Draining means finish what you have and accept nothing new. Distinct from
	// unhealthy: a draining worker's in-flight work is still valid, so killing it
	// would fail requests unnecessarily.
	Draining bool
}

// FreeBlocks is capacity remaining. Never negative; a negative value would mean
// the accounting has drifted from reality, which the registry treats as a bug.
func (w *WorkerState) FreeBlocks() int {
	if w.UsedCacheBlocks >= w.TotalCacheBlocks {
		return 0
	}
	return w.TotalCacheBlocks - w.UsedCacheBlocks
}

// Healthy reports whether the worker has heartbeated recently enough.
func (w *WorkerState) Healthy(now time.Time, timeout time.Duration) bool {
	return now.Sub(w.LastHeartbeat) < timeout
}

// AdmitDecision is the admission controller's answer.
//
// Three outcomes rather than two, because "no" has two very different meanings.
// Rejected is permanent: the request cannot be served under any near-future
// state, so telling the client immediately is kinder than queueing it. Queued is
// "not now", and the difference matters to a client deciding whether to retry.
type AdmitDecision uint8

const (
	AdmitAccepted AdmitDecision = iota
	AdmitQueued
	AdmitRejected
)

func (d AdmitDecision) String() string {
	switch d {
	case AdmitAccepted:
		return "accepted"
	case AdmitQueued:
		return "queued"
	case AdmitRejected:
		return "rejected"
	}
	return "unknown"
}

// RejectReason explains a refusal, so the metrics can distinguish overload from
// malformed input from an impossible deadline.
type RejectReason uint8

const (
	RejectNone RejectReason = iota
	RejectQueueFull
	RejectTenantQuota
	RejectTooLarge // would not fit on any worker even when empty
	RejectDeadlineInfeasible
	RejectNoCapacity
	RejectModelUnavailable
)

func (r RejectReason) String() string {
	switch r {
	case RejectNone:
		return "none"
	case RejectQueueFull:
		return "queue_full"
	case RejectTenantQuota:
		return "tenant_quota"
	case RejectTooLarge:
		return "too_large"
	case RejectDeadlineInfeasible:
		return "deadline_infeasible"
	case RejectNoCapacity:
		return "no_capacity"
	case RejectModelUnavailable:
		return "model_unavailable"
	}
	return "unknown"
}
