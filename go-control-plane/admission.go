package loom

import (
	"sort"
	"sync"
	"time"
)

// OutputLengthModel predicts how long a generation will actually be.
//
// This interface exists because the prediction is the single assumption the whole
// capacity plan rests on, and an assumption that important should be nameable,
// swappable, and reportable rather than a constant buried in an estimator. Every
// implementation must return at most req.MaxOutputTokens: the client's ceiling is
// the one hard fact available, and a model that predicts past it is predicting
// something that cannot happen.
type OutputLengthModel interface {
	Name() string
	ExpectedTokens(req *Request) int
}

// FractionOfMax predicts a fixed fraction of the declared ceiling.
//
// Crude but honest about being crude, and it has one property nothing else does:
// it is monotone in MaxOutputTokens, so a client that inflates its ceiling is
// charged for inflating it. The empirical models below are better predictors and
// lose that property.
type FractionOfMax struct {
	Fraction float64

	// Floor is a minimum in tokens, because a prediction of ~0 blocks for a request
	// that will certainly produce some output guarantees a growth failure on the
	// first decode step.
	Floor int
}

func (m FractionOfMax) Name() string { return "fraction-of-max" }

func (m FractionOfMax) ExpectedTokens(req *Request) int {
	if req.MaxOutputTokens <= 0 {
		return 0
	}
	f := m.Fraction
	if f <= 0 {
		f = 1
	}
	if f > 1 {
		f = 1
	}
	n := int(float64(req.MaxOutputTokens) * f)
	if n < m.Floor {
		n = m.Floor
	}
	if n > req.MaxOutputTokens {
		n = req.MaxOutputTokens
	}
	if n < 1 {
		n = 1
	}
	return n
}

// QuantileLengthModel predicts from an observed length distribution.
//
// A quantile rather than a mean, and the choice of quantile is the tuning dial
// between the two failure modes: the mean of a heavy-tailed output-length
// distribution under-reserves for most of the mass that matters, while the p99 is
// barely cheaper than reserving the peak. Sample is copied and sorted on
// construction so the model is immutable and safe to share across admission
// goroutines.
type QuantileLengthModel struct {
	sorted   []int
	quantile float64
}

// NewQuantileLengthModel builds a model from observed generation lengths.
func NewQuantileLengthModel(observed []int, quantile float64) *QuantileLengthModel {
	s := make([]int, len(observed))
	copy(s, observed)
	sort.Ints(s)
	if quantile < 0 {
		quantile = 0
	}
	if quantile > 1 {
		quantile = 1
	}
	return &QuantileLengthModel{sorted: s, quantile: quantile}
}

func (m *QuantileLengthModel) Name() string { return "quantile" }

func (m *QuantileLengthModel) ExpectedTokens(req *Request) int {
	if req.MaxOutputTokens <= 0 {
		return 0
	}
	if len(m.sorted) == 0 {
		return req.MaxOutputTokens
	}
	idx := int(m.quantile * float64(len(m.sorted)-1))
	n := m.sorted[idx]
	if n > req.MaxOutputTokens {
		n = req.MaxOutputTokens
	}
	if n < 1 {
		n = 1
	}
	return n
}

// ReservationMode is which of the two cache estimates admission commits to.
//
// Neither option is free, and the system has to pick one in the open:
//
//   - ReservePeak charges every request its worst case. Nothing is ever preempted
//     for lack of cache, and concurrency collapses to TotalBlocks divided by the
//     worst case — for a 32-token prompt with a 2048-token ceiling that is a
//     handful of sequences on a cache that could hold dozens.
//   - ReserveExpected charges the predicted length. Concurrency rises by roughly
//     the ratio between the ceiling and the real length, and the cost is paid when
//     the prediction is wrong: the cache runs out mid-decode and a running
//     sequence has to be preempted. The victim keeps its generated tokens, so what
//     is paid is a re-prefill over prompt-plus-generated-tokens when it resumes,
//     plus however long it waits for a second admission — not the regeneration of
//     the tokens themselves.
//
// So this is a choice between wasting capacity deterministically and wasting it
// stochastically. Paging is what makes the second option available at all; the
// contiguous policy cannot offer it, because it cannot grow a reservation it
// under-sized.
type ReservationMode uint8

const (
	ReservePeak ReservationMode = iota
	ReserveExpected
)

func (m ReservationMode) String() string {
	if m == ReserveExpected {
		return "expected"
	}
	return "peak"
}

// CostEstimator turns a request's declared shape into the numbers admission and
// scheduling decide on.
type CostEstimator struct {
	Policy CachePolicy
	Length OutputLengthModel

	// PrefillNanosPerToken and DecodeNanosPerToken are the two halves of the time
	// model, kept separate because prefill is compute-bound and roughly linear in
	// prompt length while decode is memory-bandwidth-bound and linear in generated
	// tokens. One blended per-token number cannot predict both a long-prompt and a
	// long-generation request, and deadline feasibility depends on predicting both.
	PrefillNanosPerToken int64
	DecodeNanosPerToken  int64
}

// NewCostEstimator returns an estimator with the given length model, defaulting
// to half of the declared ceiling when none is supplied.
func NewCostEstimator(policy CachePolicy, length OutputLengthModel) *CostEstimator {
	if length == nil {
		length = FractionOfMax{Fraction: 0.5, Floor: 1}
	}
	return &CostEstimator{
		Policy:               policy,
		Length:               length,
		PrefillNanosPerToken: int64(200 * time.Microsecond),
		DecodeNanosPerToken:  int64(5 * time.Millisecond),
	}
}

// Estimate computes peak and expected footprints and a service-time prediction.
func (e *CostEstimator) Estimate(req *Request) EstimatedCost {
	expected := 0
	if e.Length != nil {
		expected = e.Length.ExpectedTokens(req)
	}
	peakTokens := req.PromptTokens + req.MaxOutputTokens
	expectedTokens := req.PromptTokens + expected

	cost := EstimatedCost{
		PrefillTokens:       req.PromptTokens,
		PeakCacheBlocks:     e.Policy.BlocksFor(peakTokens),
		ExpectedCacheBlocks: e.Policy.BlocksFor(expectedTokens),
	}
	cost.EstimatedServiceNanos = int64(req.PromptTokens)*e.PrefillNanosPerToken +
		int64(expected)*e.DecodeNanosPerToken
	return cost
}

// EstimateResumed is Estimate for a sequence that has already generated tokens —
// one that was preempted and is being re-admitted.
//
// Preemption frees the KV blocks and keeps the generated tokens, so resuming is not
// a continuation: the blocks are gone, and the sequence has to re-prefill
// prompt-plus-generated-tokens before it can produce token number n+1. That makes a
// resumed sequence strictly more expensive than its first admission in both
// dimensions, and both are charged here:
//
//   - PrefillTokens covers the generated tokens too, so the service-time estimate
//     includes the recomputation instead of pretending the sequence picks up where
//     it left off.
//   - ExpectedCacheBlocks is at least what the sequence already spans. Reserving the
//     original expected footprint for a sequence that is past it guarantees a growth
//     failure on its first step back, which is a preemption loop rather than a
//     prediction error.
//
// Only the expected footprint moves. The peak is a property of the request's
// declared ceiling and does not change when the sequence is preempted.
func (e *CostEstimator) EstimateResumed(st *RequestState) EstimatedCost {
	cost := e.Estimate(&st.Request)
	if st.GeneratedTokens <= 0 {
		return cost
	}
	expected := 0
	if e.Length != nil {
		expected = e.Length.ExpectedTokens(&st.Request)
	}
	spanned := st.PromptTokens + st.GeneratedTokens
	if expectedSpan := st.PromptTokens + expected; expectedSpan > spanned {
		spanned = expectedSpan
	}
	remaining := expected - st.GeneratedTokens
	if remaining < 0 {
		remaining = 0
	}
	cost.PrefillTokens = st.PromptTokens + st.GeneratedTokens
	cost.ExpectedCacheBlocks = e.Policy.BlocksFor(spanned)
	cost.EstimatedServiceNanos = int64(cost.PrefillTokens)*e.PrefillNanosPerToken +
		int64(remaining)*e.DecodeNanosPerToken
	return cost
}

// ReserveBlocks is the number admission commits under the given mode.
func (e *CostEstimator) ReserveBlocks(cost EstimatedCost, mode ReservationMode) int {
	if mode == ReserveExpected {
		return cost.ExpectedCacheBlocks
	}
	return cost.PeakCacheBlocks
}

// TenantQuota is a hard ceiling on one tenant's consumption.
//
// Quota is not fairness, and conflating the two is the mistake this type exists to
// prevent. Fairness is a *scheduling* property: given a set of queued requests, it
// decides whose goes next, and under light load a single tenant can legitimately
// use the whole machine because nobody else wants it. Quota is an *admission*
// property: it says "never more than this, even if the machine is idle", because
// the limit is a contract or a blast-radius bound rather than a sharing rule. A
// fair-share scheduler alone cannot stop one tenant from occupying every block
// when it is the only tenant present; a quota alone cannot stop it from going
// first every time.
//
// A zero field means unlimited for that dimension.
type TenantQuota struct {
	MaxConcurrentRequests int
	MaxCacheBlocks        int
	MaxTokensPerSecond    float64
}

// tenantUsage is the live consumption a quota is checked against.
type tenantUsage struct {
	requests     int
	blocks       int
	tokensPerSec float64
}

// AdmissionMetrics is a point-in-time copy of the controller's counters.
type AdmissionMetrics struct {
	Accepted uint64
	Queued   uint64
	Rejected uint64

	ByReason map[RejectReason]uint64

	// ReconcileDriftLast is the most recent estimate-minus-reported block count.
	// Positive means the control plane over-reserved and capacity is being wasted;
	// negative means it under-reserved and the next admission may overcommit. The
	// sign is the diagnosis, so it is never stored as an absolute value.
	ReconcileDriftLast int

	// DriftSum keeps the sign so a persistent bias shows up as a trend rather than
	// cancelling out, while DriftAbsSum captures noisy-but-centred drift that
	// DriftSum alone would hide.
	DriftSum    int64
	DriftAbsSum int64
	Reconciles  uint64

	// DriftMaxOver and DriftMaxUnder are the worst excursion in each direction.
	DriftMaxOver  int
	DriftMaxUnder int
}

var _ AdmissionController = (*QuotaAdmissionController)(nil)

// QuotaAdmissionController is the admission controller: it decides whether a
// request enters the system at all, and keeps the per-tenant and per-worker
// accounting that decision needs.
type QuotaAdmissionController struct {
	mu sync.Mutex

	est  *CostEstimator
	mode ReservationMode

	quotas  map[TenantID]TenantQuota
	usage   map[TenantID]*tenantUsage
	charged map[RequestID]chargedCost

	metrics AdmissionMetrics
}

// chargedCost is what a tenant was billed at admission, kept so Complete
// refunds exactly what was charged. Recomputing the estimate at completion would
// refund a different number if the estimator or its length model were ever
// reconfigured in between, and the usage counters would drift permanently.
type chargedCost struct {
	tenant       TenantID
	blocks       int
	tokensPerSec float64
}

// NewQuotaAdmissionController builds a controller. A nil estimator gets the
// default; mode makes the peak-versus-expected choice explicit at construction,
// because it is the kind of decision that must not be implicit.
func NewQuotaAdmissionController(est *CostEstimator, mode ReservationMode, quotas map[TenantID]TenantQuota) *QuotaAdmissionController {
	if est == nil {
		est = NewCostEstimator(CachePolicy{TokensPerBlock: 16, TotalBlocks: 0}, nil)
	}
	q := make(map[TenantID]TenantQuota, len(quotas))
	for k, v := range quotas {
		q[k] = v
	}
	return &QuotaAdmissionController{
		est:     est,
		mode:    mode,
		quotas:  q,
		usage:   make(map[TenantID]*tenantUsage, len(q)),
		charged: make(map[RequestID]chargedCost, 64),
		metrics: AdmissionMetrics{ByReason: make(map[RejectReason]uint64, 8)},
	}
}

// SetQuota installs or replaces a tenant's quota.
func (c *QuotaAdmissionController) SetQuota(t TenantID, q TenantQuota) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.quotas[t] = q
}

// Mode reports the reservation choice in force, so a report can state it.
func (c *QuotaAdmissionController) Mode() ReservationMode { return c.mode }

// Estimator is the cost estimator admission decided with. Exposed so another
// component that has to agree with those numbers — the scheduling policy's
// feasibility test — can take them from here instead of being configured with a
// second copy that can drift.
func (c *QuotaAdmissionController) Estimator() *CostEstimator { return c.est }

// Admit evaluates a request and, on any non-refusal, charges it to its tenant.
//
// The order of the checks is deliberate and is part of the contract, because the
// reason reported to the client is as much of the answer as the decision: the
// permanent and structural refusals are evaluated before the transient ones, so a
// request that can never be served is never reported as "we are busy". Charging
// happens inside the same critical section as the quota check, which is what makes
// the quota a real bound under concurrent submissions rather than a check with a
// race in it.
func (c *QuotaAdmissionController) Admit(req *Request, workers []*WorkerState, q Queue, now time.Time) (AdmitDecision, RejectReason, EstimatedCost) {
	cost := c.est.Estimate(req)
	need := c.est.ReserveBlocks(cost, c.mode)

	// 1. Model availability. Nothing else is worth computing if no live worker
	// serves the model: queueing would be waiting for an event that is not pending.
	var candidates []*WorkerState
	fleetMax := 0
	for _, w := range workers {
		if w.Model != req.Model {
			continue
		}
		// fleetMax spans draining and unhealthy workers on purpose. Whether the
		// request is structurally too large is a property of the hardware that
		// exists, not of what happens to be accepting work this second.
		if w.TotalCacheBlocks > fleetMax {
			fleetMax = w.TotalCacheBlocks
		}
		if w.Draining {
			continue
		}
		candidates = append(candidates, w)
	}
	if fleetMax == 0 {
		return c.refuse(RejectModelUnavailable, cost)
	}

	// 2. Structurally too large. The request would not fit on the largest worker in
	// the fleet even if that worker were completely empty, so no amount of waiting
	// helps. This is checked against the PEAK rather than the reservation: a request
	// admitted on its expected footprint that can provably not reach its ceiling is
	// a request guaranteed to be preempted, and discovering that after it has
	// produced 1500 tokens is worse than refusing it now. Permanent refusal, and the
	// caller must not queue it.
	if cost.PeakCacheBlocks > fleetMax {
		return c.refuse(RejectTooLarge, cost)
	}

	// 3. Tenant quota: the hard ceiling, checked before any capacity question,
	// because a tenant over its quota must be refused whether the machine is full or
	// idle. Refused rather than queued: a quota breach is not a transient shortfall,
	// and letting an over-quota tenant accumulate queue depth would let it consume
	// the one resource the quota did not name.
	if ok, reason, dec := c.checkAndCharge(req, cost, need); !ok {
		return c.refuseWith(reason, dec, cost)
	}

	// From here the request is charged, so every refusal path must refund.

	// 4. Deadline feasibility. If the predicted completion is already past the
	// deadline, serving it burns capacity to produce something nobody will use, and
	// it delays requests that would still have been useful. Refusing now is the
	// cheaper failure.
	if !req.Deadline.IsZero() {
		wait := time.Duration(0)
		if !c.fits(candidates, need) && q != nil {
			// With no free capacity the request waits behind at least the current
			// head-of-line wait. This underestimates — the head still has to be served —
			// but a feasibility check must not refuse requests it is merely pessimistic
			// about, so the conservative direction here is to under-predict the wait.
			wait = q.OldestWait(now)
		}
		if now.Add(wait + time.Duration(cost.EstimatedServiceNanos)).After(req.Deadline) {
			c.refund(req.ID)
			return c.refuse(RejectDeadlineInfeasible, cost)
		}
	}

	// 5. Immediate capacity. Accepted means a specific worker can take it right now.
	if c.fits(candidates, need) {
		c.count(AdmitAccepted, RejectNone)
		return AdmitAccepted, RejectNone, cost
	}

	// 6. No capacity, in the sense that queueing is futile: every worker that is
	// actually accepting work is too small for this request even when empty. The
	// fleet contains something big enough (step 2 passed) but it is draining or
	// dead, so the request would sit in the queue until the fleet changed shape.
	if !c.couldEverFit(candidates, cost.PeakCacheBlocks) {
		c.refund(req.ID)
		return c.refuse(RejectNoCapacity, cost)
	}

	// 7. Queue capacity. Last, because it is the most transient reason of all and
	// reporting it in preference to a structural one would hide the real problem.
	if q == nil || q.Len() >= q.Cap() {
		c.refund(req.ID)
		return c.refuse(RejectQueueFull, cost)
	}

	c.count(AdmitQueued, RejectNone)
	return AdmitQueued, RejectNone, cost
}

// fits reports whether some non-draining worker can take need blocks and another
// sequence right now.
func (c *QuotaAdmissionController) fits(candidates []*WorkerState, need int) bool {
	for _, w := range candidates {
		if w.FreeBlocks() < need {
			continue
		}
		if w.MaxBatchSize > 0 && w.ActiveSequences >= w.MaxBatchSize {
			continue
		}
		return true
	}
	return false
}

// couldEverFit reports whether any worker accepting work is large enough to hold
// the request once it empties.
func (c *QuotaAdmissionController) couldEverFit(candidates []*WorkerState, peak int) bool {
	for _, w := range candidates {
		if w.TotalCacheBlocks >= peak {
			return true
		}
	}
	return false
}

// checkAndCharge tests every quota dimension and bills the tenant atomically if
// all of them pass.
func (c *QuotaAdmissionController) checkAndCharge(req *Request, cost EstimatedCost, need int) (bool, RejectReason, AdmitDecision) {
	// The rate a request demands is its expected output over its expected service
	// time. Prompt tokens are excluded: prefill produces no output tokens, so
	// charging them against a tokens/sec quota would bill a long prompt twice.
	rate := 0.0
	if cost.EstimatedServiceNanos > 0 && c.est.Length != nil {
		rate = float64(c.est.Length.ExpectedTokens(req)) / (float64(cost.EstimatedServiceNanos) / 1e9)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	quota, hasQuota := c.quotas[req.Tenant]
	u := c.usage[req.Tenant]
	if u == nil {
		u = &tenantUsage{}
		c.usage[req.Tenant] = u
	}
	if hasQuota {
		if quota.MaxConcurrentRequests > 0 && u.requests+1 > quota.MaxConcurrentRequests {
			return false, RejectTenantQuota, AdmitRejected
		}
		if quota.MaxCacheBlocks > 0 && u.blocks+need > quota.MaxCacheBlocks {
			return false, RejectTenantQuota, AdmitRejected
		}
		if quota.MaxTokensPerSecond > 0 && u.tokensPerSec+rate > quota.MaxTokensPerSecond {
			return false, RejectTenantQuota, AdmitRejected
		}
	}
	u.requests++
	u.blocks += need
	u.tokensPerSec += rate
	c.charged[req.ID] = chargedCost{tenant: req.Tenant, blocks: need, tokensPerSec: rate}
	return true, RejectNone, AdmitAccepted
}

// Complete releases a request's quota charge. It must be called on completion,
// failure, cancellation and permanent preemption alike: a charge that is only
// refunded on the happy path means a tenant's effective quota shrinks every time
// anything goes wrong.
func (c *QuotaAdmissionController) Complete(id RequestID) {
	c.refund(id)
}

func (c *QuotaAdmissionController) refund(id RequestID) {
	c.mu.Lock()
	defer c.mu.Unlock()
	ch, ok := c.charged[id]
	if !ok {
		return
	}
	delete(c.charged, id)
	u := c.usage[ch.tenant]
	if u == nil {
		return
	}
	u.requests--
	u.blocks -= ch.blocks
	u.tokensPerSec -= ch.tokensPerSec
	if u.requests <= 0 {
		// Dropping the entry at zero keeps float error in tokensPerSec from
		// accumulating across a long run of admissions and refunds.
		delete(c.usage, ch.tenant)
	}
}

// Rebill adjusts a charged request's block cost, for the case where the
// reservation grew after admission — a sequence admitted on its expected
// footprint that then outran the prediction. Without this the tenant's block
// quota would bound the estimate rather than the reality.
func (c *QuotaAdmissionController) Rebill(id RequestID, blocks int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	ch, ok := c.charged[id]
	if !ok {
		return
	}
	u := c.usage[ch.tenant]
	if u == nil {
		return
	}
	u.blocks += blocks - ch.blocks
	ch.blocks = blocks
	c.charged[id] = ch
}

// TenantUsage exposes a tenant's live charge, for metrics and assertions.
func (c *QuotaAdmissionController) TenantUsage(t TenantID) (requests, blocks int, tokensPerSec float64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	u := c.usage[t]
	if u == nil {
		return 0, 0, 0
	}
	return u.requests, u.blocks, u.tokensPerSec
}

// Reconcile applies a worker's heartbeat report of actual cache usage.
//
// The report wins, unconditionally. The control plane predicts cache growth
// because it has to decide before the fact, but the worker is the only party that
// measures it, and a controller that trusts its own prediction over a direct
// measurement drifts in one direction until it overcommits and something is killed
// mid-generation.
//
// The WorkerState is mutated in place and is not internally synchronized, so the
// caller must own it for the duration — in the real system the registry does,
// because heartbeats for one worker arrive on one connection and are therefore
// already serialized. The controller's own counters are safe under concurrency.
//
// The returned drift is estimate minus reported, and the sign is the whole value
// of the number: positive means capacity is sitting idle inside reservations that
// will never be used, negative means the next admission is being made against a
// cache fuller than the controller thinks.
//
// The estimate side comes from w.UsedCacheBlocks, which means the CALLER must have
// put its own prediction there before calling. This is the easy mistake: a caller
// that passes the WorkerState straight out of the registry is passing the previous
// heartbeat's report, so it measures the worker against itself and the drift is
// always approximately zero — the metric looks healthy precisely when it is telling
// you nothing. Overwrite w.UsedCacheBlocks with the scheduler's own reservation
// mirror first.
func (c *QuotaAdmissionController) Reconcile(w *WorkerState, reportedUsed, reportedActive int) int {
	drift := w.UsedCacheBlocks - reportedUsed
	w.UsedCacheBlocks = reportedUsed
	w.ActiveSequences = reportedActive

	c.mu.Lock()
	defer c.mu.Unlock()
	c.metrics.Reconciles++
	c.metrics.ReconcileDriftLast = drift
	c.metrics.DriftSum += int64(drift)
	if drift < 0 {
		c.metrics.DriftAbsSum += int64(-drift)
		if drift < c.metrics.DriftMaxUnder {
			c.metrics.DriftMaxUnder = drift
		}
	} else {
		c.metrics.DriftAbsSum += int64(drift)
		if drift > c.metrics.DriftMaxOver {
			c.metrics.DriftMaxOver = drift
		}
	}
	return drift
}

// Metrics returns a copy, including a copy of the reason map, so a caller cannot
// read a half-updated counter or mutate the controller's state through the map.
func (c *QuotaAdmissionController) Metrics() AdmissionMetrics {
	c.mu.Lock()
	defer c.mu.Unlock()
	m := c.metrics
	m.ByReason = make(map[RejectReason]uint64, len(c.metrics.ByReason))
	for k, v := range c.metrics.ByReason {
		m.ByReason[k] = v
	}
	return m
}

func (c *QuotaAdmissionController) refuse(reason RejectReason, cost EstimatedCost) (AdmitDecision, RejectReason, EstimatedCost) {
	c.count(AdmitRejected, reason)
	return AdmitRejected, reason, cost
}

func (c *QuotaAdmissionController) refuseWith(reason RejectReason, dec AdmitDecision, cost EstimatedCost) (AdmitDecision, RejectReason, EstimatedCost) {
	c.count(dec, reason)
	return dec, reason, cost
}

func (c *QuotaAdmissionController) count(dec AdmitDecision, reason RejectReason) {
	c.mu.Lock()
	defer c.mu.Unlock()
	switch dec {
	case AdmitAccepted:
		c.metrics.Accepted++
	case AdmitQueued:
		c.metrics.Queued++
	default:
		c.metrics.Rejected++
	}
	if reason != RejectNone {
		c.metrics.ByReason[reason]++
	}
}
