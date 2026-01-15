package loom

import (
	"math"
	"sort"
	"sync"
	"time"
)

// PolicyCostModel turns a queued request into the two numbers the policies rank
// on: how much cache it will need, and how long it will occupy a worker.
//
// It lives here, rather than the policies reading EstimatedCost, because
// EstimatedCost is computed by the admission controller and RequestState does not
// carry it. A policy that needs a cost estimate and is not given one has to
// recompute it, so the recomputation is at least explicit and swappable.
//
// The footprint must be the one the system is actually committing, which is why
// Mode is here. Testing feasibility against the worst case while the batcher
// reserves the expected footprint is not "conservative", it is a disagreement: the
// scheduler refuses to dispatch at the concurrency admission is reserving for, and
// the number it gates on is one no other component uses.
//
// ServiceNanos stays worst case deliberately — it is a ranking input, and a ranking
// that moved with a length prediction would let a request's position change as the
// predictor is retuned rather than as its declared shape changes.
type PolicyCostModel struct {
	// TokensPerBlock must match the worker's registered block size, or the
	// feasibility check is comparing blocks of two different sizes. The protocol
	// default is 16.
	TokensPerBlock int

	// Mode selects which footprint Blocks reports, and must match the admission
	// controller's. SetReservationMode on a policy is how the owner of the queue keeps
	// the two in step rather than configuring the same decision twice.
	Mode ReservationMode

	// Length predicts output length for Mode == ReserveExpected. It is only consulted
	// for a request whose carried estimate is empty, since the estimate admission
	// computed is the number that was actually committed.
	Length OutputLengthModel

	// PrefillNanosPerToken is compute-bound and roughly linear in prompt length.
	PrefillNanosPerToken int64

	// DecodeNanosPerToken is memory-bandwidth-bound and paid per generated token,
	// which is why it dominates for anything but a very long prompt.
	DecodeNanosPerToken int64
}

// DefaultPolicyCostModel is a small-model-on-CPU shaped estimate. The absolute
// numbers matter far less than their ratio, since every policy here only ever
// compares two requests' costs to each other.
func DefaultPolicyCostModel() PolicyCostModel {
	return PolicyCostModel{
		TokensPerBlock:       16,
		PrefillNanosPerToken: int64(200 * time.Microsecond),
		DecodeNanosPerToken:  int64(20 * time.Millisecond),
	}
}

// Blocks is the cache footprint a dispatch of this request would reserve, in
// blocks. The ceiling division is where internal fragmentation comes from: one
// token into a block holds the whole block.
//
// Under ReservePeak that is the worst case. Under ReserveExpected it is the
// expected footprint — the same number the admission controller charged and the
// batcher will allocate — floored by what the sequence already spans, because a
// preempted sequence being re-admitted is past its original estimate and the
// blocks it needs are not negotiable.
func (m PolicyCostModel) Blocks(st *RequestState) int {
	if m.Mode == ReserveExpected {
		need := st.Est.ExpectedCacheBlocks
		if need <= 0 && m.Length != nil {
			need = m.blocksFor(st.PromptTokens + m.Length.ExpectedTokens(&st.Request))
		}
		if spanned := m.blocksFor(st.PromptTokens + st.GeneratedTokens); spanned > need {
			need = spanned
		}
		if need > 0 {
			return need
		}
		// Nothing to go on: no carried estimate and no length model. The peak is the
		// only number available, and over-reserving in the feasibility test only costs a
		// scheduling round.
	}
	return m.blocksFor(st.PromptTokens + st.MaxOutputTokens)
}

func (m PolicyCostModel) blocksFor(tokens int) int {
	tpb := m.TokensPerBlock
	if tpb <= 0 {
		tpb = 16
	}
	if tokens < 0 {
		tokens = 0
	}
	return (tokens + tpb - 1) / tpb
}

// ServiceNanos is the predicted total occupancy of a request.
func (m PolicyCostModel) ServiceNanos(st *RequestState) int64 {
	p, d := m.PrefillNanosPerToken, m.DecodeNanosPerToken
	if p <= 0 {
		p = int64(200 * time.Microsecond)
	}
	if d <= 0 {
		d = int64(20 * time.Millisecond)
	}
	return int64(st.PromptTokens)*p + int64(st.MaxOutputTokens)*d
}

// policyBase holds what all three policies share: the cost model, the feasibility
// test, and the bounded-skip rule.
type policyBase struct {
	Cost PolicyCostModel

	// ServiceNanosFn overrides the cost model's service estimate, so a caller that
	// has a better predictor (a measured per-tenant output-length distribution, for
	// instance) can supply it without reimplementing a policy.
	ServiceNanosFn func(*RequestState) int64

	// MaxSkips bounds how far past the policy's first choice a single decision will
	// look for something that currently fits.
	//
	// Skipping at all is necessary: a request needing more cache than any worker
	// has free right now would otherwise block every smaller request behind it,
	// and head-of-line blocking on a queue whose head is the most expensive item
	// is a stall, not a scheduling decision. Skipping without a bound is the
	// opposite failure — the expensive request is passed over forever, because
	// every time capacity frees, a cheap arrival takes it.
	MaxSkips int

	// SkipGrace is how long a request may be skipped before it stops yielding.
	// Once a skipped request has waited this long, Next stops dispatching past it so
	// the running set drains and the request runs as soon as capacity exists. This is
	// the half that makes "eventually runs" true; MaxSkips alone only bounds the work
	// per decision, not the wait per request.
	//
	// The stall is scoped to the workers that could actually host the overdue request.
	// A fleet-wide stall was measured at about 14% of throughput and 43 rejections on
	// a fleet whose largest request's peak reservation was a significant fraction of
	// one worker's cache: every worker drained to admit one request, including workers
	// that could never have hosted it. Draining the candidates it is waiting for is
	// what frees it; draining the rest is pure loss.
	SkipGrace time.Duration

	// stalls counts the decisions a grace stall suppressed, and stalled names the
	// overdue request of the most recent one. Counted because the convoy is otherwise
	// invisible: throughput falls, nothing is logged, and nothing tells an operator
	// they are near the cliff.
	stallMu sync.Mutex
	stalls  uint64
	stalled RequestID
}

const (
	defaultMaxSkips  = 8
	defaultSkipGrace = 2 * time.Second
)

// SetReservationMode tells the policy which footprint admission is committing, so
// its feasibility test gates on the number the rest of the system uses. est, when
// non-nil, also supplies the block size and the length model, which removes the
// other way these two can silently disagree.
//
// Call it before the policy is in use — the scheduler does it at construction. It is
// configuration, not a runtime control, and it is not synchronized.
func (b *policyBase) SetReservationMode(mode ReservationMode, est *CostEstimator) {
	b.Cost.Mode = mode
	if est == nil {
		return
	}
	if est.Policy.TokensPerBlock > 0 {
		b.Cost.TokensPerBlock = est.Policy.TokensPerBlock
	}
	b.Cost.Length = est.Length
}

// ReservationMode reports what the feasibility test is gating on.
func (b *policyBase) ReservationMode() ReservationMode { return b.Cost.Mode }

func (b *policyBase) maxSkips() int {
	if b.MaxSkips <= 0 {
		return defaultMaxSkips
	}
	return b.MaxSkips
}

func (b *policyBase) skipGrace() time.Duration {
	if b.SkipGrace <= 0 {
		return defaultSkipGrace
	}
	return b.SkipGrace
}

func (b *policyBase) serviceNanos(st *RequestState) int64 {
	if b.ServiceNanosFn != nil {
		return b.ServiceNanosFn(st)
	}
	return b.Cost.ServiceNanos(st)
}

// feasible reports whether st could be dispatched to some worker right now.
//
// An empty worker list means the caller did not supply worker state, which is
// read as "unknown" and permits everything. Reading it as "no capacity" would
// make every policy return nil forever, turning a missing wiring step into a
// silent total stall.
func (b *policyBase) feasible(st *RequestState, workers []*WorkerState) bool {
	if len(workers) == 0 {
		return true
	}
	for _, w := range workers {
		if b.fitsOn(st, w) {
			return true
		}
	}
	return false
}

// fitsOn reports whether st could be dispatched to this worker right now.
func (b *policyBase) fitsOn(st *RequestState, w *WorkerState) bool {
	if w == nil || w.Draining {
		return false
	}
	if st.Model != "" && w.Model != "" && st.Model != w.Model {
		return false
	}
	if w.MaxBatchSize > 0 && w.ActiveSequences >= w.MaxBatchSize {
		return false
	}
	return w.FreeBlocks() >= b.Cost.Blocks(st)
}

// drainTargetFor is the worker a stalled request is waiting for: the one that is
// closest to having room for it among those that could host it at all.
//
// Scoping the grace stall needs exactly one such worker. Waiting on the emptiest
// candidate is what makes the wait shortest, and nil — no worker could ever host
// this request even empty — means stalling would accomplish nothing, since the
// capacity it is waiting for does not exist.
func (b *policyBase) drainTargetFor(st *RequestState, workers []*WorkerState) *WorkerState {
	need := b.Cost.Blocks(st)
	var best *WorkerState
	for _, w := range workers {
		if w == nil || w.Draining {
			continue
		}
		if st.Model != "" && w.Model != "" && st.Model != w.Model {
			continue
		}
		if w.TotalCacheBlocks < need {
			continue
		}
		if best == nil || w.FreeBlocks() > best.FreeBlocks() ||
			(w.FreeBlocks() == best.FreeBlocks() && w.ID < best.ID) {
			best = w
		}
	}
	return best
}

// graceStallApplies decides whether an overdue request has to stop this dispatch.
//
// It stops only the dispatches that would delay it: those that can only be placed
// on the worker it is waiting to drain. A request that fits on some other worker is
// let through, because draining a worker the overdue request is not waiting for
// costs throughput and buys it nothing — that is the convoy, and it was measured at
// about 14% of throughput plus rejections that did not otherwise happen.
//
// The guarantee the grace exists for survives: the one worker the request is
// waiting for stops being chosen for new work, so its running set drains and the
// request becomes feasible.
func (b *policyBase) graceStallApplies(overdue, choice *RequestState, workers []*WorkerState) bool {
	if choice == nil {
		// Nothing was going to be dispatched anyway, so there is no stall to report.
		return false
	}
	target := b.drainTargetFor(overdue, workers)
	if target == nil {
		return false
	}
	for _, w := range workers {
		if w == nil || w == target {
			continue
		}
		if b.fitsOn(choice, w) {
			return false
		}
	}
	return true
}

// GraceStalls reports how many dispatch decisions a SkipGrace stall has
// suppressed, and the overdue request of the most recent one.
//
// Exported because the alternative is a system that quietly trades throughput for
// one request's starvation guarantee with nothing to read: the queue looks healthy,
// the workers look healthy, and throughput is simply lower. A rising count next to
// falling throughput is the signal that the largest request's reservation has grown
// to a significant fraction of one worker's cache.
func (b *policyBase) GraceStalls() (uint64, RequestID) {
	b.stallMu.Lock()
	defer b.stallMu.Unlock()
	return b.stalls, b.stalled
}

func (b *policyBase) noteStall(overdue *RequestState) {
	b.stallMu.Lock()
	b.stalls++
	b.stalled = overdue.ID
	b.stallMu.Unlock()
}

// selectBest ranks the queue with better and returns the highest-ranked request
// that currently fits, within the skip budget.
//
// better must be a strict weak ordering: sort.SliceStable keeps push order among
// equals, which is what makes every policy deterministic for a fixed arrival
// sequence and therefore comparable in a benchmark.
func (b *policyBase) selectBest(q QueueSnapshot, workers []*WorkerState, now time.Time, better func(x, y *RequestState) bool) *RequestState {
	snap, ok := q.(QueueSnapshot)
	if !ok {
		// Without a read side there is nothing to rank. Returning nil rather than
		// guessing keeps a mis-wired queue from looking like a working one.
		return nil
	}

	var cands []*RequestState
	snap.Each(func(st *RequestState) bool {
		cands = append(cands, st)
		return true
	})
	if len(cands) == 0 {
		return nil
	}
	sort.SliceStable(cands, func(i, j int) bool { return better(cands[i], cands[j]) })

	skips := 0
	limit := b.maxSkips()
	grace := b.skipGrace()
	var choice, overdue *RequestState
	for _, st := range cands {
		if b.feasible(st, workers) {
			if choice == nil {
				choice = st
			}
			// Keep scanning: an infeasible request that has already waited out the
			// grace period must stop this dispatch even when something behind it in
			// the ranking does fit. Returning early on the first feasible candidate
			// would make the grace rule unreachable for any policy whose ranking
			// does not happen to put the starved request first — which is exactly
			// the case for shortest-job-first, where the starved request is by
			// construction ranked last.
			continue
		}
		if overdue == nil && waitOf(st, now) >= grace {
			// Highest-ranked overdue request wins: it is the one whose starvation the
			// grace is there to end.
			overdue = st
		}
		if choice == nil {
			skips++
			if skips >= limit {
				// Past the budget, give up for this round rather than walking a deep
				// queue looking for the one request that fits.
				break
			}
		}
	}
	if overdue != nil && b.graceStallApplies(overdue, choice, workers) {
		b.noteStall(overdue)
		return nil
	}
	return choice
}

func waitOf(st *RequestState, now time.Time) time.Duration {
	d := now.Sub(queuedSince(st))
	if d < 0 {
		return 0
	}
	return d
}

// FIFOPolicy schedules in arrival order.
//
// It is the correctness baseline, and it is not merely a baseline: arrival order
// is the only ordering nobody can game, its tail latency is bounded by the work
// ahead of a request rather than by the work that arrives after it, and it has no
// starvation mode to engineer around. What it gives up is mean latency, because
// one long request delays every short one behind it.
type FIFOPolicy struct {
	policyBase
}

func NewFIFOPolicy() *FIFOPolicy {
	return &FIFOPolicy{policyBase{Cost: DefaultPolicyCostModel()}}
}

func (p *FIFOPolicy) Name() string { return "fifo" }

func (p *FIFOPolicy) Next(q QueueSnapshot, workers []*WorkerState, now time.Time) *RequestState {
	return p.selectBest(q, workers, now, func(x, y *RequestState) bool {
		if x.Priority != y.Priority {
			return x.Priority > y.Priority
		}
		return queuedSince(x).Before(queuedSince(y))
	})
}

// ShortestJobFirstPolicy schedules the cheapest request first, with aging.
//
// Plain SJF minimizes mean waiting time — that result is not a heuristic, it is
// provable for a fixed set of jobs — and in a serving system the effect is large,
// because the cost spread between a 30-token completion and a 2000-token one is
// two orders of magnitude. The price is that it starves long requests outright:
// under any arrival rate that keeps the queue non-empty, a long request is passed
// over by every cheaper arrival forever, and its wait is unbounded no matter how
// much capacity is added.
//
// Aging is the fix. A request's effective cost falls as it waits, so a long
// request eventually outranks the short ones. AgingRate is the exchange rate, and
// it is a parameter rather than a constant because the starvation it prevents is
// worth demonstrating: with the rate at zero this policy starves long requests,
// and the test that shows that is next to the test that shows aging stopping it.
type ShortestJobFirstPolicy struct {
	policyBase

	// AgingRate is nanoseconds of cost credited per nanosecond waited. Zero
	// disables aging, giving pure SJF and its starvation. At 1.0 a request's
	// effective cost reaches zero once it has waited its own service time, which
	// bounds its wait at roughly (service time x queue contention) rather than
	// leaving it unbounded.
	AgingRate float64
}

func NewShortestJobFirstPolicy(agingRate float64) *ShortestJobFirstPolicy {
	return &ShortestJobFirstPolicy{
		policyBase: policyBase{Cost: DefaultPolicyCostModel()},
		AgingRate:  agingRate,
	}
}

func (p *ShortestJobFirstPolicy) Name() string { return "sjf" }

// effectiveCost is the service estimate discounted by accumulated wait. It is
// allowed to go negative: a request that has waited several times its own service
// time should outrank one that has waited nothing and costs nothing, and clamping
// at zero would flatten exactly the distinction aging is there to make.
func (p *ShortestJobFirstPolicy) effectiveCost(st *RequestState, now time.Time) float64 {
	c := float64(p.serviceNanos(st))
	if p.AgingRate > 0 {
		c -= p.AgingRate * float64(waitOf(st, now))
	}
	return c
}

func (p *ShortestJobFirstPolicy) Next(q QueueSnapshot, workers []*WorkerState, now time.Time) *RequestState {
	return p.selectBest(q, workers, now, func(x, y *RequestState) bool {
		if x.Priority != y.Priority {
			return x.Priority > y.Priority
		}
		cx, cy := p.effectiveCost(x, now), p.effectiveCost(y, now)
		if cx != cy {
			return cx < cy
		}
		return queuedSince(x).Before(queuedSince(y))
	})
}

// FairSharePolicy bounds what any one tenant can consume.
//
// The scheme is virtual time (start-time fair queueing), not deficit round robin.
// Both can hit the same long-run share, and DRR is cheaper per decision, but
// virtual time is the better fit here for two reasons. First, the quantity being
// shared is service time and it is wildly variable per request; DRR's quantum has
// to be chosen relative to the largest request or a tenant's big request stalls
// across many rounds, and there is no largest request here. Second, virtual time
// has a natural answer for a tenant that arrives mid-stream: start it at the
// current minimum virtual time, so it neither inherits a huge credit for having
// been idle (which would let it monopolize on arrival) nor starts behind (which
// would punish it for being new). A deficit counter has to pick one of those two
// wrong answers.
//
// Each tenant carries a virtual clock. Selecting a request advances that tenant's
// clock by the request's cost divided by the tenant's weight, and the policy
// always picks from the tenant whose clock is furthest behind. A tenant sending
// ten times the work therefore advances its clock ten times as fast and is
// selected proportionally less often, which is the property a tenant with huge
// prompts cannot defeat by sending more of them.
type FairSharePolicy struct {
	policyBase

	mu       sync.Mutex
	weights  map[TenantID]float64
	virtual  map[TenantID]float64
	defaultW float64
}

func NewFairSharePolicy() *FairSharePolicy {
	return &FairSharePolicy{
		policyBase: policyBase{Cost: DefaultPolicyCostModel()},
		weights:    make(map[TenantID]float64),
		virtual:    make(map[TenantID]float64),
		defaultW:   1,
	}
}

func (p *FairSharePolicy) Name() string { return "fair-share" }

// SetWeight sets a tenant's share. A weight of 2 means twice the service time of
// a weight-1 tenant over any interval in which both have work queued. Weights are
// relative, so only their ratios matter.
func (p *FairSharePolicy) SetWeight(t TenantID, w float64) {
	if w <= 0 {
		w = 1
	}
	p.mu.Lock()
	p.weights[t] = w
	p.mu.Unlock()
}

// SetDefaultWeight sets the weight for tenants with no explicit weight.
func (p *FairSharePolicy) SetDefaultWeight(w float64) {
	if w <= 0 {
		w = 1
	}
	p.mu.Lock()
	p.defaultW = w
	p.mu.Unlock()
}

// VirtualTimes copies the per-tenant virtual clocks, for metrics and tests.
func (p *FairSharePolicy) VirtualTimes() map[TenantID]float64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make(map[TenantID]float64, len(p.virtual))
	for t, v := range p.virtual {
		out[t] = v
	}
	return out
}

func (p *FairSharePolicy) weightOf(t TenantID) float64 {
	if w, ok := p.weights[t]; ok {
		return w
	}
	return p.defaultW
}

func (p *FairSharePolicy) Next(q QueueSnapshot, workers []*WorkerState, now time.Time) *RequestState {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.admitNewTenants(q)

	chosen := p.selectBest(q, workers, now, func(x, y *RequestState) bool {
		if x.Priority != y.Priority {
			return x.Priority > y.Priority
		}
		vx, vy := p.virtual[x.Tenant], p.virtual[y.Tenant]
		if vx != vy {
			return vx < vy
		}
		// Within one tenant, arrival order: fairness is between tenants, and
		// reordering inside a tenant would import SJF's starvation without asking.
		return queuedSince(x).Before(queuedSince(y))
	})
	if chosen == nil {
		return nil
	}

	// Charged at selection rather than at completion. Charging on completion would
	// let a tenant with many in-flight requests be selected repeatedly before any
	// of its cost registered, which is precisely the burst this policy exists to
	// bound. The cost is an estimate, and the error is bounded by how wrong the
	// estimate is rather than accumulating, since each request is charged once.
	p.virtual[chosen.Tenant] += float64(p.serviceNanos(chosen)) / p.weightOf(chosen.Tenant)
	return chosen
}

// admitNewTenants gives every queued tenant a virtual clock, seeding a newcomer
// at the current minimum so it starts level with the tenants already competing.
func (p *FairSharePolicy) admitNewTenants(q QueueSnapshot) {
	snap, ok := q.(QueueSnapshot)
	if !ok {
		return
	}
	minVT := math.Inf(1)
	for _, v := range p.virtual {
		if v < minVT {
			minVT = v
		}
	}
	if math.IsInf(minVT, 1) {
		minVT = 0
	}
	snap.Each(func(st *RequestState) bool {
		if _, ok := p.virtual[st.Tenant]; !ok {
			p.virtual[st.Tenant] = minVT
		}
		return true
	})

	// A tenant with nothing queued is forgotten, so an idle tenant does not carry
	// a stale clock that would hand it a windfall or a penalty when it returns —
	// it is reseeded at the minimum on its next arrival. This also keeps the map
	// bounded by live tenants rather than by all tenants ever seen.
	queued := snap.TenantCounts()
	for t := range p.virtual {
		if queued[t] == 0 {
			delete(p.virtual, t)
		}
	}
}
