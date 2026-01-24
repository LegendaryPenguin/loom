package loom

import (
	"fmt"
	"sync"
	"time"
)

// AutoscalerConfig is the scaling policy.
//
// Every threshold here comes in a pair with a gap between them, and the gap is
// the point. A single threshold with an action on each side guarantees
// oscillation: the scale-up that relieves the pressure takes the signal below the
// threshold, which triggers a scale-down, which recreates the pressure. The system
// then pays a full cold start per cycle and spends its life starting and stopping
// workers instead of serving requests.
type AutoscalerConfig struct {
	Clock Clock

	// ScaleUpWait is the head-of-line queue wait that means more capacity is
	// needed. Head-of-line wait, not utilization: a fully utilized system with an
	// empty queue is working perfectly, and a half-utilized one with a growing
	// queue is misconfigured. Utilization cannot distinguish those.
	//
	// CPU percentage is not a signal here at all. The control plane is not the
	// bottleneck and the workers are not CPU-bound — they are bound by
	// accelerator memory bandwidth and by KV-cache blocks. A worker at 100% CPU
	// may be perfectly healthy and one at 20% may be refusing admissions for
	// want of cache blocks, so a CPU-driven scaler would add workers when nothing
	// is wrong and sit still when the thing that is actually scarce runs out.
	ScaleUpWait time.Duration

	// ScaleDownWait is the wait below which capacity looks surplus. It must be
	// well below ScaleUpWait; the band between them is where no action is taken.
	ScaleDownWait time.Duration

	// ScaleUpDepth is the queue depth that, combined with cache pressure, counts
	// as demand even when head-of-line wait is still low — which happens on a
	// sharp arrival burst, before anything has had time to wait.
	ScaleUpDepth int

	// CacheUtilScaleUp and CacheUtilScaleDown bracket cache occupancy. Cache
	// blocks, not memory percentage: blocks are the resource admission actually
	// counts.
	CacheUtilScaleUp   float64
	CacheUtilScaleDown float64

	// SlowdownScaleUp is the per-sequence slowdown that counts as demand even when
	// the queue is empty. 2.0 means the typical sequence is generating at half the
	// rate it generates uncontended.
	//
	// This is the second signal, and under continuous batching it is the one that
	// fires first. Head-of-line wait assumes a queue forms when the fleet runs out of
	// capacity, which is true of a server that admits a bounded number of requests and
	// makes the rest wait. With a loose batch cap the fleet is closer to a
	// processor-sharing server: an extra request is admitted immediately and slows
	// every resident sequence a little instead of waiting. Measured at 90% offered
	// load on the standard fleet, head-of-line wait read 2.8ms at a queue depth of ~0
	// while end-to-end latency had already risen substantially — so the primary signal
	// stays flat through the whole degradation and only moves when batch slots run out,
	// which is a cliff rather than a ramp.
	//
	// Zero takes the default; any negative value disables the term and restores
	// queue-wait-only behaviour, which is what the comparison in the autoscaler tests
	// needs and what an operator who distrusts the baseline would set.
	SlowdownScaleUp float64

	// SlowdownScaleDown is the slowdown below which the fleet no longer looks
	// degraded. Like every other threshold here it is a separate number from its
	// scale-up twin, and the band between them is where nothing happens. Defaults to
	// halfway between no slowdown at all and SlowdownScaleUp.
	SlowdownScaleDown float64

	// UncontendedPerSequenceTokensPerSecond, when non-zero, is the per-sequence
	// decode rate of an uncontended worker, and overrides the learned baseline.
	//
	// Worth setting if the fleet is never lightly loaded. The learned baseline is the
	// best per-sequence rate ever observed, which is a true uncontended rate only if
	// the fleet was once uncontended; a system that has been saturated since startup
	// learns its own degraded rate as "normal" and the slowdown term reads ~1.0. That
	// failure is silent, which is why the override exists and why SlowdownSamples is
	// reported in the signals.
	UncontendedPerSequenceTokensPerSecond float64

	// SlowdownMinSamples is how many observations a worker's learned baseline needs
	// before the slowdown term is trusted at all. A baseline from one sample is a
	// single EMA reading of one batch composition, which is noise.
	SlowdownMinSamples int

	// MinTokensPerSecond, when non-zero, suppresses scale-down while aggregate
	// throughput is above it. Throughput high with the queue empty means the
	// current fleet is exactly right; shrinking it is how a healthy system gets
	// pushed into overload by its own scaler.
	MinTokensPerSecond float64

	// SustainedFor is how long demand must persist before acting. This is the
	// spike filter: a 200ms burst does not justify a worker that will not be
	// useful for 30 seconds and will then be surplus.
	SustainedFor time.Duration

	// ColdStartDuration is how long a new worker takes to load its model and
	// become useful. Until it elapses, a started worker is capacity that exists
	// but cannot serve, and scaling up again for demand it will absorb
	// double-provisions the fleet.
	ColdStartDuration time.Duration

	ScaleUpCooldown   time.Duration
	ScaleDownCooldown time.Duration

	// MaxScaleUpStep bounds a single decision. Unbounded steps turn a measurement
	// error into a fleet-sized bill.
	MaxScaleUpStep int

	MinWorkers int
	MaxWorkers int
}

func (c AutoscalerConfig) withDefaults() AutoscalerConfig {
	if c.Clock == nil {
		c.Clock = SystemClock{}
	}
	if c.ScaleUpWait <= 0 {
		c.ScaleUpWait = 500 * time.Millisecond
	}
	if c.ScaleDownWait <= 0 || c.ScaleDownWait >= c.ScaleUpWait {
		c.ScaleDownWait = c.ScaleUpWait / 5
	}
	if c.ScaleUpDepth <= 0 {
		c.ScaleUpDepth = 8
	}
	if c.CacheUtilScaleUp <= 0 {
		c.CacheUtilScaleUp = 0.85
	}
	if c.CacheUtilScaleDown <= 0 || c.CacheUtilScaleDown >= c.CacheUtilScaleUp {
		c.CacheUtilScaleDown = 0.35
	}
	if c.SlowdownScaleUp == 0 {
		// On by default, at a factor that is unambiguous: a sequence running at half its
		// uncontended rate is a user-visible halving of inter-token latency, not jitter.
		c.SlowdownScaleUp = 2.0
	}
	if c.SlowdownScaleDown <= 1 || c.SlowdownScaleDown >= c.SlowdownScaleUp {
		c.SlowdownScaleDown = 1 + (c.SlowdownScaleUp-1)/2
	}
	if c.SlowdownMinSamples <= 0 {
		c.SlowdownMinSamples = 8
	}
	if c.SustainedFor <= 0 {
		c.SustainedFor = 3 * time.Second
	}
	if c.ColdStartDuration <= 0 {
		c.ColdStartDuration = 30 * time.Second
	}
	if c.ScaleUpCooldown <= 0 {
		c.ScaleUpCooldown = c.ColdStartDuration
	}
	if c.ScaleDownCooldown <= 0 {
		// Longer than scale-up on purpose: adding a worker you did not need costs
		// money, removing one you did need costs failed requests.
		c.ScaleDownCooldown = 3 * c.ColdStartDuration
	}
	if c.MaxScaleUpStep <= 0 {
		c.MaxScaleUpStep = 2
	}
	if c.MinWorkers < 0 {
		c.MinWorkers = 0
	}
	if c.MaxWorkers < c.MinWorkers {
		c.MaxWorkers = c.MinWorkers
	}
	return c
}

// ScaleSignals is the input the autoscaler derived from the queue and the fleet,
// exposed so an operator can see the same numbers the decision was made from.
type ScaleSignals struct {
	HeadOfLineWait  time.Duration
	QueueDepth      int
	CacheUtil       float64
	TokensPerSecond float64
	ServingWorkers  int
	StartingWorkers int
	DrainingWorkers int

	// PerSequenceTokensPerSecond is the mean decode rate one resident sequence is
	// getting, across workers that are running anything.
	PerSequenceTokensPerSecond float64

	// Slowdown is that rate against the uncontended baseline: 1.0 is uncontended, 2.0
	// is half speed. Zero means not yet measurable — too few samples, or no worker
	// running anything — and is never read as "fine".
	Slowdown float64

	// SlowdownSamples is the smallest number of observations behind any worker's
	// baseline, so an operator can tell a trustworthy 1.0 from an untrained one.
	SlowdownSamples int
}

// QueueWaitAutoscaler is the Autoscaler implementation.
//
// It holds state between calls — the sustained-demand timers, the cooldown, and
// the set of cold-starting workers — because every one of the behaviours that
// stops this from oscillating is a function of history rather than of the current
// sample.
type QueueWaitAutoscaler struct {
	cfg AutoscalerConfig

	mu sync.Mutex

	// upSince and downSince are when the current run of above/below-threshold
	// observations began, zeroed the moment the signal crosses back. Only a
	// contiguous run counts, which is what makes a spike a non-event.
	upSince   time.Time
	downSince time.Time

	lastAction     time.Time
	lastActionSign int

	// starting holds the launch time of each worker this scaler asked for and
	// which has not yet finished its cold start.
	starting []time.Time

	// baselines is the per-worker uncontended per-sequence rate, learned as a
	// high-water mark. Per worker rather than per fleet because workers differ in
	// speed, and a fleet-wide baseline would read the fastest worker's rate as every
	// worker's and call the slow ones degraded forever.
	baselines map[WorkerID]*rateBaseline

	signals ScaleSignals
	actions int
}

// NewQueueWaitAutoscaler builds the scaler.
// rateBaseline is one worker's learned uncontended per-sequence decode rate.
type rateBaseline struct {
	best    float64
	samples int
}

func NewQueueWaitAutoscaler(cfg AutoscalerConfig) *QueueWaitAutoscaler {
	return &QueueWaitAutoscaler{
		cfg:       cfg.withDefaults(),
		baselines: make(map[WorkerID]*rateBaseline),
	}
}

// NoteWorkerStarted tells the scaler that a worker was launched by something
// other than its own last decision — an operator, or a restart after a crash.
// Without this the scaler would see the resulting capacity as a surprise and
// could double-provision for the same demand.
func (a *QueueWaitAutoscaler) NoteWorkerStarted(at time.Time) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.starting = append(a.starting, at)
}

// Signals returns the inputs to the most recent Evaluate.
func (a *QueueWaitAutoscaler) Signals() ScaleSignals {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.signals
}

// LastActionSign is +1 after a scale-up, -1 after a scale-down, 0 before any
// action. Reversals of this value are what oscillation looks like from outside.
func (a *QueueWaitAutoscaler) LastActionSign() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.lastActionSign
}

// Actions counts the decisions with a non-zero delta, which is the number the
// hysteresis test asserts on: a correct scaler driven by a signal hovering at a
// threshold takes very few actions, and a thresholded one takes one per sample.
func (a *QueueWaitAutoscaler) Actions() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.actions
}

// Evaluate recommends a worker-count change.
func (a *QueueWaitAutoscaler) Evaluate(q Queue, workers []*WorkerState, now time.Time) ScaleDecision {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.pruneStartingLocked(now)

	s := a.observeLocked(q, workers, now)
	a.signals = s

	// Capacity that exists but is not yet serving still counts against demand:
	// the whole purpose of modelling the cold start is to not buy the same
	// capacity twice while waiting for it.
	effective := s.ServingWorkers + s.StartingWorkers

	if effective < a.cfg.MinWorkers {
		// The floor is not subject to cooldown or to sustained demand. Being below
		// it is a configuration violation, not a load observation.
		return a.actLocked(now, a.cfg.MinWorkers-effective, "below_min_workers")
	}

	up := a.demandUpLocked(s)
	if up {
		a.downSince = time.Time{}
		if a.upSince.IsZero() {
			a.upSince = now
		}
	} else {
		a.upSince = time.Time{}
	}

	if up {
		if now.Sub(a.upSince) < a.cfg.SustainedFor {
			return ScaleDecision{Delta: 0, Reason: "demand_not_sustained"}
		}
		want := a.wantedStepLocked(s)
		if s.StartingWorkers >= want {
			// A brief spike and a sustained overload look identical for the first
			// SustainedFor; what separates them is whether the capacity already on
			// its way is enough. If it is, waiting costs nothing but a little
			// latency, and acting costs a worker that will be surplus.
			return ScaleDecision{Delta: 0, Reason: fmt.Sprintf("cold_start_in_flight:%d", s.StartingWorkers)}
		}
		if effective >= a.cfg.MaxWorkers {
			return ScaleDecision{Delta: 0, Reason: "at_max_workers"}
		}
		if !a.lastAction.IsZero() && now.Sub(a.lastAction) < a.cfg.ScaleUpCooldown {
			return ScaleDecision{Delta: 0, Reason: "scale_up_cooldown"}
		}
		delta := want - s.StartingWorkers
		if room := a.cfg.MaxWorkers - effective; delta > room {
			delta = room
		}
		if delta <= 0 {
			return ScaleDecision{Delta: 0, Reason: "at_max_workers"}
		}
		return a.actLocked(now, delta, fmt.Sprintf("queue_wait=%s depth=%d slowdown=%.2fx sustained=%s",
			s.HeadOfLineWait.Round(time.Millisecond), s.QueueDepth, s.Slowdown,
			now.Sub(a.upSince).Round(time.Millisecond)))
	}

	down := a.demandDownLocked(s)
	if !down {
		a.downSince = time.Time{}
		return ScaleDecision{Delta: 0, Reason: "within_hysteresis_band"}
	}
	if a.downSince.IsZero() {
		a.downSince = now
	}
	if now.Sub(a.downSince) < a.cfg.SustainedFor {
		return ScaleDecision{Delta: 0, Reason: "surplus_not_sustained"}
	}
	if s.StartingWorkers > 0 {
		// Removing a worker while another is still cold-starting means the fleet
		// just paid for a cold start and is now throwing away warm capacity.
		return ScaleDecision{Delta: 0, Reason: "cold_start_in_flight"}
	}
	if effective <= a.cfg.MinWorkers {
		return ScaleDecision{Delta: 0, Reason: "at_min_workers"}
	}
	if !a.lastAction.IsZero() && now.Sub(a.lastAction) < a.cfg.ScaleDownCooldown {
		return ScaleDecision{Delta: 0, Reason: "scale_down_cooldown"}
	}

	// One at a time. Removing n workers from a single observation assumes the
	// surplus is real for all n, and the next observation is a cheaper way to find
	// out than n cold starts would be to undo.
	reason := "surplus"
	victim, ok := a.leastLoaded(workers)
	if ok {
		reason = fmt.Sprintf("surplus drain=%s wait=%s util=%.2f",
			victim, s.HeadOfLineWait.Round(time.Millisecond), s.CacheUtil)
	}
	// The victim travels with the decision. It was chosen from the state this
	// Evaluate examined, and a caller re-deriving "least loaded" from newer state can
	// pick a different worker than the one the decision was reasoned about.
	dec := a.actLocked(now, -1, reason)
	dec.Target = victim
	return dec
}

func (a *QueueWaitAutoscaler) actLocked(now time.Time, delta int, reason string) ScaleDecision {
	if delta == 0 {
		return ScaleDecision{Delta: 0, Reason: reason}
	}
	a.lastAction = now
	a.actions++
	if delta > 0 {
		a.lastActionSign = 1
		for i := 0; i < delta; i++ {
			a.starting = append(a.starting, now)
		}
	} else {
		a.lastActionSign = -1
	}
	a.upSince = time.Time{}
	a.downSince = time.Time{}
	return ScaleDecision{Delta: delta, Reason: reason}
}

// pruneStartingLocked drops workers whose cold start has elapsed. After that they
// are either serving — in which case the fleet count already includes them — or
// they failed to start, in which case continuing to credit them as incoming
// capacity would suppress the scale-up that is genuinely needed.
func (a *QueueWaitAutoscaler) pruneStartingLocked(now time.Time) {
	keep := a.starting[:0]
	for _, t := range a.starting {
		if now.Sub(t) < a.cfg.ColdStartDuration {
			keep = append(keep, t)
		}
	}
	a.starting = keep
}

func (a *QueueWaitAutoscaler) observeLocked(q Queue, workers []*WorkerState, now time.Time) ScaleSignals {
	s := ScaleSignals{StartingWorkers: len(a.starting)}
	if q != nil {
		s.HeadOfLineWait = q.OldestWait(now)
		s.QueueDepth = q.Len()
	}
	var used, total int
	for _, w := range workers {
		if w == nil {
			continue
		}
		if w.Draining {
			s.DrainingWorkers++
			// A draining worker's capacity is leaving, so it must not count toward
			// the fleet size the decision is made against — otherwise scale-down
			// drains the fleet one worker at a time without ever noticing.
			continue
		}
		s.ServingWorkers++
		used += w.UsedCacheBlocks
		total += w.TotalCacheBlocks
		s.TokensPerSecond += w.RecentTokensPerSecond
	}
	if total > 0 {
		s.CacheUtil = float64(used) / float64(total)
	}
	a.observeSlowdownLocked(&s, workers)
	return s
}

// observeSlowdownLocked fills in the per-sequence rate and the slowdown against the
// uncontended baseline.
//
// The baseline is the best per-sequence rate the worker has ever shown. The
// justification is that decode is memory-bandwidth-bound, so one sequence in the
// batch is the fastest a sequence can run on that device: every additional resident
// sequence shares the same bandwidth, which makes the maximum observed
// per-sequence rate the uncontended rate rather than merely a lucky sample. What it
// cannot do is invent an observation that never happened — see
// UncontendedPerSequenceTokensPerSecond.
//
// The fleet slowdown is the *minimum* across measurable workers, which is the
// conservative reading: one straggler is a straggler, and capacity should be bought
// when the typical worker has slowed, not when any worker has.
func (a *QueueWaitAutoscaler) observeSlowdownLocked(s *ScaleSignals, workers []*WorkerState) {
	slowest, rateSum, measured := 0.0, 0.0, 0
	minSamples := 0
	untrained := false
	seen := make(map[WorkerID]bool, len(workers))
	for _, w := range workers {
		if w != nil {
			seen[w.ID] = true
		}
		if w == nil || w.Draining || w.ActiveSequences <= 0 || w.RecentTokensPerSecond <= 0 {
			continue
		}
		perSeq := w.RecentTokensPerSecond / float64(w.ActiveSequences)
		rateSum += perSeq
		measured++

		base := a.cfg.UncontendedPerSequenceTokensPerSecond
		samples := a.cfg.SlowdownMinSamples
		if base <= 0 {
			b := a.baselines[w.ID]
			if b == nil {
				b = &rateBaseline{}
				a.baselines[w.ID] = b
			}
			b.samples++
			if perSeq > b.best {
				b.best = perSeq
			}
			base, samples = b.best, b.samples
		}
		if samples < a.cfg.SlowdownMinSamples || base <= 0 {
			// Not enough history to call this worker slow. Left unmeasurable rather than
			// reported as healthy, because a scaler that reads "no data" as "fine" is the
			// blindness this term exists to remove.
			untrained = true
			continue
		}
		if minSamples == 0 || samples < minSamples {
			minSamples = samples
		}
		if f := base / perSeq; slowest == 0 || f < slowest {
			slowest = f
		}
	}
	// Baselines of workers that are gone are dropped, so a long-lived scaler's memory
	// is bounded by the live fleet rather than by every worker it has ever seen.
	for id := range a.baselines {
		if !seen[id] {
			delete(a.baselines, id)
		}
	}
	if measured == 0 {
		return
	}
	s.PerSequenceTokensPerSecond = rateSum / float64(measured)
	if untrained {
		return
	}
	s.Slowdown = slowest
	s.SlowdownSamples = minSamples
}

func (a *QueueWaitAutoscaler) demandUpLocked(s ScaleSignals) bool {
	if s.ServingWorkers == 0 && s.QueueDepth > 0 {
		return true
	}
	if s.HeadOfLineWait >= a.cfg.ScaleUpWait {
		return true
	}
	// The queue can stay empty through the whole degradation, so the slowdown term is
	// checked independently of depth and wait rather than as a tie-break on them.
	if a.cfg.SlowdownScaleUp > 0 && s.Slowdown >= a.cfg.SlowdownScaleUp {
		return true
	}
	// A burst that has not had time to wait yet still shows up as depth against a
	// cache that has no room to absorb it.
	return s.QueueDepth >= a.cfg.ScaleUpDepth && s.CacheUtil >= a.cfg.CacheUtilScaleUp
}

func (a *QueueWaitAutoscaler) demandDownLocked(s ScaleSignals) bool {
	if s.QueueDepth > 0 {
		return false
	}
	// An empty queue is exactly what a processor-sharing fleet looks like while it
	// saturates, so surplus has to mean idle *and* not slow. Without this the scaler
	// reads a degrading system as over-provisioned and shrinks it.
	if a.cfg.SlowdownScaleUp > 0 && s.Slowdown >= a.cfg.SlowdownScaleDown {
		return false
	}
	if s.HeadOfLineWait > a.cfg.ScaleDownWait {
		return false
	}
	if s.CacheUtil > a.cfg.CacheUtilScaleDown {
		return false
	}
	if a.cfg.MinTokensPerSecond > 0 && s.TokensPerSecond > a.cfg.MinTokensPerSecond {
		return false
	}
	return true
}

// wantedStepLocked sizes the scale-up from how far past the threshold the wait
// is. Twice the threshold suggests roughly twice the missing capacity; the step
// cap keeps a pathological sample from ordering the whole fleet again.
func (a *QueueWaitAutoscaler) wantedStepLocked(s ScaleSignals) int {
	want := 1
	if a.cfg.ScaleUpWait > 0 {
		if n := int(s.HeadOfLineWait / a.cfg.ScaleUpWait); n > want {
			want = n
		}
	}
	if want > a.cfg.MaxScaleUpStep {
		want = a.cfg.MaxScaleUpStep
	}
	return want
}

// ScaleDownTarget picks the worker to drain.
//
// Least loaded first, measured by in-flight sequences and then by held cache
// blocks: that worker's drain completes soonest, so the capacity actually leaves
// the fleet instead of lingering for the lifetime of the longest generation on a
// busy worker. Already-draining workers are skipped so a second scale-down does
// not target the same one twice.
func (a *QueueWaitAutoscaler) ScaleDownTarget(workers []*WorkerState) (WorkerID, bool) {
	return a.leastLoaded(workers)
}

func (a *QueueWaitAutoscaler) leastLoaded(workers []*WorkerState) (WorkerID, bool) {
	var best *WorkerState
	for _, w := range workers {
		if w == nil || w.Draining {
			continue
		}
		if best == nil || lessLoaded(w, best) {
			best = w
		}
	}
	if best == nil {
		return "", false
	}
	return best.ID, true
}

func lessLoaded(a, b *WorkerState) bool {
	if a.ActiveSequences != b.ActiveSequences {
		return a.ActiveSequences < b.ActiveSequences
	}
	if a.UsedCacheBlocks != b.UsedCacheBlocks {
		return a.UsedCacheBlocks < b.UsedCacheBlocks
	}
	// Deterministic tie-break, so a test and an operator see the same choice.
	return a.ID < b.ID
}

var _ Autoscaler = (*QueueWaitAutoscaler)(nil)
