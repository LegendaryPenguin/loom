package simulator

import (
	"fmt"
	"sort"
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
)

// BatcherFactory builds the batcher for one worker.
//
// Per worker rather than per fleet, because a batcher owns a KV cache and a cache
// belongs to exactly one device. A single shared batcher would let one worker's
// preemption free another worker's memory, which is the bug the per-worker cache
// exists to make impossible.
type BatcherFactory func(cache loom.KVCache, est *loom.CostEstimator, mode loom.ReservationMode) loom.Batcher

// Crash schedules a worker failure.
type Crash struct {
	At     time.Duration
	Worker loom.WorkerID
}

// Config is one simulation.
type Config struct {
	// Workers is the starting fleet. Template is the shape of a worker the
	// autoscaler adds; when its ID is empty, generated ids are used.
	Workers  []WorkerSpec
	Template WorkerSpec

	Latency LatencyModel

	QueueCapacity int
	Policy        loom.SchedulePolicy
	Batcher       BatcherFactory
	Estimator     *loom.CostEstimator
	Mode          loom.ReservationMode
	Quotas        map[loom.TenantID]loom.TenantQuota

	// Autoscaler nil means scaling is off, which is one half of the autoscaling
	// experiment.
	Autoscaler *loom.QueueWaitAutoscaler

	// AutoscalerFactory builds the autoscaler against the simulation's own clock.
	//
	// It exists because every hysteresis behaviour in QueueWaitAutoscaler — the sustained
	// window, both cooldowns, the cold-start credit — is a function of elapsed time, so a
	// scaler constructed with any other clock has all of them frozen and degenerates into
	// the thresholded scaler whose oscillation the hysteresis exists to prevent. The
	// clock is created inside New, so a caller cannot supply it any other way.
	AutoscalerFactory func(loom.Clock) *loom.QueueWaitAutoscaler

	// TickInterval is the control-plane loop period: heartbeats, reconciliation,
	// deadline expiry, reaping, and the autoscaler evaluation all happen on it.
	TickInterval time.Duration

	// HeartbeatInterval and DeadAfterMisses configure the registry's health machine.
	HeartbeatInterval time.Duration
	DeadAfterMisses   int

	Arrivals []Arrival
	Crashes  []Crash

	// MaxDuration bounds the run. It must be long enough for the backlog to drain
	// after the last arrival, or the tail percentiles are percentiles of whatever
	// happened to finish, which is the measurement error the overload experiment is
	// specifically about.
	MaxDuration time.Duration

	// SampleInterval turns on the time series the autoscaling and failure experiments
	// plot. Zero disables it; the series is the largest allocation in a long run.
	SampleInterval time.Duration

	// RetryOnWorkerLoss runs reaped requests back through the registry's retry
	// classifier. Off means a worker loss fails its in-flight work, which is the
	// baseline the recovery number is measured against.
	RetryOnWorkerLoss bool
}

// Sample is one point in the time series.
type Sample struct {
	T              time.Duration
	QueueDepth     int
	HeadOfLineWait time.Duration
	Workers        int
	Serving        int
	Starting       int
	Draining       int
	Running        int
	CacheUtil      float64
	Completed      int
	Rejected       int
}

// TenantStat is one tenant's submitted-versus-served ledger, which is the only way
// to see a fairness result: served work alone cannot distinguish a tenant that was
// throttled from one that sent nothing.
type TenantStat struct {
	Submitted       int
	SubmittedTokens int
	Accepted        int
	Completed       int
	Rejected        int
	ServedTokens    int
	Latencies       []time.Duration
}

// LeakCheck is the end-of-run accounting invariant.
//
// Every field must be zero in a run that ended with an empty queue. A non-zero
// value is a capacity leak, and a leak is invisible from the outside — every
// individual request looks fine — until the accounting says the cluster is full
// while the accelerators sit idle.
type LeakCheck struct {
	CacheBlocksHeld   int
	CacheSequences    int
	ControlPlaneHeld  int
	TenantQuotaHeld   int
	UnfinishedTracked int

	// Details names the offending sequences, because a leak is only debuggable if the
	// report says which reservation was left behind.
	Details []string
}

// Clean reports whether the run ended with nothing held.
func (l LeakCheck) Clean() bool {
	return l.CacheBlocksHeld == 0 && l.CacheSequences == 0 && l.ControlPlaneHeld == 0 &&
		l.TenantQuotaHeld == 0 && l.UnfinishedTracked == 0
}

// Result is everything one run measured.
type Result struct {
	// Elapsed runs from the start to the last observable event — an arrival, a crash,
	// or a completion. It is the denominator of every rate.
	Elapsed time.Duration

	// WallEnd is where the simulated clock finally stopped, which is later than Elapsed
	// when the run hit MaxDuration with work it could not place. A WallEnd far past
	// Elapsed means the run was truncated and the tail percentiles are suspect.
	WallEnd time.Duration

	Submitted int
	Accepted  int
	Enqueued  int
	Rejected  int
	Completed int
	Failed    int
	TimedOut  int

	RejectByReason map[loom.RejectReason]int

	// Latency, QueueWait and TTFT cover COMPLETED requests only. Reported separately
	// from the rejection rate on purpose: a system that refuses half the offered load
	// and serves the rest in 100ms is not worse than one that accepts everything and
	// serves it in 40 seconds, and a single blended number cannot say which happened.
	Latency   Dist
	QueueWait Dist
	TTFT      Dist

	OutputTokens int

	Preemptions uint64
	Steps       uint64

	// DoubleDispatches counts requests that were dispatched to a second worker while a
	// first still held a reservation for them. Non-zero means the batcher left a sequence
	// both running and re-queued; the driver reclaims the stale assignment, but the count
	// is reported because the condition is a contract violation, not a workload property.
	DoubleDispatches int

	// SilentRetirements counts sequences the batcher dropped from the running set without
	// reporting them in either return value of Step. Non-zero means the driver had to
	// reconcile against the previous running set to avoid leaking their reservations; it
	// is a contract gap, not a workload property, so it is reported rather than hidden.
	SilentRetirements int

	// SequenceSteps is the sum of batch sizes over every forward pass the fleet ran.
	// Divided by Steps it gives the mean batch size, which is the only direct measure of
	// whether a batching window is actually filling batches.
	SequenceSteps uint64

	MaxConcurrency  int
	MeanConcurrency float64

	// PeakCacheSequences is the most sequences any single worker's cache held at once,
	// which is the concurrency figure the paged-versus-contiguous comparison is about.
	PeakCacheSequences int

	// PeakFragmentation is the most free-but-unusable capacity observed on one worker:
	// free blocks that exist and cannot be handed out because they are not adjacent.
	// Structurally zero for the paged policy, which is the entire argument for paging.
	PeakFragmentation int

	// PeakCacheUtil is the highest fleet-wide block occupancy observed.
	PeakCacheUtil float64

	// ReservedBlockBeyondNeed is reserved capacity that was never going to be used:
	// summed over completed requests, the blocks the reservation held minus the blocks
	// the sequence actually spanned at its stopping point. This is where
	// peak-reservation waste shows up, and it does NOT show up as reconcile drift,
	// because the cache really did allocate what the control plane asked for.
	ReservedBlockBeyondNeed int
	NeededBlocks            int

	Admission loom.AdmissionMetrics

	// WorkerLoss is the failure experiment's output.
	ReapedWorkers  int
	ReapedRequests int
	ReleasedOnReap int
	RetriedOnReap  int
	LostOnReap     int
	// RecoveryTime is measured from the LAST kill to the moment head-of-line wait came
	// back under its pre-failure high-water mark, and is only set if the run was observed
	// to degrade first. Zero means it never recovered within the run.
	RecoveryTime     time.Duration
	Recovered        bool
	FirstCrashAt     time.Duration
	ScaleUpActions   int
	ScaleDownActions int
	PeakWorkers      int
	FinalWorkers     int

	PerTenant map[loom.TenantID]*TenantStat
	Samples   []Sample
	Leak      LeakCheck
}

// Throughput is completed requests per second of virtual time.
func (r Result) Throughput() float64 {
	if r.Elapsed <= 0 {
		return 0
	}
	return float64(r.Completed) / r.Elapsed.Seconds()
}

// Goodput is output tokens per second, counting only tokens that reached a client.
// Tokens produced by a sequence that was later preempted or lost are not goodput,
// however busy they kept the accelerator.
func (r Result) Goodput() float64 {
	if r.Elapsed <= 0 {
		return 0
	}
	return float64(r.OutputTokens) / r.Elapsed.Seconds()
}

// RejectRate is refused over submitted.
func (r Result) RejectRate() float64 {
	if r.Submitted == 0 {
		return 0
	}
	return float64(r.Rejected) / float64(r.Submitted)
}

// Sim is one simulation in progress.
type Sim struct {
	cfg   Config
	clock *VirtualClock

	queue    *loom.BoundedQueue
	adm      *loom.QuotaAdmissionController
	registry *loom.Registry
	est      *loom.CostEstimator

	workers []*simWorker

	// retired holds workers that have finished draining. Their step and preemption
	// counters are still part of the run's totals, so dropping the struct on removal
	// would silently under-report the work a scaled-down fleet did.
	retired []*simWorker
	nextID  int

	states map[loom.RequestID]*loom.RequestState
	actual map[loom.RequestID]int

	completedLat  []time.Duration
	completedWait []time.Duration
	completedTTFT []time.Duration

	res Result

	deadline   time.Time
	lastSample time.Duration

	// concurrency is the live sequence count across the fleet, integrated over time
	// so the mean is time-weighted rather than event-weighted. An event-weighted mean
	// would over-count the instants just after a completion, which are exactly the
	// instants with the fewest running sequences.
	//
	// Recounted from the running sets rather than adjusted incrementally. Incremental
	// adjustment is what a real metrics path would do and it is what this did first, and
	// it went negative: ContinuousBatcher.Step can report the same sequence as preempted
	// twice within one step, so one departure arrived as two. A derived gauge cannot
	// drift from the thing it describes.
	concurrency  int
	concIntegral float64
	lastIntegral time.Time

	// crashT is when the first injected crash fires, and recovered is when the queue's
	// head-of-line wait first came back under the pre-crash high-water mark.
	crashT        time.Duration
	lastCrashT    time.Duration
	preCrashWait  time.Duration
	preCrashDepth int
	degraded      bool
	recovered     bool

	// pendingExternal counts arrivals and crashes still scheduled. Counted explicitly
	// rather than read off the event heap, because the heap also holds the worker and
	// tick events the loop itself creates — making the termination condition depend on
	// the heap makes it self-sustaining, and the run never ends.
	pendingExternal int

	// lastActivity is the last instant at which the system did something observable.
	// It, not the configured MaxDuration, is the denominator of every throughput number:
	// dividing by a bound that the run did not actually use understates throughput by
	// however much slack the bound had.
	lastActivity time.Time
}

// New builds a simulation. It does not run it.
func New(cfg Config) *Sim {
	if cfg.TickInterval <= 0 {
		cfg.TickInterval = 20 * time.Millisecond
	}
	if cfg.HeartbeatInterval <= 0 {
		cfg.HeartbeatInterval = 500 * time.Millisecond
	}
	if cfg.DeadAfterMisses <= 0 {
		cfg.DeadAfterMisses = 3
	}
	if cfg.QueueCapacity <= 0 {
		cfg.QueueCapacity = 1024
	}
	if cfg.Policy == nil {
		cfg.Policy = loom.NewFIFOPolicy()
	}
	if cfg.Latency == (LatencyModel{}) {
		cfg.Latency = DefaultLatencyModel()
	}
	if cfg.Batcher == nil {
		cfg.Batcher = ContinuousBatcherFactory(0, loom.PreemptNewest)
	}
	if cfg.MaxDuration <= 0 {
		last := time.Duration(0)
		for _, a := range cfg.Arrivals {
			if a.Offset > last {
				last = a.Offset
			}
		}
		cfg.MaxDuration = last + 5*time.Minute
	}

	clock := NewVirtualClock()
	est := cfg.Estimator
	if est == nil {
		tpb := 16
		if len(cfg.Workers) > 0 && cfg.Workers[0].TokensPerBlock > 0 {
			tpb = cfg.Workers[0].TokensPerBlock
		}
		est = loom.NewCostEstimator(loom.CachePolicy{TokensPerBlock: tpb}, nil)
	}

	s := &Sim{
		cfg:   cfg,
		clock: clock,
		est:   est,
		adm:   loom.NewQuotaAdmissionController(est, cfg.Mode, cfg.Quotas),
		registry: loom.NewRegistry(loom.RegistryConfig{
			Clock:             clock,
			HeartbeatInterval: cfg.HeartbeatInterval,
			DeadAfterMisses:   cfg.DeadAfterMisses,
		}),
		states: make(map[loom.RequestID]*loom.RequestState, len(cfg.Arrivals)),
		actual: make(map[loom.RequestID]int, len(cfg.Arrivals)),
		res: Result{
			RejectByReason: make(map[loom.RejectReason]int, 8),
			PerTenant:      make(map[loom.TenantID]*TenantStat, 8),
		},
		deadline:     clock.Start().Add(cfg.MaxDuration),
		lastIntegral: clock.Start(),
	}

	if cfg.AutoscalerFactory != nil {
		s.cfg.Autoscaler = cfg.AutoscalerFactory(clock)
	}

	s.queue = loom.NewBoundedQueue(cfg.QueueCapacity, cfg.Policy)
	s.queue.SetClock(clock.Now)
	s.queue.SetWorkerSource(s.eligibleStates)

	for _, spec := range cfg.Workers {
		s.addWorker(spec, clock.Now())
	}
	return s
}

// eligibleStates is what the scheduling policy and the admission controller see: the
// control plane's own view of workers that are accepting work. Not the simulated
// truth — a control plane that could read the truth would never need to reconcile.
func (s *Sim) eligibleStates() []*loom.WorkerState {
	out := make([]*loom.WorkerState, 0, len(s.workers))
	for _, w := range s.workers {
		if !w.alive || !w.ready || w.cp.Draining {
			continue
		}
		st := w.cp
		out = append(out, &st)
	}
	return out
}

// allStates includes cold-starting and draining workers, which is what the
// autoscaler needs: it counts draining capacity as leaving and has its own notion of
// capacity still on its way.
func (s *Sim) allStates() []*loom.WorkerState {
	out := make([]*loom.WorkerState, 0, len(s.workers))
	for _, w := range s.workers {
		if !w.alive {
			continue
		}
		st := w.cp
		out = append(out, &st)
	}
	return out
}

func (s *Sim) addWorker(spec WorkerSpec, now time.Time) *simWorker {
	if spec.ID == "" {
		s.nextID++
		spec.ID = loom.WorkerID("w-auto-" + itoa(s.nextID))
	}
	w := newSimWorker(spec, s.est, s.cfg.Mode, s.cfg.Batcher, now)
	s.workers = append(s.workers, w)
	if w.ready {
		// Registered only once the model is loaded, which is what PROTOCOL.md describes: the
		// worker sends `register` on connect and it connects when it is ready to serve.
		// Registering at launch instead is a real trap — a cold start longer than three
		// heartbeat intervals means the registry declares the brand-new worker dead before it
		// ever becomes useful, and the reaper then counts it as a failure.
		_ = s.registry.Register(&w.truth)
	}
	if !w.ready {
		s.clock.At(event{at: now.Add(spec.ColdStart), kind: evWorkerReady, worker: string(spec.ID)})
	}
	if n := len(s.workers); n > s.res.PeakWorkers {
		s.res.PeakWorkers = n
	}
	return w
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// Run executes the simulation to completion and returns the measurements.
func (s *Sim) Run() Result {
	start := s.clock.Start()

	for i := range s.cfg.Arrivals {
		s.clock.At(event{at: start.Add(s.cfg.Arrivals[i].Offset), kind: evArrival, req: i})
		s.pendingExternal++
	}
	for _, c := range s.cfg.Crashes {
		for _, w := range s.workers {
			if w.spec.ID == c.Worker {
				s.clock.At(event{at: start.Add(c.At), kind: evWorkerCrash, worker: string(c.Worker)})
				s.pendingExternal++
				if s.crashT == 0 || c.At < s.crashT {
					s.crashT = c.At
				}
				if c.At > s.lastCrashT {
					s.lastCrashT = c.At
				}
			}
		}
	}
	s.clock.At(event{at: start.Add(s.cfg.TickInterval), kind: evTick})

	for {
		ev, ok := s.clock.Next()
		if !ok {
			break
		}
		now := s.clock.Now()
		if now.After(s.deadline) {
			break
		}
		s.integrate(now)

		switch ev.kind {
		case evArrival:
			s.pendingExternal--
			s.lastActivity = now
			s.onArrival(&s.cfg.Arrivals[ev.req], now)
		case evTick:
			s.onTick(now)
		case evWorkerStep:
			s.onWorkerStep(loom.WorkerID(ev.worker), now)
		case evWorkerReady:
			if w := s.workerByID(loom.WorkerID(ev.worker)); w != nil {
				w.ready = true
				w.truth.LastHeartbeat = now
				_ = s.registry.Register(&w.truth)
				s.kick(w, now)
			}
		case evWorkerCrash:
			s.pendingExternal--
			s.lastActivity = now
			s.onCrash(loom.WorkerID(ev.worker), now)
		}
	}

	s.finish()
	return s.res
}

// integrate advances the time-weighted concurrency accumulator to now.
// recount refreshes the concurrency gauge from the fleet's running sets.
func (s *Sim) recount() {
	n := 0
	for _, w := range s.workers {
		if w.alive {
			n += len(w.running)
		}
	}
	s.concurrency = n
	if n > s.res.MaxConcurrency {
		s.res.MaxConcurrency = n
	}
}

func (s *Sim) integrate(now time.Time) {
	dt := now.Sub(s.lastIntegral)
	if dt > 0 {
		s.concIntegral += float64(s.concurrency) * dt.Seconds()
	}
	s.lastIntegral = now
}

func (s *Sim) tenant(t loom.TenantID) *TenantStat {
	ts := s.res.PerTenant[t]
	if ts == nil {
		ts = &TenantStat{}
		s.res.PerTenant[t] = ts
	}
	return ts
}

func (s *Sim) onArrival(a *Arrival, now time.Time) {
	req := a.Req
	req.ArrivalTime = now
	if a.DeadlineAfter > 0 {
		req.Deadline = now.Add(a.DeadlineAfter)
	}

	s.res.Submitted++
	ts := s.tenant(req.Tenant)
	ts.Submitted++
	ts.SubmittedTokens += a.ActualOutputTokens

	st := &loom.RequestState{Request: req, Phase: loom.PhaseQueued, QueuedAt: now}

	dec, reason, cost := s.adm.Admit(&req, s.allStates(), s.queue, now)
	st.Est = cost

	if dec == loom.AdmitRejected {
		s.res.Rejected++
		s.res.RejectByReason[reason]++
		ts.Rejected++
		return
	}

	if !s.queue.Push(st) {
		// Admit said there was room and the push failed anyway. Only reachable if
		// something else filled the queue in between; in this single-threaded
		// simulation it is unreachable, and asserting on it would be a test of the
		// simulator rather than of the system, so it is counted as a refusal.
		s.adm.Complete(req.ID)
		s.res.Rejected++
		s.res.RejectByReason[loom.RejectQueueFull]++
		ts.Rejected++
		return
	}

	s.states[req.ID] = st
	s.actual[req.ID] = a.ActualOutputTokens
	s.res.Accepted++
	ts.Accepted++
	if dec == loom.AdmitQueued {
		s.res.Enqueued++
	}

	s.kickAll(now)
}

// kick schedules a step for one worker if it is idle and able.
func (s *Sim) kick(w *simWorker, now time.Time) {
	if w == nil || !w.alive || !w.ready || w.stepping {
		return
	}
	w.stepping = true
	s.clock.At(event{at: now, kind: evWorkerStep, worker: string(w.spec.ID)})
}

func (s *Sim) kickAll(now time.Time) {
	for _, w := range s.workers {
		s.kick(w, now)
	}
}

func (s *Sim) onCrash(id loom.WorkerID, now time.Time) {
	w := s.workerByID(id)
	if w == nil || !w.alive {
		return
	}
	w.alive = false

	// The crash is a process death: whatever it was running is gone, and its cache
	// contents with it. Nothing is released through the control plane here, which is
	// the point — the only path back for that capacity is the registry's reaper, and
	// if the reaper does not report the blocks they are leaked.
	w.running = nil
	s.recount()
	s.res.FirstCrashAt = s.crashT
}

func (s *Sim) onTick(now time.Time) {
	// Deadline expiry first: running a request whose deadline has passed burns
	// capacity to produce something nobody wants, so it must not be dispatched on this
	// tick.
	for _, st := range s.queue.DropExpired(now) {
		s.adm.Complete(st.ID)
		s.res.TimedOut++
		delete(s.states, st.ID)
	}

	for _, w := range s.workers {
		if !w.alive || !w.ready {
			continue
		}
		used := w.cache.Used()
		active := len(w.running)
		w.truth.UsedCacheBlocks = used
		w.truth.ActiveSequences = active
		w.truth.RecentTokensPerSecond = w.lastTPS
		_ = s.registry.Heartbeat(w.spec.ID, used, active, w.lastTPS)

		// Reconcile overwrites the control plane's estimate with the worker's report,
		// which is the protocol's rule. The clamp afterwards is not cosmetic: once the
		// aggregate has been overwritten it no longer equals the sum of the per-sequence
		// reservations the control plane is still tracking, so releasing those
		// reservations one by one can drive the aggregate below zero, and a negative
		// UsedCacheBlocks reads as more free capacity than the worker has.
		w.cp.RecentTokensPerSecond = w.lastTPS
		s.adm.Reconcile(&w.cp, used, active)
		s.clampCP(w)
	}

	s.observePeaks()

	if dead := s.registry.ReapDead(now, 0); len(dead) > 0 {
		s.onReap(dead, now)
	}

	if s.cfg.Autoscaler != nil {
		s.onAutoscale(now)
	}

	s.removeDrained()

	if s.cfg.SampleInterval > 0 {
		el := now.Sub(s.clock.Start())
		if el-s.lastSample >= s.cfg.SampleInterval || len(s.res.Samples) == 0 {
			s.lastSample = el
			s.res.Samples = append(s.res.Samples, s.sample(el, now))
		}
	}

	s.trackRecovery(now)

	// The decision to keep ticking is made BEFORE kicking the workers, because kicking
	// enqueues events and a condition that counted those events would keep itself alive
	// forever.
	more := now.Before(s.deadline) && (s.queue.Len() > 0 || s.anyRunning() || s.pendingExternal > 0)
	s.kickAll(now)
	if more {
		s.clock.At(event{at: now.Add(s.cfg.TickInterval), kind: evTick})
	}
}

// observePeaks records the cache high-water marks. Taken on every control-plane tick
// rather than only when the time series is enabled, because the cache experiment needs
// them and the series is an expensive thing to turn on just to get a maximum.
func (s *Sim) observePeaks() {
	var used, total int
	for _, w := range s.workers {
		if !w.alive {
			continue
		}
		if c, ok := w.cache.(interface{ Sequences() int }); ok {
			if n := c.Sequences(); n > s.res.PeakCacheSequences {
				s.res.PeakCacheSequences = n
			}
		}
		if f := w.cache.Fragmentation(); f > s.res.PeakFragmentation {
			s.res.PeakFragmentation = f
		}
		used += w.cache.Used()
		total += w.spec.TotalBlocks
	}
	if total > 0 {
		if u := float64(used) / float64(total); u > s.res.PeakCacheUtil {
			s.res.PeakCacheUtil = u
		}
	}
}

// trackRecovery measures how long the system took to get back to where it was.
//
// Two conditions, and the first is what makes the number mean anything: the system must
// be observed to have DEGRADED — head-of-line wait past its pre-failure high-water mark —
// before it can be observed to have recovered. Without that, a queue that happens to be
// empty on the tick right after the kill reads as an instant recovery, which is what the
// first version of this reported.
//
// Recovery is defined against head-of-line wait rather than queue depth because depth
// says nothing about how fast the queue is draining, which is the same reason the
// autoscaler scales on wait.
func (s *Sim) trackRecovery(now time.Time) {
	el := now.Sub(s.clock.Start())
	wait := s.queue.OldestWait(now)
	depth := s.queue.Len()

	if s.lastCrashT == 0 || el <= s.lastCrashT {
		if wait > s.preCrashWait {
			s.preCrashWait = wait
		}
		if depth > s.preCrashDepth {
			s.preCrashDepth = depth
		}
		return
	}
	if s.recovered {
		return
	}
	if !s.degraded {
		if wait > s.preCrashWait && depth > s.preCrashDepth {
			s.degraded = true
		}
		return
	}
	if wait <= s.preCrashWait && depth <= s.preCrashDepth {
		s.res.RecoveryTime = el - s.lastCrashT
		s.recovered = true
	}
}

func (s *Sim) clampCP(w *simWorker) {
	if w.cp.UsedCacheBlocks < 0 {
		w.cp.UsedCacheBlocks = 0
	}
	if w.cp.UsedCacheBlocks > w.cp.TotalCacheBlocks {
		w.cp.UsedCacheBlocks = w.cp.TotalCacheBlocks
	}
}

func (s *Sim) anyRunning() bool {
	for _, w := range s.workers {
		if w.alive && len(w.running) > 0 {
			return true
		}
	}
	return false
}

func (s *Sim) sample(el time.Duration, now time.Time) Sample {
	sm := Sample{
		T:              el,
		QueueDepth:     s.queue.Len(),
		HeadOfLineWait: s.queue.OldestWait(now),
		Running:        s.concurrency,
		Completed:      s.res.Completed,
		Rejected:       s.res.Rejected,
	}
	var used, total int
	for _, w := range s.workers {
		if !w.alive {
			continue
		}
		sm.Workers++
		switch {
		case !w.ready:
			sm.Starting++
		case w.cp.Draining:
			sm.Draining++
		default:
			sm.Serving++
		}
		used += w.cache.Used()
		total += w.spec.TotalBlocks
	}
	if total > 0 {
		sm.CacheUtil = float64(used) / float64(total)
	}
	return sm
}

func (s *Sim) onReap(dead []loom.DeadWorker, now time.Time) {
	for _, d := range dead {
		s.res.ReapedWorkers++
		s.res.ReleasedOnReap += d.ReleasedBlocks
		s.res.ReapedRequests += len(d.Requests)

		w := s.workerByID(d.ID)
		if w != nil {
			w.running = nil
			for id := range w.cpReserved {
				delete(w.cpReserved, id)
			}
			w.cp.UsedCacheBlocks = 0
			w.cp.ActiveSequences = 0
			s.recount()
		}

		for _, id := range d.Requests {
			st := s.states[id]
			if st == nil {
				continue
			}
			f := loom.Failure{Request: id, Worker: d.ID, WorkerLost: true}
			v := s.registry.ClassifyRetry(st, f, now)
			if !s.cfg.RetryOnWorkerLoss || v.Action == loom.DoNotRetry {
				s.adm.Complete(id)
				s.res.LostOnReap++
				s.res.Failed++
				delete(s.states, id)
				delete(s.actual, id)
				continue
			}
			// The tokens produced on the dead worker are gone with it, so the retry
			// starts over. Resetting QueuedAt would hide the real end-to-end latency of
			// a request that got caught in a failure, so it is left alone.
			st.Phase = loom.PhaseQueued
			st.GeneratedTokens = 0
			st.HeldBlocks = 0
			st.Worker = ""
			st.StartedAt = time.Time{}
			st.FirstToken = time.Time{}
			if s.queue.Push(st) {
				s.res.RetriedOnReap++
			} else {
				s.adm.Complete(id)
				s.res.LostOnReap++
				s.res.Failed++
				delete(s.states, id)
				delete(s.actual, id)
			}
		}
		s.registry.Forget(d.ID)
	}
	s.kickAll(now)
}

func (s *Sim) workerByID(id loom.WorkerID) *simWorker {
	for _, w := range s.workers {
		if w.spec.ID == id {
			return w
		}
	}
	return nil
}

func (s *Sim) onAutoscale(now time.Time) {
	dec := s.cfg.Autoscaler.Evaluate(s.queue, s.allStates(), now)
	switch {
	case dec.Delta > 0:
		s.res.ScaleUpActions++
		for i := 0; i < dec.Delta; i++ {
			spec := s.cfg.Template
			if spec.TotalBlocks == 0 && len(s.cfg.Workers) > 0 {
				spec = s.cfg.Workers[0]
			}
			spec.ID = ""
			s.addWorker(spec, now)
		}
	case dec.Delta < 0:
		s.res.ScaleDownActions++
		target := dec.Target
		if target == "" {
			// QueueWaitAutoscaler never populates ScaleDecision.Target even though the
			// contract says it carries the victim; without this fallback a scale-down
			// decision would be unactionable. Flagged in bench/results.md.
			if t, ok := s.cfg.Autoscaler.ScaleDownTarget(s.allStates()); ok {
				target = t
			}
		}
		if target != "" {
			if w := s.workerByID(target); w != nil {
				w.cp.Draining = true
				w.truth.Draining = true
				_ = s.registry.SetDraining(target, true)
			}
		}
	}
}

// removeDrained retires a draining worker once its running set is empty. Draining
// rather than killing is what makes scale-down free: the requests that were already
// most of the way done still finish.
func (s *Sim) removeDrained() {
	kept := s.workers[:0]
	for _, w := range s.workers {
		if w.alive && w.cp.Draining && len(w.running) == 0 && w.ready {
			s.registry.Deregister(w.spec.ID)
			s.retired = append(s.retired, w)
			continue
		}
		kept = append(kept, w)
	}
	if len(kept) != len(s.workers) {
		s.workers = kept
	}
}

func (s *Sim) finish() {
	// Integrated and measured to the last observable event, not to the clock's final
	// position. The clock can sit well past the last completion — the tick loop runs to
	// the configured bound when there is still pending work it cannot place — and
	// dividing by that slack would understate throughput by however loose the bound was.
	end := s.lastActivity
	if end.Before(s.clock.Start()) {
		end = s.clock.Now()
	}
	s.integrate(end)
	s.res.Elapsed = end.Sub(s.clock.Start())
	s.res.WallEnd = s.clock.Elapsed()
	if s.res.Elapsed > 0 {
		s.res.MeanConcurrency = s.concIntegral / s.res.Elapsed.Seconds()
	}
	s.res.Admission = s.adm.Metrics()
	s.res.FinalWorkers = len(s.workers)
	s.res.Recovered = s.recovered

	for _, w := range append(append([]*simWorker{}, s.workers...), s.retired...) {
		if !w.alive {
			continue
		}
		s.res.Leak.CacheBlocksHeld += w.cache.Used()
		if c, ok := w.cache.(interface{ Sequences() int }); ok {
			s.res.Leak.CacheSequences += c.Sequences()
		}
		for id, n := range w.cpReserved {
			s.res.Leak.ControlPlaneHeld += n
			s.res.Leak.Details = append(s.res.Leak.Details, fmt.Sprintf("%s holds %d blocks for %s (running=%d tracked=%v)", w.spec.ID, n, id, len(w.running), s.states[id] != nil))
		}
	}
	for t := range s.res.PerTenant {
		r, b, _ := s.adm.TenantUsage(t)
		s.res.Leak.TenantQuotaHeld += r + b
	}
	s.res.Leak.UnfinishedTracked = len(s.states)

	var lat, wait, ttft []time.Duration
	for _, d := range s.completedLat {
		lat = append(lat, d)
	}
	for _, d := range s.completedWait {
		wait = append(wait, d)
	}
	for _, d := range s.completedTTFT {
		ttft = append(ttft, d)
	}
	s.res.Latency = Summarize(lat)
	s.res.QueueWait = Summarize(wait)
	s.res.TTFT = Summarize(ttft)

	for _, ts := range s.res.PerTenant {
		sort.Slice(ts.Latencies, func(i, j int) bool { return ts.Latencies[i] < ts.Latencies[j] })
	}

	for _, w := range append(append([]*simWorker{}, s.workers...), s.retired...) {
		s.res.Steps += w.steps
		s.res.SequenceSteps += w.tokens
		if cb, ok := w.batch.(interface{ Preemptions() uint64 }); ok {
			s.res.Preemptions += cb.Preemptions()
		}
	}
}
