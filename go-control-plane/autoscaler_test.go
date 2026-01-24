package loom

import (
	"strconv"
	"strings"
	"testing"
	"time"
)

// asQueue is a Queue stub that reports exactly the two signals the autoscaler
// reads. The real queue is another component's file; depending on it here would
// make an autoscaler failure indistinguishable from a queue failure.
type asQueue struct {
	depth    int
	capacity int
	wait     time.Duration
}

func (q *asQueue) Push(*RequestState) bool               { q.depth++; return true }
func (q *asQueue) Pop() *RequestState                    { return nil }
func (q *asQueue) Remove(RequestID) bool                 { return false }
func (q *asQueue) DropExpired(time.Time) []*RequestState { return nil }
func (q *asQueue) Len() int                              { return q.depth }
func (q *asQueue) Cap() int                              { return q.capacity }
func (q *asQueue) OldestWait(time.Time) time.Duration    { return q.wait }

// The autoscaler reads only depth and wait, so the read side is a stub. Present
// because Queue requires it, not because these tests exercise it.
func (q *asQueue) Snapshot() QueueSnapshot        { return q }
func (q *asQueue) Each(func(*RequestState) bool)  {}
func (q *asQueue) TenantCount(TenantID) int       { return 0 }
func (q *asQueue) TenantCounts() map[TenantID]int { return nil }

const (
	asUpWait     = 500 * time.Millisecond
	asDownWait   = 50 * time.Millisecond
	asSustain    = 1 * time.Second
	asColdStart  = 10 * time.Second
	asUpCooldown = 10 * time.Second
	asDownCool   = 30 * time.Second
)

func asConfig() AutoscalerConfig {
	return AutoscalerConfig{
		ScaleUpWait:        asUpWait,
		ScaleDownWait:      asDownWait,
		ScaleUpDepth:       8,
		CacheUtilScaleUp:   0.85,
		CacheUtilScaleDown: 0.35,
		SustainedFor:       asSustain,
		ColdStartDuration:  asColdStart,
		ScaleUpCooldown:    asUpCooldown,
		ScaleDownCooldown:  asDownCool,
		MaxScaleUpStep:     2,
		MinWorkers:         1,
		MaxWorkers:         4,
	}
}

func asWorkers(n int, used, total, active int) []*WorkerState {
	out := make([]*WorkerState, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, &WorkerState{
			ID:                    WorkerID("w-" + strconv.Itoa(i)),
			TotalCacheBlocks:      total,
			UsedCacheBlocks:       used,
			ActiveSequences:       active,
			MaxBatchSize:          32,
			RecentTokensPerSecond: 100,
		})
	}
	return out
}

func TestAutoscalerScalesUpOnSustainedQueueWait(t *testing.T) {
	a := NewQueueWaitAutoscaler(asConfig())
	q := &asQueue{depth: 12, capacity: 256, wait: 600 * time.Millisecond}
	ws := asWorkers(2, 500, 1000, 6)

	now := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	// The first observation above the threshold must not act; a scaler that reacts
	// to one sample reacts to noise.
	d := a.Evaluate(q, ws, now)
	if d.Delta != 0 || d.Reason != "demand_not_sustained" {
		t.Fatalf("first observation acted: %+v", d)
	}

	// Still inside the sustained window.
	now = now.Add(asSustain / 2)
	if d := a.Evaluate(q, ws, now); d.Delta != 0 {
		t.Fatalf("acted before the sustained window elapsed: %+v", d)
	}

	now = now.Add(asSustain)
	d = a.Evaluate(q, ws, now)
	if d.Delta != 1 {
		t.Fatalf("sustained queue wait did not scale up: %+v", d)
	}
	if s := a.Signals(); s.HeadOfLineWait != 600*time.Millisecond || s.QueueDepth != 12 || s.ServingWorkers != 2 {
		t.Fatalf("signals = %+v", s)
	}

	// Twice the threshold asks for a bigger step, bounded by MaxScaleUpStep.
	b := NewQueueWaitAutoscaler(asConfig())
	q2 := &asQueue{depth: 60, capacity: 256, wait: 4 * asUpWait}
	t0 := now
	b.Evaluate(q2, ws, t0)
	d = b.Evaluate(q2, ws, t0.Add(2*asSustain))
	if d.Delta != 2 {
		t.Fatalf("step = %d, want 2 (capped by MaxScaleUpStep)", d.Delta)
	}
}

func TestAutoscalerEmptyFleetWithQueuedWorkScalesUp(t *testing.T) {
	cfg := asConfig()
	cfg.MinWorkers = 0
	a := NewQueueWaitAutoscaler(cfg)
	q := &asQueue{depth: 3, capacity: 256}
	now := time.Now()

	// Zero workers and queued work: wait is still near zero because nothing has had
	// time to age, but no amount of waiting will help since there is nothing to
	// serve it.
	a.Evaluate(q, nil, now)
	if d := a.Evaluate(q, nil, now.Add(2*asSustain)); d.Delta != 1 {
		t.Fatalf("empty fleet with queued work: %+v", d)
	}
}

// TestAutoscalerIgnoresBriefSpike is the behaviour that separates a scaler from a
// thermostat. A spike shorter than the sustained window must not buy a worker that
// will not be useful until long after the spike is over, and demand that
// in-flight cold-starting capacity will absorb must not buy a second one.
func TestAutoscalerIgnoresBriefSpike(t *testing.T) {
	a := NewQueueWaitAutoscaler(asConfig())
	ws := asWorkers(2, 500, 1000, 6)
	now := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	spike := &asQueue{depth: 20, capacity: 256, wait: 900 * time.Millisecond}
	calm := &asQueue{depth: 0, capacity: 256, wait: 0}

	// Three spikes, each shorter than SustainedFor, each followed by calm.
	for i := 0; i < 3; i++ {
		for el := time.Duration(0); el < asSustain-100*time.Millisecond; el += 100 * time.Millisecond {
			if d := a.Evaluate(spike, ws, now); d.Delta != 0 {
				t.Fatalf("spike %d acted at +%s: %+v", i, el, d)
			}
			now = now.Add(100 * time.Millisecond)
		}
		for el := time.Duration(0); el < 2*asSustain; el += 100 * time.Millisecond {
			if d := a.Evaluate(calm, ws, now); d.Delta != 0 {
				t.Fatalf("calm after spike %d acted: %+v", i, d)
			}
			now = now.Add(100 * time.Millisecond)
		}
	}
	if a.Actions() != 0 {
		t.Fatalf("brief spikes produced %d actions, want 0", a.Actions())
	}

	// Now sustain the load. One worker is ordered.
	var orderedAt time.Time
	for el := time.Duration(0); el <= 2*asSustain; el += 100 * time.Millisecond {
		if d := a.Evaluate(spike, ws, now); d.Delta > 0 && orderedAt.IsZero() {
			orderedAt = now
		}
		now = now.Add(100 * time.Millisecond)
	}
	if a.Actions() != 1 {
		t.Fatalf("sustained load produced %d actions, want 1", a.Actions())
	}
	if orderedAt.IsZero() {
		t.Fatal("no order recorded")
	}

	// The ordered worker is cold-starting. The load is still high and still
	// sustained, but the capacity on its way covers it, so nothing more is ordered.
	// This is the double-provisioning a cold-start-blind scaler commits.
	for now.Sub(orderedAt) < asColdStart {
		d := a.Evaluate(spike, ws, now)
		if d.Delta != 0 {
			t.Fatalf("scaled up again at +%s after the order while the worker was cold-starting: %+v",
				now.Sub(orderedAt), d)
		}
		now = now.Add(500 * time.Millisecond)
	}
	if a.Actions() != 1 {
		t.Fatalf("actions during cold start = %d, want 1", a.Actions())
	}
	if s := a.Signals(); s.StartingWorkers != 1 {
		t.Fatalf("StartingWorkers = %d during cold start, want 1", s.StartingWorkers)
	}

	// The cold start finishes and the worker joins the fleet. If the load has
	// persisted through all of that, it is real and a second worker is justified.
	ws3 := asWorkers(3, 500, 1000, 6)
	now = now.Add(time.Second)
	for i := 0; i < 5; i++ {
		d := a.Evaluate(spike, ws3, now)
		now = now.Add(time.Second)
		if d.Delta == 1 {
			break
		}
	}
	if a.Actions() != 2 {
		t.Fatalf("after the cold start completed, actions = %d, want 2", a.Actions())
	}
	// Signals report the observation the decision was made from, so the newly
	// ordered worker shows up as in-flight capacity on the following evaluation.
	a.Evaluate(spike, ws3, now)
	if s := a.Signals(); s.StartingWorkers != 1 {
		t.Fatalf("StartingWorkers = %d after the second order, want 1", s.StartingWorkers)
	}
}

func TestAutoscalerNoteWorkerStartedSuppressesDuplicateOrder(t *testing.T) {
	a := NewQueueWaitAutoscaler(asConfig())
	ws := asWorkers(2, 900, 1000, 8)
	q := &asQueue{depth: 20, capacity: 256, wait: 800 * time.Millisecond}
	now := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	// A worker launched by an operator or by a crash restart is capacity the scaler
	// did not order but must still count, or it buys the same capacity twice.
	a.NoteWorkerStarted(now)
	for el := time.Duration(0); el <= 2*asSustain; el += 200 * time.Millisecond {
		if d := a.Evaluate(q, ws, now); d.Delta != 0 {
			t.Fatalf("ordered a worker while an externally started one was coming: %+v", d)
		}
		now = now.Add(200 * time.Millisecond)
	}
}

// TestAutoscalerDoesNotOscillate drives the scaler with a signal that hovers
// around the scale-up threshold — the worst case for a single-threshold
// controller, which would act on roughly every other sample. Hysteresis means the
// action count stays tiny and the actions never reverse direction.
func TestAutoscalerDoesNotOscillate(t *testing.T) {
	a := NewQueueWaitAutoscaler(asConfig())
	now := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	const ticks = 400
	const tick = 100 * time.Millisecond

	// Runs of 15 ticks (1.5s) above the threshold alternating with 15 below. Each
	// above-run outlasts the sustained window, so a controller with no cooldown
	// would act on every run, and one with no band would act on every crossing.
	workers := 2
	var deltas []int
	var crossings int
	above := false
	for i := 0; i < ticks; i++ {
		nowAbove := (i/15)%2 == 0
		if nowAbove != above {
			crossings++
			above = nowAbove
		}
		q := &asQueue{capacity: 256}
		if nowAbove {
			q.wait = asUpWait + 10*time.Millisecond
			q.depth = 10
		} else {
			q.wait = asUpWait - 10*time.Millisecond
			q.depth = 9
		}
		ws := asWorkers(workers, 600, 1000, 6)
		d := a.Evaluate(q, ws, now)
		if d.Delta != 0 {
			deltas = append(deltas, d.Delta)
			workers += d.Delta
			if workers < 1 {
				workers = 1
			}
		}
		now = now.Add(tick)
	}

	t.Logf("hysteresis: %d threshold crossings over %d ticks produced %d scaling actions %v",
		crossings, ticks, len(deltas), deltas)

	if crossings < 20 {
		t.Fatalf("the test signal only crossed the threshold %d times; it is not exercising hysteresis", crossings)
	}
	// A single-threshold controller would act on the order of `crossings` times.
	if len(deltas) > 4 {
		t.Fatalf("oscillation: %d actions for %d crossings", len(deltas), crossings)
	}
	for _, d := range deltas {
		if d < 0 {
			t.Fatalf("scaled down while the signal hovered at the scale-up threshold: %v", deltas)
		}
	}
	// Hovering just below the up threshold is still far above the down threshold, so
	// the band must swallow it entirely rather than treating it as surplus.
	if a.LastActionSign() == -1 {
		t.Fatal("last action was a scale-down under scale-up-level load")
	}
}

func TestAutoscalerScalesDownOnSustainedSurplus(t *testing.T) {
	a := NewQueueWaitAutoscaler(asConfig())
	q := &asQueue{depth: 0, capacity: 256, wait: 0}
	ws := asWorkers(3, 100, 1000, 1)
	now := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	if d := a.Evaluate(q, ws, now); d.Delta != 0 || d.Reason != "surplus_not_sustained" {
		t.Fatalf("first surplus observation acted: %+v", d)
	}
	now = now.Add(2 * asSustain)
	d := a.Evaluate(q, ws, now)
	if d.Delta != -1 {
		t.Fatalf("sustained surplus did not scale down: %+v", d)
	}

	// One at a time. Acting restarts the sustained window, so the surplus has to be
	// re-established before the cooldown is even consulted — two independent brakes
	// rather than one.
	now = now.Add(2 * asSustain)
	if d := a.Evaluate(q, ws, now); d.Delta != 0 || d.Reason != "surplus_not_sustained" {
		t.Fatalf("the sustained window was not restarted by the action: %+v", d)
	}
	now = now.Add(2 * asSustain)
	if d := a.Evaluate(q, ws, now); d.Delta != 0 || d.Reason != "scale_down_cooldown" {
		t.Fatalf("cooldown not honoured: %+v", d)
	}
	now = now.Add(asDownCool)
	if d := a.Evaluate(q, ws, now); d.Delta != -1 {
		t.Fatalf("after the cooldown: %+v", d)
	}
}

// TestSlowdownIsSeenWithAnEmptyQueue is the processor-sharing case: the fleet
// degrades, the queue never forms, and queue wait alone reads the system as
// healthy — or worse, as over-provisioned.
func TestSlowdownIsSeenWithAnEmptyQueue(t *testing.T) {
	// Empty queue, no wait. This is what a continuous-batching fleet with a loose
	// batch cap looks like all the way into saturation: an arriving request is
	// admitted immediately and slows every resident sequence instead of waiting.
	idleQueue := func() *asQueue { return &asQueue{depth: 0, capacity: 256, wait: 0} }

	// Uncontended: one sequence per worker at 100 tokens/s.
	uncontended := func() []*WorkerState {
		ws := asWorkers(2, 100, 1000, 1)
		for _, w := range ws {
			w.RecentTokensPerSecond = 100
		}
		return ws
	}
	// Saturated: 20 resident sequences sharing 500 tokens/s, so 25 per sequence —
	// a quarter of the uncontended rate, with the queue still empty.
	degraded := func() []*WorkerState {
		ws := asWorkers(2, 300, 1000, 20)
		for _, w := range ws {
			w.RecentTokensPerSecond = 500
		}
		return ws
	}

	train := func(a *QueueWaitAutoscaler, start time.Time) time.Time {
		now := start
		for i := 0; i < 10; i++ {
			a.Evaluate(idleQueue(), uncontended(), now)
			now = now.Add(100 * time.Millisecond)
		}
		return now
	}

	start := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	// Queue-wait-only: the signal the system had.
	blind := NewQueueWaitAutoscaler(func() AutoscalerConfig {
		c := asConfig()
		c.SlowdownScaleUp = -1 // disabled
		return c
	}())
	now := train(blind, start)
	sawScaleUp := false
	for i := 0; i < 20; i++ {
		d := blind.Evaluate(idleQueue(), degraded(), now)
		if d.Delta > 0 {
			sawScaleUp = true
		}
		now = now.Add(asSustain)
	}
	if sawScaleUp {
		t.Fatal("the queue-wait-only scaler reacted; this workload is supposed to be invisible to it")
	}
	if sig := blind.Signals(); sig.HeadOfLineWait != 0 || sig.QueueDepth != 0 {
		t.Fatalf("the degraded workload formed a queue (%v, depth %d); the test is not exercising the blind spot",
			sig.HeadOfLineWait, sig.QueueDepth)
	}

	// With the slowdown term.
	a := NewQueueWaitAutoscaler(asConfig())
	now = train(a, start)
	if sig := a.Signals(); sig.Slowdown > 1.01 {
		t.Fatalf("uncontended slowdown = %.2f, want ~1.0", sig.Slowdown)
	}

	first := a.Evaluate(idleQueue(), degraded(), now)
	if first.Delta != 0 || first.Reason != "demand_not_sustained" {
		t.Fatalf("acted on the first degraded sample: %+v", first)
	}
	now = now.Add(2 * asSustain)
	d := a.Evaluate(idleQueue(), degraded(), now)
	if d.Delta <= 0 {
		sig := a.Signals()
		t.Fatalf("no scale-up on a sustained %.2fx slowdown with an empty queue: %+v (signals %+v)", sig.Slowdown, d, sig)
	}
	sig := a.Signals()
	if sig.Slowdown < 3.5 {
		t.Fatalf("slowdown = %.2f, want ~4.0 (25 tokens/s per sequence against 100 uncontended)", sig.Slowdown)
	}
	if sig.PerSequenceTokensPerSecond != 25 {
		t.Fatalf("per-sequence rate = %.2f, want 25", sig.PerSequenceTokensPerSecond)
	}
	if !strings.Contains(d.Reason, "slowdown=") {
		t.Fatalf("reason %q does not report the signal that drove it", d.Reason)
	}

	// And a degrading fleet must not be read as surplus, which is what an empty queue
	// and low cache use say on their own.
	down := NewQueueWaitAutoscaler(asConfig())
	now = train(down, start)
	for i := 0; i < 5; i++ {
		now = now.Add(2 * asSustain)
		if d := down.Evaluate(idleQueue(), degraded(), now); d.Delta < 0 {
			t.Fatalf("scaled down a fleet whose sequences are at a quarter speed: %+v", d)
		}
	}
}

// TestSlowdownNeedsATrustedBaseline covers the honest limitation: the learned
// baseline is a high-water mark, so until there is enough history the term reports
// unmeasurable rather than healthy.
func TestSlowdownNeedsATrustedBaseline(t *testing.T) {
	cfg := asConfig()
	cfg.SlowdownMinSamples = 5
	a := NewQueueWaitAutoscaler(cfg)
	now := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	ws := asWorkers(1, 300, 1000, 20)
	ws[0].RecentTokensPerSecond = 500
	q := &asQueue{depth: 0, capacity: 256, wait: 0}

	for i := 0; i < 4; i++ {
		a.Evaluate(q, ws, now.Add(time.Duration(i)*time.Second))
		if sig := a.Signals(); sig.Slowdown != 0 {
			t.Fatalf("sample %d reported slowdown %.2f from an untrained baseline", i, sig.Slowdown)
		}
	}
	// Once trained, a fleet that has only ever been seen at this rate reads as
	// uncontended — the limitation the config override exists for.
	a.Evaluate(q, ws, now.Add(5*time.Second))
	if sig := a.Signals(); sig.Slowdown == 0 || sig.Slowdown > 1.01 {
		t.Fatalf("slowdown = %.2f, want ~1.0: the only rate ever observed is the learned baseline", sig.Slowdown)
	}

	// With the uncontended rate supplied, the same observation is correctly degraded.
	cfg2 := asConfig()
	cfg2.UncontendedPerSequenceTokensPerSecond = 100
	b := NewQueueWaitAutoscaler(cfg2)
	b.Evaluate(q, ws, now)
	if sig := b.Signals(); sig.Slowdown < 3.5 {
		t.Fatalf("slowdown = %.2f with a configured baseline of 100 against 25 per sequence", sig.Slowdown)
	}
}

// TestScaleDownDecisionCarriesItsTarget pins the field the contract calls
// load-bearing: a scale-down decision names the worker to drain, chosen from the
// state the decision was reasoned about, so the caller does not have to re-derive
// "least loaded" from newer state and pick a different one.
func TestScaleDownDecisionCarriesItsTarget(t *testing.T) {
	a := NewQueueWaitAutoscaler(asConfig())
	q := &asQueue{depth: 0, capacity: 256, wait: 0}
	ws := asWorkers(3, 100, 1000, 4)
	// One clearly idlest worker, so the expected target is not a tie-break.
	ws[2].ActiveSequences = 0
	ws[2].UsedCacheBlocks = 1
	now := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	a.Evaluate(q, ws, now)
	d := a.Evaluate(q, ws, now.Add(2*asSustain))
	if d.Delta != -1 {
		t.Fatalf("sustained surplus did not scale down: %+v", d)
	}
	if d.Target == "" {
		t.Fatal("scale-down decision carries no Target, so the caller cannot act on it without re-deriving the victim")
	}
	if d.Target != ws[2].ID {
		t.Fatalf("Target = %q, want the least loaded worker %q", d.Target, ws[2].ID)
	}
	if want, _ := a.ScaleDownTarget(ws); d.Target != want {
		t.Fatalf("Target = %q but ScaleDownTarget says %q; the decision and the helper disagree", d.Target, want)
	}
	// The reason string already names the victim; Target must agree with it, or an
	// operator reading the log and a caller reading the field see different workers.
	if !strings.Contains(d.Reason, string(d.Target)) {
		t.Fatalf("reason %q does not mention the target %q", d.Reason, d.Target)
	}

	// A scale-up has no victim, and inventing one would invite a caller to drain a
	// worker on a decision to add capacity.
	up := NewQueueWaitAutoscaler(asConfig())
	busy := &asQueue{depth: 32, capacity: 256, wait: 4 * asUpWait}
	up.Evaluate(busy, ws, now)
	if d := up.Evaluate(busy, ws, now.Add(2*asSustain)); d.Delta <= 0 {
		t.Fatalf("expected a scale-up: %+v", d)
	} else if d.Target != "" {
		t.Fatalf("scale-up decision named a drain target %q", d.Target)
	}
}

func TestAutoscalerScaleDownBlockedByItsOwnConditions(t *testing.T) {
	now := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	cases := []struct {
		name string
		q    *asQueue
		ws   []*WorkerState
		cfg  func(AutoscalerConfig) AutoscalerConfig
		want string
	}{
		{
			name: "a non-empty queue is never surplus",
			q:    &asQueue{depth: 1},
			ws:   asWorkers(3, 100, 1000, 1),
			want: "within_hysteresis_band",
		},
		{
			name: "wait inside the band is neither surplus nor demand",
			q:    &asQueue{wait: (asUpWait + asDownWait) / 2},
			ws:   asWorkers(3, 100, 1000, 1),
			want: "within_hysteresis_band",
		},
		{
			name: "a nearly full cache is not surplus even with an empty queue",
			q:    &asQueue{},
			ws:   asWorkers(3, 800, 1000, 8),
			want: "within_hysteresis_band",
		},
		{
			name: "throughput above the floor keeps the fleet",
			q:    &asQueue{},
			ws:   asWorkers(3, 100, 1000, 1),
			cfg: func(c AutoscalerConfig) AutoscalerConfig {
				c.MinTokensPerSecond = 10
				return c
			},
			want: "within_hysteresis_band",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := asConfig()
			if tc.cfg != nil {
				cfg = tc.cfg(cfg)
			}
			a := NewQueueWaitAutoscaler(cfg)
			a.Evaluate(tc.q, tc.ws, now)
			d := a.Evaluate(tc.q, tc.ws, now.Add(5*asSustain))
			if d.Delta != 0 || d.Reason != tc.want {
				t.Fatalf("got %+v, want delta 0 reason %q", d, tc.want)
			}
		})
	}
}

func TestAutoscalerRespectsMinAndMax(t *testing.T) {
	now := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

	// Max: sustained demand on a full-size fleet must not order more.
	a := NewQueueWaitAutoscaler(asConfig())
	hot := &asQueue{depth: 50, capacity: 256, wait: 5 * asUpWait}
	ws := asWorkers(4, 900, 1000, 8)
	a.Evaluate(hot, ws, now)
	d := a.Evaluate(hot, ws, now.Add(5*asSustain))
	if d.Delta != 0 || d.Reason != "at_max_workers" {
		t.Fatalf("at max: %+v", d)
	}

	// A step that would overshoot the max is clipped rather than refused.
	b := NewQueueWaitAutoscaler(asConfig())
	ws3 := asWorkers(3, 900, 1000, 8)
	b.Evaluate(hot, ws3, now)
	if d := b.Evaluate(hot, ws3, now.Add(5*asSustain)); d.Delta != 1 {
		t.Fatalf("step at max-1 = %+v, want 1", d)
	}

	// Min: sustained surplus must not take the fleet below the floor.
	c := NewQueueWaitAutoscaler(asConfig())
	idle := &asQueue{}
	one := asWorkers(1, 0, 1000, 0)
	c.Evaluate(idle, one, now)
	if d := c.Evaluate(idle, one, now.Add(5*asSustain)); d.Delta != 0 || d.Reason != "at_min_workers" {
		t.Fatalf("at min: %+v", d)
	}

	// And a fleet below the floor is corrected immediately, without waiting for the
	// sustained window or the cooldown: that is a configuration violation, not load.
	cfg := asConfig()
	cfg.MinWorkers = 3
	e := NewQueueWaitAutoscaler(cfg)
	d = e.Evaluate(idle, one, now)
	if d.Delta != 2 || d.Reason != "below_min_workers" {
		t.Fatalf("below min: %+v", d)
	}
}

func TestAutoscalerScaleDownPicksLeastLoaded(t *testing.T) {
	a := NewQueueWaitAutoscaler(asConfig())
	ws := []*WorkerState{
		{ID: "busy", ActiveSequences: 9, UsedCacheBlocks: 400, TotalCacheBlocks: 1000},
		{ID: "idle", ActiveSequences: 1, UsedCacheBlocks: 50, TotalCacheBlocks: 1000},
		{ID: "medium", ActiveSequences: 4, UsedCacheBlocks: 200, TotalCacheBlocks: 1000},
	}
	got, ok := a.ScaleDownTarget(ws)
	if !ok || got != "idle" {
		t.Fatalf("ScaleDownTarget = %q,%v; want idle", got, ok)
	}

	// Equal sequence counts fall back to held blocks, because that worker's drain
	// finishes soonest.
	tie := []*WorkerState{
		{ID: "a", ActiveSequences: 2, UsedCacheBlocks: 300},
		{ID: "b", ActiveSequences: 2, UsedCacheBlocks: 100},
	}
	if got, _ := a.ScaleDownTarget(tie); got != "b" {
		t.Fatalf("block tie-break = %q, want b", got)
	}

	// An already-draining worker is not a candidate, or two consecutive
	// scale-downs drain the same worker and remove no capacity.
	drained := []*WorkerState{
		{ID: "gone", Draining: true, ActiveSequences: 0},
		{ID: "stay", ActiveSequences: 5},
	}
	if got, _ := a.ScaleDownTarget(drained); got != "stay" {
		t.Fatalf("drained worker selected again: %q", got)
	}
	if _, ok := a.ScaleDownTarget(nil); ok {
		t.Fatal("ScaleDownTarget found a victim in an empty fleet")
	}

	// And the decision names the worker, since ScaleDecision carries only a delta.
	b := NewQueueWaitAutoscaler(asConfig())
	idle := &asQueue{}
	quiet := []*WorkerState{
		{ID: "busy", ActiveSequences: 3, UsedCacheBlocks: 100, TotalCacheBlocks: 1000},
		{ID: "idle", ActiveSequences: 0, UsedCacheBlocks: 10, TotalCacheBlocks: 1000},
	}
	now := time.Now()
	b.Evaluate(idle, quiet, now)
	d := b.Evaluate(idle, quiet, now.Add(5*asSustain))
	if d.Delta != -1 || !strings.Contains(d.Reason, "drain=idle") {
		t.Fatalf("scale-down reason = %+v, want it to name idle", d)
	}
}

func TestAutoscalerDrainingWorkersAreNotCountedAsCapacity(t *testing.T) {
	a := NewQueueWaitAutoscaler(asConfig())
	ws := []*WorkerState{
		{ID: "w-1", TotalCacheBlocks: 1000, UsedCacheBlocks: 900, ActiveSequences: 8},
		{ID: "w-2", TotalCacheBlocks: 1000, UsedCacheBlocks: 10, ActiveSequences: 1, Draining: true},
	}
	q := &asQueue{depth: 20, capacity: 256, wait: 2 * asUpWait}
	now := time.Now()
	a.Evaluate(q, ws, now)
	d := a.Evaluate(q, ws, now.Add(2*asSustain))
	if d.Delta < 1 {
		t.Fatalf("a draining worker was counted as serving capacity: %+v", d)
	}
	s := a.Signals()
	if s.ServingWorkers != 1 || s.DrainingWorkers != 1 {
		t.Fatalf("signals = %+v, want 1 serving and 1 draining", s)
	}
	// The drainer's nearly-empty cache must not dilute the utilization signal.
	if s.CacheUtil < 0.89 {
		t.Fatalf("CacheUtil = %.3f, want ~0.9 from the serving worker alone", s.CacheUtil)
	}
}

// TestAutoscalerScaleUpCooldown isolates the cooldown from the cold-start brake
// by making the cooldown much longer than the cold start, then letting the ordered
// workers arrive. Without that separation a passing test proves only that one of
// the two brakes works.
func TestAutoscalerScaleUpCooldown(t *testing.T) {
	cfg := asConfig()
	cfg.MaxWorkers = 8
	cfg.ScaleUpCooldown = 3 * asColdStart
	a := NewQueueWaitAutoscaler(cfg)

	q := &asQueue{depth: 30, capacity: 256, wait: 3 * asUpWait}
	ws2 := asWorkers(2, 900, 1000, 8)
	ws4 := asWorkers(4, 900, 1000, 8)

	t0 := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	a.Evaluate(q, ws2, t0)
	ordered := t0.Add(2 * asSustain)
	if d := a.Evaluate(q, ws2, ordered); d.Delta != 2 {
		t.Fatalf("first scale-up = %+v, want 2", d)
	}

	// The ordered workers have arrived and the cold start has elapsed, so nothing
	// but the cooldown is holding the scaler back.
	at := ordered.Add(asColdStart + time.Second)
	a.Evaluate(q, ws4, at)
	if d := a.Evaluate(q, ws4, at.Add(2*asSustain)); d.Reason != "scale_up_cooldown" {
		t.Fatalf("acted inside the scale-up cooldown: %+v", d)
	}

	// Past it, the still-unrelieved load is acted on.
	// The demand has been continuously above the threshold throughout, so the
	// cooldown expiring is the only thing that changed.
	past := ordered.Add(cfg.ScaleUpCooldown + time.Second)
	if d := a.Evaluate(q, ws4, past); d.Delta != 2 {
		t.Fatalf("after the cooldown = %+v, want 2", d)
	}
}

func TestAutoscalerDefaultsAreSane(t *testing.T) {
	a := NewQueueWaitAutoscaler(AutoscalerConfig{})
	if a.cfg.ScaleDownWait >= a.cfg.ScaleUpWait {
		t.Fatalf("default thresholds leave no hysteresis band: up=%v down=%v",
			a.cfg.ScaleUpWait, a.cfg.ScaleDownWait)
	}
	if a.cfg.CacheUtilScaleDown >= a.cfg.CacheUtilScaleUp {
		t.Fatalf("default cache thresholds leave no band: up=%.2f down=%.2f",
			a.cfg.CacheUtilScaleUp, a.cfg.CacheUtilScaleDown)
	}
	if a.cfg.ScaleDownCooldown <= a.cfg.ScaleUpCooldown {
		t.Fatal("scale-down should be slower than scale-up: a surplus worker costs money, a missing one costs requests")
	}
	if a.cfg.MaxWorkers < a.cfg.MinWorkers {
		t.Fatal("max below min")
	}

	// A misconfiguration that inverts the thresholds must be repaired rather than
	// obeyed, since obeying it is the oscillation this component exists to prevent.
	bad := NewQueueWaitAutoscaler(AutoscalerConfig{
		ScaleUpWait:        time.Second,
		ScaleDownWait:      2 * time.Second,
		CacheUtilScaleUp:   0.5,
		CacheUtilScaleDown: 0.9,
	})
	if bad.cfg.ScaleDownWait >= bad.cfg.ScaleUpWait {
		t.Fatalf("inverted wait thresholds survived: %+v", bad.cfg)
	}
	if bad.cfg.CacheUtilScaleDown >= bad.cfg.CacheUtilScaleUp {
		t.Fatalf("inverted cache thresholds survived: %+v", bad.cfg)
	}
}

func TestAutoscalerNilQueueIsNotADecision(t *testing.T) {
	cfg := asConfig()
	cfg.MinWorkers = 1
	a := NewQueueWaitAutoscaler(cfg)
	ws := asWorkers(2, 100, 1000, 1)
	now := time.Now()
	// A nil queue means the signal is missing, not that the queue is empty; with no
	// demand evidence the scaler must sit still rather than shrink the fleet on a
	// measurement it does not have. Here it reads as surplus, which is the
	// conservative direction only because a nil queue cannot be distinguished from
	// an empty one — so the sustained window still has to elapse first.
	if d := a.Evaluate(nil, ws, now); d.Delta != 0 {
		t.Fatalf("acted on the first nil-queue observation: %+v", d)
	}
}
