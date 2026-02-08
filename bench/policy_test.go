package bench

import (
	"os"
	"testing"
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
	"github.com/LegendaryPenguin/loom/loadgen"
	"github.com/LegendaryPenguin/loom/simulator"
)

func policies() []struct {
	name string
	make func() loom.SchedulePolicy
} {
	return []struct {
		name string
		make func() loom.SchedulePolicy
	}{
		{"fifo", func() loom.SchedulePolicy { return loom.NewFIFOPolicy() }},
		// Aging at zero is pure shortest-job-first, starvation included. It is in the
		// table precisely so the starvation shows up as a max wait rather than being
		// asserted about in prose.
		{"sjf (no aging)", func() loom.SchedulePolicy { return loom.NewShortestJobFirstPolicy(0) }},
		{"sjf (aging=1.0)", func() loom.SchedulePolicy { return loom.NewShortestJobFirstPolicy(1.0) }},
		{"fair-share", func() loom.SchedulePolicy {
			p := loom.NewFairSharePolicy()
			for _, t := range loadgen.DefaultTenants() {
				p.SetWeight(t.ID, t.Weight)
			}
			return p
		}},
	}
}

// policyFleet tightens the per-worker batch cap.
//
// Not to make the system look bad, but because a scheduling policy only has a decision
// to make when there is a queue to choose from, and under continuous batching with a
// loose cap there isn't one: a 96-slot fleet behaves like a processor-sharing server, so
// an extra request slows everyone slightly instead of waiting. Head-of-line wait stays
// in single-digit milliseconds right up to rho = 1 and all four policies are identical.
// Tightening the cap to 8 makes slots the binding constraint, which is the regime where
// the ordering question is real.
func policyFleet() []simulator.WorkerSpec {
	ws := StandardFleet()
	for i := range ws {
		ws[i].MaxBatchSize = 8
	}
	return ws
}

func policyConfig(p loom.SchedulePolicy, wl loadgen.Workload, dur time.Duration) simulator.Config {
	ws := policyFleet()
	return simulator.Config{
		Workers:       ws,
		Template:      ws[0],
		Latency:       simulator.DefaultLatencyModel(),
		QueueCapacity: 512,
		Policy:        p,
		Batcher:       simulator.ContinuousBatcherFactory(0, loom.PreemptNewest),
		Estimator:     Estimator(1024, loom.NewQuantileLengthModel(ObservedLengths(Seed), 0.5)),
		Mode:          loom.ReserveExpected,
		TickInterval:  5 * time.Millisecond,
		Arrivals:      wl.Arrivals,
		MaxDuration:   dur + 30*time.Minute,
	}
}

// TestExpPolicies re-measures the claim recorded in policy.go's comments: that
// shortest-job-first buys mean latency and pays for it in the tail, and that FIFO's tail
// is the bounded one.
//
// The policy author's measurement was that the folklore is only half true — that SJF won
// p99 as well and FIFO only won the max. This runs the same workload through all four
// and reports what it gets, agreeing or not.
func TestExpPolicies(t *testing.T) {
	const dur = 240 * time.Second
	est := Estimator(1024, loom.NewQuantileLengthModel(ObservedLengths(Seed), 0.5))
	capacity := Capacity(policyFleet(), loom.ReserveExpected, est)

	for _, util := range []float64{0.70, 0.90, 1.10, 1.50} {
		wl := Workload(Seed, loadgen.Steady, util, dur, capacity, nil)

		tbl := NewTable(
			"Scheduling policy — "+Pct(util)+" offered load, continuous batching, expected reservation (p50 length model)",
			"policy", "served", "reject", "served/s", "lat p50", "lat p95", "lat p99", "lat max",
			"wait p50", "wait p95", "wait p99", "wait MAX", "preempt")

		for _, pc := range policies() {
			res := simulator.New(policyConfig(pc.make(), wl, dur)).Run()
			if res.Completed == 0 {
				t.Fatalf("%s completed nothing", pc.name)
			}
			if !res.Leak.Clean() {
				t.Errorf("%s leaked: %+v", pc.name, res.Leak)
			}
			tbl.Row(
				pc.name, itoa(res.Completed), Pct(res.RejectRate()), f2(res.Throughput()),
				Ms(res.Latency.P50), Ms(res.Latency.P95), Ms(res.Latency.P99), Ms(res.Latency.Max),
				Ms(res.QueueWait.P50), Ms(res.QueueWait.P95), Ms(res.QueueWait.P99), Ms(res.QueueWait.Max),
				itoa64(res.Preemptions),
			)
		}
		tbl.Print(os.Stdout)
	}
}

// TestExpPoliciesAcrossSeeds repeats the comparison over several seeds, because a single
// seed's p99 and especially its max are one or two observations and a claim about which
// policy wins the tail cannot rest on them.
func TestExpPoliciesAcrossSeeds(t *testing.T) {
	const dur = 240 * time.Second
	est := Estimator(1024, loom.NewQuantileLengthModel(ObservedLengths(Seed), 0.5))
	capacity := Capacity(policyFleet(), loom.ReserveExpected, est)

	tbl := NewTable(
		"Scheduling policy across 5 seeds — 110% offered load (min..max over seeds)",
		"policy", "lat p50", "lat p99", "wait p99", "wait MAX", "p99 wins", "max wins")

	type agg struct {
		p50, p99, wait99, waitMax []time.Duration
	}
	res := map[string]*agg{}
	p99Wins := map[string]int{}
	maxWins := map[string]int{}

	for s := 0; s < 5; s++ {
		seed := uint64(Seed + s)
		wl := Workload(seed, loadgen.Steady, 1.10, dur, capacity, nil)
		bestP99, bestMax := "", ""
		var bp99, bmax time.Duration
		for _, pc := range policies() {
			r := simulator.New(policyConfig(pc.make(), wl, dur)).Run()
			a := res[pc.name]
			if a == nil {
				a = &agg{}
				res[pc.name] = a
			}
			a.p50 = append(a.p50, r.Latency.P50)
			a.p99 = append(a.p99, r.Latency.P99)
			a.wait99 = append(a.wait99, r.QueueWait.P99)
			a.waitMax = append(a.waitMax, r.QueueWait.Max)
			if bestP99 == "" || r.Latency.P99 < bp99 {
				bestP99, bp99 = pc.name, r.Latency.P99
			}
			if bestMax == "" || r.QueueWait.Max < bmax {
				bestMax, bmax = pc.name, r.QueueWait.Max
			}
		}
		p99Wins[bestP99]++
		maxWins[bestMax]++
	}

	for _, pc := range policies() {
		a := res[pc.name]
		tbl.Row(pc.name,
			rangeMs(a.p50), rangeMs(a.p99), rangeMs(a.wait99), rangeMs(a.waitMax),
			itoa(p99Wins[pc.name]), itoa(maxWins[pc.name]))
	}
	tbl.Print(os.Stdout)
}

func rangeMs(ds []time.Duration) string {
	if len(ds) == 0 {
		return "-"
	}
	lo, hi := ds[0], ds[0]
	for _, d := range ds {
		if d < lo {
			lo = d
		}
		if d > hi {
			hi = d
		}
	}
	return Ms(lo) + ".." + Ms(hi)
}
