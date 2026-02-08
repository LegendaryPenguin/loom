package bench

import (
	"os"
	"testing"
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
	"github.com/LegendaryPenguin/loom/loadgen"
	"github.com/LegendaryPenguin/loom/simulator"
)

// TestExpQueueBound is the load-shedding experiment.
//
// The two numbers are reported side by side and never blended, because blending them
// inverts the conclusion: a system that refuses half the offered load and serves the rest
// in 100ms is not worse than one that accepts everything and serves it in 40 seconds. The
// first has told half its clients to go away, which they can act on; the second has told
// all of them to wait past the point where the answer is useful, which they cannot.
//
// "Effectively unbounded" is a capacity of one million rather than a different data
// structure, which is the whole point of the comparison: the queue type is the same, only
// the limit moves, so the difference is attributable to the limit and nothing else.
func TestExpQueueBound(t *testing.T) {
	const dur = 180 * time.Second
	est := Estimator(1024, loom.NewQuantileLengthModel(ObservedLengths(Seed), 0.5))
	capacity := Capacity(policyFleet(), loom.ReserveExpected, est)

	for _, util := range []float64{1.3, 2.0} {
		wl := Workload(Seed, loadgen.Steady, util, dur, capacity, nil)

		tbl := NewTable(
			"Queue bound under sustained overload — "+Pct(util)+" offered load, continuous batching, FIFO",
			"queue cap", "offered", "served", "rejected", "reject rate", "served/s", "goodput tok/s",
			"SERVED lat p50", "p95", "p99", "max", "wait p99", "final depth", "sim end")

		for _, cap := range []int{32, 128, 512, 1000000} {
			name := itoa(cap)
			if cap >= 1000000 {
				name = "1e6 (unbounded)"
			}
			ws := policyFleet()
			res := simulator.New(simulator.Config{
				Workers: ws, Template: ws[0], Latency: simulator.DefaultLatencyModel(),
				QueueCapacity: cap, Policy: loom.NewFIFOPolicy(),
				Batcher:   simulator.ContinuousBatcherFactory(0, loom.PreemptNewest),
				Estimator: est, Mode: loom.ReserveExpected,
				TickInterval:   5 * time.Millisecond,
				Arrivals:       wl.Arrivals,
				MaxDuration:    dur + 2*time.Hour,
				SampleInterval: 2 * time.Second,
			}).Run()

			if res.Completed == 0 {
				t.Fatalf("cap=%d completed nothing", cap)
			}
			finalDepth := 0
			if n := len(res.Samples); n > 0 {
				finalDepth = res.Samples[n-1].QueueDepth
			}
			tbl.Row(name, itoa(res.Submitted), itoa(res.Completed), itoa(res.Rejected),
				Pct(res.RejectRate()), f2(res.Throughput()), f1(res.Goodput()),
				Ms(res.Latency.P50), Ms(res.Latency.P95), Ms(res.Latency.P99), Ms(res.Latency.Max),
				Ms(res.QueueWait.P99), itoa(finalDepth), res.WallEnd.Round(time.Second).String())
		}
		tbl.Print(os.Stdout)
	}
}

// TestExpQueueDepthOverTime shows the shape of the difference rather than its summary: a
// bounded queue reaches a plateau and stays there, while an unbounded one rises for as
// long as the overload lasts. The plateau is what makes the bounded system's latency a
// property of the system; the ramp is what makes the unbounded system's latency a
// property of the run length.
func TestExpQueueDepthOverTime(t *testing.T) {
	const dur = 180 * time.Second
	est := Estimator(1024, loom.NewQuantileLengthModel(ObservedLengths(Seed), 0.5))
	capacity := Capacity(policyFleet(), loom.ReserveExpected, est)
	wl := Workload(Seed, loadgen.Steady, 2.0, dur, capacity, nil)

	run := func(cap int) simulator.Result {
		ws := policyFleet()
		return simulator.New(simulator.Config{
			Workers: ws, Template: ws[0], Latency: simulator.DefaultLatencyModel(),
			QueueCapacity: cap, Policy: loom.NewFIFOPolicy(),
			Batcher:   simulator.ContinuousBatcherFactory(0, loom.PreemptNewest),
			Estimator: est, Mode: loom.ReserveExpected,
			TickInterval: 5 * time.Millisecond, Arrivals: wl.Arrivals,
			MaxDuration: dur + 2*time.Hour, SampleInterval: 20 * time.Second,
		}).Run()
	}
	b, u := run(128), run(1000000)

	tbl := NewTable("Queue depth and head-of-line wait over time — 200% offered load for 180s",
		"t", "bounded depth", "bounded head wait", "bounded rejected", "unbounded depth", "unbounded head wait")
	n := len(b.Samples)
	if len(u.Samples) < n {
		n = len(u.Samples)
	}
	for i := 0; i < n; i++ {
		tbl.Row(b.Samples[i].T.Round(time.Second).String(),
			itoa(b.Samples[i].QueueDepth), Ms(b.Samples[i].HeadOfLineWait), itoa(b.Samples[i].Rejected),
			itoa(u.Samples[i].QueueDepth), Ms(u.Samples[i].HeadOfLineWait))
	}
	tbl.Print(os.Stdout)

	if u.QueueWait.P99 <= b.QueueWait.P99 {
		t.Errorf("unbounded queue did not produce worse waits than bounded: unbounded p99=%v bounded p99=%v",
			u.QueueWait.P99, b.QueueWait.P99)
	}
}
