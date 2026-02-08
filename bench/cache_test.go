package bench

import (
	"os"
	"testing"
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
	"github.com/LegendaryPenguin/loom/loadgen"
	"github.com/LegendaryPenguin/loom/simulator"
)

type cacheVariant struct {
	name  string
	spec  []simulator.WorkerSpec
	mode  loom.ReservationMode
	model loom.OutputLengthModel
}

// cacheVariants is the comparison, with the middle row doing the decomposition.
//
// The honest comparison is paged-with-expected-reservation against
// contiguous-with-peak-reservation, because a contiguous allocator cannot offer
// expected reservation at all — it has no way to honour a later request for more space,
// so it must take the worst case at admission. But that comparison moves two things at
// once, so the paged-with-peak row is included to separate them: how much of the win is
// paging, and how much is the reservation mode paging makes possible.
func cacheVariants() []cacheVariant {
	p50 := loom.NewQuantileLengthModel(ObservedLengths(Seed), 0.5)
	peak := loom.OutputLengthModel(loom.FractionOfMax{Fraction: 1, Floor: 1})
	return []cacheVariant{
		{"contiguous + peak (best-fit)", SingleWorker(true, loom.BestFit), loom.ReservePeak, peak},
		{"contiguous + peak (first-fit)", SingleWorker(true, loom.FirstFit), loom.ReservePeak, peak},
		{"paged + peak", SingleWorker(false, loom.BestFit), loom.ReservePeak, peak},
		{"paged + expected (p50 model)", SingleWorker(false, loom.BestFit), loom.ReserveExpected, p50},
	}
}

// TestExpCache verifies the cache author's 3.43x concurrency claim (24 versus 7)
// independently, on an identical workload, with the queue short enough that the cache is
// what limits concurrency rather than the arrival rate.
func TestExpCache(t *testing.T) {
	const dur = 180 * time.Second

	// A single reference capacity for every variant, so all four see byte-identical
	// traffic. Deliberately overloaded: the question is how much concurrency each policy
	// can sustain, and a cache that is never full cannot answer it.
	refEst := Estimator(1024, loom.FractionOfMax{Fraction: 1, Floor: 1})
	capacity := Capacity(SingleWorker(false, loom.BestFit), loom.ReservePeak, refEst)
	wl := Workload(Seed, loadgen.Steady, 2.0, dur, capacity, nil)

	tbl := NewTable(
		"KV cache policy — one 1024-block worker, batch cap 64, identical 200%-of-capacity workload",
		"cache / reservation", "peak seqs", "mean conc", "served", "reject", "goodput tok/s",
		"peak util", "peak frag", "preempt", "lat p50", "lat p95", "lat p99")

	for _, v := range cacheVariants() {
		est := Estimator(1024, v.model)
		res := simulator.New(simulator.Config{
			Workers:  v.spec,
			Template: v.spec[0],
			Latency:  simulator.DefaultLatencyModel(),
			// Short queue on purpose: with a deep one every variant eventually serves
			// everything and the difference shows up only as latency. A short queue turns the
			// difference into the thing that matters operationally — how much of the offered
			// load each policy can hold at once, and therefore how much it has to refuse.
			QueueCapacity: 64,
			Policy:        loom.NewFIFOPolicy(),
			Batcher:       simulator.ContinuousBatcherFactory(0, loom.PreemptNewest),
			Estimator:     est,
			Mode:          v.mode,
			TickInterval:  5 * time.Millisecond,
			Arrivals:      wl.Arrivals,
			MaxDuration:   dur + 30*time.Minute,
		}).Run()

		if res.Completed == 0 {
			t.Fatalf("%s completed nothing", v.name)
		}
		tbl.Row(v.name,
			itoa(res.PeakCacheSequences), f2(res.MeanConcurrency), itoa(res.Completed),
			Pct(res.RejectRate()), f1(res.Goodput()), f3(res.PeakCacheUtil), itoa(res.PeakFragmentation),
			itoa64(res.Preemptions), Ms(res.Latency.P50), Ms(res.Latency.P95), Ms(res.Latency.P99))
	}
	tbl.Print(os.Stdout)
}

// TestExpCacheInflationSweep sweeps how much larger the DECLARED output ceiling is than
// the real generation length.
//
// This is the sweep that makes the "3.43x concurrency" claim interpretable. The ratio is
// not a property of the cache implementations — it is roughly the inflation factor,
// because contiguous allocation has to reserve the declared ceiling and paging only has
// to reserve what the sequence reaches. Quoting one number without the inflation it was
// measured at states a workload property as if it were a code property.
func TestExpCacheInflationSweep(t *testing.T) {
	const dur = 120 * time.Second

	tbl := NewTable(
		"Concurrency ratio versus declared-ceiling inflation — one 1024-block worker, single tenant, 200% load",
		"inflation", "contig+peak seqs", "paged+expected seqs", "ratio",
		"contig reject", "paged reject", "contig goodput", "paged goodput", "paged preempt")

	for _, infl := range []float64{1.0, 2.0, 4.0, 8.0, 16.0} {
		tenants := []loadgen.Tenant{{
			ID: "mixed", Share: 1, Weight: 1,
			Prompts:            []loadgen.LengthClass{{Weight: 1, MinTokens: 32, MaxTokens: 300}},
			Outputs:            []loadgen.LengthClass{{Weight: 0.75, MinTokens: 8, MaxTokens: 80}, {Weight: 0.2, MinTokens: 80, MaxTokens: 400}, {Weight: 0.05, MinTokens: 400, MaxTokens: 1200}},
			MaxOutputInflation: infl,
		}}
		refEst := Estimator(1024, loom.FractionOfMax{Fraction: 1, Floor: 1})
		capacity := Capacity(SingleWorker(false, loom.BestFit), loom.ReservePeak, refEst)
		wl := Workload(Seed, loadgen.Steady, 2.0, dur, capacity, tenants)

		// The length model is fitted to the same workload, which is the generous case for
		// expected reservation: in a real deployment the fit is stale.
		lengths := make([]int, len(wl.OutputLengths))
		copy(lengths, wl.OutputLengths)

		run := func(contig bool, mode loom.ReservationMode, model loom.OutputLengthModel) simulator.Result {
			ws := SingleWorker(contig, loom.BestFit)
			return simulator.New(simulator.Config{
				Workers: ws, Template: ws[0], Latency: simulator.DefaultLatencyModel(),
				QueueCapacity: 64, Policy: loom.NewFIFOPolicy(),
				Batcher:   simulator.ContinuousBatcherFactory(0, loom.PreemptNewest),
				Estimator: Estimator(1024, model), Mode: mode,
				TickInterval: 5 * time.Millisecond, Arrivals: wl.Arrivals,
				MaxDuration: dur + 30*time.Minute,
			}).Run()
		}

		c := run(true, loom.ReservePeak, loom.FractionOfMax{Fraction: 1, Floor: 1})
		p := run(false, loom.ReserveExpected, loom.NewQuantileLengthModel(lengths, 0.5))

		ratio := 0.0
		if c.PeakCacheSequences > 0 {
			ratio = float64(p.PeakCacheSequences) / float64(c.PeakCacheSequences)
		}
		tbl.Row(f1(infl)+"x",
			itoa(c.PeakCacheSequences), itoa(p.PeakCacheSequences), f2(ratio)+"x",
			Pct(c.RejectRate()), Pct(p.RejectRate()),
			f1(c.Goodput()), f1(p.Goodput()), itoa64(p.Preemptions))
	}
	tbl.Print(os.Stdout)
}
