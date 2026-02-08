package bench

import (
	"os"
	"testing"
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
	"github.com/LegendaryPenguin/loom/loadgen"
	"github.com/LegendaryPenguin/loom/simulator"
)

// batchingVariant is one of the three batching strategies under test.
type batchingVariant struct {
	name     string
	maxBatch int
	factory  simulator.BatcherFactory
}

func batchingVariants(window time.Duration) []batchingVariant {
	return []batchingVariant{
		// No batching is the static batcher with a cap of one. It is not a separate
		// implementation because it should not be: "no batching" is a batch size, and
		// giving it its own code path would let it differ from the baseline in ways that
		// have nothing to do with batching.
		{"none (batch=1)", 1, simulator.StaticBatcherFactory(0)},
		{"static", 24, simulator.StaticBatcherFactory(window)},
		{"continuous", 24, simulator.ContinuousBatcherFactory(window, loom.PreemptNewest)},
	}
}

// batchingEstimator is peak reservation for all three variants, so the KV cache behaves
// identically and the only thing that differs is how the batch is formed. Changing the
// reservation mode at the same time would make the result uninterpretable.
func batchingEstimator() *loom.CostEstimator {
	return Estimator(1024, loom.FractionOfMax{Fraction: 1, Floor: 1})
}

func batchingConfig(v batchingVariant, wl loadgen.Workload, dur time.Duration) simulator.Config {
	ws := StandardFleet()
	for i := range ws {
		ws[i].MaxBatchSize = v.maxBatch
	}
	return simulator.Config{
		Workers:       ws,
		Template:      ws[0],
		Latency:       simulator.DefaultLatencyModel(),
		QueueCapacity: 512,
		Policy:        loom.NewFIFOPolicy(),
		Batcher:       v.factory,
		Estimator:     batchingEstimator(),
		Mode:          loom.ReservePeak,
		TickInterval:  5 * time.Millisecond,
		Arrivals:      wl.Arrivals,
		MaxDuration:   dur + 30*time.Minute,
	}
}

func TestExpBatching(t *testing.T) {
	const dur = 180 * time.Second
	capacity := Capacity(StandardFleet(), loom.ReservePeak, batchingEstimator())

	for _, util := range []float64{0.30, 0.70, 0.95} {
		// The offered load is sized against the capacity of the REFERENCE configuration
		// for every variant, including the batch-of-one one. That is the comparison that
		// matters: the same traffic arrives and the question is what each strategy does
		// with it. Sizing each variant's load to its own capacity would hide the fact
		// that one of them has a seventh of the capacity of the others.
		wl := Workload(Seed, loadgen.Steady, util, dur, capacity, nil)

		tbl := NewTable(
			"Batching strategy — offered load "+Pct(util)+" of measured fleet capacity ("+f1(capacity)+" tok/s), FIFO, peak reservation",
			"strategy", "served/s", "goodput tok/s", "reject", "ttft p50", "ttft p95", "ttft p99",
			"lat p50", "lat p95", "lat p99", "lat max", "mean conc", "steps")

		for _, v := range batchingVariants(0) {
			res := simulator.New(batchingConfig(v, wl, dur)).Run()

			if res.Completed == 0 {
				t.Fatalf("%s completed nothing", v.name)
			}
			if !res.Leak.Clean() {
				t.Errorf("%s leaked: %+v", v.name, res.Leak)
			}

			cells := []string{
				v.name,
				f2(res.Throughput()),
				f1(res.Goodput()),
				Pct(res.RejectRate()),
				Ms(res.TTFT.P50), Ms(res.TTFT.P95), Ms(res.TTFT.P99),
			}
			cells = append(cells, LatencyCells(res.Latency)...)
			cells = append(cells, f2(res.MeanConcurrency), itoa64(res.Steps))
			tbl.Row(cells...)
		}
		tbl.Print(os.Stdout)
	}
}

// TestExpBatchingWindowSweep walks the accumulation window, which is the whole
// latency/throughput dial. Both batchers are swept, because the window means something
// different to each: the static batcher always waits for company, while the continuous
// batcher only waits when nothing is running, so the dial should be nearly flat for it
// under load. That asymmetry is a design claim and this is the measurement of it.
func TestExpBatchingWindowSweep(t *testing.T) {
	const dur = 180 * time.Second
	capacity := Capacity(StandardFleet(), loom.ReservePeak, batchingEstimator())
	windows := []time.Duration{0, 1 * time.Millisecond, 2 * time.Millisecond, 5 * time.Millisecond,
		10 * time.Millisecond, 20 * time.Millisecond, 50 * time.Millisecond}

	for _, util := range []float64{0.10, 0.30, 0.70} {
		wl := Workload(Seed, loadgen.Steady, util, dur, capacity, nil)
		for _, which := range []string{"static", "continuous"} {
			tbl := NewTable(
				"Batching window sweep — "+which+" batcher, "+Pct(util)+" offered load, FIFO, peak reservation",
				"window", "served/s", "goodput tok/s", "reject", "ttft p50", "ttft p95", "ttft p99",
				"lat p50", "lat p95", "lat p99", "mean batch", "steps")

			for _, w := range windows {
				var v batchingVariant
				for _, c := range batchingVariants(w) {
					if c.name == which {
						v = c
					}
				}
				res := simulator.New(batchingConfig(v, wl, dur)).Run()
				if res.Completed == 0 {
					t.Fatalf("%s/%v completed nothing", which, w)
				}

				// Mean batch size is sequence-steps over steps: the direct measure of whether
				// the window is actually filling batches, which neither throughput nor latency
				// can show on its own.
				meanBatch := 0.0
				if res.Steps > 0 {
					meanBatch = float64(res.SequenceSteps) / float64(res.Steps)
				}

				tbl.Row(
					w.String(),
					f2(res.Throughput()), f1(res.Goodput()), Pct(res.RejectRate()),
					Ms(res.TTFT.P50), Ms(res.TTFT.P95), Ms(res.TTFT.P99),
					Ms(res.Latency.P50), Ms(res.Latency.P95), Ms(res.Latency.P99),
					f2(meanBatch), itoa64(res.Steps),
				)
			}
			tbl.Print(os.Stdout)
		}
	}
}
