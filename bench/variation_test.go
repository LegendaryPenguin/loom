package bench

import (
	"fmt"
	"os"
	"testing"
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
	"github.com/LegendaryPenguin/loom/loadgen"
	"github.com/LegendaryPenguin/loom/simulator"
)

// TestExpSeedVariation reports the headline numbers across five seeds.
//
// The suite is deterministic, so running the same experiment twice produces identical
// output and tells you nothing about how stable a result is. Varying the seed varies the
// workload, which is the only source of noise there is here, and a difference between two
// configurations is only worth reporting if it survives that variation. Where the ranges
// of two configurations overlap, this file is the evidence that the comparison is too
// noisy to conclude from.
func TestExpSeedVariation(t *testing.T) {
	const dur = 180 * time.Second
	const seeds = 5

	type stat struct {
		thr, good []float64
		p99       []time.Duration
		ttft      []time.Duration
		rej       []float64
	}

	// Batching, at 95% offered load, which is where the three strategies separate most.
	{
		capacity := Capacity(StandardFleet(), loom.ReservePeak, batchingEstimator())
		acc := map[string]*stat{}
		for s := 0; s < seeds; s++ {
			wl := Workload(uint64(Seed+s), loadgen.Steady, 0.95, dur, capacity, nil)
			for _, v := range batchingVariants(0) {
				r := simulator.New(batchingConfig(v, wl, dur)).Run()
				a := acc[v.name]
				if a == nil {
					a = &stat{}
					acc[v.name] = a
				}
				a.thr = append(a.thr, r.Throughput())
				a.good = append(a.good, r.Goodput())
				a.p99 = append(a.p99, r.Latency.P99)
				a.ttft = append(a.ttft, r.TTFT.P50)
				a.rej = append(a.rej, r.RejectRate())
			}
		}
		tbl := NewTable(fmt.Sprintf("Seed variation (%d seeds) — batching at 95%% offered load", seeds),
			"strategy", "served/s", "goodput tok/s", "ttft p50", "lat p99", "reject")
		for _, v := range batchingVariants(0) {
			a := acc[v.name]
			tbl.Row(v.name, rangeF(a.thr, 2), rangeF(a.good, 0), rangeMs(a.ttft), rangeMs(a.p99), rangePct(a.rej))
		}
		tbl.Print(os.Stdout)
	}

	// Cache policy, at 200% offered load on one worker.
	{
		refEst := Estimator(1024, loom.FractionOfMax{Fraction: 1, Floor: 1})
		capacity := Capacity(SingleWorker(false, loom.BestFit), loom.ReservePeak, refEst)
		type cs struct {
			seqs []float64
			good []float64
			rej  []float64
		}
		acc := map[string]*cs{}
		for s := 0; s < seeds; s++ {
			wl := Workload(uint64(Seed+s), loadgen.Steady, 2.0, dur, capacity, nil)
			for _, v := range cacheVariants() {
				est := Estimator(1024, v.model)
				r := simulator.New(simulator.Config{
					Workers: v.spec, Template: v.spec[0], Latency: simulator.DefaultLatencyModel(),
					QueueCapacity: 64, Policy: loom.NewFIFOPolicy(),
					Batcher:   simulator.ContinuousBatcherFactory(0, loom.PreemptNewest),
					Estimator: est, Mode: v.mode,
					TickInterval: 5 * time.Millisecond, Arrivals: wl.Arrivals,
					MaxDuration: dur + 30*time.Minute,
				}).Run()
				a := acc[v.name]
				if a == nil {
					a = &cs{}
					acc[v.name] = a
				}
				a.seqs = append(a.seqs, float64(r.PeakCacheSequences))
				a.good = append(a.good, r.Goodput())
				a.rej = append(a.rej, r.RejectRate())
			}
		}
		tbl := NewTable(fmt.Sprintf("Seed variation (%d seeds) — KV cache policy at 200%% offered load", seeds),
			"cache / reservation", "peak seqs", "goodput tok/s", "reject")
		for _, v := range cacheVariants() {
			a := acc[v.name]
			tbl.Row(v.name, rangeF(a.seqs, 0), rangeF(a.good, 0), rangePct(a.rej))
		}
		tbl.Print(os.Stdout)
	}

	// Queue bound, at 200% offered load.
	{
		est := Estimator(1024, loom.NewQuantileLengthModel(ObservedLengths(Seed), 0.5))
		capacity := Capacity(policyFleet(), loom.ReserveExpected, est)
		type qs struct {
			rej  []float64
			p50  []time.Duration
			p99  []time.Duration
			good []float64
		}
		caps := []int{32, 128, 1000000}
		acc := map[int]*qs{}
		for s := 0; s < seeds; s++ {
			wl := Workload(uint64(Seed+s), loadgen.Steady, 2.0, dur, capacity, nil)
			for _, c := range caps {
				ws := policyFleet()
				r := simulator.New(simulator.Config{
					Workers: ws, Template: ws[0], Latency: simulator.DefaultLatencyModel(),
					QueueCapacity: c, Policy: loom.NewFIFOPolicy(),
					Batcher:   simulator.ContinuousBatcherFactory(0, loom.PreemptNewest),
					Estimator: est, Mode: loom.ReserveExpected,
					TickInterval: 5 * time.Millisecond, Arrivals: wl.Arrivals,
					MaxDuration: dur + 2*time.Hour,
				}).Run()
				a := acc[c]
				if a == nil {
					a = &qs{}
					acc[c] = a
				}
				a.rej = append(a.rej, r.RejectRate())
				a.p50 = append(a.p50, r.Latency.P50)
				a.p99 = append(a.p99, r.Latency.P99)
				a.good = append(a.good, r.Goodput())
			}
		}
		tbl := NewTable(fmt.Sprintf("Seed variation (%d seeds) — queue bound at 200%% offered load", seeds),
			"queue cap", "reject", "SERVED lat p50", "SERVED lat p99", "goodput tok/s")
		for _, c := range caps {
			a := acc[c]
			name := itoa(c)
			if c >= 1000000 {
				name = "1e6 (unbounded)"
			}
			tbl.Row(name, rangePct(a.rej), rangeMs(a.p50), rangeMs(a.p99), rangeF(a.good, 0))
		}
		tbl.Print(os.Stdout)
	}
}

func rangeF(vs []float64, dp int) string {
	if len(vs) == 0 {
		return "-"
	}
	lo, hi := vs[0], vs[0]
	for _, v := range vs {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	return fmt.Sprintf("%.*f..%.*f", dp, lo, dp, hi)
}

func rangePct(vs []float64) string {
	if len(vs) == 0 {
		return "-"
	}
	lo, hi := vs[0], vs[0]
	for _, v := range vs {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	return Pct(lo) + ".." + Pct(hi)
}
