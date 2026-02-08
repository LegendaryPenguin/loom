package bench

import (
	"os"
	"testing"
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
	"github.com/LegendaryPenguin/loom/loadgen"
	"github.com/LegendaryPenguin/loom/simulator"
)

// TestExpCacheDrift measures the gap between what the control plane predicted a
// sequence's cache footprint would be and what the worker reported it to be.
//
// The sign is the whole value of the number, so it is never aggregated as an absolute:
// positive drift means capacity is sitting idle inside reservations that will never be
// used, and negative means the next admission is being made against a cache fuller than
// the controller thinks — one wastes money and the other kills a generation mid-flight.
//
// Two reservation modes, and the headline is that they fail in opposite directions. Peak
// reservation produces essentially no drift, because the cache really did allocate the
// worst case the controller asked for — its waste is invisible to drift entirely and
// shows up only as reserved-but-never-touched blocks, which is the last column.
func TestExpCacheDrift(t *testing.T) {
	const dur = 180 * time.Second
	lengths := ObservedLengths(Seed)

	tbl := NewTable(
		"Estimate versus worker report — 4 workers at 85% offered load, continuous batching",
		"reservation mode", "length model", "reconciles", "drift last", "mean signed drift",
		"mean |drift|", "worst over", "worst under", "preempt", "blocks over-reserved",
		"blocks needed", "waste", "served", "goodput tok/s")

	type variant struct {
		mode  loom.ReservationMode
		name  string
		model loom.OutputLengthModel
	}
	variants := []variant{
		{loom.ReservePeak, "peak", loom.FractionOfMax{Fraction: 1, Floor: 1}},
		{loom.ReserveExpected, "quantile p50", loom.NewQuantileLengthModel(lengths, 0.5)},
		{loom.ReserveExpected, "quantile p90", loom.NewQuantileLengthModel(lengths, 0.9)},
		{loom.ReserveExpected, "fraction 0.25", loom.FractionOfMax{Fraction: 0.25, Floor: 1}},
	}

	for _, v := range variants {
		est := Estimator(1024, v.model)
		ws := policyFleet()
		capacity := Capacity(ws, v.mode, est)
		wl := Workload(Seed, loadgen.Steady, 0.85, dur, capacity, nil)

		res := simulator.New(simulator.Config{
			Workers: ws, Template: ws[0], Latency: simulator.DefaultLatencyModel(),
			QueueCapacity: 256, Policy: loom.NewFIFOPolicy(),
			Batcher:   simulator.ContinuousBatcherFactory(0, loom.PreemptNewest),
			Estimator: est, Mode: v.mode,
			TickInterval: 20 * time.Millisecond, HeartbeatInterval: 500 * time.Millisecond,
			Arrivals: wl.Arrivals, MaxDuration: dur + 1*time.Hour,
		}).Run()

		m := res.Admission
		meanSigned, meanAbs := 0.0, 0.0
		if m.Reconciles > 0 {
			meanSigned = float64(m.DriftSum) / float64(m.Reconciles)
			meanAbs = float64(m.DriftAbsSum) / float64(m.Reconciles)
		}
		waste := 0.0
		if res.NeededBlocks > 0 {
			waste = float64(res.ReservedBlockBeyondNeed) / float64(res.NeededBlocks)
		}
		tbl.Row(v.mode.String(), v.name, itoa64(m.Reconciles), itoa(m.ReconcileDriftLast),
			f2(meanSigned), f2(meanAbs), itoa(m.DriftMaxOver), itoa(m.DriftMaxUnder),
			itoa64(res.Preemptions), itoa(res.ReservedBlockBeyondNeed), itoa(res.NeededBlocks),
			Pct(waste), itoa(res.Completed), f1(res.Goodput()))
	}
	tbl.Print(os.Stdout)
}
