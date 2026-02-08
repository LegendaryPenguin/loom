package bench

import (
	"os"
	"testing"
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
	"github.com/LegendaryPenguin/loom/loadgen"
	"github.com/LegendaryPenguin/loom/simulator"
)

// coldStart is deliberately long relative to the experiment, because a cold start that is
// short compared to the demand it is answering makes autoscaling look free. Loading a
// model is tens of seconds, and the whole reason the scaler has hysteresis is that the
// capacity it orders is not there when it orders it.
const coldStart = 20 * time.Second

// autoscaleFleet starts at one worker so the ramp has somewhere to go. Every worker,
// including the starting one, carries the cold-start delay: a fleet whose first worker is
// instantly warm would hide the startup transient.
func autoscaleFleet(n int) []simulator.WorkerSpec {
	ws := policyFleet()[:n]
	out := make([]simulator.WorkerSpec, n)
	for i := range out {
		out[i] = ws[i]
		out[i].SpeedFactor = 1
		out[i].ColdStart = 0
	}
	return out
}

func autoscaleTemplate() simulator.WorkerSpec {
	s := autoscaleFleet(1)[0]
	s.ID = ""
	s.ColdStart = coldStart
	return s
}

func newAutoscaler(clock loom.Clock, min, max int) *loom.QueueWaitAutoscaler {
	return loom.NewQueueWaitAutoscaler(loom.AutoscalerConfig{
		Clock:             clock,
		ScaleUpWait:       1500 * time.Millisecond,
		ScaleDownWait:     200 * time.Millisecond,
		ScaleUpDepth:      16,
		SustainedFor:      3 * time.Second,
		ColdStartDuration: coldStart,
		ScaleUpCooldown:   5 * time.Second,
		ScaleDownCooldown: 30 * time.Second,
		MaxScaleUpStep:    2,
		MinWorkers:        min,
		MaxWorkers:        max,
	})
}

func autoscaleConfig(shape loadgen.Shape, on bool, dur time.Duration, capacity float64) simulator.Config {
	est := Estimator(1024, loom.NewQuantileLengthModel(ObservedLengths(Seed), 0.5))
	wl := loadgen.Generate(loadgen.Config{
		Seed:                    Seed,
		Shape:                   shape,
		Duration:                dur,
		TargetUtilization:       1.0,
		CapacityTokensPerSecond: capacity,
		RampFrom:                0.2,
		RampTo:                  1.1,
		SpikeAt:                 dur / 4,
		SpikeFactor:             5,
		SpikeDuration:           25 * time.Second,
	})

	start := 1
	if !on {
		// Scaling off is compared against a fleet sized for the PEAK, not against a fleet
		// of one. A static fleet of one versus an elastic fleet of four is a comparison of
		// fleet sizes, not of scaling; the question an operator actually faces is whether
		// elasticity can match a statically over-provisioned fleet while owning fewer
		// worker-seconds.
		start = 4
	}
	cfg := simulator.Config{
		Workers:        autoscaleFleet(start),
		Template:       autoscaleTemplate(),
		Latency:        simulator.DefaultLatencyModel(),
		QueueCapacity:  256,
		Policy:         loom.NewFIFOPolicy(),
		Batcher:        simulator.ContinuousBatcherFactory(0, loom.PreemptNewest),
		Estimator:      est,
		Mode:           loom.ReserveExpected,
		TickInterval:   20 * time.Millisecond,
		Arrivals:       wl.Arrivals,
		MaxDuration:    dur + 30*time.Minute,
		SampleInterval: 10 * time.Second,
	}
	return cfg
}

func TestExpAutoscaling(t *testing.T) {
	const dur = 240 * time.Second
	est := Estimator(1024, loom.NewQuantileLengthModel(ObservedLengths(Seed), 0.5))
	// Capacity of the four-worker fleet, so "100% offered load" means the load a fully
	// scaled-out fleet can just absorb. The one-worker starting fleet therefore begins at
	// roughly 4x its own capacity, which is the situation scaling exists for.
	capacity := Capacity(autoscaleFleet(4), loom.ReserveExpected, est)

	for _, shape := range []loadgen.Shape{loadgen.Ramp, loadgen.Spike} {
		tbl := NewTable(
			"Autoscaling under a "+shape.String()+" — cold start "+coldStart.String()+", scale-up wait 1.5s, sustained 3s",
			"config", "served", "reject", "goodput tok/s", "wait p50", "wait p95", "wait p99", "wait max",
			"lat p99", "workers end", "peak workers", "scale ups", "scale downs", "worker-seconds")

		var series [2]simulator.Result
		for i, on := range []bool{false, true} {
			cfg := autoscaleConfig(shape, on, dur, capacity)
			if on {
				// Built against the simulation's clock, not a fresh one: every cooldown and
				// sustained-demand window in the scaler is a function of elapsed time.
				cfg.AutoscalerFactory = func(c loom.Clock) *loom.QueueWaitAutoscaler {
					return newAutoscaler(c, 1, 4)
				}
			}
			res := simulator.New(cfg).Run()
			series[i] = res

			name := "static fleet of 4"
			if on {
				name = "autoscaled 1..4"
			}
			ws := 0.0
			for _, s := range res.Samples {
				ws += float64(s.Serving+s.Starting) * 10
			}
			tbl.Row(name, itoa(res.Completed), Pct(res.RejectRate()), f1(res.Goodput()),
				Ms(res.QueueWait.P50), Ms(res.QueueWait.P95), Ms(res.QueueWait.P99), Ms(res.QueueWait.Max),
				Ms(res.Latency.P99), itoa(res.FinalWorkers), itoa(res.PeakWorkers),
				itoa(res.ScaleUpActions), itoa(res.ScaleDownActions), f1(ws))
		}
		tbl.Print(os.Stdout)

		ts := NewTable("Time series — "+shape.String()+", autoscaled",
			"t", "queue depth", "head wait", "serving", "starting", "draining", "running", "cache util", "completed", "rejected")
		for _, s := range series[1].Samples {
			ts.Row(s.T.Round(time.Second).String(), itoa(s.QueueDepth), Ms(s.HeadOfLineWait),
				itoa(s.Serving), itoa(s.Starting), itoa(s.Draining), itoa(s.Running),
				f3(s.CacheUtil), itoa(s.Completed), itoa(s.Rejected))
		}
		ts.Print(os.Stdout)

		if series[1].ScaleUpActions == 0 {
			t.Errorf("%s: autoscaler never scaled up", shape)
		}
	}
}
