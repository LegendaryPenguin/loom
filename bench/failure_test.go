package bench

import (
	"os"
	"testing"
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
	"github.com/LegendaryPenguin/loom/loadgen"
	"github.com/LegendaryPenguin/loom/simulator"
)

// TestExpWorkerFailure kills workers mid-run and measures what it cost.
//
// Three numbers, and the third is the one that is easy to omit and expensive to omit.
// Requests lost and recovery time are the visible costs. Capacity released is the
// invisible one: a reaped worker's reservations are held by the control plane's ledger,
// and if the reaper does not report the blocks it was holding, that capacity is leaked
// permanently and the leak looks like nothing at all until the accounting says the
// cluster is full while the accelerators sit idle. The leak check at the end of the run
// is the assertion, not the table.
func TestExpWorkerFailure(t *testing.T) {
	const dur = 180 * time.Second
	est := Estimator(1024, loom.NewQuantileLengthModel(ObservedLengths(Seed), 0.5))
	capacity := Capacity(policyFleet(), loom.ReserveExpected, est)
	wl := Workload(Seed, loadgen.Steady, 0.85, dur, capacity, nil)

	heartbeat := 500 * time.Millisecond
	deadAfter := 3 * heartbeat

	tbl := NewTable(
		"Worker failure — 4 workers at 85% load, w-2 killed at 45s, w-3 at 60s; heartbeat 500ms, dead after 3 misses ("+deadAfter.String()+")",
		"config", "served", "lost", "retried", "reaped workers", "reaped reqs", "blocks released",
		"recovery", "wait p99 whole run", "wait max", "lat p99", "leak")

	type row struct {
		name    string
		crashes []simulator.Crash
		retry   bool
		replace bool
	}
	kills := []simulator.Crash{{At: 45 * time.Second, Worker: "w-2"}, {At: 60 * time.Second, Worker: "w-3"}}
	rows := []row{
		{"no failure (baseline)", nil, true, false},
		{"2 killed, no retry, no replacement", kills, false, false},
		{"2 killed, retry, no replacement", kills, true, false},
		// Without replacement there is no recovery to measure, only a drain: the fleet has
		// permanently lost half its capacity, so head-of-line wait never comes back. The
		// replacement row is the one where "recovery time" is a real quantity, and it is
		// dominated by the cold start rather than by the detection gap.
		{"2 killed, retry, autoscaler replaces", kills, true, true},
	}

	for _, r := range rows {
		ws := policyFleet()
		cfg := simulator.Config{
			Workers: ws, Template: ws[0], Latency: simulator.DefaultLatencyModel(),
			QueueCapacity: 256, Policy: loom.NewFIFOPolicy(),
			Batcher:   simulator.ContinuousBatcherFactory(0, loom.PreemptNewest),
			Estimator: est, Mode: loom.ReserveExpected,
			TickInterval: 20 * time.Millisecond, HeartbeatInterval: heartbeat, DeadAfterMisses: 3,
			Arrivals: wl.Arrivals, Crashes: r.crashes, RetryOnWorkerLoss: r.retry,
			MaxDuration: dur + 1*time.Hour, SampleInterval: 10 * time.Second,
		}
		if r.replace {
			tmpl := ws[0]
			tmpl.ID = ""
			tmpl.ColdStart = coldStart
			cfg.Template = tmpl
			cfg.AutoscalerFactory = func(c loom.Clock) *loom.QueueWaitAutoscaler {
				return newAutoscaler(c, 4, 4)
			}
		}
		res := simulator.New(cfg).Run()

		leak := "none"
		if !res.Leak.Clean() {
			leak = "LEAK"
			t.Errorf("%s leaked capacity across a worker loss: %+v", r.name, res.Leak)
		}
		rec := "-"
		if len(r.crashes) > 0 {
			rec = "never"
			if res.Recovered {
				rec = res.RecoveryTime.Round(100 * time.Millisecond).String()
			}
		}
		tbl.Row(r.name, itoa(res.Completed), itoa(res.LostOnReap), itoa(res.RetriedOnReap),
			itoa(res.ReapedWorkers), itoa(res.ReapedRequests), itoa(res.ReleasedOnReap), rec,
			Ms(res.QueueWait.P99), Ms(res.QueueWait.Max), Ms(res.Latency.P99), leak)

		if len(r.crashes) > 0 {
			if res.ReapedWorkers < 2 {
				t.Errorf("%s: expected at least 2 reaped workers, got %d", r.name, res.ReapedWorkers)
			}
			if res.ReapedRequests > 0 && res.ReleasedOnReap == 0 {
				t.Errorf("%s: reaper reported %d lost requests but zero released blocks, which is the capacity leak",
					r.name, res.ReapedRequests)
			}
		}
	}
	tbl.Print(os.Stdout)
}

// TestExpWorkerFailureTimeline shows the recovery shape: the detection gap during which
// the control plane does not yet know the worker is gone, the backlog that builds while
// the fleet is short, and the drain back to the pre-failure level.
func TestExpWorkerFailureTimeline(t *testing.T) {
	const dur = 180 * time.Second
	est := Estimator(1024, loom.NewQuantileLengthModel(ObservedLengths(Seed), 0.5))
	capacity := Capacity(policyFleet(), loom.ReserveExpected, est)
	wl := Workload(Seed, loadgen.Steady, 0.85, dur, capacity, nil)

	ws := policyFleet()
	res := simulator.New(simulator.Config{
		Workers: ws, Template: ws[0], Latency: simulator.DefaultLatencyModel(),
		QueueCapacity: 256, Policy: loom.NewFIFOPolicy(),
		Batcher:   simulator.ContinuousBatcherFactory(0, loom.PreemptNewest),
		Estimator: est, Mode: loom.ReserveExpected,
		TickInterval: 20 * time.Millisecond, HeartbeatInterval: 500 * time.Millisecond, DeadAfterMisses: 3,
		Arrivals:          wl.Arrivals,
		Crashes:           []simulator.Crash{{At: 45 * time.Second, Worker: "w-2"}, {At: 60 * time.Second, Worker: "w-3"}},
		RetryOnWorkerLoss: true,
		MaxDuration:       dur + 1*time.Hour, SampleInterval: 5 * time.Second,
	}).Run()

	tbl := NewTable("Worker failure timeline — kills at 45s and 60s, retry on",
		"t", "serving", "running", "queue depth", "head wait", "cache util", "completed")
	for _, s := range res.Samples {
		tbl.Row(s.T.Round(time.Second).String(), itoa(s.Serving), itoa(s.Running),
			itoa(s.QueueDepth), Ms(s.HeadOfLineWait), f3(s.CacheUtil), itoa(s.Completed))
	}
	tbl.Print(os.Stdout)
}
