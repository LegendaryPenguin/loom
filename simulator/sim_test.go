package simulator_test

import (
	"testing"
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
	"github.com/LegendaryPenguin/loom/simulator"
)

func fleet(n int) []simulator.WorkerSpec {
	out := make([]simulator.WorkerSpec, n)
	for i := range out {
		out[i] = simulator.WorkerSpec{
			ID:             loom.WorkerID("w-" + string(rune('1'+i))),
			Model:          "tiny-gpt2",
			TotalBlocks:    256,
			TokensPerBlock: 16,
			MaxBatchSize:   16,
			SpeedFactor:    1,
		}
	}
	return out
}

// arrivals builds a fixed, hand-written workload so the simulator's determinism test
// does not depend on loadgen's.
func arrivals(n int) []simulator.Arrival {
	out := make([]simulator.Arrival, 0, n)
	for i := 0; i < n; i++ {
		prompt := 32 + (i*37)%200
		actual := 8 + (i*53)%180
		out = append(out, simulator.Arrival{
			Req: loom.Request{
				ID:              loom.RequestID("r-" + itoa(i)),
				Tenant:          loom.TenantID([]string{"a", "b"}[i%2]),
				Model:           "tiny-gpt2",
				PromptTokens:    prompt,
				MaxOutputTokens: actual * 3,
			},
			ActualOutputTokens: actual,
			Offset:             time.Duration(i) * 7 * time.Millisecond,
		})
	}
	return out
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

func baseConfig() simulator.Config {
	ws := fleet(2)
	return simulator.Config{
		Workers:       ws,
		Template:      ws[0],
		Latency:       simulator.DefaultLatencyModel(),
		QueueCapacity: 512,
		Policy:        loom.NewFIFOPolicy(),
		Batcher:       simulator.ContinuousBatcherFactory(0, loom.PreemptNewest),
		Estimator:     loom.NewCostEstimator(loom.CachePolicy{TokensPerBlock: 16, TotalBlocks: 256}, loom.FractionOfMax{Fraction: 0.4, Floor: 1}),
		Mode:          loom.ReserveExpected,
		TickInterval:  20 * time.Millisecond,
		Arrivals:      arrivals(300),
	}
}

func TestSimulationCompletesEverythingAndLeaksNothing(t *testing.T) {
	res := simulator.New(baseConfig()).Run()

	if res.Completed == 0 {
		t.Fatalf("nothing completed: %+v", res)
	}
	if got := res.Completed + res.Rejected + res.Failed + res.TimedOut; got != res.Submitted {
		t.Errorf("requests unaccounted for: completed=%d rejected=%d failed=%d timedout=%d submitted=%d",
			res.Completed, res.Rejected, res.Failed, res.TimedOut, res.Submitted)
	}
	if !res.Leak.Clean() {
		t.Errorf("accounting leak at end of run: %+v", res.Leak)
	}
	if res.Latency.P99 < res.Latency.P50 {
		t.Errorf("percentiles out of order: %+v", res.Latency)
	}
}

// TestDeterminism is the claim the whole benchmark suite rests on. Two runs of the
// same config must agree on every number, including the tail percentiles, or a
// difference between two experiments cannot be attributed to the thing that differed.
func TestDeterminism(t *testing.T) {
	a := simulator.New(baseConfig()).Run()
	b := simulator.New(baseConfig()).Run()

	if a.Completed != b.Completed || a.Rejected != b.Rejected || a.OutputTokens != b.OutputTokens {
		t.Fatalf("counts differ:\n a=%+v\n b=%+v", a, b)
	}
	if a.Latency != b.Latency || a.QueueWait != b.QueueWait || a.TTFT != b.TTFT {
		t.Fatalf("distributions differ:\n a=%+v\n b=%+v", a.Latency, b.Latency)
	}
	if a.Elapsed != b.Elapsed || a.Steps != b.Steps || a.Preemptions != b.Preemptions {
		t.Fatalf("run shape differs: elapsed %v/%v steps %d/%d preempt %d/%d",
			a.Elapsed, b.Elapsed, a.Steps, b.Steps, a.Preemptions, b.Preemptions)
	}
	if a.MaxConcurrency != b.MaxConcurrency || a.MeanConcurrency != b.MeanConcurrency {
		t.Fatalf("concurrency differs: %d/%d %f/%f", a.MaxConcurrency, b.MaxConcurrency, a.MeanConcurrency, b.MeanConcurrency)
	}
}

// TestRealCacheIsUsed checks that the paged and contiguous caches produce different
// sustained concurrency on an identical workload. If they did not, the simulator would
// be accounting for memory itself instead of delegating to the real implementations,
// and every cache result would be an artifact.
func TestRealCacheIsUsed(t *testing.T) {
	paged := baseConfig()
	res1 := simulator.New(paged).Run()

	contig := baseConfig()
	for i := range contig.Workers {
		contig.Workers[i].Contiguous = true
	}
	contig.Mode = loom.ReservePeak
	res2 := simulator.New(contig).Run()

	if res1.MaxConcurrency <= res2.MaxConcurrency {
		t.Errorf("paged cache did not sustain more concurrency than contiguous: paged=%d contiguous=%d",
			res1.MaxConcurrency, res2.MaxConcurrency)
	}
}
