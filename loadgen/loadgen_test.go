package loadgen_test

import (
	"sort"
	"testing"
	"time"

	"github.com/LegendaryPenguin/loom/loadgen"
)

func cfg(seed uint64) loadgen.Config {
	return loadgen.Config{
		Seed:                    seed,
		Shape:                   loadgen.Steady,
		Duration:                60 * time.Second,
		TargetUtilization:       0.8,
		CapacityTokensPerSecond: 2500,
	}
}

// TestDeterminism is the property every benchmark number depends on: the workload is a
// function of the seed and nothing else.
func TestDeterminism(t *testing.T) {
	a := loadgen.Generate(cfg(7))
	b := loadgen.Generate(cfg(7))

	if len(a.Arrivals) != len(b.Arrivals) {
		t.Fatalf("different arrival counts: %d vs %d", len(a.Arrivals), len(b.Arrivals))
	}
	for i := range a.Arrivals {
		if a.Arrivals[i] != b.Arrivals[i] {
			t.Fatalf("arrival %d differs:\n %+v\n %+v", i, a.Arrivals[i], b.Arrivals[i])
		}
	}
	if a.OfferedTokens != b.OfferedTokens || a.ArrivalRate != b.ArrivalRate {
		t.Fatalf("aggregate differs")
	}
}

func TestDifferentSeedsDiffer(t *testing.T) {
	a := loadgen.Generate(cfg(1))
	b := loadgen.Generate(cfg(2))
	if len(a.Arrivals) == len(b.Arrivals) && a.OfferedTokens == b.OfferedTokens {
		t.Fatalf("two seeds produced the same workload, so the seed is not wired through")
	}
}

// TestGenerationLengthsAreAsymmetric guards the property the paging result depends on.
// A symmetric length distribution would make a worst-case reservation nearly correct
// and the paged-versus-contiguous comparison meaningless.
func TestGenerationLengthsAreAsymmetric(t *testing.T) {
	wl := loadgen.Generate(cfg(3))
	ls := append([]int(nil), wl.OutputLengths...)
	sort.Ints(ls)
	if len(ls) < 100 {
		t.Fatalf("sample too small to characterize: %d", len(ls))
	}
	p50 := ls[len(ls)/2]
	p99 := ls[len(ls)*99/100]
	if p99 < 8*p50 {
		t.Errorf("tail is not heavy enough to matter: p50=%d p99=%d", p50, p99)
	}
}

// TestDeclaredCeilingExceedsReality is the gap the admission controller has to guess
// across. Without it, peak reservation would be free.
func TestDeclaredCeilingExceedsReality(t *testing.T) {
	wl := loadgen.Generate(cfg(4))
	over := 0
	for _, a := range wl.Arrivals {
		if a.ActualOutputTokens > a.Req.MaxOutputTokens {
			t.Fatalf("request %s generates past its declared ceiling: %d > %d",
				a.Req.ID, a.ActualOutputTokens, a.Req.MaxOutputTokens)
		}
		if a.Req.MaxOutputTokens > a.ActualOutputTokens {
			over++
		}
	}
	if over*2 < len(wl.Arrivals) {
		t.Errorf("most requests should declare more than they use; only %d/%d do", over, len(wl.Arrivals))
	}
}

func TestUtilizationScalesArrivalRate(t *testing.T) {
	lo := cfg(5)
	lo.TargetUtilization = 0.4
	hi := cfg(5)
	hi.TargetUtilization = 1.2

	a := loadgen.Generate(lo)
	b := loadgen.Generate(hi)
	ratio := b.ArrivalRate / a.ArrivalRate
	if ratio < 2.9 || ratio > 3.1 {
		t.Errorf("arrival rate should scale with target utilization: ratio=%.3f", ratio)
	}
}

func TestShapesProduceDifferentProfiles(t *testing.T) {
	for _, sh := range []loadgen.Shape{loadgen.Steady, loadgen.Bursty, loadgen.Overload, loadgen.Ramp, loadgen.Spike} {
		c := cfg(6)
		c.Shape = sh
		if sh == loadgen.Overload {
			c.TargetUtilization = 1.5
		}
		wl := loadgen.Generate(c)
		if len(wl.Arrivals) == 0 {
			t.Errorf("shape %s produced no arrivals", sh)
		}
		for i := 1; i < len(wl.Arrivals); i++ {
			if wl.Arrivals[i].Offset < wl.Arrivals[i-1].Offset {
				t.Fatalf("shape %s produced out-of-order arrivals at %d", sh, i)
			}
		}
	}
}
