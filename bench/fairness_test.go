package bench

import (
	"os"
	"sort"
	"testing"
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
	"github.com/LegendaryPenguin/loom/loadgen"
	"github.com/LegendaryPenguin/loom/simulator"
)

// adversarialTenants is the default mix plus a tenant sending very large prompts.
//
// Large prompts rather than many requests, because that is the case a per-request quota
// does not see: the request count is unremarkable, the generation is short, and each one
// reserves an enormous cache footprint and monopolizes prefill compute. It separates a
// scheduler that bounds requests from one that bounds work.
func adversarialTenants() []loadgen.Tenant {
	return append(loadgen.DefaultTenants(), loadgen.AdversarialTenant(0.25))
}

func fairnessQuotas() map[loom.TenantID]loom.TenantQuota {
	return map[loom.TenantID]loom.TenantQuota{
		// A block ceiling rather than a request ceiling, which is the only dimension that
		// catches the adversary: its request rate is modest and its footprint is not.
		"adversary":   {MaxCacheBlocks: 220, MaxConcurrentRequests: 4},
		"chat":        {MaxCacheBlocks: 2000},
		"batch":       {MaxCacheBlocks: 1400},
		"interactive": {MaxCacheBlocks: 1200},
	}
}

// TestExpTenantFairness reports served work against submitted work per tenant.
//
// Against submitted, never served alone: served output tokens cannot distinguish a tenant
// that was throttled from one that simply sent less, and the ratio is the only form of the
// number that means anything.
func TestExpTenantFairness(t *testing.T) {
	const dur = 240 * time.Second
	est := Estimator(1024, loom.NewQuantileLengthModel(ObservedLengths(Seed), 0.5))
	capacity := Capacity(policyFleet(), loom.ReserveExpected, est)
	wl := Workload(Seed, loadgen.Steady, 1.4, dur, capacity, adversarialTenants())

	type variant struct {
		name   string
		policy func() loom.SchedulePolicy
		quotas map[loom.TenantID]loom.TenantQuota
	}
	variants := []variant{
		{"fifo, no quota", func() loom.SchedulePolicy { return loom.NewFIFOPolicy() }, nil},
		{"fifo + quota", func() loom.SchedulePolicy { return loom.NewFIFOPolicy() }, fairnessQuotas()},
		{"fair-share, no quota", newWeightedFairShare, nil},
		{"fair-share + quota", newWeightedFairShare, fairnessQuotas()},
	}

	for _, v := range variants {
		ws := policyFleet()
		res := simulator.New(simulator.Config{
			Workers: ws, Template: ws[0], Latency: simulator.DefaultLatencyModel(),
			QueueCapacity: 256, Policy: v.policy(), Quotas: v.quotas,
			Batcher:   simulator.ContinuousBatcherFactory(0, loom.PreemptNewest),
			Estimator: est, Mode: loom.ReserveExpected,
			TickInterval: 20 * time.Millisecond,
			Arrivals:     wl.Arrivals, MaxDuration: dur + 1*time.Hour,
		}).Run()

		tbl := NewTable(
			"Tenant fairness — "+v.name+", 140% offered load, adversary sends 3000-7000 token prompts",
			"tenant", "weight", "submitted", "submitted tok", "accepted", "rejected", "completed",
			"served tok", "served/submitted tok", "share of served", "lat p50", "lat p95", "lat p99")

		names := make([]loom.TenantID, 0, len(res.PerTenant))
		for id := range res.PerTenant {
			names = append(names, id)
		}
		sort.Slice(names, func(i, j int) bool { return names[i] < names[j] })

		totalServed := 0
		for _, ts := range res.PerTenant {
			totalServed += ts.ServedTokens
		}
		weights := map[loom.TenantID]float64{}
		for _, tn := range adversarialTenants() {
			weights[tn.ID] = tn.Weight
		}

		for _, id := range names {
			ts := res.PerTenant[id]
			ratio, share := 0.0, 0.0
			if ts.SubmittedTokens > 0 {
				ratio = float64(ts.ServedTokens) / float64(ts.SubmittedTokens)
			}
			if totalServed > 0 {
				share = float64(ts.ServedTokens) / float64(totalServed)
			}
			d := simulator.Summarize(append([]time.Duration(nil), ts.Latencies...))
			tbl.Row(string(id), f1(weights[id]), itoa(ts.Submitted), itoa(ts.SubmittedTokens),
				itoa(ts.Accepted), itoa(ts.Rejected), itoa(ts.Completed), itoa(ts.ServedTokens),
				Pct(ratio), Pct(share), Ms(d.P50), Ms(d.P95), Ms(d.P99))
		}
		tbl.Print(os.Stdout)

		if !res.Leak.Clean() {
			t.Errorf("%s leaked: %+v", v.name, res.Leak)
		}
	}
}

func newWeightedFairShare() loom.SchedulePolicy {
	p := loom.NewFairSharePolicy()
	for _, tn := range adversarialTenants() {
		p.SetWeight(tn.ID, tn.Weight)
	}
	return p
}
