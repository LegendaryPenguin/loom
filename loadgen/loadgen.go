// Package loadgen produces seeded, reproducible inference workloads for the
// simulator.
//
// Two design decisions here do most of the work.
//
// First, load is expressed as a TARGET UTILIZATION rather than as an arrival rate.
// A rate is the wrong knob because the interesting measurements are percentiles, and
// a percentile measured at rho > 1 is not a property of the system: the queue grows
// without bound, so every latency percentile is a function of how long the run lasted.
// Reporting "p99 = 12s" from such a run says nothing except that the run was long.
// Expressing the knob as utilization makes the one regime where percentiles mean
// something the default, and makes deliberately entering the other regime an explicit
// request (Overload) whose results are reported as rejection rates and served-request
// latency instead of as a tail.
//
// Second, generation lengths come from a deliberately asymmetric distribution: most
// requests stop early and a minority run very long. That asymmetry is the whole
// reason the paging result matters. If every request ran to its declared ceiling, a
// worst-case reservation would be exactly right and a paged cache would buy nothing.
package loadgen

import (
	"math"
	"math/rand/v2"
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
	"github.com/LegendaryPenguin/loom/simulator"
)

// Shape is the arrival process.
type Shape uint8

const (
	// Steady is a homogeneous Poisson process at the target utilization. Poisson
	// rather than uniform spacing because uniform arrivals into a server with variable
	// service time understate queueing badly — most of the waiting in a real system
	// comes from arrival clustering, not from the mean rate.
	Steady Shape = iota

	// Bursty is a Poisson baseline with periodic windows at BurstFactor times the rate.
	Bursty

	// Overload is a sustained Poisson process above capacity. Percentiles from this
	// shape are reported for SERVED requests only, alongside the rejection rate.
	Overload

	// Ramp walks utilization linearly from RampFrom to RampTo across the run, which is
	// what the autoscaler has to track.
	Ramp

	// Spike is a low baseline with one step change to SpikeFactor for SpikeDuration.
	// Distinct from Bursty: a single sustained step is the case where a scaler should
	// act, where a short burst is the case where it should not.
	Spike
)

func (s Shape) String() string {
	switch s {
	case Bursty:
		return "bursty"
	case Overload:
		return "overload"
	case Ramp:
		return "ramp"
	case Spike:
		return "spike"
	}
	return "steady"
}

// LengthClass is one bucket of the prompt/generation mix.
type LengthClass struct {
	// Weight is relative, so weights need not sum to anything in particular.
	Weight float64

	MinTokens int
	MaxTokens int
}

// Tenant is one workload source.
type Tenant struct {
	ID loom.TenantID

	// Share is this tenant's fraction of the offered request rate, relative to the
	// other tenants' shares.
	Share float64

	// Weight is the fair-share weight the scheduler should be configured with. It is
	// carried here rather than only in the policy so an experiment can assert that
	// served work tracked the weight it was promised.
	Weight float64

	Prompts []LengthClass
	Outputs []LengthClass

	// MaxOutputInflation is how much larger the DECLARED ceiling is than the sampled
	// real length, as a multiplier. This is the gap the whole reservation problem lives
	// in: a client that asks for 2048 and stops at 40 is not misbehaving, it is normal.
	MaxOutputInflation float64

	// Priority is carried straight onto the request.
	Priority int
}

// Config is one workload.
type Config struct {
	Seed uint64

	Shape    Shape
	Duration time.Duration

	// TargetUtilization is the offered load as a fraction of CapacityTokensPerSecond.
	// Above 1.0 the system is deliberately in overload.
	TargetUtilization float64

	// CapacityTokensPerSecond is the fleet's decode throughput, normally from
	// simulator.NominalTokensPerSecond. The arrival rate is derived from it and the
	// mean work per request, so the same workload config produces proportionally more
	// traffic against a bigger fleet.
	CapacityTokensPerSecond float64

	Model loom.ModelID

	Tenants []Tenant

	BurstFactor   float64
	BurstEvery    time.Duration
	BurstDuration time.Duration

	RampFrom float64
	RampTo   float64

	SpikeAt       time.Duration
	SpikeFactor   float64
	SpikeDuration time.Duration

	// Deadline, when non-zero, is every request's budget measured from its own arrival.
	// Relative rather than absolute because the generated workload has no start instant;
	// the simulator converts it when the request arrives.
	Deadline time.Duration
}

// Workload is a generated arrival sequence plus the facts about it a report needs.
type Workload struct {
	Arrivals []simulator.Arrival

	// MeanWorkTokens is the mean decode-equivalent work per request, which is the
	// divisor that turned utilization into a rate. Reported so a result can state the
	// offered load it actually generated rather than the one that was asked for.
	MeanWorkTokens float64
	ArrivalRate    float64
	OfferedTokens  int
	Shape          Shape
	Utilization    float64
	OutputLengths  []int
}

// DefaultTenants is a three-tenant mix: a high-volume short-request tenant, a
// batch tenant with long generations, and a low-volume interactive one.
func DefaultTenants() []Tenant {
	return []Tenant{
		{
			ID: "chat", Share: 0.6, Weight: 1,
			Prompts:            []LengthClass{{Weight: 0.8, MinTokens: 24, MaxTokens: 160}, {Weight: 0.2, MinTokens: 300, MaxTokens: 900}},
			Outputs:            shortTailHeavy(),
			MaxOutputInflation: 4,
		},
		{
			ID: "batch", Share: 0.25, Weight: 1,
			Prompts:            []LengthClass{{Weight: 1, MinTokens: 200, MaxTokens: 1200}},
			Outputs:            []LengthClass{{Weight: 0.5, MinTokens: 200, MaxTokens: 600}, {Weight: 0.5, MinTokens: 600, MaxTokens: 1600}},
			MaxOutputInflation: 1.5,
		},
		{
			ID: "interactive", Share: 0.15, Weight: 2, Priority: 0,
			Prompts:            []LengthClass{{Weight: 1, MinTokens: 16, MaxTokens: 90}},
			Outputs:            []LengthClass{{Weight: 0.9, MinTokens: 8, MaxTokens: 60}, {Weight: 0.1, MinTokens: 60, MaxTokens: 200}},
			MaxOutputInflation: 8,
		},
	}
}

// shortTailHeavy is the asymmetric generation-length mix: ~70% of requests stop in
// under 64 tokens, ~5% run past 800. The long tail is a small minority of requests
// and the majority of the memory-time product, which is why reserving for it costs so
// much and why being able to not reserve for it is worth so much.
func shortTailHeavy() []LengthClass {
	return []LengthClass{
		{Weight: 0.70, MinTokens: 4, MaxTokens: 64},
		{Weight: 0.18, MinTokens: 64, MaxTokens: 220},
		{Weight: 0.07, MinTokens: 220, MaxTokens: 600},
		{Weight: 0.05, MinTokens: 600, MaxTokens: 1800},
	}
}

// AdversarialTenant sends very large prompts at a modest request rate.
//
// Large PROMPTS rather than large generations, because that is the attack a
// per-request or per-second quota does not see: the request count is unremarkable and
// the generation is short, but each one reserves an enormous cache footprint and
// monopolizes prefill compute. It is the case that separates a scheduler that bounds
// requests from one that bounds work.
func AdversarialTenant(share float64) Tenant {
	return Tenant{
		ID: "adversary", Share: share, Weight: 1,
		Prompts:            []LengthClass{{Weight: 1, MinTokens: 3000, MaxTokens: 7000}},
		Outputs:            []LengthClass{{Weight: 1, MinTokens: 8, MaxTokens: 40}},
		MaxOutputInflation: 2,
	}
}

// Generate builds the arrival sequence.
//
// Determinism: all randomness comes from one PCG stream seeded from cfg.Seed, drawn
// in a fixed order, so the same config and seed produce byte-identical arrivals. PCG
// rather than the legacy source because its sequence is specified, not an
// implementation detail.
func Generate(cfg Config) Workload {
	if cfg.Duration <= 0 {
		cfg.Duration = 60 * time.Second
	}
	if cfg.TargetUtilization <= 0 {
		cfg.TargetUtilization = 0.8
	}
	if cfg.CapacityTokensPerSecond <= 0 {
		cfg.CapacityTokensPerSecond = 1000
	}
	if cfg.Model == "" {
		cfg.Model = "tiny-gpt2"
	}
	if len(cfg.Tenants) == 0 {
		cfg.Tenants = DefaultTenants()
	}
	if cfg.BurstFactor <= 0 {
		cfg.BurstFactor = 4
	}
	if cfg.BurstEvery <= 0 {
		cfg.BurstEvery = 10 * time.Second
	}
	if cfg.BurstDuration <= 0 {
		cfg.BurstDuration = 1500 * time.Millisecond
	}
	if cfg.RampFrom <= 0 {
		cfg.RampFrom = 0.15
	}
	if cfg.RampTo <= 0 {
		cfg.RampTo = 1.3
	}
	if cfg.SpikeFactor <= 0 {
		cfg.SpikeFactor = 6
	}
	if cfg.SpikeDuration <= 0 {
		cfg.SpikeDuration = 8 * time.Second
	}
	if cfg.SpikeAt <= 0 {
		cfg.SpikeAt = cfg.Duration / 3
	}

	r := rand.New(rand.NewPCG(cfg.Seed, cfg.Seed^0x9e3779b97f4a7c15))

	// The mean work per request is estimated from the configured mixes rather than
	// measured from the generated sample, so the arrival rate does not depend on the
	// sample it is about to produce. Prefill tokens are converted to decode-equivalents
	// at a fixed ratio because the capacity figure is a decode-token rate.
	const prefillToDecodeEquiv = 0.02
	meanWork := 0.0
	totalShare := 0.0
	for _, t := range cfg.Tenants {
		share := t.Share
		if share <= 0 {
			share = 1
		}
		totalShare += share
		meanWork += share * (meanOf(t.Outputs) + prefillToDecodeEquiv*meanOf(t.Prompts))
	}
	if totalShare <= 0 {
		totalShare = 1
	}
	meanWork /= totalShare
	if meanWork <= 0 {
		meanWork = 1
	}

	baseRate := cfg.TargetUtilization * cfg.CapacityTokensPerSecond / meanWork

	wl := Workload{
		MeanWorkTokens: meanWork,
		ArrivalRate:    baseRate,
		Shape:          cfg.Shape,
		Utilization:    cfg.TargetUtilization,
	}

	// Cumulative tenant shares, for one uniform draw per arrival.
	cum := make([]float64, len(cfg.Tenants))
	acc := 0.0
	for i, t := range cfg.Tenants {
		share := t.Share
		if share <= 0 {
			share = 1
		}
		acc += share / totalShare
		cum[i] = acc
	}

	n := 0
	for t := time.Duration(0); t < cfg.Duration; {
		rate := instantRate(cfg, baseRate, t)
		if rate <= 0 {
			t += 10 * time.Millisecond
			continue
		}
		// Exponential inter-arrival gap: the defining property of a Poisson process, and
		// the source of the clustering that produces most real queueing.
		gap := time.Duration(r.ExpFloat64() / rate * float64(time.Second))
		t += gap
		if t >= cfg.Duration {
			break
		}

		u := r.Float64()
		ti := 0
		for ti < len(cum)-1 && u > cum[ti] {
			ti++
		}
		tn := cfg.Tenants[ti]

		prompt := sampleClass(r, tn.Prompts, 64)
		output := sampleClass(r, tn.Outputs, 32)
		infl := tn.MaxOutputInflation
		if infl < 1 {
			infl = 1
		}
		declared := int(math.Ceil(float64(output) * infl))
		if declared < output {
			declared = output
		}

		n++
		req := loom.Request{
			ID:              loom.RequestID("r-" + itoa(n)),
			Tenant:          tn.ID,
			Model:           cfg.Model,
			Priority:        tn.Priority,
			PromptTokens:    prompt,
			MaxOutputTokens: declared,
		}
		a := simulator.Arrival{Req: req, ActualOutputTokens: output, Offset: t, DeadlineAfter: cfg.Deadline}
		wl.Arrivals = append(wl.Arrivals, a)
		wl.OfferedTokens += output
		wl.OutputLengths = append(wl.OutputLengths, output)
	}
	return wl
}

// instantRate is the arrival rate at time t under the configured shape.
func instantRate(cfg Config, base float64, t time.Duration) float64 {
	switch cfg.Shape {
	case Bursty:
		if cfg.BurstEvery > 0 && t%cfg.BurstEvery < cfg.BurstDuration {
			return base * cfg.BurstFactor
		}
		return base
	case Ramp:
		f := float64(t) / float64(cfg.Duration)
		u := cfg.RampFrom + (cfg.RampTo-cfg.RampFrom)*f
		return base / cfg.TargetUtilization * u
	case Spike:
		if t >= cfg.SpikeAt && t < cfg.SpikeAt+cfg.SpikeDuration {
			return base * cfg.SpikeFactor
		}
		return base
	}
	return base
}

func meanOf(cs []LengthClass) float64 {
	if len(cs) == 0 {
		return 1
	}
	sum, w := 0.0, 0.0
	for _, c := range cs {
		cw := c.Weight
		if cw <= 0 {
			cw = 1
		}
		w += cw
		sum += cw * float64(c.MinTokens+c.MaxTokens) / 2
	}
	if w == 0 {
		return 1
	}
	return sum / w
}

// sampleClass picks a class by weight and then a length uniformly inside it. Uniform
// within the class rather than another distribution because the asymmetry that matters
// is between classes, and a second shape inside each one would add parameters nothing
// depends on.
func sampleClass(r *rand.Rand, cs []LengthClass, fallback int) int {
	if len(cs) == 0 {
		return fallback
	}
	total := 0.0
	for _, c := range cs {
		if c.Weight > 0 {
			total += c.Weight
		} else {
			total += 1
		}
	}
	u := r.Float64() * total
	acc := 0.0
	for _, c := range cs {
		cw := c.Weight
		if cw <= 0 {
			cw = 1
		}
		acc += cw
		if u <= acc {
			lo, hi := c.MinTokens, c.MaxTokens
			if hi <= lo {
				if lo < 1 {
					return 1
				}
				return lo
			}
			return lo + r.IntN(hi-lo+1)
		}
	}
	c := cs[len(cs)-1]
	if c.MaxTokens <= c.MinTokens {
		return c.MinTokens
	}
	return c.MinTokens + r.IntN(c.MaxTokens-c.MinTokens+1)
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
