# bench — the measured experiments

Every experiment here is a Go test that prints a table. They are tests rather than
standalone programs because an experiment nobody runs rots, and `go test ./bench/ -v`
runs all of them in a few seconds. Each one also asserts the minimum it needs to be a
valid measurement — that something completed, that the block accounting balanced to
zero — so a broken simulation fails loudly instead of printing a plausible table of
nonsense.

The actual output, with a reading of each result, is in [results.md](results.md).

## How to run

```sh
export PATH="/opt/homebrew/bin:$PATH"

# everything
go test ./bench/ -v -timeout 900s

# one experiment at a time (names below)
go test ./bench/ -v -timeout 900s -run TestExpBatching$
```

The suite is seeded (`bench.Seed`) and runs in virtual time, so the numbers are
reproducible to the digit. Verify that claim with:

```sh
go test ./simulator/ -run TestDeterminism -v
go test ./loadgen/   -run TestDeterminism -v
```

Build and vet gates:

```sh
go build ./... && go vet ./... && go test -race ./... -timeout 1200s
```

## What the pieces are

- **`simulator/`** — a deterministic virtual-time simulation of a worker fleet. It
  simulates exactly two things: how long a forward pass takes, and when a sequence
  stops producing tokens. Everything that *decides* — `BoundedQueue`, the three
  scheduling policies, `QuotaAdmissionController`, both batchers, `Registry`,
  `QueueWaitAutoscaler` — is the real implementation from `go-control-plane/`, and so
  is the KV-cache accounting (`PagedCache`, `ContiguousCache`). Accounting that only
  existed in the simulator would prove nothing.
- **`loadgen/`** — seeded workload generation. Load is expressed as a **target
  utilization**, not an arrival rate, because a percentile measured at ρ > 1 is a
  function of run length rather than of the system.
- **`bench/`** — the experiments.

### Why utilization is measured, not computed

`bench.Capacity` does not use a closed-form capacity estimate. It measures saturation
throughput by hammering the fleet through a reference configuration (continuous
batching, FIFO, the given reservation mode) with a queue short enough to shed rather
than hoard, and takes the goodput achieved.

This matters: the closed-form estimate in `simulator.NominalTokensPerSecond` overstates
the standard fleet by about 2.4x, because the KV cache binds before the batch cap does,
because prefill is not free, and because the scheduling policy's skip-grace rule
periodically drains the fleet to admit one large request. Sizing offered load against
the optimistic number put every early run in overload while the table claimed 80%
utilization.

## The experiments

| Test | What it shows |
|---|---|
| `TestExpBatching` | No batching vs static vs continuous: throughput, goodput, rejection, time-to-first-token, and p50/p95/p99/max latency, at 30%, 70% and 95% offered load. |
| `TestExpBatchingWindowSweep` | The accumulation window from 0 to 50ms, for both batchers, at 10%/30%/70% load, with mean batch size so the dial is visible rather than inferred. |
| `TestExpPolicies` | FIFO vs SJF vs SJF-with-aging vs fair-share: p50/p95/p99 **and max** for both end-to-end latency and queue wait, at 70%/90%/110%/150% load. |
| `TestExpPoliciesAcrossSeeds` | The same comparison over five seeds, reported as ranges, with a count of which policy won p99 and which won the max. A tail claim from one seed is a claim about one or two observations. |
| `TestExpCache` | Paged vs contiguous (both fits) on an identical overloaded workload: sustained concurrency, goodput, rejection rate, peak fragmentation. Includes `paged + peak` to separate the paging win from the reservation-mode win it enables. |
| `TestExpCacheInflationSweep` | The concurrency ratio as a function of how much larger the declared output ceiling is than the real generation length. This is what makes a single "3.43x" number interpretable. |
| `TestExpQueueBound` | Queue capacity 32/128/512/1e6 under 130% and 200% load. Latency of **served** requests and rejection rate are reported side by side and never blended. |
| `TestExpQueueDepthOverTime` | The shape of the same thing: a bounded queue plateaus, an unbounded one ramps for as long as the overload lasts. |
| `TestExpAutoscaling` | Scaling on vs off under a traffic ramp and under a spike, with a 20s cold start: queue wait, worker count over time, and worker-seconds owned. |
| `TestExpWorkerFailure` | Two workers killed mid-run: requests lost, recovery time, blocks released by the reaper, and a leak assertion. Four rows, including one where the autoscaler replaces the dead workers — without replacement there is no recovery to measure, only a drain. |
| `TestExpWorkerFailureTimeline` | The detection gap, the backlog, and the drain. |
| `TestExpCacheDrift` | Estimate-minus-report drift under peak and three expected-reservation length models, with the sign kept, plus reserved-but-never-touched blocks. |
| `TestExpTenantFairness` | Submitted vs served work per tenant, with and without quota and fair-share, against an adversarial tenant sending 3000–7000 token prompts. |
| `TestExpSeedVariation` | The headline numbers of experiments 1, 4 and 5 across five seeds, as min..max ranges. Since the suite is deterministic, the seed is the only source of noise there is, and this is where a comparison gets ruled inconclusive. |

## Reading the tables

- Latency columns are milliseconds.
- `lat` is end-to-end (arrival to last token). `wait` is arrival to first dispatch.
  `ttft` is arrival to first token. They answer different questions and a single
  latency number hides which one is bad.
- `served/s` and `goodput tok/s` are divided by the interval from the start of the run
  to the **last observable event**, not by the configured run bound.
- A mean is never reported without the percentiles beside it.
