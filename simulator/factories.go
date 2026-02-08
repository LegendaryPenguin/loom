package simulator

import (
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
)

// StaticBatcherFactory builds the baseline batcher for each worker.
func StaticBatcherFactory(maxWait time.Duration) BatcherFactory {
	return func(cache loom.KVCache, est *loom.CostEstimator, mode loom.ReservationMode) loom.Batcher {
		return loom.NewStaticBatcher(maxWait, est, mode, cache)
	}
}

// ContinuousBatcherFactory builds the continuous batcher for each worker.
func ContinuousBatcherFactory(maxWait time.Duration, victims loom.VictimPolicy) BatcherFactory {
	return func(cache loom.KVCache, est *loom.CostEstimator, mode loom.ReservationMode) loom.Batcher {
		return loom.NewContinuousBatcher(maxWait, est, mode, cache, victims)
	}
}

// NominalTokensPerSecond is the fleet's decode throughput at a full batch, which is
// what a workload generator needs to turn a target utilization into an arrival rate.
//
// It is a first-order estimate and is documented as one: it assumes every worker runs
// a full batch of average-context sequences and ignores prefill, so the real
// saturation point sits somewhat below it. That is the right direction for the error —
// a target utilization of 0.9 that is really 0.95 still measures a loaded system,
// whereas one that is really 1.1 measures a queue growing without bound and reports
// percentiles that are a function of run length.
func NominalTokensPerSecond(workers []WorkerSpec, m LatencyModel, avgContextTokens int) float64 {
	total := 0.0
	for _, raw := range workers {
		w := raw.withDefaults()
		batch := w.MaxBatchSize
		// Cache blocks, not the batch cap, are usually the binding constraint, so the
		// effective batch is the smaller of the two.
		if avgContextTokens > 0 {
			byCache := w.TotalBlocks / ((avgContextTokens + w.TokensPerBlock - 1) / w.TokensPerBlock)
			if byCache < batch {
				batch = byCache
			}
		}
		if batch < 1 {
			batch = 1
		}
		step := m.StepDuration(batch, batch*avgContextTokens, 0, w.SpeedFactor)
		if step <= 0 {
			continue
		}
		total += float64(batch) / step.Seconds()
	}
	return total
}
