package simulator

import (
	"time"
)

// LatencyModel is the cost of a forward pass.
//
// The shape matters far more than the constants, because every result in bench/ is
// a comparison between two configurations run through the same model. Three
// properties have to be right or the comparisons are meaningless:
//
//   - Prefill is roughly linear in prompt length and paid once. A long prompt is
//     expensive once; a long generation is expensive forever.
//   - A decode step costs much less than linearly in batch size. The step streams
//     the model weights through memory regardless of how many sequences ride along,
//     so the marginal sequence is cheap. This is the entire reason batching works,
//     and a model with linear batch cost would show batching as free throughput with
//     no latency cost, which is the opposite of the truth.
//   - A decode step gets mildly more expensive as context grows, because attention
//     reads the whole KV cache. Mildly: the weight streaming still dominates at the
//     context lengths here.
type LatencyModel struct {
	// PrefillNanosPerToken is the compute-bound cost of processing one prompt token.
	PrefillNanosPerToken int64

	// DecodeStepNanos is the cost of one decode step for a single sequence at zero
	// context.
	DecodeStepNanos int64

	// BatchMarginalFraction is the extra cost of each sequence past the first, as a
	// fraction of DecodeStepNanos. At 0.1 a batch of 16 costs 2.5x a batch of 1 and
	// does 16x the work.
	BatchMarginalFraction float64

	// ContextGrowthPer1K is how much a decode step slows per 1024 tokens of context,
	// as a fraction of DecodeStepNanos.
	ContextGrowthPer1K float64

	// StepOverheadNanos is the fixed per-step cost — kernel launches, sampling,
	// Python. It is what makes a batch of one wasteful in a way no token count
	// explains.
	StepOverheadNanos int64
}

// DefaultLatencyModel is shaped like a small transformer on a modest accelerator.
func DefaultLatencyModel() LatencyModel {
	return LatencyModel{
		PrefillNanosPerToken:  int64(150 * time.Microsecond),
		DecodeStepNanos:       int64(8 * time.Millisecond),
		BatchMarginalFraction: 0.10,
		ContextGrowthPer1K:    0.25,
		StepOverheadNanos:     int64(1500 * time.Microsecond),
	}
}

// StepDuration is how long one forward pass takes for a batch.
//
// prefillTokens is the prompt length of sequences entering the batch on this step,
// summed; they pay prefill and their first decode token in the same pass, which is
// what a real engine does and what makes time-to-first-token equal to queue wait
// plus one prefill rather than queue wait plus a separate round trip.
func (m LatencyModel) StepDuration(batch int, totalContextTokens, prefillTokens int, speed float64) time.Duration {
	if batch <= 0 {
		return 0
	}
	if speed <= 0 {
		speed = 1
	}

	avgCtx := float64(totalContextTokens) / float64(batch)
	perSeq := float64(m.DecodeStepNanos) * (1 + m.ContextGrowthPer1K*avgCtx/1024)
	decode := perSeq * (1 + m.BatchMarginalFraction*float64(batch-1))

	total := float64(m.StepOverheadNanos) + decode + float64(prefillTokens)*float64(m.PrefillNanosPerToken)
	return time.Duration(total / speed)
}
