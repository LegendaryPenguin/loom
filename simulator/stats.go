package simulator

import (
	"fmt"
	"sort"
	"time"
)

// Dist is a latency distribution summary.
//
// Every field except Mean is an order statistic, and that is deliberate: an average
// latency in a serving system is a number that cannot be acted on. The distribution
// of inference request cost spans two orders of magnitude, so the mean sits
// somewhere nobody experiences, and a change that halves the mean while doubling the
// p99 looks like an improvement. Mean is carried only so a report can show how far
// it is from the percentiles.
type Dist struct {
	N    int
	Min  time.Duration
	P50  time.Duration
	P90  time.Duration
	P95  time.Duration
	P99  time.Duration
	Max  time.Duration
	Mean time.Duration
}

// Summarize computes the order statistics. It sorts in place; callers that still
// need the original order must copy first.
func Summarize(ds []time.Duration) Dist {
	if len(ds) == 0 {
		return Dist{}
	}
	sort.Slice(ds, func(i, j int) bool { return ds[i] < ds[j] })
	var sum time.Duration
	for _, d := range ds {
		sum += d
	}
	return Dist{
		N:    len(ds),
		Min:  ds[0],
		P50:  quantile(ds, 0.50),
		P90:  quantile(ds, 0.90),
		P95:  quantile(ds, 0.95),
		P99:  quantile(ds, 0.99),
		Max:  ds[len(ds)-1],
		Mean: sum / time.Duration(len(ds)),
	}
}

// quantile uses nearest-rank on sorted data. Nearest-rank rather than interpolated:
// an interpolated p99 of a sample of 50 invents a value between two observations,
// and a benchmark should only ever report latencies that something actually
// experienced.
func quantile(sorted []time.Duration, q float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(q * float64(len(sorted)-1))
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// Ms formats a duration in milliseconds for a table column.
func Ms(d time.Duration) string {
	return fmt.Sprintf("%.1f", float64(d)/float64(time.Millisecond))
}
