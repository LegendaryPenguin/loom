// Package bench holds the measured experiments.
//
// Each experiment is a Go test that prints a table. They are tests rather than
// separate programs for one reason: an experiment that nobody runs rots, and
// `go test ./bench/ -v` runs all of them. Each also asserts the minimum it needs to
// be a valid measurement — that something completed, that the accounting balanced —
// so a broken simulation fails loudly instead of printing a plausible table of
// nonsense.
//
// Every experiment is seeded and every number is reproducible. bench/README.md has
// the exact commands and bench/results.md has the output.
package bench

import (
	"fmt"
	"io"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
	"github.com/LegendaryPenguin/loom/loadgen"
	"github.com/LegendaryPenguin/loom/simulator"
)

// Seed is the default seed. Experiments that report a range across runs use
// Seed, Seed+1, ... so the set of repeats is itself reproducible.
const Seed = 20240601

// TokensPerBlock is fixed across every experiment and every component.
//
// It has to match in three places — the worker's registration, the cost estimator's
// cache policy, and the scheduling policies' cost model — because each of them counts
// blocks and a mismatch means two of them are counting different-sized blocks while
// comparing the numbers. The protocol's default is 16.
const TokensPerBlock = 16

// StandardFleet is the fleet most experiments run on: four workers, deliberately
// heterogeneous in speed, with cache capacity that binds before the batch cap does.
//
// Cache-bound rather than batch-cap-bound on purpose. A fleet whose batch cap binds
// first would make the KV-cache policy irrelevant, and the cache is the resource the
// whole design is about.
func StandardFleet() []simulator.WorkerSpec {
	speeds := []float64{1.0, 1.0, 0.75, 1.4}
	out := make([]simulator.WorkerSpec, 4)
	for i := range out {
		out[i] = simulator.WorkerSpec{
			ID:             loom.WorkerID(fmt.Sprintf("w-%d", i+1)),
			Model:          "tiny-gpt2",
			TotalBlocks:    1024,
			TokensPerBlock: TokensPerBlock,
			MaxBatchSize:   24,
			SpeedFactor:    speeds[i],
			ColdStart:      0,
		}
	}
	return out
}

// SingleWorker is the fleet for the cache experiment. One worker, because sustained
// concurrency on one cache is the quantity the paged-versus-contiguous claim is about,
// and spreading the load over four would measure the placement policy instead.
func SingleWorker(contiguous bool, fit loom.FitPolicy) []simulator.WorkerSpec {
	return []simulator.WorkerSpec{{
		ID:             "w-1",
		Model:          "tiny-gpt2",
		TotalBlocks:    1024,
		TokensPerBlock: TokensPerBlock,
		MaxBatchSize:   64,
		SpeedFactor:    1,
		Contiguous:     contiguous,
		Fit:            fit,
	}}
}

// ObservedLengths is a sample of the generation-length distribution, used to build the
// QuantileLengthModel the expected-reservation mode predicts with.
//
// Taken from the generator rather than invented, which is the honest version of what a
// real deployment does: fit the length model to measured traffic. It is also the
// generous case for expected reservation — the model has seen the exact distribution it
// will face — so any preemption it still causes is a lower bound on what a real
// mis-fitted model would cause.
func ObservedLengths(seed uint64) []int {
	wl := loadgen.Generate(loadgen.Config{
		Seed:                    seed,
		Shape:                   loadgen.Steady,
		Duration:                120 * time.Second,
		TargetUtilization:       0.8,
		CapacityTokensPerSecond: 2500,
	})
	return wl.OutputLengths
}

// Estimator builds a cost estimator over the standard block size.
func Estimator(totalBlocks int, length loom.OutputLengthModel) *loom.CostEstimator {
	return loom.NewCostEstimator(loom.CachePolicy{TokensPerBlock: TokensPerBlock, TotalBlocks: totalBlocks}, length)
}

// Capacity is the fleet's MEASURED saturation throughput in output tokens per second.
//
// Measured rather than computed, and the difference is not academic: the closed-form
// estimate in simulator.NominalTokensPerSecond assumes every worker runs a full batch
// forever, and it overstates this fleet by roughly 2.4x because the KV cache binds
// before the batch cap does, because prefill is not free, and because the scheduling
// policy's skip-grace rule periodically drains the fleet to let a large request in. An
// offered load expressed against the optimistic number would put every experiment in
// overload while claiming 80% utilization, and a percentile measured under rho > 1 is a
// function of run length rather than of the system.
//
// So capacity is defined operationally: hammer the fleet through the reference
// configuration — continuous batching, FIFO, the given reservation mode — with a queue
// short enough that it sheds rather than accumulates, and take the goodput it achieves.
// Results are cached per fleet shape because the measurement costs a full simulation.
func Capacity(ws []simulator.WorkerSpec, mode loom.ReservationMode, est *loom.CostEstimator) float64 {
	key := capKey(ws, mode)
	capMu.Lock()
	if v, ok := capCache[key]; ok {
		capMu.Unlock()
		return v
	}
	capMu.Unlock()

	nominal := simulator.NominalTokensPerSecond(ws, simulator.DefaultLatencyModel(), 420)
	probe := loadgen.Generate(loadgen.Config{
		Seed:                    Seed,
		Shape:                   loadgen.Steady,
		Duration:                90 * time.Second,
		TargetUtilization:       8, // far past capacity; the short queue sheds the excess rather than hoarding it
		CapacityTokensPerSecond: nominal,
	})
	res := simulator.New(simulator.Config{
		Workers:  ws,
		Template: ws[0],
		Latency:  simulator.DefaultLatencyModel(),
		// A short queue is the point: with a deep one the fleet would stay saturated but
		// the run would end with a huge backlog, and the drain tail would drag the
		// measured rate down by however long it took.
		QueueCapacity: 48,
		Policy:        loom.NewFIFOPolicy(),
		Batcher:       simulator.ContinuousBatcherFactory(0, loom.PreemptNewest),
		Estimator:     est,
		Mode:          mode,
		TickInterval:  5 * time.Millisecond,
		Arrivals:      probe.Arrivals,
		MaxDuration:   10 * time.Minute,
	}).Run()

	v := res.Goodput()
	capMu.Lock()
	capCache[key] = v
	capMu.Unlock()
	return v
}

var (
	capMu    sync.Mutex
	capCache = map[string]float64{}
)

func capKey(ws []simulator.WorkerSpec, mode loom.ReservationMode) string {
	var b strings.Builder
	fmt.Fprintf(&b, "mode=%v;", mode)
	for _, w := range ws {
		fmt.Fprintf(&b, "%d/%d/%d/%.2f/%v/%v;", w.TotalBlocks, w.TokensPerBlock, w.MaxBatchSize, w.SpeedFactor, w.Contiguous, w.Fit)
	}
	return b.String()
}

// Workload is the standard workload builder, so every experiment that says it used the
// same load really did. cap is the measured capacity the utilization is relative to.
func Workload(seed uint64, shape loadgen.Shape, util float64, dur time.Duration, capacity float64, tenants []loadgen.Tenant) loadgen.Workload {
	return loadgen.Generate(loadgen.Config{
		Seed:                    seed,
		Shape:                   shape,
		Duration:                dur,
		TargetUtilization:       util,
		CapacityTokensPerSecond: capacity,
		Tenants:                 tenants,
	})
}

// Table is a small fixed-width table writer. Hand-rolled because the output has to be
// pasteable into results.md verbatim and stable across runs, which a generic formatter
// with its own column heuristics is not.
type Table struct {
	title string
	head  []string
	rows  [][]string
}

func NewTable(title string, head ...string) *Table {
	return &Table{title: title, head: head}
}

func (t *Table) Row(cells ...string) { t.rows = append(t.rows, cells) }

func (t *Table) Rowf(format string, args ...any) {
	t.rows = append(t.rows, strings.Split(fmt.Sprintf(format, args...), "|"))
}

func (t *Table) Print(w io.Writer) {
	fmt.Fprintf(w, "\n%s\n", t.title)
	fmt.Fprintf(w, "%s\n", strings.Repeat("=", len(t.title)))
	tw := tabwriter.NewWriter(w, 2, 8, 2, ' ', 0)
	fmt.Fprintln(tw, strings.Join(t.head, "\t"))
	sep := make([]string, len(t.head))
	for i, h := range t.head {
		sep[i] = strings.Repeat("-", len(h))
	}
	fmt.Fprintln(tw, strings.Join(sep, "\t"))
	for _, r := range t.rows {
		fmt.Fprintln(tw, strings.Join(r, "\t"))
	}
	tw.Flush()
}

// Pct formats a fraction as a percentage.
func Pct(f float64) string { return fmt.Sprintf("%.1f%%", 100*f) }

// Ms is the millisecond formatter used in every latency column.
func Ms(d time.Duration) string { return simulator.Ms(d) }

// LatencyCells is the p50/p95/p99/max group that appears in most tables. Never a mean
// on its own: a mean latency in a system whose request costs span two orders of
// magnitude is a number nobody experiences.
func LatencyCells(d simulator.Dist) []string {
	return []string{Ms(d.P50), Ms(d.P95), Ms(d.P99), Ms(d.Max)}
}
