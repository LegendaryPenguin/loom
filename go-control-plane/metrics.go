package loom

import (
	"bufio"
	"io"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// The exporter is hand-written rather than built on a client library.
//
// The Prometheus text format is a dozen lines of printing: a name, optional
// labels, a float, and for a histogram a set of cumulative `_bucket` series plus
// `_sum` and `_count`. Taking a dependency — and its transitive dependencies, and
// its global default registry, and its init-time side effects — to avoid writing
// that is a poor trade in a project whose whole point is that the resource
// accounting is explicit and inspectable.
//
// Concurrency: counters and gauges are atomics, because they are touched on every
// request from every goroutine in the system and a single mutex over all of them
// would make the metrics the contended thing in a system built to study
// contention. Histograms take a per-histogram mutex instead, because an
// observation must advance a bucket, the sum, and the count together — with
// atomics a concurrent scrape can observe a count that includes an observation
// whose sum has not landed, and a histogram whose average is wrong in a way that
// looks like a real latency anomaly is worse than one that is briefly locked.

// The bucket boundaries are upper bounds in the metric's own unit. They are
// chosen to be dense where the decisions are: single-digit milliseconds through a
// few seconds for queue wait and time-to-first-token, because that is the range
// where a serving system goes from fine to unusable, and a histogram whose
// boundaries straddle that range in one jump cannot show the transition.
var (
	latencyBucketsSeconds = []float64{
		0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
		1, 2.5, 5, 10, 30, 60,
	}
	batchSizeBuckets  = []float64{1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256}
	tokensPerSecBucks = []float64{1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000}
)

// counter is a monotonic count. Nothing in the API can decrease it; a counter
// that can go down breaks every rate() over it.
type counter struct {
	v atomic.Uint64
}

func (c *counter) inc()         { c.v.Add(1) }
func (c *counter) add(n uint64) { c.v.Add(n) }
func (c *counter) get() uint64  { return c.v.Load() }

// gauge is a current value, signed because cache drift — estimate minus actual —
// is informative in both directions and clamping it at zero would hide the
// dangerous half.
type gauge struct {
	v atomic.Int64
}

func (g *gauge) set(n int64) { g.v.Store(n) }
func (g *gauge) get() int64  { return g.v.Load() }

// labelled is a fixed set of child series keyed by one label value. The key set
// is fixed at construction so the hot path never writes to a map, which is both
// the race and the allocation that a dynamic-label metric costs.
type labelled struct {
	label string
	keys  []string
	kids  map[string]*counter
}

func newLabelledCounter(label string, keys []string) *labelled {
	l := &labelled{label: label, keys: append([]string(nil), keys...), kids: make(map[string]*counter, len(keys))}
	sort.Strings(l.keys)
	for _, k := range l.keys {
		l.kids[k] = &counter{}
	}
	return l
}

func (l *labelled) inc(key string) {
	if c, ok := l.kids[key]; ok {
		c.inc()
	}
}

type labelledGauge struct {
	label string
	keys  []string
	kids  map[string]*gauge
}

func newLabelledGauge(label string, keys []string) *labelledGauge {
	l := &labelledGauge{label: label, keys: append([]string(nil), keys...), kids: make(map[string]*gauge, len(keys))}
	sort.Strings(l.keys)
	for _, k := range l.keys {
		l.kids[k] = &gauge{}
	}
	return l
}

func (l *labelledGauge) set(key string, v int64) {
	if g, ok := l.kids[key]; ok {
		g.set(v)
	}
}

// histogram is a real bucketed histogram.
//
// Bucketed rather than an average, because an average is exactly the statistic
// that hides overload. A mean latency of 200ms is produced both by a system where
// every request takes 200ms and by one where nine in ten take 20ms and the tenth
// takes two seconds, and only the second one is failing its users. The behaviour
// this project exists to demonstrate lives in the tail.
type histogram struct {
	bounds []float64

	mu     sync.Mutex
	counts []uint64 // one per bound, plus a final +Inf slot
	sum    float64
	n      uint64
}

func newHistogram(bounds []float64) *histogram {
	return &histogram{bounds: bounds, counts: make([]uint64, len(bounds)+1)}
}

func (h *histogram) observe(v float64) {
	// NaN would corrupt the sum permanently and silently; drop it at the door.
	if math.IsNaN(v) {
		return
	}
	i := sort.SearchFloat64s(h.bounds, v)
	// SearchFloat64s finds the first bound >= v, which is the bucket whose upper
	// bound includes v since Prometheus boundaries are inclusive.
	h.mu.Lock()
	h.counts[i]++
	h.sum += v
	h.n++
	h.mu.Unlock()
}

// snapshot returns cumulative bucket counts, the sum, and the observation count,
// read under one lock so the three are mutually consistent.
func (h *histogram) snapshot() ([]uint64, float64, uint64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	cum := make([]uint64, len(h.counts))
	var running uint64
	for i, c := range h.counts {
		running += c
		cum[i] = running
	}
	return cum, h.sum, h.n
}

// Metrics is the control plane's instrumentation and its /metrics exporter.
type Metrics struct {
	namespace string

	arrived   counter
	accepted  counter
	completed counter
	failed    counter
	cancelled counter
	preempted counter
	timedOut  counter

	workerFailures counter
	reaped         counter
	reapedRequests counter

	rejected *labelled // by RejectReason
	retries  *labelled // by RetryAction

	queueDepth       gauge
	queueCapacity    gauge
	activeSequences  gauge
	cacheBlocksUsed  gauge
	cacheBlocksTotal gauge

	// cacheDrift is estimated minus actual blocks. Positive means the control
	// plane over-reserved and is wasting capacity it could be admitting into;
	// negative means it under-reserved and the next admission may overcommit the
	// device. The sign is the whole message, which is why this is a signed gauge
	// and not two counters.
	cacheDrift gauge

	workersByHealth *labelledGauge
	drainingWorkers gauge

	queueWait    *histogram
	timeToFirst  *histogram
	totalLatency *histogram
	batchSize    *histogram
	tokensPerSec *histogram
}

// NewMetrics builds the metric set. namespace prefixes every series; empty
// defaults to "loom".
func NewMetrics(namespace string) *Metrics {
	if namespace == "" {
		namespace = "loom"
	}
	reasons := []string{
		RejectQueueFull.String(), RejectTenantQuota.String(), RejectTooLarge.String(),
		RejectDeadlineInfeasible.String(), RejectNoCapacity.String(), RejectModelUnavailable.String(),
		RejectNone.String(),
	}
	actions := []string{RetryElsewhere.String(), RetrySame.String(), DoNotRetry.String()}
	states := []string{HealthHealthy.String(), HealthSuspect.String(), HealthDead.String()}

	return &Metrics{
		namespace:       namespace,
		rejected:        newLabelledCounter("reason", reasons),
		retries:         newLabelledCounter("action", actions),
		workersByHealth: newLabelledGauge("state", states),
		queueWait:       newHistogram(latencyBucketsSeconds),
		timeToFirst:     newHistogram(latencyBucketsSeconds),
		totalLatency:    newHistogram(latencyBucketsSeconds),
		batchSize:       newHistogram(batchSizeBuckets),
		tokensPerSec:    newHistogram(tokensPerSecBucks),
	}
}

// Counter mutators.

func (m *Metrics) RequestArrived()   { m.arrived.inc() }
func (m *Metrics) RequestAccepted()  { m.accepted.inc() }
func (m *Metrics) RequestCompleted() { m.completed.inc() }
func (m *Metrics) RequestFailed()    { m.failed.inc() }
func (m *Metrics) RequestCancelled() { m.cancelled.inc() }
func (m *Metrics) RequestPreempted() { m.preempted.inc() }
func (m *Metrics) RequestTimedOut()  { m.timedOut.inc() }

// RequestRejected records a refusal under its reason, so the graphs can tell
// overload apart from an impossible deadline apart from a request too large for
// any worker. A single rejected counter makes those three indistinguishable, and
// they call for three different responses.
func (m *Metrics) RequestRejected(r RejectReason) { m.rejected.inc(r.String()) }

// Retry records a classifier verdict, including the do-not-retry verdicts: the
// ratio of abandoned to retried failures is how a poison-request problem becomes
// visible.
func (m *Metrics) Retry(a RetryAction) { m.retries.inc(a.String()) }

// WorkerFailed counts one worker loss.
func (m *Metrics) WorkerFailed() { m.workerFailures.inc() }

// WorkerReaped counts one reaped worker and the requests taken off it. The
// request count is tracked separately from requests_failed_total because a reaped
// request is usually retried rather than failed, and conflating the two makes a
// single worker loss look like a burst of client-visible errors.
func (m *Metrics) WorkerReaped(requests int) {
	m.reaped.inc()
	if requests > 0 {
		m.reapedRequests.add(uint64(requests))
	}
}

// Gauge setters.

func (m *Metrics) SetQueueDepth(depth, capacity int) {
	m.queueDepth.set(int64(depth))
	m.queueCapacity.set(int64(capacity))
}

func (m *Metrics) SetActiveSequences(n int) { m.activeSequences.set(int64(n)) }

func (m *Metrics) SetCacheBlocks(used, total int) {
	m.cacheBlocksUsed.set(int64(used))
	m.cacheBlocksTotal.set(int64(total))
}

// SetCacheDrift records estimated minus actual blocks across the fleet.
func (m *Metrics) SetCacheDrift(blocks int) { m.cacheDrift.set(int64(blocks)) }

// SetWorkersByHealth publishes the registry's health census.
func (m *Metrics) SetWorkersByHealth(counts map[HealthState]int) {
	for _, s := range []HealthState{HealthHealthy, HealthSuspect, HealthDead} {
		m.workersByHealth.set(s.String(), int64(counts[s]))
	}
}

func (m *Metrics) SetDrainingWorkers(n int) { m.drainingWorkers.set(int64(n)) }

// Histogram observations.

func (m *Metrics) ObserveQueueWait(d time.Duration)        { m.queueWait.observe(d.Seconds()) }
func (m *Metrics) ObserveTimeToFirstToken(d time.Duration) { m.timeToFirst.observe(d.Seconds()) }
func (m *Metrics) ObserveTotalLatency(d time.Duration)     { m.totalLatency.observe(d.Seconds()) }
func (m *Metrics) ObserveBatchSize(n int)                  { m.batchSize.observe(float64(n)) }
func (m *Metrics) ObserveTokensPerSecond(tps float64)      { m.tokensPerSec.observe(tps) }

// Handler serves the exposition format at whatever path it is mounted on.
func (m *Metrics) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		if r.Method == http.MethodHead {
			return
		}
		// Buffered: the scrape is one write to the socket rather than one per
		// series, and a slow scraper cannot interleave with another goroutine's
		// observations in a way that produces a torn line.
		bw := bufio.NewWriter(w)
		_ = m.Export(bw)
		_ = bw.Flush()
	})
}

// ServeMux returns a mux with the handler mounted at /metrics, for callers that
// want a standalone metrics listener.
func (m *Metrics) ServeMux() *http.ServeMux {
	mux := http.NewServeMux()
	mux.Handle("/metrics", m.Handler())
	return mux
}

// Export renders the Prometheus text exposition format.
//
// Series are emitted in a fixed order with every series of a family contiguous
// and preceded by its HELP and TYPE, which the format requires and which also
// makes the output diffable between scrapes while debugging.
func (m *Metrics) Export(w io.Writer) error {
	var b strings.Builder
	b.Grow(8 << 10)

	m.writeCounter(&b, "requests_arrived_total", "Requests received by the gateway.", m.arrived.get())
	m.writeCounter(&b, "requests_accepted_total", "Requests admitted for service.", m.accepted.get())
	m.writeLabelled(&b, "requests_rejected_total", "Requests refused, by reason.", m.rejected)
	m.writeCounter(&b, "requests_completed_total", "Requests that produced a full result.", m.completed.get())
	m.writeCounter(&b, "requests_failed_total", "Requests that ended in an error.", m.failed.get())
	m.writeCounter(&b, "requests_cancelled_total", "Requests abandoned by the client.", m.cancelled.get())
	m.writeCounter(&b, "requests_preempted_total", "Running sequences evicted to free cache blocks.", m.preempted.get())
	m.writeCounter(&b, "requests_timed_out_total", "Requests dropped because their deadline passed.", m.timedOut.get())
	m.writeCounter(&b, "worker_failures_total", "Workers lost.", m.workerFailures.get())
	m.writeCounter(&b, "workers_reaped_total", "Workers declared dead and reaped.", m.reaped.get())
	m.writeCounter(&b, "reaped_requests_total", "Requests recovered from dead workers.", m.reapedRequests.get())
	m.writeLabelled(&b, "retries_total", "Retry classifier verdicts, by action.", m.retries)

	m.writeGauge(&b, "queue_depth", "Requests currently queued.", m.queueDepth.get())
	m.writeGauge(&b, "queue_capacity", "Queue bound.", m.queueCapacity.get())
	m.writeGauge(&b, "active_sequences", "Sequences currently running on workers.", m.activeSequences.get())
	m.writeLabelledGauge(&b, "workers", "Workers by health state.", m.workersByHealth)
	m.writeGauge(&b, "workers_draining", "Workers finishing in-flight work and taking none.", m.drainingWorkers.get())
	m.writeGauge(&b, "cache_blocks_used", "KV-cache blocks allocated.", m.cacheBlocksUsed.get())
	m.writeGauge(&b, "cache_blocks_total", "KV-cache blocks in the fleet.", m.cacheBlocksTotal.get())
	m.writeGauge(&b, "cache_blocks_drift", "Estimated minus actual blocks; positive wastes capacity, negative risks overcommit.", m.cacheDrift.get())

	m.writeHistogram(&b, "queue_wait_seconds", "Time between admission and dispatch.", m.queueWait)
	m.writeHistogram(&b, "time_to_first_token_seconds", "Time between arrival and the first generated token.", m.timeToFirst)
	m.writeHistogram(&b, "request_latency_seconds", "Total time from arrival to completion.", m.totalLatency)
	m.writeHistogram(&b, "batch_size", "Sequences per forward pass.", m.batchSize)
	m.writeHistogram(&b, "tokens_per_second", "Per-worker decode throughput samples.", m.tokensPerSec)

	_, err := io.WriteString(w, b.String())
	return err
}

func (m *Metrics) name(suffix string) string { return m.namespace + "_" + suffix }

func (m *Metrics) writeHeader(b *strings.Builder, name, help, typ string) {
	b.WriteString("# HELP ")
	b.WriteString(name)
	b.WriteByte(' ')
	b.WriteString(escapeHelp(help))
	b.WriteByte('\n')
	b.WriteString("# TYPE ")
	b.WriteString(name)
	b.WriteByte(' ')
	b.WriteString(typ)
	b.WriteByte('\n')
}

func (m *Metrics) writeCounter(b *strings.Builder, suffix, help string, v uint64) {
	n := m.name(suffix)
	m.writeHeader(b, n, help, "counter")
	b.WriteString(n)
	b.WriteByte(' ')
	b.WriteString(strconv.FormatUint(v, 10))
	b.WriteByte('\n')
}

func (m *Metrics) writeGauge(b *strings.Builder, suffix, help string, v int64) {
	n := m.name(suffix)
	m.writeHeader(b, n, help, "gauge")
	b.WriteString(n)
	b.WriteByte(' ')
	b.WriteString(strconv.FormatInt(v, 10))
	b.WriteByte('\n')
}

func (m *Metrics) writeLabelled(b *strings.Builder, suffix, help string, l *labelled) {
	n := m.name(suffix)
	m.writeHeader(b, n, help, "counter")
	for _, k := range l.keys {
		b.WriteString(n)
		b.WriteString("{")
		b.WriteString(l.label)
		b.WriteString(`="`)
		b.WriteString(escapeLabel(k))
		b.WriteString(`"} `)
		b.WriteString(strconv.FormatUint(l.kids[k].get(), 10))
		b.WriteByte('\n')
	}
}

func (m *Metrics) writeLabelledGauge(b *strings.Builder, suffix, help string, l *labelledGauge) {
	n := m.name(suffix)
	m.writeHeader(b, n, help, "gauge")
	for _, k := range l.keys {
		b.WriteString(n)
		b.WriteString("{")
		b.WriteString(l.label)
		b.WriteString(`="`)
		b.WriteString(escapeLabel(k))
		b.WriteString(`"} `)
		b.WriteString(strconv.FormatInt(l.kids[k].get(), 10))
		b.WriteByte('\n')
	}
}

func (m *Metrics) writeHistogram(b *strings.Builder, suffix, help string, h *histogram) {
	n := m.name(suffix)
	m.writeHeader(b, n, help, "histogram")
	cum, sum, count := h.snapshot()
	for i, bound := range h.bounds {
		b.WriteString(n)
		b.WriteString(`_bucket{le="`)
		b.WriteString(formatFloat(bound))
		b.WriteString(`"} `)
		b.WriteString(strconv.FormatUint(cum[i], 10))
		b.WriteByte('\n')
	}
	// The +Inf bucket is mandatory and must equal _count; without it a scraper
	// cannot compute the fraction of observations above the last finite bound,
	// which is precisely the tail this histogram exists to show.
	b.WriteString(n)
	b.WriteString(`_bucket{le="+Inf"} `)
	b.WriteString(strconv.FormatUint(count, 10))
	b.WriteByte('\n')
	b.WriteString(n)
	b.WriteString("_sum ")
	b.WriteString(formatFloat(sum))
	b.WriteByte('\n')
	b.WriteString(n)
	b.WriteString("_count ")
	b.WriteString(strconv.FormatUint(count, 10))
	b.WriteByte('\n')
}

func formatFloat(f float64) string {
	if math.IsInf(f, 1) {
		return "+Inf"
	}
	if math.IsInf(f, -1) {
		return "-Inf"
	}
	return strconv.FormatFloat(f, 'g', -1, 64)
}

// escapeHelp escapes the two characters that would otherwise end or continue a
// HELP line in the wrong place.
func escapeHelp(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	return strings.ReplaceAll(s, "\n", `\n`)
}

func escapeLabel(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `"`, `\"`)
	return strings.ReplaceAll(s, "\n", `\n`)
}
