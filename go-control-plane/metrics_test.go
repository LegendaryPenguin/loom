package loom

import (
	"math"
	"net/http"
	"net/http/httptest"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// promSeries is one parsed sample.
type promSeries struct {
	name   string
	labels map[string]string
	value  float64
}

// promFamily is a metric family as the parser reconstructed it.
type promFamily struct {
	name   string
	typ    string
	help   string
	series []promSeries
}

// parsePrometheus is a deliberately strict parser for the text exposition format.
//
// Strict on purpose: the point of asserting on parsed output rather than on
// substrings is that a malformed line — a missing TYPE, a label without quotes, a
// value that is not a float — fails here rather than silently producing a series
// no scraper will accept. A substring assertion passes on output Prometheus would
// reject.
func parsePrometheus(t *testing.T, text string) map[string]*promFamily {
	t.Helper()

	families := make(map[string]*promFamily)
	// order records the family each metric name belongs to, so a series whose name
	// is a histogram suffix is attributed to the right family.
	seenHelp := make(map[string]bool)
	seenType := make(map[string]bool)

	for i, line := range strings.Split(strings.TrimRight(text, "\n"), "\n") {
		ln := i + 1
		if line == "" {
			t.Fatalf("line %d: blank line in exposition output", ln)
		}
		if strings.HasPrefix(line, "#") {
			parts := strings.SplitN(line, " ", 4)
			if len(parts) < 3 {
				t.Fatalf("line %d: malformed comment %q", ln, line)
			}
			switch parts[1] {
			case "HELP":
				name := parts[2]
				if seenHelp[name] {
					t.Fatalf("line %d: duplicate HELP for %s", ln, name)
				}
				seenHelp[name] = true
				if len(parts) < 4 || parts[3] == "" {
					t.Fatalf("line %d: HELP for %s has no text", ln, name)
				}
				fam(families, name).help = parts[3]
			case "TYPE":
				name := parts[2]
				if seenType[name] {
					t.Fatalf("line %d: duplicate TYPE for %s", ln, name)
				}
				seenType[name] = true
				if len(parts) < 4 {
					t.Fatalf("line %d: TYPE for %s has no kind", ln, name)
				}
				switch parts[3] {
				case "counter", "gauge", "histogram", "summary", "untyped":
				default:
					t.Fatalf("line %d: unknown metric type %q", ln, parts[3])
				}
				if !seenHelp[name] {
					t.Fatalf("line %d: TYPE for %s precedes its HELP", ln, name)
				}
				fam(families, name).typ = parts[3]
			default:
				t.Fatalf("line %d: unexpected comment keyword %q", ln, parts[1])
			}
			continue
		}

		sp := strings.LastIndex(line, " ")
		if sp < 0 {
			t.Fatalf("line %d: sample %q has no value", ln, line)
		}
		head, rawVal := line[:sp], line[sp+1:]
		v, err := strconv.ParseFloat(rawVal, 64)
		if err != nil {
			t.Fatalf("line %d: value %q is not a float: %v", ln, rawVal, err)
		}

		name, labels := parseHead(t, ln, head)
		family := attribute(t, families, name)
		family.series = append(family.series, promSeries{name: name, labels: labels, value: v})
	}

	for name, f := range families {
		if f.typ == "" {
			t.Fatalf("family %s has no TYPE line", name)
		}
		if f.help == "" {
			t.Fatalf("family %s has no HELP line", name)
		}
		if len(f.series) == 0 {
			t.Fatalf("family %s declared but has no series", name)
		}
	}
	return families
}

func fam(m map[string]*promFamily, name string) *promFamily {
	if f, ok := m[name]; ok {
		return f
	}
	f := &promFamily{name: name}
	m[name] = f
	return f
}

// attribute maps a sample name onto its declared family, which for a histogram
// means stripping the _bucket/_sum/_count suffix.
func attribute(t *testing.T, m map[string]*promFamily, name string) *promFamily {
	t.Helper()
	if f, ok := m[name]; ok && f.typ != "" {
		return f
	}
	for _, suffix := range []string{"_bucket", "_sum", "_count"} {
		if strings.HasSuffix(name, suffix) {
			base := strings.TrimSuffix(name, suffix)
			if f, ok := m[base]; ok && f.typ == "histogram" {
				return f
			}
		}
	}
	t.Fatalf("sample %s has no declared family", name)
	return nil
}

func parseHead(t *testing.T, ln int, head string) (string, map[string]string) {
	t.Helper()
	open := strings.IndexByte(head, '{')
	if open < 0 {
		if !validMetricName(head) {
			t.Fatalf("line %d: %q is not a valid metric name", ln, head)
		}
		return head, nil
	}
	name := head[:open]
	if !validMetricName(name) {
		t.Fatalf("line %d: %q is not a valid metric name", ln, name)
	}
	if !strings.HasSuffix(head, "}") {
		t.Fatalf("line %d: unterminated label set in %q", ln, head)
	}
	body := head[open+1 : len(head)-1]
	labels := map[string]string{}
	if body == "" {
		t.Fatalf("line %d: empty label set in %q", ln, head)
	}
	for _, pair := range splitLabels(body) {
		eq := strings.IndexByte(pair, '=')
		if eq < 0 {
			t.Fatalf("line %d: label %q has no value", ln, pair)
		}
		k, v := pair[:eq], pair[eq+1:]
		if !validLabelName(k) {
			t.Fatalf("line %d: %q is not a valid label name", ln, k)
		}
		if len(v) < 2 || v[0] != '"' || v[len(v)-1] != '"' {
			t.Fatalf("line %d: label value %q is not quoted", ln, v)
		}
		labels[k] = v[1 : len(v)-1]
	}
	return name, labels
}

// splitLabels splits on commas that are not inside a quoted value.
func splitLabels(body string) []string {
	var out []string
	inQuote := false
	start := 0
	for i := 0; i < len(body); i++ {
		switch body[i] {
		case '"':
			if i == 0 || body[i-1] != '\\' {
				inQuote = !inQuote
			}
		case ',':
			if !inQuote {
				out = append(out, body[start:i])
				start = i + 1
			}
		}
	}
	return append(out, body[start:])
}

func validMetricName(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		ok := c == '_' || c == ':' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9' && i > 0)
		if !ok {
			return false
		}
	}
	return true
}

func validLabelName(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		ok := c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9' && i > 0)
		if !ok {
			return false
		}
	}
	return true
}

func (f *promFamily) find(t *testing.T, labels map[string]string) promSeries {
	t.Helper()
	for _, s := range f.series {
		match := len(s.labels) == len(labels)
		if match {
			for k, v := range labels {
				if s.labels[k] != v {
					match = false
					break
				}
			}
		}
		if match {
			return s
		}
	}
	t.Fatalf("family %s has no series with labels %v (have %v)", f.name, labels, f.series)
	return promSeries{}
}

func mustFamily(t *testing.T, fams map[string]*promFamily, name string) *promFamily {
	t.Helper()
	f, ok := fams[name]
	if !ok {
		keys := make([]string, 0, len(fams))
		for k := range fams {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		t.Fatalf("missing family %s; have %v", name, keys)
	}
	return f
}

func exportText(t *testing.T, m *Metrics) string {
	t.Helper()
	var sb strings.Builder
	if err := m.Export(&sb); err != nil {
		t.Fatalf("Export: %v", err)
	}
	return sb.String()
}

func TestMetricsExportParsesAsPrometheus(t *testing.T) {
	m := NewMetrics("")

	m.RequestArrived()
	m.RequestArrived()
	m.RequestAccepted()
	m.RequestRejected(RejectQueueFull)
	m.RequestRejected(RejectQueueFull)
	m.RequestRejected(RejectDeadlineInfeasible)
	m.RequestCompleted()
	m.RequestFailed()
	m.RequestCancelled()
	m.RequestPreempted()
	m.RequestTimedOut()
	m.WorkerFailed()
	m.WorkerReaped(3)
	m.Retry(RetryElsewhere)
	m.Retry(DoNotRetry)

	m.SetQueueDepth(17, 256)
	m.SetActiveSequences(9)
	m.SetCacheBlocks(912, 2048)
	m.SetCacheDrift(-24)
	m.SetDrainingWorkers(1)
	m.SetWorkersByHealth(map[HealthState]int{HealthHealthy: 4, HealthSuspect: 1, HealthDead: 2})

	m.ObserveQueueWait(120 * time.Millisecond)
	m.ObserveTimeToFirstToken(184 * time.Millisecond)
	m.ObserveTotalLatency(2 * time.Second)
	m.ObserveBatchSize(12)
	m.ObserveTokensPerSecond(143.2)

	fams := parsePrometheus(t, exportText(t, m))

	counters := map[string]float64{
		"loom_requests_arrived_total":   2,
		"loom_requests_accepted_total":  1,
		"loom_requests_completed_total": 1,
		"loom_requests_failed_total":    1,
		"loom_requests_cancelled_total": 1,
		"loom_requests_preempted_total": 1,
		"loom_requests_timed_out_total": 1,
		"loom_worker_failures_total":    1,
		"loom_workers_reaped_total":     1,
		"loom_reaped_requests_total":    3,
	}
	for name, want := range counters {
		f := mustFamily(t, fams, name)
		if f.typ != "counter" {
			t.Fatalf("%s type = %q, want counter", name, f.typ)
		}
		if got := f.find(t, nil).value; got != want {
			t.Fatalf("%s = %v, want %v", name, got, want)
		}
	}

	rej := mustFamily(t, fams, "loom_requests_rejected_total")
	if got := rej.find(t, map[string]string{"reason": "queue_full"}).value; got != 2 {
		t.Fatalf("queue_full rejections = %v, want 2", got)
	}
	if got := rej.find(t, map[string]string{"reason": "deadline_infeasible"}).value; got != 1 {
		t.Fatalf("deadline_infeasible rejections = %v, want 1", got)
	}
	// Every reason is exported even at zero, so a dashboard query does not go
	// blank the first time a new failure mode appears.
	if got := rej.find(t, map[string]string{"reason": "tenant_quota"}).value; got != 0 {
		t.Fatalf("tenant_quota = %v, want 0", got)
	}
	if len(rej.series) != 7 {
		t.Fatalf("rejection series = %d, want 7 (one per reason)", len(rej.series))
	}

	retries := mustFamily(t, fams, "loom_retries_total")
	if got := retries.find(t, map[string]string{"action": "elsewhere"}).value; got != 1 {
		t.Fatalf("retries elsewhere = %v", got)
	}
	if got := retries.find(t, map[string]string{"action": "none"}).value; got != 1 {
		t.Fatalf("abandoned retries = %v", got)
	}

	gauges := map[string]float64{
		"loom_queue_depth":        17,
		"loom_queue_capacity":     256,
		"loom_active_sequences":   9,
		"loom_cache_blocks_used":  912,
		"loom_cache_blocks_total": 2048,
		"loom_cache_blocks_drift": -24,
		"loom_workers_draining":   1,
	}
	for name, want := range gauges {
		f := mustFamily(t, fams, name)
		if f.typ != "gauge" {
			t.Fatalf("%s type = %q, want gauge", name, f.typ)
		}
		if got := f.find(t, nil).value; got != want {
			t.Fatalf("%s = %v, want %v", name, got, want)
		}
	}

	workers := mustFamily(t, fams, "loom_workers")
	for state, want := range map[string]float64{"healthy": 4, "suspect": 1, "dead": 2} {
		if got := workers.find(t, map[string]string{"state": state}).value; got != want {
			t.Fatalf("workers{state=%q} = %v, want %v", state, got, want)
		}
	}

	for _, h := range []string{
		"loom_queue_wait_seconds",
		"loom_time_to_first_token_seconds",
		"loom_request_latency_seconds",
		"loom_batch_size",
		"loom_tokens_per_second",
	} {
		f := mustFamily(t, fams, h)
		if f.typ != "histogram" {
			t.Fatalf("%s type = %q, want histogram", h, f.typ)
		}
		assertHistogramWellFormed(t, f, 1)
	}
}

// assertHistogramWellFormed checks the invariants a scraper relies on: buckets
// are cumulative and non-decreasing, the +Inf bucket equals _count, and _count is
// the number of observations.
func assertHistogramWellFormed(t *testing.T, f *promFamily, wantCount float64) {
	t.Helper()

	type bucket struct {
		le    float64
		value float64
	}
	var buckets []bucket
	var sum, count float64
	var haveSum, haveCount, haveInf bool

	for _, s := range f.series {
		switch {
		case strings.HasSuffix(s.name, "_bucket"):
			le, ok := s.labels["le"]
			if !ok {
				t.Fatalf("%s: bucket series without an le label", f.name)
			}
			if le == "+Inf" {
				haveInf = true
				if s.value != wantCount {
					t.Fatalf("%s: +Inf bucket = %v, want %v (_count)", f.name, s.value, wantCount)
				}
				continue
			}
			v, err := strconv.ParseFloat(le, 64)
			if err != nil {
				t.Fatalf("%s: le=%q is not a float", f.name, le)
			}
			buckets = append(buckets, bucket{v, s.value})
		case strings.HasSuffix(s.name, "_sum"):
			haveSum, sum = true, s.value
		case strings.HasSuffix(s.name, "_count"):
			haveCount, count = true, s.value
		default:
			t.Fatalf("%s: unexpected series %s in a histogram family", f.name, s.name)
		}
	}
	if !haveInf {
		t.Fatalf("%s: no +Inf bucket, so the tail above the last finite bound is unknowable", f.name)
	}
	if !haveSum || !haveCount {
		t.Fatalf("%s: sum=%v count=%v present flags", f.name, haveSum, haveCount)
	}
	if count != wantCount {
		t.Fatalf("%s: _count = %v, want %v", f.name, count, wantCount)
	}
	if len(buckets) == 0 {
		t.Fatalf("%s: no finite buckets", f.name)
	}
	sort.Slice(buckets, func(i, j int) bool { return buckets[i].le < buckets[j].le })
	for i := 1; i < len(buckets); i++ {
		if buckets[i].value < buckets[i-1].value {
			t.Fatalf("%s: buckets are not cumulative at le=%v: %v < %v",
				f.name, buckets[i].le, buckets[i].value, buckets[i-1].value)
		}
	}
	if last := buckets[len(buckets)-1].value; last > count {
		t.Fatalf("%s: last finite bucket %v exceeds _count %v", f.name, last, count)
	}
	if sum < 0 && count > 0 {
		t.Fatalf("%s: negative sum %v over non-negative observations", f.name, sum)
	}
}

func TestMetricsHistogramBucketsAreCumulative(t *testing.T) {
	m := NewMetrics("loom")

	// Observations chosen to land in known buckets, including one above every
	// finite bound so the +Inf bucket is exercised.
	waits := []time.Duration{
		500 * time.Microsecond, // <= 0.001
		3 * time.Millisecond,   // <= 0.005
		3 * time.Millisecond,
		40 * time.Millisecond, // <= 0.05
		700 * time.Millisecond,
		90 * time.Second, // +Inf only
	}
	var total float64
	for _, d := range waits {
		m.ObserveQueueWait(d)
		total += d.Seconds()
	}

	fams := parsePrometheus(t, exportText(t, m))
	f := mustFamily(t, fams, "loom_queue_wait_seconds")
	assertHistogramWellFormed(t, f, float64(len(waits)))

	bucketAt := func(le string) float64 {
		for _, s := range f.series {
			if strings.HasSuffix(s.name, "_bucket") && s.labels["le"] == le {
				return s.value
			}
		}
		t.Fatalf("no bucket le=%s", le)
		return 0
	}
	// Cumulative means each bound counts everything at or below it, not just the
	// observations that fell in that bound's own slot. Getting this wrong makes
	// every quantile computed from the histogram wrong.
	if got := bucketAt("0.001"); got != 1 {
		t.Fatalf("le=0.001 = %v, want 1", got)
	}
	if got := bucketAt("0.005"); got != 3 {
		t.Fatalf("le=0.005 = %v, want 3", got)
	}
	if got := bucketAt("0.05"); got != 4 {
		t.Fatalf("le=0.05 = %v, want 4", got)
	}
	if got := bucketAt("1"); got != 5 {
		t.Fatalf("le=1 = %v, want 5", got)
	}
	if got := bucketAt("60"); got != 5 {
		t.Fatalf("le=60 = %v, want 5 — the 90s observation is above every finite bound", got)
	}
	if got := bucketAt("+Inf"); got != 6 {
		t.Fatalf("le=+Inf = %v, want 6", got)
	}

	sumSeries := f.find(t, nil)
	if !strings.HasSuffix(sumSeries.name, "_sum") && !strings.HasSuffix(sumSeries.name, "_count") {
		t.Fatalf("unlabelled series in a histogram should be _sum or _count, got %s", sumSeries.name)
	}
	var sum float64
	for _, s := range f.series {
		if strings.HasSuffix(s.name, "_sum") {
			sum = s.value
		}
	}
	if math.Abs(sum-total) > 1e-9 {
		t.Fatalf("_sum = %v, want %v", sum, total)
	}

	// An upper bound is inclusive: an observation exactly on a boundary belongs to
	// that bucket, not the next one.
	b := NewMetrics("loom")
	b.ObserveQueueWait(time.Millisecond)
	bf := mustFamily(t, parsePrometheus(t, exportText(t, b)), "loom_queue_wait_seconds")
	for _, s := range bf.series {
		if strings.HasSuffix(s.name, "_bucket") && s.labels["le"] == "0.001" && s.value != 1 {
			t.Fatalf("an observation exactly on a bound landed outside it: %v", s.value)
		}
	}

	// NaN cannot be placed in a bucket and would poison the sum forever.
	c := NewMetrics("loom")
	c.ObserveTokensPerSecond(math.NaN())
	cf := mustFamily(t, parsePrometheus(t, exportText(t, c)), "loom_tokens_per_second")
	assertHistogramWellFormed(t, cf, 0)
}

func TestMetricsCountersOnlyIncrease(t *testing.T) {
	m := NewMetrics("loom")
	names := []string{
		"loom_requests_arrived_total",
		"loom_requests_accepted_total",
		"loom_requests_completed_total",
		"loom_requests_failed_total",
		"loom_requests_cancelled_total",
		"loom_requests_preempted_total",
		"loom_requests_timed_out_total",
		"loom_worker_failures_total",
		"loom_workers_reaped_total",
	}
	prev := map[string]float64{}

	for round := 0; round < 25; round++ {
		m.RequestArrived()
		if round%2 == 0 {
			m.RequestAccepted()
			m.RequestCompleted()
		}
		if round%3 == 0 {
			m.RequestFailed()
			m.RequestRejected(RejectNoCapacity)
		}
		if round%5 == 0 {
			m.RequestCancelled()
			m.RequestPreempted()
			m.RequestTimedOut()
			m.WorkerFailed()
			m.WorkerReaped(2)
		}
		// Gauges move in both directions, which is exactly why they are a separate
		// kind; a counter that did this would break every rate() over it.
		m.SetQueueDepth(25-round, 256)

		fams := parsePrometheus(t, exportText(t, m))
		for _, n := range names {
			got := mustFamily(t, fams, n).find(t, nil).value
			if got < prev[n] {
				t.Fatalf("round %d: %s decreased from %v to %v", round, n, prev[n], got)
			}
			prev[n] = got
		}
		rej := mustFamily(t, fams, "loom_requests_rejected_total").
			find(t, map[string]string{"reason": "no_capacity"}).value
		if rej < prev["rej"] {
			t.Fatalf("round %d: labelled counter decreased from %v to %v", round, prev["rej"], rej)
		}
		prev["rej"] = rej
	}

	if prev["loom_requests_arrived_total"] != 25 {
		t.Fatalf("arrived = %v, want 25", prev["loom_requests_arrived_total"])
	}
}

func TestMetricsHandler(t *testing.T) {
	m := NewMetrics("loom")
	m.RequestArrived()
	m.ObserveTotalLatency(42 * time.Millisecond)

	srv := httptest.NewServer(m.ServeMux())
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatalf("GET /metrics: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Fatalf("Content-Type = %q", ct)
	}
	var sb strings.Builder
	buf := make([]byte, 4096)
	for {
		n, err := resp.Body.Read(buf)
		sb.Write(buf[:n])
		if err != nil {
			break
		}
	}
	fams := parsePrometheus(t, sb.String())
	if got := mustFamily(t, fams, "loom_requests_arrived_total").find(t, nil).value; got != 1 {
		t.Fatalf("served arrived = %v, want 1", got)
	}

	rec := httptest.NewRecorder()
	m.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/metrics", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST /metrics = %d, want 405", rec.Code)
	}

	rec = httptest.NewRecorder()
	m.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodHead, "/metrics", nil))
	if rec.Code != http.StatusOK || rec.Body.Len() != 0 {
		t.Fatalf("HEAD /metrics = %d with %d bytes", rec.Code, rec.Body.Len())
	}
}

func TestMetricsConcurrentObservation(t *testing.T) {
	m := NewMetrics("loom")

	const writers = 8
	const perWriter = 500

	var writersWG sync.WaitGroup
	for i := 0; i < writers; i++ {
		i := i
		writersWG.Add(1)
		go func() {
			defer writersWG.Done()
			for j := 0; j < perWriter; j++ {
				m.RequestArrived()
				m.RequestRejected(RejectQueueFull)
				m.Retry(RetrySame)
				m.ObserveQueueWait(time.Duration(j) * time.Millisecond)
				m.ObserveTimeToFirstToken(time.Duration(j%50) * time.Millisecond)
				m.ObserveBatchSize(j % 33)
				m.ObserveTokensPerSecond(float64(j))
				m.SetQueueDepth(j, 256)
				m.SetActiveSequences(j % 17)
				m.SetCacheDrift(i - j)
				m.SetWorkersByHealth(map[HealthState]int{HealthHealthy: i, HealthSuspect: j % 3})
			}
		}()
	}

	// A scraper runs against the same metrics, because a torn scrape is the race
	// that would actually reach an operator. It is waited for separately: joining it
	// into the writers' group would deadlock, since it only stops once they finish.
	stop := make(chan struct{})
	var scraperWG sync.WaitGroup
	scraperWG.Add(1)
	go func() {
		defer scraperWG.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			var sb strings.Builder
			if err := m.Export(&sb); err != nil {
				t.Errorf("Export: %v", err)
				return
			}
		}
	}()

	writersWG.Wait()
	close(stop)
	scraperWG.Wait()

	fams := parsePrometheus(t, exportText(t, m))
	want := float64(writers * perWriter)
	if got := mustFamily(t, fams, "loom_requests_arrived_total").find(t, nil).value; got != want {
		t.Fatalf("arrived = %v, want %v", got, want)
	}
	if got := mustFamily(t, fams, "loom_requests_rejected_total").
		find(t, map[string]string{"reason": "queue_full"}).value; got != want {
		t.Fatalf("rejected = %v, want %v", got, want)
	}
	assertHistogramWellFormed(t, mustFamily(t, fams, "loom_queue_wait_seconds"), want)
	assertHistogramWellFormed(t, mustFamily(t, fams, "loom_batch_size"), want)
}

func TestMetricsNamespaceAndEscaping(t *testing.T) {
	m := NewMetrics("custom")
	m.RequestArrived()
	text := exportText(t, m)
	if !strings.Contains(text, "custom_requests_arrived_total 1\n") {
		t.Fatalf("namespace not applied:\n%s", text)
	}
	fams := parsePrometheus(t, text)
	if _, ok := fams["loom_requests_arrived_total"]; ok {
		t.Fatal("default namespace leaked into a custom one")
	}

	if got := escapeLabel(`a"b\c`); got != `a\"b\\c` {
		t.Fatalf("escapeLabel = %q", got)
	}
	if got := escapeHelp("one\ntwo"); got != `one\ntwo` {
		t.Fatalf("escapeHelp = %q — an unescaped newline would split the HELP line", got)
	}
	if got := formatFloat(math.Inf(1)); got != "+Inf" {
		t.Fatalf("formatFloat(+Inf) = %q", got)
	}
	if got := formatFloat(math.Inf(-1)); got != "-Inf" {
		t.Fatalf("formatFloat(-Inf) = %q", got)
	}
}

func TestMetricsRegistryIntegration(t *testing.T) {
	// The health census and the reap accounting are the two numbers the registry
	// produces that only exist to be exported, so they are wired here rather than
	// left to a caller that might not.
	clk := newRegClock()
	r := newTestRegistry(clk)
	m := NewMetrics("loom")

	for _, id := range []WorkerID{"w-1", "w-2"} {
		if err := r.Register(regWorker(id, 1024)); err != nil {
			t.Fatal(err)
		}
	}
	if err := r.Assign("r-1", "w-1", 30); err != nil {
		t.Fatal(err)
	}
	clk.advance(3 * regInterval)
	if err := r.Heartbeat("w-2", 0, 0, 0); err != nil {
		t.Fatal(err)
	}

	for _, d := range r.ReapDead(clk.Now(), 0) {
		m.WorkerFailed()
		m.WorkerReaped(len(d.Requests))
		for range d.Requests {
			m.Retry(RetryElsewhere)
		}
	}
	m.SetWorkersByHealth(r.HealthCounts(clk.Now()))

	fams := parsePrometheus(t, exportText(t, m))
	if got := mustFamily(t, fams, "loom_reaped_requests_total").find(t, nil).value; got != 1 {
		t.Fatalf("reaped requests = %v, want 1", got)
	}
	workers := mustFamily(t, fams, "loom_workers")
	if got := workers.find(t, map[string]string{"state": "dead"}).value; got != 1 {
		t.Fatalf("dead workers = %v, want 1", got)
	}
	if got := workers.find(t, map[string]string{"state": "healthy"}).value; got != 1 {
		t.Fatalf("healthy workers = %v, want 1", got)
	}
}
