package loom

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

type gwHarness struct {
	*harness
	gw  *Gateway
	srv *httptest.Server
}

func newGatewayHarness(t *testing.T, opts harnessOpts, tune func(*GatewayConfig)) *gwHarness {
	t.Helper()
	h := newHarness(t, opts)
	cfg := GatewayConfig{
		Scheduler: h.sched, Registry: h.reg, Metrics: h.met,
		DefaultModel: "m", MaxBodyBytes: 512,
		// Short on purpose: several tests deliberately leave a request in flight, and a
		// buffered handler waits for its request to finish.
		RequestTimeout: 2 * time.Second,
	}
	if tune != nil {
		tune(&cfg)
	}
	gw := NewGateway(cfg)
	srv := httptest.NewServer(gw.Handler())
	t.Cleanup(srv.Close)
	// Registered after srv.Close so it runs before it: the scheduler finishing its
	// requests is what lets the open handlers return, and closing the test server
	// first would wait on handlers nothing has released.
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = h.sched.Shutdown(ctx)
	})
	return &gwHarness{harness: h, gw: gw, srv: srv}
}

func (g *gwHarness) post(t *testing.T, path, tenant, body string, header map[string]string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, g.srv.URL+path, strings.NewReader(body))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if tenant != "" {
		req.Header.Set("X-Loom-Tenant", tenant)
	}
	for k, v := range header {
		req.Header.Set(k, v)
	}
	resp, err := g.srv.Client().Do(req)
	if err != nil {
		t.Fatalf("post %s: %v", path, err)
	}
	return resp
}

func (g *gwHarness) get(t *testing.T, path string) *http.Response {
	t.Helper()
	resp, err := g.srv.Client().Get(g.srv.URL + path)
	if err != nil {
		t.Fatalf("get %s: %v", path, err)
	}
	return resp
}

// decodeRefusal reads the body of a refusal and insists it names a reason. The
// whole purpose of RejectReason is that an operator can tell overload from a bad
// request, and a status code alone cannot.
func decodeRefusal(t *testing.T, resp *http.Response) errorBody {
	t.Helper()
	defer resp.Body.Close()
	var eb errorBody
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if err := json.Unmarshal(body, &eb); err != nil {
		t.Fatalf("refusal body %q is not JSON: %v", body, err)
	}
	if eb.Reason == "" {
		t.Fatalf("refusal body %q names no reason", body)
	}
	return eb
}

func TestGatewayRejectsMalformedBodyWith400(t *testing.T) {
	t.Parallel()
	g := newGatewayHarness(t, harnessOpts{}, nil)
	g.addWorker("w-1", "m", 64, 4)

	resp := g.post(t, "/v1/generate", "t-1", `{"prompt": "hi", `, nil)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status %d, want 400", resp.StatusCode)
	}
	if eb := decodeRefusal(t, resp); eb.Reason != "malformed_json" {
		t.Fatalf("reason %q", eb.Reason)
	}
}

func TestGatewayRejectsInvalidFieldsWith400(t *testing.T) {
	t.Parallel()
	g := newGatewayHarness(t, harnessOpts{}, nil)
	g.addWorker("w-1", "m", 64, 4)

	for name, body := range map[string]string{
		"no prompt":      `{"max_output_tokens": 16}`,
		"no output cap":  `{"prompt": "hello"}`,
		"negative wait":  `{"prompt": "hello", "max_output_tokens": 8, "deadline_ms": -1}`,
		"absurd request": `{"prompt": "hello", "max_output_tokens": 1000000000}`,
	} {
		resp := g.post(t, "/v1/generate", "t-1", body, nil)
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("%s: status %d, want 400", name, resp.StatusCode)
		}
		if eb := decodeRefusal(t, resp); eb.Message == "" {
			t.Fatalf("%s: refusal did not say what was wrong", name)
		}
	}
}

func TestGatewayRequiresTenantIdentification(t *testing.T) {
	t.Parallel()
	g := newGatewayHarness(t, harnessOpts{}, nil)
	g.addWorker("w-1", "m", 64, 4)

	resp := g.post(t, "/v1/generate", "", `{"prompt":"hi","max_output_tokens":8}`, nil)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status %d, want 400", resp.StatusCode)
	}
	if eb := decodeRefusal(t, resp); eb.Reason != "missing_tenant" {
		t.Fatalf("reason %q", eb.Reason)
	}
}

func TestGatewayRejectsOversizedBody(t *testing.T) {
	t.Parallel()
	g := newGatewayHarness(t, harnessOpts{}, nil)
	g.addWorker("w-1", "m", 64, 4)

	// MaxBodyBytes is 512 in the harness; this is comfortably past it.
	body := `{"prompt":"` + strings.Repeat("x", 4096) + `","max_output_tokens":8}`
	resp := g.post(t, "/v1/generate", "t-1", body, nil)
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("status %d, want 413", resp.StatusCode)
	}
	if eb := decodeRefusal(t, resp); eb.Reason != "body_too_large" {
		t.Fatalf("reason %q", eb.Reason)
	}
}

func TestGatewayRejectsQueueFullWith429(t *testing.T) {
	t.Parallel()
	// A worker that is registered but has no scheduler slot: healthy from admission's
	// point of view, and nothing drains the queue behind it.
	g := newGatewayHarness(t, harnessOpts{queueCapacity: 1}, nil)
	if err := g.reg.Register(&WorkerState{ID: "w-1", Model: "m", TotalCacheBlocks: 64,
		MaxBatchSize: 1, LastHeartbeat: time.Now()}); err != nil {
		t.Fatalf("register: %v", err)
	}

	body := `{"prompt":"` + strings.Repeat("a", 64) + `","max_output_tokens":256,"stream":false}`
	// In the background: a buffered handler does not return until its request
	// finishes, and this one is meant to sit in the queue.
	go func() {
		resp := g.post(t, "/v1/generate", "t-1", body, nil)
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}()

	waitFor(t, 2*time.Second, "queue to fill", func() bool { return g.queue.Len() == 1 })

	second := g.post(t, "/v1/generate", "t-1", body, nil)
	if second.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("status %d, want 429", second.StatusCode)
	}
	eb := decodeRefusal(t, second)
	if eb.Reason != RejectQueueFull.String() {
		t.Fatalf("reason %q, want %s", eb.Reason, RejectQueueFull)
	}
	if eb.RetryAfterMS <= 0 {
		t.Fatal("a queue-full refusal should tell the client it is worth retrying")
	}
}

func TestGatewayRejectsTenantQuotaWith429(t *testing.T) {
	t.Parallel()
	g := newGatewayHarness(t, harnessOpts{
		quotas: map[TenantID]TenantQuota{"t-1": {MaxConcurrentRequests: 1}},
	}, nil)
	fc := g.addWorker("w-1", "m", 64, 4)

	body := `{"id":"r-1","prompt":"hello there","max_output_tokens":32}`
	go func() {
		resp := g.post(t, "/v1/generate", "t-1", body, nil)
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}()
	waitFor(t, 2*time.Second, "first dispatched", func() bool { return fc.sawDispatch("r-1") })

	second := g.post(t, "/v1/generate", "t-1", `{"id":"r-2","prompt":"hello","max_output_tokens":32}`, nil)
	if second.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("status %d, want 429", second.StatusCode)
	}
	if eb := decodeRefusal(t, second); eb.Reason != RejectTenantQuota.String() {
		t.Fatalf("reason %q, want %s", eb.Reason, RejectTenantQuota)
	}
}

func TestGatewayRejectsUnknownModelWith503(t *testing.T) {
	t.Parallel()
	g := newGatewayHarness(t, harnessOpts{}, nil)
	g.addWorker("w-1", "m", 64, 4)

	resp := g.post(t, "/v1/generate", "t-1",
		`{"model":"some-other-model","prompt":"hi","max_output_tokens":8}`, nil)
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status %d, want 503", resp.StatusCode)
	}
	if eb := decodeRefusal(t, resp); eb.Reason != RejectModelUnavailable.String() {
		t.Fatalf("reason %q, want %s", eb.Reason, RejectModelUnavailable)
	}
}

func TestGatewayRejectsNoCapacityWith503(t *testing.T) {
	t.Parallel()
	g := newGatewayHarness(t, harnessOpts{}, nil)
	g.addWorker("w-1", "m", 64, 4)

	// Draining: the fleet contains a worker big enough, but nothing is accepting
	// work, so queueing would be waiting for an event that is not pending.
	if err := g.reg.SetDraining("w-1", true); err != nil {
		t.Fatalf("set draining: %v", err)
	}

	resp := g.post(t, "/v1/generate", "t-1", `{"prompt":"hi","max_output_tokens":8}`, nil)
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status %d, want 503", resp.StatusCode)
	}
	if eb := decodeRefusal(t, resp); eb.Reason != RejectNoCapacity.String() {
		t.Fatalf("reason %q, want %s", eb.Reason, RejectNoCapacity)
	}
}

func TestGatewayRejectsInfeasibleDeadlineWith504(t *testing.T) {
	t.Parallel()
	g := newGatewayHarness(t, harnessOpts{}, nil)
	g.addWorker("w-1", "m", 64, 4)

	// 512 output tokens cannot be generated in one millisecond on any worker, so
	// serving it would burn capacity for a result that arrives too late.
	resp := g.post(t, "/v1/generate", "t-1",
		`{"prompt":"hi","max_output_tokens":512,"deadline_ms":1}`, nil)
	if resp.StatusCode != http.StatusGatewayTimeout {
		t.Fatalf("status %d, want 504", resp.StatusCode)
	}
	if eb := decodeRefusal(t, resp); eb.Reason != RejectDeadlineInfeasible.String() {
		t.Fatalf("reason %q, want %s", eb.Reason, RejectDeadlineInfeasible)
	}
}

func TestGatewayBufferedResponseCarriesTheResult(t *testing.T) {
	t.Parallel()
	g := newGatewayHarness(t, harnessOpts{}, nil)
	fc := g.addWorker("w-1", "m", 64, 4)

	type reply struct {
		resp *http.Response
		body []byte
	}
	got := make(chan reply, 1)
	go func() {
		resp := g.post(t, "/v1/generate", "t-1",
			`{"id":"r-buf","prompt":"tell me a story","max_output_tokens":16}`, nil)
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		got <- reply{resp, b}
	}()

	waitFor(t, 2*time.Second, "dispatch", func() bool { return fc.sawDispatch("r-buf") })
	fc.emit(WorkerEvent{Kind: EventFirstToken, Request: "r-buf", Tokens: 1})
	fc.emit(WorkerEvent{Kind: EventToken, Request: "r-buf", Tokens: 1, Blocks: 2, Err: "once"})
	fc.emit(WorkerEvent{Kind: EventToken, Request: "r-buf", Tokens: 2, Blocks: 2, Err: " upon"})
	fc.emit(WorkerEvent{Kind: EventCompleted, Request: "r-buf", Tokens: 2, Err: "eos"})

	select {
	case r := <-got:
		if r.resp.StatusCode != http.StatusOK {
			t.Fatalf("status %d: %s", r.resp.StatusCode, r.body)
		}
		var out generateResponse
		if err := json.Unmarshal(r.body, &out); err != nil {
			t.Fatalf("body %q: %v", r.body, err)
		}
		if out.Text != "once upon" {
			t.Fatalf("text %q", out.Text)
		}
		if out.Status != StatusCompleted.String() || out.StopReason != "eos" {
			t.Fatalf("result %+v", out)
		}
		if out.TTFTMillis < 0 {
			t.Fatalf("ttft %d", out.TTFTMillis)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("buffered response never arrived")
	}
}

// TestGatewayStreamsFirstChunkBeforeLast is the test that makes
// time-to-first-token observable to a client rather than only to the server's
// histogram: the final token is not even emitted until the first frame has been
// read, so a server that buffered the response could not pass.
func TestGatewayStreamsFirstChunkBeforeLast(t *testing.T) {
	t.Parallel()
	g := newGatewayHarness(t, harnessOpts{}, nil)
	fc := g.addWorker("w-1", "m", 64, 4)

	req, err := http.NewRequest(http.MethodPost, g.srv.URL+"/v1/generate",
		strings.NewReader(`{"id":"r-stream","prompt":"stream please","max_output_tokens":16,"stream":true}`))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("X-Loom-Tenant", "t-1")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := g.srv.Client().Do(req)
	if err != nil {
		t.Fatalf("do: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content type %q", ct)
	}

	waitFor(t, 2*time.Second, "dispatch", func() bool { return fc.sawDispatch("r-stream") })

	frames := make(chan streamChunk, 8)
	readErr := make(chan error, 1)
	go func() {
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			line := bytes.TrimPrefix(sc.Bytes(), []byte("data: "))
			if len(line) == 0 {
				continue
			}
			var c streamChunk
			if err := json.Unmarshal(line, &c); err != nil {
				readErr <- err
				return
			}
			frames <- c
		}
		readErr <- sc.Err()
	}()

	fc.emit(WorkerEvent{Kind: EventFirstToken, Request: "r-stream", Tokens: 1})

	var first streamChunk
	select {
	case first = <-frames:
	case err := <-readErr:
		t.Fatalf("stream read: %v", err)
	case <-time.After(2 * time.Second):
		t.Fatal("the first frame never arrived, so the response was buffered")
	}
	if first.Type != "first_token" {
		t.Fatalf("first frame %+v", first)
	}

	// Only now does the rest of the generation happen. The first frame is already in
	// the client's hands.
	fc.emit(WorkerEvent{Kind: EventToken, Request: "r-stream", Tokens: 1, Blocks: 2, Err: "a"})
	fc.emit(WorkerEvent{Kind: EventToken, Request: "r-stream", Tokens: 2, Blocks: 2, Err: "b"})
	fc.emit(WorkerEvent{Kind: EventCompleted, Request: "r-stream", Tokens: 2, Err: "eos"})

	var text strings.Builder
	var done *streamChunk
	for done == nil {
		select {
		case c := <-frames:
			switch c.Type {
			case "token":
				text.WriteString(c.Text)
			case "done":
				cc := c
				done = &cc
			}
		case <-time.After(3 * time.Second):
			t.Fatalf("stream never terminated; got %q so far", text.String())
		}
	}
	if text.String() != "ab" {
		t.Fatalf("streamed text %q", text.String())
	}
	if done.Status != StatusCompleted.String() || done.StopReason != "eos" {
		t.Fatalf("terminal frame %+v", done)
	}
}

func TestGatewayStatusAndCancelEndpoints(t *testing.T) {
	t.Parallel()
	g := newGatewayHarness(t, harnessOpts{}, nil)
	fc := g.addWorker("w-1", "m", 64, 4)

	go func() {
		resp := g.post(t, "/v1/generate", "t-1",
			`{"id":"r-cancel","prompt":"a long one","max_output_tokens":512}`, nil)
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}()
	waitFor(t, 2*time.Second, "dispatch", func() bool { return fc.sawDispatch("r-cancel") })

	resp := g.get(t, "/v1/requests/r-cancel")
	var st statusResponse
	if err := json.NewDecoder(resp.Body).Decode(&st); err != nil {
		t.Fatalf("status body: %v", err)
	}
	resp.Body.Close()
	if st.Status != StatusRunning.String() || st.Worker != "w-1" {
		t.Fatalf("status %+v", st)
	}

	if missing := g.get(t, "/v1/requests/nope"); missing.StatusCode != http.StatusNotFound {
		t.Fatalf("unknown request status %d, want 404", missing.StatusCode)
	} else {
		missing.Body.Close()
	}

	cancel := g.post(t, "/v1/generate/r-cancel/cancel", "t-1", "", nil)
	if cancel.StatusCode != http.StatusAccepted {
		t.Fatalf("cancel status %d, want 202", cancel.StatusCode)
	}
	cancel.Body.Close()

	waitFor(t, 2*time.Second, "cancellation to land", func() bool {
		h, ok := g.sched.Lookup("r-cancel")
		return ok && h.Result().Status == StatusCancelled
	})
	waitFor(t, 2*time.Second, "blocks released", func() bool { return g.cache("w-1").Used() == 0 })
}

func TestGatewayReadyzReflectsWorkerAvailability(t *testing.T) {
	t.Parallel()
	g := newGatewayHarness(t, harnessOpts{}, nil)

	resp := g.get(t, "/readyz")
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("readyz with no workers: status %d, want 503", resp.StatusCode)
	}
	var body map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&body)
	resp.Body.Close()
	if body["ready"] != false || body["reason"] != RejectModelUnavailable.String() {
		t.Fatalf("readyz body %v", body)
	}

	// Liveness is a different question: a control plane with no workers is alive and
	// must not be restarted for it, because restarting discards the queue.
	if h := g.get(t, "/healthz"); h.StatusCode != http.StatusOK {
		t.Fatalf("healthz %d", h.StatusCode)
	} else {
		h.Body.Close()
	}

	g.addWorker("w-1", "m", 64, 4)

	resp = g.get(t, "/readyz")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("readyz with a healthy worker: status %d, want 200", resp.StatusCode)
	}
	body = nil
	_ = json.NewDecoder(resp.Body).Decode(&body)
	resp.Body.Close()
	if body["ready"] != true {
		t.Fatalf("readyz body %v", body)
	}

	// A different model has no worker at all, which is a per-model question.
	notReady := g.get(t, "/readyz?model=other")
	if notReady.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("readyz for an unserved model: status %d, want 503", notReady.StatusCode)
	}
	notReady.Body.Close()
}

func TestGatewayServesMetrics(t *testing.T) {
	t.Parallel()
	g := newGatewayHarness(t, harnessOpts{}, nil)
	g.addWorker("w-1", "m", 64, 4)

	resp := g.post(t, "/v1/generate", "t-1", `{"prompt":"hi"}`, nil)
	resp.Body.Close()

	m := g.get(t, "/metrics")
	defer m.Body.Close()
	body, err := io.ReadAll(m.Body)
	if err != nil {
		t.Fatalf("read metrics: %v", err)
	}
	if m.StatusCode != http.StatusOK {
		t.Fatalf("metrics status %d", m.StatusCode)
	}
	if !strings.Contains(string(body), "loom_requests_arrived_total") {
		t.Fatalf("metrics output missing the arrival counter:\n%s", body)
	}
}
