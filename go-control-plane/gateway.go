package loom

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// GatewayConfig configures the client-facing API.
type GatewayConfig struct {
	Scheduler *Scheduler
	Registry  *Registry
	Metrics   *Metrics
	Clock     Clock

	// MaxBodyBytes bounds a request body. An unbounded body is a trivial way to
	// exhaust the control plane's memory from outside, and the control plane is the
	// one process whose failure takes the whole fleet's work with it.
	MaxBodyBytes int64

	// TenantHeader names the header carrying the tenant. Quota and fairness are both
	// per-tenant, so an unattributed request cannot be accounted for at all.
	TenantHeader string

	// DefaultTenant is used when the header is absent. Empty makes the header
	// mandatory, which is the right default for a multi-tenant deployment and the
	// wrong one for a single-tenant benchmark, so it is configurable rather than
	// decided here.
	DefaultTenant TenantID

	// DefaultModel is served when a request names no model.
	DefaultModel ModelID

	// MaxPromptTokens and MaxOutputTokens bound a single request. Validation at the
	// edge, because a request that cannot be served is cheapest to refuse before it
	// consumes an admission decision.
	MaxPromptTokens int
	MaxOutputTokens int

	// StreamFlushInterval bounds how long a streamed chunk waits for company. Zero
	// flushes every chunk, which is what makes time-to-first-token observable to the
	// client rather than only to the server's histogram.
	StreamFlushInterval time.Duration

	// RequestTimeout bounds a buffered (non-streaming) response. A client that
	// cannot be told "later" must at least be told "too late".
	RequestTimeout time.Duration

	Logf func(format string, args ...any)
}

func (c GatewayConfig) withDefaults() GatewayConfig {
	if c.Clock == nil {
		c.Clock = SystemClock{}
	}
	if c.MaxBodyBytes <= 0 {
		c.MaxBodyBytes = 1 << 20
	}
	if c.TenantHeader == "" {
		c.TenantHeader = "X-Loom-Tenant"
	}
	if c.MaxPromptTokens <= 0 {
		c.MaxPromptTokens = 1 << 20
	}
	if c.MaxOutputTokens <= 0 {
		c.MaxOutputTokens = 1 << 16
	}
	if c.RequestTimeout <= 0 {
		c.RequestTimeout = 5 * time.Minute
	}
	return c
}

// Gateway is the HTTP front end.
type Gateway struct {
	cfg GatewayConfig
	mux *http.ServeMux

	// seq makes generated request ids unique without a clock dependency, since two
	// requests in the same nanosecond are routine under load.
	seq atomic.Uint64
}

// NewGateway builds the gateway and its routes.
func NewGateway(cfg GatewayConfig) *Gateway {
	g := &Gateway{cfg: cfg.withDefaults()}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/generate", g.handleGenerate)
	mux.HandleFunc("POST /v1/generate/{id}/cancel", g.handleCancel)
	mux.HandleFunc("GET /v1/requests/{id}", g.handleStatus)
	mux.HandleFunc("GET /healthz", g.handleHealthz)
	mux.HandleFunc("GET /readyz", g.handleReadyz)
	if g.cfg.Metrics != nil {
		mux.Handle("GET /metrics", g.cfg.Metrics.Handler())
	}
	g.mux = mux
	return g
}

// Handler is the gateway's router.
func (g *Gateway) Handler() http.Handler { return g.mux }

func (g *Gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) { g.mux.ServeHTTP(w, r) }

// generateRequest is the wire shape of a submission.
type generateRequest struct {
	ID     string `json:"id"`
	Model  string `json:"model"`
	Prompt string `json:"prompt"`

	// PromptTokens lets a client that has already tokenized say so. Absent, the
	// gateway estimates, and the estimate is deliberately crude — the number only
	// has to be good enough to size a reservation, and a wrong one shows up as cache
	// drift rather than as a wrong answer.
	PromptTokens    int `json:"prompt_tokens"`
	MaxOutputTokens int `json:"max_output_tokens"`

	Priority int `json:"priority"`

	// DeadlineMS is relative, because an absolute timestamp from a client with a
	// skewed clock is worse than no deadline at all.
	DeadlineMS int `json:"deadline_ms"`

	Stream bool `json:"stream"`
}

// errorBody is what a refusal says.
//
// The reason is a machine-readable field, not only prose: the whole point of
// RejectReason is that an operator — and a client's retry logic — can tell
// overload from a bad request, and a body that only carries an HTTP status has
// thrown that distinction away.
type errorBody struct {
	Error   string `json:"error"`
	Reason  string `json:"reason"`
	Message string `json:"message"`

	// RetryAfterMS is advice, present only on the refusals where waiting could
	// plausibly help.
	RetryAfterMS int `json:"retry_after_ms,omitempty"`
}

func (g *Gateway) handleGenerate(w http.ResponseWriter, r *http.Request) {
	tenant, ok := g.tenantOf(r)
	if !ok {
		g.refuse(w, http.StatusBadRequest, "missing_tenant",
			fmt.Sprintf("header %s is required", g.cfg.TenantHeader), 0)
		return
	}

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, g.cfg.MaxBodyBytes))
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			g.refuse(w, http.StatusRequestEntityTooLarge, "body_too_large",
				fmt.Sprintf("request body exceeds %d bytes", g.cfg.MaxBodyBytes), 0)
			return
		}
		g.refuse(w, http.StatusBadRequest, "unreadable_body", err.Error(), 0)
		return
	}

	var in generateRequest
	if err := json.Unmarshal(body, &in); err != nil {
		g.refuse(w, http.StatusBadRequest, "malformed_json", err.Error(), 0)
		return
	}

	req, msg := g.buildRequest(&in, tenant)
	if req == nil {
		g.refuse(w, http.StatusBadRequest, "invalid_request", msg, 0)
		return
	}

	cancelled := make(chan struct{})
	req.Cancelled = cancelled
	h, dec, reason := g.cfg.Scheduler.Submit(req, in.Prompt)
	if h == nil {
		status, label := statusForReject(reason)
		g.refuse(w, status, label, rejectMessage(reason, dec), retryAfterMS(reason))
		return
	}

	if g.wantsStream(r, &in) {
		g.serveStream(w, r, h, cancelled)
		return
	}
	g.serveBuffered(w, r, h, cancelled)
}

// buildRequest validates and normalizes. It returns nil and a reason on anything
// it will not accept, rather than clamping silently: a client that asked for 10
// million output tokens has a bug, and serving a quietly different request than
// the one asked for is how that bug survives to production.
func (g *Gateway) buildRequest(in *generateRequest, tenant TenantID) (*Request, string) {
	if strings.TrimSpace(in.Prompt) == "" && in.PromptTokens <= 0 {
		return nil, "prompt must not be empty"
	}
	if in.MaxOutputTokens <= 0 {
		return nil, "max_output_tokens must be positive"
	}
	if in.MaxOutputTokens > g.cfg.MaxOutputTokens {
		return nil, fmt.Sprintf("max_output_tokens exceeds limit of %d", g.cfg.MaxOutputTokens)
	}
	promptTokens := in.PromptTokens
	if promptTokens <= 0 {
		promptTokens = estimatePromptTokens(in.Prompt)
	}
	if promptTokens > g.cfg.MaxPromptTokens {
		return nil, fmt.Sprintf("prompt_tokens exceeds limit of %d", g.cfg.MaxPromptTokens)
	}
	if in.DeadlineMS < 0 {
		return nil, "deadline_ms must not be negative"
	}
	model := ModelID(in.Model)
	if model == "" {
		model = g.cfg.DefaultModel
	}
	if model == "" {
		return nil, "model is required"
	}

	now := g.cfg.Clock.Now()
	id := RequestID(in.ID)
	if id == "" {
		id = RequestID(fmt.Sprintf("r-%d-%d", now.UnixNano(), g.seq.Add(1)))
	}
	req := &Request{
		ID: id, Tenant: tenant, Model: model, Priority: in.Priority,
		PromptTokens: promptTokens, MaxOutputTokens: in.MaxOutputTokens,
		ArrivalTime: now,
	}
	if in.DeadlineMS > 0 {
		req.Deadline = now.Add(time.Duration(in.DeadlineMS) * time.Millisecond)
	}
	return req, ""
}

// estimatePromptTokens is a four-bytes-per-token approximation.
//
// The control plane does not have the model's tokenizer and should not: loading
// one per model into the scheduler would make the scheduler depend on the ML
// runtime it exists to be independent of. The error is bounded, it is in the
// reservation rather than in the result, and the heartbeat reconciliation measures
// it — which is a better answer than a dependency.
func estimatePromptTokens(prompt string) int {
	if prompt == "" {
		return 0
	}
	n := (len(prompt) + 3) / 4
	if n < 1 {
		n = 1
	}
	return n
}

func (g *Gateway) wantsStream(r *http.Request, in *generateRequest) bool {
	if in.Stream {
		return true
	}
	accept := r.Header.Get("Accept")
	return strings.Contains(accept, "text/event-stream") || strings.Contains(accept, "application/x-ndjson")
}

// streamChunk is one streamed frame. Shared by both framings so a client choosing
// between them gets the same fields.
type streamChunk struct {
	Type   string `json:"type"`
	Text   string `json:"text,omitempty"`
	Tokens int    `json:"tokens,omitempty"`

	Status     string `json:"status,omitempty"`
	StopReason string `json:"stop_reason,omitempty"`
	Error      string `json:"error,omitempty"`

	// TTFTMillis is sent on the first-token frame so the client can report the
	// metric itself rather than inferring it from its own receive timestamps.
	TTFTMillis int64 `json:"ttft_ms,omitempty"`
}

func (g *Gateway) serveStream(w http.ResponseWriter, r *http.Request, h *Handle, cancelled chan struct{}) {
	sse := strings.Contains(r.Header.Get("Accept"), "text/event-stream")
	if sse {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
	} else {
		w.Header().Set("Content-Type", "application/x-ndjson")
	}
	w.Header().Set("X-Loom-Request-Id", string(h.ID()))
	w.WriteHeader(http.StatusOK)

	flusher, _ := w.(http.Flusher)
	if flusher != nil {
		// The headers go out before the first token exists. Without this the client
		// blocks in its own round trip until the first frame, which makes the accepted
		// request indistinguishable from a slow refusal and hides the very interval
		// streaming exists to expose.
		flusher.Flush()
	}
	start := g.cfg.Clock.Now()
	closeOnce := func() func() {
		done := false
		return func() {
			if !done {
				done = true
				close(cancelled)
			}
		}
	}()
	defer closeOnce()

	write := func(c streamChunk) bool {
		b, err := json.Marshal(c)
		if err != nil {
			return false
		}
		if sse {
			_, err = fmt.Fprintf(w, "data: %s\n\n", b)
		} else {
			_, err = fmt.Fprintf(w, "%s\n", b)
		}
		if err != nil {
			return false
		}
		if flusher != nil {
			// Flushed per frame on purpose. Buffering a streamed response until the
			// handler returns makes the stream indistinguishable from a buffered one, and
			// time-to-first-token unobservable, which is the only reason to stream at all.
			flusher.Flush()
		}
		return true
	}

	for {
		select {
		case c, ok := <-h.Chunks():
			if !ok {
				g.writeTerminal(write, h)
				return
			}
			frame := streamChunk{Type: "token", Text: c.Text, Tokens: c.Tokens}
			if c.First {
				frame.Type = "first_token"
				frame.TTFTMillis = g.cfg.Clock.Now().Sub(start).Milliseconds()
			}
			if !write(frame) {
				// The client is gone. Cancelling rather than finishing the generation is
				// the point: capacity spent on a response nobody will read is capacity
				// taken from a request somebody is waiting for.
				closeOnce()
				return
			}
		case <-r.Context().Done():
			closeOnce()
			return
		}
	}
}

func (g *Gateway) writeTerminal(write func(streamChunk) bool, h *Handle) {
	res := h.Result()
	write(streamChunk{
		Type: "done", Status: res.Status.String(), StopReason: res.StopReason,
		Error: res.Err, Tokens: res.GeneratedTokens,
	})
}

// generateResponse is the buffered reply.
type generateResponse struct {
	ID              string `json:"id"`
	Status          string `json:"status"`
	Text            string `json:"text"`
	GeneratedTokens int    `json:"generated_tokens"`
	StopReason      string `json:"stop_reason,omitempty"`
	Error           string `json:"error,omitempty"`

	QueueWaitMS int64 `json:"queue_wait_ms"`
	TTFTMillis  int64 `json:"ttft_ms,omitempty"`
	TotalMS     int64 `json:"total_ms"`

	// DroppedChunks is non-zero only if the server had to discard output, which a
	// buffered client should never see; reported so that if it ever happens it is not
	// a silent truncation.
	DroppedChunks int `json:"dropped_chunks,omitempty"`
}

func (g *Gateway) serveBuffered(w http.ResponseWriter, r *http.Request, h *Handle, cancelled chan struct{}) {
	var text strings.Builder
	timeout := time.NewTimer(g.cfg.RequestTimeout)
	defer timeout.Stop()

	for collecting := true; collecting; {
		select {
		case c, ok := <-h.Chunks():
			if !ok {
				collecting = false
				break
			}
			text.WriteString(c.Text)
		case <-r.Context().Done():
			close(cancelled)
			// No body: the client has already stopped reading, and the status code is
			// for the logs rather than for it.
			w.WriteHeader(499)
			return
		case <-timeout.C:
			close(cancelled)
			g.refuse(w, http.StatusGatewayTimeout, RejectDeadlineInfeasible.String(),
				"request exceeded the gateway's response timeout", 0)
			return
		}
	}

	res := h.Result()
	status := http.StatusOK
	switch res.Status {
	case StatusTimedOut:
		status = http.StatusGatewayTimeout
	case StatusFailed:
		status = http.StatusInternalServerError
	case StatusCancelled:
		status = 499
	}

	out := generateResponse{
		ID: string(res.ID), Status: res.Status.String(), Text: text.String(),
		GeneratedTokens: res.GeneratedTokens, StopReason: res.StopReason, Error: res.Err,
		DroppedChunks: res.DroppedChunks,
	}
	if !res.StartedAt.IsZero() {
		out.QueueWaitMS = res.StartedAt.Sub(res.ArrivalTime).Milliseconds()
	}
	if !res.FirstTokenAt.IsZero() {
		out.TTFTMillis = res.FirstTokenAt.Sub(res.ArrivalTime).Milliseconds()
	}
	if !res.CompletedAt.IsZero() {
		out.TotalMS = res.CompletedAt.Sub(res.ArrivalTime).Milliseconds()
	}
	writeJSON(w, status, out)
}

func (g *Gateway) handleCancel(w http.ResponseWriter, r *http.Request) {
	id := RequestID(r.PathValue("id"))
	if id == "" {
		g.refuse(w, http.StatusBadRequest, "missing_id", "request id is required", 0)
		return
	}
	if _, ok := g.cfg.Scheduler.Lookup(id); !ok {
		g.refuse(w, http.StatusNotFound, "unknown_request", "no such request", 0)
		return
	}
	g.cfg.Scheduler.Cancel(id)
	writeJSON(w, http.StatusAccepted, map[string]string{"id": string(id), "status": "cancelling"})
}

type statusResponse struct {
	ID              string `json:"id"`
	Status          string `json:"status"`
	Phase           string `json:"phase"`
	GeneratedTokens int    `json:"generated_tokens"`
	PromptTokens    int    `json:"prompt_tokens"`
	Worker          string `json:"worker,omitempty"`
	Attempts        int    `json:"attempts"`
	HeldBlocks      int    `json:"held_blocks"`
	StopReason      string `json:"stop_reason,omitempty"`
	Error           string `json:"error,omitempty"`
	Tenant          string `json:"tenant"`
	Model           string `json:"model"`
}

func (g *Gateway) handleStatus(w http.ResponseWriter, r *http.Request) {
	h, ok := g.cfg.Scheduler.Lookup(RequestID(r.PathValue("id")))
	if !ok {
		g.refuse(w, http.StatusNotFound, "unknown_request", "no such request", 0)
		return
	}
	res := h.Result()
	writeJSON(w, http.StatusOK, statusResponse{
		ID: string(res.ID), Status: res.Status.String(), Phase: res.Phase.String(),
		GeneratedTokens: res.GeneratedTokens, PromptTokens: res.PromptTokens,
		Worker: string(res.Worker), Attempts: res.Attempts, HeldBlocks: res.HeldBlocks,
		StopReason: res.StopReason, Error: res.Err,
		Tenant: string(res.Tenant), Model: string(res.Model),
	})
}

// handleHealthz answers for the process, not the fleet.
//
// Liveness and readiness are deliberately different questions: a control plane
// with no workers is perfectly alive and must not be restarted for it — restarting
// it would discard the queue — but it is not ready to serve.
func (g *Gateway) handleHealthz(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

func (g *Gateway) handleReadyz(w http.ResponseWriter, r *http.Request) {
	model := ModelID(r.URL.Query().Get("model"))
	if model == "" {
		model = g.cfg.DefaultModel
	}
	now := g.cfg.Clock.Now()

	healthy := 0
	for _, ws := range g.cfg.Registry.Healthy(now) {
		if model != "" && ws.Model != model {
			continue
		}
		healthy++
	}
	accepting := g.cfg.Scheduler == nil || g.cfg.Scheduler.Accepting()
	ready := healthy > 0 && accepting

	status := http.StatusOK
	if !ready {
		status = http.StatusServiceUnavailable
	}
	reason := ""
	switch {
	case !accepting:
		reason = "shutting_down"
	case healthy == 0:
		reason = RejectModelUnavailable.String()
	}
	writeJSON(w, status, map[string]any{
		"ready": ready, "healthy_workers": healthy,
		"model": string(model), "reason": reason,
	})
}

func (g *Gateway) tenantOf(r *http.Request) (TenantID, bool) {
	t := TenantID(strings.TrimSpace(r.Header.Get(g.cfg.TenantHeader)))
	if t == "" {
		t = g.cfg.DefaultTenant
	}
	return t, t != ""
}

// statusForReject maps a refusal to HTTP.
//
// The mapping is the contract a client's retry logic is written against, so each
// case is chosen for what the client should do rather than for how the server
// feels: 429 means the same request will work later, 503 means it needs a
// different fleet, 400 means it will never work, 504 means it is already too late.
func statusForReject(reason RejectReason) (int, string) {
	switch reason {
	case RejectQueueFull, RejectTenantQuota:
		return http.StatusTooManyRequests, reason.String()
	case RejectNoCapacity, RejectModelUnavailable:
		return http.StatusServiceUnavailable, reason.String()
	case RejectDeadlineInfeasible:
		return http.StatusGatewayTimeout, reason.String()
	case RejectTooLarge:
		return http.StatusBadRequest, reason.String()
	}
	return http.StatusInternalServerError, reason.String()
}

func rejectMessage(reason RejectReason, dec AdmitDecision) string {
	switch reason {
	case RejectQueueFull:
		return "the request queue is full; retry shortly"
	case RejectTenantQuota:
		return "tenant quota exhausted"
	case RejectTooLarge:
		return "the request cannot fit on any worker in the fleet even when idle"
	case RejectDeadlineInfeasible:
		return "the deadline cannot be met, so the request was refused rather than served late"
	case RejectNoCapacity:
		return "no worker accepting work can serve a request of this size"
	case RejectModelUnavailable:
		return "no healthy worker serves the requested model"
	}
	return "request refused (" + dec.String() + ")"
}

func retryAfterMS(reason RejectReason) int {
	switch reason {
	case RejectQueueFull, RejectNoCapacity, RejectModelUnavailable:
		return 500
	case RejectTenantQuota:
		return 1000
	}
	return 0
}

func (g *Gateway) refuse(w http.ResponseWriter, status int, reason, message string, retryAfterMS int) {
	if retryAfterMS > 0 {
		w.Header().Set("Retry-After", strconv.Itoa((retryAfterMS+999)/1000))
	}
	writeJSON(w, status, errorBody{
		Error: http.StatusText(status), Reason: reason, Message: message,
		RetryAfterMS: retryAfterMS,
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		http.Error(w, `{"error":"encoding failed"}`, http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(b)+1))
	w.WriteHeader(status)
	b = append(b, '\n')
	_, _ = w.Write(b)
}
