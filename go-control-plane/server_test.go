package loom

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"
)

// TestServerEndToEndOverRealSockets wires the whole control plane the way cmd/ does
// and drives it from both ends at once: a worker on the TCP socket speaking
// PROTOCOL.md, and a client on the HTTP API. The unit tests mock one side or the
// other, so this is the only place the two framings meet.
func TestServerEndToEndOverRealSockets(t *testing.T) {
	t.Parallel()

	srv, err := NewServer(Config{
		HTTPAddr: "127.0.0.1:0", WorkerAddr: "127.0.0.1:0",
		DefaultModel: "tiny-gpt2", DefaultTenant: "t-1",
		QueueCapacity: 16, TokensPerBlock: 16,
		Policy: PolicySJF, Batcher: BatcherContinuous,
		ReservationMode:   ReserveExpected,
		HeartbeatInterval: 10 * time.Second,
		Tick:              time.Millisecond,
		RequestTimeout:    5 * time.Second,
	})
	if err != nil {
		t.Fatalf("new server: %v", err)
	}
	if err := srv.Start(); err != nil {
		t.Fatalf("start: %v", err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
	})

	conn, err := net.Dial("tcp", srv.WorkerAddr())
	if err != nil {
		t.Fatalf("worker dial: %v", err)
	}
	defer conn.Close()
	worker := newFakeWorker(t, conn)

	worker.send(`{"type":"register","worker_id":"w-1","model_id":"tiny-gpt2",` +
		`"total_cache_blocks":256,"tokens_per_block":16,"max_batch_size":8,"device":"cpu"}`)

	client := &http.Client{Timeout: 5 * time.Second}
	waitFor(t, 3*time.Second, "readyz to go green after the worker registers", func() bool {
		resp, err := client.Get("http://" + srv.HTTPAddr() + "/readyz")
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		_, _ = io.Copy(io.Discard, resp.Body)
		return resp.StatusCode == http.StatusOK
	})

	type reply struct {
		status int
		body   []byte
	}
	got := make(chan reply, 1)
	go func() {
		req, err := http.NewRequest(http.MethodPost, "http://"+srv.HTTPAddr()+"/v1/generate",
			strings.NewReader(`{"id":"r-1","prompt":"once upon a time","max_output_tokens":16}`))
		if err != nil {
			got <- reply{0, []byte(err.Error())}
			return
		}
		req.Header.Set("X-Loom-Tenant", "t-1")
		resp, err := client.Do(req)
		if err != nil {
			got <- reply{0, []byte(err.Error())}
			return
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		got <- reply{resp.StatusCode, b}
	}()

	msg := worker.read()
	if msg["type"] != msgDispatch {
		t.Fatalf("expected a dispatch, got %v", msg)
	}
	batch := msg["batch"].([]any)
	item := batch[0].(map[string]any)
	if item["request_id"] != "r-1" {
		t.Fatalf("dispatch item %v", item)
	}
	if item["prompt"] != "once upon a time" {
		t.Fatalf("the prompt did not survive the trip to the worker: %v", item)
	}
	if item["reserved_blocks"].(float64) <= 0 {
		t.Fatalf("worker was dispatched a zero reservation: %v", item)
	}

	worker.send(`{"type":"first_token","request_id":"r-1","latency_ms":12}`)
	worker.send(`{"type":"token","request_id":"r-1","token":"there","generated":1,"blocks":2}`)
	worker.send(`{"type":"token","request_id":"r-1","token":" was","generated":2,"blocks":2}`)
	worker.send(`{"type":"heartbeat","worker_id":"w-1","used_cache_blocks":2,"active_sequences":1,` +
		`"tokens_per_second":88.5,"queue_depth":0}`)
	worker.send(`{"type":"completed","request_id":"r-1","generated":2,"stop_reason":"eos"}`)

	select {
	case r := <-got:
		if r.status != http.StatusOK {
			t.Fatalf("status %d: %s", r.status, r.body)
		}
		var out generateResponse
		if err := json.Unmarshal(r.body, &out); err != nil {
			t.Fatalf("body %q: %v", r.body, err)
		}
		if out.Text != "there was" {
			t.Fatalf("text %q", out.Text)
		}
		if out.Status != StatusCompleted.String() || out.GeneratedTokens != 2 {
			t.Fatalf("result %+v", out)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the HTTP response never arrived")
	}

	// The heartbeat is authoritative, so the reconciliation must have happened by the
	// time the request finished.
	waitFor(t, 2*time.Second, "heartbeat reconciliation", func() bool {
		return srv.Admission.Metrics().Reconciles > 0
	})
}

func TestServerRejectsUnknownPolicyAndBatcher(t *testing.T) {
	t.Parallel()
	if _, err := NewServer(Config{Policy: "nonesuch"}); err == nil {
		t.Fatal("an unknown policy must be an error the operator can read, not a silent default")
	}
	if _, err := NewServer(Config{Batcher: "nonesuch"}); err == nil {
		t.Fatal("an unknown batcher must be an error")
	}
}

func TestServerUnboundedQueueResolvesToALimit(t *testing.T) {
	t.Parallel()
	srv, err := NewServer(Config{QueueCapacity: 0, HTTPAddr: "127.0.0.1:0", WorkerAddr: "127.0.0.1:0"})
	if err != nil {
		t.Fatalf("new server: %v", err)
	}
	// "Unbounded" is a configuration the experiments need and an operator should not
	// get: it is large, not infinite, so a misconfigured run degrades into latency
	// rather than into an out-of-memory kill of the control plane.
	if got := srv.Queue.Cap(); got != effectivelyUnboundedQueue {
		t.Fatalf("queue capacity %d", got)
	}
}
