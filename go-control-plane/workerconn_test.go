package loom

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net"
	"os"
	"testing"
	"time"
)

// fakeWorker is a hand-written peer that speaks PROTOCOL.md over a real socket.
//
// Hand-written rather than driven by the production encoder: a round-trip test
// where both ends share the marshalling code proves only that the code agrees with
// itself. These messages are the ones in PROTOCOL.md, typed out.
type fakeWorker struct {
	t    *testing.T
	conn net.Conn
	sc   *bufio.Scanner
}

func newFakeWorker(t *testing.T, conn net.Conn) *fakeWorker {
	t.Helper()
	sc := bufio.NewScanner(conn)
	sc.Buffer(make([]byte, 0, 4096), 1<<20)
	return &fakeWorker{t: t, conn: conn, sc: sc}
}

func (f *fakeWorker) send(line string) {
	f.t.Helper()
	if err := f.conn.SetWriteDeadline(time.Now().Add(2 * time.Second)); err != nil {
		f.t.Fatalf("set write deadline: %v", err)
	}
	if _, err := f.conn.Write([]byte(line + "\n")); err != nil {
		f.t.Fatalf("worker write: %v", err)
	}
}

func (f *fakeWorker) read() map[string]any {
	f.t.Helper()
	if err := f.conn.SetReadDeadline(time.Now().Add(2 * time.Second)); err != nil {
		f.t.Fatalf("set read deadline: %v", err)
	}
	if !f.sc.Scan() {
		f.t.Fatalf("worker read: %v", f.sc.Err())
	}
	var m map[string]any
	if err := json.Unmarshal(f.sc.Bytes(), &m); err != nil {
		f.t.Fatalf("worker decode %q: %v", f.sc.Bytes(), err)
	}
	return m
}

// dialFakeWorker returns the control plane's side and the worker's side of a real
// TCP connection.
func dialFakeWorker(t *testing.T, cfg WorkerConnConfig) (*WorkerConn, *fakeWorker) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	t.Cleanup(func() { _ = ln.Close() })

	type accepted struct {
		conn net.Conn
		err  error
	}
	ch := make(chan accepted, 1)
	go func() {
		c, err := ln.Accept()
		ch <- accepted{c, err}
	}()

	client, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	a := <-ch
	if a.err != nil {
		t.Fatalf("accept: %v", a.err)
	}

	// The control plane holds the dialled end here; in production the worker dials
	// and the server accepts, which makes no difference to the framing.
	wc := NewWorkerConn(client, cfg)
	t.Cleanup(func() {
		_ = wc.Close()
		_ = a.conn.Close()
	})
	return wc, newFakeWorker(t, a.conn)
}

func mustEvent(t *testing.T, events <-chan WorkerEvent) WorkerEvent {
	t.Helper()
	select {
	case ev, ok := <-events:
		if !ok {
			t.Fatal("event channel closed early")
		}
		return ev
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for a worker event")
	}
	return WorkerEvent{}
}

func TestWorkerConnRoundTripsEveryMessage(t *testing.T) {
	t.Parallel()

	regs := make(chan Registration, 4)
	beats := make(chan Heartbeat, 4)
	wc, worker := dialFakeWorker(t, WorkerConnConfig{
		OnRegister:  func(_ *WorkerConn, r Registration) { regs <- r },
		OnHeartbeat: func(_ *WorkerConn, h Heartbeat) { beats <- h },
		Prompts:     func(RequestID) string { return "once upon a time" },
	})

	worker.send(`{"type":"register","worker_id":"w-1","model_id":"tiny-gpt2","total_cache_blocks":2048,` +
		`"tokens_per_block":16,"max_batch_size":32,"device":"cpu","unknown_field":"ignored"}`)

	select {
	case reg := <-regs:
		if reg.Worker != "w-1" || reg.Model != "tiny-gpt2" {
			t.Fatalf("register identity: %+v", reg)
		}
		if reg.TotalCacheBlocks != 2048 || reg.TokensPerBlock != 16 || reg.MaxBatchSize != 32 {
			t.Fatalf("register capacity: %+v", reg)
		}
		if reg.Device != "cpu" {
			t.Fatalf("register device: %+v", reg)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no register callback")
	}

	worker.send(`{"type":"heartbeat","worker_id":"w-1","used_cache_blocks":912,"active_sequences":7,` +
		`"tokens_per_second":143.2,"queue_depth":0}`)
	select {
	case hb := <-beats:
		if hb.UsedCacheBlocks != 912 || hb.ActiveSequences != 7 || hb.TokensPerSecond != 143.2 {
			t.Fatalf("heartbeat: %+v", hb)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no heartbeat callback")
	}

	worker.send(`{"type":"first_token","request_id":"r-7","latency_ms":184}`)
	worker.send(`{"type":"token","request_id":"r-7","token":" the","generated":12,"blocks":3}`)
	worker.send(`{"type":"preempted","request_id":"r-8","generated":31}`)
	worker.send(`{"type":"failed","request_id":"r-9","error":"cuda oom","retryable":true}`)
	worker.send(`{"type":"completed","request_id":"r-7","generated":48,"stop_reason":"eos"}`)

	ev := mustEvent(t, wc.Events())
	if ev.Kind != EventFirstToken || ev.Request != "r-7" || ev.Worker != "w-1" {
		t.Fatalf("first_token event: %+v", ev)
	}

	ev = mustEvent(t, wc.Events())
	if ev.Kind != EventToken || ev.Tokens != 12 || ev.Blocks != 3 || ev.Err != " the" {
		t.Fatalf("token event: %+v", ev)
	}

	ev = mustEvent(t, wc.Events())
	if ev.Kind != EventPreempted || ev.Request != "r-8" || ev.Tokens != 31 {
		t.Fatalf("preempted event: %+v", ev)
	}

	ev = mustEvent(t, wc.Events())
	if ev.Kind != EventFailed || ev.Request != "r-9" || ev.Err != "cuda oom" {
		t.Fatalf("failed event: %+v", ev)
	}
	f, ok := wc.TakeFailure("r-9")
	if !ok || !f.Transient {
		t.Fatalf("retryable flag lost: %+v ok=%v", f, ok)
	}
	if _, again := wc.TakeFailure("r-9"); again {
		t.Fatal("failure record survived being taken, so a later attempt would inherit it")
	}

	ev = mustEvent(t, wc.Events())
	if ev.Kind != EventCompleted || ev.Tokens != 48 || ev.Err != "eos" {
		t.Fatalf("completed event: %+v", ev)
	}

	// Control plane to worker.
	st := &RequestState{Request: Request{ID: "r-7", PromptTokens: 11, MaxOutputTokens: 128}, HeldBlocks: 12}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := wc.Dispatch(ctx, []*RequestState{st}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	msg := worker.read()
	if msg["type"] != msgDispatch {
		t.Fatalf("dispatch type: %v", msg)
	}
	batch, ok := msg["batch"].([]any)
	if !ok || len(batch) != 1 {
		t.Fatalf("dispatch batch: %v", msg)
	}
	item := batch[0].(map[string]any)
	if item["request_id"] != "r-7" || item["prompt"] != "once upon a time" {
		t.Fatalf("dispatch item: %v", item)
	}
	if item["reserved_blocks"].(float64) != 12 || item["max_output_tokens"].(float64) != 128 {
		t.Fatalf("dispatch reservation: %v", item)
	}

	if err := wc.Cancel(ctx, "r-7"); err != nil {
		t.Fatalf("cancel: %v", err)
	}
	if msg := worker.read(); msg["type"] != msgCancel || msg["request_id"] != "r-7" {
		t.Fatalf("cancel message: %v", msg)
	}

	if err := wc.Drain(ctx); err != nil {
		t.Fatalf("drain: %v", err)
	}
	if msg := worker.read(); msg["type"] != msgDrain {
		t.Fatalf("drain message: %v", msg)
	}
}

func TestWorkerConnRefusesDispatchBeforeRegister(t *testing.T) {
	t.Parallel()
	wc, _ := dialFakeWorker(t, WorkerConnConfig{})
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	err := wc.Dispatch(ctx, []*RequestState{{Request: Request{ID: "r-1"}, HeldBlocks: 1}})
	if !errors.Is(err, ErrNotRegistered) {
		t.Fatalf("want ErrNotRegistered, got %v", err)
	}
}

func TestWorkerConnMalformedMessagesAreSkippedNotFatal(t *testing.T) {
	t.Parallel()
	logged := make(chan string, 8)
	wc, worker := dialFakeWorker(t, WorkerConnConfig{
		Logf: func(format string, args ...any) {
			select {
			case logged <- format:
			default:
			}
		},
	})

	worker.send(`{"type":"register","worker_id":"w-1","model_id":"m","total_cache_blocks":64}`)
	worker.send(`{this is not json`)
	worker.send(`{"type":"some_future_message","request_id":"r-1"}`)
	// A valid event after the bad ones proves the connection survived both.
	worker.send(`{"type":"token","request_id":"r-1","token":"x","generated":1,"blocks":1}`)

	ev := mustEvent(t, wc.Events())
	if ev.Kind != EventToken || ev.Request != "r-1" {
		t.Fatalf("connection did not survive bad input: %+v", ev)
	}
	select {
	case <-wc.Closed():
		t.Fatal("malformed input closed the connection")
	default:
	}
	if len(logged) < 2 {
		t.Fatalf("want both the malformed line and the unknown type logged, got %d", len(logged))
	}
}

func TestWorkerConnVanishedWorkerErrorsRatherThanHangs(t *testing.T) {
	t.Parallel()
	wc, worker := dialFakeWorker(t, WorkerConnConfig{})
	worker.send(`{"type":"register","worker_id":"w-1","model_id":"m","total_cache_blocks":64}`)
	for {
		if _, ok := wc.Registration(); ok {
			break
		}
		time.Sleep(time.Millisecond)
	}

	// The worker dies mid-flight: no close handshake at the protocol level, just a
	// socket that is gone.
	_ = worker.conn.Close()

	st := &RequestState{Request: Request{ID: "r-1", MaxOutputTokens: 8}, HeldBlocks: 2}
	deadline := time.Now().Add(3 * time.Second)
	var lastErr error
	for time.Now().Before(deadline) {
		ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		lastErr = wc.Dispatch(ctx, []*RequestState{st})
		cancel()
		if lastErr != nil {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if lastErr == nil {
		t.Fatal("dispatch to a dead worker reported success")
	}

	select {
	case <-wc.Closed():
	case <-time.After(2 * time.Second):
		t.Fatal("connection to a dead worker never closed")
	}
	// Events closing is how the scheduler learns to release the worker's capacity.
	select {
	case _, ok := <-wc.Events():
		if ok {
			// Drain whatever was buffered, then require closure.
			for range wc.Events() {
			}
		}
	case <-time.After(2 * time.Second):
		t.Fatal("event channel never closed after the worker vanished")
	}
}

// pipeWorkerConn uses an unbuffered in-memory connection, which is how a worker
// that has stopped reading is modelled deterministically: a TCP socket's kernel
// buffers would absorb a test's worth of messages before blocking.
func pipeWorkerConn(t *testing.T, cfg WorkerConnConfig) *WorkerConn {
	t.Helper()
	ours, theirs := net.Pipe()
	wc := NewWorkerConn(ours, cfg)
	t.Cleanup(func() {
		_ = wc.Close()
		_ = theirs.Close()
	})
	return wc
}

func registerOverPipe(t *testing.T, wc *WorkerConn) {
	t.Helper()
	// Registration is forced directly because the peer in these tests never reads or
	// writes; what is under test is the write path, not the handshake.
	wc.mu.Lock()
	wc.reg = Registration{Worker: "w-stalled", Model: "m", TotalCacheBlocks: 64}
	wc.haveReg = true
	wc.mu.Unlock()
}

func TestWorkerConnStalledWorkerHitsTheDeadline(t *testing.T) {
	t.Parallel()
	wc := pipeWorkerConn(t, WorkerConnConfig{WriteTimeout: time.Hour})
	registerOverPipe(t, wc)

	st := &RequestState{Request: Request{ID: "r-1", MaxOutputTokens: 8}, HeldBlocks: 2}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	start := time.Now()
	err := wc.Dispatch(ctx, []*RequestState{st})
	elapsed := time.Since(start)

	// Either shape counts, and which one wins is a race between the caller's own
	// select and the socket deadline derived from the same instant. What matters is
	// that the deadline fired at all: the call returned, with an error, well before
	// the connection's hour-long write timeout.
	if !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, os.ErrDeadlineExceeded) {
		t.Fatalf("want a deadline error, got %v", err)
	}
	if elapsed > 2*time.Second {
		t.Fatalf("a stalled worker pinned the caller for %s", elapsed)
	}
}

func TestWorkerConnFullSendQueueRejects(t *testing.T) {
	t.Parallel()
	// Capacity one so the queue fills the moment the writer is stuck on its first
	// message, which is exactly the backpressure case PROTOCOL.md describes.
	wc := pipeWorkerConn(t, WorkerConnConfig{SendQueue: 1, WriteTimeout: time.Hour})
	registerOverPipe(t, wc)

	// No per-call deadlines here: a deadline would close the socket and the test
	// would be measuring the timeout rather than the rejection. One sender ends up
	// inside the blocked write, one in the queue, and every other must be refused.
	const senders = 8
	results := make(chan error, senders)
	for i := 0; i < senders; i++ {
		go func(i int) {
			st := &RequestState{
				Request:    Request{ID: RequestID("r-" + string(rune('a'+i))), MaxOutputTokens: 8},
				HeldBlocks: 2,
			}
			results <- wc.Dispatch(context.Background(), []*RequestState{st})
		}(i)
	}

	full := 0
	timeout := time.After(3 * time.Second)
	for full == 0 {
		select {
		case err := <-results:
			if errors.Is(err, ErrSendQueueFull) {
				full++
			} else if err != nil && !errors.Is(err, ErrWorkerClosed) {
				t.Fatalf("unexpected dispatch result: %v", err)
			}
		case <-timeout:
			t.Fatal("a worker that never reads should make the bounded send queue reject")
		}
	}
	// Closing releases the senders still parked on the full queue, which is also the
	// behaviour a reaped worker depends on.
	_ = wc.Close()
}

func TestWorkerConnCloseIsIdempotentAndUnblocksCallers(t *testing.T) {
	t.Parallel()
	wc := pipeWorkerConn(t, WorkerConnConfig{SendQueue: 1, WriteTimeout: time.Hour})
	registerOverPipe(t, wc)

	blocked := make(chan error, 1)
	go func() {
		// No deadline on this one: closing the connection is what must unblock it,
		// because a reaped worker's dispatcher cannot be left waiting for a timeout.
		blocked <- wc.Dispatch(context.Background(), []*RequestState{
			{Request: Request{ID: "r-1", MaxOutputTokens: 4}, HeldBlocks: 1},
		})
	}()
	time.Sleep(20 * time.Millisecond)

	if err := wc.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := wc.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}

	select {
	case err := <-blocked:
		if err == nil {
			t.Fatal("dispatch on a closed connection reported success")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("close did not unblock the dispatcher")
	}
}
