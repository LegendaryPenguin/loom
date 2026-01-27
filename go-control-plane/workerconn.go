package loom

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net"
	"sync"
	"time"
)

// Errors a WorkerConn returns. They are distinguishable values because the
// scheduler reacts differently to each: a full send queue is backpressure and the
// batch should be retried against another worker, a closed connection means the
// worker is gone and its reservations must be released, and a timeout means the
// worker is alive but not reading and may still be holding the work.
var (
	ErrSendQueueFull  = errors.New("loom: worker send queue full")
	ErrWorkerClosed   = errors.New("loom: worker connection closed")
	ErrNotRegistered  = errors.New("loom: worker has not registered")
	ErrEmptyDispatch  = errors.New("loom: dispatch with no requests")
	ErrMessageTooLong = errors.New("loom: worker message exceeds line limit")
)

// Protocol message type strings. Spelled once so a typo is a compile error in one
// place rather than a message the peer silently skips.
const (
	msgRegister   = "register"
	msgHeartbeat  = "heartbeat"
	msgFirstToken = "first_token"
	msgToken      = "token"
	msgCompleted  = "completed"
	msgFailed     = "failed"
	msgPreempted  = "preempted"
	msgDispatch   = "dispatch"
	msgCancel     = "cancel"
	msgDrain      = "drain"
)

// Registration is the worker's `register` message.
//
// It is surfaced through its own callback rather than on the Events channel
// because WorkerEvent has no room for any of these fields, and they are the
// fields the control plane needs most: the worker's own measurement of its cache
// capacity is what every admission decision for it is made against.
type Registration struct {
	Worker           WorkerID
	Model            ModelID
	TotalCacheBlocks int
	TokensPerBlock   int
	MaxBatchSize     int
	Device           string
	At               time.Time
}

// Heartbeat is the worker's periodic report. UsedCacheBlocks is authoritative per
// PROTOCOL.md and overwrites the control plane's estimate.
type Heartbeat struct {
	Worker          WorkerID
	UsedCacheBlocks int
	ActiveSequences int
	TokensPerSecond float64
	QueueDepth      int
	At              time.Time
}

// WorkerConnConfig configures one connection.
type WorkerConnConfig struct {
	// SendQueue bounds outbound messages awaiting the socket. Bounded because
	// PROTOCOL.md makes backpressure explicit: a worker that is not reading must
	// cause a visible rejection, not an invisible queue that hides how far behind
	// it is.
	SendQueue int

	// EventBuffer bounds inbound events awaiting the scheduler. Events are never
	// dropped when it fills — a dropped `completed` leaks a reservation forever —
	// so the reader blocks, and the resulting missed heartbeats make the stall
	// visible as a worker health problem instead of as silent corruption.
	EventBuffer int

	// WriteTimeout is the ceiling on one socket write when the caller's context
	// carries no deadline of its own.
	WriteTimeout time.Duration

	// MaxLineBytes caps one inbound line. A peer that never sends a newline would
	// otherwise grow the read buffer without limit.
	MaxLineBytes int

	Clock Clock

	// Prompts supplies the prompt text for a request id.
	//
	// It is a lookup rather than a field on RequestState because Request carries
	// only PromptTokens — the scheduler reasons about cost, which is a function of
	// the token count, and never about the text. PROTOCOL.md's dispatch does need
	// the text, so the component that holds it (the gateway) provides it here
	// instead of the control plane's core types growing a payload they never read.
	Prompts func(RequestID) string

	// OnRegister and OnHeartbeat are called from the reader goroutine, in arrival
	// order, and must not block for long.
	OnRegister  func(*WorkerConn, Registration)
	OnHeartbeat func(*WorkerConn, Heartbeat)

	// OnClose fires once when the connection is finished, so the server can drop
	// its reference without polling.
	OnClose func(*WorkerConn, error)

	// Logf receives skipped messages. Unknown types and malformed lines are logged
	// and skipped rather than treated as fatal, so a newer control plane can talk
	// to an older worker without taking it down.
	Logf func(format string, args ...any)
}

func (c WorkerConnConfig) withDefaults() WorkerConnConfig {
	if c.SendQueue <= 0 {
		c.SendQueue = 32
	}
	if c.EventBuffer <= 0 {
		c.EventBuffer = 256
	}
	if c.WriteTimeout <= 0 {
		c.WriteTimeout = 2 * time.Second
	}
	if c.MaxLineBytes <= 0 {
		c.MaxLineBytes = 1 << 20
	}
	if c.Clock == nil {
		c.Clock = SystemClock{}
	}
	return c
}

// WorkerConn speaks PROTOCOL.md to one worker process over a socket.
//
// Reads and writes are each owned by a single goroutine. The alternative —
// writing from whichever goroutine happens to be dispatching — means a slow
// worker stalls a caller that holds scheduler state, and the one thing the
// control plane cannot afford is for a worker to be able to stop it.
type WorkerConn struct {
	cfg  WorkerConnConfig
	conn net.Conn

	events chan WorkerEvent
	sendq  chan *outbound

	closeOnce sync.Once
	closed    chan struct{}

	// writerDone gates Close's socket teardown on the writer having stopped, so an
	// in-progress write is not torn down underneath itself.
	writerDone chan struct{}
	readerDone chan struct{}

	mu       sync.Mutex
	reg      Registration
	haveReg  bool
	closeErr error

	// failures holds the `retryable` flag of each reported failure.
	//
	// WorkerEvent cannot carry it, and it is the one bit that decides between
	// retrying and refusing: a control plane that guesses "retryable" turns one
	// poison request into an infinite loop. Kept here and consumed by request id
	// so the single ordered event channel stays the only sequencing mechanism.
	failures map[RequestID]Failure
}

var _ WorkerClient = (*WorkerConn)(nil)

// NewWorkerConn starts the reader and writer for conn. The connection is live
// immediately; the first inbound message is expected to be `register`, and
// Dispatch refuses until it arrives, because dispatching to a worker whose cache
// capacity is unknown is a guess the protocol exists to avoid.
func NewWorkerConn(conn net.Conn, cfg WorkerConnConfig) *WorkerConn {
	cfg = cfg.withDefaults()
	w := &WorkerConn{
		cfg:        cfg,
		conn:       conn,
		events:     make(chan WorkerEvent, cfg.EventBuffer),
		sendq:      make(chan *outbound, cfg.SendQueue),
		closed:     make(chan struct{}),
		writerDone: make(chan struct{}),
		readerDone: make(chan struct{}),
		failures:   make(map[RequestID]Failure),
	}
	go w.readLoop()
	go w.writeLoop()
	return w
}

// Events is the ordered stream of request progress. Closed when the connection
// ends, which is how the scheduler learns to stop reading without a second
// signal.
func (w *WorkerConn) Events() <-chan WorkerEvent { return w.events }

// Closed is closed when the connection is finished, for a caller that wants to
// wait rather than poll.
func (w *WorkerConn) Closed() <-chan struct{} { return w.closed }

// ID is the worker's id, empty until it registers.
func (w *WorkerConn) ID() WorkerID {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.reg.Worker
}

// Registration returns the worker's register message, and whether it has arrived.
func (w *WorkerConn) Registration() (Registration, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.reg, w.haveReg
}

// TakeFailure consumes the retryability classification recorded for a failed
// request. Consuming rather than reading keeps a retry of the same id from
// inheriting the previous attempt's verdict.
func (w *WorkerConn) TakeFailure(id RequestID) (Failure, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	f, ok := w.failures[id]
	if ok {
		delete(w.failures, id)
	}
	return f, ok
}

// Err reports why the connection ended, nil for a clean close.
func (w *WorkerConn) Err() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.closeErr
}

// Close ends the connection. Idempotent, because both the reader noticing EOF and
// the scheduler reaping a dead worker legitimately call it.
func (w *WorkerConn) Close() error {
	w.closeWith(nil)
	return nil
}

func (w *WorkerConn) closeWith(cause error) {
	w.closeOnce.Do(func() {
		w.mu.Lock()
		w.closeErr = cause
		w.mu.Unlock()

		close(w.closed)
		// Closing the socket is what unblocks a reader parked in Read and a writer
		// parked in Write; without it a worker that neither sends nor reads would
		// hold both goroutines until the process exited.
		_ = w.conn.Close()

		if w.cfg.OnClose != nil {
			go w.cfg.OnClose(w, cause)
		}
	})
}

// outbound is one framed message plus the caller's reply channel. The reply
// carries the socket's verdict back, so Dispatch reports a write failure to the
// scheduler rather than returning success for a batch that never left the host.
type outbound struct {
	line     []byte
	deadline time.Time
	reply    chan error
}

// Dispatch sends a batch of new work.
//
// Only the requests the worker does not already have should be passed; the
// protocol's at-least-once rule makes a duplicate harmless, but re-sending the
// whole running set every tick would make the socket traffic grow with batch size
// times tick rate for no benefit.
func (w *WorkerConn) Dispatch(ctx context.Context, batch []*RequestState) error {
	if len(batch) == 0 {
		return ErrEmptyDispatch
	}
	if _, ok := w.Registration(); !ok {
		return ErrNotRegistered
	}
	msg := wireDispatch{Type: msgDispatch, Batch: make([]wireDispatchItem, 0, len(batch))}
	for _, st := range batch {
		if st == nil {
			continue
		}
		blocks := st.HeldBlocks
		if blocks <= 0 {
			// Never dispatch a zero reservation: the worker must not exceed what it was
			// given, and zero would make every decode step an overrun.
			blocks = st.Est.ExpectedCacheBlocks
		}
		prompt := ""
		if w.cfg.Prompts != nil {
			prompt = w.cfg.Prompts(st.ID)
		}
		msg.Batch = append(msg.Batch, wireDispatchItem{
			RequestID:       string(st.ID),
			Prompt:          prompt,
			PromptTokens:    st.PromptTokens,
			MaxOutputTokens: st.MaxOutputTokens,
			ReservedBlocks:  blocks,
		})
	}
	if len(msg.Batch) == 0 {
		return ErrEmptyDispatch
	}
	return w.send(ctx, msg)
}

// Cancel asks the worker to abandon a sequence. Best effort by contract: the
// control plane releases its own accounting without waiting, since the case that
// matters most is a worker that will never answer.
func (w *WorkerConn) Cancel(ctx context.Context, id RequestID) error {
	return w.send(ctx, wireCancel{Type: msgCancel, RequestID: string(id)})
}

// Drain tells the worker to finish what it has and accept nothing new.
func (w *WorkerConn) Drain(ctx context.Context) error {
	return w.send(ctx, wireDrain{Type: msgDrain})
}

func (w *WorkerConn) send(ctx context.Context, v any) error {
	line, err := json.Marshal(v)
	if err != nil {
		return err
	}
	line = append(line, '\n')

	deadline, ok := ctx.Deadline()
	if !ok {
		deadline = w.cfg.Clock.Now().Add(w.cfg.WriteTimeout)
	}
	o := &outbound{line: line, deadline: deadline, reply: make(chan error, 1)}

	select {
	case <-w.closed:
		return ErrWorkerClosed
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	select {
	case w.sendq <- o:
	default:
		// The explicit rejection PROTOCOL.md requires. Queueing behind a worker that
		// is not draining its socket would hide its depth from the one component that
		// has to see it.
		return ErrSendQueueFull
	}

	select {
	case err := <-o.reply:
		return err
	case <-ctx.Done():
		// The message may still be written; the caller's deadline has passed either
		// way, and the alternative is a control plane pinned by an unresponsive peer.
		return ctx.Err()
	case <-w.closed:
		return ErrWorkerClosed
	}
}

func (w *WorkerConn) writeLoop() {
	defer close(w.writerDone)
	bw := bufio.NewWriter(w.conn)
	for {
		select {
		case <-w.closed:
			w.failPending(ErrWorkerClosed)
			return
		case o := <-w.sendq:
			err := w.writeOne(bw, o)
			o.reply <- err
			if err != nil {
				w.closeWith(err)
				w.failPending(err)
				return
			}
		}
	}
}

func (w *WorkerConn) writeOne(bw *bufio.Writer, o *outbound) error {
	// A per-message deadline rather than one for the connection's lifetime: the
	// property wanted is that no single call can hang, and a connection-wide
	// deadline would also kill an idle but healthy worker.
	if err := w.conn.SetWriteDeadline(o.deadline); err != nil {
		return err
	}
	if _, err := bw.Write(o.line); err != nil {
		return err
	}
	// Flushed per message because the peer is waiting on a newline; a buffered
	// dispatch that sits unflushed is a batch the worker never starts.
	return bw.Flush()
}

func (w *WorkerConn) failPending(err error) {
	for {
		select {
		case o := <-w.sendq:
			o.reply <- err
		default:
			return
		}
	}
}

func (w *WorkerConn) readLoop() {
	defer close(w.readerDone)
	// Events is closed exactly once, by the reader, after the last event it will
	// ever produce. A consumer ranging over it therefore sees every event and then
	// termination, with no separate liveness check to get wrong.
	defer close(w.events)

	sc := bufio.NewScanner(w.conn)
	sc.Buffer(make([]byte, 0, 64<<10), w.cfg.MaxLineBytes)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var in wireIn
		if err := json.Unmarshal(line, &in); err != nil {
			// Skipped, not fatal. One corrupt line must not cost the control plane a
			// worker and every sequence running on it.
			w.logf("loom: worker %s: skipping malformed message: %v", w.ID(), err)
			continue
		}
		if !w.handle(&in) {
			return
		}
	}
	err := sc.Err()
	if errors.Is(err, bufio.ErrTooLong) {
		err = ErrMessageTooLong
	}
	w.closeWith(err)
}

// handle applies one decoded message. It returns false when the reader should
// stop, which only happens on shutdown.
func (w *WorkerConn) handle(in *wireIn) bool {
	now := w.cfg.Clock.Now()
	switch in.Type {
	case msgRegister:
		reg := Registration{
			Worker:           WorkerID(in.WorkerID),
			Model:            ModelID(in.ModelID),
			TotalCacheBlocks: in.TotalCacheBlocks,
			TokensPerBlock:   in.TokensPerBlock,
			MaxBatchSize:     in.MaxBatchSize,
			Device:           in.Device,
			At:               now,
		}
		w.mu.Lock()
		first := !w.haveReg
		w.reg, w.haveReg = reg, true
		w.mu.Unlock()
		if first && w.cfg.OnRegister != nil {
			w.cfg.OnRegister(w, reg)
		}
		return true

	case msgHeartbeat:
		if w.cfg.OnHeartbeat != nil {
			w.cfg.OnHeartbeat(w, Heartbeat{
				Worker:          w.resolveWorker(in.WorkerID),
				UsedCacheBlocks: in.UsedCacheBlocks,
				ActiveSequences: in.ActiveSequences,
				TokensPerSecond: in.TokensPerSecond,
				QueueDepth:      in.QueueDepth,
				At:              now,
			})
		}
		return true

	case msgFirstToken:
		// Tokens is 1 rather than in.Generated: first_token is by definition the
		// first, and the worker's latency_ms measures only its own prefill, while the
		// time-to-first-token a client experiences includes the queue wait the control
		// plane is the only party that can see. The scheduler times it from arrival.
		return w.emit(WorkerEvent{Kind: EventFirstToken, Request: RequestID(in.RequestID), Tokens: 1, Timestamp: now})

	case msgToken:
		return w.emit(WorkerEvent{
			Kind: EventToken, Request: RequestID(in.RequestID),
			Tokens: in.Generated, Blocks: in.Blocks,
			// Err carries the event's string payload, which for a token is the token
			// text. Streaming it to the client is the whole reason the field is read.
			Err:       in.Token,
			Timestamp: now,
		})

	case msgCompleted:
		return w.emit(WorkerEvent{
			Kind: EventCompleted, Request: RequestID(in.RequestID),
			Tokens: in.Generated, Err: in.StopReason, Timestamp: now,
		})

	case msgFailed:
		id := RequestID(in.RequestID)
		w.mu.Lock()
		w.failures[id] = Failure{Request: id, Worker: w.reg.Worker, Transient: in.Retryable, Err: in.Error}
		w.mu.Unlock()
		return w.emit(WorkerEvent{Kind: EventFailed, Request: id, Err: in.Error, Timestamp: now})

	case msgPreempted:
		return w.emit(WorkerEvent{Kind: EventPreempted, Request: RequestID(in.RequestID), Tokens: in.Generated, Timestamp: now})

	default:
		w.logf("loom: worker %s: skipping unknown message type %q", w.ID(), in.Type)
		return true
	}
}

// resolveWorker prefers the id on the wire but falls back to the registered one,
// because every message after `register` is unambiguous on a per-worker
// connection and an older worker may well omit it.
func (w *WorkerConn) resolveWorker(onWire string) WorkerID {
	if onWire != "" {
		return WorkerID(onWire)
	}
	return w.ID()
}

func (w *WorkerConn) emit(ev WorkerEvent) bool {
	w.mu.Lock()
	ev.Worker = w.reg.Worker
	w.mu.Unlock()
	select {
	case w.events <- ev:
		return true
	case <-w.closed:
		return false
	}
}

func (w *WorkerConn) logf(format string, args ...any) {
	if w.cfg.Logf != nil {
		w.cfg.Logf(format, args...)
	}
}

// wireIn is every inbound field in one struct.
//
// One struct rather than a type switch over seven: the fields do not collide, an
// unknown field is ignored by the decoder as the protocol requires, and a single
// decode per line is both simpler and cheaper than decoding the envelope and then
// the body.
type wireIn struct {
	Type string `json:"type"`

	WorkerID         string `json:"worker_id"`
	ModelID          string `json:"model_id"`
	TotalCacheBlocks int    `json:"total_cache_blocks"`
	TokensPerBlock   int    `json:"tokens_per_block"`
	MaxBatchSize     int    `json:"max_batch_size"`
	Device           string `json:"device"`

	UsedCacheBlocks int     `json:"used_cache_blocks"`
	ActiveSequences int     `json:"active_sequences"`
	TokensPerSecond float64 `json:"tokens_per_second"`
	QueueDepth      int     `json:"queue_depth"`

	RequestID  string `json:"request_id"`
	LatencyMS  int    `json:"latency_ms"`
	Token      string `json:"token"`
	Generated  int    `json:"generated"`
	Blocks     int    `json:"blocks"`
	StopReason string `json:"stop_reason"`
	Error      string `json:"error"`
	Retryable  bool   `json:"retryable"`
}

type wireDispatchItem struct {
	RequestID string `json:"request_id"`
	Prompt    string `json:"prompt"`

	// PromptTokens is sent even though the worker will tokenize the prompt itself,
	// so a mismatch between the control plane's accounting and the worker's reality
	// is detectable at the worker rather than only as unexplained cache drift.
	PromptTokens    int `json:"prompt_tokens"`
	MaxOutputTokens int `json:"max_output_tokens"`
	ReservedBlocks  int `json:"reserved_blocks"`
}

type wireDispatch struct {
	Type  string             `json:"type"`
	Batch []wireDispatchItem `json:"batch"`
}

type wireCancel struct {
	Type      string `json:"type"`
	RequestID string `json:"request_id"`
}

type wireDrain struct {
	Type string `json:"type"`
}
