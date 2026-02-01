package loom

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"
)

// Policy and batcher names, as a config file or a flag spells them. Strings rather
// than an enum because the experiments sweep them from the command line, and a
// name that does not resolve must be an error the operator can read.
const (
	PolicyFIFO      = "fifo"
	PolicySJF       = "sjf"
	PolicyFairShare = "fair-share"

	BatcherContinuous = "continuous"
	BatcherStatic     = "static"
)

// effectivelyUnboundedQueue is what a zero queue capacity resolves to.
//
// Not a truly unbounded queue: the experiment that needs "unbounded" needs a queue
// that never refuses at the loads under test, and a real infinity would turn a
// misconfigured run into an out-of-memory kill of the control plane rather than
// into the latency blowup the experiment is trying to show.
const effectivelyUnboundedQueue = 1 << 20

// Config is everything needed to run a control plane.
//
// It is deliberately complete enough to express every experiment: the scheduling
// policy, the batching mode and its window, paged versus contiguous cache,
// peak versus expected reservation, autoscaling on or off, and a bounded or
// effectively unbounded queue. An experiment that needs a code change to run is an
// experiment that will not be run.
type Config struct {
	HTTPAddr   string
	WorkerAddr string

	// Namespace prefixes the metric series.
	Namespace string

	// QueueCapacity is the hard queue bound; zero means effectively unbounded, which
	// exists to be measured against the bounded case rather than to be deployed.
	QueueCapacity int

	Policy string

	// SJFAgingRate keeps shortest-job-first from starving long requests outright.
	SJFAgingRate float64

	// TenantWeights are the fair-share weights.
	TenantWeights map[TenantID]float64

	// TokensPerBlock is the KV-cache block granularity used for every accounting
	// decision. A worker reporting a different value is a misconfiguration the
	// server logs, because every reservation the control plane computes would
	// otherwise be wrong by that ratio.
	TokensPerBlock int

	// ContiguousCache selects the naive allocator, kept so paging has something to
	// be measured against.
	ContiguousCache bool
	ContiguousFit   FitPolicy

	// ReservationMode chooses between reserving the declared ceiling and reserving
	// the expected length. Explicit because it is the system's central tradeoff:
	// deterministic waste against stochastic preemption.
	ReservationMode ReservationMode

	// ExpectedOutputFraction feeds the default output-length model.
	ExpectedOutputFraction float64

	Batcher     string
	BatchWindow time.Duration
	Victims     VictimPolicy

	Quotas map[TenantID]TenantQuota

	TenantHeader  string
	DefaultTenant TenantID
	DefaultModel  ModelID
	MaxBodyBytes  int64

	HeartbeatInterval  time.Duration
	SuspectAfterMisses int
	DeadAfterMisses    int
	Retry              RetryPolicy

	// Autoscaling turns the autoscaler on. Off is a valid configuration and the
	// baseline the autoscaling experiments are compared against.
	Autoscaling bool
	Autoscaler  AutoscalerConfig

	Tick            time.Duration
	DispatchTimeout time.Duration
	RequestTimeout  time.Duration

	// WorkerSendQueue bounds a worker connection's outbound queue, which is where
	// the protocol's explicit backpressure shows up.
	WorkerSendQueue int

	ShutdownGrace time.Duration

	Clock Clock
	Logf  func(format string, args ...any)
}

func (c Config) withDefaults() Config {
	if c.HTTPAddr == "" {
		c.HTTPAddr = ":8080"
	}
	if c.WorkerAddr == "" {
		c.WorkerAddr = ":9090"
	}
	if c.Namespace == "" {
		c.Namespace = "loom"
	}
	if c.QueueCapacity <= 0 {
		c.QueueCapacity = effectivelyUnboundedQueue
	}
	if c.Policy == "" {
		c.Policy = PolicyFIFO
	}
	if c.TokensPerBlock <= 0 {
		c.TokensPerBlock = 16
	}
	if c.ExpectedOutputFraction <= 0 {
		c.ExpectedOutputFraction = 0.5
	}
	if c.Batcher == "" {
		c.Batcher = BatcherContinuous
	}
	if c.HeartbeatInterval <= 0 {
		c.HeartbeatInterval = 500 * time.Millisecond
	}
	if c.Tick <= 0 {
		c.Tick = 5 * time.Millisecond
	}
	if c.DispatchTimeout <= 0 {
		c.DispatchTimeout = 2 * time.Second
	}
	if c.ShutdownGrace <= 0 {
		c.ShutdownGrace = 30 * time.Second
	}
	if c.Clock == nil {
		c.Clock = SystemClock{}
	}
	return c
}

// Server assembles and runs the control plane.
type Server struct {
	cfg Config

	Metrics   *Metrics
	Registry  *Registry
	Queue     *BoundedQueue
	Admission *QuotaAdmissionController
	Scheduler *Scheduler
	Gateway   *Gateway
	Estimator *CostEstimator
	Scaler    *QueueWaitAutoscaler

	httpSrv      *http.Server
	httpLn       net.Listener
	workerLn     net.Listener
	workerConns  sync.Map // WorkerID -> *WorkerConn, for shutdown
	acceptDone   chan struct{}
	serveDone    chan struct{}
	shutdownOnce sync.Once
}

// NewServer builds every component from cfg and wires them together. Nothing
// listens until Start.
func NewServer(cfg Config) (*Server, error) {
	cfg = cfg.withDefaults()

	policy, err := buildPolicy(cfg)
	if err != nil {
		return nil, err
	}
	if cfg.Batcher != BatcherContinuous && cfg.Batcher != BatcherStatic {
		return nil, fmt.Errorf("loom: unknown batcher %q", cfg.Batcher)
	}

	m := NewMetrics(cfg.Namespace)
	registry := NewRegistry(RegistryConfig{
		Clock:              cfg.Clock,
		HeartbeatInterval:  cfg.HeartbeatInterval,
		SuspectAfterMisses: cfg.SuspectAfterMisses,
		DeadAfterMisses:    cfg.DeadAfterMisses,
		Retry:              cfg.Retry,
	})
	queue := NewBoundedQueue(cfg.QueueCapacity, policy)
	queue.SetClock(cfg.Clock.Now)

	// TotalBlocks is zero in the estimator's policy on purpose: the estimator only
	// needs the block granularity to convert tokens to blocks, and capacity is a
	// per-worker number that comes from the worker's own registration.
	est := NewCostEstimator(
		CachePolicy{TokensPerBlock: cfg.TokensPerBlock, Contiguous: cfg.ContiguousCache},
		FractionOfMax{Fraction: cfg.ExpectedOutputFraction, Floor: 1},
	)
	admission := NewQuotaAdmissionController(est, cfg.ReservationMode, cfg.Quotas)

	var scaler *QueueWaitAutoscaler
	var auto Autoscaler
	if cfg.Autoscaling {
		ac := cfg.Autoscaler
		if ac.Clock == nil {
			ac.Clock = cfg.Clock
		}
		scaler = NewQueueWaitAutoscaler(ac)
		auto = scaler
	}

	s := &Server{
		cfg: cfg, Metrics: m, Registry: registry, Queue: queue,
		Admission: admission, Estimator: est, Scaler: scaler,
		acceptDone: make(chan struct{}), serveDone: make(chan struct{}),
	}

	s.Scheduler = NewScheduler(SchedulerConfig{
		Queue: queue, Registry: registry, Admission: admission, Metrics: m,
		Autoscaler:       auto,
		NewCache:         s.newCache,
		NewBatcher:       s.newBatcher,
		Clock:            cfg.Clock,
		Tick:             cfg.Tick,
		HeartbeatTimeout: registry.DeadAfter(),
		DispatchTimeout:  cfg.DispatchTimeout,
		OnScaleUp:        s.onScaleUp,
		Logf:             cfg.Logf,
	})

	s.Gateway = NewGateway(GatewayConfig{
		Scheduler: s.Scheduler, Registry: registry, Metrics: m, Clock: cfg.Clock,
		MaxBodyBytes: cfg.MaxBodyBytes, TenantHeader: cfg.TenantHeader,
		DefaultTenant: cfg.DefaultTenant, DefaultModel: cfg.DefaultModel,
		RequestTimeout: cfg.RequestTimeout, Logf: cfg.Logf,
	})
	return s, nil
}

func buildPolicy(cfg Config) (SchedulePolicy, error) {
	switch cfg.Policy {
	case PolicyFIFO:
		return NewFIFOPolicy(), nil
	case PolicySJF:
		return NewShortestJobFirstPolicy(cfg.SJFAgingRate), nil
	case PolicyFairShare:
		p := NewFairSharePolicy()
		for t, w := range cfg.TenantWeights {
			p.SetWeight(t, w)
		}
		return p, nil
	}
	return nil, fmt.Errorf("loom: unknown policy %q", cfg.Policy)
}

// newCache builds one worker's cache. Capacity comes from the worker's own
// registration rather than from config: the worker knows its device memory and the
// control plane does not, and a configured guess eventually guesses high and
// causes an out-of-memory kill mid-generation.
func (s *Server) newCache(w *WorkerState) KVCache {
	policy := CachePolicy{
		TokensPerBlock: s.cfg.TokensPerBlock,
		TotalBlocks:    w.TotalCacheBlocks,
		Contiguous:     s.cfg.ContiguousCache,
	}
	if s.cfg.ContiguousCache {
		return NewContiguousCache(policy, s.cfg.ContiguousFit)
	}
	return NewPagedCache(policy)
}

func (s *Server) newBatcher(w *WorkerState, cache KVCache) Batcher {
	if s.cfg.Batcher == BatcherStatic {
		return NewStaticBatcher(s.cfg.BatchWindow, s.Estimator, s.cfg.ReservationMode, cache)
	}
	return NewContinuousBatcher(s.cfg.BatchWindow, s.Estimator, s.cfg.ReservationMode, cache, s.cfg.Victims)
}

func (s *Server) onScaleUp(d ScaleDecision) {
	s.logf("loom: autoscaler wants %+d workers: %s", d.Delta, d.Reason)
}

// Start begins listening. It returns once both listeners are bound, so a caller
// can read the addresses and connect without racing the accept loop.
func (s *Server) Start() error {
	workerLn, err := net.Listen("tcp", s.cfg.WorkerAddr)
	if err != nil {
		return fmt.Errorf("loom: worker listener: %w", err)
	}
	httpLn, err := net.Listen("tcp", s.cfg.HTTPAddr)
	if err != nil {
		_ = workerLn.Close()
		return fmt.Errorf("loom: http listener: %w", err)
	}
	s.workerLn, s.httpLn = workerLn, httpLn
	s.httpSrv = &http.Server{
		Handler: s.Gateway.Handler(),
		// No write timeout: a streamed generation legitimately lasts as long as the
		// generation does, and a timeout here would cut off exactly the long requests
		// the scheduler works hardest to serve.
		ReadHeaderTimeout: 10 * time.Second,
	}

	s.Scheduler.Start()
	go s.acceptWorkers()
	go func() {
		defer close(s.serveDone)
		if err := s.httpSrv.Serve(httpLn); err != nil && !errors.Is(err, http.ErrServerClosed) {
			s.logf("loom: http serve: %v", err)
		}
	}()
	return nil
}

// HTTPAddr and WorkerAddr report the bound addresses, which differ from the
// configured ones when port 0 was requested.
func (s *Server) HTTPAddr() string {
	if s.httpLn == nil {
		return s.cfg.HTTPAddr
	}
	return s.httpLn.Addr().String()
}

func (s *Server) WorkerAddr() string {
	if s.workerLn == nil {
		return s.cfg.WorkerAddr
	}
	return s.workerLn.Addr().String()
}

// Run starts the server and blocks until ctx is cancelled, then shuts down
// gracefully.
func (s *Server) Run(ctx context.Context) error {
	if err := s.Start(); err != nil {
		return err
	}
	<-ctx.Done()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), s.cfg.ShutdownGrace)
	defer cancel()
	return s.Shutdown(shutdownCtx)
}

// Shutdown stops accepting, drains in-flight work, and closes the workers.
//
// The order matters: the HTTP listener closes first so no new request is admitted,
// then the scheduler drains, then the worker sockets close. Closing the worker
// sockets first would fail every in-flight generation, which is the thing draining
// exists to avoid.
func (s *Server) Shutdown(ctx context.Context) error {
	var err error
	s.shutdownOnce.Do(func() {
		if s.workerLn != nil {
			_ = s.workerLn.Close()
		}
		if s.httpSrv != nil {
			// Closing the listener but letting in-flight responses finish; the streamed
			// ones end when the scheduler finishes their requests.
			if e := s.httpSrv.Shutdown(ctx); e != nil {
				err = e
			}
		}
		if e := s.Scheduler.Shutdown(ctx); e != nil && err == nil {
			err = e
		}
		s.workerConns.Range(func(_, v any) bool {
			if wc, ok := v.(*WorkerConn); ok {
				_ = wc.Close()
			}
			return true
		})
		<-s.acceptDone
	})
	return err
}

func (s *Server) acceptWorkers() {
	defer close(s.acceptDone)
	for {
		conn, err := s.workerLn.Accept()
		if err != nil {
			// A closed listener is the shutdown path, not an error worth reporting.
			return
		}
		if tcp, ok := conn.(*net.TCPConn); ok {
			// Keepalive so a worker host that vanishes without closing its socket is
			// eventually noticed by the kernel as well as by the heartbeat timeout.
			_ = tcp.SetKeepAlive(true)
			_ = tcp.SetKeepAlivePeriod(2 * s.cfg.HeartbeatInterval)
		}
		s.serveWorker(conn)
	}
}

func (s *Server) serveWorker(conn net.Conn) {
	NewWorkerConn(conn, WorkerConnConfig{
		SendQueue:    s.cfg.WorkerSendQueue,
		WriteTimeout: s.cfg.DispatchTimeout,
		Clock:        s.cfg.Clock,
		Prompts:      s.Scheduler.Prompt,
		Logf:         s.cfg.Logf,
		OnRegister:   s.onWorkerRegister,
		OnHeartbeat:  func(_ *WorkerConn, hb Heartbeat) { s.Scheduler.Heartbeat(hb) },
		OnClose:      s.onWorkerClose,
	})
}

func (s *Server) onWorkerRegister(wc *WorkerConn, reg Registration) {
	if reg.TokensPerBlock > 0 && reg.TokensPerBlock != s.cfg.TokensPerBlock {
		// Not fatal, but every reservation computed for this worker is wrong by the
		// ratio, so it must be loud rather than silently absorbed into cache drift.
		s.logf("loom: worker %s reports tokens_per_block=%d but the control plane accounts in %d",
			reg.Worker, reg.TokensPerBlock, s.cfg.TokensPerBlock)
	}
	ws := &WorkerState{
		ID: reg.Worker, Model: reg.Model,
		TotalCacheBlocks: reg.TotalCacheBlocks,
		MaxBatchSize:     reg.MaxBatchSize,
		LastHeartbeat:    reg.At,
	}
	if err := s.Scheduler.AddWorker(ws, wc); err != nil {
		s.logf("loom: refusing worker %s: %v", reg.Worker, err)
		_ = wc.Close()
		return
	}
	s.workerConns.Store(reg.Worker, wc)
	// On its own goroutine: ConsumeEvents runs for the life of the connection, and
	// this callback is on the connection's reader goroutine, which is the thing
	// producing the events it would be consuming.
	go s.Scheduler.ConsumeEvents(reg.Worker, wc)
}

func (s *Server) onWorkerClose(wc *WorkerConn, cause error) {
	id := wc.ID()
	if id == "" {
		return
	}
	s.workerConns.Delete(id)
	if cause != nil {
		s.logf("loom: worker %s disconnected: %v", id, cause)
	}
	// ConsumeEvents also reports the loss when the event channel closes; the
	// scheduler treats a second report for an unknown worker as a no-op, which is
	// cheaper than coordinating which of the two paths gets there first.
	s.Scheduler.WorkerGone(id)
}

func (s *Server) logf(format string, args ...any) {
	if s.cfg.Logf != nil {
		s.cfg.Logf(format, args...)
	}
}
