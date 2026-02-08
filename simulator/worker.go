package simulator

import (
	"fmt"
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
)

// WorkerSpec configures one simulated worker.
type WorkerSpec struct {
	ID    loom.WorkerID
	Model loom.ModelID

	// TotalBlocks and TokensPerBlock are the worker's cache capacity, reported at
	// registration in the real protocol. TokensPerBlock must match the cost
	// estimator's, or admission is counting blocks of a different size than the
	// worker allocates.
	TotalBlocks    int
	TokensPerBlock int

	MaxBatchSize int

	// SpeedFactor scales every latency; 2.0 is a worker twice as fast. Heterogeneity
	// matters because a fleet of identical workers hides the question of where to
	// place a request, and a real fleet is never identical for long.
	SpeedFactor float64

	// ColdStart is how long after launch the worker can serve. Until it elapses the
	// worker is capacity that exists and cannot be used, which is the cost the
	// autoscaler's hysteresis is paying to avoid.
	ColdStart time.Duration

	// Contiguous selects ContiguousCache over PagedCache for this worker's KV cache.
	Contiguous bool
	Fit        loom.FitPolicy
}

func (s WorkerSpec) withDefaults() WorkerSpec {
	if s.Model == "" {
		s.Model = "tiny-gpt2"
	}
	if s.TokensPerBlock <= 0 {
		s.TokensPerBlock = 16
	}
	if s.TotalBlocks <= 0 {
		s.TotalBlocks = 512
	}
	if s.MaxBatchSize <= 0 {
		s.MaxBatchSize = 16
	}
	if s.SpeedFactor <= 0 {
		s.SpeedFactor = 1
	}
	return s
}

// simWorker is one simulated worker process.
//
// It owns a real loom.KVCache and a real loom.Batcher. The only thing it simulates
// is how long a step takes and when a sequence stops producing tokens; every
// capacity decision inside it is made by the control plane code under test.
type simWorker struct {
	spec  WorkerSpec
	cache loom.KVCache
	batch loom.Batcher

	// truth is what the worker would report in a heartbeat: cache usage measured from
	// the cache itself, not predicted.
	truth loom.WorkerState

	// cp is the control plane's view of this worker. It is a separate struct, not a
	// pointer to truth, and that separation is the point: the control plane advances
	// its own number from the reservations it made and only learns the real one on a
	// heartbeat. Collapsing them into one would make the estimate-versus-actual drift
	// identically zero by construction and the drift experiment vacuous.
	cp loom.WorkerState

	running []*loom.RequestState

	// cpReserved is the per-sequence block count the control plane believes it
	// reserved, which is what it has to release when the sequence leaves. Recomputed
	// from the estimator rather than read out of the cache, because reading the cache
	// would be the control plane consulting the ground truth it does not have.
	cpReserved map[loom.RequestID]int

	// ready is false during the cold start.
	ready bool
	alive bool

	// stepping guards against two in-flight step events for one worker, which would
	// let a worker execute two forward passes at once.
	stepping bool

	steps        uint64
	tokens       uint64
	prefillSteps uint64

	// lastTPS is the decayed throughput the heartbeat reports. The autoscaler reads
	// it, so it has to come from observed work rather than from the config.
	lastTPS       float64
	lastStepStart time.Time
	lastStepBatch int
}

func newSimWorker(spec WorkerSpec, est *loom.CostEstimator, mode loom.ReservationMode, mk BatcherFactory, now time.Time) *simWorker {
	spec = spec.withDefaults()
	policy := loom.CachePolicy{
		TokensPerBlock: spec.TokensPerBlock,
		TotalBlocks:    spec.TotalBlocks,
		Contiguous:     spec.Contiguous,
	}
	var cache loom.KVCache
	if spec.Contiguous {
		cache = loom.NewContiguousCache(policy, spec.Fit)
	} else {
		cache = loom.NewPagedCache(policy)
	}

	st := loom.WorkerState{
		ID:               spec.ID,
		Model:            spec.Model,
		TotalCacheBlocks: spec.TotalBlocks,
		MaxBatchSize:     spec.MaxBatchSize,
		LastHeartbeat:    now,
	}
	w := &simWorker{
		spec:       spec,
		cache:      cache,
		batch:      mk(cache, est, mode),
		truth:      st,
		cp:         st,
		cpReserved: make(map[loom.RequestID]int, 32),
		ready:      spec.ColdStart <= 0,
		alive:      true,
	}
	return w
}

// dropRunning removes one sequence from the worker's running set.
func (w *simWorker) dropRunning(id loom.RequestID) {
	out := w.running[:0]
	for _, st := range w.running {
		if st.ID != id {
			out = append(out, st)
		}
	}
	w.running = out
}

// contextTokens is the total KV positions the batch spans, which is what makes a
// late decode step cost more than an early one.
func (w *simWorker) contextTokens() int {
	n := 0
	for _, st := range w.running {
		n += st.PromptTokens + st.GeneratedTokens
	}
	return n
}

func (w *simWorker) String() string {
	return fmt.Sprintf("%s(blocks=%d batch=%d speed=%.2f)", w.spec.ID, w.spec.TotalBlocks, w.spec.MaxBatchSize, w.spec.SpeedFactor)
}
