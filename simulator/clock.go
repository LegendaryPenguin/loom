// Package simulator is a deterministic, virtual-time simulation of a worker
// fleet driven by the real control plane components.
//
// It exists because neither of the obvious alternatives can answer the questions
// bench/ asks. Running real model inference on CPU is three or four orders of
// magnitude too slow to push a scheduler into sustained overload, so the regime
// where every interesting scheduling decision happens is unreachable. Driving the
// scheduler with sleeps on a wall clock is reachable but worthless: a p99 measured
// against the host OS scheduler moves when the machine is busy, so a regression and
// a noisy neighbour are indistinguishable.
//
// Virtual time fixes both. Ten minutes of simulated traffic runs in well under a
// second, and the result is byte-for-byte reproducible from a seed, so a number
// that changes means the code changed.
//
// What is simulated is strictly the *cost* of inference: how long a forward pass
// takes and how many tokens come out. Everything that decides what runs — the
// queue, the policies, the admission controller, the batchers, the registry, the
// autoscaler — is the real implementation from go-control-plane, and so is the
// KV-cache accounting. Accounting that only exists in the simulator proves nothing
// about the system.
package simulator

import (
	"container/heap"
	"time"
)

// eventKind is what the simulation is waking up to do.
type eventKind uint8

const (
	evArrival eventKind = iota
	evTick
	evWorkerStep
	evWorkerReady
	evWorkerCrash
)

// event is one scheduled wakeup.
//
// seq breaks ties between events at the same virtual instant, and it is the single
// reason this simulation is reproducible at all. Two events scheduled for the same
// nanosecond have to be ordered somehow; without an explicit tiebreak the order
// comes from heap internals, which depend on insertion history in ways that change
// when unrelated code changes. seq is assigned monotonically on push, so
// same-instant events fire in the order they were created.
type event struct {
	at   time.Time
	kind eventKind
	seq  uint64

	// worker is the WorkerID the event is about, carried as a string so this file stays
	// independent of the control plane's types. An ID rather than a fleet index: the fleet
	// slice is compacted when a drained worker is retired, so an index scheduled before a
	// scale-down names a different worker — or no worker — by the time it fires.
	worker string

	req int // index into the arrival list, for evArrival
}

type eventHeap []event

func (h eventHeap) Len() int { return len(h) }
func (h eventHeap) Less(i, j int) bool {
	if !h[i].at.Equal(h[j].at) {
		return h[i].at.Before(h[j].at)
	}
	return h[i].seq < h[j].seq
}
func (h eventHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *eventHeap) Push(x any)   { *h = append(*h, x.(event)) }
func (h *eventHeap) Pop() any {
	old := *h
	n := len(old)
	e := old[n-1]
	*h = old[:n-1]
	return e
}

// VirtualClock is the simulation's time source and event queue in one.
//
// They are one type because they are one concept: virtual time only advances when
// an event is dequeued, so a clock that could be read independently of the queue
// would be a clock nobody advanced. It satisfies loom.Clock, which is what lets the
// registry's health machine and the autoscaler's cooldowns run on simulated time
// without knowing they are being simulated.
type VirtualClock struct {
	now   time.Time
	start time.Time

	pending eventHeap
	seq     uint64
}

// NewVirtualClock starts a clock at an arbitrary fixed instant.
//
// Fixed rather than time.Now: every timestamp in a result then depends only on the
// seed and the config, so two runs produce identical output and a diff over the
// table is meaningful.
func NewVirtualClock() *VirtualClock {
	t := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
	return &VirtualClock{now: t, start: t}
}

func (c *VirtualClock) Now() time.Time   { return c.now }
func (c *VirtualClock) Start() time.Time { return c.start }

// Elapsed is virtual time since the start, which is the denominator of every
// throughput number in bench/.
func (c *VirtualClock) Elapsed() time.Duration { return c.now.Sub(c.start) }

// At schedules an event. A time in the past is clamped to now rather than
// rejected: a step whose computed duration rounds to zero is a legitimate thing for
// a cost model to produce, and it must not be able to move the clock backwards.
func (c *VirtualClock) At(e event) {
	if e.at.Before(c.now) {
		e.at = c.now
	}
	e.seq = c.seq
	c.seq++
	heap.Push(&c.pending, e)
}

// Next dequeues the earliest event and advances the clock to it.
func (c *VirtualClock) Next() (event, bool) {
	if len(c.pending) == 0 {
		return event{}, false
	}
	e := heap.Pop(&c.pending).(event)
	c.now = e.at
	return e, true
}

func (c *VirtualClock) Pending() int { return len(c.pending) }
