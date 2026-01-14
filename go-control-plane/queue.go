package loom

import (
	"container/heap"
	"sync"
	"time"
)

// Snapshot returns the read-only view handed to a scheduling policy.
//
// BoundedQueue is its own snapshot: the view methods are reads under the same
// mutex, so there is nothing to copy and no window in which the policy sees a
// queue that has since changed.
func (q *BoundedQueue) Snapshot() QueueSnapshot { return q }

func (v queueView) Snapshot() QueueSnapshot { return v }

// qentry is the queue's internal node for one request.
//
// The storage is an intrusive doubly linked list in push order plus a map from
// RequestID to the node, which is what makes the operations the scheduler loop
// performs cheap in the right places:
//
//   - Push/Pop at the ends, and OldestWait from the head: O(1).
//   - Remove by id, which cancellation drives and which a slice would make O(n)
//     with an O(n) shift on top: O(1), because the map finds the node and the
//     node already knows its neighbours.
//   - Policy iteration in arrival order: O(n) by pointer chase, which is inherent
//     — a policy that ranks the queue has to see the queue.
//
// A slice would iterate with better locality, but every cancellation and every
// expiry would be a linear scan plus a memmove, and cancellation is common in a
// serving system where clients give up. A single heap was rejected because the
// ordering is the policy's business and it changes per policy, so the container
// must not bake one in.
//
// The deadline heap is a second index over the same nodes, carried so DropExpired
// costs O(k log n) in the number of actually-expired requests instead of O(n) per
// scheduler tick. heapIdx is -1 for a request with no deadline, which is never in
// the heap.
type qentry struct {
	st   *RequestState
	prev *qentry
	next *qentry

	heapIdx int
}

// deadlineHeap is a min-heap of entries by Deadline. It stores back-pointers into
// the entries so an arbitrary Remove can fix the heap in O(log n) rather than
// leaving tombstones that DropExpired would have to re-validate forever.
type deadlineHeap []*qentry

func (h deadlineHeap) Len() int           { return len(h) }
func (h deadlineHeap) Less(i, j int) bool { return h[i].st.Deadline.Before(h[j].st.Deadline) }
func (h deadlineHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i]; h[i].heapIdx = i; h[j].heapIdx = j }
func (h *deadlineHeap) Push(x any)        { e := x.(*qentry); e.heapIdx = len(*h); *h = append(*h, e) }
func (h *deadlineHeap) Pop() any {
	old := *h
	n := len(old)
	e := old[n-1]
	old[n-1] = nil
	*h = old[:n-1]
	e.heapIdx = -1
	return e
}

// BoundedQueue is the bounded request queue.
//
// Bounded with refusal at the limit, never growth and never blocking: an
// unbounded queue turns overload into unbounded latency, and a Push that blocked
// would push the backpressure into the gateway's request goroutine, where it
// becomes a stuck connection instead of an honest 503.
//
// Safe for concurrent use. One mutex rather than a lock per index, because every
// operation touches at least two of the three indexes (list, id map, tenant
// counts) and they must agree; splitting the lock would buy nothing but a window
// in which Len and TenantCount disagree.
type BoundedQueue struct {
	mu       sync.Mutex
	capacity int

	head *qentry
	tail *qentry
	n    int

	byID    map[RequestID]*qentry
	tenants map[TenantID]int
	expiry  deadlineHeap

	policy SchedulePolicy

	// workers is how Pop learns current worker state. Queue.Pop takes no
	// arguments but SchedulePolicy.Next needs workers to judge whether a request
	// can currently fit, so the owner of the queue supplies a source. When it is
	// nil the policy is called with no workers, which every policy here treats as
	// "feasibility unknown, do not filter" — the alternative, treating it as "no
	// capacity", would make Pop silently always return nil.
	workers func() []*WorkerState

	// now is injectable so tests can drive deadline and aging behaviour without
	// sleeping. Production leaves it nil and gets time.Now.
	now func() time.Time
}

var _ Queue = (*BoundedQueue)(nil)
var _ QueueSnapshot = (*BoundedQueue)(nil)

// NewBoundedQueue builds a queue of the given capacity selecting with policy.
// A nil policy means arrival order, so a queue is usable before a policy is
// configured rather than panicking in the gateway's hot path.
func NewBoundedQueue(capacity int, policy SchedulePolicy) *BoundedQueue {
	if capacity < 0 {
		capacity = 0
	}
	if policy == nil {
		policy = NewFIFOPolicy()
	}
	return &BoundedQueue{
		capacity: capacity,
		byID:     make(map[RequestID]*qentry),
		tenants:  make(map[TenantID]int),
		policy:   policy,
	}
}

// SetWorkerSource installs the callback Pop uses to get worker state. The source
// is called outside the queue's lock-free sections but inside its mutex, so it
// must not call back into the queue.
func (q *BoundedQueue) SetWorkerSource(fn func() []*WorkerState) {
	q.mu.Lock()
	q.workers = fn
	q.mu.Unlock()
}

// SetClock overrides the queue's notion of now, for tests.
func (q *BoundedQueue) SetClock(fn func() time.Time) {
	q.mu.Lock()
	q.now = fn
	q.mu.Unlock()
}

// Policy reports the configured policy, so metrics can label by it.
func (q *BoundedQueue) Policy() SchedulePolicy {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.policy
}

func (q *BoundedQueue) clock() time.Time {
	if q.now != nil {
		return q.now()
	}
	return time.Now()
}

// Push enqueues st, reporting false when the queue is at capacity or already
// holds that id. A rejected Push leaves the queue exactly as it was — the caller
// is about to turn the false into a RejectQueueFull and must not find the request
// half-enqueued when it does.
func (q *BoundedQueue) Push(st *RequestState) bool {
	if st == nil {
		return false
	}
	q.mu.Lock()
	defer q.mu.Unlock()

	if q.n >= q.capacity {
		return false
	}
	if _, dup := q.byID[st.ID]; dup {
		// At-least-once dispatch means duplicate arrivals are expected. Admitting
		// the same id twice would hand two copies to the scheduler and leak one
		// request's reservation.
		return false
	}

	if st.QueuedAt.IsZero() {
		st.QueuedAt = q.clock()
	}
	st.Phase = PhaseQueued

	e := &qentry{st: st, heapIdx: -1}
	if q.tail == nil {
		q.head, q.tail = e, e
	} else {
		e.prev = q.tail
		q.tail.next = e
		q.tail = e
	}
	q.n++
	q.byID[st.ID] = e
	q.tenants[st.Tenant]++
	if !st.Deadline.IsZero() {
		heap.Push(&q.expiry, e)
	}
	return true
}

// Pop returns the request the policy selects and removes it, or nil when the
// queue is empty or the policy judges nothing currently runnable. Those two cases
// are deliberately indistinguishable here; a caller that needs to tell them apart
// reads Len.
func (q *BoundedQueue) Pop() *RequestState {
	q.mu.Lock()
	defer q.mu.Unlock()

	if q.n == 0 {
		return nil
	}

	var workers []*WorkerState
	if q.workers != nil {
		workers = q.workers()
	}

	// The policy is called with a read-only view rather than with q itself: the
	// lock is held, and q's exported methods take it again. The view's methods are
	// the unlocked internals, so a policy sees a consistent queue and cannot
	// deadlock against the Pop that invoked it.
	st := q.policy.Next(queueView{q}, workers, q.clock())
	if st == nil {
		return nil
	}
	e, ok := q.byID[st.ID]
	if !ok {
		// The policy returned something not in this queue. Refusing is the only
		// safe answer: dispatching it would hand out a request nobody is tracking.
		return nil
	}
	q.unlink(e)
	return e.st
}

// Remove drops a request by id. O(1): the map locates the node and the node
// unlinks itself. Cancellation and deadline expiry both land here, and both are
// driven by client behaviour rather than by load, so they cannot be allowed to
// cost a scan of a queue that is deep precisely when the system is busiest.
func (q *BoundedQueue) Remove(id RequestID) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	e, ok := q.byID[id]
	if !ok {
		return false
	}
	q.unlink(e)
	return true
}

// DropExpired removes every request whose deadline has passed and returns them.
//
// Returning rather than discarding is the whole point: the caller reports them
// failed, and the metrics show deadline loss instead of showing requests that
// simply evaporated. Running one of these would burn accelerator capacity
// producing a result the client has already stopped waiting for.
func (q *BoundedQueue) DropExpired(now time.Time) []*RequestState {
	q.mu.Lock()
	defer q.mu.Unlock()

	var out []*RequestState
	for len(q.expiry) > 0 {
		e := q.expiry[0]
		if e.st.Deadline.After(now) {
			break
		}
		q.unlink(e)
		out = append(out, e.st)
	}
	return out
}

// Len is the current depth.
func (q *BoundedQueue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.n
}

// Cap is the hard limit.
func (q *BoundedQueue) Cap() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.capacity
}

// OldestWait is how long the request at the head has been waiting, zero when
// empty. This is the autoscaler's primary signal: depth alone cannot distinguish
// a deep queue draining fast from a shallow one that is stuck.
func (q *BoundedQueue) OldestWait(now time.Time) time.Duration {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.oldestWait(now)
}

// Each walks the queue in push order under the lock. Exported for policies; see
// QueueSnapshot for why the callback must not mutate the queue.
func (q *BoundedQueue) Each(fn func(st *RequestState) bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.each(fn)
}

// TenantCount is the queued count for one tenant.
func (q *BoundedQueue) TenantCount(t TenantID) int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.tenants[t]
}

// TenantCounts copies the per-tenant depths.
func (q *BoundedQueue) TenantCounts() map[TenantID]int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.tenantCounts()
}

// --- unlocked internals. Callers must hold q.mu. ---

func (q *BoundedQueue) unlink(e *qentry) {
	if e.prev != nil {
		e.prev.next = e.next
	} else {
		q.head = e.next
	}
	if e.next != nil {
		e.next.prev = e.prev
	} else {
		q.tail = e.prev
	}
	e.prev, e.next = nil, nil

	delete(q.byID, e.st.ID)
	if c := q.tenants[e.st.Tenant]; c <= 1 {
		// Deleting at zero rather than leaving a zero entry keeps the map from
		// growing without bound in a system with many short-lived tenants.
		delete(q.tenants, e.st.Tenant)
	} else {
		q.tenants[e.st.Tenant] = c - 1
	}
	if e.heapIdx >= 0 {
		heap.Remove(&q.expiry, e.heapIdx)
		e.heapIdx = -1
	}
	q.n--
}

func (q *BoundedQueue) each(fn func(st *RequestState) bool) {
	for e := q.head; e != nil; e = e.next {
		if !fn(e.st) {
			return
		}
	}
}

func (q *BoundedQueue) tenantCounts() map[TenantID]int {
	out := make(map[TenantID]int, len(q.tenants))
	for t, c := range q.tenants {
		out[t] = c
	}
	return out
}

func (q *BoundedQueue) oldestWait(now time.Time) time.Duration {
	if q.head == nil {
		return 0
	}
	d := now.Sub(queuedSince(q.head.st))
	if d < 0 {
		return 0
	}
	return d
}

// queuedSince is when a request started waiting for its *current* turn. QueuedAt is
// preferred because a re-queued preempted request has an old ArrivalTime but a fresh
// wait — preemption resets QueuedAt for exactly this reason — and ArrivalTime is the
// fallback for a request a caller built without setting QueuedAt.
//
// The two timestamps are not interchangeable and the choice is per consumer:
//
//   - Scheduling uses this. Aging, head-of-line wait and the autoscaler's primary
//     signal all exist to stop a request from waiting unboundedly for a slot, and the
//     time a preempted request spent running is not waiting. Charging it as wait
//     would hand it an aging credit it did not earn, let it dominate OldestWait, and
//     trip SkipGrace the instant it is re-queued.
//   - Client-facing latency uses ArrivalTime. End-to-end latency and
//     time-to-first-token are what the client experienced, and the client does not
//     care how many times the request was rescheduled.
func queuedSince(st *RequestState) time.Time {
	if !st.QueuedAt.IsZero() {
		return st.QueuedAt
	}
	return st.ArrivalTime
}

// queueView is the read-only, already-locked view handed to a policy.
//
// It satisfies Queue so it can be passed to SchedulePolicy.Next, but the three
// mutating methods are refused rather than implemented. A policy that tries to
// mutate is a bug; refusing makes that bug show up as "this policy never
// schedules anything", which is loud and harmless, instead of as corrupted queue
// state or a self-deadlock.
type queueView struct{ q *BoundedQueue }

var _ Queue = queueView{}
var _ QueueSnapshot = queueView{}

func (v queueView) Push(*RequestState) bool               { return false }
func (v queueView) Pop() *RequestState                    { return nil }
func (v queueView) Remove(RequestID) bool                 { return false }
func (v queueView) DropExpired(time.Time) []*RequestState { return nil }
func (v queueView) Len() int                              { return v.q.n }
func (v queueView) Cap() int                              { return v.q.capacity }

func (v queueView) OldestWait(now time.Time) time.Duration { return v.q.oldestWait(now) }
func (v queueView) Each(fn func(*RequestState) bool)       { v.q.each(fn) }
func (v queueView) TenantCount(t TenantID) int             { return v.q.tenants[t] }
func (v queueView) TenantCounts() map[TenantID]int         { return v.q.tenantCounts() }
