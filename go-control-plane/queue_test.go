package loom

import (
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

var schedTestBase = time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

// schedTestReq builds a queued request. Tests set QueuedAt explicitly so that
// wait-time behaviour is driven by arithmetic rather than by wall-clock sleeps.
func schedTestReq(id string, tenant string, queuedAt time.Time) *RequestState {
	return &RequestState{
		Request: Request{
			ID:              RequestID(id),
			Tenant:          TenantID(tenant),
			Model:           "m",
			PromptTokens:    64,
			MaxOutputTokens: 64,
			ArrivalTime:     queuedAt,
		},
		QueuedAt: queuedAt,
	}
}

func schedTestQueue(capacity int, p SchedulePolicy) *BoundedQueue {
	q := NewBoundedQueue(capacity, p)
	q.SetClock(func() time.Time { return schedTestBase })
	return q
}

func TestQueuePushPopLenCap(t *testing.T) {
	q := schedTestQueue(3, NewFIFOPolicy())
	if got := q.Cap(); got != 3 {
		t.Fatalf("Cap() = %d, want 3", got)
	}
	if got := q.Len(); got != 0 {
		t.Fatalf("Len() on empty = %d, want 0", got)
	}
	if st := q.Pop(); st != nil {
		t.Fatalf("Pop() on empty = %v, want nil", st.ID)
	}

	for i, id := range []string{"a", "b", "c"} {
		if !q.Push(schedTestReq(id, "t1", schedTestBase.Add(time.Duration(i)*time.Second))) {
			t.Fatalf("Push(%s) = false, want true", id)
		}
	}
	if got := q.Len(); got != 3 {
		t.Fatalf("Len() = %d, want 3", got)
	}

	for _, want := range []RequestID{"a", "b", "c"} {
		st := q.Pop()
		if st == nil || st.ID != want {
			t.Fatalf("Pop() = %v, want %s", st, want)
		}
	}
	if got := q.Len(); got != 0 {
		t.Fatalf("Len() after draining = %d, want 0", got)
	}
}

func TestQueuePushAtCapacityRejectsWithoutMutating(t *testing.T) {
	q := schedTestQueue(2, NewFIFOPolicy())
	q.Push(schedTestReq("a", "t1", schedTestBase))
	q.Push(schedTestReq("b", "t1", schedTestBase.Add(time.Second)))

	before := q.TenantCounts()
	if q.Push(schedTestReq("c", "t2", schedTestBase.Add(2*time.Second))) {
		t.Fatal("Push at capacity = true, want false")
	}
	if got := q.Len(); got != 2 {
		t.Fatalf("Len() after rejected Push = %d, want 2", got)
	}
	after := q.TenantCounts()
	if len(after) != len(before) || after["t1"] != before["t1"] {
		t.Fatalf("rejected Push mutated tenant counts: before %v after %v", before, after)
	}
	if _, ok := after["t2"]; ok {
		t.Fatal("rejected Push recorded a count for the rejected tenant")
	}
	// The queue must still contain exactly what it did before.
	var ids []RequestID
	q.Each(func(st *RequestState) bool { ids = append(ids, st.ID); return true })
	if len(ids) != 2 || ids[0] != "a" || ids[1] != "b" {
		t.Fatalf("contents after rejected Push = %v, want [a b]", ids)
	}
}

func TestQueuePushDuplicateIDRejected(t *testing.T) {
	// Dispatch is at-least-once, so duplicate ids reach the gateway in normal
	// operation; admitting both would leak one request's reservation.
	q := schedTestQueue(4, NewFIFOPolicy())
	if !q.Push(schedTestReq("a", "t1", schedTestBase)) {
		t.Fatal("first Push = false")
	}
	if q.Push(schedTestReq("a", "t1", schedTestBase)) {
		t.Fatal("duplicate Push = true, want false")
	}
	if got := q.Len(); got != 1 {
		t.Fatalf("Len() = %d, want 1", got)
	}
}

func TestQueueRemove(t *testing.T) {
	q := schedTestQueue(4, NewFIFOPolicy())
	q.Push(schedTestReq("a", "t1", schedTestBase))
	q.Push(schedTestReq("b", "t2", schedTestBase.Add(time.Second)))
	q.Push(schedTestReq("c", "t1", schedTestBase.Add(2*time.Second)))

	if !q.Remove("b") {
		t.Fatal("Remove(present) = false, want true")
	}
	if q.Remove("b") {
		t.Fatal("Remove(already removed) = true, want false")
	}
	if q.Remove("zzz") {
		t.Fatal("Remove(absent) = true, want false")
	}
	if got := q.Len(); got != 2 {
		t.Fatalf("Len() = %d, want 2", got)
	}
	if got := q.TenantCount("t2"); got != 0 {
		t.Fatalf("TenantCount(t2) after Remove = %d, want 0", got)
	}
	if got := q.TenantCount("t1"); got != 2 {
		t.Fatalf("TenantCount(t1) = %d, want 2", got)
	}

	// Removing the head must not strand the rest, and removing the tail must not
	// leave a dangling tail pointer that a later Push would chain onto.
	if st := q.Pop(); st == nil || st.ID != "a" {
		t.Fatalf("Pop() = %v, want a", st)
	}
	if st := q.Pop(); st == nil || st.ID != "c" {
		t.Fatalf("Pop() = %v, want c", st)
	}
	if st := q.Pop(); st != nil {
		t.Fatalf("Pop() = %v, want nil", st.ID)
	}
	if !q.Push(schedTestReq("d", "t3", schedTestBase)) {
		t.Fatal("Push after full drain = false")
	}
	if st := q.Pop(); st == nil || st.ID != "d" {
		t.Fatalf("Pop() = %v, want d", st)
	}
}

func TestQueueRemoveHeadUpdatesOldestWait(t *testing.T) {
	q := schedTestQueue(4, NewFIFOPolicy())
	q.Push(schedTestReq("old", "t1", schedTestBase))
	q.Push(schedTestReq("new", "t1", schedTestBase.Add(10*time.Second)))

	now := schedTestBase.Add(30 * time.Second)
	if got := q.OldestWait(now); got != 30*time.Second {
		t.Fatalf("OldestWait = %v, want 30s", got)
	}
	q.Remove("old")
	if got := q.OldestWait(now); got != 20*time.Second {
		t.Fatalf("OldestWait after removing head = %v, want 20s", got)
	}
}

func TestQueueDropExpired(t *testing.T) {
	q := schedTestQueue(8, NewFIFOPolicy())

	expired := []string{"e1", "e2"}
	for i, id := range expired {
		st := schedTestReq(id, "t1", schedTestBase.Add(time.Duration(i)*time.Second))
		st.Deadline = schedTestBase.Add(5 * time.Second)
		q.Push(st)
	}
	live := schedTestReq("live", "t1", schedTestBase.Add(2*time.Second))
	live.Deadline = schedTestBase.Add(time.Hour)
	q.Push(live)
	noDeadline := schedTestReq("forever", "t2", schedTestBase.Add(3*time.Second))
	q.Push(noDeadline)

	got := q.DropExpired(schedTestBase.Add(10 * time.Second))
	if len(got) != 2 {
		t.Fatalf("DropExpired returned %d requests, want 2", len(got))
	}
	seen := map[RequestID]bool{}
	for _, st := range got {
		seen[st.ID] = true
	}
	for _, id := range expired {
		if !seen[RequestID(id)] {
			t.Fatalf("DropExpired did not return %s; got %v", id, seen)
		}
	}

	if q.Len() != 2 {
		t.Fatalf("Len() after DropExpired = %d, want 2", q.Len())
	}
	// The dropped ones must be gone from every index, not merely unlinked.
	if q.Remove("e1") || q.Remove("e2") {
		t.Fatal("an expired request was still present after DropExpired")
	}
	if got := q.TenantCount("t1"); got != 1 {
		t.Fatalf("TenantCount(t1) = %d, want 1", got)
	}
	var left []RequestID
	q.Each(func(st *RequestState) bool { left = append(left, st.ID); return true })
	if len(left) != 2 || left[0] != "live" || left[1] != "forever" {
		t.Fatalf("remaining = %v, want [live forever]", left)
	}

	if more := q.DropExpired(schedTestBase.Add(10 * time.Second)); len(more) != 0 {
		t.Fatalf("second DropExpired returned %d, want 0", len(more))
	}
}

func TestQueueExpiredRequestIsNeverDispatched(t *testing.T) {
	// Serving a request whose deadline passed while it queued burns capacity
	// producing a result nobody wants, so the drop must happen before the pop.
	q := schedTestQueue(4, NewFIFOPolicy())
	dead := schedTestReq("dead", "t1", schedTestBase)
	dead.Deadline = schedTestBase.Add(time.Second)
	q.Push(dead)
	q.Push(schedTestReq("alive", "t1", schedTestBase.Add(time.Second)))

	now := schedTestBase.Add(5 * time.Second)
	dropped := q.DropExpired(now)
	if len(dropped) != 1 || dropped[0].ID != "dead" {
		t.Fatalf("DropExpired = %v, want [dead]", dropped)
	}
	st := q.Pop()
	if st == nil || st.ID != "alive" {
		t.Fatalf("Pop() = %v, want alive", st)
	}
}

func TestQueueOldestWait(t *testing.T) {
	q := schedTestQueue(8, NewFIFOPolicy())
	now := schedTestBase.Add(time.Minute)

	if got := q.OldestWait(now); got != 0 {
		t.Fatalf("OldestWait on empty = %v, want 0", got)
	}

	q.Push(schedTestReq("a", "t1", schedTestBase))
	if got := q.OldestWait(now); got != time.Minute {
		t.Fatalf("OldestWait with one = %v, want 1m", got)
	}

	q.Push(schedTestReq("b", "t1", schedTestBase.Add(30*time.Second)))
	q.Push(schedTestReq("c", "t1", schedTestBase.Add(45*time.Second)))
	if got := q.OldestWait(now); got != time.Minute {
		t.Fatalf("OldestWait with several = %v, want 1m (head of line)", got)
	}

	q.Pop()
	if got := q.OldestWait(now); got != 30*time.Second {
		t.Fatalf("OldestWait after popping head = %v, want 30s", got)
	}

	// A clock that runs behind a request's QueuedAt must not produce a negative
	// wait; the autoscaler compares this against thresholds and a negative value
	// would read as a healthy queue.
	if got := q.OldestWait(schedTestBase.Add(-time.Hour)); got != 0 {
		t.Fatalf("OldestWait with now before QueuedAt = %v, want 0", got)
	}
}

func TestQueueTenantCountsMaintainedIncrementally(t *testing.T) {
	q := schedTestQueue(16, NewFIFOPolicy())
	for i := 0; i < 5; i++ {
		q.Push(schedTestReq("a"+string(rune('0'+i)), "heavy", schedTestBase.Add(time.Duration(i)*time.Second)))
	}
	q.Push(schedTestReq("l0", "light", schedTestBase.Add(10*time.Second)))

	if got := q.TenantCount("heavy"); got != 5 {
		t.Fatalf("TenantCount(heavy) = %d, want 5", got)
	}
	if got := q.TenantCount("light"); got != 1 {
		t.Fatalf("TenantCount(light) = %d, want 1", got)
	}
	if got := q.TenantCount("absent"); got != 0 {
		t.Fatalf("TenantCount(absent) = %d, want 0", got)
	}

	counts := q.TenantCounts()
	counts["heavy"] = 999 // the returned map must be a copy
	if got := q.TenantCount("heavy"); got != 5 {
		t.Fatalf("TenantCounts returned a live map: TenantCount(heavy) = %d", got)
	}

	for i := 0; i < 5; i++ {
		q.Pop()
	}
	if _, ok := q.TenantCounts()["heavy"]; ok {
		t.Fatal("tenant with no queued requests still present in TenantCounts")
	}
}

func TestQueueConcurrentPushPop(t *testing.T) {
	const (
		producers = 8
		perProd   = 250
	)
	q := NewBoundedQueue(producers*perProd, NewFIFOPolicy())

	var pushed, popped, removed int64
	var wg sync.WaitGroup

	for p := 0; p < producers; p++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			for i := 0; i < perProd; i++ {
				id := RequestID(string(rune('A'+p)) + "-" + strconv.Itoa(i))
				st := schedTestReq(string(id), "t"+strconv.Itoa(p%3), schedTestBase.Add(time.Duration(i)*time.Millisecond))
				if q.Push(st) {
					atomic.AddInt64(&pushed, 1)
				}
			}
		}(p)
	}

	stop := make(chan struct{})
	for c := 0; c < 4; c++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				if st := q.Pop(); st != nil {
					atomic.AddInt64(&popped, 1)
				}
			}
		}()
	}

	// Concurrent readers and removers, since the gateway cancels while the
	// scheduler pops and the autoscaler reads OldestWait.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			_ = q.Len()
			_ = q.OldestWait(time.Now())
			_ = q.TenantCounts()
			q.Each(func(*RequestState) bool { return true })
			if q.Remove("A-0") {
				atomic.AddInt64(&removed, 1)
			}
		}
	}()

	time.Sleep(50 * time.Millisecond)
	close(stop)
	wg.Wait()

	for q.Pop() != nil {
		atomic.AddInt64(&popped, 1)
	}
	if got := atomic.LoadInt64(&pushed) - atomic.LoadInt64(&removed); got != atomic.LoadInt64(&popped) {
		t.Fatalf("pushed-removed = %d but popped = %d: the queue lost or duplicated requests",
			got, atomic.LoadInt64(&popped))
	}
	if q.Len() != 0 {
		t.Fatalf("Len() = %d after draining, want 0", q.Len())
	}
	if len(q.TenantCounts()) != 0 {
		t.Fatalf("TenantCounts = %v after draining, want empty", q.TenantCounts())
	}
}

func TestQueueZeroCapacityAlwaysRejects(t *testing.T) {
	q := schedTestQueue(0, NewFIFOPolicy())
	if q.Push(schedTestReq("a", "t1", schedTestBase)) {
		t.Fatal("Push into zero-capacity queue = true, want false")
	}
	if q.Len() != 0 || q.Pop() != nil {
		t.Fatal("zero-capacity queue is not empty")
	}
}

func TestQueueNilPolicyDefaultsToArrivalOrder(t *testing.T) {
	q := NewBoundedQueue(4, nil)
	q.SetClock(func() time.Time { return schedTestBase })
	q.Push(schedTestReq("a", "t1", schedTestBase))
	q.Push(schedTestReq("b", "t1", schedTestBase.Add(time.Second)))
	if st := q.Pop(); st == nil || st.ID != "a" {
		t.Fatalf("Pop() = %v, want a", st)
	}
	if q.Policy().Name() != "fifo" {
		t.Fatalf("default policy = %s, want fifo", q.Policy().Name())
	}
}

// A policy cannot mutate the queue it is selecting from: SchedulePolicy.Next
// receives a QueueSnapshot, which has no Push, Pop, Remove or DropExpired. That
// used to be a runtime rule — a mutating policy degraded to scheduling nothing —
// and it is now a compile-time one, which is strictly better: the failure mode
// cannot be written rather than being caught after the fact.
//
// This test therefore no longer needs a misbehaving policy. What it checks instead
// is the property that made the rule necessary: the view a policy reads reflects
// the queue exactly, so a policy cannot be handed stale contents.
func TestPolicyViewMatchesTheQueue(t *testing.T) {
	q := schedTestQueue(4, &inspectingPolicy{})
	q.Push(schedTestReq("a", "t1", schedTestBase))
	q.Push(schedTestReq("b", "t2", schedTestBase.Add(time.Second)))

	seen := q.Snapshot()
	if seen.Len() != 2 {
		t.Fatalf("snapshot Len() = %d, want 2", seen.Len())
	}
	if seen.TenantCount("t1") != 1 || seen.TenantCount("t2") != 1 {
		t.Fatalf("per-tenant counts wrong: t1=%d t2=%d",
			seen.TenantCount("t1"), seen.TenantCount("t2"))
	}

	var order []RequestID
	seen.Each(func(st *RequestState) bool {
		order = append(order, st.ID)
		return true
	})
	if len(order) != 2 || order[0] != "a" || order[1] != "b" {
		t.Fatalf("Each visited %v, want [a b] in arrival order", order)
	}

	// Removing through the queue is reflected in the view, since the view is the
	// queue rather than a copy taken at some earlier instant.
	q.Remove("a")
	if n := q.Snapshot().Len(); n != 1 {
		t.Fatalf("snapshot Len() = %d after Remove, want 1", n)
	}
}

type inspectingPolicy struct{}

func (inspectingPolicy) Name() string { return "inspecting" }
func (inspectingPolicy) Next(q QueueSnapshot, _ []*WorkerState, _ time.Time) *RequestState {
	var first *RequestState
	q.Each(func(st *RequestState) bool {
		first = st
		return false
	})
	return first
}

func TestQueuePopRejectsForeignRequest(t *testing.T) {
	q := schedTestQueue(4, &foreignPolicy{})
	q.Push(schedTestReq("a", "t1", schedTestBase))
	if st := q.Pop(); st != nil {
		t.Fatalf("Pop() = %v, want nil for a request the queue does not hold", st.ID)
	}
	if q.Len() != 1 {
		t.Fatalf("Len() = %d, want 1", q.Len())
	}
}

type foreignPolicy struct{}

func (foreignPolicy) Name() string { return "foreign" }
func (foreignPolicy) Next(QueueSnapshot, []*WorkerState, time.Time) *RequestState {
	return schedTestReq("not-in-queue", "t1", schedTestBase)
}
