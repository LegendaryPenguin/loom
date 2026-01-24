package loom

import (
	"strconv"
	"sync"
	"testing"
	"time"
)

// regClock is a hand-advanced Clock. The health machine is defined purely in
// elapsed time, so a test that proved it by sleeping would be asserting on the OS
// scheduler rather than on the state machine, and would fail on a loaded machine
// for reasons that have nothing to do with the code.
type regClock struct {
	mu sync.Mutex
	t  time.Time
}

func newRegClock() *regClock {
	return &regClock{t: time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)}
}

func (c *regClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *regClock) advance(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

const regInterval = 500 * time.Millisecond

func newTestRegistry(clk Clock) *Registry {
	return NewRegistry(RegistryConfig{
		Clock:              clk,
		HeartbeatInterval:  regInterval,
		SuspectAfterMisses: 1,
		DeadAfterMisses:    3,
		Retry:              RetryPolicy{MaxAttempts: 3, MinDeadlineHeadroom: 250 * time.Millisecond},
	})
}

func regWorker(id WorkerID, blocks int) *WorkerState {
	return &WorkerState{
		ID:               id,
		Model:            "tiny-gpt2",
		TotalCacheBlocks: blocks,
		MaxBatchSize:     32,
	}
}

func TestRegistryRegisterHeartbeatGet(t *testing.T) {
	clk := newRegClock()
	r := newTestRegistry(clk)

	if err := r.Register(regWorker("w-1", 2048)); err != nil {
		t.Fatalf("Register: %v", err)
	}
	if err := r.Register(regWorker("w-1", 2048)); err != ErrWorkerExists {
		t.Fatalf("duplicate Register = %v, want ErrWorkerExists", err)
	}
	if err := r.Heartbeat("w-nope", 0, 0, 0); err != ErrUnknownWorker {
		t.Fatalf("Heartbeat on unknown = %v, want ErrUnknownWorker", err)
	}

	clk.advance(100 * time.Millisecond)
	if err := r.Heartbeat("w-1", 912, 7, 143.2); err != nil {
		t.Fatalf("Heartbeat: %v", err)
	}

	got, ok := r.Get("w-1")
	if !ok {
		t.Fatal("Get: worker missing")
	}
	if got.UsedCacheBlocks != 912 || got.ActiveSequences != 7 || got.RecentTokensPerSecond != 143.2 {
		t.Fatalf("heartbeat not applied: %+v", got)
	}
	if got.LastHeartbeat != clk.Now() {
		t.Fatalf("LastHeartbeat = %v, want %v", got.LastHeartbeat, clk.Now())
	}
	if got.FreeBlocks() != 2048-912 {
		t.Fatalf("FreeBlocks = %d, want %d", got.FreeBlocks(), 2048-912)
	}

	// Get must hand back a copy; otherwise the next heartbeat mutates a struct the
	// caller is already reading.
	got.UsedCacheBlocks = 0
	again, _ := r.Get("w-1")
	if again.UsedCacheBlocks != 912 {
		t.Fatal("Get returned an aliased pointer into registry state")
	}
}

// TestSlowColdStartIsNotReaped covers the integration trap: a 20s model load
// against a 1.5s dead threshold used to reap brand-new workers and count them as
// failures, so a fleet that lost two workers reported four.
func TestSlowColdStartIsNotReaped(t *testing.T) {
	clk := newRegClock()
	r := NewRegistry(RegistryConfig{
		Clock:              clk,
		HeartbeatInterval:  regInterval,
		SuspectAfterMisses: 1,
		DeadAfterMisses:    3,
		WarmupTimeout:      30 * time.Second,
	})
	if err := r.Register(regWorker("cold", 128)); err != nil {
		t.Fatal(err)
	}

	// Well past three missed heartbeats, which is where the reaper used to take it.
	clk.advance(20 * time.Second)
	if got := r.Warming(clk.Now()); len(got) != 1 || got[0] != "cold" {
		t.Fatalf("Warming = %v, want [cold]: a worker that has never reported is still starting", got)
	}
	if dead := r.ReapDead(clk.Now(), r.DeadAfter()); len(dead) != 0 {
		t.Fatalf("reaped %v during a cold start; that is a phantom worker failure", dead)
	}

	// Its first report ends the warm-up, and from then on its silence means it stopped.
	if err := r.Heartbeat("cold", 0, 0, 0); err != nil {
		t.Fatal(err)
	}
	if got := r.Warming(clk.Now()); len(got) != 0 {
		t.Fatalf("Warming = %v after the first heartbeat", got)
	}
	clk.advance(3 * regInterval)
	dead := r.ReapDead(clk.Now(), r.DeadAfter())
	if len(dead) != 1 || dead[0].ID != "cold" {
		t.Fatalf("reaped %v, want the worker that went silent after reporting", dead)
	}

	// A worker holding work is never warming: something dispatched to it, so its
	// reservations have to come back whether or not it ever heartbeated.
	clk2 := newRegClock()
	r2 := NewRegistry(RegistryConfig{
		Clock:              clk2,
		HeartbeatInterval:  regInterval,
		SuspectAfterMisses: 1,
		DeadAfterMisses:    3,
		WarmupTimeout:      30 * time.Second,
	})
	if err := r2.Register(regWorker("busy", 128)); err != nil {
		t.Fatal(err)
	}
	if err := r2.Assign("r-1", "busy", 7); err != nil {
		t.Fatal(err)
	}
	clk2.advance(3 * regInterval)
	dead = r2.ReapDead(clk2.Now(), r2.DeadAfter())
	if len(dead) != 1 || dead[0].ReleasedBlocks != 7 {
		t.Fatalf("reaped %v, want the silent worker and its 7 blocks back", dead)
	}

	// And the warm-up is bounded, so a loader that hangs forever is still reaped
	// rather than becoming invisible.
	clk3 := newRegClock()
	r3 := NewRegistry(RegistryConfig{
		Clock:              clk3,
		HeartbeatInterval:  regInterval,
		SuspectAfterMisses: 1,
		DeadAfterMisses:    3,
		WarmupTimeout:      10 * time.Second,
	})
	if err := r3.Register(regWorker("stuck", 128)); err != nil {
		t.Fatal(err)
	}
	clk3.advance(11 * time.Second)
	if dead := r3.ReapDead(clk3.Now(), r3.DeadAfter()); len(dead) != 1 {
		t.Fatalf("reaped %v, want the worker that never finished starting", dead)
	}
}

func TestRegistryHealthProgression(t *testing.T) {
	clk := newRegClock()
	r := newTestRegistry(clk)
	if err := r.Register(regWorker("w-1", 128)); err != nil {
		t.Fatal(err)
	}

	steps := []struct {
		advance time.Duration
		want    HealthState
		why     string
	}{
		{0, HealthHealthy, "just registered"},
		{regInterval - time.Millisecond, HealthHealthy, "within the interval"},
		{time.Millisecond, HealthSuspect, "one missed heartbeat"},
		{regInterval, HealthSuspect, "two missed heartbeats, still not dead"},
		{regInterval, HealthDead, "three missed heartbeats"},
	}
	for _, s := range steps {
		clk.advance(s.advance)
		got, ok := r.Health("w-1", clk.Now())
		if !ok {
			t.Fatal("worker vanished")
		}
		if got != s.want {
			t.Fatalf("%s: health = %v, want %v", s.why, got, s.want)
		}
	}

	// A suspect worker must be recoverable, which is the entire reason the state
	// exists: a GC pause is not a crash.
	clk2 := newRegClock()
	r2 := newTestRegistry(clk2)
	if err := r2.Register(regWorker("w-2", 128)); err != nil {
		t.Fatal(err)
	}
	clk2.advance(regInterval + time.Millisecond)
	if h, _ := r2.Health("w-2", clk2.Now()); h != HealthSuspect {
		t.Fatalf("health = %v, want suspect", h)
	}
	if got := r2.Healthy(clk2.Now()); len(got) != 0 {
		t.Fatalf("suspect worker is still eligible for new work: %d", len(got))
	}
	if err := r2.Heartbeat("w-2", 10, 1, 50); err != nil {
		t.Fatal(err)
	}
	if h, _ := r2.Health("w-2", clk2.Now()); h != HealthHealthy {
		t.Fatalf("after heartbeat health = %v, want healthy", h)
	}
	if got := r2.Healthy(clk2.Now()); len(got) != 1 {
		t.Fatalf("recovered worker not eligible again: %d", len(got))
	}
}

func TestRegistryHealthCounts(t *testing.T) {
	clk := newRegClock()
	r := newTestRegistry(clk)
	for _, id := range []WorkerID{"w-1", "w-2", "w-3", "w-4"} {
		if err := r.Register(regWorker(id, 128)); err != nil {
			t.Fatal(err)
		}
	}

	// Push every worker well past the dead threshold, then bring some back, so the
	// census is built from three genuinely different states rather than from the
	// registration timestamp.
	clk.advance(10 * regInterval)
	for _, id := range []WorkerID{"w-1", "w-2"} {
		if err := r.Heartbeat(id, 0, 0, 0); err != nil {
			t.Fatal(err)
		}
	}
	clk.advance(regInterval + time.Millisecond)
	if err := r.Heartbeat("w-1", 0, 0, 0); err != nil {
		t.Fatal(err)
	}

	counts := r.HealthCounts(clk.Now())
	if counts[HealthHealthy] != 1 || counts[HealthSuspect] != 1 || counts[HealthDead] != 2 {
		t.Fatalf("census = %v, want healthy=1 suspect=1 dead=2", counts)
	}
}

func TestRegistryReapDeadReturnsExactRequests(t *testing.T) {
	clk := newRegClock()
	r := newTestRegistry(clk)
	for _, id := range []WorkerID{"w-dead", "w-live"} {
		if err := r.Register(regWorker(id, 1024)); err != nil {
			t.Fatal(err)
		}
	}
	assign := func(req RequestID, w WorkerID, blocks int) {
		if err := r.Assign(req, w, blocks); err != nil {
			t.Fatalf("Assign(%s,%s): %v", req, w, err)
		}
	}
	assign("r-1", "w-dead", 12)
	assign("r-2", "w-dead", 7)
	assign("r-3", "w-live", 20)

	// At-least-once dispatch: the same assignment arriving twice is a no-op, not a
	// second copy of the request.
	assign("r-1", "w-dead", 12)
	if got := r.RequestsOn("w-dead"); len(got) != 2 {
		t.Fatalf("duplicate Assign created a second copy: %v", got)
	}
	if err := r.Assign("r-1", "w-live", 12); err != ErrRequestAssigned {
		t.Fatalf("cross-worker reassignment = %v, want ErrRequestAssigned", err)
	}

	clk.advance(3 * regInterval)
	if err := r.Heartbeat("w-live", 20, 1, 100); err != nil {
		t.Fatal(err)
	}

	reaped := r.ReapDead(clk.Now(), 0)
	if len(reaped) != 1 {
		t.Fatalf("reaped %d workers, want 1: %v", len(reaped), reaped)
	}
	if reaped[0].ID != "w-dead" {
		t.Fatalf("reaped %q, want w-dead", reaped[0].ID)
	}
	got := reaped[0].Requests
	want := []RequestID{"r-1", "r-2"}
	if len(got) != len(want) {
		t.Fatalf("reaped requests = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("reaped requests = %v, want %v", got, want)
		}
	}

	// The live worker keeps its request. A reaper that sweeps up more than the dead
	// worker's work fails requests that were running perfectly well.
	if on := r.RequestsOn("w-live"); len(on) != 1 || on[0] != "r-3" {
		t.Fatalf("live worker's requests = %v, want [r-3]", on)
	}
	if w, ok := r.WorkerOf("r-3"); !ok || w != "w-live" {
		t.Fatalf("WorkerOf(r-3) = %q,%v", w, ok)
	}
	if _, ok := r.WorkerOf("r-1"); ok {
		t.Fatal("reaped request still has an assignment")
	}

	// A second reap must not hand the same requests out again: re-queueing them
	// twice duplicates the work and double-releases their reservations.
	if again := r.ReapDead(clk.Now(), 0); len(again) != 0 {
		t.Fatalf("second ReapDead returned %v, want nothing", again)
	}

	// The dead worker stays visible for the metrics until explicitly forgotten.
	if len(r.All()) != 2 {
		t.Fatalf("All() = %d workers, want 2", len(r.All()))
	}
	r.Forget("w-dead")
	if len(r.All()) != 1 {
		t.Fatalf("after Forget, All() = %d, want 1", len(r.All()))
	}
}

// regCapacity is the reservation ledger the registry does not own. It stands in
// for whatever holds cache reservations so the test can assert that a worker loss
// actually returns capacity to it.
type regCapacity struct {
	total    int
	reserved int
}

func (c *regCapacity) reserve(blocks int) bool {
	if c.total-c.reserved < blocks {
		return false
	}
	c.reserved += blocks
	return true
}

func (c *regCapacity) release(blocks int) { c.reserved -= blocks }
func (c *regCapacity) free() int          { return c.total - c.reserved }

// TestRegistryWorkerLossDoesNotLeakCapacity is the test this component exists
// for. A worker loss that re-queues the requests but drops their reservations
// leaks capacity permanently, and the leak is invisible from any single request's
// point of view — every one of them succeeds — until the ledger says the fleet is
// full while the accelerators sit idle.
func TestRegistryWorkerLossDoesNotLeakCapacity(t *testing.T) {
	clk := newRegClock()
	r := newTestRegistry(clk)
	cap1, cap2 := 100, 100
	ledger := &regCapacity{total: cap1 + cap2}

	if err := r.Register(regWorker("w-1", cap1)); err != nil {
		t.Fatal(err)
	}
	if err := r.Register(regWorker("w-2", cap2)); err != nil {
		t.Fatal(err)
	}

	// Fill w-1 completely: four 25-block requests.
	for i, req := range []RequestID{"r-1", "r-2", "r-3", "r-4"} {
		if !ledger.reserve(25) {
			t.Fatalf("reservation %d refused with %d free", i, ledger.free())
		}
		if err := r.Assign(req, "w-1", 25); err != nil {
			t.Fatal(err)
		}
	}
	// And w-2, so the fleet is genuinely full and the next admission must fail.
	for _, req := range []RequestID{"r-5", "r-6", "r-7", "r-8"} {
		if !ledger.reserve(25) {
			t.Fatalf("reservation refused with %d free", ledger.free())
		}
		if err := r.Assign(req, "w-2", 25); err != nil {
			t.Fatal(err)
		}
	}
	if ledger.free() != 0 {
		t.Fatalf("free = %d, want 0 before the loss", ledger.free())
	}
	if ledger.reserve(25) {
		t.Fatal("a full fleet admitted a request")
	}

	// w-1 dies. w-2 keeps heartbeating.
	clk.advance(3 * regInterval)
	if err := r.Heartbeat("w-2", 100, 4, 120); err != nil {
		t.Fatal(err)
	}

	dead := r.ReapDead(clk.Now(), 0)
	if len(dead) != 1 || dead[0].ID != "w-1" {
		t.Fatalf("reaped = %+v, want only w-1", dead)
	}
	if n := len(dead[0].Requests); n != 4 {
		t.Fatalf("reaped %d requests, want 4", n)
	}
	if dead[0].ReleasedBlocks != 100 {
		t.Fatalf("ReleasedBlocks = %d, want 100 — the leak is exactly this number",
			dead[0].ReleasedBlocks)
	}

	// This is the step a naive implementation omits.
	ledger.release(dead[0].ReleasedBlocks)

	if ledger.free() != 100 {
		t.Fatalf("free after the loss = %d, want 100", ledger.free())
	}
	// And the system admits again, which is the symptom an operator would see.
	if !ledger.reserve(25) {
		t.Fatal("capacity leaked: the fleet still refuses admission after reaping")
	}
	if err := r.Assign("r-9", "w-2", 25); err != nil {
		t.Fatalf("re-dispatch of reaped work failed: %v", err)
	}

	// The dead worker's own accounting must be zeroed too, or a resurrected or
	// re-registered worker inherits phantom usage.
	w1, _ := r.Get("w-1")
	if w1.UsedCacheBlocks != 0 || w1.ActiveSequences != 0 {
		t.Fatalf("dead worker retains usage: %+v", w1)
	}
	if err := r.Heartbeat("w-1", 10, 1, 5); err != ErrWorkerDead {
		t.Fatalf("reaped worker heartbeat = %v, want ErrWorkerDead", err)
	}
}

func TestRegistryReleaseOnCompletion(t *testing.T) {
	clk := newRegClock()
	r := newTestRegistry(clk)
	if err := r.Register(regWorker("w-1", 256)); err != nil {
		t.Fatal(err)
	}
	if err := r.Assign("r-1", "w-1", 8); err != nil {
		t.Fatal(err)
	}
	// Decode grows the sequence past a block boundary.
	if err := r.UpdateAssignment("r-1", 11); err != nil {
		t.Fatal(err)
	}
	if err := r.UpdateAssignment("r-nope", 3); err != ErrUnknownAssignment {
		t.Fatalf("UpdateAssignment on unknown = %v", err)
	}

	w, blocks, ok := r.Release("r-1")
	if !ok || w != "w-1" || blocks != 11 {
		t.Fatalf("Release = %q,%d,%v; want w-1,11,true", w, blocks, ok)
	}
	if _, _, ok := r.Release("r-1"); ok {
		t.Fatal("double Release succeeded, which would double-credit capacity")
	}
	st, _ := r.Get("w-1")
	if st.ActiveSequences != 0 {
		t.Fatalf("ActiveSequences = %d, want 0", st.ActiveSequences)
	}
}

func TestRegistryDrainingLeavesHealthyStaysInAll(t *testing.T) {
	clk := newRegClock()
	r := newTestRegistry(clk)
	if err := r.Register(regWorker("w-1", 256)); err != nil {
		t.Fatal(err)
	}
	if err := r.Register(regWorker("w-2", 256)); err != nil {
		t.Fatal(err)
	}
	if err := r.Assign("r-1", "w-1", 9); err != nil {
		t.Fatal(err)
	}

	if err := r.SetDraining("w-1", true); err != nil {
		t.Fatal(err)
	}
	if err := r.SetDraining("w-nope", true); err != ErrUnknownWorker {
		t.Fatalf("SetDraining on unknown = %v", err)
	}

	healthy := r.Healthy(clk.Now())
	if len(healthy) != 1 || healthy[0].ID != "w-2" {
		t.Fatalf("Healthy = %v, want only w-2", healthy)
	}
	if len(r.All()) != 2 {
		t.Fatalf("All = %d, want 2 — a draining worker's in-flight work is still real", len(r.All()))
	}
	if on := r.RequestsOn("w-1"); len(on) != 1 {
		t.Fatalf("draining worker lost its in-flight request: %v", on)
	}

	// It keeps heartbeating, so it must not drift into suspect or get reaped.
	for i := 0; i < 5; i++ {
		clk.advance(regInterval - time.Millisecond)
		if err := r.Heartbeat("w-1", 9, 1, 40); err != nil {
			t.Fatalf("draining worker heartbeat rejected: %v", err)
		}
		if err := r.Heartbeat("w-2", 0, 0, 0); err != nil {
			t.Fatal(err)
		}
	}
	if h, _ := r.Health("w-1", clk.Now()); h != HealthHealthy {
		t.Fatalf("draining worker health = %v, want healthy", h)
	}
	if got := r.ReapDead(clk.Now(), 0); len(got) != 0 {
		t.Fatalf("heartbeating drainer was reaped: %v", got)
	}

	// A drainer must not be handed new work even by mistake at the registry level.
	if got := r.Healthy(clk.Now()); len(got) != 1 {
		t.Fatalf("Healthy = %d, want 1", len(got))
	}

	// Undraining restores eligibility, which is what a cancelled scale-down needs.
	if err := r.SetDraining("w-1", false); err != nil {
		t.Fatal(err)
	}
	if len(r.Healthy(clk.Now())) != 2 {
		t.Fatal("un-drained worker not eligible again")
	}
}

func TestRegistryDeregisterReturnsInFlight(t *testing.T) {
	clk := newRegClock()
	r := newTestRegistry(clk)
	if err := r.Register(regWorker("w-1", 256)); err != nil {
		t.Fatal(err)
	}
	if err := r.Assign("r-1", "w-1", 4); err != nil {
		t.Fatal(err)
	}
	if err := r.Assign("r-2", "w-1", 6); err != nil {
		t.Fatal(err)
	}
	got := r.Deregister("w-1")
	if len(got) != 2 || got[0] != "r-1" || got[1] != "r-2" {
		t.Fatalf("Deregister = %v, want [r-1 r-2]", got)
	}
	if _, ok := r.Get("w-1"); ok {
		t.Fatal("worker still present after Deregister")
	}
	if r.Deregister("w-1") != nil {
		t.Fatal("Deregister of an unknown worker returned requests")
	}
	if _, ok := r.WorkerOf("r-1"); ok {
		t.Fatal("assignment index still holds a deregistered worker's request")
	}
}

func TestRegistryConcurrentHeartbeats(t *testing.T) {
	clk := newRegClock()
	r := newTestRegistry(clk)
	const workers = 8
	ids := make([]WorkerID, 0, workers)
	for i := 0; i < workers; i++ {
		id := WorkerID("w-" + strconv.Itoa(i))
		ids = append(ids, id)
		if err := r.Register(regWorker(id, 1024)); err != nil {
			t.Fatal(err)
		}
	}

	var wg sync.WaitGroup
	stop := make(chan struct{})

	for _, id := range ids {
		id := id
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; ; i++ {
				select {
				case <-stop:
					return
				default:
				}
				_ = r.Heartbeat(id, i%512, i%16, float64(i))
				req := RequestID(string(id) + "-" + strconv.Itoa(i))
				if r.Assign(req, id, 3) == nil {
					r.Release(req)
				}
			}
		}()
	}
	// Readers and the reaper run concurrently with the writers, because the race
	// that matters is a scrape or a reap crossing a heartbeat.
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				_ = r.Healthy(clk.Now())
				_ = r.All()
				_ = r.HealthCounts(clk.Now())
				_ = r.ReapDead(clk.Now(), 10*time.Second)
				clk.advance(time.Microsecond)
			}
		}()
	}

	time.Sleep(50 * time.Millisecond)
	close(stop)
	wg.Wait()

	if len(r.All()) != workers {
		t.Fatalf("All = %d, want %d", len(r.All()), workers)
	}
}

func TestRegistryClassifyRetry(t *testing.T) {
	clk := newRegClock()
	r := newTestRegistry(clk)
	if err := r.Register(regWorker("w-1", 256)); err != nil {
		t.Fatal(err)
	}
	if err := r.Register(regWorker("w-2", 256)); err != nil {
		t.Fatal(err)
	}
	now := clk.Now()

	state := func(attempts int, deadline time.Time, worker WorkerID) *RequestState {
		return &RequestState{
			Request:  Request{ID: "r-1", Deadline: deadline},
			Attempts: attempts,
			Worker:   worker,
		}
	}

	cases := []struct {
		name       string
		st         *RequestState
		f          Failure
		now        time.Time
		wantAction RetryAction
		wantReason string
	}{
		{
			name:       "permanent failure is never retried",
			st:         state(0, time.Time{}, "w-1"),
			f:          Failure{Worker: "w-1", Transient: false},
			now:        now,
			wantAction: DoNotRetry,
			wantReason: "permanent_failure",
		},
		{
			name:       "transient on a healthy worker retries in place",
			st:         state(1, time.Time{}, "w-1"),
			f:          Failure{Worker: "w-1", Transient: true},
			now:        now,
			wantAction: RetrySame,
			wantReason: "transient_worker_healthy",
		},
		{
			name:       "worker loss retries elsewhere",
			st:         state(1, time.Time{}, "w-1"),
			f:          Failure{Worker: "w-1", WorkerLost: true},
			now:        now,
			wantAction: RetryElsewhere,
			wantReason: "worker_lost",
		},
		{
			name:       "attempt cap stops the loop",
			st:         state(3, time.Time{}, "w-1"),
			f:          Failure{Worker: "w-1", Transient: true},
			now:        now,
			wantAction: DoNotRetry,
			wantReason: "attempt_cap",
		},
		{
			name:       "a passed deadline is not retried",
			st:         state(0, now.Add(-time.Millisecond), "w-1"),
			f:          Failure{Worker: "w-1", Transient: true},
			now:        now,
			wantAction: DoNotRetry,
			wantReason: "deadline_passed",
		},
		{
			name:       "too little deadline left to be worth retrying",
			st:         state(0, now.Add(50*time.Millisecond), "w-1"),
			f:          Failure{Worker: "w-1", Transient: true},
			now:        now,
			wantAction: DoNotRetry,
			wantReason: "deadline_insufficient_headroom",
		},
		{
			name:       "enough deadline left still retries",
			st:         state(0, now.Add(5*time.Second), "w-1"),
			f:          Failure{Worker: "w-1", Transient: true},
			now:        now,
			wantAction: RetrySame,
			wantReason: "transient_worker_healthy",
		},
		{
			name:       "deadline is reported ahead of the attempt cap",
			st:         state(99, now.Add(-time.Second), "w-1"),
			f:          Failure{Worker: "w-1", Transient: true},
			now:        now,
			wantAction: DoNotRetry,
			wantReason: "deadline_passed",
		},
		{
			name:       "missing request state can only be retried elsewhere",
			st:         nil,
			f:          Failure{Worker: "w-1", Transient: true},
			now:        now,
			wantAction: RetryElsewhere,
			wantReason: "no_request_state",
		},
		{
			name:       "transient on an unknown worker goes elsewhere",
			st:         state(0, time.Time{}, "w-gone"),
			f:          Failure{Worker: "w-gone", Transient: true},
			now:        now,
			wantAction: RetryElsewhere,
			wantReason: "transient_worker_ineligible",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := r.ClassifyRetry(tc.st, tc.f, tc.now)
			if v.Action != tc.wantAction || v.Reason != tc.wantReason {
				t.Fatalf("got %v/%q, want %v/%q", v.Action, v.Reason, tc.wantAction, tc.wantReason)
			}
		})
	}

	// A drained worker is not somewhere to retry in place, even though it is alive.
	if err := r.SetDraining("w-1", true); err != nil {
		t.Fatal(err)
	}
	v := r.ClassifyRetry(state(0, time.Time{}, "w-1"), Failure{Worker: "w-1", Transient: true}, now)
	if v.Action != RetryElsewhere || v.Reason != "transient_worker_ineligible" {
		t.Fatalf("draining worker verdict = %v/%q", v.Action, v.Reason)
	}

	// A suspect worker likewise: not trusted with more work yet.
	if err := r.SetDraining("w-1", false); err != nil {
		t.Fatal(err)
	}
	late := now.Add(regInterval + time.Millisecond)
	v = r.ClassifyRetry(state(0, time.Time{}, "w-1"), Failure{Worker: "w-1", Transient: true}, late)
	if v.Action != RetryElsewhere {
		t.Fatalf("suspect worker verdict = %v, want elsewhere", v.Action)
	}

	if got := r.RetryPolicy().MaxAttempts; got != 3 {
		t.Fatalf("RetryPolicy().MaxAttempts = %d, want 3", got)
	}
}

func TestRegistryDefaultsFillIn(t *testing.T) {
	r := NewRegistry(RegistryConfig{})
	if r.DeadAfter() != 3*500*time.Millisecond {
		t.Fatalf("DeadAfter = %v, want 1.5s (three 500ms intervals from PROTOCOL.md)", r.DeadAfter())
	}
	if err := r.Register(&WorkerState{}); err != ErrUnknownWorker {
		t.Fatalf("Register of an id-less worker = %v", err)
	}
	if err := r.Register(nil); err != ErrUnknownWorker {
		t.Fatalf("Register(nil) = %v", err)
	}
	if _, ok := r.Health("absent", time.Now()); ok {
		t.Fatal("Health of an absent worker reported ok")
	}
	if err := r.Assign("r", "absent", 1); err != ErrUnknownWorker {
		t.Fatalf("Assign to absent worker = %v", err)
	}
}
