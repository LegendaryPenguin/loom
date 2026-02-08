package simulator

import (
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
)

// onWorkerStep is the heart of the simulation. One event means "the forward pass
// that was in flight has finished", so the handler retires that pass and then forms
// and schedules the next one.
func (s *Sim) onWorkerStep(id loom.WorkerID, now time.Time) {
	w := s.workerByID(id)
	if w == nil {
		// The worker was retired after this step was scheduled, which is routine on a
		// scale-down. Nothing to do: a retired worker's running set was already empty.
		return
	}
	w.stepping = false
	if !w.alive || !w.ready {
		return
	}

	if len(w.running) > 0 {
		s.retireStep(w, now)
	}
	s.dispatch(w, now)
}

// retireStep applies the effects of a completed forward pass: one token per live
// sequence, the cache growth that token may require, the preemptions that growth may
// force, and the completions it may cause.
func (s *Sim) retireStep(w *simWorker, now time.Time) {
	w.steps++

	prev := append([]*loom.RequestState(nil), w.running...)
	before := sumGenerated(w.running)
	// The third return value, retired, is the batcher reporting what it dropped. The
	// reconciliation below predates it and still covers the same ground.
	survivors, preempted, _ := w.batch.Step(w.running, s.queue, &w.cp, now)

	if sumGenerated(w.running) == before {
		// The batcher did not advance any token. ContinuousBatcher.Step does; StaticBatcher.Step
		// is a deliberate no-op, because a static batch is never re-formed mid-flight and so
		// has no growth failure to report. Either way a forward pass did happen, so the driver
		// produces the tokens the batcher did not. Detected by comparing progress rather than
		// by type-switching on the batcher, so a third Batcher implementation needs no change.
		survivors = s.advanceManually(w, now)
	}

	// Deduplicated by id: the batcher can report one sequence twice in a single step —
	// evicted as another sequence's victim, then evicted again on its own turn — and
	// un-billing a reservation twice is how an accounting ledger goes negative.
	seen := make(map[loom.RequestID]bool, len(preempted))
	for _, st := range preempted {
		if seen[st.ID] {
			continue
		}
		seen[st.ID] = true
		// The batcher has already released the blocks and pushed the request back onto the
		// queue. What it cannot do is un-bill the control plane's own reservation ledger,
		// because that ledger is the driver's.
		s.releaseCP(w, st.ID)
		_, _, _ = s.registry.Release(st.ID)
		if st.Phase == loom.PhaseDone {
			// The queue was full, so the preempted request could not be retried and the
			// batcher failed it visibly rather than losing it.
			s.adm.Complete(st.ID)
			s.res.Failed++
			delete(s.states, st.ID)
			delete(s.actual, st.ID)
		}
	}

	live := survivors[:0:0]
	var done []loom.RequestID
	for _, st := range survivors {
		if a := s.actual[st.ID]; a > 0 && st.GeneratedTokens >= a {
			// The real stopping point, which the control plane does not know. Marking the
			// phase here is how the batcher learns about it on its next call, through the
			// same batcherDone path a worker's `completed` message would drive.
			st.Phase = loom.PhaseDone
		}
		if st.Phase == loom.PhaseDone {
			s.complete(w, st, now)
			done = append(done, st.ID)
			continue
		}
		live = append(live, st)
	}

	// Sequences the batcher retired without telling anyone. ContinuousBatcher.Step
	// releases the cache for a member that was already finished when the step began and
	// then drops it from both return values, so a driver that trusts the return values
	// alone never learns the sequence left and leaks the reservation it was still
	// tracking. Reachable in practice because a preempted request keeps its
	// GeneratedTokens and so can be re-admitted already complete.
	accounted := make(map[loom.RequestID]bool, len(live)+len(seen)+len(done))
	for _, st := range live {
		accounted[st.ID] = true
	}
	for id := range seen {
		accounted[id] = true
	}
	for _, id := range done {
		accounted[id] = true
	}
	for _, st := range prev {
		if accounted[st.ID] {
			continue
		}
		accounted[st.ID] = true
		s.res.SilentRetirements++
		if s.states[st.ID] != nil {
			s.complete(w, st, now)
		} else {
			s.releaseCP(w, st.ID)
			_, _, _ = s.registry.Release(st.ID)
		}
	}
	w.running = live
	s.recount()

	// Reported tokens per second, which is what the heartbeat carries and the
	// autoscaler reads. Measured over the step that just ran rather than configured,
	// because a configured number cannot show a worker slowing down under a large batch.
	if d := now.Sub(w.lastStepStart); d > 0 && w.lastStepBatch > 0 {
		inst := float64(w.lastStepBatch) / d.Seconds()
		if w.lastTPS == 0 {
			w.lastTPS = inst
		} else {
			// Exponential decay, matching the "recent" in RecentTokensPerSecond. A plain
			// instantaneous value would make the autoscaler react to one unlucky step.
			w.lastTPS = 0.7*w.lastTPS + 0.3*inst
		}
	}
}

// advanceManually produces one token per live sequence for a batcher whose Step does
// not. It mirrors what a worker does: extend the sequence, and take another cache
// block when the sequence crosses a block boundary.
func (s *Sim) advanceManually(w *simWorker, now time.Time) []*loom.RequestState {
	tpb := w.spec.TokensPerBlock
	out := make([]*loom.RequestState, 0, len(w.running))
	for _, st := range w.running {
		if st.Phase == loom.PhaseDone {
			out = append(out, st)
			continue
		}
		if st.MaxOutputTokens > 0 && st.GeneratedTokens >= st.MaxOutputTokens {
			st.Phase = loom.PhaseDone
			out = append(out, st)
			continue
		}
		st.Phase = loom.PhaseDecode
		st.GeneratedTokens++
		if st.FirstToken.IsZero() {
			st.FirstToken = now
		}
		want := (st.PromptTokens + st.GeneratedTokens + tpb - 1) / tpb
		have := cacheBlocks(w.cache, st.ID)
		if want > have && !w.cache.Grow(st.ID) {
			// A static batch has no eviction path, so a growth failure is a hard failure
			// for this request rather than a preemption. With peak reservation it cannot
			// happen, which is the baseline's one advantage.
			st.Phase = loom.PhaseDone
			s.releaseCP(w, st.ID)
			w.cache.Release(st.ID)
			_, _, _ = s.registry.Release(st.ID)
			s.adm.Complete(st.ID)
			s.res.Failed++
			delete(s.states, st.ID)
			delete(s.actual, st.ID)
			continue
		}
		st.HeldBlocks = cacheBlocks(w.cache, st.ID)
		_ = s.registry.UpdateAssignment(st.ID, st.HeldBlocks)
		out = append(out, st)
	}
	return out
}

// complete finalizes a request that produced its last token.
func (s *Sim) complete(w *simWorker, st *loom.RequestState, now time.Time) {
	st.CompletedAt = now
	st.Phase = loom.PhaseDone
	s.lastActivity = now

	// Reserved-but-never-needed capacity, counted at the one instant it can be: the
	// sequence has stopped, so the blocks it actually spans is finally known, and the
	// difference against what it still holds is capacity nothing was ever going to use.
	// This is where peak-reservation waste shows up, and it is invisible to reconcile
	// drift, because the cache really did allocate what the control plane asked for.
	held := cacheBlocks(w.cache, st.ID)
	needed := (st.PromptTokens + st.GeneratedTokens + w.spec.TokensPerBlock - 1) / w.spec.TokensPerBlock
	if held > needed {
		s.res.ReservedBlockBeyondNeed += held - needed
	}
	s.res.NeededBlocks += needed

	// Released through the cache rather than left to the batcher's next FormBatch
	// so the capacity is available to the sequence admitted on this very step. Release
	// is idempotent, so the batcher releasing it again is harmless.
	w.cache.Release(st.ID)
	s.releaseCP(w, st.ID)
	_, _, _ = s.registry.Release(st.ID)
	s.adm.Complete(st.ID)

	s.res.Completed++
	s.res.OutputTokens += st.GeneratedTokens
	s.completedLat = append(s.completedLat, now.Sub(st.ArrivalTime))
	if !st.StartedAt.IsZero() {
		s.completedWait = append(s.completedWait, st.StartedAt.Sub(st.ArrivalTime))
	}
	if !st.FirstToken.IsZero() {
		s.completedTTFT = append(s.completedTTFT, st.FirstToken.Sub(st.ArrivalTime))
	}

	ts := s.tenant(st.Tenant)
	ts.Completed++
	ts.ServedTokens += st.GeneratedTokens
	ts.Latencies = append(ts.Latencies, now.Sub(st.ArrivalTime))

	delete(s.states, st.ID)
	delete(s.actual, st.ID)
}

// dispatch forms the next batch and schedules the forward pass that runs it.
func (s *Sim) dispatch(w *simWorker, now time.Time) {
	if !w.alive || !w.ready {
		return
	}

	wasRunning := make(map[loom.RequestID]bool, len(w.running))
	for _, st := range w.running {
		wasRunning[st.ID] = true
	}

	// A draining worker is handed a nil queue: the batchers have no notion of
	// draining, so this is how the driver stops one from taking new work while still
	// letting it finish and release what it already holds.
	q := loom.Queue(s.queue)
	if w.cp.Draining {
		q = nil
	}
	batch, _ := w.batch.FormBatch(w.running, q, &w.cp, now)

	prefillTokens := 0
	for _, st := range batch {
		if wasRunning[st.ID] {
			continue
		}
		if st.StartedAt.IsZero() {
			st.StartedAt = now
		}
		st.Worker = w.spec.ID
		st.Attempts++
		// The reservation the control plane believes it made. Recomputed from the
		// estimator with the same inputs batchFit used, because the control plane has no
		// other way to know: the cache is the worker's, and reading its real allocation
		// here would be the control plane consulting ground truth it does not have.
		// A request can arrive here while another worker still believes it is running it.
		// ContinuousBatcher.Step can evict a sequence — releasing its blocks and pushing it
		// back onto the queue — and then keep advancing it in the same step, so it is
		// simultaneously queued and in a running set; whichever worker pops it next dispatches
		// a second copy. Reclaiming the stale assignment is the only way the reservation
		// ledger stays correct, and the occurrences are counted rather than quietly fixed.
		if prevW := s.holderOf(st.ID); prevW != nil && prevW != w {
			s.res.DoubleDispatches++
			prevW.dropRunning(st.ID)
			s.releaseCP(prevW, st.ID)
			prevW.cache.Release(st.ID)
			prevW.cp.ActiveSequences = len(prevW.running)
		}

		need := s.est.ReserveBlocks(s.est.Estimate(&st.Request), s.cfg.Mode)
		if need < 1 {
			need = 1
		}
		w.cpReserved[st.ID] = need
		w.cp.UsedCacheBlocks += need
		s.clampCP(w)
		_ = s.registry.Assign(st.ID, w.spec.ID, need)
		prefillTokens += st.PromptTokens
	}
	// FormBatch can also drop a member of the running set — it evicts anything already
	// finished and releases its cache — and like Step it reports nothing when it does. A
	// driver that only diffs Step's return values still leaks those reservations, which is
	// reachable because a sequence can end up both running here and queued elsewhere.
	inBatch := make(map[loom.RequestID]bool, len(batch))
	for _, st := range batch {
		inBatch[st.ID] = true
	}
	for id := range wasRunning {
		if inBatch[id] {
			continue
		}
		s.res.SilentRetirements++
		if st := s.states[id]; st != nil {
			s.complete(w, st, now)
		} else {
			s.releaseCP(w, id)
			_, _, _ = s.registry.Release(id)
		}
	}

	w.running = batch
	w.cp.ActiveSequences = len(batch)
	s.recount()

	if len(batch) == 0 {
		if mw := w.batch.MaxWait(); mw > 0 && s.queue.Len() > 0 {
			// The batcher is holding the window open. Nothing else will wake it, so the
			// wakeup has to be scheduled for when the window closes.
			w.stepping = true
			s.clock.At(event{at: now.Add(mw), kind: evWorkerStep, worker: string(w.spec.ID)})
		}
		return
	}

	dur := s.cfg.Latency.StepDuration(len(batch), w.contextTokens(), prefillTokens, w.spec.SpeedFactor)
	if prefillTokens > 0 {
		w.prefillSteps++
	}
	w.tokens += uint64(len(batch))
	w.lastStepStart = now
	w.lastStepBatch = len(batch)
	w.stepping = true
	s.clock.At(event{at: now.Add(dur), kind: evWorkerStep, worker: string(w.spec.ID)})
}

// holderOf finds the worker whose reservation ledger still carries id.
func (s *Sim) holderOf(id loom.RequestID) *simWorker {
	for _, w := range s.workers {
		if _, ok := w.cpReserved[id]; ok {
			return w
		}
	}
	return nil
}

// releaseCP un-bills the control plane's reservation ledger for one sequence.
func (s *Sim) releaseCP(w *simWorker, id loom.RequestID) {
	n, ok := w.cpReserved[id]
	if !ok {
		return
	}
	delete(w.cpReserved, id)
	w.cp.UsedCacheBlocks -= n
	s.clampCP(w)
}

func sumGenerated(sts []*loom.RequestState) int {
	n := 0
	for _, st := range sts {
		n += st.GeneratedTokens
	}
	return n
}

// cacheBlocks reads the per-sequence block count both cache implementations expose.
// The KVCache contract itself only has aggregates, which is correct for a scheduler
// but not enough to tell over-reservation from real occupancy.
func cacheBlocks(c loom.KVCache, id loom.RequestID) int {
	if bc, ok := c.(interface{ Blocks(loom.RequestID) int }); ok {
		return bc.Blocks(id)
	}
	return 0
}
