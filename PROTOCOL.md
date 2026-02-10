# The control plane / worker protocol

The Go control plane and the Python workers are separate processes that share only
this document. Nothing is generated from a schema and nothing is shared by import,
because the point of the split is that the two halves can be reasoned about — and
replaced — independently.

The boundary is drawn where the problem changes. Deciding *what* runs is a
concurrency and resource-accounting problem; actually running a forward pass is a
Python and ML-runtime problem. Putting the scheduler in the worker would mean one
scheduler per worker, each with a partial view, which is how you get two workers
that each think they have capacity for the same request.

## Transport

Newline-delimited JSON over a TCP socket, one connection per worker, bidirectional.

JSON rather than a binary encoding: every message here is tiny except the token
payloads, the message rate is per-decode-step rather than per-token-per-sequence,
and being able to read a session with `nc` while debugging a scheduling decision
has repeatedly been worth more than the bytes. If a profile ever shows framing in
the top ten, the framing is the thing to change, not the boundary.

Every message is a single JSON object on one line with a `"type"` field. Unknown
fields must be ignored, and unknown `type` values must be logged and skipped
rather than treated as fatal — a newer control plane must be able to talk to an
older worker without taking it down.

## Worker → control plane

### `register`

Sent once, immediately on connect. The control plane will not dispatch to a worker
that has not registered.

```json
{
  "type": "register",
  "worker_id": "w-1",
  "model_id": "tiny-gpt2",
  "total_cache_blocks": 2048,
  "tokens_per_block": 16,
  "max_batch_size": 32,
  "device": "cpu"
}
```

`total_cache_blocks` is the worker's own measurement of its capacity, not a
configured guess. The worker knows its device memory and its model's per-token
cache footprint; the control plane does not, and a control plane that assumes will
eventually assume wrong and cause an out-of-memory kill mid-generation.

### `heartbeat`

Every `heartbeat_interval_ms` (default 500). Three consecutive misses and the
control plane declares the worker dead, reaps its requests, and releases their
reservations.

Note the interaction with `register`, which is a real trap: three misses is 1.5s
while loading a model takes tens of seconds, so a worker that registers at launch
rather than when it is ready is declared dead before it has served anything — and
the symptom is an inflated worker-failure count, not an obviously wrong one. The
rule above is "connect when ready, register on connect", and a worker that follows
it is never affected. The registry nonetheless treats a worker that has registered
and never heartbeated, and holds no requests, as warming rather than dead for a
configured warm-up window, so a slow start is visible as a warming worker instead
of as a failure.

```json
{
  "type": "heartbeat",
  "worker_id": "w-1",
  "used_cache_blocks": 912,
  "active_sequences": 7,
  "tokens_per_second": 143.2,
  "queue_depth": 0
}
```

`used_cache_blocks` is authoritative and **overwrites** the control plane's
estimate. The control plane predicts cache growth to make admission decisions, but
the worker is the only party that knows the truth, and a scheduler that trusts its
own prediction over a direct report will drift until it overcommits.

### `grow_request`

```json
{"type": "grow_request", "request_id": "r-7", "want_blocks": 4, "free_blocks": 112}
```

The sequence needs more cache than its reservation allows. This message exists
because the rule below — a worker must not exceed `reserved_blocks` — is
unenforceable without it: the only protocol-legal alternative is to preempt a
sequence that the worker has free capacity to serve, which turns an
under-estimate into preemption churn and hides the cause. The control plane
answers with `grant` or lets the sequence be preempted.

### `first_token`, `token`, `completed`, `failed`, `preempted`

```json
{"type": "first_token", "request_id": "r-7", "latency_ms": 184}
{"type": "token",       "request_id": "r-7", "token": " the", "generated": 12, "blocks": 3}
{"type": "completed",   "request_id": "r-7", "generated": 48, "stop_reason": "eos"}
{"type": "failed",      "request_id": "r-7", "error": "...", "retryable": true}
{"type": "preempted",   "request_id": "r-7", "generated": 31, "reason": "cache_exhausted"}
```

`first_token` is separate from `token` because time-to-first-token and
inter-token latency are different user-visible properties with different causes:
the first is dominated by queue wait plus prefill, the rest by decode throughput
and batch composition. A single latency number hides which one is bad.

`stop_reason` is one of `eos`, `length`, `cancelled`. `retryable` on `failed`
distinguishes a transient worker problem (retry elsewhere) from a request that
will fail identically everywhere (do not retry, and do not burn capacity proving
it) — an unqualified failure forces the control plane to guess, and guessing
"retryable" turns one bad request into an infinite loop.

`reason` on `preempted` is one of `cache_exhausted` (the worker genuinely ran out
of blocks — add workers) or `reservation_exceeded` (the control plane's estimate was
too low — fix the estimator). Those call for opposite responses, so a single
undifferentiated `preempted` leaves the operator unable to act on it.

`preempted` reports that the worker evicted a running sequence to free cache. The
victim's blocks are freed; the `generated` count it carries is kept, and the control
plane must re-queue the request with that progress intact. Resuming it is a
recomputation, not a continuation: the KV blocks are gone, so the worker re-prefills
the prompt **plus the tokens already generated** and then continues decoding from
there. That is the cost of a preemption — one prefill over a longer prompt, plus the
wait for a second admission — and it is why the metrics must count preemptions: a
system quietly preempting constantly looks healthy from the outside while spending a
growing share of its prefill budget re-reading its own output.

## Control plane → worker

### `dispatch`

```json
{
  "type": "dispatch",
  "batch": [
    {
      "request_id": "r-7",
      "prompt": "...",
      "max_output_tokens": 128,
      "reserved_blocks": 12
    }
  ]
}
```

`reserved_blocks` is the control plane's allocation. The worker must not exceed it
without asking, which is what keeps two schedulers' decisions from colliding on
one device.

### `cancel`

```json
{"type": "cancel", "request_id": "r-7"}
```

Must free the sequence's blocks promptly and report `completed` with
`stop_reason: "cancelled"`. Prompt matters: a cancelled request whose memory is
only reclaimed at natural completion defeats the purpose of cancelling.

### `grant`

```json
{"type": "grant", "request_id": "r-7", "extra_blocks": 4}
```

Permission to exceed the original reservation. Sent in response to a `grow_request`.

### `drain`

```json
{"type": "drain"}
```

Finish in-flight work, accept nothing new, keep heartbeating. The worker exits
when its running set empties. Draining rather than killing is how scale-down
avoids failing requests that were already most of the way done.

## The rules that are easy to get wrong

**At-least-once dispatch.** A `dispatch` whose acknowledgement is lost may be
redelivered. A worker that already holds `request_id` must treat the duplicate as
a no-op rather than starting a second copy — and the control plane must not assume
exactly-once, because it cannot have it over a socket that can drop.

**Cache accounting is the worker's truth and the control plane's estimate.** The
control plane plans with `expected_cache_blocks`, a prediction from prompt length
and typical output length, and reconciles against `used_cache_blocks` on every
heartbeat. A large positive gap means capacity is being wasted on
over-reservation; negative means the next admission may overcommit.

One caveat, measured rather than assumed: **this metric is structurally blind
under peak reservation.** If the control plane reserves each request's worst case,
the worker can never exceed the reservation, so drift is identically zero — it was
zero across 39,808 reconciles in a run where 66% of all reserved blocks were never
needed. A metric that reads perfect while two thirds of the cache is wasted is
worse than no metric, because it invites the conclusion that nothing is wrong.
Drift is only informative under expected reservation, where it came out unbiased
(mean signed 0.00 blocks) with excursions of +108/−5. Waste needs its own
measurement: reserved-but-never-touched blocks.

**A worker serves one model.** Loading a second model into the same device
invalidates every cache-capacity number the worker reported at registration.
Switching models means draining and re-registering.

**Backpressure is explicit, and distinguishable from a fault.** A worker whose
dispatch channel is full replies `{"type": "rejected", "request_id": "...",
"reason": "backpressure"}` rather than buffering the batch or reporting a failure.
Modelling overload as `failed(retryable=true)` would make it indistinguishable from
a genuine worker fault, and those need to be counted separately — one means the
fleet is too small, the other means a worker is sick. Buffering at the worker hides queue depth from the
scheduler, which is the one component that needs to see it to make a decision.
