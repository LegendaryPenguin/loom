#!/usr/bin/env bash
#
# Shows the one result in this project that needed no caveats, and the two that did.
#
# Runs against the real control plane and a real Python worker over a socket, not
# the simulator — the simulator is for the load experiments, and a demo that only
# ever drives a simulation would be proving the wrong thing.

set -uo pipefail
set +m   # suppress job-control notices; the cleanup trap kills background curls
cd "$(dirname "$0")/.."

BLUE=$'\033[34m'; GREEN=$'\033[32m'; DIM=$'\033[2m'; RED=$'\033[31m'; OFF=$'\033[0m'
step() { printf '\n%s== %s%s\n' "$BLUE" "$1" "$OFF"; }
note() { printf '%s   %s%s\n' "$DIM" "$1" "$OFF"; }

RUN=$(mktemp -d)
HTTP=18080
WORKER=19090
cleanup() { jobs -p | xargs -r kill 2>/dev/null || true; rm -rf "$RUN"; }
trap cleanup EXIT

req() {
    # $1 = max_output_tokens, $2 = prompt
    curl -s -m 60 -H 'Content-Type: application/json' -H "X-Loom-Tenant: demo" \
        -d "{\"model\":\"tiny-gpt2\",\"prompt\":\"$2\",\"max_output_tokens\":$1}" \
        "http://127.0.0.1:$HTTP/v1/generate"
}

step "Step 1: build"
go build -o "$RUN/loom-control-plane" ./cmd/loom-control-plane

# loom_worker lives under python-worker/, so it is not importable from the repo root.
export PYTHONPATH="$PWD/python-worker${PYTHONPATH:+:$PYTHONPATH}"

step "Step 2: start the control plane and one worker"
note "the worker reports its OWN cache capacity at registration — the control plane"
note "does not know the device and a control plane that assumes will assume wrong"
"$RUN/loom-control-plane" -http-addr ":$HTTP" -worker-addr ":$WORKER" \
    -policy sjf -batcher continuous -batch-window 10ms \
    -queue-capacity 8 > "$RUN/cp.log" 2>&1 &
sleep 1
python3 -m loom_worker --addr "127.0.0.1:$WORKER" --model tiny-gpt2 \
    --simulated --blocks 64 --tokens-per-block 16 --seed 7 > "$RUN/w1.log" 2>&1 &

for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$HTTP/readyz" >/dev/null 2>&1; then break; fi
    sleep 0.25
done
if ! curl -sf "http://127.0.0.1:$HTTP/readyz" >/dev/null 2>&1; then
    printf '%s   worker never became ready; logs in %s%s\n' "$RED" "$RUN" "$OFF"; exit 1
fi
note "readyz is true — readiness means a healthy worker for the requested model exists,"
note "not merely that the process is up"

step "Step 3: one request, streaming"
note "proving: time-to-first-token is observable by a CLIENT, which is the only reason"
note "that metric is worth having"
# head closes the pipe, which SIGPIPEs curl; under pipefail that would abort the run.
{ curl -sN -m 60 -H 'Content-Type: application/json' -H 'X-Loom-Tenant: demo' \
    -d '{"model":"tiny-gpt2","prompt":"the cairn marks","max_output_tokens":24,"stream":true}' \
    "http://127.0.0.1:$HTTP/v1/generate" || true; } | head -6 || true

step "Step 4: a mixed burst — short requests behind a long one"
note "proving: continuous batching admits a queued request into capacity freed by a"
note "finished one, so a short request does NOT wait for the long one to end. Under"
note "static batching the long member holds the slot and everything behind it stalls."
# Collect only the request PIDs. A bare `wait` would also wait on the control plane
# and the worker, which never exit.
( req 160 "a long generation that holds its slot" > "$RUN/long.json" ) &
LONG_PID=$!
sleep 0.2
SHORT_PIDS=()
for i in 1 2 3 4 5 6; do
    ( req 16 "short $i" > "$RUN/short-$i.json" ) &
    SHORT_PIDS+=($!)
done
for pid in "${SHORT_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
DONE=$(grep -l '"text"\|"tokens"' "$RUN"/short-*.json 2>/dev/null | wc -l | tr -d ' ')
printf '   %s of 6 short requests completed\n' "$DONE"
if kill -0 "$LONG_PID" 2>/dev/null; then
    printf '   and the 160-token generation is STILL running — that is the result\n'
else
    printf '   (the long generation had already finished; raise its token count to see the effect)\n'
fi
wait "$LONG_PID" 2>/dev/null || true

step "Step 5: overload against a bounded queue"
note "proving: the queue REFUSES rather than growing. An unbounded queue would accept"
note "all of these and serve them in a p50 of 80s instead of 7s — measured in bench/."
for _ in $(seq 1 90); do
    curl -s -o /dev/null -m 20 \
        -H 'Content-Type: application/json' -H 'X-Loom-Tenant: flood' \
        -d '{"model":"tiny-gpt2","prompt":"flood","max_output_tokens":48}' \
        "http://127.0.0.1:$HTTP/v1/generate" >/dev/null 2>&1 &
done
# Read the counters while the burst is still resolving rather than waiting it out:
# the accept/reject split is decided at admission, which has already happened.
sleep 8
STATS=$(curl -s "http://127.0.0.1:$HTTP/metrics")
ACC=$(echo "$STATS" | grep -E '^loom_requests_accepted_total' | awk '{print $2}' | head -1)
printf '   accepted: %s\n' "${ACC:-0}"
printf '   rejected, by reason:\n'
echo "$STATS" | grep -E '^loom_requests_rejected_total\{' | grep -v ' 0$' | sed 's/^/     /' || true
REJ=$(echo "$STATS" | grep -E '^loom_requests_rejected_total\{' | awk '{s+=$2} END {print s+0}')
if [ "${REJ:-0}" -gt 0 ]; then
    printf '%s   %s refused rather than queued — the queue has a limit and enforces it%s\n' "$GREEN" "$REJ" "$OFF"
    note "each reason is counted separately, so overload is distinguishable from a bad request"
else
    printf '%s   nothing was refused: this burst fit. The step did not demonstrate shedding.%s\n' "$RED" "$OFF"
    note "raise the burst size or lower -queue-capacity to see it"
fi

step "Step 6: kill the worker"
note "proving: its cache reservations are RELEASED. A worker loss that leaks capacity is"
note "invisible until the system stops admitting anything at all."
pkill -f 'loom_worker' 2>/dev/null || true
sleep 3
curl -s "http://127.0.0.1:$HTTP/readyz" -o /dev/null -w '   readyz now: %{http_code} (503 expected, no healthy worker)\n'
curl -s "http://127.0.0.1:$HTTP/metrics" | grep -E '^loom_worker_failures_total|^loom_cache_blocks_used' | sed 's/^/   /'

step "Step 7: bring a worker back"
python3 -m loom_worker --addr "127.0.0.1:$WORKER" --model tiny-gpt2 \
    --simulated --blocks 64 --tokens-per-block 16 --seed 8 > "$RUN/w2.log" 2>&1 &
for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$HTTP/readyz" >/dev/null 2>&1; then break; fi
    sleep 0.25
done
req 16 "after recovery" > "$RUN/recover.json"
if grep -q '"text"' "$RUN/recover.json" 2>/dev/null; then
    printf '%s   served a request after full worker loss and replacement%s\n' "$GREEN" "$OFF"
else
    printf '%s   recovery request did not succeed; see %s%s\n' "$RED" "$RUN" "$OFF"
fi

step "Done"
note "What this demo does NOT show, because it cannot at this scale: the paged-versus-"
note "contiguous cache result, the scheduling-policy tails, and the autoscaler. Those"
note "need thousands of requests under controlled load — see bench/results.md, and read"
note "section 11, where the measurements disagree with the code's own comments."
