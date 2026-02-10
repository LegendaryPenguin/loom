# results

Every table below is verbatim output. Reproduce with the commands in
[README.md](README.md).

**Machine.** Apple M4 Pro, 14 cores, 48 GiB RAM. macOS 26.6.2 (build 25G83),
darwin/arm64. Go 1.27.1 (`/opt/homebrew/bin/go`).

**What the machine does and does not affect.** Nothing in these tables is a wall-clock
measurement. The simulation runs in virtual time, so every latency, throughput and
percentile is a function of the seed, the config and the cost model only — the host
affects how long `go test` takes (about 12 seconds for the whole suite) and nothing
else. Two runs on the same machine, or on different machines, produce byte-identical
tables. `simulator.TestDeterminism` and `loadgen.TestDeterminism` assert this.

**Where the noise comes from, then.** Only the seed. Varying it varies the workload,
which is the sole source of variation in the suite, and that is what the seed-variation
section at the bottom reports. Where two configurations' ranges overlap there, the
comparison is called out as inconclusive rather than quoted as a win.

**Gates.** `go build ./...`, `go vet ./...` and `go test -race ./... -timeout 1200s`
all pass (race run: `bench` 239.8s, `go-control-plane` 3.9s, `loadgen` 2.2s,
`simulator` 3.3s). Without `-race` the whole `bench` suite is about 12 seconds.

---

## Contents

- [1. Batching: none vs static vs continuous](#1-batching-none-vs-static-vs-continuous)
- [2. Batching window sweep](#2-batching-window-sweep)
- [3. Scheduling policy](#3-scheduling-policy)
- [4. KV cache: paged vs contiguous](#4-kv-cache-paged-vs-contiguous)
- [5. Queue bound under overload](#5-queue-bound-under-overload)
- [6. Autoscaling on vs off](#6-autoscaling-on-vs-off)
- [7. Worker failure](#7-worker-failure)
- [8. Estimate-vs-actual cache drift](#8-estimate-vs-actual-cache-drift)
- [9. Tenant fairness](#9-tenant-fairness)
- [10. Seed variation](#10-seed-variation)
- [11. Contradictions and defects found](#11-contradictions-and-defects-found)

---

## 1. Batching: none vs static vs continuous

```
Batching strategy — offered load 30.0% of measured fleet capacity (1332.0 tok/s), FIFO, peak reservation
==========================================================================================================
strategy        served/s  goodput tok/s  reject  ttft p50  ttft p95  ttft p99  lat p50  lat p95  lat p99  lat max  mean conc  steps
--------        --------  -------------  ------  --------  --------  --------  -------  -------  -------  -------  ---------  -----
none (batch=1)  1.08      332.2          0.0%    14650.7   27506.1   28992.9   17637.9  30809.3  37718.0  51060.2  3.74       75099
static          1.22      375.5          0.0%    684.4     6027.8    7654.0    2786.9   18491.7  24165.2  29676.6  4.81       55140
continuous      1.26      388.0          0.0%    30.8      147.2     187.1     616.3    17225.9  25793.8  31263.9  4.70       51492

Batching strategy — offered load 70.0% of measured fleet capacity (1332.0 tok/s), FIFO, peak reservation
==========================================================================================================
strategy        served/s  goodput tok/s  reject  ttft p50  ttft p95  ttft p99  lat p50   lat p95   lat p99   lat max   mean conc  steps
--------        --------  -------------  ------  --------  --------  --------  -------   -------   -------   -------   ---------  -----
none (batch=1)  1.17      353.0          0.0%    154275.5  290247.5  312566.5  157929.4  295061.0  314676.1  323586.2  3.93       177767
static          2.69      814.3          0.0%    12138.2   18976.8   21395.1   15105.4   29722.2   36788.0   44289.7   13.77      58900
continuous      2.92      883.1          0.0%    37.7      175.7     227.8     1016.8    19702.7   29768.0   35594.4   12.98      54875

Batching strategy — offered load 95.0% of measured fleet capacity (1332.0 tok/s), FIFO, peak reservation
==========================================================================================================
strategy        served/s  goodput tok/s  reject  ttft p50  ttft p95  ttft p99  lat p50   lat p95   lat p99   lat max   mean conc  steps
--------        --------  -------------  ------  --------  --------  --------  -------   -------   -------   -------   ---------  -----
none (batch=1)  1.19      352.3          8.5%    212535.4  417888.5  424739.4  214986.4  418626.2  425630.7  436993.9  3.92       215847
static          2.93      870.3          0.0%    38662.4   68954.8   71296.4   44291.5   74720.7   84611.6   101543.4  15.08      73861
continuous      3.96      1177.5         0.0%    46.2      1526.1    2916.7    1222.0    20768.0   28160.6   33956.5   19.28      50277
```

**Reading.** At 95% offered load continuous batching serves 3.33x the requests of no
batching and 1.35x of static batching, at 3.34x and 1.35x the goodput. The headline is
not throughput, though — it is time-to-first-token, where continuous batching is
**4600x** better than no batching (46ms vs 212s) and **837x** better than static
(46ms vs 38.7s). That gap is structural, not a tuning difference: a static batch owns
its slots until its longest member finishes, so a request that arrives while a
1800-token generation is running waits for the generation rather than for capacity.
Continuous batching admits it on the next step.

The no-batching column saturates at 1.19 served/s regardless of offered load, which is
its capacity, and from 70% offered load onwards its latency is pure backlog.

End-to-end p99 is nearly flat across the three at 30% load (25.8s / 24.2s / 37.7s) and
that is the honest caveat: p99 end-to-end latency here is dominated by *service time* of
the long generations in the tail of the output-length distribution, not by scheduling.
Continuous batching does not make a 1800-token generation finish sooner. What it changes
is everything that queues behind it.

Note `mean conc` for continuous at 95% (19.28) versus static (15.08): the continuous
batcher keeps more sequences resident at the same offered load, because it backfills
slots that short sequences vacate.

## 2. Batching window sweep

```
Batching window sweep — static batcher, 10.0% offered load, FIFO, peak reservation
====================================================================================
window  served/s  goodput tok/s  reject  ttft p50  ttft p95  ttft p99  lat p50  lat p95  lat p99  mean batch  steps
------  --------  -------------  ------  --------  --------  --------  -------  -------  -------  ----------  -----
0s      0.38      118.1          0.0%    29.4      421.5     2791.3    713.3    17120.2  19313.2  1.01        22001
1ms     0.38      118.1          0.0%    33.4      432.9     2802.7    718.1    17122.8  19316.9  1.01        22001
2ms     0.38      118.1          0.0%    34.2      432.9     2802.7    718.1    17122.8  19316.9  1.01        22001
5ms     0.38      118.1          0.0%    37.7      432.9     2802.7    723.1    17127.8  19321.9  1.01        22001
10ms    0.38      118.1          0.0%    42.7      442.9     2812.7    728.1    17132.8  19326.9  1.01        22001
20ms    0.38      118.1          0.0%    52.7      462.9     2832.7    738.1    17142.8  19336.9  1.01        22001
50ms    0.39      120.5          0.0%    82.7      522.9     2892.7    768.1    17172.8  19366.9  1.01        22001

Batching window sweep — continuous batcher, 10.0% offered load, FIFO, peak reservation
========================================================================================
window  served/s  goodput tok/s  reject  ttft p50  ttft p95  ttft p99  lat p50  lat p95  lat p99  mean batch  steps
------  --------  -------------  ------  --------  --------  --------  -------  -------  -------  ----------  -----
0s      0.38      118.1          0.0%    27.7      173.8     189.7     642.1    17120.2  19341.4  1.01        21894
1ms     0.38      118.1          0.0%    27.2      133.8     188.1     607.0    17121.2  19818.4  1.08        20415
2ms     0.38      118.0          0.0%    28.2      134.8     189.1     609.3    17250.8  19894.0  1.16        19046
5ms     0.38      117.2          0.0%    30.9      137.0     193.5     599.1    17253.8  19881.5  1.24        17843
10ms    0.38      118.1          0.0%    32.3      143.4     188.8     692.7    17315.9  21486.7  1.54        14374
20ms    0.38      117.1          0.0%    37.3      144.2     197.3     704.8    17326.0  22168.4  1.78        12443
50ms    0.38      117.1          0.0%    63.9      169.2     194.8     702.7    17357.6  22155.2  1.78        12423

Batching window sweep — static batcher, 30.0% offered load, FIFO, peak reservation
====================================================================================
window  served/s  goodput tok/s  reject  ttft p50  ttft p95  ttft p99  lat p50  lat p95  lat p99  mean batch  steps
------  --------  -------------  ------  --------  --------  --------  -------  -------  -------  ----------  -----
0s      1.22      375.5          0.0%    684.4     6027.8    7654.0    2786.9   18491.7  24165.2  1.36        55140
1ms     1.23      379.3          0.0%    641.6     5420.5    7403.9    2946.3   17124.4  23078.1  1.34        56108
2ms     1.23      379.3          0.0%    641.6     5425.5    7403.9    2946.3   17124.4  23078.1  1.34        56108
5ms     1.23      379.3          0.0%    494.5     5440.5    7753.7    2831.9   16502.5  23083.1  1.35        55639
10ms    1.23      379.3          0.0%    490.7     5460.5    7768.7    2800.6   16517.5  23088.1  1.35        55630
20ms    1.24      380.3          0.0%    767.9     4557.4    6192.2    2703.7   16225.8  23098.1  1.35        55682
50ms    1.22      376.3          0.0%    562.3     5684.0    7814.7    2473.8   17434.0  23506.6  1.38        54547

Batching window sweep — continuous batcher, 30.0% offered load, FIFO, peak reservation
========================================================================================
window  served/s  goodput tok/s  reject  ttft p50  ttft p95  ttft p99  lat p50  lat p95  lat p99  mean batch  steps
------  --------  -------------  ------  --------  --------  --------  -------  -------  -------  ----------  -----
0s      1.26      388.0          0.0%    30.8      147.2     187.1     616.3    17225.9  25793.8  1.46        51492
1ms     1.24      381.4          0.0%    32.1      168.9     197.2     675.5    15725.9  21985.0  1.45        51950
2ms     1.24      382.4          0.0%    32.6      154.3     189.1     670.1    15546.8  19783.7  1.46        51330
5ms     1.24      380.4          0.0%    33.2      171.7     225.3     699.6    17801.7  25919.4  1.76        42703
10ms    1.19      365.0          0.0%    39.5      186.8     236.4     863.7    21488.6  27191.3  2.47        30344
20ms    1.22      376.5          0.0%    43.3      178.6     234.8     887.5    20197.9  23475.0  2.97        25264
50ms    1.24      380.7          0.0%    42.3      163.6     203.1     867.4    20360.8  23590.9  3.20        23483

Batching window sweep — static batcher, 70.0% offered load, FIFO, peak reservation
====================================================================================
window  served/s  goodput tok/s  reject  ttft p50  ttft p95  ttft p99  lat p50  lat p95  lat p99  mean batch  steps
------  --------  -------------  ------  --------  --------  --------  -------  -------  -------  ----------  -----
0s      2.69      814.3          0.0%    12138.2   18976.8   21395.1   15105.4  29722.2  36788.0  3.02        58900
1ms     2.69      814.5          0.0%    12152.2   18989.8   21413.0   15097.1  29657.7  36802.1  3.02        58900
2ms     2.69      814.5          0.0%    12165.2   19001.8   21430.0   15098.1  29657.7  36809.1  3.02        58900
5ms     2.69      814.4          0.0%    12182.8   19027.8   21471.0   15124.3  29667.7  36825.1  3.02        58900
10ms    2.65      801.9          0.0%    10653.6   16512.4   18346.4   13885.6  27546.1  35183.7  3.03        58582
20ms    2.65      801.6          0.0%    10719.0   16604.5   18436.4   13945.6  27626.1  35253.7  3.03        58582
50ms    2.64      800.5          0.0%    11000.9   16894.2   18724.9   14133.8  27866.1  35463.7  3.03        58582

Batching window sweep — continuous batcher, 70.0% offered load, FIFO, peak reservation
========================================================================================
window  served/s  goodput tok/s  reject  ttft p50  ttft p95  ttft p99  lat p50  lat p95  lat p99  mean batch  steps
------  --------  -------------  ------  --------  --------  --------  -------  -------  -------  ----------  -----
0s      2.92      883.1          0.0%    37.7      175.7     227.8     1016.8   19702.7  29768.0  3.24        54875
1ms     3.01      911.8          0.0%    38.7      170.8     203.8     912.4    17892.2  24277.8  3.21        55334
2ms     2.93      887.5          0.0%    37.3      172.9     223.7     970.6    19009.9  28281.6  3.24        54791
5ms     2.93      888.7          0.0%    38.0      173.8     224.0     910.6    19624.9  25989.4  3.31        53626
10ms    2.93      887.7          0.0%    38.6      170.9     232.6     1012.8   19775.1  25359.4  3.62        49107
20ms    2.91      881.9          0.0%    40.8      173.2     237.0     1096.4   19912.3  26674.6  3.91        45456
50ms    2.96      897.6          0.0%    41.6      165.9     213.0     1059.4   19607.8  25923.9  3.84        46353
```

**Reading.** The dial is real but it is much weaker than the framing in `batcher.go`
("the window is the whole latency-throughput dial") suggests, and it works differently
for the two batchers.

For the **continuous** batcher the window does what it says. At 10% load it raises mean
batch size from 1.01 to 1.78 and cuts forward passes by 43% (21894 → 12423) for 36ms of
TTFT p50 (27.7 → 63.9ms). At 30% load it raises mean batch from 1.46 to 3.20 and cuts
steps by 54% for 12ms of TTFT. The accelerator does less than half the work for the same
served load — which is a cost and energy result, not a throughput result, because served
throughput is flat (0.38, then 1.24–1.26) at these loads: the system was not
throughput-limited to begin with.

For the **static** batcher the window is nearly pure cost. Mean batch size is pinned at
1.01 across the whole 0–50ms sweep at 10% load, and steps do not move at all (22001
everywhere), while TTFT p50 rises from 29.4ms to 82.7ms — exactly the window. The reason
is arithmetic: at 10% load the mean inter-arrival gap is about 2.6 seconds, so a 50ms
window almost never catches a second request, and the static batcher holds its one
request for the full window before dispatching it. At 70% load the static batcher's
window stops mattering for the opposite reason — the backlog fills the batch to the cap
immediately, so the window never opens (mean batch 3.02, identical for every window).

So the window only pays when arrivals are closer together than the window and the batch
is not already full, and that is a narrow band. Where it does pay — the continuous
batcher at low load — what it buys is fewer forward passes, not more throughput.

The 10ms continuous row at 30% load is the one place the window costs end-to-end
latency measurably (p95 17.2s → 21.5s), which is consistent with larger batches making
each step slower for everyone in it.

## 3. Scheduling policy

```
Scheduling policy — 70.0% offered load, continuous batching, expected reservation (p50 length model)
======================================================================================================
policy           served  reject  served/s  lat p50  lat p95  lat p99  lat max  wait p50  wait p95  wait p99  wait MAX  preempt
------           ------  ------  --------  -------  -------  -------  -------  --------  --------  --------  --------  -------
fifo             886     0.0%    3.49      1094.9   20421.9  27824.0  37767.5  2.4       8.7       13.0      116.8     0
sjf (no aging)   886     0.0%    3.37      1067.1   20421.9  28961.8  37767.5  2.5       8.2       11.6      20.5      0
sjf (aging=1.0)  886     0.0%    3.37      1067.1   20421.9  28961.8  37767.5  2.5       8.2       11.6      20.5      0
fair-share       886     0.0%    3.49      1094.9   20421.9  27824.0  37767.5  2.4       8.7       13.0      116.8     0

Scheduling policy — 90.0% offered load, continuous batching, expected reservation (p50 length model)
======================================================================================================
policy           served  reject  served/s  lat p50  lat p95  lat p99  lat max  wait p50  wait p95  wait p99  wait MAX  preempt
------           ------  ------  --------  -------  -------  -------  -------  --------  --------  --------  --------  -------
fifo             1130    0.0%    4.32      1418.8   22769.2  30494.8  45806.4  3.5       1152.1    2454.6    2971.3    0
sjf (no aging)   1130    0.0%    4.26      1220.4   21961.2  30341.0  38579.0  3.5       369.2     1351.1    5550.1    0
sjf (aging=1.0)  1130    0.0%    4.44      1220.4   22496.0  30998.9  39563.2  3.5       394.9     1088.6    5436.2    0
fair-share       1130    0.0%    4.27      1363.9   23107.0  30998.9  42505.4  3.7       938.9     3004.2    4281.2    0

Scheduling policy — 110.0% offered load, continuous batching, expected reservation (p50 length model)
=======================================================================================================
policy           served  reject  served/s  lat p50  lat p95  lat p99  lat max   wait p50  wait p95  wait p99  wait MAX  preempt
------           ------  ------  --------  -------  -------  -------  -------   --------  --------  --------  --------  -------
fifo             1404    0.0%    5.25      7259.2   28157.3  36300.7  50851.8   4839.7    8276.7    10107.7   10653.9   0
sjf (no aging)   1404    0.0%    5.28      1643.8   29442.8  50687.8  102756.6  221.8     5876.3    25198.7   80951.2   0
sjf (aging=1.0)  1404    0.0%    5.13      1642.3   28909.0  49244.9  97491.4   217.5     7113.7    19743.9   68226.2   0
fair-share       1404    0.0%    5.39      6768.2   27805.1  36978.4  46870.4   3006.9    13010.8   14979.4   16189.1   0

Scheduling policy — 150.0% offered load, continuous batching, expected reservation (p50 length model)
=======================================================================================================
policy           served  reject  served/s  lat p50  lat p95   lat p99   lat max   wait p50  wait p95  wait p99  wait MAX  preempt
------           ------  ------  --------  -------  -------   -------   -------   --------  --------  --------  --------  -------
fifo             1898    0.0%    5.54      50915.8  85832.0   98523.8   118065.9  47718.4   77555.0   81710.1   82592.5   0
sjf (no aging)   1898    0.0%    5.46      1664.8   106046.6  233183.5  315301.1  287.1     84298.0   222234.5  295235.4  0
sjf (aging=1.0)  1898    0.0%    5.55      20593.1  88596.9   142056.0  203323.1  15272.7   70249.7   117460.3  169367.5  0
fair-share       1898    0.0%    5.53      46443.1  96346.5   105490.7  128012.9  42722.3   92230.3   100415.4  103961.4  0

Scheduling policy across 5 seeds — 110% offered load (min..max over seeds)
============================================================================
policy           lat p50         lat p99           wait p99         wait MAX          p99 wins  max wins
------           -------         -------           --------         --------          --------  --------
fifo             2369.8..7259.2  32560.1..36300.7  4113.3..10107.7  4590.6..10653.9   1         5
sjf (no aging)   1182.6..1643.8  33429.9..50687.8  4461.9..25198.7  12386.0..80951.2  0         0
sjf (aging=1.0)  1192.7..1642.3  35098.7..49244.9  4781.9..19743.9  12596.5..68226.2  0         0
fair-share       1913.3..6768.2  32332.8..36978.4  4046.8..14979.4  5050.4..16498.2   4         0
```

**I disagree with the policy author's recorded measurement.** `policy.go`'s comments
record the finding that the common claim "SJF better mean, FIFO better tail" is only half
true — that SJF won p99 too, and FIFO only won the *max*. That is not what I measure.

At 110% offered load, over five seeds:

- **SJF wins p50 decisively**, by 3.1x to 4.4x on end-to-end latency (1.18–1.64s vs
  2.37–7.26s) and by about 22x on queue wait at the single-seed point (222ms vs 4840ms).
  This half of the claim reproduces, strongly.
- **FIFO wins the max in 5 of 5 seeds**, as claimed. Queue-wait max 4.6–10.7s for FIFO
  against 12.4–81.0s for SJF — a 7.6x gap at the worst seed.
- **FIFO and fair-share also win p99.** FIFO took p99 on one seed and fair-share on four;
  **SJF took it on zero of five.** FIFO's end-to-end p99 range (32.6–36.3s) sits entirely
  below SJF's (33.4–50.7s), and the two ranges barely overlap at the bottom. On queue-wait
  p99 the separation is wider: FIFO 4.1–10.1s against SJF 4.5–25.2s.

So on this workload the folklore is not half true, it is true: SJF buys the median and
pays for it in the tail, in p99 as well as in the max. I think the difference is the
output-length distribution. The claim that SJF wins p99 needs a tail short enough that a
long request passed over a few times is still not the p99 observation. Here 5% of
requests generate 600–1800 tokens against a median near 45, so SJF's starved requests
*are* the p99, and no amount of good behaviour on the other 95% can hide them.

Two further things in the table:

- **Aging works, and is not sufficient.** At 150% load aging cuts SJF's queue-wait max
  from 295s to 169s (−43%) and p99 from 222s to 117s (−47%), while giving back some of
  the p50 win (1.66s → 20.6s). It bounds starvation, it does not remove it. `AgingRate`
  at 1.0 means a request's effective cost reaches zero only after it has waited its own
  service time, and for a 1800-token generation that is tens of seconds.
- **Below saturation the policy does not matter at all.** At 70% load all four policies
  produce identical served counts and near-identical percentiles (queue wait p50 2.4ms).
  There is nothing queued to choose between. Note that this table uses a deliberately
  tightened batch cap of 8 per worker precisely so that a queue forms below ρ = 1; with
  the standard cap of 24 the fleet behaves like a processor-sharing server and head-of-line
  wait stays in single-digit milliseconds right up to ρ = 1, with every policy identical.
  See finding F8 below — that is itself one of the more interesting results here.

Throughput is within 5% across all four policies at every load, so none of this is a
throughput tradeoff; it is purely about who waits.

## 4. KV cache: paged vs contiguous

```
KV cache policy — one 1024-block worker, batch cap 64, identical 200%-of-capacity workload
============================================================================================
cache / reservation            peak seqs  mean conc  served  reject  goodput tok/s  peak util  peak frag  preempt  lat p50  lat p95  lat p99
-------------------            ---------  ---------  ------  ------  -------------  ---------  ---------  -------  -------  -------  -------
contiguous + peak (best-fit)   18         4.79       238     38.3%   279.4          0.998      400        0        55920.2  89227.1  95367.9
contiguous + peak (first-fit)  17         4.40       227     41.2%   265.7          0.995      387        0        54370.2  98977.0  106717.2
paged + peak                   17         5.25       246     36.3%   298.5          1.000      0          0        50515.2  79654.2  92025.0
paged + expected (p50 model)   26         11.40      226     17.9%   205.5          1.000      0          396      34886.1  81536.8  101566.3

Concurrency ratio versus declared-ceiling inflation — one 1024-block worker, single tenant, 200% load
=======================================================================================================
inflation  contig+peak seqs  paged+expected seqs  ratio  contig reject  paged reject  contig goodput  paged goodput  paged preempt
---------  ----------------  -------------------  -----  -------------  ------------  --------------  -------------  -------------
1.0x       35                42                   1.20x  0.0%           0.0%          583.2           584.2          0
2.0x       24                42                   1.75x  3.5%           0.0%          528.0           584.2          0
4.0x       20                42                   2.10x  25.7%          0.0%          398.9           584.2          0
8.0x       22                42                   1.91x  58.2%          0.0%          226.3           583.5          0
16.0x      13                53                   4.08x  58.0%          1.5%          173.5           523.0          0
```

**I do not reproduce the cache author's 3.43x (24 vs 7) as a general number, and I think
the number is a property of the workload rather than of the code.** On the realistic
three-tenant mix I get **26 vs 18, a ratio of 1.44x**, not 3.43x. Neither of the author's
two figures appears: nothing sustained 7 sequences and nothing sustained 24 on the
contiguous side.

The inflation sweep is why. The ratio is roughly the factor by which the declared output
ceiling exceeds the real generation length, because that is exactly the quantity
contiguous allocation has to reserve and paging does not:

| declared-ceiling inflation | ratio |
|---|---|
| 1x | 1.20x |
| 2x | 1.75x |
| 4x | 2.10x |
| 8x | 1.91x |
| 16x | 4.08x |

3.43x is reachable, somewhere between 8x and 16x inflation. Quoting it without the
inflation it was measured at states a workload property as if it were a code property.
(The 8x row breaking monotonicity is expected noise: peak concurrency is a maximum over
the run, so it is one observation, and the seed-variation table below shows a 15..19 range
for the contiguous rows on a fixed configuration.)

**The decomposition changes the story, and in paging's favour.** Comparing
`paged + peak` against `contiguous + peak` isolates paging alone: 17 vs 18 peak sequences,
which is no win at all on concurrency. Almost the entire concurrency difference comes from
the *reservation mode*, not from the allocator. That is still an argument for paging —
`contiguous` cannot offer expected reservation, because it has no way to honour a later
request for more space — but it is a different argument than "paging raises concurrency".
Paging's direct contribution is `peak frag`: 400 blocks on the best-fit contiguous cache,
**39% of the whole 1024-block cache free and unusable because it is not adjacent**, against
a structural zero for paged. That is the real allocator-level result, and the one
`contiguous+peak` pays for in rejections even at identical concurrency.

**The operational headline is the rejection rate, not the concurrency.** At 8x inflation
contiguous+peak refuses 58.2% of offered load and delivers 226 tok/s; paged+expected
refuses 0.0% and delivers 583 tok/s — **2.6x the goodput**. At 4x it is 25.7% vs 0.0% and
1.46x. Concurrency is the mechanism; what an operator sees is load shedding.

**One result points the other way and I cannot explain it away.** On the realistic mix,
`paged + expected` has the best concurrency (26) and the best rejection rate (17.9%) but
the *worst* goodput (205.5 tok/s against 298.5 for `paged + peak`) and it preempts 396
times. Packing 26 sequences onto one worker makes every forward pass slower for all of
them, and preemption discards work. But see finding F4: the batcher does not reset
`GeneratedTokens` on preemption, so the simulated sequence resumes where it left off and
my numbers *understate* the real cost of those 396 preemptions. The honest conclusion is
that `paged + expected` with a p50 length model is over-subscribed on this fleet, and the
p90 model in §8 is the better operating point. The seed-variation table flags this
comparison as inconclusive anyway: paged+expected goodput ranges 205..385 against
contiguous's 255..325, and those overlap.

## 5. Queue bound under overload

```
Queue bound under sustained overload — 130.0% offered load, continuous batching, FIFO
=======================================================================================
queue cap        offered  served  rejected  reject rate  served/s  goodput tok/s  SERVED lat p50  p95      p99      max      wait p99  final depth  sim end
---------        -------  ------  --------  -----------  --------  -------------  --------------  ---      ---      ---      --------  -----------  -------
32               1238     1058    180       14.5%        5.17      1535.9         6610.5          28292.9  38345.0  48081.0  8373.6    0            3m25s
128              1238     1146    92        7.4%         5.23      1541.2         20083.2         38274.4  49536.2  64934.9  24832.6   0            3m39s
512              1238     1238    0         0.0%         5.11      1519.7         22729.5         46392.9  55253.6  68093.2  36617.3   0            4m2s
1e6 (unbounded)  1238     1238    0         0.0%         5.11      1519.7         22729.5         46392.9  55253.6  68093.2  36617.3   0            4m2s

Queue bound under sustained overload — 200.0% offered load, continuous batching, FIFO
=======================================================================================
queue cap        offered  served  rejected  reject rate  served/s  goodput tok/s  SERVED lat p50  p95       p99       max       wait p99  final depth  sim end
---------        -------  ------  --------  -----------  --------  -------------  --------------  ---       ---       ---       --------  -----------  -------
32               1898     1098    800       42.1%        5.36      1553.4         7067.4          28477.4   38788.6   50864.2   8665.3    0            3m25s
128              1898     1183    715       37.7%        5.22      1550.3         23399.2         45086.5   54159.2   65592.3   25547.9   0            3m47s
512              1898     1548    350       18.4%        5.31      1566.3         64002.6         100793.5  114345.1  135836.2  91139.1   0            4m52s
1e6 (unbounded)  1898     1898    0         0.0%         5.42      1570.5         80619.0         140167.8  151947.0  170393.7  139923.4  0            5m50s

Queue depth and head-of-line wait over time — 200% offered load for 180s
==========================================================================
t      bounded depth  bounded head wait  bounded rejected  unbounded depth  unbounded head wait
-      -------------  -----------------  ----------------  ---------------  -------------------
0s     0              0.0                0                 0                0.0
20s    51             5721.7             0                 51               5721.7
40s    126            15572.1            34                160              15572.1
1m0s   127            21758.5            132               256              23334.4
1m20s  123            20609.7            234               373              35468.8
1m40s  127            23326.6            335               466              44606.2
2m0s   128            24327.5            449               561              51470.0
2m20s  127            20925.6            539               665              61949.3
2m40s  127            23664.3            650               788              72814.0
3m0s   125            19319.3            715               857              80653.0
3m20s  16             22336.1            715               745              90233.4
3m40s  0              0.0                715               607              97957.0
```

**Reading.** This is the cleanest result in the suite.

At 200% offered load the goodput is **identical to within 1.3%** across every queue
capacity from 32 to one million (1550–1571 tok/s). The queue bound does not cost the
system any work. What it buys is where the cost of overload lands:

| queue cap | reject | served lat p50 | served lat p99 |
|---|---|---|---|
| 32 | 42.1% | 7.1s | 38.8s |
| 128 | 37.7% | 23.4s | 54.2s |
| 512 | 18.4% | 64.0s | 114.3s |
| 1e6 | 0.0% | 80.6s | 152.0s |

A cap of 32 refuses 42% of the offered load and serves the other 58% in a p50 of 7
seconds. The unbounded queue refuses nobody and serves everybody in a p50 of 81 seconds
and a p99 of 152 seconds. These are the same requests, the same fleet and the same amount
of work done; the only thing that changed is whether the system said no.

The time series shows the mechanism. The bounded queue reaches a plateau at depth ~127
within 40 seconds and holds it for the rest of the overload, so head-of-line wait
oscillates around 20–24 seconds and *stops growing*. The unbounded queue rises
monotonically to 857 and its head wait to 98 seconds and is still climbing when the
arrivals stop. A latency percentile from the bounded run is a property of the system. A
latency percentile from the unbounded run is a statement about how long I chose to run
the experiment — had I run 360 seconds of overload instead of 180, the bounded p99 would
be unchanged and the unbounded p99 would roughly double.

Note the 512-cap row at 130% load is byte-identical to the unbounded row: at that load the
backlog never reaches 512, so the bound is not a bound. A queue limit only does something
once it is reached, which is worth stating because it means the limit has to be set
against the overload you expect, not against the load you expect.

## 6. Autoscaling on vs off

```
Autoscaling under a ramp — cold start 20s, scale-up wait 1.5s, sustained 3s
=============================================================================
config             served  reject  goodput tok/s  wait p50  wait p95  wait p99  wait max  lat p99  workers end  peak workers  scale ups  scale downs  worker-seconds
------             ------  ------  -------------  --------  --------  --------  --------  -------  -----------  ------------  ---------  -----------  --------------
static fleet of 4  786     0.0%    894.0          3.0       455.2     1493.7    3001.2    26474.5  4            4             0          0            1080.0
autoscaled 1..4    786     0.0%    878.3          4320.1    14807.8   15936.8   16197.9   36748.3  3            4             5          3            800.0

Autoscaling under a spike — cold start 20s, scale-up wait 1.5s, sustained 3s
==============================================================================
config             served  reject  goodput tok/s  wait p50  wait p95  wait p99  wait max  lat p99  workers end  peak workers  scale ups  scale downs  worker-seconds
------             ------  ------  -------------  --------  --------  --------  --------  -------  -----------  ------------  ---------  -----------  --------------
static fleet of 4  1470    14.5%   1487.8         27462.8   45486.5   48775.4   49225.7   66815.6  4            4             0          0            1160.0
autoscaled 1..4    1402    18.5%   1396.3         31392.4   45859.3   47302.4   47802.5   69775.2  3            4             2          1            1070.0
```

```
Time series — ramp, autoscaled
================================
t      queue depth  head wait  serving  starting  draining  running  cache util  completed  rejected
-      -----------  ---------  -------  --------  --------  -------  ----------  ---------  --------
0s     0            0.0        1        0         0         0        0.000       0          0
10s    0            0.0        1        0         0         1        0.036       10         0
20s    0            0.0        1        0         0         3        0.205       14         0
30s    1            999.8      1        0         0         8        0.375       25         0
40s    0            0.0        1        0         0         6        0.384       35         0
50s    4            962.0      1        0         0         8        0.605       50         0
1m0s   9            3934.9     1        2         0         8        0.145       67         0
1m10s  27           8857.4     1        2         0         8        0.188       77         0
1m20s  8            2326.6     3        1         0         23       0.264       110        0
1m30s  0            0.0        3        1         0         18       0.277       145        0
1m40s  0            0.0        4        0         0         9        0.167       174        0
1m50s  0            0.0        4        0         0         18       0.235       198        0
2m0s   0            0.0        4        0         0         19       0.304       229        0
2m10s  0            0.0        3        0         1         21       0.311       261        0
2m20s  0            0.0        3        0         0         16       0.344       302        0
2m30s  0            0.0        3        0         0         12       0.160       336        0
2m40s  3            915.9      2        0         0         16       0.470       365        0
2m50s  24           4480.2     2        1         0         16       0.308       395        0
3m0s   53           10898.4    2        1         0         16       0.340       414        0
3m10s  63           14726.2    3        1         0         24       0.367       440        0
3m20s  66           15428.2    3        1         0         24       0.376       484        0
3m30s  44           10198.2    4        0         0         32       0.450       541        0
3m40s  39           7621.1     4        0         0         32       0.502       599        0
3m50s  28           6159.5     4        0         0         32       0.438       658        0
4m0s   28           3766.4     4        0         0         32       0.422       726        0
4m10s  0            0.0        4        0         0         27       0.529       759        0
4m20s  0            0.0        3        0         1         4        0.109       782        0

Time series — spike, autoscaled
=================================
t      queue depth  head wait  serving  starting  draining  running  cache util  completed  rejected
-      -----------  ---------  -------  --------  --------  -------  ----------  ---------  --------
0s     0            0.0        1        0         0         0        0.000       0          0
10s    14           3797.3     1        0         0         8        0.351       18         0
20s    63           13329.5    1        2         0         8        0.208       22         0
30s    103          17777.4    1        2         0         8        0.164       39         0
40s    69           16028.2    3        1         0         24       0.339       95         0
50s    84           18097.8    3        1         0         24       0.361       136        0
1m0s   69           14542.9    4        0         0         32       0.430       191        0
1m10s  256          16436.3    4        0         0         32       0.509       232        29
1m20s  256          19035.2    4        0         0         32       0.441       285        231
1m30s  242          25650.9    4        0         0         32       0.356       357        318
1m40s  245          34291.9    4        0         0         32       0.523       402        318
1m50s  249          42735.4    4        0         0         32       0.570       447        318
2m0s   236          45137.8    4        0         0         31       0.453       509        318
2m10s  226          46052.7    4        0         0         32       0.486       567        318
2m20s  220          42589.4    4        0         0         32       0.457       636        318
2m30s  221          41300.2    4        0         0         32       0.518       687        318
2m40s  209          37979.5    4        0         0         32       0.448       750        318
2m50s  215          39255.1    4        0         0         32       0.489       799        318
3m0s   212          40770.2    4        0         0         32       0.461       851        318
3m10s  180          35788.5    4        0         0         32       0.450       930        318
3m20s  182          35715.5    4        0         0         32       0.468       983        318
3m30s  169          33364.1    4        0         0         32       0.449       1046       318
3m40s  148          29667.3    4        0         0         32       0.447       1111       318
3m50s  133          25822.3    4        0         0         32       0.468       1184       318
4m0s   141          27999.8    4        0         0         32       0.558       1229       318
4m10s  86           26583.3    4        0         0         32       0.458       1284       318
4m20s  38           25571.9    4        0         0         32       0.510       1332       318
4m30s  0            0.0        4        0         0         17       0.305       1385       318
4m40s  0            0.0        3        0         0         6        0.206       1396       318
```

**Autoscaling does not improve latency here. It saves money.** On the ramp both
configurations serve all 786 requests with zero rejections, but the static fleet of four
holds queue wait p50 at 3.0ms against 4320ms for the autoscaled fleet, and p99 at 1.49s
against 15.9s. The autoscaled fleet paid for that with 800 worker-seconds against 1080,
a **26% saving**. On the spike the two are within 6% on every latency percentile and
within 4 points on rejection (14.5% static, 18.5% autoscaled) for an 8% worker-second
saving.

That is the correct shape of the result and it should not be surprising: the capacity an
autoscaler orders is not there when it orders it. The ramp time series shows the cost
directly. At 50s, one worker with a cache at 60% and a queue of 4. At 60s the scaler has
ordered two workers (`starting 2`) and queue depth is 9. By 1m10s the queue is at 27 and
head wait at 8.9s, with both ordered workers *still cold*. They land at 1m20s and the
queue collapses to 8 within ten seconds and to 0 by 1m30s. The entire 15.9s wait p99 is
earned in that one 20-second window, and it is exactly the cold start.

**The hysteresis works, and the scale-down is visibly premature once.** Five scale-ups
and three scale-downs across 4.5 minutes of a monotonically rising ramp is not
oscillation — a thresholded scaler sampling every 20ms would have taken thousands of
actions. But the drain at 2m10s (`draining 1`, going 4 → 3 → 2 by 2m40s) happens while
the ramp is still climbing, and it is immediately punished: queue depth goes 3 → 24 → 53
→ 63 over the next 40 seconds and the scaler has to buy the capacity back, paying another
cold start. `demandDownLocked` requires only an empty queue, head wait under 200ms and
cache utilisation under 35% for `SustainedFor` = 3s, and during a ramp those conditions
are met in the gaps between arrival clusters. On a rising load, three seconds of quiet is
not evidence of surplus. `MinTokensPerSecond` exists to suppress exactly this and is
unset in my config; a deployment on a ramping workload should set it.

**The spike result is partly an artifact of the queue bound and I would not quote it as a
scaling result.** From 1m10s the queue is pinned at its 256 cap and 318 requests are
rejected; after that both configurations are shedding, so what the table compares is two
load-shedding systems rather than two scaling systems. The spike factor of 5 is simply
past what four workers can absorb.

## 7. Worker failure

```
Worker failure — 4 workers at 85% load, w-2 killed at 45s, w-3 at 60s; heartbeat 500ms, dead after 3 misses (1.5s)
====================================================================================================================
config                                served  lost  retried  reaped workers  reaped reqs  blocks released  recovery  wait p99 whole run  wait max  lat p99  leak
------                                ------  ----  -------  --------------  -----------  ---------------  --------  ------------------  --------  -------  ----
no failure (baseline)                 807     0     0        0               0            0                -         2316.3              3095.5    28105.2  none
2 killed, no retry, no replacement    797     10    0        2               10           374              2m44.9s   46377.3             46803.6   60742.8  none
2 killed, retry, no replacement       807     0     10       2               10           390              2m55.7s   56872.0             57240.0   69944.4  none
2 killed, retry, autoscaler replaces  807     0     10       2               10           390              25s       9265.4              33751.1   30051.6  none
```

```
Worker failure timeline — kills at 45s and 60s, retry on
==========================================================
t      serving  running  queue depth  head wait  cache util  completed
-      -------  -------  -----------  ---------  ----------  ---------
0s     4        0        0            0.0        0.000       0
5s     4        5        0            0.0        0.042       12
10s    4        11       0            0.0        0.149       26
15s    4        15       0            0.0        0.238       37
20s    4        18       0            0.0        0.276       61
25s    4        22       0            0.0        0.315       86
30s    4        20       0            0.0        0.309       114
35s    4        23       0            0.0        0.310       136
40s    4        22       0            0.0        0.298       150
45s    3        16       0            0.0        0.295       171
50s    3        24       7            1186.0     0.466       187
55s    3        24       6            1801.8     0.460       207
1m0s   2        16       18           3597.2     0.582       221
1m5s   2        16       34           7418.6     0.400       235
1m10s  2        16       56           11351.1    0.484       240
1m15s  2        16       59           14588.9    0.498       250
1m20s  2        16       63           15494.0    0.469       266
1m25s  2        16       63           15956.6    0.478       286
1m30s  2        16       69           16896.6    0.480       302
1m35s  2        16       73           15971.5    0.444       325
1m40s  2        16       78           14488.8    0.375       350
1m45s  2        16       93           16548.9    0.500       364
1m50s  2        16       112          20028.2    0.463       370
1m55s  2        16       114          22334.5    0.542       389
2m0s   2        16       114          23288.3    0.463       408
2m5s   2        15       127          27131.4    0.526       418
2m10s  2        16       144          31013.1    0.511       422
2m15s  2        16       144          33776.9    0.507       438
2m20s  2        16       140          32258.0    0.445       464
2m25s  2        16       158          36479.5    0.471       473
2m30s  2        16       156          36884.8    0.480       495
2m35s  2        16       165          37392.5    0.470       509
2m40s  2        16       170          39482.7    0.432       523
2m45s  2        16       163          37371.1    0.393       552
2m50s  2        16       179          40347.6    0.476       564
2m55s  2        16       196          41644.9    0.467       577
3m0s   2        16       201          44058.4    0.558       590
3m5s   2        16       181          44325.8    0.393       610
3m10s  2        16       163          45492.9    0.347       628
3m15s  2        16       142          45392.1    0.401       649
3m20s  2        16       119          45269.0    0.387       672
3m25s  2        16       99           45488.3    0.498       692
3m30s  2        16       93           48644.8    0.674       698
3m35s  2        16       79           50866.0    0.552       712
3m40s  2        16       62           51958.1    0.545       729
3m45s  2        16       43           53525.3    0.407       748
3m50s  2        16       25           55738.5    0.414       766
3m55s  2        16       2            55391.2    0.475       789
4m0s   2        8        0            0.0        0.351       799
4m5s   2        6        0            0.0        0.341       801
4m10s  2        2        0            0.0        0.136       805
```

**No capacity is leaked.** This is the result the experiment exists for. Two kills, 10
in-flight requests destroyed with their workers, 374–390 cache blocks reported back by
`Registry.ReapDead`, and at the end of the run every cache holds zero blocks, every
per-sequence reservation ledger is empty, every tenant's quota charge is zero and no
request is still tracked. The `leak` column is the `LeakCheck` struct and it is zero in
every row. `DeadWorker.ReleasedBlocks` is what makes that possible — the driver cannot
release capacity it is not told about, and the interface tells it.

**Requests lost is a retry-policy choice, not a failure cost.** Ten requests were running
on the two dead workers. With retry off, all ten fail. With retry on,
`Registry.ClassifyRetry` returns `RetryElsewhere` (reason `worker_lost`) for all ten, they
are re-queued, and the served count returns exactly to the no-failure baseline of 807.
The work is redone, not the request lost — but note that 374 of those blocks and all the
tokens generated before the kill are thrown away, so the retried run's tail is *worse*
than the no-retry run's (latency p99 69.9s vs 60.7s): retrying ten requests cost
everybody else a little.

**Detection is fast; recovery is a different question.** The timeline shows head-of-line
wait going from 0 at 45s to 1186ms at 50s — the kill at 45s is detected within the
1.5-second dead threshold and the three-state health machine moves the worker out of
`Healthy()` on the first missed heartbeat. That part is cheap. What is not cheap is that
the fleet has permanently lost half its capacity: at 85% offered load on four workers,
two workers are at 170% of their load, so the queue rises monotonically to 201 and head
wait to 55 seconds, and the "recovery" of 2m45–2m56s in the table is not a recovery at
all, it is the backlog draining after the arrivals stopped. That is why the fourth row
exists.

**With replacement, recovery is the cold start and nothing else.** 25 seconds, against a
20-second cold start plus a 1.5-second detection threshold plus the scaler's 3-second
sustained-demand window — 24.5s of accounted-for delay out of 25. And it holds queue wait
p99 to 9.3s against 56.9s without replacement, 6.1x better. There is nothing to tune here:
the recovery time of a worker failure is the time to load a model, and the only lever is
to have spare capacity already warm.

## 8. Estimate-vs-actual cache drift

```
Estimate versus worker report — 4 workers at 85% offered load, continuous batching
====================================================================================
reservation mode  length model   reconciles  drift last  mean signed drift  mean |drift|  worst over  worst under  preempt  blocks over-reserved  blocks needed  waste  served  goodput tok/s
----------------  ------------   ----------  ----------  -----------------  ------------  ----------  -----------  -------  --------------------  -------------  -----  ------  -------------
peak              peak           39808       0           0.00               0.00          0           0            0        16949                 25695          66.0%  649     980.6
expected          quantile p50   40336       0           0.00               0.62          108         -5           0        637                   31262          2.0%   807     1193.2
expected          quantile p90   41372       0           0.00               0.08          50          -2           0        10020                 31262          32.1%  807     1163.3
expected          fraction 0.25  40336       0           0.00               0.32          63          -3           0        332                   31262          1.1%   807     1193.2
```

**PROTOCOL.md calls the estimate-report gap "the system's most interesting metric". Under
peak reservation it is identically zero, and that is a blind spot rather than a clean bill
of health.** Over 39808 reconciliations the drift is exactly 0 in every direction — mean,
absolute mean, worst over, worst under. The reason is structural: under `ReservePeak` the
cache really does allocate the worst case the controller asked for, so the controller's
estimate and the worker's report agree by construction. Meanwhile **66% of every reserved
block was never needed** (16949 wasted against 25695 actually spanned), and peak
reservation serves 649 requests against expected reservation's 807 at 22% less goodput.
So a dashboard watching drift would show a perfectly calibrated system that is wasting two
thirds of its memory. The waste column, not drift, is the metric for that failure mode.

**Under expected reservation the drift exists, is unbiased, and is small.** Mean signed
drift is 0.00 blocks for all three length models — the reconcile loop has no systematic
bias in either direction, which is the main thing one wants to know about it. Mean
absolute drift is under a block (0.08–0.62). Excursions are not negligible: the p50 model
reaches +108 blocks, about 10% of one worker's cache, in the over-reserved direction, and
−5 in the under-reserved direction. The asymmetry is the right shape — positive drift is
idle capacity, negative drift is the one that overcommits — and the dangerous direction
stays an order of magnitude smaller than the wasteful one.

**The quantile dial is the whole tradeoff and p90 is the sweet spot on this fleet.** p50
gives 2.0% waste; p90 gives 32.1% waste for 2.5% less goodput and a drift excursion
halved (+50 vs +108). `fraction 0.25` happens to land almost exactly on this
workload's real ratio and gives the best of both (1.1% waste, same goodput as p50) — but
that is a coincidence of this generator's 1.5x–8x per-tenant inflation, and
`FractionOfMax` is the one model that stays monotone in `MaxOutputTokens`, which means a
client that inflates its ceiling is charged for inflating it. The quantile models lose
that property, which the admission code's comments say plainly.

Zero preemptions in every row, which is why I treat §4's 396 preemptions as a symptom of
the single-worker configuration being over-subscribed rather than of expected reservation
in general.

## 9. Tenant fairness

```
Tenant fairness — fifo, no quota, 140% offered load, adversary sends 3000-7000 token prompts
==============================================================================================
tenant       weight  submitted  submitted tok  accepted  rejected  completed  served tok  served/submitted tok  share of served  lat p50  lat p95  lat p99
------       ------  ---------  -------------  --------  --------  ---------  ----------  --------------------  ---------------  -------  -------  -------
adversary    1.0     404        10170          314       90        313        7874        77.4%                 2.0%             42843.2  49269.7  49833.9
batch        1.0     404        315231         334       70        334        260250      82.6%                 67.4%            53499.5  78434.7  86466.4
chat         1.0     929        138013         745       184       745        108752      78.8%                 28.1%            43091.9  56305.8  79143.2
interactive  2.0     260        11290          214       46        214        9471        83.9%                 2.5%             42091.1  49431.1  50135.6

Tenant fairness — fifo + quota, 140% offered load, adversary sends 3000-7000 token prompts
============================================================================================
tenant       weight  submitted  submitted tok  accepted  rejected  completed  served tok  served/submitted tok  share of served  lat p50  lat p95  lat p99
------       ------  ---------  -------------  --------  --------  ---------  ----------  --------------------  ---------------  -------  -------  -------
adversary    1.0     404        10170          26        378       26         638         6.3%                  0.2%             3735.8   6208.2   6228.9
batch        1.0     404        315231         340       64        340        264768      84.0%                 63.8%            15809.1  34730.8  41954.9
chat         1.0     929        138013         929       0         929        138013      100.0%                33.3%            5039.2   16878.7  32476.0
interactive  2.0     260        11290          260       0         260        11290       100.0%                2.7%             4220.8   7417.9   8394.9

Tenant fairness — fair-share, no quota, 140% offered load, adversary sends 3000-7000 token prompts
====================================================================================================
tenant       weight  submitted  submitted tok  accepted  rejected  completed  served tok  served/submitted tok  share of served  lat p50  lat p95  lat p99
------       ------  ---------  -------------  --------  --------  ---------  ----------  --------------------  ---------------  -------  -------  -------
adversary    1.0     404        10170          322       82        322        8053        79.2%                 2.2%             1692.8   2998.0   3718.2
batch        1.0     404        315231         307       97        307        240739      76.4%                 64.4%            65740.3  97801.2  110571.3
chat         1.0     929        138013         751       178       751        116212      84.2%                 31.1%            61713.6  83718.2  94192.2
interactive  2.0     260        11290          209       51        209        8970        79.5%                 2.4%             1354.0   4038.4   5956.1

Tenant fairness — fair-share + quota, 140% offered load, adversary sends 3000-7000 token prompts
==================================================================================================
tenant       weight  submitted  submitted tok  accepted  rejected  completed  served tok  served/submitted tok  share of served  lat p50  lat p95  lat p99
------       ------  ---------  -------------  --------  --------  ---------  ----------  --------------------  ---------------  -------  -------  -------
adversary    1.0     404        10170          32        372       32         781         7.7%                  0.2%             1173.5   1817.8   2035.9
batch        1.0     404        315231         348       56        348        274809      87.2%                 64.9%            15440.0  33722.7  39494.2
chat         1.0     929        138013         923       6         923        136781      99.1%                 32.3%            15992.0  32094.1  44952.2
interactive  2.0     260        11290          260       0         260        11290       100.0%                2.7%             1120.4   2382.4   4482.1

**`admission.go`'s claim that quota is not fairness is exactly right, and this is the
measurement.** The two mechanisms do completely different things and only one of them
stops the adversary.

**Quota is what protects the other tenants.** With FIFO and no quota, every tenant gets
76–84% of its submitted work served and everyone's p50 latency is 42–53 seconds. Adding a
220-block ceiling on the adversary cuts it to 6.3% served and 26 accepted requests out of
404 — and chat and interactive go from 78.8% and 83.9% to **100.0% each, with p50 latency
falling 8.5x (43.1s → 5.0s) and 10x (42.1s → 4.2s)**. One tenant sending 404 requests of
3000–7000 prompt tokens each was consuming enough prefill compute and enough cache to
hold back the whole system, and a per-tenant *block* ceiling was what saw it. Note which
quota dimension did the work: the adversary's request rate is modest and its generations
are 8–40 tokens, so a `MaxConcurrentRequests` or tokens-per-second limit would have let
it straight through.

**Fair-share without quota does not protect anyone — it protects the adversary.** Compare
the no-quota rows: fair-share moves the adversary's p50 latency from 42.8s to **1.7s**, a
25x improvement, while `batch`'s p50 gets *worse* (53.5s → 65.7s) and `chat`'s gets worse
(43.1s → 61.7s). Served-work ratios barely move (adversary 77.4% → 79.2%). This is
start-time fair queueing doing precisely what it is designed to do: the adversary's
requests are *cheap in service time* (short generations), so virtual time advances slowly
for it and it is selected often. Fair-share equalizes service time, and the adversary's
attack is not service time — it is prefill compute and cache footprint, neither of which
`PolicyCostModel.ServiceNanos` weights heavily. A fair-share scheduler alone cannot stop
one tenant from occupying every block, and here it actively helps it.

**Together they are the best configuration.** Fair-share plus quota is the only row where
`interactive` — the weight-2 tenant — gets 100% of its work served at a p50 of 1.1s, the
lowest in the table, while `batch` gets its highest served ratio (87.2%) and `chat`
effectively all of its work (99.1%). The weights also finally show up: interactive's p50
(1.1s) sits well below chat's (16.0s) at equal-ish request size, which is the weight-2
share being honoured.

**The weight is not visible in share-of-served, and that is correct.** `batch` takes
63–67% of served tokens in every configuration regardless of weight or quota, because it
submitted 68% of the offered tokens. Share of served tracks share of submitted, not
weight. This is why the table reports served-against-submitted: a tenant's share of output
tells you what it asked for, not how it was treated.

## 10. Seed variation

```
Seed variation (5 seeds) — batching at 95% offered load
=========================================================
strategy        served/s    goodput tok/s  ttft p50            lat p99             reject
--------        --------    -------------  --------            -------             ------
none (batch=1)  1.19..1.34  352..355       190779.5..214785.5  379280.3..425630.7  4.9%..11.3%
static          2.93..3.39  828..894       20829.6..50247.7    63107.1..89229.5    0.0%..0.0%
continuous      3.93..4.30  1083..1178     36.8..63.9          27403.3..28805.8    0.0%..0.0%

Seed variation (5 seeds) — KV cache policy at 200% offered load
=================================================================
cache / reservation            peak seqs  goodput tok/s  reject
-------------------            ---------  -------------  ------
contiguous + peak (best-fit)   15..18     255..325       27.4%..48.6%
contiguous + peak (first-fit)  15..17     264..314       25.9%..46.9%
paged + peak                   15..19     285..339       22.9%..44.1%
paged + expected (p50 model)   26..35     205..385       3.7%..46.9%

Seed variation (5 seeds) — queue bound at 200% offered load
=============================================================
queue cap        reject        SERVED lat p50    SERVED lat p99      goodput tok/s
---------        ------        --------------    --------------      -------------
32               36.2%..42.3%  6219.3..7153.9    35952.9..38788.6    1550..1565
128              32.8%..38.2%  21493.1..23399.2  51176.4..54159.2    1550..1605
1e6 (unbounded)  0.0%..0.0%    63169.9..80619.0  134110.1..151947.0  1571..1614
```

**What survives seed variation.**

- **Batching.** Fully separated on every metric. Continuous's served/s range
  (3.93..4.30) does not touch static's (2.93..3.39) or no-batching's (1.19..1.34), and
  TTFT p50 ranges are three orders of magnitude apart with no overlap at all. Continuous
  batching's p99 range (27.4..28.8s) is also both tighter and lower than static's
  (63.1..89.2s), so the win is not a tail artifact.
- **Queue bound.** Fully separated. Served p50 ranges (6.2..7.2s / 21.5..23.4s /
  63.2..80.6s) do not overlap, rejection ranges do not overlap, and goodput ranges overlap
  almost exactly — which is the whole claim: shedding costs nothing in work done and buys
  a 10x improvement in served latency.
- **Cache concurrency.** Separated on the metric that matters to the claim:
  paged+expected 26..35 against every peak-reservation variant at 15..19. No overlap.

**What does not survive, and I will not claim it.**

- **Cache goodput.** paged+expected 205..385 overlaps contiguous+best-fit 255..325 and
  paged+peak 285..339. **This comparison is too noisy to conclude from at this sample
  size**, and that is consistent with §4's unexplained goodput inversion. The concurrency
  and rejection-rate results stand; the goodput result does not.
- **Cache rejection rate for paged+expected**: 3.7%..46.9% is a 12x range across seeds,
  far wider than any other row. The 200%-load single-worker configuration sits right at
  the edge where the p50 length model's under-reservation either holds or collapses into
  preemption, and which side it lands on is seed-dependent. Any single number from that
  cell is meaningless.
- **Contiguous best-fit vs first-fit.** 15..18 against 15..17 on concurrency and
  255..325 against 264..314 on goodput. Indistinguishable. `cache.go` argues best-fit is
  right for worst-case reservations and I cannot confirm or refute that from these runs;
  the fit policy appears not to matter on this workload.
- **Static batcher's TTFT p50**: 20.8..50.2s, a 2.4x range. Static batching's latency is
  dominated by which long generation happens to be holding a batch, so its tail is
  inherently seed-sensitive. The comparison against continuous survives anyway because the
  gap is 400x.

---

## 11. Contradictions and defects found

These are observations about `go-control-plane/` and `PROTOCOL.md`, reported rather than
fixed. Several of them changed numbers above, and where they did I say so.

> **Status: all of F1–F9 have since been fixed**, each with a test that was confirmed to
> fail before the fix and pass after. The findings are left here as written, because the
> record of what an adversarial measurement pass found in code that already passed 256
> tests is the useful part — rewriting them into a list of resolved tickets would delete it.
>
> Four residual limitations came out of fixing them, and they are real:
>
> * **F8a is enforced per-decision, not per-placement.** `Queue.Pop()` is worker-agnostic,
>   so the policy can decide *whether* to dispatch but not *where*. The stall now stops
>   choosing work that only the drain target could take, but nothing stops a caller then
>   placing that work on the target. Closing it needs `Pop`/`Next` to name the requesting
>   worker.
> * **F8b's slowdown baseline is only as good as the fleet's history.** It is a per-worker
>   high-water mark of per-sequence throughput, so a fleet saturated since startup learns
>   its degraded rate as uncontended and the term reads ~1.0. Mitigated by reporting
>   *unmeasurable* rather than healthy below a sample threshold, and by a configurable
>   override. It cannot be solved from `WorkerState` alone.
> * **F7's alignment does not reach `simulator`/`bench`.** They construct policies directly
>   and never call `SetReservationMode`, so their feasibility check still gates on the peak.
>   Only the real `Scheduler` path is aligned, which means the numbers in this document are
>   unchanged by that fix.
> * **F4 is priced but not fully propagated.** `batchFit` now uses `EstimateResumed`, so a
>   re-admitted victim gets a reservation covering prompt-plus-generated tokens. The
>   simulator still mirrors it with the original estimate, which under-bills a resumed
>   sequence in its cache-utilisation figure. No leak — release is by id — but the mirror
>   is slightly stale.
>
> One consequence for the numbers above: **the preemption costs in this document remain
> underestimates**, because they were measured before resumed sequences were priced. The
> direction of the error is known and stated at each affected result.

**F1 — `ScaleDecision.Target` is never populated.** `contracts.go` documents `Target` as
"the worker to drain on a scale-down", "carried here rather than left to the caller
because the autoscaler chose it from the state it just examined; a caller re-deriving
'least loaded' from newer state can pick a different worker than the one the decision was
reasoned about." `QueueWaitAutoscaler.actLocked` returns `ScaleDecision{Delta, Reason}`
and never sets `Target`. It *does* compute the victim — `a.leastLoaded(workers)` in
`Evaluate` — but only to interpolate into the reason string. So every scale-down decision
is unactionable as documented, and the caller is forced into exactly the re-derivation the
comment warns against. `simulator/sim.go:onAutoscale` falls back to
`ScaleDownTarget(allStates())` and the fallback is load-bearing: without it no worker is
ever drained. One-line fix at the `actLocked(now, -1, reason)` call site.

**F2 — `ContinuousBatcher.Step` can report the same sequence as preempted twice in one
step.** The decode loop sets `st.Phase = PhaseDecode` at the top of each iteration without
checking whether an earlier iteration already preempted `st` as another sequence's victim.
A resurrected victim can then be preempted again, appearing twice in the returned
`preempted` slice. A driver that un-bills a reservation per entry double-releases.
Observed: my concurrency gauge went negative until I deduplicated by id
(`simulator/step.go`) and switched the gauge from incremental to derived.

**F3 — worse, a preempted sequence can survive the same step.** Same root cause. If `st`
is preempted as a victim at iteration `i` (blocks released, pushed back onto the queue by
`preemptLocked`) and its own iteration `j > i` comes later, `st.Phase` is set back to
`PhaseDecode` and it ends up in `survivors`. It is now simultaneously in a worker's
running set and in the shared queue, so the next worker to `Pop` dispatches a second copy
of a request that is still running. I had to add a cross-worker reclaim
(`Sim.holderOf`) to keep the reservation ledger correct. It is rare but real — one
occurrence in a 1997-request run — and in a real deployment it means two workers
generating the same request's tokens from the same prefix.

**F4 — preemption does not discard the victim's tokens, though three comments say it
does.** `cache.go` ("the reservation tracks the sequence's actual length"),
`batcher.go` ("Preemption discards every token the victim has already generated"),
`admission.go` ("discarding the tokens it had already produced") and `PROTOCOL.md` all
describe preemption as losing progress. `ContinuousBatcher.preemptLocked` releases blocks,
zeroes `HeldBlocks` and sets `Phase = PhaseQueued`, but leaves `GeneratedTokens` untouched,
so the re-admitted sequence resumes where it stopped. **Every preemption cost in this
document is therefore an underestimate** — notably §4's 396 preemptions, which cost only a
re-prefill here rather than the 396 sequences' worth of regenerated decode they would cost
in reality.

**F5 — `Batcher.Step` and `FormBatch` retire sequences without reporting them.** `Step`'s
doc comment says it exists specifically so that a failure during a step can be reported,
because "a contract without this method leaves that path implementable only outside the
interface". But its first loop drops any already-finished member — releasing its cache and
setting `Phase = PhaseDone` — and returns it in *neither* `survivors` nor `preempted`.
`ContinuousBatcher.FormBatch` does the same. A driver that diffs only the return values
leaks those reservations. This is reachable precisely because of F4: a preempted request
keeps its `GeneratedTokens` and can be re-admitted already complete. I found it as a
369-block leak in the fairness run and had to reconcile against the previous running set in
both `retireStep` and `dispatch` (`Result.SilentRetirements` counts the occurrences).

**F6 — preemption does not reset `QueuedAt`, though `queue.go` implies it should.**
`queuedSince`'s comment justifies preferring `QueuedAt` over `ArrivalTime` on the grounds
that "a re-queued preempted request has an old `ArrivalTime` but a fresh wait".
`preemptLocked` never touches `QueuedAt`, and `BoundedQueue.Push` only sets it when zero,
so a preempted request's wait keeps accumulating from its original enqueue. The
consequences are not obviously wrong but they are not what the comment describes: the
request inherits a large aging credit under SJF, it dominates `OldestWait` and therefore
the autoscaler's primary signal, and it trips `SkipGrace` immediately. I left it as
implemented.

**F7 — `PolicyCostModel` always uses the peak, so the scheduler's feasibility test
contradicts the controller's reservation.** `policyBase.feasible` needs
`w.FreeBlocks() >= Cost.Blocks(st)`, and `Blocks` is unconditionally
`(PromptTokens + MaxOutputTokens) / TokensPerBlock` — the worst case. Under
`ReserveExpected` the batcher then allocates far less. The comment defends worst-case
feasibility as "the conservative direction", and for a single decision it is, but the
aggregate effect is that the scheduler refuses to dispatch at the concurrency the
admission controller is actually reserving for. The policy is gating on a number no
component is using.

**F8 — the `SkipGrace` convoy, and the autoscaler's signal going blind.** Two related
behaviours, both correct per their comments and both with large costs.

`selectBest` returns nil for the *whole fleet* as soon as any queued request has waited
`SkipGrace` (default 2s) and does not currently fit, which drains every worker to admit
one large request. On an early 400-block-per-worker fleet this cost about 14% of
throughput and 43 rejections that did not occur with the grace effectively disabled; I
moved the standard fleet to 1024 blocks so that `batch`-tenant requests (up to 225 blocks
peak) fit routinely, and the effect mostly disappeared. It is a real cliff, though: the
convoy appears when the largest request's peak reservation is a significant fraction of a
worker's cache, and nothing warns you that you are near it.

The second is more consequential. Under continuous batching with a loose batch cap, **the
queue stays empty while the system degrades.** At ρ = 0.9 on the standard fleet,
head-of-line wait is 2.8ms and queue depth is ~0, while end-to-end latency has risen
substantially — because 96 concurrent slots make the fleet behave like a processor-sharing
server, so an extra request slows everyone a little instead of waiting. `QueueWaitAutoscaler`
scales on head-of-line wait precisely because (per its own comment) "a fully utilized
system with an empty queue is working perfectly". Here that is false: the system is
saturating and the queue is empty, and the scaler will not react until slots run out, at
which point the transition is a cliff rather than a ramp. I had to tighten the batch cap to
8 in the policy experiment to produce a queue at all. This looks like the single most
important gap I found, and it is a design question rather than a bug: the signal needs a
slowdown term (tokens/sec per sequence, or inter-token latency) alongside queue wait.

**F9 — a cold start longer than the dead threshold reaps brand-new workers.** With a 20s
cold start and `3 x 500ms` to death, registering a worker at launch meant the registry
declared it dead 1.5s later and the reaper counted it as a failure; my first failure run
reported 4 reaped workers for 2 kills. `PROTOCOL.md` is consistent — `register` is sent
"once, immediately on connect", and a worker connects when it is ready — so this is an
integration trap rather than a control-plane bug, but it is a sharp one and nothing in the
registry would tell you it had happened other than an inflated failure count. The
simulator now registers on ready.

**F10 — the closed-form capacity estimate overstates the fleet by ~2.4x.** Not a defect in
`go-control-plane/`; a defect in my own first version of `simulator.NominalTokensPerSecond`,
recorded because it invalidated the first two rounds of every table here. Assuming a full
batch at all times ignores that the KV cache binds before the batch cap, that prefill is not
free, and F8's convoy. `bench.Capacity` now measures saturation instead.

**F11 — my own driver bug, for completeness.** The simulator's control-plane tick loop
rescheduled itself whenever the event heap was non-empty, and the loop's own worker-step
events kept it non-empty, so every run advanced the virtual clock to `MaxDuration`. Since
`Elapsed` was the denominator of every rate, throughput was understated by roughly 5x and
`MeanConcurrency` by the same factor, in a way that looked plausible. `Result.Elapsed` is
now measured to the last observable event and `Result.WallEnd` reports where the clock
actually stopped, so a truncated run is visible rather than silent.
