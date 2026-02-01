// Command loom-control-plane runs the inference control plane: the HTTP API
// clients submit to, and the socket workers connect back on.
//
// Every flag maps to a field of loom.Config, so an experiment is a command line
// rather than a build.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	loom "github.com/LegendaryPenguin/loom/go-control-plane"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	log.SetPrefix("loom: ")

	cfg, err := parseFlags(os.Args[1:])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	srv, err := loom.NewServer(cfg)
	if err != nil {
		log.Fatal(err)
	}

	// SIGTERM is the deploy signal, and the whole point of the graceful path is that
	// a deploy does not fail the generations that are already most of the way done.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := srv.Start(); err != nil {
		log.Fatal(err)
	}
	log.Printf("listening: http=%s workers=%s policy=%s batcher=%s reservation=%s",
		srv.HTTPAddr(), srv.WorkerAddr(), cfg.Policy, cfg.Batcher, cfg.ReservationMode)

	<-ctx.Done()
	log.Print("draining")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownGrace)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("shutdown: %v", err)
		os.Exit(1)
	}
	log.Print("stopped")
}

func parseFlags(args []string) (loom.Config, error) {
	fs := flag.NewFlagSet("loom-control-plane", flag.ContinueOnError)
	var cfg loom.Config

	fs.StringVar(&cfg.HTTPAddr, "http-addr", ":8080", "address for the client HTTP API")
	fs.StringVar(&cfg.WorkerAddr, "worker-addr", ":9090", "address workers connect to")
	fs.StringVar(&cfg.Namespace, "metrics-namespace", "loom", "prefix for metric series")

	fs.IntVar(&cfg.QueueCapacity, "queue-capacity", 1024,
		"hard queue bound; 0 for effectively unbounded, which exists to be measured, not deployed")
	fs.StringVar(&cfg.Policy, "policy", loom.PolicyFIFO, "scheduling policy: fifo, sjf, fair-share")
	fs.Float64Var(&cfg.SJFAgingRate, "sjf-aging-rate", 1.0,
		"how fast a waiting request's effective cost decays under sjf; 0 starves long requests")

	fs.IntVar(&cfg.TokensPerBlock, "tokens-per-block", 16, "KV-cache block granularity in tokens")
	fs.BoolVar(&cfg.ContiguousCache, "contiguous-cache", false,
		"use the naive contiguous allocator instead of paged blocks")
	fs.Float64Var(&cfg.ExpectedOutputFraction, "expected-output-fraction", 0.5,
		"expected output length as a fraction of the declared ceiling")

	fs.StringVar(&cfg.Batcher, "batcher", loom.BatcherContinuous, "batching mode: continuous, static")
	fs.DurationVar(&cfg.BatchWindow, "batch-window", 0,
		"how long to hold a request hoping for company; the latency/throughput dial")

	fs.StringVar(&cfg.TenantHeader, "tenant-header", "X-Loom-Tenant", "header carrying the tenant id")
	fs.Int64Var(&cfg.MaxBodyBytes, "max-body-bytes", 1<<20, "maximum request body size")
	fs.DurationVar(&cfg.RequestTimeout, "request-timeout", 5*time.Minute,
		"ceiling on a buffered response")

	fs.DurationVar(&cfg.HeartbeatInterval, "heartbeat-interval", 500*time.Millisecond,
		"worker heartbeat period from PROTOCOL.md")
	fs.IntVar(&cfg.SuspectAfterMisses, "suspect-after-misses", 1,
		"missed heartbeats before a worker stops receiving new work")
	fs.IntVar(&cfg.DeadAfterMisses, "dead-after-misses", 3,
		"missed heartbeats before a worker is reaped and its reservations released")
	fs.IntVar(&cfg.Retry.MaxAttempts, "max-attempts", 3, "total attempts per request, including the first")
	fs.DurationVar(&cfg.Retry.MinDeadlineHeadroom, "retry-headroom", 250*time.Millisecond,
		"deadline that must remain for a retry to be worth starting")

	fs.BoolVar(&cfg.Autoscaling, "autoscaling", false, "enable the autoscaler")
	fs.DurationVar(&cfg.Autoscaler.ScaleUpWait, "scale-up-wait", 500*time.Millisecond,
		"head-of-line queue wait that means more capacity is needed")
	fs.DurationVar(&cfg.Autoscaler.ScaleDownWait, "scale-down-wait", 100*time.Millisecond,
		"head-of-line wait below which capacity looks surplus")
	fs.DurationVar(&cfg.Autoscaler.SustainedFor, "scale-sustained-for", 3*time.Second,
		"how long demand must persist before acting; the spike filter")
	fs.DurationVar(&cfg.Autoscaler.ColdStartDuration, "cold-start", 30*time.Second,
		"how long a new worker takes to become useful")
	fs.IntVar(&cfg.Autoscaler.MinWorkers, "min-workers", 1, "worker floor")
	fs.IntVar(&cfg.Autoscaler.MaxWorkers, "max-workers", 8, "worker ceiling")

	fs.DurationVar(&cfg.Tick, "tick", 5*time.Millisecond, "scheduler period for time-driven work")
	fs.DurationVar(&cfg.DispatchTimeout, "dispatch-timeout", 2*time.Second,
		"per-call deadline on a dispatch to a worker")
	fs.IntVar(&cfg.WorkerSendQueue, "worker-send-queue", 32,
		"outbound queue per worker; a full queue rejects rather than buffering")
	fs.DurationVar(&cfg.ShutdownGrace, "shutdown-grace", 30*time.Second,
		"how long to let in-flight generations finish on shutdown")

	defaultModel := fs.String("default-model", "", "model served when a request names none")
	defaultTenant := fs.String("default-tenant", "", "tenant used when the header is absent; empty makes it mandatory")
	reservation := fs.String("reservation", "peak",
		"cache reservation mode: peak reserves the declared ceiling, expected reserves the prediction")
	victims := fs.String("victim-policy", "newest",
		"preemption victim: newest or longest-remaining")
	quotas := fs.String("quota", "",
		"per-tenant quota as tenant=requests:blocks, comma separated")
	weights := fs.String("tenant-weights", "", "fair-share weights as tenant=weight, comma separated")

	if err := fs.Parse(args); err != nil {
		return cfg, err
	}

	cfg.DefaultModel = loom.ModelID(*defaultModel)
	cfg.DefaultTenant = loom.TenantID(*defaultTenant)

	switch *reservation {
	case "peak":
		cfg.ReservationMode = loom.ReservePeak
	case "expected":
		cfg.ReservationMode = loom.ReserveExpected
	default:
		return cfg, fmt.Errorf("unknown -reservation %q", *reservation)
	}

	switch *victims {
	case "newest":
		cfg.Victims = loom.PreemptNewest
	case "longest-remaining":
		cfg.Victims = loom.PreemptLongestRemaining
	default:
		return cfg, fmt.Errorf("unknown -victim-policy %q", *victims)
	}

	var err error
	if cfg.Quotas, err = parseQuotas(*quotas); err != nil {
		return cfg, err
	}
	if cfg.TenantWeights, err = parseWeights(*weights); err != nil {
		return cfg, err
	}

	cfg.Logf = log.Printf
	return cfg, nil
}

// parseQuotas reads tenant=requests:blocks pairs. A zero in either position means
// unlimited for that dimension, matching TenantQuota.
func parseQuotas(spec string) (map[loom.TenantID]loom.TenantQuota, error) {
	if strings.TrimSpace(spec) == "" {
		return nil, nil
	}
	out := make(map[loom.TenantID]loom.TenantQuota)
	for _, entry := range strings.Split(spec, ",") {
		tenant, limits, ok := strings.Cut(strings.TrimSpace(entry), "=")
		if !ok || tenant == "" {
			return nil, fmt.Errorf("bad -quota entry %q, want tenant=requests:blocks", entry)
		}
		reqs, blocks, ok := strings.Cut(limits, ":")
		if !ok {
			return nil, fmt.Errorf("bad -quota entry %q, want tenant=requests:blocks", entry)
		}
		r, err := strconv.Atoi(reqs)
		if err != nil {
			return nil, fmt.Errorf("bad -quota request count in %q: %w", entry, err)
		}
		b, err := strconv.Atoi(blocks)
		if err != nil {
			return nil, fmt.Errorf("bad -quota block count in %q: %w", entry, err)
		}
		out[loom.TenantID(tenant)] = loom.TenantQuota{MaxConcurrentRequests: r, MaxCacheBlocks: b}
	}
	return out, nil
}

func parseWeights(spec string) (map[loom.TenantID]float64, error) {
	if strings.TrimSpace(spec) == "" {
		return nil, nil
	}
	out := make(map[loom.TenantID]float64)
	for _, entry := range strings.Split(spec, ",") {
		tenant, weight, ok := strings.Cut(strings.TrimSpace(entry), "=")
		if !ok || tenant == "" {
			return nil, fmt.Errorf("bad -tenant-weights entry %q, want tenant=weight", entry)
		}
		w, err := strconv.ParseFloat(weight, 64)
		if err != nil {
			return nil, fmt.Errorf("bad -tenant-weights value in %q: %w", entry, err)
		}
		if w <= 0 {
			return nil, fmt.Errorf("tenant weight in %q must be positive", entry)
		}
		out[loom.TenantID(tenant)] = w
	}
	return out, nil
}
