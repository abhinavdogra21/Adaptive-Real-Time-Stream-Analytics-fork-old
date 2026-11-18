"""
Distributed Hedged Scatter-Gather & Straggler Benchmark
======================================================
Simulates a multi-partition stream cluster (4 partitions, 8 workers: 4 primaries + 4 replicas)
under realistic distributed network conditions with synthetic straggler injection (5% GC stalls).

Compares:
  1. Baseline Unhedged Scatter-Gather:
     Wait for all primary partition workers to respond: T_query = max_p(T_primary_p).
  2. Hedged Scatter-Gather (Jeff Dean's 'The Tail at Scale'):
     If a primary does not reply within P95 normal threshold (e.g. 5.0 ms), fire a duplicate
     hedged request to the partition replica: T_p = min(T_primary_p, T_hedge + T_replica_p).

Collects:
  - Percentiles (P50, P90, P95, P99, P99.9, Max)
  - Hedged duplicate request overhead percentage
  - Latency CDF arrays for publication figures

Outputs:
  - `scatter_gather_results.json`
"""

import asyncio
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from router.scatter_gather import HedgedScatterGatherEngine


def run_benchmark(
    n_queries: int = 1500,
    n_partitions: int = 4,
    straggler_rate: float = 0.05,
    normal_mean_ms: float = 2.5,
    normal_std_ms: float = 0.7,
    straggler_min_ms: float = 35.0,
    straggler_max_ms: float = 75.0,
    hedge_timeouts: Tuple[float, ...] = (4.5, 6.0),
    seed: int = 42,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    # Topology: 4 partitions, each with Primary and Replica
    # Worker metrics: primary has slightly better base latency score
    cluster_metrics = {}
    for p in range(n_partitions):
        cluster_metrics[f"worker-{p}-a"] = {
            "partition": p, "replica": "A",
            "template": "high", "p95_latency_ms": 3.2, "memory_percent": 25.0,
        }
        cluster_metrics[f"worker-{p}-b"] = {
            "partition": p, "replica": "B",
            "template": "medium", "p95_latency_ms": 4.5, "memory_percent": 30.0,
        }

    class MockStore:
        def get_cluster_metrics(self):
            return cluster_metrics

    # Pre-generate latencies for all queries across all partitions to ensure fair comparison
    # shape: (n_queries, n_partitions)
    primary_latencies = np.random.lognormal(
        mean=np.log(normal_mean_ms), sigma=0.25, size=(n_queries, n_partitions)
    )
    replica_latencies = np.random.lognormal(
        mean=np.log(normal_mean_ms * 1.08), sigma=0.25, size=(n_queries, n_partitions)
    )

    # Inject straggler stalls on primary
    straggler_mask = np.random.random((n_queries, n_partitions)) < straggler_rate
    straggler_delays = np.random.uniform(straggler_min_ms, straggler_max_ms, size=(n_queries, n_partitions))
    primary_latencies[straggler_mask] = straggler_delays[straggler_mask]

    configs = [
        {"name": "Unhedged Scatter-Gather", "hedged": False, "timeout_ms": 0.0},
    ]
    for ht in hedge_timeouts:
        configs.append({
            "name": f"Hedged (T_hedge = {ht:.1f}ms)",
            "hedged": True,
            "timeout_ms": ht,
        })

    benchmark_data = {
        "n_queries": n_queries,
        "n_partitions": n_partitions,
        "straggler_rate": straggler_rate,
        "results": [],
    }

    print(f"\n==========================================================================")
    print(f"  BENCHMARK: Hedged Scatter-Gather vs Tail Stragglers ({n_partitions} Partitions)")
    print(f"  Queries: {n_queries:,} | Straggler Rate: {straggler_rate * 100:.1f}% ({straggler_min_ms}-{straggler_max_ms}ms)")
    print(f"==========================================================================")

    for cfg in configs:
        name = cfg["name"]
        hedged = cfg["hedged"]
        ht = cfg["timeout_ms"]

        query_latencies = []
        hedged_fired_total = 0

        for q in range(n_queries):
            part_latencies = []
            for p in range(n_partitions):
                t_prim = primary_latencies[q, p]
                t_repl = replica_latencies[q, p]

                if not hedged:
                    part_lat = t_prim
                else:
                    if t_prim <= ht:
                        part_lat = t_prim
                    else:
                        # Hedge fired after ht ms
                        hedged_fired_total += 1
                        part_lat = min(t_prim, ht + t_repl)

                part_latencies.append(part_lat)

            # Scatter-gather overall latency is bounded by the slowest partition
            query_lat = max(part_latencies)
            query_latencies.append(query_lat)

        arr = np.asarray(query_latencies)
        p50 = float(np.percentile(arr, 50))
        p90 = float(np.percentile(arr, 90))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        p999 = float(np.percentile(arr, 99.9))
        pmax = float(np.max(arr))
        mean = float(np.mean(arr))
        std = float(np.std(arr))

        total_part_queries = n_queries * n_partitions
        hedge_overhead_pct = (hedged_fired_total / total_part_queries) * 100.0

        res_entry = {
            "name": name,
            "hedged": hedged,
            "hedge_timeout_ms": ht,
            "mean_ms": round(mean, 2),
            "std_ms": round(std, 2),
            "p50_ms": round(p50, 2),
            "p90_ms": round(p90, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "p999_ms": round(p999, 2),
            "max_ms": round(pmax, 2),
            "hedged_requests_fired": hedged_fired_total,
            "traffic_overhead_pct": round(hedge_overhead_pct, 2),
            "raw_latencies": [round(float(x), 3) for x in arr.tolist()],
        }
        benchmark_data["results"].append(res_entry)

        print(f"\nConfiguration: {name}")
        print(f"  P50:   {p50:5.2f} ms")
        print(f"  P90:   {p90:5.2f} ms")
        print(f"  P95:   {p95:5.2f} ms")
        print(f"  P99:   {p99:5.2f} ms")
        print(f"  P99.9: {p999:5.2f} ms")
        print(f"  Max:   {pmax:5.2f} ms")
        print(f"  Traffic Overhead: {hedge_overhead_pct:.2f}% ({hedged_fired_total}/{total_part_queries} duplicate requests)")

    out_file = ROOT / "scatter_gather_results.json"
    out_file.write_text(json.dumps(benchmark_data, indent=2))
    print(f"\n✓ Saved benchmark telemetry to {out_file}")
    return benchmark_data


if __name__ == "__main__":
    run_benchmark()
