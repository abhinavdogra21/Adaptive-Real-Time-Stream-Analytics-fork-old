"""
Comparative Experiment Runner — Phase 1 vs. Phase 2 (RL Q-Learning) Benchmark.

Evaluates 4 configurations under a realistic multi-regime workload
(normal traffic, bursty latency spikes, concept drift, and memory pressure):

  1. Baseline: Static (medium) + Round-Robin router
  2. Phase 1 Adaptive: Rule-Based controller + Round-Robin router
  3. Phase 1 Full: Rule-Based controller + Load-Aware router
  4. Phase 2 (Ours): Online Q-Learning (RL) controller + Load-Aware router

Measures real empirical CMS error against exact ground truth, SLA violation
rate, composite RL reward, P99 latency, memory footprint, and flapping count.

Usage:
  python3 scripts/run_simulation.py --events 60000 --nodes 3 --zipf 1.2
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.rl_agent import (
    LinUCBControllerAgent,
    QLearningControllerAgent,
    TEMPLATE_ORDER,
)
from stream.worker import CountMinSketch, HyperLogLog, MisraGries, TEMPLATES


# ---------------------------------------------------------------------------
# Zipfian sampler with Concept Drift support
# ---------------------------------------------------------------------------

class ZipfSampler:
    def __init__(self, n: int = 5_000, s: float = 1.2, seed: int = 42):
        self.rng = random.Random(seed)
        weights = [1.0 / (k ** s) for k in range(1, n + 1)]
        total = sum(weights)
        self._cdf, cum = [], 0.0
        for w in weights:
            cum += w / total
            self._cdf.append(cum)
        self.n = n
        self.shift_offset = 0

    def trigger_drift(self) -> None:
        self.shift_offset = (self.shift_offset + 500) % self.n

    def sample(self) -> str:
        r = self.rng.random()
        lo, hi = 0, self.n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cdf[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        idx = (lo + self.shift_offset) % self.n
        return f"item_{idx + 1}"


# ---------------------------------------------------------------------------
# Simulated worker node with realistic load/memory/error dynamics
# ---------------------------------------------------------------------------

# Realistic relative compute & memory pressure profiles per template
_TEMPLATE_LAT_FACTOR = {"low": 0.45, "medium": 1.00, "high": 2.35}
_TEMPLATE_MEM_BASE   = {"low": 42.0, "medium": 60.0, "high": 78.0}


class SimulatedNode:
    """In-process worker node using the real Phase-2 state-preserving sketches."""

    def __init__(self, node_id: str, template: str = "medium", hw_speed: float = 1.0):
        self.node_id = node_id
        self.hw_speed = hw_speed  # Heterogeneous worker capacity (e.g., fast vs slow node)
        self.template = template
        self.cms = None
        self.hll = None
        self.mg = None
        self.switches = 0
        self.exact_counts: Counter = Counter()
        self._epoch_exact: Counter = Counter()
        self._epoch_cms_base: dict = {}
        self.apply_template(template, count_switch=False)

    def apply_template(self, tpl: str, count_switch: bool = True) -> None:
        if count_switch and tpl == self.template:
            return
        p = TEMPLATES[tpl]
        if self.cms is None:
            self.cms = CountMinSketch(p["cms_width"], p["cms_depth"], conservative=True)
            self.hll = HyperLogLog(p["hll_p"])
            self.mg = MisraGries(p["mg_k"])
        else:
            self.cms = self.cms.resize_preserving(p["cms_width"], p["cms_depth"])
            self.hll = self.hll.change_precision(p["hll_p"])
            self.mg.resize(p["mg_k"])
            self._epoch_exact.clear()
            self._epoch_cms_base.clear()
        if count_switch:
            self.switches += 1
        self.template = tpl

    def process(self, key: str) -> None:
        if key not in self._epoch_cms_base and len(self._epoch_cms_base) < 64:
            self._epoch_cms_base[key] = self.cms.query(key)
        self.cms.update(key)
        self.hll.add(key)
        self.mg.add(key)
        self.exact_counts[key] += 1
        if key in self._epoch_cms_base:
            self._epoch_exact[key] += 1

    def empirical_rel_error(self, sample_keys: list) -> float:
        """Weighted empirical relative CMS error (WAPE) over the active template epoch."""
        total_err = 0.0
        total_exact = 0
        for k, exact in self._epoch_exact.items():
            if exact > 0:
                delta_est = max(0, self.cms.query(k) - self._epoch_cms_base.get(k, 0))
                total_err += abs(delta_est - exact)
                total_exact += exact
        return (total_err / total_exact) if total_exact > 0 else 0.0

    def simulated_metrics(self, regime: dict, rng: random.Random, sample_keys: list) -> dict:
        """
        Compute realistic P99 latency, memory %, and empirical error for the
        current workload regime (traffic burst, memory pressure, or concept drift).
        """
        lat_mult = _TEMPLATE_LAT_FACTOR[self.template]
        base_lat = regime["base_lat_ms"] * lat_mult / self.hw_speed
        p99_ms = max(0.5, base_lat * rng.uniform(0.92, 1.08))

        mem_pct = min(
            99.0,
            max(10.0, _TEMPLATE_MEM_BASE[self.template] + regime["mem_offset"] + rng.uniform(-2.0, 2.0)),
        )

        # Combine real measured CMS error with tail-collision error under drift/skew
        measured_err = self.empirical_rel_error(sample_keys)
        # Scale by theoretical collision pressure when template is low under heavy skew
        width_pressure = (math.e / self.cms.width) * regime["skew_mult"]
        rel_error = max(measured_err, width_pressure)

        return {
            "template": self.template,
            "p99_latency_ms": p99_ms,
            "p95_latency_ms": p99_ms * 0.88,
            "memory_percent": mem_pct,
            "rel_error": rel_error,
            "memory_kb": (self.cms.memory_bytes() + self.hll.memory_bytes() + self.mg.memory_bytes()) / 1024.0,
        }


# ---------------------------------------------------------------------------
# Multi-Regime Workload Schedule (exercises SLA pressure + accuracy demand)
# ---------------------------------------------------------------------------

WORKLOAD_REGIMES = [
    {"name": "normal",        "base_lat_ms": 60.0,  "mem_offset": 0.0,  "skew_mult": 4.0,  "drift": False},
    {"name": "accuracy_skew", "base_lat_ms": 55.0,  "mem_offset": 0.0,  "skew_mult": 14.0, "drift": True},
    {"name": "traffic_burst", "base_lat_ms": 230.0, "mem_offset": 8.0,  "skew_mult": 6.0,  "drift": False},
    {"name": "mem_pressure",  "base_lat_ms": 95.0,  "mem_offset": 24.0, "skew_mult": 5.0,  "drift": False},
]

LAT_SLA_MS = 200.0
MEM_SLA_PCT = 80.0
ERR_SLA = 0.05


def _router_score(m: dict, alpha_q: float = 0.95, latency_sla_ms: float = 50.0) -> float:
    cap = {"low": 0.30, "medium": 0.65, "high": 1.00}[m["template"]]
    return 0.5 * (cap / alpha_q) - 0.3 * (m["p95_latency_ms"] / latency_sla_ms) - 0.2 * (m["memory_percent"] / 100.0)


def run_config(
    name: str,
    n_nodes: int,
    controller_mode: str,   # "static" | "rule" | "linucb" | "rl"
    smart_router: bool,
    n_events: int,
    items: int,
    zipf_s: float,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    sampler = ZipfSampler(n=items, s=zipf_s, seed=seed)

    # Heterogeneous cluster (node-0 fastest, node-1 baseline, node-2 constrained)
    speeds = [1.25, 1.00, 0.78]
    nodes = [
        SimulatedNode(f"node-{i}", "medium", hw_speed=speeds[i % len(speeds)])
        for i in range(n_nodes)
    ]

    # Shared reward evaluator so all configs are scored on the EXACT same objective!
    scorer_agent = QLearningControllerAgent(
        latency_sla_ms=LAT_SLA_MS,
        memory_high_pct=MEM_SLA_PCT,
        error_high=ERR_SLA,
        seed=seed,
    )

    # Phase 2 ML Agents
    rl_agent = QLearningControllerAgent(
        latency_sla_ms=LAT_SLA_MS,
        memory_high_pct=MEM_SLA_PCT,
        error_high=ERR_SLA,
        alpha=0.30,
        gamma=0.85,
        epsilon=0.25,
        epsilon_min=0.01,
        epsilon_decay=0.97,
        seed=seed,
    )
    linucb_agent = LinUCBControllerAgent(
        latency_sla_ms=LAT_SLA_MS,
        memory_high_pct=MEM_SLA_PCT,
        error_high=ERR_SLA,
        alpha_ucb=0.5,
    )

    # Brief offline pre-training over the MDP dynamics (identical budget for both ML agents)
    if controller_mode == "rl":
        for _ in range(180):
            for reg in WORKLOAD_REGIMES:
                for n_idx in range(n_nodes):
                    spd = speeds[n_idx % len(speeds)]
                    tpl = TEMPLATE_ORDER[rng.randrange(3)]
                    lat = max(0.5, reg["base_lat_ms"] * _TEMPLATE_LAT_FACTOR[tpl] / spd)
                    mem = min(99.0, _TEMPLATE_MEM_BASE[tpl] + reg["mem_offset"])
                    err = (math.e / TEMPLATES[tpl]["cms_width"]) * reg["skew_mult"]
                    s = rl_agent.discretize_state(lat, mem, err, tpl)
                    a, _ = rl_agent.select_action(s, explore=True)
                    next_tpl = TEMPLATE_ORDER[a]
                    nlat = max(0.5, reg["base_lat_ms"] * _TEMPLATE_LAT_FACTOR[next_tpl] / spd)
                    nmem = min(99.0, _TEMPLATE_MEM_BASE[next_tpl] + reg["mem_offset"])
                    nerr = (math.e / TEMPLATES[next_tpl]["cms_width"]) * reg["skew_mult"]
                    s_next = rl_agent.discretize_state(nlat, nmem, nerr, next_tpl)
                    r = rl_agent.compute_reward(
                        p99_ms=nlat,
                        memory_pct=nmem,
                        rel_error=nerr,
                        switched=(next_tpl != tpl),
                        template=next_tpl,
                    )
                    rl_agent.update(s, a, r, s_next)
        rl_agent._last_sa.clear()
        rl_agent.switches = 0
        rl_agent.recent_rewards.clear()
        rl_agent.epsilon = 0.01

    elif controller_mode == "linucb":
        for _ in range(180):
            for reg in WORKLOAD_REGIMES:
                for n_idx in range(n_nodes):
                    spd = speeds[n_idx % len(speeds)]
                    tpl = TEMPLATE_ORDER[rng.randrange(3)]
                    lat = max(0.5, reg["base_lat_ms"] * _TEMPLATE_LAT_FACTOR[tpl] / spd)
                    mem = min(99.0, _TEMPLATE_MEM_BASE[tpl] + reg["mem_offset"])
                    err = (math.e / TEMPLATES[tpl]["cms_width"]) * reg["skew_mult"]
                    x = linucb_agent.build_context(lat, mem, err, tpl, 0.0)
                    a, _ = linucb_agent.select_action(x, explore=True)
                    next_tpl = TEMPLATE_ORDER[a]
                    nlat = max(0.5, reg["base_lat_ms"] * _TEMPLATE_LAT_FACTOR[next_tpl] / spd)
                    nmem = min(99.0, _TEMPLATE_MEM_BASE[next_tpl] + reg["mem_offset"])
                    nerr = (math.e / TEMPLATES[next_tpl]["cms_width"]) * reg["skew_mult"]
                    r = linucb_agent.compute_reward(
                        p99_ms=nlat,
                        memory_pct=nmem,
                        rel_error=nerr,
                        switched=(next_tpl != tpl),
                        template=next_tpl,
                    )
                    linucb_agent.update(x, a, r)
        linucb_agent._last_ctx.clear()
        linucb_agent.switches = 0
        linucb_agent.recent_rewards.clear()
        linucb_agent.alpha_ucb = 0.08  # Low online exploration after convergence

    control_interval = max(500, n_events // 80)  # ~80 control intervals across the run
    regime_period = max(control_interval * 5, n_events // 16)

    rr_idx = 0
    total_checks = 0
    sla_violations = 0
    rewards = []
    routed_p99s = []
    routed_errs = []
    mem_kbs = []

    sample_keys = [f"item_{i}" for i in range(1, min(65, items + 1))]
    last_regime_idx = -1

    for ev in range(1, n_events + 1):
        regime_idx = (ev // regime_period) % len(WORKLOAD_REGIMES)
        regime = WORKLOAD_REGIMES[regime_idx]
        if regime_idx != last_regime_idx:
            last_regime_idx = regime_idx
            if regime["drift"]:
                sampler.trigger_drift()

        key = sampler.sample()
        for node in nodes:
            node.process(key)

        if ev % control_interval == 0:
            metrics_map = {
                node.node_id: node.simulated_metrics(regime, rng, sample_keys)
                for node in nodes
            }

            # Evaluate routed query performance (router picks best node or round-robin)
            if smart_router:
                best_node = max(nodes, key=lambda n: _router_score(metrics_map[n.node_id]))
            else:
                best_node = nodes[rr_idx % n_nodes]
                rr_idx += 1

            bm = metrics_map[best_node.node_id]
            routed_p99s.append(bm["p99_latency_ms"])
            routed_errs.append(bm["rel_error"])
            mem_kbs.append(sum(m["memory_kb"] for m in metrics_map.values()) / n_nodes)

            # Evaluate SLA & composite reward across cluster + run controller
            for node in nodes:
                m = metrics_map[node.node_id]
                total_checks += 1
                violated = (
                    m["p99_latency_ms"] > LAT_SLA_MS
                    or m["memory_percent"] > MEM_SLA_PCT
                    or m["rel_error"] > ERR_SLA
                )
                if violated:
                    sla_violations += 1

                switched = False
                if controller_mode == "rule":
                    cur = node.template
                    idx = TEMPLATE_ORDER.index(cur)
                    if m["p99_latency_ms"] > LAT_SLA_MS and idx > 0:
                        node.apply_template(TEMPLATE_ORDER[idx - 1])
                        switched = True
                    elif m["rel_error"] > ERR_SLA and m["memory_percent"] < MEM_SLA_PCT and idx < 2:
                        node.apply_template(TEMPLATE_ORDER[idx + 1])
                        switched = True
                    elif m["memory_percent"] > MEM_SLA_PCT and idx > 0:
                        node.apply_template(TEMPLATE_ORDER[idx - 1])
                        switched = True
                elif controller_mode == "linucb":
                    res = linucb_agent.step_worker(
                        worker_id=node.node_id,
                        p99_ms=m["p99_latency_ms"],
                        memory_pct=m["memory_percent"],
                        rel_error=m["rel_error"],
                        current_template=node.template,
                        heavy_hitters=node.mg.heavy_hitters(10),
                        explore=True,
                    )
                    if res["switched"]:
                        node.apply_template(res["target_template"])
                        switched = True
                elif controller_mode == "rl":
                    res = rl_agent.step_worker(
                        worker_id=node.node_id,
                        p99_ms=m["p99_latency_ms"],
                        memory_pct=m["memory_percent"],
                        rel_error=m["rel_error"],
                        current_template=node.template,
                        heavy_hitters=node.mg.heavy_hitters(10),
                        explore=True,
                    )
                    if res["switched"]:
                        node.apply_template(res["target_template"])
                        switched = True

                r = scorer_agent.compute_reward(
                    p99_ms=m["p99_latency_ms"],
                    memory_pct=m["memory_percent"],
                    rel_error=m["rel_error"],
                    switched=switched,
                    template=node.template,
                )
                rewards.append(r)

    avg_reward = sum(rewards) / max(len(rewards), 1)
    sla_viol_pct = (sla_violations / max(total_checks, 1)) * 100.0
    avg_routed_p99 = sum(routed_p99s) / max(len(routed_p99s), 1)
    avg_routed_err_pct = (sum(routed_errs) / max(len(routed_errs), 1)) * 100.0
    avg_mem_kb = sum(mem_kbs) / max(len(mem_kbs), 1)
    total_switches = sum(n.switches for n in nodes)

    return {
        "config": name,
        "controller_mode": controller_mode,
        "smart_router": smart_router,
        "avg_reward": round(avg_reward, 4),
        "sla_violation_pct": round(sla_viol_pct, 2),
        "routed_p99_ms": round(avg_routed_p99, 2),
        "empirical_cms_err_pct": round(avg_routed_err_pct, 3),
        "avg_mem_kb": round(avg_mem_kb, 1),
        "template_switches": total_switches,
    }


def main():
    parser = argparse.ArgumentParser(
        description="BTP Phase 1 vs. Phase 2 (RL Q-Learning) Comparative Simulation"
    )
    parser.add_argument("--events", type=int,   default=40_000, help="Total events to simulate")
    parser.add_argument("--nodes",  type=int,   default=3,      help="Number of worker nodes")
    parser.add_argument("--zipf",   type=float, default=1.2,    help="Zipf skewness parameter")
    parser.add_argument("--items",  type=int,   default=5_000,  help="Vocabulary size")
    args = parser.parse_args()

    configs = [
        ("1. Static + Round-Robin (Baseline)",     "static", False),
        ("2. Rule-Based + Round-Robin (Phase 1)",  "rule",   False),
        ("3. Rule-Based + Smart Router (Phase 1)", "rule",   True),
        ("4. LinUCB Bandit + Smart Router (Ph.2)", "linucb", True),
        ("5. Q-Learning RL + Smart Router (Ph.2)", "rl",     True),
    ]

    print(f"\nRunning {args.events:,} events across {args.nodes} heterogeneous nodes "
          f"(Zipf s={args.zipf}, vocab={args.items:,}, multi-regime bursts + drift)\n")
    print(f"{'Configuration':<42} {'Reward':>8} {'SLA Viol%':>10} {'P99 (ms)':>10} {'CMS Err%':>10} {'Mem(KB)':>9} {'Sw':>5}")
    print("-" * 100)

    results = []
    for name, mode, smart in configs:
        r = run_config(name, args.nodes, mode, smart, args.events, args.items, args.zipf)
        results.append(r)
        print(
            f"{r['config']:<42} {r['avg_reward']:>+8.4f} {r['sla_violation_pct']:>9.2f}% "
            f"{r['routed_p99_ms']:>9.2f} {r['empirical_cms_err_pct']:>9.3f}% "
            f"{r['avg_mem_kb']:>9.1f} {r['template_switches']:>5}"
        )

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Simulation complete. Results saved to {out_path}")


if __name__ == "__main__":
    main()
