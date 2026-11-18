"""
Generate publication-quality Phase 2 figures for the BTP report & slides.

Outputs (300 DPI PNGs in `LNMIIT BTP Report Template/Figures/Phase2/`):
  1. fig1_rl_convergence.png        — Q-Learning vs LinUCB reward convergence curve
  2. fig2_controller_comparison.png — 5-config benchmark (Reward, SLA, Error, Switches)
  3. fig3_cu_cms_comparison.png     — Standard CMS vs Conservative-Update CMS error

Usage:
  python3 scripts/plot_results.py
"""

import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from controller.rl_agent import LinUCBControllerAgent, QLearningControllerAgent, TEMPLATE_ORDER
from scripts.run_simulation import (
    ERR_SLA,
    LAT_SLA_MS,
    MEM_SLA_PCT,
    WORKLOAD_REGIMES,
    ZipfSampler,
    _TEMPLATE_LAT_FACTOR,
    _TEMPLATE_MEM_BASE,
)
from stream.worker import CountMinSketch, TEMPLATES

OUT_DIR = ROOT / "LNMIIT BTP Report Template" / "Figures" / "Phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Clean academic styling
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _rolling(arr, w=10):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Figure 1: Online Learning Convergence Curve (Q-Learning vs LinUCB)
# ---------------------------------------------------------------------------

def plot_convergence(bench_results: list) -> Path:
    speeds = [1.25, 1.00, 0.78]
    episodes = 120
    ticks_per_regime = 5  # Matches the real multi-regime dwell time in run_simulation.py

    ql = QLearningControllerAgent(
        latency_sla_ms=LAT_SLA_MS, memory_high_pct=MEM_SLA_PCT, error_high=ERR_SLA,
        alpha=0.25, gamma=0.85, epsilon=0.35, epsilon_min=0.01, epsilon_decay=0.992, seed=42,
    )
    lucb = LinUCBControllerAgent(
        latency_sla_ms=LAT_SLA_MS, memory_high_pct=MEM_SLA_PCT, error_high=ERR_SLA,
        alpha_ucb=0.45,
    )

    ql_ep_rewards, lucb_ep_rewards = [], []
    ql_tpls = ["medium"] * 3
    lucb_tpls = ["medium"] * 3

    for ep in range(episodes):
        ep_q, ep_l = [], []
        for reg in WORKLOAD_REGIMES:
            for _ in range(ticks_per_regime):
                for i, spd in enumerate(speeds):
                    # Q-Learning step
                    tpl_q = ql_tpls[i]
                    lat_q = max(0.5, reg["base_lat_ms"] * _TEMPLATE_LAT_FACTOR[tpl_q] / spd)
                    mem_q = min(99.0, _TEMPLATE_MEM_BASE[tpl_q] + reg["mem_offset"])
                    err_q = (math.e / TEMPLATES[tpl_q]["cms_width"]) * reg["skew_mult"]
                    s = ql.discretize_state(lat_q, mem_q, err_q, tpl_q)
                    a, _ = ql.select_action(s, explore=True)
                    nt_q = TEMPLATE_ORDER[a]
                    nlat_q = max(0.5, reg["base_lat_ms"] * _TEMPLATE_LAT_FACTOR[nt_q] / spd)
                    nmem_q = min(99.0, _TEMPLATE_MEM_BASE[nt_q] + reg["mem_offset"])
                    nerr_q = (math.e / TEMPLATES[nt_q]["cms_width"]) * reg["skew_mult"]
                    ns = ql.discretize_state(nlat_q, nmem_q, nerr_q, nt_q)
                    rq = ql.compute_reward(nlat_q, nmem_q, nerr_q, switched=(nt_q != tpl_q), template=nt_q)
                    ql.update(s, a, rq, ns)
                    ql_tpls[i] = nt_q
                    ep_q.append(rq)

                    # LinUCB step
                    tpl_l = lucb_tpls[i]
                    lat_l = max(0.5, reg["base_lat_ms"] * _TEMPLATE_LAT_FACTOR[tpl_l] / spd)
                    mem_l = min(99.0, _TEMPLATE_MEM_BASE[tpl_l] + reg["mem_offset"])
                    err_l = (math.e / TEMPLATES[tpl_l]["cms_width"]) * reg["skew_mult"]
                    x = lucb.build_context(lat_l, mem_l, err_l, tpl_l, 0.0)
                    al, _ = lucb.select_action(x, explore=True)
                    nt_l = TEMPLATE_ORDER[al]
                    nlat_l = max(0.5, reg["base_lat_ms"] * _TEMPLATE_LAT_FACTOR[nt_l] / spd)
                    nmem_l = min(99.0, _TEMPLATE_MEM_BASE[nt_l] + reg["mem_offset"])
                    nerr_l = (math.e / TEMPLATES[nt_l]["cms_width"]) * reg["skew_mult"]
                    rl = lucb.compute_reward(nlat_l, nmem_l, nerr_l, switched=(nt_l != tpl_l), template=nt_l)
                    lucb.update(x, al, rl)
                    lucb_tpls[i] = nt_l
                    ep_l.append(rl)

        ql_ep_rewards.append(sum(ep_q) / len(ep_q))
        lucb_ep_rewards.append(sum(ep_l) / len(ep_l))

    w = 6
    xs = np.arange(w, episodes + 1)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(xs, _rolling(ql_ep_rewards, w), label="Q-Learning RL (Phase 2 — Sequential MDP)", color="#2563eb", lw=2.3)
    ax.plot(xs, _rolling(lucb_ep_rewards, w), label="LinUCB Contextual Bandit (Phase 2 — Continuous)", color="#7c3aed", lw=2.0, linestyle="-.")

    # Reference baselines from benchmark
    static_r = bench_results[0]["avg_reward"]
    rule_r = bench_results[2]["avg_reward"]
    ax.axhline(static_r, color="#64748b", linestyle="--", lw=1.4, label=f"Static Baseline ({static_r:+.3f})")
    ax.axhline(rule_r, color="#ea580c", linestyle=":", lw=1.6, label=f"Rule-Based Phase 1 ({rule_r:+.3f})")

    ax.set_title("Online Policy Convergence: Q-Learning vs. LinUCB vs. Baselines")
    ax.set_xlabel("Training Episode (Multi-Regime Workload Cycles)")
    ax.set_ylabel("Mean Episode Reward (Rolling Avg, w=6)")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()

    out = OUT_DIR / "fig1_rl_convergence.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: 5-Configuration Comparative Benchmark (4-panel)
# ---------------------------------------------------------------------------

def plot_controller_comparison(bench_results: list) -> Path:
    short_labels = [
        "Static\n+ RR",
        "Rule\n+ RR",
        "Rule\n+ Smart",
        "LinUCB\n+ Smart",
        "Q-Learn\n+ Smart",
    ]
    colors = ["#94a3b8", "#f59e0b", "#ea580c", "#7c3aed", "#2563eb"]

    rewards = [r["avg_reward"] for r in bench_results]
    slas    = [r["sla_violation_pct"] for r in bench_results]
    errs    = [r["empirical_cms_err_pct"] for r in bench_results]
    sws     = [r["template_switches"] for r in bench_results]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4))

    panels = [
        (axes[0, 0], rewards, "Composite Pareto Reward (Higher is Better)", "Mean Reward", "{:+.3f}"),
        (axes[0, 1], slas,    "SLA Violation Rate (Lower is Better)", "Violations (%)", "{:.1f}%"),
        (axes[1, 0], errs,    "Empirical CMS Relative Error (Lower is Better)", "WAPE Error (%)", "{:.2f}%"),
        (axes[1, 1], sws,     "Template Switches / Flapping (Lower is Better)", "Total Switches", "{:d}"),
    ]

    for ax, vals, title, ylabel, fmt_str in panels:
        bars = ax.bar(short_labels, vals, color=colors, width=0.58, edgecolor="#1e293b", linewidth=0.6)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel(ylabel)
        ymax = max(vals) if max(vals) > 0 else 1.0
        ax.set_ylim(0, ymax * 1.22)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + ymax * 0.03,
                fmt_str.format(v),
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )

    fig.suptitle("Phase 1 vs. Phase 2 Comparative Evaluation (40K Zipfian Events, 3 Heterogeneous Nodes)",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = OUT_DIR / "fig2_controller_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: Standard CMS vs Conservative-Update CMS Error under Zipf Skew
# ---------------------------------------------------------------------------

def plot_cu_cms_comparison() -> Path:
    sampler = ZipfSampler(n=5_000, s=1.2, seed=99)
    cms_std = CountMinSketch(width=512, depth=3, conservative=False)
    cms_cu  = CountMinSketch(width=512, depth=3, conservative=True)
    exact   = Counter()

    checkpoints = list(range(5_000, 55_000, 5_000))
    err_std_list, err_cu_list = [], []

    for ev in range(1, checkpoints[-1] + 1):
        k = sampler.sample()
        cms_std.update(k)
        cms_cu.update(k)
        exact[k] += 1

        if ev in checkpoints:
            # Measure WAPE across top-200 active items
            top_items = exact.most_common(200)
            tot_exact = sum(c for _, c in top_items)
            e_std = sum(cms_std.query(k) - c for k, c in top_items) / tot_exact * 100.0
            e_cu  = sum(cms_cu.query(k) - c for k, c in top_items) / tot_exact * 100.0
            err_std_list.append(e_std)
            err_cu_list.append(e_cu)

    xs = [c / 1000 for c in checkpoints]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(xs, err_std_list, marker="o", lw=2.2, color="#ef4444", label="Standard CMS (Phase 1 — 512×3)")
    ax.plot(xs, err_cu_list,  marker="s", lw=2.2, color="#16a34a", label="Conservative-Update CMS (Phase 2 — 512×3)")
    ax.fill_between(xs, err_cu_list, err_std_list, color="#16a34a", alpha=0.10, label="Error Reduction via Conservative Update")

    ax.set_title("Count-Min Sketch Overestimation Under Heavy-Tailed Traffic (Zipf s = 1.2)")
    ax.set_xlabel("Stream Volume (Thousands of Events)")
    ax.set_ylabel("Weighted Relative Error — WAPE (%)")
    ax.legend(loc="center right", frameon=True, fontsize=9)
    fig.tight_layout()

    out = OUT_DIR / "fig3_cu_cms_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 4: Hedged Scatter-Gather Tail Latency & CDF under Stragglers
# ---------------------------------------------------------------------------

def plot_hedged_scatter_gather() -> Path:
    res_path = ROOT / "scatter_gather_results.json"
    if not res_path.exists():
        from scripts.run_scatter_gather_benchmark import run_benchmark
        run_benchmark()
    bench = json.loads(res_path.read_text())

    results = bench["results"]
    unhedged = results[0]
    h45 = results[1]
    h60 = results[2]

    fig, (ax_cdf, ax_bar) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # Panel A: Latency CDF
    for res, color, ls, lw, label in [
        (unhedged, "#ef4444", "-", 2.2, "Unhedged (Wait for Straggler)"),
        (h60,      "#10b981", ":", 2.0, "Hedged ($T_{hedge} = 6.0$ ms)"),
        (h45,      "#2563eb", "--", 2.2, "Hedged ($T_{hedge} = 4.5$ ms)"),
    ]:
        sorted_vals = np.sort(res["raw_latencies"])
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax_cdf.plot(sorted_vals, cdf, color=color, linestyle=ls, linewidth=lw, label=label)

    ax_cdf.axvline(h45["p99_ms"], color="#2563eb", linestyle=":", alpha=0.6)
    ax_cdf.axvline(unhedged["p99_ms"], color="#ef4444", linestyle=":", alpha=0.6)
    ax_cdf.text(h45["p99_ms"] + 1, 0.45, f"Hedged P99\n{h45['p99_ms']:.1f} ms", color="#2563eb", fontsize=8.5, fontweight="bold")
    ax_cdf.text(unhedged["p99_ms"] - 22, 0.45, f"Unhedged P99\n{unhedged['p99_ms']:.1f} ms", color="#ef4444", fontsize=8.5, fontweight="bold")

    ax_cdf.set_title("(a) Cumulative Latency Distribution (CDF)")
    ax_cdf.set_xlabel("Scatter-Gather Query Latency (ms)")
    ax_cdf.set_ylabel(r"Cumulative Probability $P(T \leq t)$")
    ax_cdf.set_xlim(0, 80)
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.legend(loc="lower right", frameon=True, fontsize=8.8)

    # Panel B: Grouped Percentiles Bar Chart
    percentiles = ["P50", "P90", "P95", "P99", "P99.9"]
    p_unhedged = [unhedged["p50_ms"], unhedged["p90_ms"], unhedged["p95_ms"], unhedged["p99_ms"], unhedged["p999_ms"]]
    p_h45      = [h45["p50_ms"], h45["p90_ms"], h45["p95_ms"], h45["p99_ms"], h45["p999_ms"]]

    x = np.arange(len(percentiles))
    width = 0.36

    rects1 = ax_bar.bar(x - width/2, p_unhedged, width, label=f"Unhedged (0% extra req)", color="#f87171", edgecolor="#1e293b", linewidth=0.6)
    rects2 = ax_bar.bar(x + width/2, p_h45, width, label=f"Hedged 4.5ms (+{h45['traffic_overhead_pct']}% extra req)", color="#3b82f6", edgecolor="#1e293b", linewidth=0.6)

    ax_bar.set_title("(b) Tail Latency Percentiles Under 5% Node Stragglers")
    ax_bar.set_ylabel("Latency (ms)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(percentiles)
    ax_bar.legend(loc="upper left", frameon=True, fontsize=8.8)
    ax_bar.set_ylim(0, max(p_unhedged) * 1.20)

    for bar, val in zip(rects1, p_unhedged):
        ax_bar.text(bar.get_x() + bar.get_width() / 2.0, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    for bar, val in zip(rects2, p_h45):
        ax_bar.text(bar.get_x() + bar.get_width() / 2.0, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

    fig.tight_layout()
    out = OUT_DIR / "fig4_hedged_scatter_gather.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main():
    res_path = ROOT / "experiment_results.json"
    if not res_path.exists():
        raise FileNotFoundError("Run `python3 scripts/run_simulation.py` first.")
    bench_results = json.loads(res_path.read_text())

    f1 = plot_convergence(bench_results)
    f2 = plot_controller_comparison(bench_results)
    f3 = plot_cu_cms_comparison()
    f4 = plot_hedged_scatter_gather()
    print(f"✓ Generated 4 figures in {OUT_DIR}:")
    for p in (f1, f2, f3, f4):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
