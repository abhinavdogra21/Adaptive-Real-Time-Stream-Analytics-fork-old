"""
Unit tests for Phase 1 fixes (state-preserving sketches, conservative update,
time-decay) and Phase 2 ML Adaptive Controller (Online Tabular Q-Learning +
Concept-Drift Detection).

Runs 100% offline — no Kafka or Redis required.
Run:
    pytest tests/ -v
"""

import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.controller import AdaptiveController, _extract_rel_error
from controller.rl_agent import (
    LinUCBControllerAgent,
    QLearningControllerAgent,
    TEMPLATE_ORDER,
)
from stream.worker import CountMinSketch, HyperLogLog, MisraGries


# ── Phase 1 Sketch Fixes ──────────────────────────────────────────────

class TestPhase1SketchFixes:
    def test_conservative_update_reduces_zipf_overestimation(self):
        """Conservative Update CMS should strictly reduce or match overestimation vs standard CMS."""
        rng = random.Random(7)
        cms_std = CountMinSketch(width=256, depth=4, conservative=False)
        cms_cu = CountMinSketch(width=256, depth=4, conservative=True)
        exact = {}
        for _ in range(5000):
            # Skewed traffic
            k = f"item_{int(rng.paretovariate(1.5)) % 500}"
            exact[k] = exact.get(k, 0) + 1
            cms_std.update(k)
            cms_cu.update(k)

        err_std = sum(cms_std.query(k) - c for k, c in exact.items())
        err_cu = sum(cms_cu.query(k) - c for k, c in exact.items())
        # Both preserve no-undercount
        for k, c in exact.items():
            assert cms_cu.query(k) >= c
        assert err_cu <= err_std

    def test_cms_resize_preserving_downgrade_and_multihop_no_data_loss(self):
        """Downgrading & upgrading repeatedly must never undercount historical keys."""
        cms = CountMinSketch(width=2048, depth=5)
        for _ in range(300):
            cms.update("hot_key")
        for i in range(500):
            cms.update(f"k_{i}")

        # Downgrade 2048x5 -> 512x3 (exact modular folding)
        cms = cms.resize_preserving(512, 3)
        assert cms.query("hot_key") >= 300
        assert cms.total_events() == 800

        # Upgrade 512x3 -> 8192x7
        cms = cms.resize_preserving(8192, 7)
        for _ in range(200):
            cms.update("hot_key")
        assert cms.query("hot_key") >= 500

        # Second hop: downgrade -> upgrade again (must not drop history)
        cms = cms.resize_preserving(2048, 5)
        cms = cms.resize_preserving(8192, 7)
        for _ in range(100):
            cms.update("hot_key")
        assert cms.query("hot_key") >= 600
        assert cms.total_events() == 1100

    def test_cms_exponential_decay_bounds_counts(self):
        cms = CountMinSketch(width=512, depth=3)
        for _ in range(1000):
            cms.update("item_1")
        before = cms.query("item_1")
        cms.decay(0.5)
        after = cms.query("item_1")
        assert after < before
        assert abs(after - before * 0.5) <= 2

    def test_hll_change_precision_preserves_cardinality(self):
        hll = HyperLogLog(precision=12)
        for i in range(4000):
            hll.add(f"user_{i}")
        orig = hll.count()

        down = hll.change_precision(10)
        assert down.precision == 10
        assert abs(down.count() - orig) / orig < 0.25

        up = hll.change_precision(14)
        assert up.precision == 14
        assert abs(up.count() - orig) / orig < 0.25

    def test_mg_resize_preserves_heavy_hitters(self):
        mg = MisraGries(k=200, window_size=50000)
        for _ in range(1000):
            mg.add("king")
        for _ in range(500):
            mg.add("queen")
        for i in range(300):
            mg.add(f"tail_{i}")

        mg.resize(50)
        hh = dict(mg.heavy_hitters(5))
        assert mg.k == 50
        assert "king" in hh and "queen" in hh
        assert hh["king"] > hh["queen"]


# ── Phase 2 ML (Q-Learning) Controller ────────────────────────────────

class TestRLControllerAgent:
    def test_state_discretization_bounds(self):
        agent = QLearningControllerAgent(latency_sla_ms=200.0, memory_high_pct=80.0, error_high=0.05)
        assert agent.discretize_state(20.0, 30.0, 0.005, "low") == (0, 0, 0, 0)
        assert agent.discretize_state(160.0, 70.0, 0.03, "medium") == (2, 1, 1, 1)
        assert agent.discretize_state(260.0, 92.0, 0.09, "high") == (3, 2, 2, 2)

    def test_reward_prefers_sla_compliance_and_penalizes_flapping(self):
        agent = QLearningControllerAgent(latency_sla_ms=200.0, memory_high_pct=80.0, error_high=0.05)
        r_good = agent.compute_reward(p99_ms=50.0, memory_pct=45.0, rel_error=0.01, switched=False, template="medium")
        r_switch = agent.compute_reward(p99_ms=50.0, memory_pct=45.0, rel_error=0.01, switched=True, template="medium")
        r_sla_viol = agent.compute_reward(p99_ms=350.0, memory_pct=92.0, rel_error=0.01, switched=False, template="high")
        assert r_good > 0
        assert r_good > r_switch
        assert r_good > r_sla_viol

    def test_q_learning_converges_to_optimal_policy(self):
        """
        Train the Q-learning agent on two clear regimes:
          - Overload regime: high latency -> optimal action must be 'low' (0)
          - Accuracy-critical healthy regime -> optimal action must be 'high' (2)
        """
        agent = QLearningControllerAgent(
            latency_sla_ms=200.0,
            memory_high_pct=80.0,
            error_high=0.05,
            alpha=0.35,
            gamma=0.80,
            epsilon=0.30,
            seed=123,
        )

        # Simulate overload regime: only 'low' stays under SLA
        overload_lat = {"low": 80.0, "medium": 240.0, "high": 450.0}
        overload_mem = {"low": 55.0, "medium": 75.0, "high": 92.0}
        overload_err = {"low": 0.02, "medium": 0.01, "high": 0.004}

        for _ in range(150):
            s = agent.discretize_state(260.0, 82.0, 0.01, "high")
            a, _ = agent.select_action(s, explore=True)
            nt = TEMPLATE_ORDER[a]
            ns = agent.discretize_state(overload_lat[nt], overload_mem[nt], overload_err[nt], nt)
            r = agent.compute_reward(overload_lat[nt], overload_mem[nt], overload_err[nt], switched=(nt != "high"), template=nt)
            agent.update(s, a, r, ns)

        best_overload_action, _ = agent.select_action(
            agent.discretize_state(260.0, 82.0, 0.01, "high"), explore=False
        )
        assert TEMPLATE_ORDER[best_overload_action] == "low"

        # Simulate accuracy-critical healthy regime: only 'high' satisfies error SLA
        acc_lat = {"low": 25.0, "medium": 45.0, "high": 70.0}
        acc_mem = {"low": 35.0, "medium": 45.0, "high": 55.0}
        acc_err = {"low": 0.09, "medium": 0.06, "high": 0.008}

        for _ in range(150):
            s = agent.discretize_state(25.0, 35.0, 0.09, "low")
            a, _ = agent.select_action(s, explore=True)
            nt = TEMPLATE_ORDER[a]
            ns = agent.discretize_state(acc_lat[nt], acc_mem[nt], acc_err[nt], nt)
            r = agent.compute_reward(acc_lat[nt], acc_mem[nt], acc_err[nt], switched=(nt != "low"), template=nt)
            agent.update(s, a, r, ns)

        best_acc_action, _ = agent.select_action(
            agent.discretize_state(25.0, 35.0, 0.09, "low"), explore=False
        )
        assert TEMPLATE_ORDER[best_acc_action] == "high"

    def test_concept_drift_detector_flags_rank_shift(self):
        agent = QLearningControllerAgent(seed=1)
        # Stable heavy hitters -> no drift
        hh1 = [["item_1", 500], ["item_2", 300], ["item_3", 200]]
        drifted1, _ = agent.detect_concept_drift("w0", hh1)
        assert not drifted1
        drifted2, score2 = agent.detect_concept_drift("w0", hh1)
        assert not drifted2 and score2 == 0.0

        # Sudden rank swap -> drift detected!
        hh_shifted = [["item_999", 550], ["item_888", 300], ["item_777", 150]]
        drifted3, score3 = agent.detect_concept_drift("w0", hh_shifted)
        assert drifted3 and score3 >= 0.9

    def test_q_table_persistence_roundtrip(self):
        agent = QLearningControllerAgent(seed=42)
        s = (1, 0, 2, 0)
        agent.update(s, 2, 0.85, (1, 0, 0, 2))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.json"
            agent.save(path)

            loaded = QLearningControllerAgent(seed=42)
            assert loaded.load(path)
            assert loaded.steps == agent.steps
            assert loaded.q_table[s] == agent.q_table[s]

    def test_controller_empirical_error_and_mode_switch_offline(self):
        """Controller works offline with mock store and uses empirical error properly."""
        class _MockStore:
            def __init__(self):
                self.commands = []
                self.logs = []
                self.stats = None
            def send_command(self, wid, cmd):
                self.commands.append((wid, cmd))
            def log_controller_decision(self, wid, msg):
                self.logs.append((wid, msg))
            def save_controller_stats(self, stats):
                self.stats = stats

        mock = _MockStore()
        ctrl = AdaptiveController(store=mock, mode="rule", q_table_path=None)

        # Empirical error > 0.05 with healthy memory MUST trigger upgrade in Rule mode
        metrics = {
            "template": "low",
            "p99_latency_ms": 15.0,
            "memory_percent": 40.0,
            "cms_total_count": 10000,
            "cms_error_estimate": 53.0,
            "cms_empirical_rel_error": 0.08,
        }
        assert _extract_rel_error(metrics) == 0.08
        decision = ctrl._evaluate_worker_rule("w0", metrics)
        assert decision is not None and "upgrade low→medium" in decision
        assert mock.commands[-1] == ("w0", {"action": "set_template", "template": "medium"})

        # Mode switch to RL and LinUCB
        assert ctrl.set_mode("rl") == "rl"
        assert mock.stats["mode"] == "rl"
        assert ctrl.set_mode("linucb") == "linucb"
        assert mock.stats["mode"] == "linucb"
        assert "LinUCB" in mock.stats["algorithm"]


# ── Phase 2 Second ML Baseline: LinUCB Contextual Bandit ──────────────

class TestLinUCBControllerAgent:
    def test_linucb_learns_optimal_arm_for_continuous_context(self):
        agent = LinUCBControllerAgent(
            latency_sla_ms=200.0, memory_high_pct=80.0, error_high=0.05, alpha_ucb=0.4
        )
        # Train on overload context (low=good, high=bad) and healthy-error context (high=good)
        x_overload = agent.build_context(280.0, 88.0, 0.01, "high", 0.0)
        x_healthy_err = agent.build_context(30.0, 35.0, 0.09, "low", 0.0)

        for _ in range(40):
            agent.update(x_overload, 0, 0.80)   # low succeeds under overload
            agent.update(x_overload, 1, 0.10)
            agent.update(x_overload, 2, -1.20)  # high violates SLA

            agent.update(x_healthy_err, 0, -0.50)  # low has high error
            agent.update(x_healthy_err, 1, 0.20)
            agent.update(x_healthy_err, 2, 0.95)   # high excels

        a_over, _ = agent.select_action(x_overload, explore=False)
        a_err, _ = agent.select_action(x_healthy_err, explore=False)
        assert TEMPLATE_ORDER[a_over] == "low"
        assert TEMPLATE_ORDER[a_err] == "high"

    def test_linucb_persistence_roundtrip(self):
        agent = LinUCBControllerAgent()
        x = agent.build_context(100.0, 50.0, 0.02, "medium", 0.1)
        agent.update(x, 1, 0.72)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "linucb.json"
            agent.save(path)

            loaded = LinUCBControllerAgent()
            assert loaded.load(path)
            assert loaded.steps == 1
            assert abs(float(loaded.b[1].sum()) - float(agent.b[1].sum())) < 1e-9

