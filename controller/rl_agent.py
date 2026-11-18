"""
Reinforcement Learning (Tabular Q-Learning) Agent for Adaptive Sketch Tuning
============================================================================

Phase 2 Intelligence Upgrade — replaces static if-else thresholds with an
online Q-Learning agent that learns the optimal sketch precision template
("low" | "medium" | "high") under changing latency, memory, and accuracy
conditions.

MDP Formulation (matches Section 5 / Future Work of the BTP report):
  State S  = (latency_bin, memory_bin, error_bin, current_template_idx)
             4 × 3 × 3 × 3 = 108 discrete states
             (4 latency bins including proactive approaching-SLA bucket)
  Action A = {0: "low", 1: "medium", 2: "high"}
  Reward R = -(w_err * ErrCost + w_lat * LatCost + w_mem * MemCost + w_sw * SwitchCost)

Bellman Update:
  Q(s, a) ← Q(s, a) + α · [ r + γ · max_{a'} Q(s', a') - Q(s, a) ]
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

TEMPLATE_ORDER = ["low", "medium", "high"]
TEMPLATE_TO_IDX = {t: i for i, t in enumerate(TEMPLATE_ORDER)}


class QLearningControllerAgent:
    """
    Lightweight, numpy-free online Tabular Q-Learning agent with:
      - Epsilon-greedy exploration with exponential decay
      - Nonlinear SLA-violation reward shaping + anti-flapping switch penalty
      - JSON persistence for the learned Q-table
      - Optional heavy-hitter distribution drift detector (L1 total variation)
    """

    def __init__(
        self,
        latency_sla_ms: float = 200.0,
        memory_high_pct: float = 80.0,
        error_high: float = 0.05,
        alpha: float = 0.25,
        gamma: float = 0.90,
        epsilon: float = 0.25,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.985,
        w_err: float = 0.45,
        w_lat: float = 0.35,
        w_mem: float = 0.20,
        w_switch: float = 0.08,
        seed: Optional[int] = None,
    ):
        self.latency_sla_ms = max(latency_sla_ms, 1e-6)
        self.memory_high_pct = max(memory_high_pct, 1.0)
        self.error_high = max(error_high, 1e-6)

        # Q-learning hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Reward weights
        self.w_err = w_err
        self.w_lat = w_lat
        self.w_mem = w_mem
        self.w_switch = w_switch

        self._rng = random.Random(seed)

        # Q-table: state_tuple -> [q_low, q_medium, q_high]
        self.q_table: Dict[Tuple[int, int, int, int], List[float]] = {}

        # Per-worker last (state, action) for online TD(0) update on next tick
        self._last_sa: Dict[str, Tuple[Tuple[int, int, int, int], int]] = {}
        # Per-worker previous heavy-hitter distribution for concept-drift detection
        self._prev_hh_dist: Dict[str, Dict[str, float]] = {}

        # Telemetry
        self.steps: int = 0
        self.switches: int = 0
        self.drift_events: int = 0
        self.recent_rewards: deque = deque(maxlen=100)

    # ------------------------------------------------------------------
    # State Discretization
    # ------------------------------------------------------------------

    def discretize_state(
        self,
        p99_ms: float,
        memory_pct: float,
        rel_error: float,
        current_template: str,
    ) -> Tuple[int, int, int, int]:
        """Map continuous worker metrics into one of 108 discrete MDP states (4×3×3×3)."""
        # Latency bucket: 0 = low, 1 = moderate, 2 = approaching SLA, 3 = SLA violated
        if p99_ms < 0.35 * self.latency_sla_ms:
            lat_b = 0
        elif p99_ms < 0.70 * self.latency_sla_ms:
            lat_b = 1
        elif p99_ms <= self.latency_sla_ms:
            lat_b = 2
        else:
            lat_b = 3

        # Memory bucket: 0 = low, 1 = moderate, 2 = pressure/high
        if memory_pct < 0.75 * self.memory_high_pct:
            mem_b = 0
        elif memory_pct <= self.memory_high_pct:
            mem_b = 1
        else:
            mem_b = 2

        # Error bucket: 0 = accurate, 1 = rising, 2 = accuracy SLA violated
        if rel_error < 0.5 * self.error_high:
            err_b = 0
        elif rel_error <= self.error_high:
            err_b = 1
        else:
            err_b = 2

        tpl_idx = TEMPLATE_TO_IDX.get(current_template, 1)
        return (lat_b, mem_b, err_b, tpl_idx)

    # ------------------------------------------------------------------
    # Reward Function
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        p99_ms: float,
        memory_pct: float,
        rel_error: float,
        switched: bool = False,
        template: str = "medium",
    ) -> float:
        """
        Composite negative-cost reward balancing accuracy, latency SLA, memory,
        and stability (anti-flapping).
        """
        # Normalized error cost + steep violation penalty
        err_ratio = rel_error / self.error_high
        err_cost = min(err_ratio, 3.0)
        if err_ratio > 1.0:
            err_cost += 1.5 * (err_ratio - 1.0)

        # Normalized latency cost + steep SLA violation penalty
        lat_ratio = p99_ms / self.latency_sla_ms
        lat_cost = min(lat_ratio, 3.0)
        if lat_ratio > 1.0:
            lat_cost += 2.0 * (lat_ratio - 1.0)

        # Normalized memory cost + pressure penalty
        mem_ratio = memory_pct / self.memory_high_pct
        mem_cost = min(mem_ratio, 2.5)
        if mem_ratio > 1.0:
            mem_cost += 2.0 * (mem_ratio - 1.0)

        # Small switching penalty to discourage decision flapping
        sw_cost = 1.0 if switched else 0.0

        # Tiny precision bonus when all SLAs are met so agent prefers higher accuracy
        # when headroom is available instead of stagnating on 'low'
        precision_bonus = 0.0
        if lat_ratio <= 1.0 and mem_ratio <= 1.0:
            precision_bonus = 0.05 * TEMPLATE_TO_IDX.get(template, 1)

        cost = (
            self.w_err * err_cost
            + self.w_lat * lat_cost
            + self.w_mem * mem_cost
            + self.w_switch * sw_cost
        )
        # Positive baseline (+1.0) so SLA-compliant actions earn positive reward
        # and SLA-violating actions earn negative reward (prevents 0-init optimism bias)
        return round(1.0 - cost + precision_bonus, 6)

    # ------------------------------------------------------------------
    # Q-Table Access, Action Selection & Bellman Update
    # ------------------------------------------------------------------

    def _q_values(self, state: Tuple[int, int, int, int]) -> List[float]:
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        return self.q_table[state]

    def select_action(
        self,
        state: Tuple[int, int, int, int],
        explore: bool = True,
    ) -> Tuple[int, bool]:
        """
        Epsilon-greedy action selection.
        Returns (action_idx, was_exploratory).
        """
        if explore and self._rng.random() < self.epsilon:
            return self._rng.randrange(len(TEMPLATE_ORDER)), True

        q_vals = self._q_values(state)
        max_q = max(q_vals)
        # Tie-break cleanly: prefer keeping current template if tied
        best_actions = [i for i, q in enumerate(q_vals) if abs(q - max_q) < 1e-12]
        cur_tpl = state[3]
        if cur_tpl in best_actions:
            return cur_tpl, False
        return best_actions[0], False

    def update(
        self,
        state: Tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int, int],
    ) -> float:
        """Perform one TD(0) Q-learning update and return the TD error."""
        q_cur = self._q_values(state)
        q_next = self._q_values(next_state)
        td_target = reward + self.gamma * max(q_next)
        td_error = td_target - q_cur[action]
        q_cur[action] += self.alpha * td_error

        self.steps += 1
        self.recent_rewards.append(reward)
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return td_error

    # ------------------------------------------------------------------
    # Sketch-Based Concept Drift Detection (Total Variation Distance)
    # ------------------------------------------------------------------

    def detect_concept_drift(
        self,
        worker_id: str,
        heavy_hitters: List[List],
        threshold: float = 0.35,
    ) -> Tuple[bool, float]:
        """
        Compute L1 Total Variation distance between consecutive top-K
        heavy-hitter distributions. Detects sudden rank shifts (e.g. producer
        `--shift-interval`) mathematically in O(K) time.
        """
        if not heavy_hitters:
            return False, 0.0
        total = sum(cnt for _, cnt in heavy_hitters)
        if total <= 0:
            return False, 0.0

        curr_dist = {item: cnt / total for item, cnt in heavy_hitters}
        prev_dist = self._prev_hh_dist.get(worker_id)
        self._prev_hh_dist[worker_id] = curr_dist

        if not prev_dist:
            return False, 0.0

        all_keys = set(prev_dist.keys()) | set(curr_dist.keys())
        tvd = 0.5 * sum(abs(curr_dist.get(k, 0.0) - prev_dist.get(k, 0.0)) for k in all_keys)
        drifted = tvd >= threshold
        if drifted:
            self.drift_events += 1
        return drifted, round(tvd, 4)

    # ------------------------------------------------------------------
    # End-to-End Online Step (called per worker per control cycle)
    # ------------------------------------------------------------------

    def step_worker(
        self,
        worker_id: str,
        p99_ms: float,
        memory_pct: float,
        rel_error: float,
        current_template: str,
        heavy_hitters: Optional[List] = None,
        explore: bool = True,
    ) -> dict:
        """
        Observe current worker state, update Q(s_{t-1}, a_{t-1}) with the
        realized reward, and pick the next precision template.
        """
        curr_state = self.discretize_state(p99_ms, memory_pct, rel_error, current_template)
        drifted, drift_score = self.detect_concept_drift(worker_id, heavy_hitters or [])

        # Complete TD update for the previous transition of this worker
        last_reward = None
        if worker_id in self._last_sa:
            prev_state, prev_action = self._last_sa[worker_id]
            switched_prev = (prev_action != prev_state[3])
            last_reward = self.compute_reward(
                p99_ms=p99_ms,
                memory_pct=memory_pct,
                rel_error=rel_error,
                switched=switched_prev,
                template=current_template,
            )
            self.update(prev_state, prev_action, last_reward, curr_state)

        # If concept drift was detected, nudge exploration gently (bounded) to re-adapt
        if drifted and self.epsilon < 0.08:
            self.epsilon = min(0.08, self.epsilon + 0.02)

        action_idx, exploratory = self.select_action(curr_state, explore=explore)
        target_template = TEMPLATE_ORDER[action_idx]
        switched = (target_template != current_template)
        if switched:
            self.switches += 1

        self._last_sa[worker_id] = (curr_state, action_idx)

        return {
            "state": curr_state,
            "action_idx": action_idx,
            "target_template": target_template,
            "switched": switched,
            "exploratory": exploratory,
            "reward": last_reward,
            "q_values": [round(q, 4) for q in self._q_values(curr_state)],
            "epsilon": round(self.epsilon, 4),
            "drift_detected": drifted,
            "drift_score": drift_score,
        }

    # ------------------------------------------------------------------
    # Persistence & Telemetry
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        avg_r = (
            sum(self.recent_rewards) / len(self.recent_rewards)
            if self.recent_rewards
            else 0.0
        )
        return {
            "algorithm": "Tabular Q-Learning (Online RL)",
            "steps": self.steps,
            "states_visited": len(self.q_table),
            "total_states": 108,
            "epsilon": round(self.epsilon, 4),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "avg_recent_reward": round(avg_r, 4),
            "total_switches": self.switches,
            "drift_events": self.drift_events,
            "recent_rewards": [round(r, 4) for r in list(self.recent_rewards)[-20:]],
        }

    def save(self, path: str | Path) -> None:
        data = {
            "steps": self.steps,
            "epsilon": self.epsilon,
            "q_table": {",".join(map(str, k)): v for k, v in self.q_table.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
            self.steps = int(data.get("steps", 0))
            self.epsilon = float(data.get("epsilon", self.epsilon))
            self.q_table = {
                tuple(int(x) for x in k.split(",")): [float(v) for v in vals]
                for k, vals in data.get("q_table", {}).items()
            }
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Second ML Baseline: LinUCB Disjoint Contextual Bandit (Li et al., WWW 2010)
# ---------------------------------------------------------------------------

class LinUCBControllerAgent:
    """
    Continuous-feature Linear Upper Confidence Bound (LinUCB) Contextual Bandit.

    Contrasts with QLearningControllerAgent:
      - Uses continuous context vectors x_t ∈ R^6 directly (no state binning).
      - Balances exploration and exploitation analytically via confidence
        ellipsoids (alpha_ucb * sqrt(x^T A_a^{-1} x)) instead of ε-greedy.
      - Uses the exact same reward objective and L1 concept-drift detector so
        comparisons against Q-Learning and Rule-Based are strictly apples-to-apples.
    """

    DIM = 6  # [bias, lat_ratio, mem_ratio, err_ratio, cur_tpl_norm, drift_score]

    def __init__(
        self,
        latency_sla_ms: float = 200.0,
        memory_high_pct: float = 80.0,
        error_high: float = 0.05,
        alpha_ucb: float = 0.6,
        reg_lambda: float = 1.0,
        w_err: float = 0.45,
        w_lat: float = 0.35,
        w_mem: float = 0.20,
        w_switch: float = 0.08,
    ):
        self.latency_sla_ms = max(latency_sla_ms, 1e-6)
        self.memory_high_pct = max(memory_high_pct, 1.0)
        self.error_high = max(error_high, 1e-6)
        self.alpha_ucb = alpha_ucb
        self.reg_lambda = reg_lambda

        # Reuse identical reward & drift helper via a lightweight delegate
        self._scorer = QLearningControllerAgent(
            latency_sla_ms=latency_sla_ms,
            memory_high_pct=memory_high_pct,
            error_high=error_high,
            w_err=w_err,
            w_lat=w_lat,
            w_mem=w_mem,
            w_switch=w_switch,
        )

        # Disjoint ridge-regression parameters per arm (0:low, 1:medium, 2:high)
        self.A: List[np.ndarray] = [
            reg_lambda * np.eye(self.DIM, dtype=np.float64)
            for _ in TEMPLATE_ORDER
        ]
        self.b: List[np.ndarray] = [
            np.zeros(self.DIM, dtype=np.float64)
            for _ in TEMPLATE_ORDER
        ]

        # Per-worker last (context_vec, action_idx, prev_template_idx)
        self._last_ctx: Dict[str, Tuple[np.ndarray, int, int]] = {}

        # Telemetry
        self.steps: int = 0
        self.switches: int = 0
        self.drift_events: int = 0
        self.recent_rewards: deque = deque(maxlen=100)

    def build_context(
        self,
        p99_ms: float,
        memory_pct: float,
        rel_error: float,
        current_template: str,
        drift_score: float = 0.0,
    ) -> np.ndarray:
        """Continuous feature vector x_t ∈ R^6."""
        return np.array([
            1.0,
            min(p99_ms / self.latency_sla_ms, 2.5),
            min(memory_pct / self.memory_high_pct, 2.0),
            min(rel_error / self.error_high, 2.5),
            TEMPLATE_TO_IDX.get(current_template, 1) / 2.0,
            min(max(drift_score, 0.0), 1.0),
        ], dtype=np.float64)

    def compute_reward(
        self,
        p99_ms: float,
        memory_pct: float,
        rel_error: float,
        switched: bool = False,
        template: str = "medium",
    ) -> float:
        return self._scorer.compute_reward(p99_ms, memory_pct, rel_error, switched, template)

    def select_action(self, x: np.ndarray, explore: bool = True) -> Tuple[int, List[float]]:
        """Compute UCB score p_{t,a} = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)."""
        ucb_scores: List[float] = []
        alpha = self.alpha_ucb if explore else 0.0
        for a in range(len(TEMPLATE_ORDER)):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mean_est = float(theta @ x)
            bonus = alpha * math.sqrt(max(float(x @ A_inv @ x), 0.0))
            ucb_scores.append(mean_est + bonus)
        best_a = int(np.argmax(ucb_scores))
        return best_a, ucb_scores

    def update(self, x: np.ndarray, action: int, reward: float) -> None:
        """Rank-1 ridge regression update for the chosen arm."""
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x
        self.steps += 1
        self.recent_rewards.append(reward)

    def step_worker(
        self,
        worker_id: str,
        p99_ms: float,
        memory_pct: float,
        rel_error: float,
        current_template: str,
        heavy_hitters: Optional[List] = None,
        explore: bool = True,
    ) -> dict:
        drifted, drift_score = self._scorer.detect_concept_drift(worker_id, heavy_hitters or [])
        if drifted:
            self.drift_events += 1

        last_reward = None
        if worker_id in self._last_ctx:
            prev_x, prev_a, prev_tpl_idx = self._last_ctx[worker_id]
            switched_prev = (prev_a != prev_tpl_idx)
            last_reward = self.compute_reward(
                p99_ms=p99_ms,
                memory_pct=memory_pct,
                rel_error=rel_error,
                switched=switched_prev,
                template=current_template,
            )
            self.update(prev_x, prev_a, last_reward)

        x = self.build_context(p99_ms, memory_pct, rel_error, current_template, drift_score)
        action_idx, ucb_scores = self.select_action(x, explore=explore)
        target_template = TEMPLATE_ORDER[action_idx]
        cur_idx = TEMPLATE_TO_IDX.get(current_template, 1)
        switched = (target_template != current_template)
        if switched:
            self.switches += 1

        self._last_ctx[worker_id] = (x, action_idx, cur_idx)

        return {
            "action_idx": action_idx,
            "target_template": target_template,
            "switched": switched,
            "exploratory": False,
            "reward": last_reward,
            "ucb_scores": [round(s, 4) for s in ucb_scores],
            "epsilon": round(self.alpha_ucb, 4),
            "drift_detected": drifted,
            "drift_score": drift_score,
        }

    def get_stats(self) -> dict:
        avg_r = (
            sum(self.recent_rewards) / len(self.recent_rewards)
            if self.recent_rewards
            else 0.0
        )
        return {
            "algorithm": "LinUCB Contextual Bandit (Continuous-Feature ML)",
            "steps": self.steps,
            "feature_dim": self.DIM,
            "alpha_ucb": round(self.alpha_ucb, 4),
            "epsilon": round(self.alpha_ucb, 4),
            "avg_recent_reward": round(avg_r, 4),
            "total_switches": self.switches,
            "drift_events": self.drift_events,
            "recent_rewards": [round(r, 4) for r in list(self.recent_rewards)[-20:]],
        }

    def save(self, path: str | Path) -> None:
        data = {
            "steps": self.steps,
            "alpha_ucb": self.alpha_ucb,
            "A": [m.tolist() for m in self.A],
            "b": [v.tolist() for v in self.b],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
            self.steps = int(data.get("steps", 0))
            self.alpha_ucb = float(data.get("alpha_ucb", self.alpha_ucb))
            self.A = [np.array(m, dtype=np.float64) for m in data["A"]]
            self.b = [np.array(v, dtype=np.float64) for v in data["b"]]
            return True
        except Exception:
            return False

