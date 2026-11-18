"""
Adaptive Controller — Phase 2 ML-Driven (Q-Learning) + Phase 1 Rule-Based Controller.

Runs as a background asyncio task alongside the FastAPI server.
Every CONTROLLER_INTERVAL seconds it:
  1. Reads cluster metrics (P99 latency, memory %, empirical CMS error, heavy hitters) from Redis.
  2. In `rl` mode (Phase 2 default): feeds worker state into the Online Tabular Q-Learning agent
     (`QLearningControllerAgent`), updates Q(s,a) via Bellman TD(0), checks for concept drift,
     and selects the optimal precision template ("low" | "medium" | "high").
  3. In `rule` mode (Phase 1 baseline): applies fixed-threshold rules (fixed to use real
     empirical CMS relative error instead of constant e/width).
  4. Issues state-preserving template commands to workers via Redis pub/sub and publishes
     live controller telemetry for the dashboard.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from controller.rl_agent import (
    LinUCBControllerAgent,
    QLearningControllerAgent,
    TEMPLATE_ORDER,
)
from state.store import StateStore

load_dotenv()

LATENCY_HIGH_MS     = float(os.getenv("LATENCY_HIGH_MS",     "200"))
MEMORY_HIGH_PCT     = float(os.getenv("MEMORY_HIGH_PERCENT", "80"))
ERROR_HIGH          = float(os.getenv("ERROR_HIGH",          "0.05"))
CONTROLLER_INTERVAL = float(os.getenv("CONTROLLER_INTERVAL_SEC", "5"))
CONTROLLER_MODE     = os.getenv("CONTROLLER_MODE", "rl").lower()  # "rl" | "linucb" | "rule"

VALID_MODES = {"rl", "linucb", "rule"}
_DEFAULT_QTABLE_PATH = Path(__file__).parent / "q_table.json"
_DEFAULT_LINUCB_PATH = Path(__file__).parent / "linucb_state.json"


def _upgrade(current: str) -> str:
    """Upgrade to the next precision level, if possible."""
    idx = TEMPLATE_ORDER.index(current) if current in TEMPLATE_ORDER else 1
    return TEMPLATE_ORDER[min(idx + 1, len(TEMPLATE_ORDER) - 1)]


def _downgrade(current: str) -> str:
    """Downgrade to the previous precision level, if possible."""
    idx = TEMPLATE_ORDER.index(current) if current in TEMPLATE_ORDER else 1
    return TEMPLATE_ORDER[max(idx - 1, 0)]


def _extract_rel_error(metrics: dict) -> float:
    """
    Prefer the worker's real empirical CMS relative error (measured against
    the exact shadow sample). Fall back to theoretical e/width only if absent.
    """
    emp = metrics.get("cms_empirical_rel_error")
    if emp is not None:
        return float(emp)
    total = metrics.get("cms_total_count", 1) or 1
    return float(metrics.get("cms_error_estimate", 0.0)) / total


class AdaptiveController:
    """
    Adaptive parameter controller supporting three live-switchable modes:
      - "rl"     (Phase 2): Online Tabular Q-Learning (108-state MDP)
      - "linucb" (Phase 2): Continuous-Feature LinUCB Contextual Bandit
      - "rule"   (Phase 1): Fixed-threshold heuristic rules
    """

    def __init__(
        self,
        store: Optional[StateStore] = None,
        mode: str = CONTROLLER_MODE,
        q_table_path: Optional[Path] = _DEFAULT_QTABLE_PATH,
        linucb_path: Optional[Path] = _DEFAULT_LINUCB_PATH,
    ):
        self._store = store
        self.mode = mode if mode in VALID_MODES else "rl"
        self._running = False
        self._q_table_path = q_table_path
        self._linucb_path = linucb_path

        # Track last decision per worker to avoid flapping
        self._last_decision: Dict[str, float] = {}
        self._cooldown_sec = 10  # minimum seconds between template commands per worker

        # Phase 2 Online Q-Learning Agent
        self.agent = QLearningControllerAgent(
            latency_sla_ms=LATENCY_HIGH_MS,
            memory_high_pct=MEMORY_HIGH_PCT,
            error_high=ERROR_HIGH,
        )
        if self._q_table_path and self._q_table_path.exists():
            if self.agent.load(self._q_table_path):
                print(f"[Controller] Loaded persisted Q-table from {self._q_table_path} "
                      f"({self.agent.steps} prior steps)")

        # Phase 2 Second ML Baseline: Continuous-Feature LinUCB Contextual Bandit
        self.linucb_agent = LinUCBControllerAgent(
            latency_sla_ms=LATENCY_HIGH_MS,
            memory_high_pct=MEMORY_HIGH_PCT,
            error_high=ERROR_HIGH,
        )
        if self._linucb_path and self._linucb_path.exists():
            if self.linucb_agent.load(self._linucb_path):
                print(f"[Controller] Loaded persisted LinUCB state from {self._linucb_path} "
                      f"({self.linucb_agent.steps} prior steps)")

    @property
    def store(self) -> StateStore:
        if self._store is None:
            self._store = StateStore()
        return self._store

    def set_mode(self, mode: str) -> str:
        """Switch controller mode live between 'rl', 'linucb', and 'rule'."""
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
        self.mode = mode
        self._publish_stats()
        return self.mode

    def get_stats(self) -> dict:
        """Return live telemetry for the API & dashboard."""
        if self.mode == "linucb":
            stats = self.linucb_agent.get_stats()
        else:
            stats = self.agent.get_stats()
        stats["mode"] = self.mode
        stats["thresholds"] = {
            "latency_high_ms": LATENCY_HIGH_MS,
            "memory_high_pct": MEMORY_HIGH_PCT,
            "error_high": ERROR_HIGH,
        }
        return stats

    def _publish_stats(self) -> None:
        try:
            self.store.save_controller_stats(self.get_stats())
        except Exception:
            pass

    async def run(self) -> None:
        """Main controller loop — run as an asyncio background task."""
        self._running = True
        print(f"[Controller] Started in '{self.mode.upper()}' mode. "
              f"Interval={CONTROLLER_INTERVAL}s, "
              f"Thresholds: P99>{LATENCY_HIGH_MS}ms, "
              f"mem>{MEMORY_HIGH_PCT}%, err>{ERROR_HIGH}")
        self._publish_stats()
        while self._running:
            await asyncio.sleep(CONTROLLER_INTERVAL)
            await self._control_cycle()

    async def _control_cycle(self) -> None:
        """Single control cycle: evaluate all workers and issue commands."""
        cluster = self.store.get_cluster_metrics()
        if not cluster:
            self._publish_stats()
            return

        for worker_id, metrics in cluster.items():
            if self.mode == "rl":
                self._evaluate_worker_ml(worker_id, metrics, self.agent, label="Q-Learn")
            elif self.mode == "linucb":
                self._evaluate_worker_ml(worker_id, metrics, self.linucb_agent, label="LinUCB")
            else:
                self._evaluate_worker_rule(worker_id, metrics)

        self._publish_stats()
        if self.mode == "rl" and self._q_table_path and self.agent.steps > 0 and self.agent.steps % 10 == 0:
            try:
                self.agent.save(self._q_table_path)
            except Exception:
                pass
        elif self.mode == "linucb" and self._linucb_path and self.linucb_agent.steps > 0 and self.linucb_agent.steps % 10 == 0:
            try:
                self.linucb_agent.save(self._linucb_path)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Phase 2: Online ML (Q-Learning or LinUCB) Evaluation
    # ------------------------------------------------------------------

    def _evaluate_worker_ml(self, worker_id: str, metrics: dict, agent, label: str = "RL") -> Optional[str]:
        now = time.time()
        current_template = metrics.get("template", "medium")
        p99              = float(metrics.get("p99_latency_ms", 0.0))
        memory_pct       = float(metrics.get("memory_percent", 0.0))
        rel_error        = _extract_rel_error(metrics)
        heavy_hitters    = metrics.get("mg_heavy_hitters", [])

        res = agent.step_worker(
            worker_id=worker_id,
            p99_ms=p99,
            memory_pct=memory_pct,
            rel_error=rel_error,
            current_template=current_template,
            heavy_hitters=heavy_hitters,
            explore=True,
        )

        if res["drift_detected"]:
            drift_msg = (
                f"[{label}] Concept drift detected (TVD={res['drift_score']:.2f})"
            )
            self.store.log_controller_decision(worker_id, drift_msg)
            print(f"[Controller] {worker_id}: {drift_msg}")

        # Respect cooldown before issuing template-switch commands
        last = self._last_decision.get(worker_id, 0)
        if not res["switched"] or (now - last < self._cooldown_sec):
            return None

        target = res["target_template"]
        direction = "upgrade" if TEMPLATE_ORDER.index(target) > TEMPLATE_ORDER.index(current_template) else "downgrade"
        tag = "explore" if res["exploratory"] else "exploit"
        r_str = f"R={res['reward']:+.2f}" if res["reward"] is not None else "R=init"
        decision = (
            f"[{label}:{tag}] {direction} {current_template}→{target} "
            f"(P99={p99:.1f}ms, mem={memory_pct:.0f}%, err={rel_error:.3f}, {r_str})"
        )
        self._send_template(worker_id, target)
        self._last_decision[worker_id] = now
        self.store.log_controller_decision(worker_id, decision)
        print(f"[Controller] {worker_id}: {decision}")
        return decision

    # ------------------------------------------------------------------
    # Phase 1: Fixed Rule-Based Evaluation (bug-fixed to use empirical error)
    # ------------------------------------------------------------------

    def _evaluate_worker_rule(self, worker_id: str, metrics: dict) -> Optional[str]:
        now = time.time()
        last = self._last_decision.get(worker_id, 0)
        if now - last < self._cooldown_sec:
            return None

        current_template = metrics.get("template", "medium")
        p99              = float(metrics.get("p99_latency_ms", 0.0))
        memory_pct       = float(metrics.get("memory_percent", 0.0))
        rel_error        = _extract_rel_error(metrics)

        decision = None

        # Rule 1: Latency constraint violated → reduce precision
        if p99 > LATENCY_HIGH_MS:
            new_template = _downgrade(current_template)
            if new_template != current_template:
                decision = (
                    f"[Rule] P99={p99:.1f}ms > {LATENCY_HIGH_MS}ms → "
                    f"downgrade {current_template}→{new_template}"
                )
                self._send_template(worker_id, new_template)

        # Rule 2: Empirical accuracy degraded AND memory available → increase precision
        elif rel_error > ERROR_HIGH and memory_pct < MEMORY_HIGH_PCT:
            new_template = _upgrade(current_template)
            if new_template != current_template:
                decision = (
                    f"[Rule] emp_error={rel_error:.4f} > {ERROR_HIGH} "
                    f"AND mem={memory_pct:.1f}% OK → "
                    f"upgrade {current_template}→{new_template}"
                )
                self._send_template(worker_id, new_template)

        # Rule 3: Memory pressure → offload (shrink MG window & downgrade if high)
        elif memory_pct > MEMORY_HIGH_PCT:
            decision = (
                f"[Rule] memory={memory_pct:.1f}% > {MEMORY_HIGH_PCT}% → "
                f"shrink MG window"
            )
            self._send_mg_resize(worker_id, new_k=50)

        if decision:
            self._last_decision[worker_id] = now
            self.store.log_controller_decision(worker_id, decision)
            print(f"[Controller] {worker_id}: {decision}")
        return decision

    # ------------------------------------------------------------------
    # Command helpers
    # ------------------------------------------------------------------

    def _send_template(self, worker_id: str, template: str) -> None:
        self.store.send_command(worker_id, {
            "action": "set_template",
            "template": template,
        })

    def _send_mg_resize(self, worker_id: str, new_k: int) -> None:
        self.store.send_command(worker_id, {
            "action": "resize_mg",
            "k": new_k,
        })

    def stop(self) -> None:
        self._running = False
        if self._q_table_path and self.agent.steps > 0:
            try:
                self.agent.save(self._q_table_path)
            except Exception:
                pass
        if self._linucb_path and self.linucb_agent.steps > 0:
            try:
                self.linucb_agent.save(self._linucb_path)
            except Exception:
                pass
