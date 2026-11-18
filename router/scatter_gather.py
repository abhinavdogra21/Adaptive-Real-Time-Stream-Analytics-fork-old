"""
Distributed Scatter-Gather Engine with Tail-Latency Hedging
============================================================

Implements the Partitioned Ingestion + Hedged Monoid Merge Architecture:
  1. SCATTER: Identifies all active partitions and routes sub-queries to the
     optimal primary worker for each partition (scored via Eq. 3).
  2. HEDGE: If a primary worker does not respond within `hedge_timeout_ms`
     (P95 expected latency), fires a duplicate request to the replica worker.
     Whichever responds first is accepted (Jeff Dean's "The Tail at Scale").
  3. GATHER & MONOID MERGE:
     - Count-Min Sketch: Element-wise frequency addition over disjoint partitions
     - HyperLogLog: Coordinate-wise register maximum (R_global[i] = max_p R_p[i])
     - Misra-Gries: Space-Saving Monoid merge (Agarwal et al., ACM TODS 2013)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from router.router import LoadAwareRouter
from state.store import StateStore
from stream.worker import CountMinSketch, HyperLogLog, MisraGries


class HedgedScatterGatherEngine:
    """
    Distributed query orchestrator with:
      - Partition-aware worker grouping (Primary + Hedged Replica)
      - Straggler-mitigating asynchronous request hedging
      - Algebraic Monoid merges across partial sketch summaries
    """

    def __init__(
        self,
        store: Optional[StateStore] = None,
        router: Optional[LoadAwareRouter] = None,
        default_hedge_timeout_ms: float = 6.0,
    ):
        self._store = store
        self._router = router or LoadAwareRouter(store=self.store)
        self.default_hedge_timeout_ms = default_hedge_timeout_ms

    @property
    def store(self) -> StateStore:
        if self._store is None:
            self._store = StateStore()
        return self._store

    @property
    def router(self) -> LoadAwareRouter:
        return self._router

    # ------------------------------------------------------------------
    # Partition Topology Discovery
    # ------------------------------------------------------------------

    def get_partition_topology(self) -> Dict[int, List[dict]]:
        """
        Group all active workers by partition ID.
        If workers do not have an explicit partition, assigns them to partition 0
        or distinct partitions based on worker ID.
        """
        cluster = self.store.get_cluster_metrics()
        if not cluster:
            return {}

        partitions: Dict[int, List[dict]] = {}
        for wid, m in cluster.items():
            p_id = m.get("partition")
            if p_id is None or p_id < 0:
                # Default heuristic if not explicitly set: worker-0 -> 0, worker-1 -> 1
                try:
                    p_id = int(wid.split("-")[-1].replace("worker", "").replace("w", "") or "0")
                except ValueError:
                    p_id = 0
            p_id = int(p_id)
            if p_id not in partitions:
                partitions[p_id] = []
            partitions[p_id].append({"worker_id": wid, "metrics": m})

        return partitions

    def _select_partition_pair(
        self,
        candidates: List[dict],
        query_type: str = "frequency",
    ) -> Tuple[dict, Optional[dict]]:
        """
        Select (primary_worker, hedged_replica) for a given partition.
        Primary is chosen using LoadAwareRouter utility scoring (Eq. 3).
        """
        if not candidates:
            raise ValueError("No candidates for partition")
        if len(candidates) == 1:
            return candidates[0], None

        cluster_sub = {c["worker_id"]: c["metrics"] for c in candidates}
        ranked = self.router._rank_nodes(cluster_sub, alpha_q=0.95, latency_sla_ms=50.0)
        best_wid = ranked[0][0]
        second_wid = ranked[1][0]

        primary = next(c for c in candidates if c["worker_id"] == best_wid)
        replica = next(c for c in candidates if c["worker_id"] == second_wid)
        return primary, replica

    # ------------------------------------------------------------------
    # Asynchronous Hedged Request Execution
    # ------------------------------------------------------------------

    async def _execute_hedged_call(
        self,
        partition_id: int,
        primary_wid: str,
        replica_wid: Optional[str],
        rpc_func: Callable[[str], Any],
        hedge_timeout_ms: float,
        hedged_enabled: bool = True,
    ) -> Tuple[Any, str, bool, float]:
        """
        Dispatches request to primary. If it does not complete within
        `hedge_timeout_ms` and `hedged_enabled` is True, dispatches duplicate
        hedged request to replica. Returns (payload, winning_worker_id, was_hedged, latency_ms).
        """
        t0 = time.perf_counter()

        async def _call(wid: str):
            res = await asyncio.to_thread(rpc_func, wid)
            return wid, res

        primary_task = asyncio.create_task(_call(primary_wid))

        if not hedged_enabled or not replica_wid or hedge_timeout_ms <= 0:
            wid, res = await primary_task
            lat_ms = (time.perf_counter() - t0) * 1000.0
            return res, wid, False, lat_ms

        # Wait up to hedge_timeout_ms for primary
        done, _ = await asyncio.wait([primary_task], timeout=hedge_timeout_ms / 1000.0)

        if primary_task in done:
            wid, res = primary_task.result()
            lat_ms = (time.perf_counter() - t0) * 1000.0
            return res, wid, False, lat_ms

        # Primary exceeded hedge threshold! Fire duplicate hedged request to replica
        replica_task = asyncio.create_task(_call(replica_wid))
        done, pending = await asyncio.wait([primary_task, replica_task], return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        winner_task = done.pop()
        wid, res = winner_task.result()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return res, wid, True, lat_ms

    # ------------------------------------------------------------------
    # Distributed Queries (Scatter-Gather with Monoid Merge)
    # ------------------------------------------------------------------

    async def scatter_gather_frequency(
        self,
        key: str,
        hedge_timeout_ms: Optional[float] = None,
        hedged: bool = True,
        mock_fetcher: Optional[Callable[[str, str], dict]] = None,
    ) -> dict:
        """
        Scatter-gather frequency query with CMS additive Monoid merge.
        Returns total frequency across all partitions, error bounds, and hedging stats.
        """
        t_start = time.perf_counter()
        ht_ms = self.default_hedge_timeout_ms if hedge_timeout_ms is None else hedge_timeout_ms
        topology = self.get_partition_topology()
        if not topology:
            return {"error": "No active partitions found", "partitions": 0}

        def _fetch(wid: str):
            if mock_fetcher:
                return mock_fetcher(wid, key)
            return self.store.query_worker_frequency(wid, key, timeout_sec=2.0)

        tasks = []
        part_keys = sorted(topology.keys())
        for p_id in part_keys:
            primary, replica = self._select_partition_pair(topology[p_id], "frequency")
            replica_id = replica["worker_id"] if replica else None
            tasks.append(self._execute_hedged_call(
                partition_id=p_id,
                primary_wid=primary["worker_id"],
                replica_wid=replica_id,
                rpc_func=_fetch,
                hedge_timeout_ms=ht_ms,
                hedged_enabled=hedged,
            ))

        results = await asyncio.gather(*tasks)

        # Monoid Merge across partitions
        total_freq = 0
        total_events = 0
        total_err_bound = 0.0
        responders = []
        hedged_fired = 0

        for (res, wid, was_hedged, lat_ms) in results:
            if was_hedged:
                hedged_fired += 1
            responders.append({"worker_id": wid, "hedged": was_hedged, "latency_ms": round(lat_ms, 2)})
            if res:
                total_freq += res.get("estimated_frequency", 0)
                total_events += res.get("cms_total_count", 0)
                total_err_bound += res.get("cms_error_bound", 0.0)

        total_lat_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "key": key,
            "estimated_frequency": total_freq,
            "cms_total_count": total_events,
            "cms_error_bound": round(total_err_bound, 2),
            "partitions": len(part_keys),
            "hedged_requests_fired": hedged_fired,
            "responders": responders,
            "total_scatter_gather_latency_ms": round(total_lat_ms, 2),
            "monoid_operator": "Element-wise Count-Min Addition (⊕)",
        }

    async def scatter_gather_cardinality(
        self,
        hedge_timeout_ms: Optional[float] = None,
        hedged: bool = True,
        mock_fetcher: Optional[Callable[[str], dict]] = None,
    ) -> dict:
        """
        Scatter-gather cardinality query with HyperLogLog coordinate-wise max Monoid merge.
        """
        t_start = time.perf_counter()
        ht_ms = self.default_hedge_timeout_ms if hedge_timeout_ms is None else hedge_timeout_ms
        topology = self.get_partition_topology()
        if not topology:
            return {"error": "No active partitions found", "partitions": 0}

        def _fetch(wid: str):
            if mock_fetcher:
                return mock_fetcher(wid)
            return self.store.query_worker_registers(wid, timeout_sec=2.0)

        tasks = []
        part_keys = sorted(topology.keys())
        for p_id in part_keys:
            primary, replica = self._select_partition_pair(topology[p_id], "cardinality")
            replica_id = replica["worker_id"] if replica else None
            tasks.append(self._execute_hedged_call(
                partition_id=p_id,
                primary_wid=primary["worker_id"],
                replica_wid=replica_id,
                rpc_func=_fetch,
                hedge_timeout_ms=ht_ms,
                hedged_enabled=hedged,
            ))

        results = await asyncio.gather(*tasks)

        # Monoid Merge across partitions
        hll_sketches: List[HyperLogLog] = []
        responders = []
        hedged_fired = 0

        for (res, wid, was_hedged, lat_ms) in results:
            if was_hedged:
                hedged_fired += 1
            responders.append({"worker_id": wid, "hedged": was_hedged, "latency_ms": round(lat_ms, 2)})
            if res and "registers" in res:
                h = HyperLogLog(precision=res.get("precision", 12))
                h.registers = list(res["registers"])
                hll_sketches.append(h)

        if hll_sketches:
            merged_hll = HyperLogLog.merge_all(hll_sketches)
            cardinality = merged_hll.count()
            rel_err = merged_hll.relative_error()
            prec = merged_hll.precision
        else:
            cardinality, rel_err, prec = 0, 0.0, 12

        total_lat_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "estimated_cardinality": cardinality,
            "relative_error": round(rel_err, 4),
            "hll_precision": prec,
            "partitions": len(part_keys),
            "hedged_requests_fired": hedged_fired,
            "responders": responders,
            "total_scatter_gather_latency_ms": round(total_lat_ms, 2),
            "monoid_operator": "Coordinate-wise HyperLogLog Maximum (⊕)",
        }

    async def scatter_gather_heavy_hitters(
        self,
        n: int = 10,
        hedge_timeout_ms: Optional[float] = None,
        hedged: bool = True,
        mock_fetcher: Optional[Callable[[str], dict]] = None,
    ) -> dict:
        """
        Scatter-gather heavy hitters query with Misra-Gries Space-Saving Monoid merge.
        """
        t_start = time.perf_counter()
        ht_ms = self.default_hedge_timeout_ms if hedge_timeout_ms is None else hedge_timeout_ms
        topology = self.get_partition_topology()
        if not topology:
            return {"error": "No active partitions found", "partitions": 0}

        def _fetch(wid: str):
            if mock_fetcher:
                return mock_fetcher(wid)
            return self.store.query_worker_heavy_hitters_partial(wid, timeout_sec=2.0)

        tasks = []
        part_keys = sorted(topology.keys())
        for p_id in part_keys:
            primary, replica = self._select_partition_pair(topology[p_id], "heavy_hitters")
            replica_id = replica["worker_id"] if replica else None
            tasks.append(self._execute_hedged_call(
                partition_id=p_id,
                primary_wid=primary["worker_id"],
                replica_wid=replica_id,
                rpc_func=_fetch,
                hedge_timeout_ms=ht_ms,
                hedged_enabled=hedged,
            ))

        results = await asyncio.gather(*tasks)

        mg_sketches: List[MisraGries] = []
        responders = []
        hedged_fired = 0

        for (res, wid, was_hedged, lat_ms) in results:
            if was_hedged:
                hedged_fired += 1
            responders.append({"worker_id": wid, "hedged": was_hedged, "latency_ms": round(lat_ms, 2)})
            if res and "counts" in res:
                m = MisraGries(k=res.get("k", 200))
                m.counts = dict(res["counts"])
                m.window_total = res.get("window_total", 0)
                mg_sketches.append(m)

        if mg_sketches:
            merged_mg = MisraGries.merge_all(mg_sketches)
            hh = merged_mg.heavy_hitters(n)
            win_tot = merged_mg.window_total
            err_b = merged_mg.error_bound()
            k_val = merged_mg.k
        else:
            hh, win_tot, err_b, k_val = [], 0, 0.0, 200

        total_lat_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "heavy_hitters": hh,
            "window_total": win_tot,
            "error_bound": round(err_b, 2),
            "mg_k": k_val,
            "partitions": len(part_keys),
            "hedged_requests_fired": hedged_fired,
            "responders": responders,
            "total_scatter_gather_latency_ms": round(total_lat_ms, 2),
            "monoid_operator": "Agarwal et al. Space-Saving Monoid Merge (⊕)",
        }
