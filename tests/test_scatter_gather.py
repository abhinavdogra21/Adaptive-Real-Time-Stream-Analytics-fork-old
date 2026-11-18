"""
Unit & Integration Tests for Hedged Scatter-Gather Engine & Monoid Merge Algebra
================================================================================
Verifies:
  1. Monoid Algebraic Properties (Associativity, Identity, Idempotence)
     - Count-Min Sketch: Abelian Monoid (Additive)
     - HyperLogLog: Bounded Upper Semilattice (Coordinate-wise Max)
     - Misra-Gries: Space-Saving Monoid Merge (Agarwal et al.)
  2. Hedged Request Scatter-Gather Execution:
     - Partition discovery & Primary/Replica pair election
     - Straggler mitigation: P95 timeout triggers replica dispatch
     - Multi-partition aggregation across Frequency, Cardinality, Heavy Hitters
"""

import asyncio
import os
import sys
import time
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.scatter_gather import HedgedScatterGatherEngine
from stream.worker import CountMinSketch, HyperLogLog, MisraGries


# =====================================================================
# 1. Monoid Merge Algebraic Invariants
# =====================================================================

class TestMonoidAlgebraicProperties:
    """Verifies that sketch merges satisfy formal algebraic monoid laws."""

    def test_cms_monoid_identity_and_associativity(self):
        # A: events 1..100
        cms_a = CountMinSketch(width=512, depth=3)
        for i in range(100):
            cms_a.update(f"user_{i}", count=2)

        # B: events 50..150
        cms_b = CountMinSketch(width=512, depth=3)
        for i in range(50, 150):
            cms_b.update(f"user_{i}", count=3)

        # C: events 100..200
        cms_c = CountMinSketch(width=512, depth=3)
        for i in range(100, 200):
            cms_c.update(f"user_{i}", count=5)

        # Identity element 0 (empty CMS)
        cms_zero = CountMinSketch(width=512, depth=3)
        merged_identity = cms_a.merge(cms_zero)
        assert merged_identity.table == cms_a.table
        assert merged_identity.total == cms_a.total

        # Associativity: (A ⊕ B) ⊕ C == A ⊕ (B ⊕ C)
        ab_c = (cms_a.merge(cms_b)).merge(cms_c)
        a_bc = cms_a.merge(cms_b.merge(cms_c))

        assert ab_c.table == a_bc.table
        assert ab_c.total == a_bc.total

        # Commutativity: A ⊕ B == B ⊕ A
        ab = cms_a.merge(cms_b)
        ba = cms_b.merge(cms_a)
        assert ab.table == ba.table

        # Monoid merge all
        merged_all = CountMinSketch.merge_all([cms_a, cms_b, cms_c])
        assert merged_all.table == ab_c.table

    def test_hll_monoid_semilattice_properties(self):
        # HyperLogLog forms a join-semilattice: Associative, Commutative, Idempotent
        hll_a = HyperLogLog(precision=10)
        hll_b = HyperLogLog(precision=10)
        hll_c = HyperLogLog(precision=10)

        for i in range(500):
            hll_a.add(f"ip_a_{i}")
        for i in range(500):
            hll_b.add(f"ip_b_{i}")
        for i in range(500):
            hll_c.add(f"ip_c_{i}")

        # Identity element 0 (all registers 0)
        hll_zero = HyperLogLog(precision=10)
        merged_ident = hll_a.merge(hll_zero)
        assert merged_ident.registers == hll_a.registers

        # Idempotence: A ⊕ A == A
        merged_idem = hll_a.merge(hll_a)
        assert merged_idem.registers == hll_a.registers

        # Associativity: (A ⊕ B) ⊕ C == A ⊕ (B ⊕ C)
        ab_c = (hll_a.merge(hll_b)).merge(hll_c)
        a_bc = hll_a.merge(hll_b.merge(hll_c))
        assert ab_c.registers == a_bc.registers

        # Commutativity: A ⊕ B == B ⊕ A
        assert (hll_a.merge(hll_b)).registers == (hll_b.merge(hll_a)).registers

        # Merge cardinality estimation accuracy
        total_est = ab_c.count()
        # 1500 distinct items with precision 10 (std err ~3.25%)
        assert 1300 <= total_est <= 1700

    def test_misra_gries_space_saving_monoid_merge(self):
        # MG A: 1000 items, heavy hitter "item_alpha" has 400 counts
        mg_a = MisraGries(k=50)
        for _ in range(400):
            mg_a.add("item_alpha")
        for i in range(600):
            mg_a.add(f"noise_a_{i}")

        # MG B: 1000 items, heavy hitter "item_beta" has 350 counts, "item_alpha" has 100 counts
        mg_b = MisraGries(k=50)
        for _ in range(350):
            mg_b.add("item_beta")
        for _ in range(100):
            mg_b.add("item_alpha")
        for i in range(550):
            mg_b.add(f"noise_b_{i}")

        # Identity: A ⊕ 0 has identical counts to A
        mg_zero = MisraGries(k=50)
        merged_ident = mg_a.merge(mg_zero)
        assert merged_ident.counts == mg_a.counts
        assert merged_ident.window_total == mg_a.window_total

        # Merge A and B
        merged_ab = mg_a.merge(mg_b)
        hh = dict(merged_ab.heavy_hitters(5))
        assert "item_alpha" in hh
        assert "item_beta" in hh
        # item_alpha total true count = 500
        assert hh["item_alpha"] >= 400


# =====================================================================
# 2. Distributed Hedged Scatter-Gather Engine Tests
# =====================================================================

class TestHedgedScatterGatherEngine:
    """Verifies scatter-gather routing, replica hedging, and aggregation."""

    def _make_mock_store(self):
        store = MagicMock()
        store.get_cluster_metrics.return_value = {
            "worker-0-a": {
                "partition": 0, "replica": "A",
                "template": "high", "p95_latency_ms": 3.0, "memory_percent": 25.0,
            },
            "worker-0-b": {
                "partition": 0, "replica": "B",
                "template": "medium", "p95_latency_ms": 8.0, "memory_percent": 30.0,
            },
            "worker-1-a": {
                "partition": 1, "replica": "A",
                "template": "high", "p95_latency_ms": 4.0, "memory_percent": 20.0,
            },
            "worker-1-b": {
                "partition": 1, "replica": "B",
                "template": "medium", "p95_latency_ms": 12.0, "memory_percent": 35.0,
            },
        }
        return store

    def test_partition_topology_and_pair_selection(self):
        mock_store = self._make_mock_store()
        engine = HedgedScatterGatherEngine(store=mock_store)

        topology = engine.get_partition_topology()
        assert 0 in topology
        assert 1 in topology
        assert len(topology[0]) == 2
        assert len(topology[1]) == 2

        # Primary for partition 0 should be worker-0-a (p95=3ms, template=high)
        primary_0, replica_0 = engine._select_partition_pair(topology[0])
        assert primary_0["worker_id"] == "worker-0-a"
        assert replica_0["worker_id"] == "worker-0-b"

        # Primary for partition 1 should be worker-1-a (p95=4ms, template=high)
        primary_1, replica_1 = engine._select_partition_pair(topology[1])
        assert primary_1["worker_id"] == "worker-1-a"
        assert replica_1["worker_id"] == "worker-1-b"

    def test_scatter_gather_frequency_additive_monoid(self):
        mock_store = self._make_mock_store()
        engine = HedgedScatterGatherEngine(store=mock_store)

        def mock_fetch(wid: str, key: str):
            if "worker-0" in wid:
                return {"estimated_frequency": 140, "cms_total_count": 5000, "cms_error_bound": 12.5}
            else:
                return {"estimated_frequency": 260, "cms_total_count": 8000, "cms_error_bound": 18.2}

        res = asyncio.run(engine.scatter_gather_frequency("item_42", mock_fetcher=mock_fetch))

        assert res["key"] == "item_42"
        assert res["estimated_frequency"] == 400  # 140 + 260
        assert res["cms_total_count"] == 13000   # 5000 + 8000
        assert res["partitions"] == 2
        assert res["hedged_requests_fired"] == 0

    def test_scatter_gather_cardinality_coordinate_wise_max(self):
        mock_store = self._make_mock_store()
        engine = HedgedScatterGatherEngine(store=mock_store)

        # Worker 0 observed 100 items; Worker 1 observed 100 overlapping items + 100 new items
        hll_0 = HyperLogLog(precision=10)
        hll_1 = HyperLogLog(precision=10)
        for i in range(100):
            hll_0.add(f"id_{i}")
        for i in range(50, 200):
            hll_1.add(f"id_{i}")

        def mock_fetch(wid: str):
            if "worker-0" in wid:
                return {"registers": hll_0.registers, "precision": 10, "cardinality": hll_0.count()}
            else:
                return {"registers": hll_1.registers, "precision": 10, "cardinality": hll_1.count()}

        res = asyncio.run(engine.scatter_gather_cardinality(mock_fetcher=mock_fetch))

        # Total distinct union = 200 items
        assert res["partitions"] == 2
        assert 170 <= res["estimated_cardinality"] <= 230
        assert res["monoid_operator"] == "Coordinate-wise HyperLogLog Maximum (⊕)"

    def test_straggler_hedging_mitigation(self):
        """
        Simulate a straggler: primary worker-0-a suffers a 35ms GC stall.
        The hedge threshold is set to 5ms.
        Verify that:
          1. Hedged request to replica worker-0-b fires.
          2. Result is returned from worker-0-b in ~6-8ms, bypassing the 35ms straggler.
          3. was_hedged flag is True.
        """
        mock_store = self._make_mock_store()
        engine = HedgedScatterGatherEngine(store=mock_store, default_hedge_timeout_ms=5.0)

        def mock_fetch(wid: str, key: str):
            if wid == "worker-0-a":
                # Artificial straggler pause
                time.sleep(0.040)
                return {"estimated_frequency": 50, "cms_total_count": 1000, "cms_error_bound": 5.0}
            elif wid == "worker-0-b":
                # Fast replica
                time.sleep(0.002)
                return {"estimated_frequency": 50, "cms_total_count": 1000, "cms_error_bound": 5.0}
            else:
                time.sleep(0.002)
                return {"estimated_frequency": 50, "cms_total_count": 1000, "cms_error_bound": 5.0}

        t0 = time.perf_counter()
        res = asyncio.run(engine.scatter_gather_frequency("item_straggler", hedge_timeout_ms=5.0, hedged=True, mock_fetcher=mock_fetch))

        assert res["hedged_requests_fired"] >= 1
        # Straggler primary would have taken 40ms; hedged query returned in ~7-15ms!
        assert res["total_scatter_gather_latency_ms"] < 25.0
        # Check responder for partition 0 is indeed the replica worker-0-b
        part0_resp = next(r for r in res["responders"] if "worker-0" in r["worker_id"])
        assert part0_resp["worker_id"] == "worker-0-b"
        assert part0_resp["hedged"] is True
        assert part0_resp["latency_ms"] < 25.0

    def test_unhedged_suffers_full_straggler_delay(self):
        """When hedged=False, query must wait for the full straggler delay."""
        mock_store = self._make_mock_store()
        engine = HedgedScatterGatherEngine(store=mock_store)

        def mock_fetch(wid: str, key: str):
            if wid == "worker-0-a":
                time.sleep(0.030)
                return {"estimated_frequency": 10, "cms_total_count": 100, "cms_error_bound": 2.0}
            return {"estimated_frequency": 10, "cms_total_count": 100, "cms_error_bound": 2.0}

        t0 = time.perf_counter()
        res = asyncio.run(engine.scatter_gather_frequency("item_slow", hedged=False, mock_fetcher=mock_fetch))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert res["hedged_requests_fired"] == 0
        assert elapsed_ms >= 28.0
        part0_resp = next(r for r in res["responders"] if "worker-0" in r["worker_id"])
        assert part0_resp["worker_id"] == "worker-0-a"
