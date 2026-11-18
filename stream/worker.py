"""
Python Analytics Worker — consumes from Kafka, runs CMS/HLL/Misra-Gries,
writes metrics to Redis every REPORT_INTERVAL_SEC seconds.

Run:
    python stream/worker.py --worker-id worker-0 --template medium

Multiple instances can run simultaneously (each with a unique --worker-id).
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import threading
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import psutil
import redis
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

KAFKA_SERVERS   = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC", "stream_events")
REDIS_HOST      = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT      = int(os.getenv("REDIS_PORT", 6379))
REPORT_INTERVAL = float(os.getenv("WORKER_REPORT_INTERVAL_SEC", "2"))
GROUP_ID        = os.getenv("KAFKA_GROUP_ID", "btp-analytics-workers")

# ---------------------------------------------------------------------------
# Precision templates (mirrors cpp/worker/worker.cpp)
# ---------------------------------------------------------------------------
TEMPLATES = {
    "low":    {"cms_width": 512,  "cms_depth": 3, "hll_p": 10, "mg_k": 50},
    "medium": {"cms_width": 2048, "cms_depth": 5, "hll_p": 12, "mg_k": 200},
    "high":   {"cms_width": 8192, "cms_depth": 7, "hll_p": 14, "mg_k": 500},
}

# ---------------------------------------------------------------------------
# Count-Min Sketch (with Conservative Updating & State-Preserving Resize)
# ---------------------------------------------------------------------------
class CountMinSketch:
    def __init__(self, width: int, depth: int, conservative: bool = True):
        self.width = width
        self.depth = depth
        self.conservative = conservative
        self.table = [[0] * width for _ in range(depth)]
        self.total = 0
        self._fallback: "CountMinSketch | None" = None  # Preserves history on upgrade

    def _hash(self, key: str, row: int) -> int:
        h = hashlib.md5(f"{key}:{row}".encode()).digest()
        return int.from_bytes(h[:8], "little") % self.width

    def update(self, key: str, count: int = 1):
        indices = [self._hash(key, row) for row in range(self.depth)]
        if self.conservative and count > 0:
            # Conservative Update: only raise counters to (current_min + count)
            # Preserves no-undercount guarantee while suppressing Zipfian collision inflation.
            min_val = min(self.table[row][indices[row]] for row in range(self.depth))
            target = min_val + count
            for row in range(self.depth):
                col = indices[row]
                if self.table[row][col] < target:
                    self.table[row][col] = target
        else:
            for row in range(self.depth):
                self.table[row][indices[row]] += count
        self.total += count

    def query(self, key: str) -> int:
        est = min(self.table[row][self._hash(key, row)] for row in range(self.depth))
        if self._fallback is not None:
            est += self._fallback.query(key)
        return est

    def total_events(self) -> int:
        return self.total + (self._fallback.total_events() if self._fallback else 0)

    def error_estimate(self) -> float:
        return (math.e / self.width) * self.total

    def memory_bytes(self) -> int:
        return self.width * self.depth * 8

    def merge(self, other: "CountMinSketch") -> "CountMinSketch":
        """
        Monoid binary merge operation (⊕) for Count-Min Sketch.
        Element-wise addition over matching dimensions (or power-of-2 folding).
        Guarantees associativity: (A ⊕ B) ⊕ C == A ⊕ (B ⊕ C)
        and identity: A ⊕ 0 == A.
        """
        if self.width == other.width and self.depth == other.depth:
            merged = CountMinSketch(self.width, self.depth, conservative=self.conservative)
            for r in range(self.depth):
                s_row = self.table[r]
                o_row = other.table[r]
                m_row = merged.table[r]
                for c in range(self.width):
                    m_row[c] = s_row[c] + o_row[c]
            merged.total = self.total + other.total
            return merged
        return CountMinSketch._merge_folded(self, other)

    @staticmethod
    def merge_all(sketches: List["CountMinSketch"]) -> "CountMinSketch":
        """Monoid fold across an arbitrary list of Count-Min Sketches."""
        if not sketches:
            return CountMinSketch(2048, 5)
        res = sketches[0]
        for s in sketches[1:]:
            res = res.merge(s)
        return res

    def decay(self, factor: float = 0.5) -> None:
        """
        Exponential time-decay: scales all counters by `factor` (0 < factor < 1),
        bounding CMS error ε·N over infinite streams (Future Scope §7).
        """
        if not (0.0 < factor < 1.0):
            raise ValueError("decay factor must be in (0, 1)")
        for r in range(self.depth):
            row = self.table[r]
            for c in range(self.width):
                row[c] = int(row[c] * factor)
        self.total = int(self.total * factor)
        if self._fallback is not None:
            self._fallback.decay(factor)
            if self._fallback.total <= 0:
                self._fallback = None

    @staticmethod
    def _merge_folded(a: "CountMinSketch", b: "CountMinSketch") -> "CountMinSketch":
        """
        Merge two CMS sketches into a single sketch of (min_width, min_depth)
        via exact power-of-2 modular folding. Guarantees zero event loss and
        bounds fallback depth to 1.
        """
        w = min(a.width, b.width)
        d = min(a.depth, b.depth)
        merged = CountMinSketch(w, d, conservative=a.conservative)
        for src in (a, b):
            for r in range(d):
                src_row = src.table[r]
                dst_row = merged.table[r]
                if src.width == w:
                    for c in range(w):
                        dst_row[c] += src_row[c]
                else:
                    for c in range(src.width):
                        dst_row[c % w] += src_row[c]
        merged.total = a.total + b.total
        return merged

    def resize_preserving(self, new_width: int, new_depth: int) -> "CountMinSketch":
        """
        Migrate to a new (width, depth) without wiping accumulated sketch state.
        - Downgrade (power-of-2 width divisor): exact modular bucket folding.
        - Upgrade: start fresh high-precision table and merge any prior history
          into a single bounded read-only fallback sketch (zero event loss).
        """
        new_cms = CountMinSketch(new_width, new_depth, conservative=self.conservative)
        if new_width <= self.width and self.width % new_width == 0 and new_depth <= self.depth:
            for r in range(new_depth):
                old_row = self.table[r]
                new_row = new_cms.table[r]
                for c in range(self.width):
                    new_row[c % new_width] += old_row[c]
            new_cms.total = self.total
            if self._fallback is not None:
                # Fold fallback into new_cms directly if compatible, else keep merged
                if self._fallback.width >= new_width and self._fallback.width % new_width == 0 and self._fallback.depth >= new_depth:
                    new_cms = CountMinSketch._merge_folded(new_cms, self._fallback)
                else:
                    new_cms._fallback = self._fallback
        else:
            # Upgrade: collapse (self + self._fallback) into a single fallback sketch
            archived = (
                CountMinSketch._merge_folded(self, self._fallback)
                if self._fallback is not None
                else self
            )
            archived._fallback = None
            new_cms._fallback = archived
        return new_cms

# ---------------------------------------------------------------------------
# HyperLogLog (with Precision-Change Folding)
# ---------------------------------------------------------------------------
class HyperLogLog:
    def __init__(self, precision: int):
        self.precision = precision
        self.m = 1 << precision
        self.registers = [0] * self.m
        self._prev: "HyperLogLog | None" = None  # Read-only lower-p history on upgrade
        # Bias correction constant
        if self.m >= 128:
            self._alpha = 0.7213 / (1 + 1.079 / self.m)
        elif self.m == 64:
            self._alpha = 0.709
        elif self.m == 32:
            self._alpha = 0.697
        else:
            self._alpha = 0.673

    def _hash64(self, item: str) -> int:
        h = hashlib.md5(item.encode()).digest()
        return int.from_bytes(h[:8], "little")

    def add(self, item: str):
        h = self._hash64(item)
        idx = h >> (64 - self.precision)
        w   = h & ((1 << (64 - self.precision)) - 1)
        rho = 1
        while rho <= (64 - self.precision) and not (w & (1 << (64 - self.precision - rho))):
            rho += 1
        self.registers[idx] = max(self.registers[idx], rho)

    def _raw_count(self) -> int:
        z = sum(2.0 ** (-r) for r in self.registers)
        est = self._alpha * self.m * self.m / z
        # Small range correction
        if est <= 2.5 * self.m:
            zeros = self.registers.count(0)
            if zeros > 0:
                est = self.m * math.log(self.m / zeros)
        # Large range correction
        elif est > (1 << 32) / 30:
            est = -(1 << 32) * math.log(1 - est / (1 << 32))
        return int(est)

    def count(self) -> int:
        native = self._raw_count()
        if self._prev is None:
            return native
        # Fold current higher-p registers down to _prev.precision and take exact HLL union
        folded = self._fold_registers_only(self._prev.precision)
        folded.merge(self._prev)
        return max(native, folded._raw_count())

    def relative_error(self) -> float:
        return 1.04 / math.sqrt(self.m)

    def memory_bytes(self) -> int:
        return self.m  # 1 byte per register

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        if self.precision != other.precision:
            raise ValueError("Cannot merge HyperLogLog sketches of different precision")
        for i in range(self.m):
            if other.registers[i] > self.registers[i]:
                self.registers[i] = other.registers[i]
        return self

    @staticmethod
    def merge_all(sketches: List["HyperLogLog"]) -> "HyperLogLog":
        """
        Monoid fold across an arbitrary list of HyperLogLog sketches.
        Coordinate-wise maximum over registers: R_merged[i] = max_p R_p[i].
        Guarantees associativity and identity.
        """
        if not sketches:
            return HyperLogLog(12)
        target_p = min(s.precision for s in sketches)
        merged = sketches[0]._fold_registers_only(target_p)
        for s in sketches[1:]:
            merged.merge(s._fold_registers_only(target_p))
        return merged

    def _fold_registers_only(self, new_precision: int) -> "HyperLogLog":
        """Fold registers from higher precision to lower precision via max-binning."""
        folded = HyperLogLog(new_precision)
        if folded.m == self.m:
            folded.registers = list(self.registers)
        else:
            factor = self.m // folded.m
            for i in range(folded.m):
                folded.registers[i] = max(self.registers[i * factor : (i + 1) * factor])
        return folded

    def change_precision(self, new_precision: int) -> "HyperLogLog":
        """
        Transition to a new precision while preserving exact deduplicated cardinality:
          - Downgrade: fold registers via max-binning (and merge any prior history).
          - Upgrade: allocate fresh higher-precision registers and retain a single
            flattened lower-precision snapshot for deduplicated union queries.
        """
        if new_precision == self.precision:
            return self
        if new_precision < self.precision:
            folded = self._fold_registers_only(new_precision)
            if self._prev is not None:
                if self._prev.precision >= new_precision:
                    folded.merge(self._prev._fold_registers_only(new_precision))
                else:
                    folded._prev = self._prev
            return folded

        # Upgrade: collapse (self U self._prev) into a single flat history snapshot
        if self._prev is not None:
            flat_prev = self._fold_registers_only(self._prev.precision)
            flat_prev.merge(self._prev)
        else:
            flat_prev = HyperLogLog(self.precision)
            flat_prev.registers = list(self.registers)

        upgraded = HyperLogLog(new_precision)
        upgraded._prev = flat_prev
        return upgraded

# ---------------------------------------------------------------------------
# Sliding-window Misra-Gries (with State-Preserving Resize)
# ---------------------------------------------------------------------------
class MisraGries:
    def __init__(self, k: int, window_size: int = 10000):
        self.k = k
        self.window_size = window_size
        self.counts: Dict[str, int] = {}
        self.window: deque = deque()
        self.window_total = 0

    def add(self, item: str):
        # Add to window
        self.window.append(item)
        self.window_total += 1
        # Evict if over window size
        if len(self.window) > self.window_size:
            old = self.window.popleft()
            self.window_total -= 1
            if old in self.counts:
                self.counts[old] -= 1
                if self.counts[old] <= 0:
                    del self.counts[old]

        # Standard MG update
        if item in self.counts:
            self.counts[item] += 1
        elif len(self.counts) < self.k - 1:
            self.counts[item] = 1
        else:
            min_val = min(self.counts.values()) if self.counts else 0
            self.counts = {k: v - min_val for k, v in self.counts.items() if v > min_val}

    def resize(self, new_k: int) -> None:
        """Adjust k in-place while retaining the top heavy-hitter candidates."""
        self.k = max(2, int(new_k))
        if len(self.counts) >= self.k:
            top = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)[: self.k - 1]
            self.counts = dict(top)

    def heavy_hitters(self, n: int = 20) -> List[Tuple[str, int]]:
        return sorted(self.counts.items(), key=lambda x: x[1], reverse=True)[:n]

    def error_bound(self) -> float:
        return self.window_total / max(self.k, 1)

    def memory_bytes(self) -> int:
        return self.k * 100

    def merge(self, other: "MisraGries") -> "MisraGries":
        """
        Monoid binary merge (⊕) for Misra-Gries heavy hitters (Agarwal et al. PODS 2012).
        Combines candidate frequency maps and preserves bounded additive error:
            error <= (N_A + N_B) / min(k_A, k_B)
        """
        target_k = min(self.k, other.k)
        merged = MisraGries(target_k, window_size=max(self.window_size, other.window_size))
        merged.window_total = self.window_total + other.window_total

        combined: Dict[str, int] = {}
        for item, cnt in self.counts.items():
            combined[item] = combined.get(item, 0) + cnt
        for item, cnt in other.counts.items():
            combined[item] = combined.get(item, 0) + cnt

        if len(combined) < target_k:
            merged.counts = combined
        else:
            sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
            kth_val = sorted_items[target_k - 1][1] if len(sorted_items) >= target_k else 0
            merged.counts = {item: cnt - kth_val for item, cnt in sorted_items if cnt > kth_val}
            if len(merged.counts) >= target_k:
                min_v = min(merged.counts.values())
                merged.counts = {k: v - min_v for k, v in merged.counts.items() if v > min_v}
        return merged

    @staticmethod
    def merge_all(sketches: List["MisraGries"]) -> "MisraGries":
        """Monoid fold across an arbitrary list of Misra-Gries candidate sketches."""
        if not sketches:
            return MisraGries(200)
        res = sketches[0]
        for s in sketches[1:]:
            res = res.merge(s)
        return res

# ---------------------------------------------------------------------------
# Worker Node
# ---------------------------------------------------------------------------
class AnalyticsWorker:
    _MAX_EXACT_SAMPLE = 64  # Tiny exact shadow sample for measuring empirical CMS error

    def __init__(
        self,
        worker_id: str,
        template: str = "medium",
        partition: int = 0,
        replica: str = "A",
    ):
        self.worker_id = worker_id
        self.template  = template
        self.partition = partition
        self.replica   = replica
        self._latencies: deque = deque(maxlen=500)
        self._lock = threading.Lock()
        self._running = True
        self._cmd_thread = None
        self._sample_exact: Dict[str, int] = {}
        self._sample_cms_base: Dict[str, int] = {}
        self.cms = None
        self.hll = None
        self.mg  = None

        # Redis
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.redis.ping()
        print(f"[Worker:{worker_id}] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")

        # Sketches
        self._apply_template(template)

    def _apply_template(self, tpl: str):
        p = TEMPLATES.get(tpl, TEMPLATES["medium"])
        with self._lock:
            if self.cms is None:
                self.cms = CountMinSketch(p["cms_width"], p["cms_depth"])
                self.hll = HyperLogLog(p["hll_p"])
                self.mg  = MisraGries(p["mg_k"])
            else:
                # Zero-loss state-preserving migration
                self.cms = self.cms.resize_preserving(p["cms_width"], p["cms_depth"])
                self.hll = self.hll.change_precision(p["hll_p"])
                self.mg.resize(p["mg_k"])
                # Reset shadow error baseline so empirical error measures the NEW template
                self._sample_exact.clear()
                self._sample_cms_base: Dict[str, int] = {}
            self.template = tpl
        print(f"[Worker:{self.worker_id}] Applied template '{tpl}' (state preserved): "
              f"CMS {p['cms_width']}×{p['cms_depth']}, HLL p={p['hll_p']}, MG k={p['mg_k']}")

    def process(self, key: str):
        t0 = time.perf_counter()
        with self._lock:
            # Record baseline CMS estimate before first observation in current epoch
            if key not in self._sample_exact and len(self._sample_exact) < self._MAX_EXACT_SAMPLE:
                self._sample_cms_base[key] = self.cms.query(key)
                self._sample_exact[key] = 0
            self.cms.update(key)
            self.hll.add(key)
            self.mg.add(key)
            if key in self._sample_exact:
                self._sample_exact[key] += 1
        lat_ms = (time.perf_counter() - t0) * 1000
        self._latencies.append(lat_ms)
        return lat_ms

    def _percentile(self, p: float) -> float:
        lats = list(self._latencies)
        if not lats:
            return 0.0
        s = sorted(lats)
        idx = min(int(len(s) * p), len(s) - 1)
        return s[idx]

    def p99(self) -> float:
        return self._percentile(0.99)

    def p95(self) -> float:
        return self._percentile(0.95)

    def p50(self) -> float:
        return self._percentile(0.50)

    def empirical_cms_rel_error(self) -> float:
        """
        Weighted relative overestimation (WAPE) over the active template epoch:
            sum(|(f_hat(k) - base(k)) - delta_exact(k)|) / sum(delta_exact(k))
        """
        if not self._sample_exact:
            return 0.0
        total_err = 0.0
        total_exact = 0
        for k, exact in self._sample_exact.items():
            if exact > 0:
                delta_est = max(0, self.cms.query(k) - self._sample_cms_base.get(k, 0))
                total_err += abs(delta_est - exact)
                total_exact += exact
        return (total_err / total_exact) if total_exact > 0 else 0.0

    def report_metrics(self):
        """Write current sketch state to Redis."""
        with self._lock:
            hh = self.mg.heavy_hitters(20)
            emp_err = self.empirical_cms_rel_error()
            metrics = {
                "worker_id":          self.worker_id,
                "template":           self.template,
                "partition":          self.partition,
                "replica":            self.replica,
                "ts":                 time.time(),
                # CMS
                "cms_width":               self.cms.width,
                "cms_depth":               self.cms.depth,
                "cms_total_count":         self.cms.total_events(),
                "cms_error_estimate":      round(self.cms.error_estimate(), 2),
                "cms_empirical_rel_error": round(emp_err, 6),
                # HLL
                "hll_precision":      self.hll.precision,
                "hll_cardinality":    self.hll.count(),
                "hll_relative_error": round(self.hll.relative_error(), 6),
                # MG
                "mg_k":               self.mg.k,
                "mg_heavy_hitters":   hh,
                "mg_window_total":    self.mg.window_total,
                "mg_error_bound":     round(self.mg.error_bound(), 2),
                # System
                "p50_latency_ms":     round(self.p50(), 4),
                "p95_latency_ms":     round(self.p95(), 4),
                "p99_latency_ms":     round(self.p99(), 4),
                "memory_percent":     round(psutil.virtual_memory().percent, 1),
                "cpu_percent":        round(psutil.cpu_percent(interval=None), 1),
            }

        key = f"worker:{self.worker_id}:metrics"
        self.redis.set(key, json.dumps(metrics), ex=30)
        self.redis.sadd("btp:workers", self.worker_id)

        # Latency history ring buffer
        hist_key = f"worker:{self.worker_id}:latency_history"
        self.redis.rpush(hist_key, metrics["p99_latency_ms"])
        self.redis.ltrim(hist_key, -60, -1)
        self.redis.expire(hist_key, 120)

    def _listen_for_commands(self):
        """Listen on Redis pub/sub for controller commands & live RPC point queries."""
        channel = f"ctrl:commands:{self.worker_id}"
        pubsub  = self.redis.pubsub()
        pubsub.subscribe(channel)
        print(f"[Worker:{self.worker_id}] Listening for commands on {channel}")
        for msg in pubsub.listen():
            if not self._running:
                break
            if msg["type"] != "message":
                continue
            try:
                cmd = json.loads(msg["data"])
                action = cmd.get("action")
                if action == "set_template":
                    self._apply_template(cmd["template"])
                elif action == "resize_mg":
                    new_k = cmd.get("k", 50)
                    with self._lock:
                        self.mg.resize(new_k)
                    print(f"[Worker:{self.worker_id}] Resized MG to k={new_k} (state preserved)")
                elif action == "query_frequency":
                    key = cmd.get("key", "")
                    reply_key = cmd.get("reply_key")
                    if reply_key:
                        with self._lock:
                            est = self.cms.query(key)
                            err_bound = round(self.cms.error_estimate(), 2)
                            total_cnt = self.cms.total_events()
                        resp = json.dumps({
                            "key": key,
                            "estimated_frequency": est,
                            "cms_error_bound": err_bound,
                            "cms_total_count": total_cnt,
                            "worker_id": self.worker_id,
                            "partition": self.partition,
                            "replica": self.replica,
                        })
                        self.redis.rpush(reply_key, resp)
                        self.redis.expire(reply_key, 10)
                elif action == "query_cardinality_registers":
                    reply_key = cmd.get("reply_key")
                    if reply_key:
                        with self._lock:
                            regs = list(self.hll.registers)
                            prec = self.hll.precision
                            card = self.hll.count()
                        resp = json.dumps({
                            "registers": regs,
                            "precision": prec,
                            "cardinality": card,
                            "worker_id": self.worker_id,
                            "partition": self.partition,
                            "replica": self.replica,
                        })
                        self.redis.rpush(reply_key, resp)
                        self.redis.expire(reply_key, 10)
                elif action == "query_heavy_hitters_partial":
                    reply_key = cmd.get("reply_key")
                    if reply_key:
                        with self._lock:
                            counts = dict(self.mg.counts)
                            k = self.mg.k
                            win_tot = self.mg.window_total
                        resp = json.dumps({
                            "counts": counts,
                            "k": k,
                            "window_total": win_tot,
                            "worker_id": self.worker_id,
                            "partition": self.partition,
                            "replica": self.replica,
                        })
                        self.redis.rpush(reply_key, resp)
                        self.redis.expire(reply_key, 10)
            except Exception as e:
                print(f"[Worker:{self.worker_id}] Bad command: {e}")

    def run(self, kafka_servers: str, topic: str):
        from kafka import KafkaConsumer

        # Start command listener thread
        self._cmd_thread = threading.Thread(target=self._listen_for_commands, daemon=True)
        self._cmd_thread.start()

        # Connect to Kafka with retries
        # Each worker needs ALL events (analytics fan-out), so use unique group_id
        unique_group = f"{GROUP_ID}-{self.worker_id}"
        consumer = None
        for attempt in range(1, 11):
            try:
                if self.partition is not None and self.partition >= 0:
                    from kafka import TopicPartition
                    consumer = KafkaConsumer(
                        bootstrap_servers=kafka_servers,
                        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                        auto_offset_reset="latest",
                        enable_auto_commit=False,
                        consumer_timeout_ms=1000,
                    )
                    consumer.assign([TopicPartition(topic, self.partition)])
                    print(f"[Worker:{self.worker_id}] Assigned to partition {self.partition} of '{topic}' at {kafka_servers}")
                else:
                    consumer = KafkaConsumer(
                        topic,
                        bootstrap_servers=kafka_servers,
                        group_id=unique_group,
                        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                        auto_offset_reset="latest",
                        enable_auto_commit=True,
                        consumer_timeout_ms=1000,
                    )
                    print(f"[Worker:{self.worker_id}] Subscribed to topic '{topic}' at {kafka_servers}")
                break
            except Exception as e:
                print(f"[Worker:{self.worker_id}] Kafka not ready (attempt {attempt}/10): {e}")
                time.sleep(3)

        if consumer is None:
            raise RuntimeError("Could not connect to Kafka after retries")

        last_report = time.time()
        total       = 0

        print(f"[Worker:{self.worker_id}] Consuming from topic '{topic}'…")
        try:
            while self._running:
                for msg in consumer:
                    if not self._running:
                        break
                    key = msg.value.get("key", "")
                    if key:
                        self.process(key)
                        total += 1

                    # Report metrics every REPORT_INTERVAL seconds
                    now = time.time()
                    if now - last_report >= REPORT_INTERVAL:
                        self.report_metrics()
                        last_report = now
                        print(f"[Worker:{self.worker_id}] Processed {total:,} events | "
                              f"HLL={self.hll.count():,} | "
                              f"P99={self.p99():.3f}ms | "
                              f"Template={self.template}")

                # Also report when idle (consumer poll timeout) to keep Redis key alive
                now = time.time()
                if now - last_report >= REPORT_INTERVAL:
                    self.report_metrics()
                    last_report = now

        except KeyboardInterrupt:
            print(f"\n[Worker:{self.worker_id}] Interrupted")
        finally:
            self._running = False
            consumer.close()
            self.report_metrics()  # Final flush
            print(f"[Worker:{self.worker_id}] Done. Total events: {total:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTP Python Analytics Worker")
    parser.add_argument("--worker-id", default="worker-0",   help="Unique worker identifier")
    parser.add_argument("--template",  default="medium",      choices=["low", "medium", "high"])
    parser.add_argument("--partition", default=0, type=int,   help="Assigned Kafka partition (0..P-1)")
    parser.add_argument("--replica",   default="A",           help="Replica identifier (e.g. A or B)")
    parser.add_argument("--servers",   default=KAFKA_SERVERS, help="Kafka bootstrap servers")
    parser.add_argument("--topic",     default=KAFKA_TOPIC,   help="Kafka topic to consume")
    args = parser.parse_args()

    worker = AnalyticsWorker(
        worker_id=args.worker_id,
        template=args.template,
        partition=args.partition,
        replica=args.replica,
    )
    worker.run(args.servers, args.topic)
