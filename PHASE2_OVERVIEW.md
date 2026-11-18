# ⚡ Phase 2 Technical Report & Overview
### Adaptive Real-Time Stream Analytics using Probabilistic Sketches with Load-Aware BTP Request Routing
**LNMIIT BTP Phase 2** · *Distributed Systems, Machine Learning & Stream Analytics*

---

## 📌 Executive Summary

In Phase 1, we established a stream ingestion pipeline using Kafka, Redis, and three probabilistic sketches (Count-Min Sketch, HyperLogLog, and Sliding-Window Misra-Gries) paired with a reactive rule-based controller.

In **Phase 2**, we elevated the system to a **production-grade, distributed, self-tuning stream analytics platform**. The major research and engineering highlights of Phase 2 are:
1. **Online Machine Learning Adaptive Control**: Replaced brittle static if-else rules with an **Online Tabular Q-Learning (RL)** agent and a **LinUCB Contextual Bandit** baseline.
2. **Concept Drift Detection**: Integrated $L_1$ Total Variation Distance monitoring across heavy-hitter snapshots to detect stream distribution shifts automatically.
3. **Zero-Data-Loss Sketch Migration & Conservative Updating**: Added power-of-2 modular folding, zero-loss multi-hop resizing, and Conservative Updating ($>15\times$ overestimation reduction).
4. **Partitioned Ingestion Engine + Hedged Monoid Merge Architecture**: Horizontally scaled stream ingestion across Kafka partitions and implemented formal algebraic Monoid merges $(\mathcal{S}, \oplus, \mathbf{0})$ with **Jeff Dean's (*The Tail at Scale*) Asynchronous Hedged Requests**, dropping P99 tail latency by **$>8.5\times$** under worker stragglers.
5. **Rigorous Offline Verification & Benchmarks**: 49 unit tests runnable 100% offline in $<0.5\text{ s}$, 4 publication-grade 300 DPI figures, and complete LaTeX report updates.

---

## 🏗️ Detailed Architecture & Advancements

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       STREAM PRODUCER                                       │
│          Zipfian Traffic Generator (s=1.2) + Periodic Concept Drift Shift Rotation          │
│                Key-Based Hash Partitioning: partition_id = CRC32(key) % P                   │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 KAFKA MESSAGE BROKER (Topic)                                │
│          Partition 0                 Partition 1                 Partition P-1              │
└──────────────┬───────────────────────────────┬───────────────────────────────┬──────────────┘
               │                               │                               │
       ┌───────┴───────┐               ┌───────┴───────┐               ┌───────┴───────┐
       ▼               ▼               ▼               ▼               ▼               ▼
 ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
 │Worker 0-A │   │Worker 0-B │   │Worker 1-A │   │Worker 1-B │   │Worker P-A │   │Worker P-B │
 │ (Primary) │   │ (Replica) │   │ (Primary) │   │ (Replica) │   │ (Primary) │   │ (Replica) │
 │ CMS+HLL+MG│   │ CMS+HLL+MG│   │ CMS+HLL+MG│   │ CMS+HLL+MG│   │ CMS+HLL+MG│   │ CMS+HLL+MG│
 └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
       │               │               │               │               │               │
       └───────────────┴───────────────┼───────────────┴───────────────┴───────────────┘
                                       ▼ Heartbeats, RPC Replies & Telemetry
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      REDIS STATE STORE                                      │
│  - worker:{id}:metrics (TTL 30s)    - ctrl:commands:{id} (PubSub)    - rpc:reply:{uuid}     │
│  - controller:stats (RL telemetry)  - btp:workers (Active set)       - controller:log       │
└──────────────────────────────────────▲──────────────────────────────────────────────────────┘
                                       │
                 ┌─────────────────────┴──────────────────────┐
                 │                                            │
┌────────────────┴───────────────────────────┐   ┌────────────┴───────────────────────────────┐
│     FASTAPI BACKEND + SCATTER-GATHER       │   │           ML ADAPTIVE CONTROLLER           │
│  - Load-Aware Router (Utility Eq. 3)       │   │  - Tabular Q-Learning (108-State MDP)      │
│  - Hedged Scatter-Gather Engine            │   │  - LinUCB Contextual Bandit (6D Context)   │
│  - Monoid Merge: CMS (⊕), HLL (max), MG(SS)│   │  - L1 TVD Concept Drift Detector           │
│  - Asynchronous P95 Tail Latency Hedging   │   │  - Live Mode Switch: rl | linucb | rule    │
└────────────────────────────────────────────┘   └────────────────────────────────────────────┘
```

---

## 🔬 Core Components Implemented in Phase 2

### 1. Machine Learning Adaptive Controller (`controller/rl_agent.py` & `controller/controller.py`)

- **108-State Markov Decision Process (MDP):**
  $$\mathcal{S} = (\text{latency\_bin [4]}, \text{memory\_bin [3]}, \text{empirical\_error\_bin [3]}, \text{current\_template [3]})$$
  Includes a proactive **approaching-SLA** latency bin ($70\%\text{--}100\%$ of SLA) allowing the agent to downscale precision *before* SLA breach occurs.
- **Action Space:** Direct action selection $\mathcal{A} \in \{\text{low}, \text{medium}, \text{high}\}$, enabling instantaneous single-step transitions during burst traffic without multi-hop lag.
- **Shaped Positive-Baseline Reward Function:**
  $$R_t = 1.0 - \left(w_e \cdot C_{\text{err}} + w_l \cdot C_{\text{lat}} + w_m \cdot C_{\text{mem}} + w_s \cdot \mathbb{I}[\text{switched}]\right) + B_{\text{prec}}$$
  Strictly penalizes latency ($>200\text{ ms}$) and memory ($>80\%$) SLA violations while dampening oscillation (anti-flapping switching cost $w_s$).
- **LinUCB Contextual Bandit Baseline:** Continuous 6-dimensional context vector:
  $$x_t = [1.0,\; \hat{\ell}_{\text{norm}},\; \hat{m}_{\text{norm}},\; \hat{e}_{\text{norm}},\; \text{template\_id}/2,\; \text{drift\_score}]^\top$$
  Updates arm parameters via rank-1 ridge regression $\hat{\theta}_a = A_a^{-1} b_a$ and upper confidence bound exploration.
- **Concept Drift Detection:** Monitors $L_1$ Total Variation Distance between consecutive heavy-hitter snapshots:
  $$D_{\text{TV}}(P_{t-1}, P_t) = \frac{1}{2}\sum_{k} |P_{t-1}(k) - P_t(k)|$$
  When $D_{\text{TV}} \ge \tau$, flags distribution shift and triggers controller adaptation.
- **Live Algorithm Switch:** `/control/mode` toggles dynamically between `rl`, `linucb`, and `rule` without system restarts.

---

### 2. Zero-Loss Sketch Migration & Accuracy Hardening (`stream/worker.py` & C++)

- **Conservative Updating (Count-Min Sketch):**
  Only increments counters achieving the minimum hash value $\min_{j} \mathbf{C}[j, h_j(x)]$, reducing Zipfian overestimation by **$>15\times$** (from $\approx 6.5\%$ down to $\approx 0.4\%$) with zero extra memory.
- **Exponential Time Decay:**
  Multiplies counters by $\lambda = e^{-\alpha \Delta t}$ to bound historical counts and favor recent stream behavior.
- **Zero-Data-Loss Migration:**
  - *Downgrade:* Folds counters into new dimensions using modular arithmetic $(i \pmod{w_{\text{new}}})$ with exact conservation of totals.
  - *Upgrade:* Retains read-only previous sketch snapshot in `_fallback` to guarantee zero historical data loss.
  - *HyperLogLog:* Max-bin register folding on downgrade; frozen snapshot union on upgrade.
  - *Misra-Gries:* In-place dynamic $k$ resizing retaining top-$k$ candidates.
- **Empirical Ground-Truth Tracking:**
  Bounded 64-key FIFO shadow sample tracks real-time Weighted Absolute Percentage Error (WAPE).

---

### 3. Partitioned Ingestion Engine + Hedged Monoid Merge (`router/scatter_gather.py`)

- **Monoid Merge Algebra $(\mathcal{S}, \oplus, \mathbf{0})$:**
  Probabilistic sketches form algebraic Monoids, allowing the system to scale writes horizontally across $P$ nodes and merge tiny $(\sim 50\text{--}100\text{ KB})$ summaries into global results in $<500\ \mu\text{s}$:
  - **Count-Min Sketch:** Element-wise matrix addition $\mathbf{C} = \sum_p \mathbf{C}_p$ (Abelian Monoid).
  - **HyperLogLog:** Coordinate-wise register maximum $\mathbf{R}[i] = \max_p \mathbf{R}_p[i]$ (Join-semilattice).
  - **Misra-Gries:** Space-Saving Monoid merge (Agarwal et al., ACM TODS 2013).
- **Asynchronous Hedged Scatter-Gather:**
  Inspired by Jeffrey Dean's *"The Tail at Scale"*:
  1. Sub-queries dispatched asynchronously to partition primaries.
  2. If a worker exceeds $T_{\text{hedge}}$ (empirical P95 latency, $\approx 4.5\text{ ms}$), a duplicate hedged request fires to the hot replica.
  3. First completion accepted; slower straggler cancelled.

---

## 📊 Experimental Results

### Multi-Regime 5-Configuration Benchmark (40,000 Events, 3 Nodes)

| # | Configuration | Pareto Reward | SLA Violations (%) | P99 (ms) | CMS WAPE (%) | Flapping Switches |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Static + Round-Robin** | $+0.5324$ | $41.67\%$ | $111.48$ | $0.962\%$ | $0$ |
| 2 | **Rule-Based + Round-Robin (Phase 1)** | $+0.5179$ | $8.75\%$ | $71.37$ | $2.291\%$ | $21$ |
| 3 | **Rule-Based + Smart Router (Phase 1)** | $+0.5179$ | $8.75\%$ | $56.46$ | $2.465\%$ | $21$ |
| 4 | **LinUCB Bandit + Smart Router (Phase 2)** | $+0.5466$ | $28.75\%$ | $70.77$ | $1.510\%$ | $54$ |
| 5 | **Q-Learning RL + Smart Router (Phase 2)** | **$+0.5644$** | **$11.67\%$** | **$63.39$** | **$1.913\%$** | **$16$** |

*Takeaway: Sequential Q-Learning achieves the highest reward, 24% fewer switches than Rule-Based (70% fewer than LinUCB), and 72% fewer SLA violations than Static.*

### Distributed Hedged Scatter-Gather Benchmark (1,500 Queries, 4 Partitions, 5% Stragglers)

| Configuration | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Max (ms) | Network Overhead |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Unhedged Scatter-Gather** | $3.37$ | $53.89$ | $63.89$ | $73.00$ | $74.90$ | $0.00\%$ |
| **Hedged ($T_{\text{hedge}} = 6.0\text{ ms}$)** | $3.37$ | $8.53$ | $9.14$ | $9.99$ | $11.08$ | $4.85\%$ |
| **Hedged ($T_{\text{hedge}} = 4.5\text{ ms}$)** | **$3.37$** | **$7.03$** | **$7.64$** | **$8.49$** | **$9.58$** | **$5.65\%$** |

*Takeaway: Hedging cuts P99 latency by **$8.6\times$** (from 73.00 ms to 8.49 ms) with only **5.65%** extra network requests.*

---

## 🧪 Testing & Verification

### 1. Offline Unit Tests (No External Services Required)
All 49 tests run in `<0.5 seconds`:
```bash
pytest tests/ -v
```
```
============================== 49 passed in 0.46s ==============================
```

### 2. Regenerating Publication Figures (300 DPI)
```bash
# 1. Run multi-regime simulation
python3 scripts/run_simulation.py

# 2. Run hedged scatter-gather benchmark
python3 scripts/run_scatter_gather_benchmark.py

# 3. Generate 4 publication figures
python3 scripts/plot_results.py
```
Output figures saved in `LNMIIT BTP Report Template/Figures/Phase2/`:
- `fig1_rl_convergence.png` — Q-Learning vs LinUCB online policy convergence
- `fig2_controller_comparison.png` — 5-configuration Pareto trade-off comparison
- `fig3_cu_cms_comparison.png` — Conservative-Update CMS error suppression
- `fig4_hedged_scatter_gather.png` — CDF & tail latency percentiles under stragglers

---

## 🐳 Docker Deployment & End-to-End Verification

### Starting Infrastructure Containers
```bash
docker compose up -d zookeeper kafka redis
```

### Verifying Container Health
```bash
docker compose ps
```
Both Kafka, Zookeeper, and Redis run in isolated Docker containers:
- Kafka broker: `localhost:9092`
- Zookeeper: `localhost:2181`
- Redis state store: `localhost:6379`

### Running the End-to-End Distributed Pipeline
Open 4 terminal tabs:

**Tab 1 — FastAPI Server & Web Dashboard:**
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Tab 2 — Analytics Workers (Partitions 0 and 1):**
```bash
# Partition 0 Primary & Replica
python3 stream/worker.py --worker-id worker-0-a --partition 0 --replica A --template high &
python3 stream/worker.py --worker-id worker-0-b --partition 0 --replica B --template medium &

# Partition 1 Primary & Replica
python3 stream/worker.py --worker-id worker-1-a --partition 1 --replica A --template high &
python3 stream/worker.py --worker-id worker-1-b --partition 1 --replica B --template medium &
```

**Tab 3 — Machine Learning Adaptive Controller:**
```bash
python3 controller/controller.py --mode rl
```

**Tab 4 — Stream Producer with Partitioning & Concept Drift:**
```bash
python3 stream/producer.py --rate 5000 --partitions 2 --shift-interval 30
```

### Sample Distributed Live Queries
```bash
# 1. Distributed Frequency Query (CMS Additive Monoid Merge across partitions with hedging)
curl "http://localhost:8000/query/distributed/frequency?key=item_1"

# 2. Distributed Cardinality Query (HyperLogLog Coordinate-Wise Max Monoid Merge)
curl "http://localhost:8000/query/distributed/cardinality"

# 3. Distributed Heavy Hitters Query (Space-Saving Monoid Merge)
curl "http://localhost:8000/query/distributed/heavy-hitters?n=10"

# 4. Partition Topology Status
curl "http://localhost:8000/metrics/partitions"

# 5. Live RL Telemetry
curl "http://localhost:8000/metrics/controller-stats"

# 6. Switch Controller Algorithm Live to LinUCB or Rule-Based
curl -X POST "http://localhost:8000/control/mode" -H "Content-Type: application/json" -d '{"mode": "linucb"}'
```

---

## 🎓 Why This Excels for BTP & Top FAANG Interviews

1. **Concrete Systems Implementation**: Not just theoretical slides or simulated toy scripts; it is a full, running, distributed C++/Python/Kafka/Redis system with real-time UI telemetry.
2. **Advanced Distributed Systems Principles**: Directly demonstrates Jeff Dean's *"The Tail at Scale"* straggler hedging, Monoid algebras for distributed stream merging, and partition fault-isolation.
3. **Machine Learning in Infrastructure**: Replaces heuristics with principled MDP formulation (TD(0) Q-Learning) and Contextual Bandits (LinUCB) with continuous features, comparing sequential lookahead against myopic greedy actions.
4. **Zero-Downtime Migration**: Mathematical guarantees on preserving sketch counts across dynamic precision adjustments without data loss or pipeline interruption.
