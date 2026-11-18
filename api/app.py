"""
FastAPI Analytics API — serves sketch query results + cluster metrics + dashboard.

Endpoints:
  GET  /                           → HTML dashboard
  GET  /query/frequency?key=X      → CMS frequency estimate
  GET  /query/cardinality          → HLL cardinality estimate
  GET  /query/heavy-hitters?n=10   → Misra-Gries top-N
  GET  /metrics/cluster            → all worker health metrics
  GET  /metrics/sketches           → sketch parameters per worker
  GET  /metrics/router?q=frequency → router scores for each node
  GET  /metrics/controller-log     → recent controller decisions
  GET  /metrics/query-log          → recent routing decisions
  POST /control/set-params         → manually override a worker's template
  GET  /health                     → liveness probe

The adaptive controller runs as a background asyncio task on startup.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from state.store import StateStore
from router.router import LoadAwareRouter
from router.scatter_gather import HedgedScatterGatherEngine
from controller.controller import AdaptiveController

# Singletons (initialised at startup)
store: StateStore = None
router_instance: LoadAwareRouter = None
controller: AdaptiveController = None
scatter_engine: HedgedScatterGatherEngine = None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    global store, router_instance, controller, scatter_engine
    store           = StateStore()
    router_instance = LoadAwareRouter(store=store)
    controller      = AdaptiveController(store=store)
    scatter_engine  = HedgedScatterGatherEngine(store=store, router=router_instance)
    asyncio.create_task(controller.run())
    print("[API] Server started. Controller & Scatter-Gather Engine running in background.")
    yield
    if controller:
        controller.stop()

app = FastAPI(
    title="BTP Sketch Analytics API",
    description="Adaptive Real-Time Stream Analytics with Probabilistic Sketches",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static dashboard files
_DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"
if (_DASHBOARD_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(_DASHBOARD_DIR / "static")), name="static")

# Prevent browser caching of static files during development
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheStaticMiddleware)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def dashboard():
    index = _DASHBOARD_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Dashboard not found — run from project root"})


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Query endpoints (routed to best worker)
# ---------------------------------------------------------------------------

@app.get("/query/frequency")
async def query_frequency(key: str = Query(..., description="Item key to query")):
    """
    Return the live CMS frequency estimate for the given key.
    Automatically routes to the best worker via the load-aware router and
    queries that worker's Count-Min Sketch over Redis RPC.
    """
    best_wid, score, ranked = router_instance.route("frequency")
    if not best_wid:
        raise HTTPException(503, "No workers available")

    metrics = store.get_worker_metrics(best_wid)
    if not metrics:
        raise HTTPException(503, f"Worker {best_wid} has no metrics")

    # Execute live point-frequency RPC against the routed worker's CMS
    rpc_res = await asyncio.to_thread(store.query_worker_frequency, best_wid, key, 1.0)
    estimated_freq = rpc_res.get("estimated_frequency") if rpc_res else None
    cms_error_bound = (
        rpc_res.get("cms_error_bound")
        if rpc_res
        else metrics.get("cms_error_estimate", 0)
    )
    cms_total_count = (
        rpc_res.get("cms_total_count")
        if rpc_res
        else metrics.get("cms_total_count", 0)
    )

    cms_w = metrics.get("cms_width") or 2048
    cms_d = metrics.get("cms_depth") or 5

    return {
        "key":                 key,
        "estimated_frequency": estimated_freq,
        "routed_to":           best_wid,
        "router_score":        round(score, 4),
        "all_scores":          [{"worker": w, "score": round(s, 4)} for w, s in ranked],
        "cms_error_bound":     cms_error_bound,
        "cms_total_count":     cms_total_count,
        "empirical_rel_error": metrics.get("cms_empirical_rel_error"),
        "rpc_live":            rpc_res is not None,
        "note": (
            f"Conservative-Update CMS estimate ≤ true_count + {cms_error_bound} "
            f"(ε={round(2.71828 / cms_w, 5)}, δ={round(2.71828 ** -cms_d, 5)})"
        ),
        "worker_metrics": {
            "template":       metrics.get("template"),
            "p95_latency_ms": metrics.get("p95_latency_ms"),
            "p99_latency_ms": metrics.get("p99_latency_ms"),
            "memory_percent": metrics.get("memory_percent"),
        },
    }


@app.get("/query/cardinality")
async def query_cardinality():
    """Return the HLL cardinality estimate from the best worker."""
    best_wid, score, ranked = router_instance.route("cardinality")
    if not best_wid:
        raise HTTPException(503, "No workers available")

    metrics = store.get_worker_metrics(best_wid)
    return {
        "estimated_cardinality": metrics.get("hll_cardinality", 0),
        "relative_error":        round(metrics.get("hll_relative_error", 0.0), 4),
        "hll_precision":         metrics.get("hll_precision", 12),
        "routed_to":             best_wid,
        "router_score":          round(score, 4),
    }


@app.get("/query/heavy-hitters")
async def query_heavy_hitters(n: int = Query(10, ge=1, le=100)):
    """Return the top-N Misra-Gries heavy hitters from the best worker."""
    best_wid, score, ranked = router_instance.route("heavy_hitters")
    if not best_wid:
        raise HTTPException(503, "No workers available")

    metrics = store.get_worker_metrics(best_wid)
    hh = metrics.get("mg_heavy_hitters", [])[:n]
    return {
        "heavy_hitters":    hh,
        "window_total":     metrics.get("mg_window_total", 0),
        "error_bound":      round(metrics.get("mg_error_bound", 0), 2),
        "mg_k":             metrics.get("mg_k", 200),
        "routed_to":        best_wid,
        "router_score":     round(score, 4),
    }


# ---------------------------------------------------------------------------
# Distributed Scatter-Gather Endpoints (Partitioned Ingestion + Monoid Merge)
# ---------------------------------------------------------------------------

@app.get("/query/distributed/frequency")
async def query_distributed_frequency(
    key: str = Query(..., description="Item key to query across all partitions"),
    hedged: bool = Query(True, description="Enable hedged request execution on stragglers"),
    hedge_timeout_ms: float = Query(6.0, ge=0.5, le=100.0, description="Hedge dispatch timeout in ms"),
):
    """
    Scatter-gather frequency query:
      1. Dispatches parallel sub-queries to primary partition workers.
      2. If primary exceeds `hedge_timeout_ms`, dispatches hedged replica request.
      3. Performs element-wise Count-Min Monoid merge (sum of partition frequencies).
    """
    if not scatter_engine:
        raise HTTPException(503, "Scatter-Gather engine not ready")
    return await scatter_engine.scatter_gather_frequency(
        key=key,
        hedge_timeout_ms=hedge_timeout_ms,
        hedged=hedged,
    )


@app.get("/query/distributed/cardinality")
async def query_distributed_cardinality(
    hedged: bool = Query(True, description="Enable hedged request execution on stragglers"),
    hedge_timeout_ms: float = Query(6.0, ge=0.5, le=100.0, description="Hedge dispatch timeout in ms"),
):
    """
    Scatter-gather cardinality query:
      1. Gathers HLL register vectors from all partitions in parallel with hedging.
      2. Performs coordinate-wise maximum Monoid merge: R_global[i] = max_p R_p[i].
      3. Returns cluster-wide distinct count.
    """
    if not scatter_engine:
        raise HTTPException(503, "Scatter-Gather engine not ready")
    return await scatter_engine.scatter_gather_cardinality(
        hedge_timeout_ms=hedge_timeout_ms,
        hedged=hedged,
    )


@app.get("/query/distributed/heavy-hitters")
async def query_distributed_heavy_hitters(
    n: int = Query(10, ge=1, le=100),
    hedged: bool = Query(True, description="Enable hedged request execution on stragglers"),
    hedge_timeout_ms: float = Query(6.0, ge=0.5, le=100.0, description="Hedge dispatch timeout in ms"),
):
    """
    Scatter-gather heavy hitters query:
      1. Gathers candidate summaries from all partitions in parallel with hedging.
      2. Performs Agarwal et al. Space-Saving Monoid merge.
      3. Returns global top-N heavy hitters.
    """
    if not scatter_engine:
        raise HTTPException(503, "Scatter-Gather engine not ready")
    return await scatter_engine.scatter_gather_heavy_hitters(
        n=n,
        hedge_timeout_ms=hedge_timeout_ms,
        hedged=hedged,
    )


@app.get("/metrics/partitions")
async def metrics_partitions():
    """Return active partition topology (Partitions, Primaries, Hedged Replicas)."""
    if not scatter_engine:
        return {}
    topo = scatter_engine.get_partition_topology()
    result = {}
    for p_id, candidates in topo.items():
        primary, replica = scatter_engine._select_partition_pair(candidates)
        result[f"partition_{p_id}"] = {
            "primary": primary["worker_id"],
            "replica": replica["worker_id"] if replica else None,
            "total_nodes": len(candidates),
        }
    return {"partitions": result, "total_partitions": len(topo)}


# ---------------------------------------------------------------------------
# Metrics endpoints
# ---------------------------------------------------------------------------

@app.get("/metrics/cluster")
async def metrics_cluster():
    """Return live health metrics for all workers."""
    cluster = store.get_cluster_metrics()
    latency_histories = {
        wid: store.get_latency_history(wid)
        for wid in cluster
    }
    return {
        "workers":          cluster,
        "latency_histories": latency_histories,
        "worker_count":     len(cluster),
    }


@app.get("/metrics/sketches")
async def metrics_sketches():
    """Return current sketch parameters for all workers."""
    cluster = store.get_cluster_metrics()
    result = {}
    for wid, m in cluster.items():
        result[wid] = {
            "template":     m.get("template"),
            "cms_width":    m.get("cms_width"),
            "cms_depth":    m.get("cms_depth"),
            "hll_precision":m.get("hll_precision"),
            "mg_k":         m.get("mg_k"),
        }
    return result


@app.get("/metrics/router")
async def metrics_router(q: str = Query("frequency")):
    """Return router scores for each node for the given query type."""
    return {"query_type": q, "ranked_nodes": router_instance.get_node_scores(q)}


@app.get("/metrics/controller-log")
async def metrics_controller_log(n: int = Query(20, ge=1, le=100)):
    return {"decisions": store.get_controller_log(n)}


@app.get("/metrics/controller-stats")
async def metrics_controller_stats():
    """Return live RL/Rule controller telemetry (mode, epsilon, rewards, drift events)."""
    if controller:
        return controller.get_stats()
    cached = store.get_controller_stats()
    return cached or {"mode": "unknown"}


@app.get("/metrics/query-log")
async def metrics_query_log(n: int = Query(50, ge=1, le=200)):
    return {"routing_decisions": store.get_query_log(n)}


# ---------------------------------------------------------------------------
# Control endpoints
# ---------------------------------------------------------------------------

class SetParamsRequest(BaseModel):
    worker_id: str
    template: str  # "low" | "medium" | "high"


class SetModeRequest(BaseModel):
    mode: str  # "rl" | "linucb" | "rule"


@app.post("/control/mode")
async def control_set_mode(req: SetModeRequest):
    """Switch adaptive controller live between 'rl', 'linucb', and 'rule'."""
    if req.mode not in {"rl", "linucb", "rule"}:
        raise HTTPException(400, "mode must be 'rl', 'linucb', or 'rule'")
    controller.set_mode(req.mode)
    store.log_controller_decision(
        "cluster",
        f"Manual override: controller mode switched to '{req.mode.upper()}'"
    )
    return {"status": "ok", "mode": req.mode}


@app.post("/control/set-params")
async def control_set_params(req: SetParamsRequest):
    """Manually override a worker's precision template."""
    valid = {"low", "medium", "high"}
    if req.template not in valid:
        raise HTTPException(400, f"template must be one of {valid}")
    store.send_command(req.worker_id, {
        "action": "set_template",
        "template": req.template,
    })
    store.log_controller_decision(
        req.worker_id,
        f"Manual override: template set to '{req.template}'"
    )
    return {"status": "ok", "worker_id": req.worker_id, "template": req.template}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api.app:app", host=host, port=port, reload=True)
