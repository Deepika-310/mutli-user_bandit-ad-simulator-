"""FastAPI service: serve a bandit as a decision API with SQL-backed feedback.

Flow for a real ad slot:
    POST /bandits                         create a bandit for N ads
    POST /bandits/{id}/select             "which ad do I show?" -> logs an impression
    POST /impressions/{impression_id}/click   the user clicked
    GET  /bandits/{id}/stats | /ctr-trend    monitor it
An impression with no click counts as a miss, which is how ad feedback arrives in
practice (a click is an event; a non-click is silence).
"""
import os
import threading

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query

from analysis import summarize
from bandit import ALGORITHMS, make_agent
from service import storage
from service.schemas import (
    ArmStats,
    BanditCreate,
    BanditOut,
    ClickOut,
    SelectOut,
    SimulateIn,
    SimulateOut,
    StatsOut,
    TrendPoint,
)
from simulation import run_experiment

DESCRIPTION = (
    "Multi-armed bandit ad selection as a REST service. The agent is rebuilt from "
    "per-arm impression and click counts stored in SQLite on every request."
)


def create_app(db_path=None, seed=None):
    """App factory. ``seed`` makes arm selection reproducible (used by tests)."""
    db_path = db_path or os.environ.get("BANDIT_DB", "bandit.db")
    storage.init_db(db_path)
    seed_rng = np.random.default_rng(seed)
    seed_lock = threading.Lock()  # Generator objects are not thread-safe

    app = FastAPI(title="Bandit Ad Service", version="1.0.0", description=DESCRIPTION)

    def get_store():
        conn = storage.connect(db_path)
        try:
            yield storage.Store(conn)
        finally:
            conn.close()

    def bandit_or_404(store, bandit_id):
        bandit = store.get_bandit(bandit_id)
        if bandit is None:
            raise HTTPException(404, f"bandit {bandit_id} not found")
        return bandit

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/algorithms", response_model=list[str])
    def algorithms():
        return ALGORITHMS

    @app.post("/bandits", response_model=BanditOut, status_code=201)
    def create_bandit(body: BanditCreate, store: storage.Store = Depends(get_store)):
        params = {"epsilon": body.epsilon, "decay_scale": body.decay_scale,
                  "ucb_c": body.ucb_c}
        return store.create_bandit(body.name, body.algorithm, body.n_arms, params)

    @app.get("/bandits", response_model=list[BanditOut])
    def list_bandits(store: storage.Store = Depends(get_store)):
        return store.list_bandits()

    @app.get("/bandits/{bandit_id}", response_model=BanditOut)
    def get_bandit(bandit_id: int, store: storage.Store = Depends(get_store)):
        return bandit_or_404(store, bandit_id)

    @app.post("/bandits/{bandit_id}/select", response_model=SelectOut)
    def select_ad(bandit_id: int, store: storage.Store = Depends(get_store)):
        with seed_lock:
            request_rng = np.random.default_rng(int(seed_rng.integers(2**63)))
        # One write transaction: read counts, choose, log the impression atomically.
        with store.transaction():
            bandit = bandit_or_404(store, bandit_id)
            pulls, clicks = store.arm_stats(bandit_id, bandit["n_arms"])
            agent = make_agent(bandit["algorithm"], bandit["n_arms"], 1, request_rng,
                               **bandit["params"])
            agent.load_stats(pulls, clicks)
            arm = int(agent.select_arm()[0])
            impression_id, seq = store.add_impression(bandit_id, arm)
        return SelectOut(impression_id=impression_id, arm=arm, seq=seq)

    @app.post("/impressions/{impression_id}/click", response_model=ClickOut)
    def record_click(impression_id: int, store: storage.Store = Depends(get_store)):
        outcome = store.record_click(impression_id)
        if outcome is None:
            raise HTTPException(404, f"impression {impression_id} not found")
        return ClickOut(impression_id=impression_id, clicked=True,
                        duplicate=outcome == "duplicate")

    @app.get("/bandits/{bandit_id}/stats", response_model=StatsOut)
    def stats(bandit_id: int, store: storage.Store = Depends(get_store)):
        bandit = bandit_or_404(store, bandit_id)
        pulls, clicks = store.arm_stats(bandit_id, bandit["n_arms"])
        arms = [ArmStats(arm=i, impressions=p, clicks=c, ctr=c / p if p else None)
                for i, (p, c) in enumerate(zip(pulls, clicks, strict=True))]
        total_p, total_c = sum(pulls), sum(clicks)
        seen = [a for a in arms if a.impressions]
        best = max(seen, key=lambda a: a.ctr).arm if seen else None
        return StatsOut(bandit=bandit, impressions=total_p, clicks=total_c,
                        ctr=total_c / total_p if total_p else None,
                        best_arm_estimate=best, arms=arms)

    @app.get("/bandits/{bandit_id}/ctr-trend", response_model=list[TrendPoint])
    def ctr_trend(bandit_id: int, bucket: int = Query(100, ge=1, le=100000),
                  store: storage.Store = Depends(get_store)):
        bandit_or_404(store, bandit_id)
        return store.ctr_trend(bandit_id, bucket)

    @app.post("/simulate", response_model=SimulateOut)
    def simulate(body: SimulateIn):
        shift_at = body.steps // 2 if body.shift else None
        experiments = [run_experiment(name, body.probs, body.steps, body.n_runs, body.seed,
                                      shift_at) for name in body.algorithms]
        return SimulateOut(settings=body.model_dump(), summary=summarize(experiments))

    return app
