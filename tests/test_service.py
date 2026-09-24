import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bandit import ALGORITHMS, EpsilonGreedy, ThompsonSampling
from service import storage
from service.main import create_app
from service.traffic import drive_traffic


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(str(tmp_path / "test.db"), seed=123))


def make_bandit(client, algorithm="Thompson Sampling", n_arms=3, **extra):
    resp = client.post("/bandits", json={"name": "t", "algorithm": algorithm,
                                         "n_arms": n_arms, **extra})
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---- bandit CRUD and validation ----------------------------------------------------

def test_health_and_algorithms(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/algorithms").json() == ALGORITHMS


def test_create_and_fetch_bandit(client):
    created = make_bandit(client, "UCB1", 4, ucb_c=1.5)
    assert created["params"]["ucb_c"] == 1.5
    assert client.get(f"/bandits/{created['id']}").json() == created
    assert [b["id"] for b in client.get("/bandits").json()] == [created["id"]]


@pytest.mark.parametrize("body", [
    {"name": "x", "algorithm": "Nope", "n_arms": 3},
    {"name": "x", "algorithm": "UCB1", "n_arms": 1},
    {"name": "x", "algorithm": "UCB1", "n_arms": 99},
    {"name": "", "algorithm": "UCB1", "n_arms": 3},
    {"name": "x", "algorithm": "Epsilon-Greedy", "n_arms": 3, "epsilon": 1.5},
])
def test_invalid_bandit_is_rejected_with_422(client, body):
    assert client.post("/bandits", json=body).status_code == 422


def test_unknown_ids_return_404(client):
    assert client.get("/bandits/999").status_code == 404
    assert client.post("/bandits/999/select").status_code == 404
    assert client.get("/bandits/999/stats").status_code == 404
    assert client.get("/bandits/999/ctr-trend").status_code == 404
    assert client.post("/impressions/999/click").status_code == 404


# ---- select / click flow ------------------------------------------------------------

def test_select_logs_impressions_with_increasing_seq(client):
    bid = make_bandit(client)["id"]
    shown = [client.post(f"/bandits/{bid}/select").json() for _ in range(5)]
    assert [s["seq"] for s in shown] == [1, 2, 3, 4, 5]
    assert all(0 <= s["arm"] < 3 for s in shown)
    assert client.get(f"/bandits/{bid}/stats").json()["impressions"] == 5


def test_click_is_recorded_once(client):
    bid = make_bandit(client)["id"]
    imp = client.post(f"/bandits/{bid}/select").json()["impression_id"]
    first = client.post(f"/impressions/{imp}/click").json()
    second = client.post(f"/impressions/{imp}/click").json()
    assert first["duplicate"] is False and second["duplicate"] is True
    stats = client.get(f"/bandits/{bid}/stats").json()
    assert stats["clicks"] == 1  # the duplicate did not double count


def test_stats_match_events(client):
    bid = make_bandit(client, "Epsilon-Greedy", 2, epsilon=1.0)["id"]  # pure random
    for i in range(40):
        imp = client.post(f"/bandits/{bid}/select").json()
        if i % 4 == 0:
            client.post(f"/impressions/{imp['impression_id']}/click")
    stats = client.get(f"/bandits/{bid}/stats").json()
    assert stats["impressions"] == 40 and stats["clicks"] == 10
    assert sum(a["impressions"] for a in stats["arms"]) == 40
    assert stats["ctr"] == pytest.approx(0.25)


def test_bandits_are_isolated_from_each_other(client):
    a, b = make_bandit(client)["id"], make_bandit(client)["id"]
    for _ in range(3):
        client.post(f"/bandits/{a}/select")
    assert client.get(f"/bandits/{b}/stats").json()["impressions"] == 0


def test_same_seed_gives_same_selection_sequence(tmp_path):
    def arms(db):
        c = TestClient(create_app(str(tmp_path / db), seed=5))
        bid = make_bandit(c)["id"]
        return [c.post(f"/bandits/{bid}/select").json()["arm"] for _ in range(15)]
    assert arms("a.db") == arms("b.db")


# ---- learning behaviour through the API -----------------------------------------------

@pytest.mark.parametrize("algorithm", ["Thompson Sampling", "UCB1", "Epsilon-Greedy"])
def test_service_learns_the_best_ad(client, algorithm):
    rates = [0.10, 0.20, 0.60]
    bid = make_bandit(client, algorithm, 3, ucb_c=0.5)["id"]
    drive_traffic(client, bid, rates, n_users=300, seed=1)
    stats = client.get(f"/bandits/{bid}/stats").json()
    assert stats["best_arm_estimate"] == 2
    share_best = stats["arms"][2]["impressions"] / stats["impressions"]
    assert share_best > 0.6


def test_ctr_trend_improves_as_it_learns(client):
    bid = make_bandit(client, "Thompson Sampling", 3)["id"]
    drive_traffic(client, bid, [0.05, 0.10, 0.50], n_users=400, seed=2)
    trend = client.get(f"/bandits/{bid}/ctr-trend", params={"bucket": 100}).json()
    assert [t["block"] for t in trend] == [0, 1, 2, 3]
    assert sum(t["impressions"] for t in trend) == 400
    assert trend[-1]["ctr"] > trend[0]["ctr"]
    # window function: cumulative CTR over all blocks equals overall CTR
    stats = client.get(f"/bandits/{bid}/stats").json()
    assert trend[-1]["cumulative_ctr"] == pytest.approx(stats["ctr"])


def test_ctr_trend_validates_bucket(client):
    bid = make_bandit(client)["id"]
    assert client.get(f"/bandits/{bid}/ctr-trend", params={"bucket": 0}).status_code == 422


# ---- agent rebuilt from SQL counts == agent that saw every event -------------------

def test_load_stats_equals_incremental_updates():
    rng = np.random.default_rng(0)
    arms = rng.integers(0, 3, 200)
    rewards = (rng.random(200) < 0.4).astype(int)
    live = EpsilonGreedy(3, rng=np.random.default_rng(1))
    for a, r in zip(arms, rewards, strict=True):
        live.update(np.array([a]), np.array([r]))
    pulls = np.bincount(arms, minlength=3)
    clicks = np.bincount(arms, weights=rewards, minlength=3)
    rebuilt = EpsilonGreedy(3, rng=np.random.default_rng(1))
    rebuilt.load_stats(pulls, clicks)
    assert np.allclose(rebuilt.Q, live.Q) and np.allclose(rebuilt.N, live.N)
    assert rebuilt.t == 200


def test_load_stats_rebuilds_thompson_posterior():
    ts = ThompsonSampling(2, rng=np.random.default_rng(0))
    ts.load_stats([10, 4], [3, 4])
    assert ts.alpha[0].tolist() == [4, 5] and ts.beta[0].tolist() == [8, 1]


@pytest.mark.parametrize("pulls, clicks", [([1, 1], [2, 0]), ([1], [1]), ([1, 1], [-1, 0])])
def test_load_stats_rejects_inconsistent_counts(pulls, clicks):
    with pytest.raises(ValueError):
        EpsilonGreedy(2, rng=np.random.default_rng(0)).load_stats(pulls, clicks)


# ---- SQL layer ---------------------------------------------------------------------------

def test_foreign_key_blocks_orphan_impressions(tmp_path):
    path = str(tmp_path / "fk.db")
    storage.init_db(path)
    store = storage.Store(storage.connect(path))
    with pytest.raises(sqlite3.IntegrityError), store.transaction():
        store.add_impression(bandit_id=42, arm=0)


def test_failed_transaction_rolls_back(tmp_path):
    path = str(tmp_path / "rb.db")
    storage.init_db(path)
    store = storage.Store(storage.connect(path))
    bandit = store.create_bandit("b", "UCB1", 2, {})
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.add_impression(bandit["id"], 0)
            raise RuntimeError("boom")
    assert store.arm_stats(bandit["id"], 2) == ([0, 0], [0, 0])


# ---- /simulate ----------------------------------------------------------------------------

def test_simulate_endpoint_returns_summary(client):
    resp = client.post("/simulate", json={"probs": [0.3, 0.5, 0.7], "steps": 300,
                                          "n_runs": 10, "seed": 1})
    assert resp.status_code == 200
    rows = resp.json()["summary"]
    assert [r["Algorithm"] for r in rows] == ALGORITHMS


@pytest.mark.parametrize("body", [
    {"probs": [0.5]},
    {"probs": [0.5, 1.5]},
    {"probs": [0.2, 0.4], "steps": 10**6},
    {"probs": [0.2, 0.4], "algorithms": ["Nope"]},
])
def test_simulate_rejects_bad_input(client, body):
    assert client.post("/simulate", json=body).status_code == 422


def test_concurrent_impressions_get_unique_sequence_numbers(tmp_path):
    # Each thread has its own connection, like separate API workers sharing one DB file.
    from concurrent.futures import ThreadPoolExecutor

    path = str(tmp_path / "conc.db")
    storage.init_db(path)
    setup = storage.Store(storage.connect(path))
    bandit_id = setup.create_bandit("b", "UCB1", 2, {})["id"]

    def worker(_):
        store = storage.Store(storage.connect(path))
        seqs = []
        for _ in range(25):
            with store.transaction():
                seqs.append(store.add_impression(bandit_id, 0)[1])
        store.conn.close()
        return seqs

    with ThreadPoolExecutor(8) as pool:
        seqs = [s for chunk in pool.map(worker, range(8)) for s in chunk]
    assert sorted(seqs) == list(range(1, 201))
