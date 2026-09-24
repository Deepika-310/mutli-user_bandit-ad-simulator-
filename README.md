# Multi-Armed Bandit Ad Simulator and Decision Service

**Which ad should you show when you don't know which one people click?** This project implements four bandit algorithms and lets you (1) *measure* them in a simulator, (2) *explore* them in a Streamlit dashboard, and (3) *serve* them as a REST API backed by SQL, with tests, Docker and CI.

| Algorithm | Idea |
|---|---|
| **Epsilon-Greedy** | With probability ε show a random ad, otherwise the best-looking one |
| **Decaying Epsilon** | Same, with ε shrinking as `ε / (1 + t/τ)` |
| **UCB1** | Show `argmax Q + sqrt(c · ln t / N)` (optimism under uncertainty) |
| **Thompson Sampling** | Sample each ad's click rate from its Beta posterior, show the max |

## Measured results

Every number is reproducible: `python benchmark.py` (seed 42, 200 runs per cell, ~40 s) writes [results/benchmark.md](results/benchmark.md) and `results/benchmark.csv`.

**Regret** = expected clicks lost vs. always showing the best ad (lower is better), mean over 200 runs.

| Scenario | Random | e-greedy | Best algorithm | Its regret | Less regret than random | Less than e-greedy |
|---|---|---|---|---|---|---|
| 3 ads, clear gaps (5,000 users) | 999 | 112 | UCB1 c=0.25 | 14.7 | **99%** | **87%** |
| 4 ads, 2-5% CTR (10,000 users) | 150 | 46 | Thompson | 36.7 | **76%** | **20%** |
| 10 ads, small gaps (10,000 users) | 2,250 | 316 | UCB1 c=0.25 | 96.7 | **96%** | **69%** |
| Click rates reverse halfway (4,000 users) | 801 | 754 | UCB1 c=0.25 + step 0.1 | 83.7 | **90%** | **89%** |

Honest reading of the full tables:
- **Nothing wins everywhere.** UCB1 with the textbook constant `c=2` *loses* to plain e-greedy in the realistic low-CTR scenario, because its exploration bonus is on the scale of 1 while the click rates are around 0.03. Thompson Sampling is the most robust across scenarios, and needs no tuning.
- **Sample-average estimators cannot track change.** After the reversal, e-greedy's regret stays at 94% of random play (754 vs 801); adding a constant step size cuts it to 156 (**-79%**), and discounting Thompson's posterior cuts 429 to 180 (**-58%**).

### What each engineering change bought (1,000 runs)

| Change | Before | After |
|---|---|---|
| Decaying epsilon: original `ε/t` + lowest-index ties -> random ties -> `ε/(1+t/50)` | **82%** of runs locked onto a bad ad, regret 631 | **8%**, regret 54 |
| Pseudo-regret instead of realised regret | 77% of runs dip below zero, spread 28.6 | 0% dip, spread 20.9 (1.9x fewer runs for the same confidence) |
| Shared simulated-user randomness (common random numbers) | CI on "Thompson minus e-greedy" clicks: 2.44 | 1.47 (2.75x variance reduction) |
| Agents batched across runs | ~270 ms per run | ~2 ms per run (~120x; timing varies by machine) |

## Architecture

```
                 +------------------+        +--------------------------+
 Streamlit  ---> |  simulation.py   | <----- |  bandit.py (agents)      |
 dashboard       |  environment +   |        |  EpsilonGreedy / UCB /   |
 (app.py)        |  run_experiment  |        |  ThompsonSampling        |
                 +--------+---------+        +------------+-------------+
                          |                               ^ load_stats(pulls, clicks)
                 analysis.py / plots.py                   |
                                                +---------+-----------+
 client --HTTP--> service/main.py (FastAPI) --> | service/storage.py  | --> SQLite
                  Pydantic validation           | SQL, transactions   |     (WAL, indexes)
                                                +---------------------+
```

The agent API is two methods (`select_arm`, `update`) plus `load_stats`, so the same algorithm code drives both the simulator and the service. The service is **stateless**: on every `/select` it computes per-arm impression/click counts in SQL, rebuilds the agent from them, chooses an ad and logs the impression, all inside one `BEGIN IMMEDIATE` transaction.

## REST API

```bash
uvicorn --factory service.main:create_app     # docs at http://127.0.0.1:8000/docs
python -m service.traffic --users 2000        # drive it with simulated users
```

| Endpoint | Purpose |
|---|---|
| `POST /bandits` | Create a bandit (algorithm, number of ads, parameters); 422 on invalid input |
| `POST /bandits/{id}/select` | "Which ad do I show?" Logs an impression |
| `POST /impressions/{id}/click` | Record a click (idempotent) |
| `GET /bandits/{id}/stats` | Per-ad impressions, clicks, CTR (SQL `GROUP BY`) |
| `GET /bandits/{id}/ctr-trend` | CTR per block + cumulative CTR (SQL window functions) |
| `POST /simulate` | Run a multi-seed experiment and return the summary table |

Checked against a real uvicorn server: 1,500 simulated users sent 72% of impressions to the best ad; 16 threads x 50 concurrent requests produced 800 unique, gap-free sequence numbers.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py                       # dashboard
docker compose up --build                  # API on :8000, dashboard on :8501
```

Development: `pip install -r requirements-dev.txt`, then `ruff check .` and `pytest` (79 tests). CI (GitHub Actions) runs lint, tests on Python 3.10 and 3.12, a benchmark smoke test, and builds the Docker image.

## Project structure

```
bandit.py       Agents, batched over independent runs; random tie-breaking; load_stats()
simulation.py   Bernoulli ad environment (optional mid-run shift) + run_experiment()
analysis.py     Regret, convergence, confidence intervals, plain-English interpretation
plots.py        Matplotlib figures (object-oriented API)
app.py          Streamlit dashboard
benchmark.py    Reproducible benchmark and ablations -> results/
service/        FastAPI app, Pydantic schemas, SQLite storage layer, traffic driver
tests/          pytest suite (agents, simulator, benchmark, API, SQL, concurrency)
```

## Design decisions worth knowing

- **Pseudo-regret** uses the ads' true click rates, so it is non-negative and low-variance; realised regret is noisy and can go negative.
- **Random tie-breaking**: `np.argmax` returns the lowest index, silently favouring ad 0 while all estimates are equal.
- **An impression without a click counts as a miss**, as in real ad feedback. The trade-off: a click that arrives late is briefly counted as a miss.
- **SQLite in WAL mode**: readers don't block the writer. For multi-node production use, swap in PostgreSQL; the SQL used here is standard (window functions, CTEs).

## Limitations

- Click rates are simulated (Bernoulli), not real ad data; results show algorithm behaviour, not real-world CTR lift.
- No context features (user, device), so "best ad" is global; the natural extension is a contextual bandit (LinUCB).
- The service assumes the sample-average update, so constant-step and discounted variants exist only in the simulator.
- Docker and CI files were written but I have not built the image or run the workflow in my environment.
