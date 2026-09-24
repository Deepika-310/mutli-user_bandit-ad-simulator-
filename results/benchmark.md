# Benchmark results

Seed 42, 200 runs per algorithm per scenario, Python 3.10.11, NumPy 2.2.6.
Regret = expected clicks lost vs always showing the best ad (lower is better). 'Best ad found' = share of runs where the best ad got >= 50% of impressions in the last 20% of users. 'Users to converge' = users until the best ad is shown >= 90% of the time (median over runs that get there; after the shift in the last scenario).

## Part 1: algorithm comparison

### Easy: 3 ads with clear gaps
click rates [0.3, 0.5, 0.7], 5000 users

| Algorithm | Final regret | +/- 95% CI | Less regret than random | Less regret than e-greedy | Best ad found (runs) | Median users to converge |
|---|---|---|---|---|---|---|
| Uniform random (baseline) | 999.4 | 1.8 | 0% | -791% | 0% | never |
| Epsilon-Greedy (e=0.1) | 112.2 | 3.4 | 89% | +0% | 100% | 113 |
| Decaying Epsilon | 74.5 | 28.4 | 93% | +34% | 96% | 99 |
| UCB1 (c=2) | 82.3 | 2.0 | 92% | +27% | 100% | 359 |
| UCB1 (c=0.25) | 14.7 | 1.2 | 99% | +87% | 100% | 108 |
| Thompson Sampling | 16.8 | 1.1 | 98% | +85% | 100% | 130 |

### Realistic: 4 ads with 2-5% click rates
click rates [0.02, 0.03, 0.04, 0.05], 10000 users

| Algorithm | Final regret | +/- 95% CI | Less regret than random | Less regret than e-greedy | Best ad found (runs) | Median users to converge |
|---|---|---|---|---|---|---|
| Uniform random (baseline) | 150.0 | 0.2 | 0% | -228% | 0% | never |
| Epsilon-Greedy (e=0.1) | 45.8 | 5.3 | 69% | +0% | 86% | 704 |
| Decaying Epsilon | 95.6 | 12.8 | 36% | -109% | 44% | 211 |
| UCB1 (c=2) | 121.8 | 0.7 | 19% | -166% | 0% | 7671 |
| UCB1 (c=0.25) | 79.5 | 1.5 | 47% | -74% | 78% | 2853 |
| Thompson Sampling | 36.7 | 2.3 | 76% | +20% | 99% | 3338 |

### Many ads: 10 ads, small gaps
click rates [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55], 10000 users

| Algorithm | Final regret | +/- 95% CI | Less regret than random | Less regret than e-greedy | Best ad found (runs) | Median users to converge |
|---|---|---|---|---|---|---|
| Uniform random (baseline) | 2250.3 | 1.8 | 0% | -612% | 0% | never |
| Epsilon-Greedy (e=0.1) | 316.0 | 20.2 | 86% | +0% | 95% | 527 |
| Decaying Epsilon | 429.1 | 73.4 | 81% | -36% | 56% | 271 |
| UCB1 (c=2) | 482.1 | 5.8 | 79% | -53% | 98% | 3218 |
| UCB1 (c=0.25) | 96.7 | 4.4 | 96% | +69% | 100% | 695 |
| Thompson Sampling | 114.7 | 5.0 | 95% | +64% | 100% | 1421 |

### Changing world: rates reverse halfway
click rates [0.3, 0.5, 0.7], 4000 users, rates reverse at user 2000

| Algorithm | Final regret | +/- 95% CI | Less regret than random | Less regret than e-greedy | Best ad found (runs) | Median users to converge |
|---|---|---|---|---|---|---|
| Uniform random (baseline) | 801.1 | 1.4 | 0% | -6% | 0% | never |
| Epsilon-Greedy (e=0.1) | 754.2 | 10.0 | 6% | +0% | 10% | 1766 |
| Decaying Epsilon | 794.5 | 12.0 | 1% | -5% | 4% | 1093 |
| UCB1 (c=2) | 105.6 | 3.8 | 87% | +86% | 100% | 81 |
| UCB1 (c=0.25) | 144.1 | 16.5 | 82% | +81% | 98% | 293 |
| Thompson Sampling | 428.9 | 21.8 | 46% | +43% | 86% | 1065 |
| Epsilon-Greedy (e=0.1, step 0.1) | 156.1 | 4.7 | 81% | +79% | 100% | 95 |
| UCB1 (c=0.25, step 0.1) | 83.7 | 5.8 | 90% | +89% | 100% | 53 |
| Thompson Sampling (discount 0.99) | 179.5 | 2.1 | 78% | +76% | 100% | 139 |

## Part 2: what each v2 change bought

### Decaying Epsilon: original vs fixed (1000 runs, easy scenario, 2000 users)

| Variant | Runs locked onto a bad ad | Final regret |
|---|---|---|
| v1: eps/t, lowest-index ties | 82.3% | 630.5 +/- 18.1 |
| + random tie-breaking | 46.4% | 253.2 +/- 18.2 |
| + new schedule eps/(1+t/50)  (= v2) | 7.9% | 54.1 +/- 7.1 |

### Regret estimator (1000 runs)

| Estimator | Std of final regret | 95% CI half-width (50 runs) | Runs dipping below 0 |
|---|---|---|---|
| Realised (v1): best rate - actual click | 28.6 | 7.9 | 77% |
| Pseudo (v2): best rate - shown ad's rate | 20.9 | 5.8 | 0% |

Pseudo-regret has 1.4x lower spread, so the same confidence needs 1.9x fewer runs.

### Common random numbers: Thompson minus e-greedy (1000 runs)

| Metric | CI half-width, independent randomness | CI half-width, shared randomness (v2) | Variance reduction |
|---|---|---|---|
| Total clicks | 2.44 | 1.47 | 2.75x |
| Final pseudo-regret | 1.65 | 1.44 | 1.31x |

Sharing the simulated users' randomness clearly helps click-based comparisons. For regret-based ones the effect is small (about 1.0-1.3x in my runs), because pseudo-regret ignores the click draws and depends only on the ads chosen.

### Vectorising across runs (timing, varies by machine)

| Approach | Time per run (ms) |
|---|---|
| One run at a time (loop over runs) | 273.2 |
| Batched: 200 runs in one loop (v2) | 2.24 |

Batching is 122x faster per run.
