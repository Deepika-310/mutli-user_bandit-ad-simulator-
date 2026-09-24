"""Drive the service with simulated users: the end-to-end demo.

    uvicorn --factory service.main:create_app   # terminal 1
    python -m service.traffic           # terminal 2
Works with any httpx-style client, including FastAPI's TestClient.
"""
import argparse

import numpy as np


def drive_traffic(client, bandit_id, click_rates, n_users, seed=0):
    """Send ``n_users`` simulated users through select -> (maybe) click."""
    rng = np.random.default_rng(seed)
    for _ in range(n_users):
        resp = client.post(f"/bandits/{bandit_id}/select")
        resp.raise_for_status()
        shown = resp.json()
        if rng.random() < click_rates[shown["arm"]]:
            client.post(f"/impressions/{shown['impression_id']}/click").raise_for_status()


def main():
    import httpx

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--algorithm", default="Thompson Sampling")
    parser.add_argument("--users", type=int, default=2000)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.03, 0.05, 0.08])
    args = parser.parse_args()

    with httpx.Client(base_url=args.url, timeout=30) as client:
        bandit = client.post("/bandits", json={
            "name": "traffic-demo", "algorithm": args.algorithm,
            "n_arms": len(args.rates)}).json()
        drive_traffic(client, bandit["id"], args.rates, args.users)
        stats = client.get(f"/bandits/{bandit['id']}/stats").json()
        trend = client.get(f"/bandits/{bandit['id']}/ctr-trend",
                           params={"bucket": max(1, args.users // 10)}).json()

    print(f"true click rates: {args.rates}")
    for a in stats["arms"]:
        print(f"  ad {a['arm']}: {a['impressions']:>5} impressions, "
              f"{a['clicks']:>4} clicks, CTR {a['ctr'] or 0:.3f}")
    print("CTR per block:", " ".join(f"{p['ctr']:.3f}" for p in trend))


if __name__ == "__main__":
    main()
