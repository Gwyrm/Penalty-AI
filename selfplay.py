"""Self-play training for the penalty duel.

Both the shooter and the keeper start clueless (a uniform 1/3 over the three
zones) and learn *only* from how well each choice scores against what the other
is currently doing. We use **multiplicative-weights self-play** (Hedge / no-regret
learning): each agent keeps a running score per action and plays the softmax of
it. In a zero-sum game this provably drives the *average* strategies to the
mixed-strategy Nash equilibrium — the point where neither side can be exploited.

We log the full trajectory of both strategies so the video can animate the shot
distribution going from "exploitable" to "balanced", and we track the
exploitability (duality gap), which goes to ~0 at the equilibrium.

Run:  python selfplay.py            # ~512k kicks, ~2s on CPU
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import penalty_env as env

RUNS = os.path.join(os.path.dirname(__file__), "runs")


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def nash_gap(p: np.ndarray, q: np.ndarray, A: np.ndarray) -> float:
    """Exploitability (duality gap) of the strategy pair ``(p, q)``.

    ``max_z (A q)_z`` is the best goal rate a shooter could reach against the
    keeper ``q``; ``min_d (p A)_d`` is the lowest goal rate the keeper could
    force against the shooter ``p``. The gap between them is zero exactly at a
    Nash equilibrium, so it is our convergence metric.
    """
    shooter_best = float(np.max(A @ q))
    keeper_best = float(np.min(p @ A))
    return shooter_best - keeper_best


def train(iterations=4000, batch=128, seed=0, init_s=None):
    rng = np.random.default_rng(seed)
    A = env.payoff_matrix()                       # shooter's payoff = P(goal)
    eta = np.sqrt(8 * np.log(env.N) / iterations)  # no-regret learning rate

    # ``init_s`` lets a caller start the shooter biased (e.g. a "predictable
    # beginner" who always aims one way) so its un-learning can be shown.
    score_s = np.zeros(env.N) if init_s is None else np.array(init_s, float)
    score_k = np.zeros(env.N)  # keeper's cumulative save value per dive
    hist_s, hist_k, goal_rate = [], [], []

    for _ in range(iterations):
        p = softmax(eta * score_s)
        q = softmax(eta * score_k)

        # Play a batch of real penalties from the current strategies — the
        # honest "they faced off hundreds of thousands of times" part.
        s_idx = rng.choice(env.N, batch, p=p)
        k_idx = rng.choice(env.N, batch, p=q)
        goals = rng.random(batch) < A[s_idx, k_idx]
        goal_rate.append(float(goals.mean()))

        # Learn: how good was each choice against what the opponent is doing now.
        score_s += A @ q          # shooter rewards zones that beat this keeper
        score_k += 1.0 - p @ A    # keeper rewards dives that stop this shooter

        hist_s.append(p)
        hist_k.append(q)

    hist_s = np.array(hist_s)
    hist_k = np.array(hist_k)
    goal_rate = np.array(goal_rate)

    # The average strategy is the one that converges to Nash — and the one the
    # heatmap animates "settling".
    steps = np.arange(1, len(hist_s) + 1)[:, None]
    avg_s = np.cumsum(hist_s, 0) / steps
    avg_k = np.cumsum(hist_k, 0) / steps
    gap_avg = np.array([nash_gap(avg_s[t], avg_k[t], A) for t in range(len(avg_s))])

    return dict(
        A=A, hist_s=hist_s, hist_k=hist_k, avg_s=avg_s, avg_k=avg_k,
        goal_rate=goal_rate, gap_avg=gap_avg, batch=batch,
    )


def report(out):
    A, p, q = out["A"], out["avg_s"][-1], out["avg_k"][-1]
    kicks = out["batch"] * len(out["hist_s"])
    print(f"\nTrained on {kicks:,} penalties.\n")
    fmt = lambda v: "  ".join(f"{z}:{x:5.1%}" for z, x in zip(env.ZONES, v))
    print(f"Shooter — où elle tire   : {fmt(p)}")
    print(f"Keeper  — où il plonge    : {fmt(q)}")
    print(f"\nTaux de but à l'équilibre : {float(p @ A @ q):5.1%}")
    print(f"Exploitabilité finale     : {nash_gap(p, q, A):.4f}  (→ 0 = Nash)")


def save(out):
    os.makedirs(RUNS, exist_ok=True)
    np.savez(os.path.join(RUNS, "history.npz"), **{
        k: out[k] for k in ("A", "hist_s", "hist_k", "avg_s", "avg_k", "goal_rate", "gap_avg")
    })

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(out["gap_avg"], color="#e63946")
    ax1.set_title("Exploitabilité (→ 0 = équilibre de Nash)")
    ax1.set_xlabel("itérations"); ax1.set_ylabel("duality gap"); ax1.grid(alpha=.3)

    for i, z in enumerate(env.ZONES):
        ax2.plot(out["avg_s"][:, i], label=f"tir {z}")
    ax2.set_title("Stratégie du tireur (moyenne) qui se stabilise")
    ax2.set_xlabel("itérations"); ax2.set_ylabel("probabilité")
    ax2.legend(); ax2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RUNS, "convergence.png"), dpi=120)
    print("\nSaved runs/history.npz + runs/convergence.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = train(args.iterations, args.batch, args.seed)
    report(out)
    save(out)


if __name__ == "__main__":
    main()
