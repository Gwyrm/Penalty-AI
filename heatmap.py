"""Animated shot heatmap — the visual centrepiece of the video.

Reads ``runs/history.npz`` (written by ``selfplay.py``) and renders the goal
mouth split into three zones, shaded by how often the shooter aims there. The
animation shows the strategy going from lopsided/exploitable to balanced, with
the exploitability counting down to ~0. Output: ``runs/heatmap.mp4``.

Run:  python heatmap.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

import penalty_env as env

RUNS = os.path.join(os.path.dirname(__file__), "runs")
N_FRAMES = 150


def main():
    data = np.load(os.path.join(RUNS, "history.npz"))
    avg_s, gap = data["avg_s"], data["gap_avg"]

    # Subsample to a smooth ~5s clip at 30 fps.
    idx = np.linspace(0, len(avg_s) - 1, N_FRAMES).astype(int)
    dist, gap = avg_s[idx], gap[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 3); ax.set_ylim(-0.6, 2.2); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 3, 1.6, fill=False, lw=4, ec="white"))
    ax.plot([1, 1], [0, 1.6], color="white", lw=1, alpha=.4)
    ax.plot([2, 2], [0, 1.6], color="white", lw=1, alpha=.4)
    fig.patch.set_facecolor("#0d1b2a"); ax.set_facecolor("#0d1b2a")

    cells = [Rectangle((i, 0), 1, 1.6, color="#e63946", alpha=0.0) for i in range(env.N)]
    labels = [ax.text(i + .5, .8, "", ha="center", va="center",
                      color="white", fontsize=20, fontweight="bold") for i in range(env.N)]
    for c in cells:
        ax.add_patch(c)
    for i, z in enumerate(env.ZONES):
        ax.text(i + .5, -0.35, z, ha="center", color="white", fontsize=13)
    title = ax.text(1.5, 2.0, "", ha="center", color="white", fontsize=15, fontweight="bold")
    sub = ax.text(1.5, 1.78, "", ha="center", color="#a8dadc", fontsize=12)

    def update(f):
        p = dist[f]
        for i, (c, lab) in enumerate(zip(cells, labels)):
            c.set_alpha(float(0.12 + 0.88 * p[i] / max(p.max(), 1e-6)))
            lab.set_text(f"{p[i]:.0%}")
        title.set_text("Où l'IA tire ses penaltys")
        sub.set_text(f"exploitabilité : {gap[f]:.3f}  →  0 = imbattable")
        return cells + labels + [title, sub]

    anim = animation.FuncAnimation(fig, update, frames=N_FRAMES, blit=False)
    out = os.path.join(RUNS, "heatmap.mp4")
    anim.save(out, writer=animation.FFMpegWriter(fps=30, bitrate=2400))
    fig.savefig(os.path.join(RUNS, "heatmap_final.png"), dpi=120,
                facecolor=fig.get_facecolor())
    print(f"Saved {out} + runs/heatmap_final.png")


if __name__ == "__main__":
    main()
