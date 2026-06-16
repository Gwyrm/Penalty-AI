"""Penalty duel — the game model + a tiny simultaneous-move environment.

Two agents face off: a *shooter* picks a target zone, a *keeper* picks a dive
direction, both at the same time. Nothing about strategy is hard-coded into the
agents — they only ever see the outcome of a kick (goal / no goal). Self-play
on this single reward is enough to make them converge to the mixed-strategy
Nash equilibrium of the game.

This module is the single source of truth for the payoff model, shared by the
trainer (``selfplay.py``) and the renderers (``heatmap.py``, ``replay.py``).
"""
from __future__ import annotations

import numpy as np

# The three zones a shooter can aim at / a keeper can dive to.
ZONES = ["Gauche", "Centre", "Droite"]
N = len(ZONES)

# Probability the shot misses the goal entirely (keeper-independent).
# Corners are harder to hit cleanly than the centre.
MISS_PROB = np.array([0.12, 0.04, 0.12])

# Probability the keeper SAVES when he dives the same way as the shot.
# A central ball is easy to block; reaching a corner in time is hard.
SAVE_IF_MATCHED = np.array([0.55, 0.90, 0.55])


def goal_prob(shot: int, dive: int) -> float:
    """Expected goal probability for a shot at ``shot`` vs a dive to ``dive``."""
    on_target = 1.0 - MISS_PROB[shot]
    if shot == dive:
        return float(on_target * (1.0 - SAVE_IF_MATCHED[shot]))
    return float(on_target)


def payoff_matrix() -> np.ndarray:
    """``A[shot, dive]`` = expected goal probability (the shooter's payoff).

    The game is zero-sum: the keeper's payoff is ``1 - A`` (a save/miss).
    """
    return np.array([[goal_prob(s, d) for d in range(N)] for s in range(N)])


def play(shot: int, dive: int, rng: np.random.Generator) -> int:
    """Sample a single penalty. Returns 1 for a goal, 0 for a save or miss."""
    return int(rng.random() < goal_prob(shot, dive))


def play_detailed(shot: int, dive: int, rng: np.random.Generator) -> str:
    """Sample a penalty and say *how* it ended: "miss", "save" or "goal".

    Consistent with ``goal_prob``: a shot is missed off target with
    ``MISS_PROB``; otherwise the keeper saves only if he dived the right way
    (``SAVE_IF_MATCHED``); anything else is a goal. Used by the renderer so the
    animation matches the outcome.
    """
    if rng.random() < MISS_PROB[shot]:
        return "miss"
    if shot == dive and rng.random() < SAVE_IF_MATCHED[shot]:
        return "save"
    return "goal"


if __name__ == "__main__":
    A = payoff_matrix()
    print("Payoff matrix A[shot, dive] = P(goal):")
    header = "          " + "  ".join(f"{z:>7}" for z in ZONES)
    print(header)
    for s in range(N):
        row = "  ".join(f"{A[s, d]:7.2f}" for d in range(N))
        print(f"shot {ZONES[s]:>7}  {row}")
