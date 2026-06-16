"""HQ replay of penalty duels, rendered with the trained strategies.

Headless (SDL dummy driver) so it runs without a window: it draws each kick to
a 1080x1920 surface, dumps frames, and muxes them with ffmpeg into a vertical
clip ready for the edit. Uses the final averaged strategies from
``runs/history.npz`` and animates the real sampled outcome (goal / save / miss).

Run:  python replay.py --duels 8
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless
import numpy as np
import pygame

import penalty_env as env

RUNS = os.path.join(os.path.dirname(__file__), "runs")
W, H, FPS = 1080, 1920, 30

# Palette
LINE = (236, 240, 235)
NET = (120, 162, 132)
KIT = (244, 163, 0)
GLOVE = (245, 245, 245)
SKIN = (228, 184, 146)
GOLD = (255, 214, 10)
RED = (230, 57, 70)
GREY = (188, 192, 190)

# Goal box + zones (pixels)
GL, GR = int(0.14 * W), int(0.86 * W)
GT, GB = int(0.16 * H), int(0.36 * H)
ZX = {0: int(0.28 * W), 1: int(0.50 * W), 2: int(0.72 * W)}
TY = int(GB - 0.28 * (GB - GT))          # low-corner target height
SPOT = (int(0.50 * W), int(0.72 * H))
REST_POS = (int(0.50 * W), GB - 28)      # keeper's hips at rest (goal centre)
REST_REACH = (int(0.50 * W), GB - 112)   # gloves at rest (hands up, ready)
ARC = 150

# Phase lengths (frames)
ANTIC, FLIGHT, HOLD = 10, 20, 18


def smooth(t):
    return t * t * (3 - 2 * t)


def ease_out(t):
    return 1 - (1 - t) ** 2


def lerp2(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def ball_arc(start, end, t):
    x = start[0] + (end[0] - start[0]) * t
    y = start[1] + (end[1] - start[1]) * t - ARC * math.sin(math.pi * t)
    return x, y


def draw_pitch(s):
    for i in range(14):                       # mown stripes
        y0, y1 = int(i / 14 * H), int((i + 1) / 14 * H)
        s.fill((26, 122, 62) if i % 2 == 0 else (22, 108, 55), (0, y0, W, y1 - y0))
    pygame.draw.line(s, LINE, (int(0.10 * W), int(0.58 * H)), (int(0.90 * W), int(0.58 * H)), 4)
    pygame.draw.arc(s, LINE, (int(0.34 * W), int(0.50 * H), int(0.32 * W), int(0.20 * H)), 0, math.pi, 4)
    pygame.draw.circle(s, LINE, SPOT, 7)


def draw_goal(s):
    for x in range(GL, GR + 1, 34):           # net
        pygame.draw.line(s, NET, (x, GT), (x, GB), 1)
    for y in range(GT, GB + 1, 28):
        pygame.draw.line(s, NET, (GL, y), (GR, y), 1)
    pygame.draw.rect(s, LINE, (GL, GT, GR - GL, GB - GT), 9)


def draw_keeper(s, pos, lean, reach):
    """A diving keeper of *fixed* size: hips at ``pos``, a constant-length torso
    tilted by ``lean`` (-1 left … +1 right), head on top, and one arm reaching to
    ``reach`` (the gloves). The body never stretches — the keeper slides toward
    the ball instead, so the arm stays short.
    """
    hx, hy = pos
    tx = lean * 42
    sh = (hx + tx, hy - 92)                                   # shoulder (fixed torso height)
    pygame.draw.line(s, KIT, pos, sh, 46)                     # torso
    pygame.draw.circle(s, KIT, (int(hx), int(hy)), 23)
    pygame.draw.circle(s, KIT, (int(sh[0]), int(sh[1])), 23)
    head = (sh[0] + tx * 0.25, sh[1] - 34)
    pygame.draw.circle(s, SKIN, (int(head[0]), int(head[1])), 22)
    pygame.draw.line(s, KIT, sh, reach, 24)                   # arm
    pygame.draw.circle(s, GLOVE, (int(reach[0]), int(reach[1])), 17)
    pygame.draw.circle(s, (40, 40, 40), (int(reach[0]), int(reach[1])), 17, 2)


def draw_ball(s, pos, r, spin):
    x, y = pos
    pygame.draw.circle(s, (250, 250, 250), (int(x), int(y)), r)
    pygame.draw.circle(s, (38, 38, 38), (int(x), int(y)), r, 2)
    pts = [(x + 0.46 * r * math.cos(spin + k * 2 * math.pi / 5),
            y + 0.46 * r * math.sin(spin + k * 2 * math.pi / 5)) for k in range(5)]
    pygame.draw.polygon(s, (32, 32, 32), pts)


def text(s, font, msg, center, color):
    surf = font.render(msg, True, color)
    s.blit(surf, surf.get_rect(center=center))


def ball_end(shot, dive, outcome):
    """Where the ball finishes for this outcome (wide on a miss, on the gloves on
    a save, in the net on a goal)."""
    if outcome == "miss":
        return (GL - 95, TY) if shot == 0 else (GR + 95, TY) if shot == 2 else (ZX[1], GT - 95)
    if outcome == "save":
        return (ZX[shot], TY)                      # ball meets the gloves
    return (ZX[shot], TY + (60 if dive == shot else 0))


def scene(s, shot, dive, outcome, phase, t, trail):
    """Draw one frame of a duel (pitch, goal, keeper, ball) — no text overlay.

    Reused by ``replay.py`` and ``story.py``. ``trail`` is a mutable list that
    accumulates ball positions across frames for the motion trail.
    """
    end = ball_end(shot, dive, outcome)
    zone = (ZX[dive], TY)                                       # where the keeper commits
    lean = max(-1.0, min(1.0, (ZX[dive] - 0.5 * W) / (0.36 * W)))
    dive_pos = (REST_POS[0] + 0.55 * (ZX[dive] - REST_POS[0]), REST_POS[1] + 6)
    draw_pitch(s)
    draw_goal(s)
    if phase == "antic":
        pos = (REST_POS[0], REST_POS[1] + 4 * math.sin(t * 6))
        ln, reach, ball, spin = 0.0, REST_REACH, SPOT, 0.0
    elif phase == "flight":
        kf = smooth(max(0.0, (t - 0.12) / 0.88))
        pos = lerp2(REST_POS, dive_pos, kf)
        ln = lean * kf
        reach = lerp2(REST_REACH, zone, kf)
        bt = ease_out(t)
        ball, spin = ball_arc(SPOT, end, bt), bt * 26
        trail.append(ball)
    else:
        pos, ln, reach, ball, spin = dive_pos, lean, zone, end, 26
    tr = pygame.Surface((W, H), pygame.SRCALPHA)
    for k, p in enumerate(trail[-7:]):
        pygame.draw.circle(tr, (255, 255, 255, int(26 * (k + 1) / 7)), (int(p[0]), int(p[1])), 20)
    s.blit(tr, (0, 0))
    draw_keeper(s, pos, ln, reach)
    draw_ball(s, ball, 24, spin)
    return ball


def duel_frames(s, fonts, shot, dive, outcome, idx, total, tally):
    f_big, f_cap, f_tally = fonts
    label, color = {"goal": ("BUT !", GOLD), "save": ("ARRÊT !", RED), "miss": ("RATÉ", GREY)}[outcome]
    trail = []

    def plural(n, word):
        return f"{n} {word}{'s' if n > 1 else ''}"

    def overlay():
        text(s, f_cap, f"PENALTY {idx}/{total}", (W // 2, int(0.105 * H)), LINE)
        tline = "   ·   ".join([plural(tally["goal"], "but"),
                                 plural(tally["save"], "arrêt"),
                                 plural(tally["miss"], "raté")])
        text(s, f_tally, tline, (W // 2, int(0.135 * H)), (210, 230, 215))

    out = []
    for ph, length in (("antic", ANTIC), ("flight", FLIGHT), ("hold", HOLD)):
        for f in range(length):
            scene(s, shot, dive, outcome, ph, f / (length - 1), trail)
            overlay()
            if ph == "hold":
                if outcome == "goal" and f < 4:    # impact flash
                    fl = pygame.Surface((W, H), pygame.SRCALPHA)
                    fl.fill((255, 255, 255, 150 - f * 38))
                    s.blit(fl, (0, 0))
                text(s, f_big, label, (W // 2, int(0.46 * H)), color)
            out.append(pygame.image.tostring(s, "RGB"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duels", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = np.load(os.path.join(RUNS, "history.npz"))
    p, q = data["avg_s"][-1], data["avg_k"][-1]
    rng = np.random.default_rng(args.seed)

    pygame.init()
    surf = pygame.Surface((W, H))
    fonts = (pygame.font.SysFont("Arial", 150, bold=True),
             pygame.font.SysFont("Arial", 46, bold=True),
             pygame.font.SysFont("Arial", 40, bold=True))

    tally = {"goal": 0, "save": 0, "miss": 0}
    with tempfile.TemporaryDirectory() as tmp:
        n = 0
        for i in range(args.duels):
            shot = int(rng.choice(env.N, p=p))
            dive = int(rng.choice(env.N, p=q))
            outcome = env.play_detailed(shot, dive, rng)
            tally[outcome] += 1
            print(f"  penalty {i+1}: tir {env.ZONES[shot]:>6} / plonge {env.ZONES[dive]:>6} -> {outcome}")
            for raw in duel_frames(surf, fonts, shot, dive, outcome, i + 1, args.duels, tally):
                pygame.image.save(pygame.image.fromstring(raw, (W, H), "RGB"),
                                  os.path.join(tmp, f"{n:05d}.png"))
                n += 1
        out = os.path.join(RUNS, "replay.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
            "-i", os.path.join(tmp, "%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", out,
        ], check=True)
        print(f"\nSaved {out}  ({n} frames, {args.duels} duels)  —  "
              f"{tally['goal']} buts / {tally['save']} arrêts / {tally['miss']} ratés")


if __name__ == "__main__":
    main()
