"""Build the pedagogical 3-act video: how two AIs learn the optimal penalty.

The story is driven by *real* self-play data. A shooter starts predictable
(always the same corner); against the best possible keeper it only scores ~48%.
As it learns to vary, the keeper can no longer guess, and its *guaranteed* goal
rate climbs to the Nash value (~71%). Each act pairs concrete duels with the
strategy heatmap and a one-line lesson — so the clip teaches, not just dazzles.

Output: runs/story.mp4   (vertical 1080x1920, ~35s)

Run:  python story.py
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import wave

import numpy as np
import pygame

import penalty_env as env
import selfplay as sp
import replay as R   # reuses the duel renderer, palette and geometry

W, H, FPS = R.W, R.H, R.FPS
ORANGE, GREEN = (244, 140, 0), (46, 196, 120)
A = env.payoff_matrix()


def soft_br(p, sharp=8.0):
    """The exploiting keeper: dives where it concedes the *least* (soft argmin)."""
    q = np.exp(-sharp * (p @ A))
    return q / q.sum()


def secured(p):
    """Goal rate the shooter can guarantee against the best keeper = min_d (pA)_d."""
    return float(np.min(p @ A))


# ---------------------------------------------------------------- beats --------
def card(sink, fonts, lines, sub, hold=None, bg=None):
    f_title, f_sub = fonts["title"], fonts["sub"]
    for _ in range(hold if hold is not None else sink.hold_card):
        if bg is not None:
            bg.step(sink.s)        # penaltys qui tournent en fond → jamais de plan figé
        else:
            R.draw_pitch(sink.s)
        panel = pygame.Surface((W, int(0.46 * H)), pygame.SRCALPHA)
        panel.fill((9, 20, 14, 165))
        sink.s.blit(panel, (0, int(0.30 * H)))
        y = int(0.40 * H)
        for ln in lines:
            R.text(sink.s, f_title, ln, (W // 2, y), R.LINE)
            y += int(0.072 * H)
        if sub:
            R.text(sink.s, f_sub, sub, (W // 2, y + 12), (193, 222, 203))
        sink.add()


def duels(sink, fonts, p, q, caption, n, seed):
    rng = np.random.default_rng(seed)
    f_cap, f_huge = fonts["cap"], fonts["huge"]
    res = {"goal": ("BUT !", R.GOLD), "save": ("ARRÊT !", R.RED), "miss": ("RATÉ", R.GREY)}
    for _ in range(n):
        shot = int(rng.choice(env.N, p=p))
        dive = int(rng.choice(env.N, p=q))
        outcome = env.play_detailed(shot, dive, rng)
        label, color = res[outcome]
        trail = []
        for ph, length in (("antic", R.ANTIC), ("flight", R.FLIGHT), ("hold", R.HOLD)):
            for f in range(length):
                R.scene(sink.s, shot, dive, outcome, ph, f / (length - 1), trail)
                R.text(sink.s, f_cap, caption, (W // 2, int(0.105 * H)), R.LINE)
                if ph == "hold":
                    if outcome == "goal" and f < 4:
                        fl = pygame.Surface((W, H), pygame.SRCALPHA)
                        fl.fill((255, 255, 255, 150 - f * 38))
                        sink.s.blit(fl, (0, 0))
                    R.text(sink.s, f_huge, label, (W // 2, int(0.46 * H)), color)
                sink.add()


def heat(sink, fonts, dist, title, stat, lesson, color, hold=None):
    s = sink.s
    zw = (R.GR - R.GL) / 3
    mx = max(dist.max(), 1e-6)
    for _ in range(hold if hold is not None else sink.hold_heat):
        R.draw_pitch(s)
        for i in range(3):
            x0 = R.GL + i * zw
            cell = pygame.Surface((int(zw), R.GB - R.GT), pygame.SRCALPHA)
            cell.fill((*color, int(40 + 205 * dist[i] / mx)))
            s.blit(cell, (int(x0), R.GT))
        R.draw_goal(s)
        for i in range(3):
            x0 = R.GL + i * zw
            R.text(s, fonts["big"], f"{dist[i]:.0%}", (int(x0 + zw / 2), (R.GT + R.GB) // 2), (255, 255, 255))
            R.text(s, fonts["small"], env.ZONES[i], (int(x0 + zw / 2), R.GB + 38), (225, 235, 228))
        R.text(s, fonts["cap"], title, (W // 2, int(0.105 * H)), R.LINE)
        band = pygame.Surface((W, int(0.18 * H)), pygame.SRCALPHA)
        band.fill((9, 20, 14, 185))
        s.blit(band, (0, int(0.46 * H)))
        R.text(s, fonts["stat"], stat, (W // 2, int(0.52 * H)), color)
        R.text(s, fonts["lesson"], lesson, (W // 2, int(0.60 * H)), (240, 244, 240))
        sink.add()


# ---------------------------------------------------------------- driver -------
class Sink:
    def __init__(self, tmp, s):
        self.tmp, self.s, self.n = tmp, s, 0
        self.hold_card, self.hold_heat = 52, 72

    def add(self):
        pygame.image.save(self.s, os.path.join(self.tmp, f"{self.n:05d}.png"))
        self.n += 1


class BgSim:
    """Un penalty qui ne s'arrête jamais — tourne en fond derrière les cartes
    pour qu'il n'y ait aucun plan statique (rétention)."""

    def __init__(self, p, q, seed=99):
        self.rng = np.random.default_rng(seed)
        self.p, self.q = p, q
        self.seq = [("antic", R.ANTIC), ("flight", R.FLIGHT), ("hold", R.HOLD)]
        self._new()

    def _new(self):
        self.shot = int(self.rng.choice(env.N, p=self.p))
        self.dive = int(self.rng.choice(env.N, p=self.q))
        self.outcome = env.play_detailed(self.shot, self.dive, self.rng)
        self.trail = []
        self.pi, self.fi = 0, 0

    def step(self, s):
        ph, length = self.seq[self.pi]
        R.scene(s, self.shot, self.dive, self.outcome, ph, self.fi / max(length - 1, 1), self.trail)
        self.fi += 1
        if self.fi >= length:
            self.fi, self.pi = 0, self.pi + 1
            if self.pi >= len(self.seq):
                self._new()


def main():
    # Real self-play with a predictable (always-left) shooter that un-learns it.
    out = sp.train(iterations=1500, batch=128, seed=0, init_s=[30.0, 0.0, 0.0])
    avg = out["avg_s"]
    sec = np.array([secured(avg[t]) for t in range(len(avg))])
    # Three checkpoints: predictable start, mid transition, converged Nash.
    p1 = avg[3]
    lo, hi = sec[3], sec[-1]
    i2 = int(np.argmax(sec >= (lo + hi) / 2))
    p2, p3 = avg[max(i2, 8)], avg[-1]
    acts = [(p1, R.RED, ORANGE), (p2, ORANGE, ORANGE), (p3, GREEN, GREEN)]
    for k, (p, _, _) in enumerate(acts, 1):
        print(f"  Acte {k}: tir {np.round(p, 2)}  but garanti {secured(p):.0%}")

    pygame.init()
    surf = pygame.Surface((W, H))
    fonts = {
        "title": pygame.font.SysFont("Arial", 72, bold=True),
        "sub": pygame.font.SysFont("Arial", 46),
        "cap": pygame.font.SysFont("Arial", 44, bold=True),
        "huge": pygame.font.SysFont("Arial", 150, bold=True),
        "big": pygame.font.SysFont("Arial", 64, bold=True),
        "small": pygame.font.SysFont("Arial", 38),
        "stat": pygame.font.SysFont("Arial", 80, bold=True),
        "lesson": pygame.font.SysFont("Arial", 50, bold=True),
    }

    # --- ACTE 4 : théorie des jeux (Nash) vs IA adaptative face à un vrai gardien ---
    # Un vrai gardien n'est pas parfait : il a un tic (ici, il plonge trop à droite).
    p_nash = acts[2][0]
    # Vraie distribution de plongeon des gardiens d'élite (Bar-Eli et al., 2007,
    # 286 penaltys) : ils ne restent au centre que 6% du temps — alors que c'est
    # leur meilleure option. Le "trou" exploitable est donc... le centre.
    q_bias = np.array([0.493, 0.063, 0.444]); q_bias = q_bias / q_bias.sum()
    # L'IA exploitante : elle vise là où le gardien n'est presque jamais (best-response).
    p_expl = sp.softmax(12.0 * (A @ q_bias))
    nash_vs_bias = float(p_nash @ A @ q_bias)   # Nash : ~invariant, quel que soit le gardien
    expl_vs_bias = float(p_expl @ A @ q_bias)    # exploite le tic → marque plus
    nash_floor = secured(p_nash)                 # plancher garanti de Nash (~71%)
    expl_floor = secured(p_expl)                 # plancher de l'IA si le gardien s'adapte
    gain = expl_vs_bias - nash_vs_bias
    print(f"\n  Acte 4 — gardien biaisé {np.round(q_bias, 2)}")
    print(f"    Nash   : {nash_vs_bias:.0%} de buts (plancher garanti {nash_floor:.0%})")
    print(f"    IA     : {expl_vs_bias:.0%} de buts (+{gain*100:.0f} pts)  "
          f"mais plancher chute à {expl_floor:.0%}")

    with tempfile.TemporaryDirectory() as tmp:
        sink = Sink(tmp, surf)
        bg = BgSim(p_nash, soft_br(p_nash))   # penaltys en fond derrière les cartes

        # Each group's on-screen duration is driven by its narration wav, so the
        # voice and visuals stay in sync. Slack (voice longer than the action) is
        # absorbed by the last shot of the group — the explanatory card/heatmap
        # held while the voice-over talks. `head` plays at natural speed; `tail`
        # is stretched to make the group span its segment.
        groups = [
            ("hook", None,
             lambda h: card(sink, fonts, ["DEUX IA.", "AUCUNE RÈGLE.", "JUSTE DES PENALTYS."],
                            "But ou arrêt — elles apprennent seules.", hold=h, bg=bg)),
            ("act1",
             lambda: (card(sink, fonts, ["ACTE 1", "Le tireur débutant"], "Il tire toujours du même côté.", bg=bg),
                      duels(sink, fonts, acts[0][0], soft_br(acts[0][0]),
                            "Toujours à gauche → le gardien anticipe", n=2, seed=11)),
             lambda h: heat(sink, fonts, acts[0][0], "Sa stratégie",
                            f"BUT GARANTI : {secured(acts[0][0]):.0%}", "Prévisible = arrêté", R.RED, hold=h)),
            ("act2",
             lambda: (card(sink, fonts, ["ACTE 2", "Il apprend à varier"], "Se faire arrêter, ça pique.", bg=bg),
                      duels(sink, fonts, acts[1][0], soft_br(acts[1][0]),
                            "Il change de côté → le gardien hésite", n=2, seed=7)),
             lambda h: heat(sink, fonts, acts[1][0], "Sa stratégie",
                            f"BUT GARANTI : {secured(acts[1][0]):.0%}", "Moins lisible", ORANGE, hold=h)),
            ("act3",
             lambda: (card(sink, fonts, ["ACTE 3", "L'équilibre parfait"], "Impossible à exploiter.", bg=bg),
                      duels(sink, fonts, acts[2][0], soft_br(acts[2][0]),
                            "Imprévisible → le gardien ne peut plus deviner", n=2, seed=3)),
             lambda h: heat(sink, fonts, acts[2][0], "Sa stratégie",
                            f"BUT GARANTI : {secured(acts[2][0]):.0%}", "ÉQUILIBRE DE NASH = 75% réel", GREEN, hold=h)),
            ("act4_intro",
             lambda: card(sink, fonts, ["ACTE 4", "Et un VRAI gardien ?"],
                          "Données réelles : 286 penaltys.", bg=bg),
             lambda h: heat(sink, fonts, q_bias, "Où va un vrai gardien",
                            "AU CENTRE : 6%", "Il plonge presque toujours (Bar-Eli, 2007)", ORANGE, hold=h)),
            ("act4_nash",
             lambda: (card(sink, fonts, ["LA THÉORIE DES JEUX", "joue à la lettre"], "Imbattable… mais polie.", bg=bg),
                      duels(sink, fonts, p_nash, q_bias, "Nash joue pareil, quoi qu'il arrive", n=2, seed=21)),
             lambda h: heat(sink, fonts, p_nash, "Stratégie de Nash",
                            f"BUTS : {nash_vs_bias:.0%}", "Solide — mais laisse des buts", GREEN, hold=h)),
            ("act4_ai",
             lambda: (card(sink, fonts, ["L'IA, elle,", "S'ADAPTE"], "Elle a vu le trou.", bg=bg),
                      duels(sink, fonts, p_expl, q_bias, "Elle tire au centre, là où il n'est jamais", n=2, seed=5)),
             lambda h: heat(sink, fonts, p_expl, "Stratégie de l'IA",
                            f"BUTS : {expl_vs_bias:.0%}", f"+{gain*100:.0f} pts : tire au centre", ORANGE, hold=h)),
            ("act4_twist",
             lambda: card(sink, fonts, ["MAIS elle a", "BAISSÉ SA GARDE"], "Si le gardien reste au centre…", bg=bg),
             lambda h: heat(sink, fonts, p_expl, "Si le gardien s'adapte",
                            f"GARANTI : {expl_floor:.0%}",
                            f"De {nash_floor:.0%} à {expl_floor:.0%} : exploitable", R.RED, hold=h)),
            ("outro", None,
             lambda h: card(sink, fonts, ["Nash te rend IMBATTABLE.", "L'IA te fait GAGNER PLUS —", "mais à tes risques."],
                            "Théorie des jeux × IA — Gwyrm/Penalty-AI", hold=h, bg=bg)),
        ]

        voice_dir = os.path.join(R.RUNS, "voice")
        have_voice = all(os.path.isfile(os.path.join(voice_dir, f"{n}.wav")) for n, _, _ in groups)

        def wav_seconds(name):
            with wave.open(os.path.join(voice_dir, f"{name}.wav"), "rb") as w:
                return w.getnframes() / float(w.getframerate())

        for name, head, tail in groups:
            start = sink.n
            if head is not None:
                head()
            used = sink.n - start
            if have_voice:
                budget = int(round(wav_seconds(name) * FPS))
                tail(max(18, budget - used))   # stretch the last shot to fill the narration
            else:
                tail(None)                     # no audio: natural pacing
        if not have_voice:
            print("  (pas de runs/voice/*.wav — vidéo muette, pacing naturel)")

        dst = os.path.join(R.RUNS, "story.mp4")
        if have_voice:
            # Concatène les wav dans l'ordre des groupes → piste voix alignée.
            voice_all = os.path.join(R.RUNS, "voice_all.wav")
            srcs = [os.path.join(voice_dir, f"{n}.wav") for n, _, _ in groups]
            with wave.open(srcs[0], "rb") as r0:
                params = r0.getparams()
            with wave.open(voice_all, "wb") as w:
                w.setparams(params)
                for s in srcs:
                    with wave.open(s, "rb") as r:
                        w.writeframes(r.readframes(r.getnframes()))
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", os.path.join(tmp, "%05d.png"), "-i", voice_all,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                "-shortest", dst,
            ], check=True)
        else:
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                "-i", os.path.join(tmp, "%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", dst,
            ], check=True)
        print(f"\nSaved {dst}  ({sink.n} frames ≈ {sink.n / FPS:.0f}s)")


if __name__ == "__main__":
    main()
