# ⚽ Penalty-AI — deux IA réinventent la théorie des jeux au tir au but

Un tireur et un gardien, chacun piloté par une IA. **Aucune règle, aucune
stratégie pré-codée** : ils ne reçoivent qu'une récompense (but / arrêt) et
s'affrontent des centaines de milliers de fois en *self-play*. À partir de ce
seul signal, ils convergent seuls vers la **stratégie optimale** du penalty :
une **stratégie mixte** (équilibre de Nash) où chacun devient imprévisible et
ne peut plus être exploité.

> Épisode 3 de la série *IA × jeu vidéo* — spécial Coupe du monde 2026.

## L'idée en une image

Au début, le tireur a un biais → le gardien l'exploite → le tireur s'adapte →
le gardien s'adapte… La course-poursuite se stabilise sur un point unique :
l'équilibre où l'**exploitabilité tombe à zéro**. C'est exactement la solution
que la théorie des jeux prédit pour un penalty — redécouverte par essai-erreur.

## Le jeu

Le tireur vise une zone (Gauche / Centre / Droite), le gardien plonge dans une
direction, **simultanément**. Le modèle de but (`penalty_env.py`) est volontairement
réaliste : viser un coin rapporte plus mais se rate plus souvent ; un ballon
central est facile à arrêter si le gardien reste. Du coup l'équilibre n'est pas
un bête 50/50 mais une distribution pondérée — plus intéressante et plus honnête.

## Lancer

```bash
pip install -r requirements.txt

# 1. (optionnel) voix off XTTS-v2, un .wav par segment dans runs/voice/
#    via le pipeline tts.py du playbook (env avec TTS + torch) :
COQUI_TOS_AGREED=1 python /chemin/tiktok-ia-playbook/tts.py \
    --config voiceover.py --out runs/voice

python story.py           # 🎬 la vidéo pédagogique 4 actes (runs/story.mp4, ~58s 9:16)
                          #    se cale automatiquement sur runs/voice/*.wav s'ils existent,
                          #    sinon rendu muet au pacing naturel

python selfplay.py        # entraîne en self-play (~512k penaltys, qq secondes CPU)
python heatmap.py         # heatmap animée des tirs (runs/heatmap.mp4)
python replay.py --duels 8  # replay HQ vertical 9:16 des duels (runs/replay.mp4)
```

`selfplay.py` écrit `runs/history.npz` (trajectoire complète des deux stratégies
+ exploitabilité) et `runs/convergence.png`.

## Stack

- **Self-play par poids multiplicatifs** (Hedge / no-regret) — agents tabulaires
  sur 3 zones ; la stratégie *moyenne* converge prouvablement vers le Nash.
- **Métrique de convergence** : exploitabilité (duality gap) → 0 = Nash.
- **Viz** : matplotlib (heatmap) + pygame headless → ffmpeg (replay 1080×1920).

## Acte 4 — théorie des jeux vs IA, face à un vrai gardien

L'équilibre de Nash est *imbattable* mais pas *maximalement exploitant* : contre
un gardien parfait il garantit ~71 % quoi qu'il arrive, et n'en tire jamais plus.
Or les vrais gardiens ne sont pas parfaits. D'après **Bar-Eli et al. (2007)** (286
penaltys d'élite), ils ne **restent au centre que 6 % du temps** — alors que c'est
statistiquement leur meilleure option. Le « trou » est donc… le centre.

Une IA *adaptative* (best-response) repère ce biais et tire au centre : elle monte
à **~88 %** de buts (+18 sur Nash). Mais en abandonnant l'équilibre, elle **baisse
sa garde** : si le gardien comprend et reste au centre, son taux garanti s'effondre
de 70 % à ~17 %. C'est la dichotomie GTO vs exploitatif du poker, version penalty :

> **Nash te rend imbattable. L'IA te fait gagner plus — mais à tes risques.**

## Pourquoi c'est plus qu'un gadget

Le penalty est l'exemple-école d'un jeu à somme nulle à coups simultanés : la
solution optimale n'est pas une action fixe mais le fait de **randomiser**. Voir
deux agents découvrir ça tout seuls, sans qu'on leur explique, c'est de la
théorie des jeux qui émerge sous les yeux.

## Ce que c'est — et ce que ce n'est pas

C'est un **modèle de théorie des jeux**, pas une simulation physique ni un vrai
jeu de foot : 3 zones, une matrice de gains, des **agents tabulaires** (pas de
réseaux de neurones). Le rendu est du dessin pygame fait main. Ce qui est
rigoureux, c'est l'**apprentissage** (self-play sans regret → Nash) et le fait
que l'équilibre trouvé (~39/22/39, ~71 % de buts) recoupe les études économiques
réelles sur les penaltys (Palacios-Huerta, 2003). À comprendre comme « **la
théorie des jeux du penalty** », pas comme « une IA qui apprend à jouer au foot ».

---

*Faceless / open-source. Code : [github.com/Gwyrm/Penalty-AI](https://github.com/Gwyrm/Penalty-AI)*
