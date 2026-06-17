"""Voix off Penalty-AI — Épisode 3 (IA × jeu, spécial CdM 2026).

Script calé beat par beat sur story.py (Actes 1-4) et sur les VRAIS chiffres
imprimés par le rendu : 48% → 60% → 70% (Nash), puis Acte 4 (gardien réel,
Bar-Eli 2007) : Nash 71% / IA 88% (+18 buts) / plancher garanti 70% → 17%.

NB voix : pas de « : » dans les phrases (XTTS le lit « deux-points ») — virgules
ou points uniquement.

Génération (env TTS+torch) :
    COQUI_TOS_AGREED=1 ~/Projets/AI-Dream/.venv/bin/python \
        ~/Projets/tiktok-ia-playbook/tts.py \
        --config voiceover.py --out runs/voice
"""

SEGMENTS = [
    ("question",
     "Comment marquer son penalty à tous les coups ?"),

    ("hook",
     "Deux IA, aucune règle. "
     "Elles vont chercher la réponse."),

    ("act1",
     "Au début, le tireur vise toujours le même coin. "
     "Prévisible, arrêté une fois sur deux."),

    ("act2",
     "Il apprend à varier, le gardien hésite, "
     "et il marque six fois sur dix, désormais."),

    ("act3",
     "Puis il devient imprévisible. "
     "Soixante-dix pour cent garantis. "
     "C'est l'équilibre de Nash, la théorie des jeux."),

    ("act4_intro",
     "Mais un vrai gardien ne reste presque jamais au centre, "
     "à peine six pour cent du temps."),

    ("act4_nash",
     "Nash s'en fiche, il garde ses soixante et onze pour cent. "
     "Imbattable, mais il ne profite pas du défaut."),

    ("act4_ai",
     "L'IA, elle, voit le trou et tire au centre. "
     "Quatre-vingt-huit pour cent, dix-huit buts de plus."),

    ("act4_twist",
     "Mais elle a baissé sa garde. Si le gardien reste au centre, "
     "son garanti chute à dix-sept."),

    ("outro",
     "Nash te rend imbattable. "
     "L'IA te fait gagner plus, mais à tes risques et périls."),
]
