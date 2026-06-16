"""Voix off Penalty-AI — Épisode 3 (IA × jeu, spécial CdM 2026).

Script calé beat par beat sur story.py (Actes 1-4) et sur les VRAIS chiffres
imprimés par le rendu : 48% → 60% → 70% (Nash), puis Acte 4 (gardien biaisé) :
Nash 71% / IA 82% (+11 pts) / plancher garanti 70% → 44%.

Génération (env TTS+torch) :
    COQUI_TOS_AGREED=1 ~/Projets/AI-Dream/.venv/bin/python \
        ~/Projets/tiktok-ia-playbook/tts.py \
        --config voiceover.py --out runs/voice
"""

SEGMENTS = [
    ("hook",
     "Deux IA, aucune règle. "
     "Elles vont redécouvrir la théorie des jeux."),

    ("act1",
     "Au début, le tireur vise toujours le même coin. "
     "Prévisible : arrêté une fois sur deux."),

    ("act2",
     "Il apprend à varier. Le gardien hésite. Soixante pour cent."),

    ("act3",
     "Puis il devient imprévisible. "
     "Soixante-dix pour cent garantis : l'équilibre de Nash."),

    ("act4_intro",
     "Mais un vrai gardien ne reste presque jamais au centre : "
     "six pour cent du temps."),

    ("act4_nash",
     "Nash s'en fiche : il garde ses soixante et onze pour cent. "
     "Imbattable, mais il ne profite pas du défaut."),

    ("act4_ai",
     "L'IA, elle, voit le trou et tire au centre. "
     "Quatre-vingt-huit pour cent : dix-huit buts de plus."),

    ("act4_twist",
     "Mais elle a baissé sa garde : si le gardien reste au centre, "
     "son garanti chute à dix-sept."),

    ("outro",
     "Nash te rend imbattable. "
     "L'IA te fait gagner plus, mais à tes risques."),
]
