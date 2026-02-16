"""
This module contains all prompts used for correcting transcriptions across different corpora.
Each prompt is optimized to minimize changes while fixing speaker labels and obvious transcription errors.

TWO-PASS SYSTEM:
- Pass 1: Speaker diarization (SPEAKER_XX -> actual role)
- Pass 2: Transcription error correction
"""

from typing import Final

# ============================================================================
# PASS 1: SPEAKER DIARIZATION PROMPTS
# ============================================================================

DIARIZATION_PROMPTS: Final[dict[str, str]] = {
    "neurochirurgie": """
Tu es un expert en analyse de conversations médicales en neurochirurgie.

CONTEXTE : Tu reçois une transcription où les locuteurs sont identifiés comme "Locuteur SPEAKER_00", "Locuteur SPEAKER_01", etc.
Ces étiquettes SPEAKER_XX ont été attribuées par reconnaissance vocale : un même SPEAKER_XX correspond à la même personne physique, sauf erreur d'analyse.

TA MISSION : Identifier le vrai rôle de chaque SPEAKER_XX unique parmi :
- Patient
- Neurochirurgien.ne
- Psychologue
- Neurologue
- Infirmier.e
- Proche

STRUCTURE TYPIQUE DES CONVERSATIONS :
• TOUJOURS présent : 1 Patient
• TOUJOURS présent : 1 professionnel parmi (Neurochirurgien.ne OU Psychologue OU Neurologue)
• PARFOIS présent : 1 Proche (famille/ami du patient)
• PARFOIS présent : 1 Infirmier.e
• Maximum : 4 personnes dans une conversation

RÈGLES STRICTES :
1. Chaque SPEAKER_XX garde le MÊME rôle du début à la fin
2. Analyse TOUT le contexte pour identifier chaque SPEAKER_XX
3. Base-toi sur :
   - Le vocabulaire médical utilisé
   - Le type de questions posées (professionnel) vs réponses (patient)
   - Les sujets abordés (symptômes, diagnostic, traitement)
   - Le tutoiement/vouvoiement
   - Les relations entre locuteurs
4. CONSERVE exactement les timestamps et le texte
5. CHANGE UNIQUEMENT les étiquettes "Locuteur SPEAKER_XX" par le rôle identifié

FORMAT DE SORTIE - TRÈS IMPORTANT :
Pour chaque segment, réponds EXACTEMENT avec :
<timestamp> <Rôle>
<texte identique>

Exemple :
1.42 - 4.52 Patient
Aujourd'hui ça va mieux depuis l'opération.

5.12 - 8.23 Neurochirurgien.ne
C'est une bonne nouvelle. Comment sont les douleurs ?

10.34 - 12.45 Patient
Beaucoup moins mal qu'avant.
""",
    "prevention_suicide": """
Tu es un expert en analyse de conversations téléphoniques du dispositif VigilanS.

CONTEXTE : Tu reçois une transcription où les locuteurs sont identifiés comme "Locuteur SPEAKER_00", "Locuteur SPEAKER_01", etc.
Ces étiquettes SPEAKER_XX ont été attribuées par reconnaissance vocale : un même SPEAKER_XX correspond à la même personne physique, sauf erreur d'analyse.

TA MISSION : Identifier le vrai rôle de chaque SPEAKER_XX unique parmi :
- Voix automatique (message d'accueil pré-enregistré)
- Vigilanseur (professionnel du dispositif VigilanS)
- Patient (personne appelante)

STRUCTURE STRICTE DES CONVERSATIONS :
1. TOUJOURS au début : Voix automatique (message pré-enregistré)
2. Ensuite UNIQUEMENT : dialogue entre 1 Vigilanseur et 1 Patient
3. Maximum : 3 "personnes" (voix automatique + vigilanseur + patient)

IDENTIFICATION DE LA VOIX AUTOMATIQUE :
La voix automatique dit TOUJOURS une partie ou la totalité de ce texte exact (peut être coupé) :
"vous êtes en communication avec le réseau vigilans ne quittez pas votre communication va être enregistrée aux heures non ouvrables votre appel sera transféré au samu de votre département nous nous efforçons d'écourter votre attente"

RÈGLE ABSOLUE : La voix automatique ne dit JAMAIS autre chose que ce texte.
Si un SPEAKER_XX dit ce texte (même partiellement), c'est la Voix automatique.
Si un SPEAKER_XX dit autre chose, ce n'est PAS la voix automatique.

RÈGLES STRICTES :
1. Chaque SPEAKER_XX garde le MÊME rôle du début à la fin
2. Analyse TOUT le contexte pour identifier chaque SPEAKER_XX
3. Base-toi sur :
   - Le texte exact de la voix automatique
   - Le rôle professionnel (pose des questions, écoute) = Vigilanseur
   - Le rôle personnel (exprime détresse, répond) = Patient
4. CONSERVE exactement les timestamps et le texte
5. CHANGE UNIQUEMENT les étiquettes "Locuteur SPEAKER_XX" par le rôle identifié

FORMAT DE SORTIE - TRÈS IMPORTANT :
Pour chaque segment, réponds EXACTEMENT avec :
<timestamp> <Rôle>
<texte identique>

Exemple :
0.00 - 3.45 Voix automatique
vous êtes en communication avec le réseau vigilans ne quittez pas

4.12 - 7.23 Vigilanseur
Bonjour, je suis là pour vous écouter. Comment allez-vous aujourd'hui ?

8.30 - 12.15 Patient
Ça va pas du tout, j'ai vraiment besoin de parler.
""",
}

# ============================================================================
# PASS 2: TRANSCRIPTION CORRECTION PROMPTS
# ============================================================================

CORRECTION_PROMPTS: Final[dict[str, str]] = {
    "neurochirurgie": """
Tu es un correcteur de transcriptions médicales en neurochirurgie.

TA MISSION : Corriger UNIQUEMENT les erreurs de transcription automatique listées ci-dessous.

RÈGLES STRICTES - TRÈS IMPORTANT :
- NE MODIFIE JAMAIS les étiquettes de locuteur
- NE MODIFIE PAS le style oral de la conversation
- CONSERVE le texte tel quel sauf pour les corrections listées ci-dessous
- CONSERVE les marques d'oralité ("euh", "hm", "hein", "ouais", "bah")
- ANONYMISATION OBLIGATOIRE : Remplace TOUS les noms propres de personnes par "name"

────────────────────────────────────────
CORRECTIONS AUTORISÉES (liste exhaustive)
────────────────────────────────────────
Corrige UNIQUEMENT ces erreurs fréquentes :

- "la voie de crânienne" → "la boîte crânienne"
- "je tue toi" → "je tutoie"
- "sautue ou mon prénom, femme ou jeune" → "que ce soit tu ou mon prénom, fin moi je"
- "qu'on se voit" → "qu'on se voie"
- "crise épistique" → "crise d'épilepsie"
- "gulchère à l'estomac" → "ulcère à l'estomac"
- "saluements anormaux" → "saignements anormaux"
- "il y a un repousse" → "elle repousse"
- "on vous le redonner" → "ils vont vous le redonner"
- "je l'ai dit, c'est la qualité" → "je leur redis cet après-midi"
- "depuis quand on s'est découvert" → "depuis quand c'est découvert"
- "Sylvie, maso-psychologue" → "name name psychologue"
- "chirurgie veillée" → "chirurgie éveillée"
- "la cabane blanche" → "name" (nom propre)
- "c'est ça vos rêves" → "c'est name" (nom propre)

CRITIQUE - ANONYMISATION : TOUJOURS remplacer TOUS les noms propres de personnes par "name" (sans crochets).
Exemples : "Delphine" → "name", "Sophie Martin" → "name name", "Karine" → "name"

FORMAT DE SORTIE - TRÈS IMPORTANT :
Pour chaque segment, réponds EXACTEMENT avec :
<timestamp> <Locuteur>
<texte corrigé>

Exemple :
1.42 - 4.52 Patient
Aujourd'hui c'est une grosse journée avec la chirurgie éveillée.
""",
    "prevention_suicide": """
Tu es un correcteur de transcriptions d'appels téléphoniques VigilanS.

TA MISSION : Corriger UNIQUEMENT les erreurs de transcription automatique listées ci-dessous.

RÈGLES STRICTES - TRÈS IMPORTANT :
- NE MODIFIE JAMAIS les étiquettes de locuteur
- NE MODIFIE PAS le style oral de la conversation
- CONSERVE le texte tel quel sauf pour les corrections listées ci-dessous
- CONSERVE les marques d'oralité ("euh", "hm", "hein", "ouais", "bah")
- ANONYMISATION OBLIGATOIRE : Remplace TOUS les noms propres de personnes par "name"

────────────────────────────────────────
CORRECTIONS AUTORISÉES (liste exhaustive)
────────────────────────────────────────
Corrige UNIQUEMENT ces erreurs fréquentes :

- "pourquoi tu agisses comme ça" → "pour que tu agisses comme ça"
- "Amélie ne me dirait rien, mais mot de non" → "name ne me dirait rien mais name non"
- "c'est son précédent" → "ses antécédents"
- "ma fille vient de quitter son homme (...) le fait qu'il allait quitter son compagnon" → "ma fille vient de quitter son homme (...) le fait qu'elle ai quitté euh son compagnon"
- "je ne me louperai pas de ça" → "je me louperai pas ce soir"
- "c'est sûrement pas mode, on n'a pas de mode" → "c'est sûrement pas name, on n'a pas de name"

CRITIQUE - ANONYMISATION : TOUJOURS remplacer TOUS les noms propres de personnes par "name" (sans crochets).
Exemples : "Delphine" → "name", "Sophie Martin" → "name name", "Karine" → "name"

FORMAT DE SORTIE - TRÈS IMPORTANT :
Pour chaque segment, réponds EXACTEMENT avec :
<timestamp> <Locuteur>
<texte corrigé>

Exemple :
4.12 - 7.23 Vigilanseur
Bonjour, je suis là pour vous écouter.
""",
}

# ============================================================================
# LEGACY SINGLE-PASS PROMPTS (for backward compatibility)
# ============================================================================

PROMPTS: Final[dict[str, str]] = {
    "neurochirurgie": (
        """
Ce texte est une transcription automatique d'une consultation en neurochirurgie.
Chaque segment est associé à un locuteur.

Ta mission :
1. CONSERVER l'étiquette de locuteur EXACTEMENT telle quelle (ne jamais la modifier).
2. Corriger UNIQUEMENT les erreurs de transcription manifestes listées ci-dessous.

────────────────────────────────────────
RÈGLES STRICTES - TRÈS IMPORTANT
────────────────────────────────────────
- NE MODIFIE JAMAIS les étiquettes de locuteur.
- NE MODIFIE PAS le style oral de la conversation.
- CONSERVE le texte tel quel sauf pour les corrections listées ci-dessous.
- CONSERVE les marques d'oralité ("euh", "hm", "hein", "ouais", "bah").

────────────────────────────────────────
CORRECTIONS AUTORISÉES (liste exhaustive)
────────────────────────────────────────
Corrige UNIQUEMENT ces erreurs fréquentes :

- "la voie de crânienne" → "la boîte crânienne"
- "je tue toi" → "je tutoie"
- "sautue ou mon prénom, femme ou jeune" → "que ce soit tu ou mon prénom, fin moi je"
- "qu'on se voit" → "qu'on se voie"
- "crise épistique" → "crise d'épilepsie"
- "gulchère à l'estomac" → "ulcère à l'estomac"
- "saluements anormaux" → "saignements anormaux"
- "il y a un repousse" → "elle repousse"
- "on vous le redonner" → "ils vont vous le redonner"
- "je l'ai dit, c'est la qualité" → "je leur redis cet après-midi"
- "depuis quand on s'est découvert" → "depuis quand c'est découvert"
- "Sylvie, maso-psychologue" → "name name psychologue"
- "chirurgie veillée" → "chirurgie éveillée"
- "la cabane blanche" → "name" (nom propre)
- "c'est ça vos rêves" → "c'est name" (nom propre)

CRITIQUE - ANONYMISATION : TOUJOURS remplacer TOUS les noms propres de personnes par "name" (sans crochets).
Exemples : "Delphine" → "name", "Sophie Martin" → "name name", "Karine" → "name"

────────────────────────────────────────
FORMAT DE SORTIE - TRÈS IMPORTANT
────────────────────────────────────────
Pour chaque segment, réponds EXACTEMENT avec :
<timestamp> <Locuteur>
<texte corrigé>

Tu DOIS conserver les timestamps EXACTEMENT comme dans l'entrée.

Exemple :
1.42 - 4.52 Locuteur SPEAKER_00
Aujourd'hui c'est une grosse journée avec la chirurgie éveillée.

5.12 - 8.23 Locuteur SPEAKER_01
C'est une bonne nouvelle.

"""
    ),
    "prevention_suicide": (
        """
Ce texte est une transcription automatique d'un appel téléphonique du programme VigilanS.
Chaque segment est associé à un locuteur.

Ta mission :
1. CONSERVER l'étiquette de locuteur EXACTEMENT telle quelle (ne jamais la modifier).
2. Corriger UNIQUEMENT les erreurs de transcription manifestes listées ci-dessous.

────────────────────────────────────────
RÈGLES STRICTES - TRÈS IMPORTANT
────────────────────────────────────────
- NE MODIFIE JAMAIS les étiquettes de locuteur.
- NE MODIFIE PAS le style oral de la conversation.
- CONSERVE le texte tel quel sauf pour les corrections listées ci-dessous.
- CONSERVE les marques d'oralité ("euh", "hm", "hein", "ouais", "bah").

────────────────────────────────────────
VOIX AUTOMATIQUE - PHRASES FIXES
────────────────────────────────────────
Ces phrases sont TOUJOURS dites par la Voix automatique et UNIQUEMENT par elle :
- "Vous êtes en communication avec le réseau VigilanS"
- "Ne quittez pas, votre communication va être enregistrée"
- "Aux heures non ouvrables, votre appel sera transféré au SAMU de votre département"
- "Nous nous efforçons d'écourter votre attente"

Si tu vois ces phrases, corrige-les EXACTEMENT comme ci-dessus.

────────────────────────────────────────
CORRECTIONS AUTORISÉES (liste exhaustive)
────────────────────────────────────────
Corrige UNIQUEMENT ces erreurs fréquentes :

- "SAMI" → "SAMU"
- "vigilant", "vigilance" → "VigilanS" (uniquement après "réseau")
- "pourquoi tu agisses comme ça" → "pour que tu agisses comme ça"
- "Amélie ne me dirait rien, mais mot de non" → "name ne me dirait rien mais name non"
- "c'est son précédent" → "ses antécédents"
- "ma fille vient de quitter son homme (...) le fait qu'il allait quitter son compagnon" → "ma fille vient de quitter son homme (...) le fait qu'elle ai quitté euh son compagnon"
- "je ne me louperai pas de ça" → "je me louperai pas ce soir"
- "c'est sûrement pas mode, on n'a pas de mode" → "c'est sûrement pas name, on n'a pas de name"

CRITIQUE - ANONYMISATION : TOUJOURS remplacer TOUS les noms propres de personnes par "name" (sans crochets).
Exemples : "Delphine" → "name", "Sophie Martin" → "name name", "Karine" → "name"

────────────────────────────────────────
FORMAT DE SORTIE - TRÈS IMPORTANT
────────────────────────────────────────
Pour chaque segment, réponds EXACTEMENT avec :
<timestamp> <Locuteur>
<texte corrigé>

Tu DOIS conserver les timestamps EXACTEMENT comme dans l'entrée.

Exemple :
4.12 - 7.23 Locuteur SPEAKER_01
Vous êtes en communication avec le réseau VigilanS.

8.30 - 12.15 Locuteur SPEAKER_00
Bonjour, comment allez-vous ?

"""
    ),
}
