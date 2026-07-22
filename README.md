---
title: SlothDrive
emoji: 🦥
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# SlothDrive — apprendre l'anglais en discutant

Un compagnon IA avec qui parler anglais chaque jour : il discute, corrige en douceur,
se souvient de toi, te lit ses réponses, et fait un bilan à la fin de chaque conversation.

## Chaîne de secours (le point clé)

- **Gemini** est utilisé en priorité (tier gratuit très généreux).
- Quand les crédits Gemini sont épuisés, l'appli **bascule automatiquement sur Groq**.
- Un indicateur dans la barre latérale montre qui répond (🟢 Gemini / 🟡 Groq).

La voix (transcription + lecture) passe toujours par Groq, indépendamment de la bascule.

## Déploiement sur Hugging Face Spaces

1. Crée un Space → **SDK : Streamlit**.
2. Dépose `app.py`, `requirements.txt`, `README.md`.
3. Dans **Settings → Variables and secrets**, ajoute :
   - `GEMINI_API_KEY` = clé Google AI Studio (aistudio.google.com)
   - `GROQ_API_KEY` = clé Groq (console.groq.com/keys)
4. C'est prêt.

## Mémoire persistante (recommandé)

Sur un Space gratuit le disque est **éphémère** (mémoire effacée au redémarrage).
Pour la conserver : crée un **dataset privé** HF et ajoute deux secrets :
- `HF_DATASET_REPO` = `ton-pseudo/companion-memory`
- `HF_TOKEN` = token HF avec accès en écriture

## Modèles (modifiables en haut de `app.py`)

- Chat prioritaire : `gemini-2.5-flash` (alt peu d'hallucination : `gemini-2.5-flash-lite`)
- Chat secours : `openai/gpt-oss-120b` (alt : `llama-3.3-70b-versatile`)
- Voix → texte : `whisper-large-v3-turbo` (Groq)
- Texte → voix : `canopylabs/orpheus-v1-english` (Groq)
