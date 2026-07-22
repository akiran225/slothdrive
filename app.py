# -*- coding: utf-8 -*-
"""
SlothDrive — V2 (bascule automatique Gemini -> Groq)
Un compagnon IA pour apprendre l'anglais en discutant chaque jour.

Chaîne de secours :
1. Gemini (tier gratuit, priorité)  -> quand les crédits sont épuisés :
2. Groq (bascule automatique)

Voix -> texte : Groq Whisper. Texte -> voix : Groq Orpheus (bouton on/off).

Secrets à définir dans le Space :
- GEMINI_API_KEY  (aistudio.google.com)
- GROQ_API_KEY    (console.groq.com/keys)
Optionnels (mémoire persistante) : HF_DATASET_REPO, HF_TOKEN
"""

import os
import json
import datetime as dt

import streamlit as st
from groq import Groq

try:
    from google import genai
    from google.genai import types as gtypes
    _HAS_GEMINI_SDK = True
except Exception:
    _HAS_GEMINI_SDK = False

# Charge un fichier .env en local (aucun effet sur Hugging Face, où les secrets
# sont déjà injectés comme variables d'environnement).
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------------------------------------------------------------------
# CONFIGURATION (vérifiée juillet 2026 — change ici sans toucher au reste)
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash"          # alt "peu d'hallucination" : "gemini-2.5-flash-lite"
GROQ_CHAT_MODEL = "openai/gpt-oss-120b"    # gros modèle => invente moins ; alt "llama-3.3-70b-versatile"
GROQ_IS_REASONING = True                   # les gpt-oss sont "reasoning"
STT_MODEL = "whisper-large-v3-turbo"
TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "troy"

MEMORY_PATH = "memory.json"


def _secret(name):
    """Lit une clé où qu'elle soit : variables d'environnement (Hugging Face, .env local)
    ou st.secrets (Streamlit Community Cloud)."""
    v = os.environ.get(name)
    if v:
        return v
    try:
        return st.secrets[name]
    except Exception:
        return None

LEVELS = {
    "A1 — débutant": (
        "Use very simple English (short sentences, common words, present tense). Speak slowly. "
        "You may add a short French hint in parentheses when the user is stuck. Keep replies to 1-3 sentences."
    ),
    "A2 — élémentaire": (
        "Use simple everyday English. Introduce a few useful words and explain them briefly. "
        "A short French hint only if the user seems lost. Keep replies to 2-4 sentences."
    ),
    "B1 — intermédiaire": (
        "Use natural conversational English, idioms and richer vocabulary. Avoid French. "
        "Encourage longer answers."
    ),
    "B2 — avancé": (
        "Use rich, natural English including idioms, humour and abstract topics. Never use French. "
        "Challenge the user with follow-up questions and light debate."
    ),
}

# ---------------------------------------------------------------------------
# CLIENTS
# ---------------------------------------------------------------------------
@st.cache_resource
def get_groq():
    key = _secret("GROQ_API_KEY")
    return Groq(api_key=key) if key else None


@st.cache_resource
def get_gemini():
    key = _secret("GEMINI_API_KEY")
    if key and _HAS_GEMINI_SDK:
        try:
            return genai.Client(api_key=key)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# MÉMOIRE LONG TERME
# ---------------------------------------------------------------------------
def _empty_memory():
    return {"profile": {}, "vocab_learned": [], "frequent_errors": [], "sessions": [], "updated_at": None}


def load_memory():
    repo, token = _secret("HF_DATASET_REPO"), _secret("HF_TOKEN")
    if repo and token:
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo_id=repo, filename=MEMORY_PATH,
                                repo_type="dataset", token=token)
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    if os.path.exists(MEMORY_PATH):
        try:
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return _empty_memory()


def save_memory(memory):
    memory["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    repo, token = _secret("HF_DATASET_REPO"), _secret("HF_TOKEN")
    if repo and token:
        try:
            from huggingface_hub import HfApi
            HfApi().upload_file(path_or_fileobj=MEMORY_PATH, path_in_repo=MEMORY_PATH,
                                repo_id=repo, repo_type="dataset", token=token)
        except Exception as e:
            st.warning(f"Mémoire locale OK mais pas synchronisée sur HF ({e}).")


def memory_context(memory):
    if not any([memory["profile"], memory["vocab_learned"], memory["frequent_errors"]]):
        return "You don't know this user yet. Ask a few friendly questions to get to know them."
    lines = []
    if memory["profile"]:
        lines.append("What you remember about the user: "
                     + "; ".join(f"{k}: {v}" for k, v in memory["profile"].items()) + ".")
    if memory["frequent_errors"]:
        lines.append("Recurring mistakes to gently watch for: " + "; ".join(memory["frequent_errors"][:8]) + ".")
    if memory["vocab_learned"]:
        lines.append("Words already practised (reuse them): " + ", ".join(memory["vocab_learned"][-15:]) + ".")
    if memory["sessions"]:
        lines.append("Last time you talked about: " + memory["sessions"][-1].get("topic", "various things") + ".")
    return " ".join(lines)


def build_system_prompt(level_key, memory):
    return (
        "You are 'Companion', a warm, curious and patient English-speaking companion who helps the "
        "user become fluent simply by chatting every day, like a friend who is also a teacher.\n\n"
        "CORE RULES:\n"
        "- Never judge. Mistakes are normal.\n"
        "- Do NOT correct every sentence. Let the conversation flow; correct at the right moment, only what matters.\n"
        "- Correct kindly in one line, e.g. \"I understood you! We'd usually say 'I went to work'. Want to try again?\" then keep going.\n"
        "- Show real personality: humour, curiosity, opinions, small anecdotes. Ask one question at a time and bounce off answers.\n"
        "- Focus on understanding what the user MEANS before improving how they say it.\n"
        "- Never claim to be human. Never claim certainty about pronunciation you can't verify.\n\n"
        f"LEVEL — {level_key}:\n{LEVELS[level_key]}\n\n"
        f"MEMORY:\n{memory_context(memory)}\n\n"
        "Keep replies focused and conversational."
    )


# ---------------------------------------------------------------------------
# CHAT : bascule automatique Gemini -> Groq
# ---------------------------------------------------------------------------
def _gemini_reply(client, system_prompt, history):
    contents = [
        gtypes.Content(
            role=("model" if m["role"] == "assistant" else "user"),
            parts=[gtypes.Part(text=m["content"])],
        )
        for m in history
    ]
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=gtypes.GenerateContentConfig(
            system_instruction=system_prompt, temperature=0.7, max_output_tokens=400,
        ),
    )
    return (resp.text or "").strip()


def _groq_reply(client, system_prompt, history):
    kwargs = dict(
        model=GROQ_CHAT_MODEL,
        messages=[{"role": "system", "content": system_prompt}] + history,
        temperature=0.7, max_tokens=400,
    )
    if GROQ_IS_REASONING:
        kwargs["reasoning_effort"] = "low"
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception:
        kwargs.pop("reasoning_effort", None)
        resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()


def chat_reply(system_prompt, history):
    """Renvoie (texte, fournisseur). Essaie Gemini, bascule sur Groq si épuisé/indispo."""
    gemini = get_gemini()
    groq = get_groq()

    if gemini and not st.session_state.get("gemini_exhausted"):
        try:
            return _gemini_reply(gemini, system_prompt, history), "Gemini"
        except Exception as e:
            # Quota atteint ou erreur -> on arrête Gemini pour la session et on bascule
            if "429" in str(e) or "quota" in str(e).lower() or "exhaust" in str(e).lower():
                st.session_state.gemini_exhausted = True
                st.toast("Crédits Gemini épuisés — bascule sur Groq.", icon="🔁")
    if groq:
        return _groq_reply(groq, system_prompt, history), "Groq"
    return ("(Aucun fournisseur disponible : vérifie GEMINI_API_KEY / GROQ_API_KEY.)", "—")


# ---------------------------------------------------------------------------
# VOIX (Groq) — indépendant de la bascule chat
# ---------------------------------------------------------------------------
def transcribe(audio_bytes):
    client = get_groq()
    if not client:
        return None
    tr = client.audio.transcriptions.create(
        model=STT_MODEL, file=("speech.wav", audio_bytes), language="en", response_format="text",
    )
    return tr if isinstance(tr, str) else getattr(tr, "text", "")


def synthesize(text):
    client = get_groq()
    if not client:
        return None
    try:
        speech = client.audio.speech.create(
            model=TTS_MODEL, voice=TTS_VOICE, input=text[:1000], response_format="wav",
        )
        return speech.read()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# BILAN + MISE À JOUR MÉMOIRE (utilise le fournisseur chat courant)
# ---------------------------------------------------------------------------
def _json_completion(prompt):
    """Petit appel JSON via Gemini si dispo, sinon Groq."""
    gemini, groq = get_gemini(), get_groq()
    raw = ""
    if gemini and not st.session_state.get("gemini_exhausted"):
        try:
            r = gemini.models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
                config=gtypes.GenerateContentConfig(temperature=0.2, max_output_tokens=500),
            )
            raw = (r.text or "").strip()
        except Exception:
            raw = ""
    if not raw and groq:
        kw = dict(model=GROQ_CHAT_MODEL, messages=[{"role": "user", "content": prompt}],
                  temperature=0.2, max_tokens=500)
        if GROQ_IS_REASONING:
            kw["reasoning_effort"] = "low"
        try:
            raw = groq.chat.completions.create(**kw).choices[0].message.content.strip()
        except Exception:
            raw = ""
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return None


def analyze_session(history):
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    prompt = (
        "You are an English coach. Analyse this conversation and return ONLY a JSON object with keys:\n"
        '{"grammar_errors": int, "new_words": [str], "strengths": [str], "next_focus": str, '
        '"topic": str, "encouragement": str}\n'
        "Count errors from USER turns only. 'encouragement' is one warm sentence in French.\n\n"
        f"Conversation:\n{transcript}"
    )
    return _json_completion(prompt)


def update_memory(history, memory):
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    prompt = (
        "From this conversation, extract durable facts about the user. Return ONLY a JSON object:\n"
        '{"profile": {"key": "value"}, "frequent_errors": [str], "vocab_learned": [str]}\n'
        "profile = studies, work, hobbies, projects, events (short values). "
        "frequent_errors = recurring English mistakes. vocab_learned = new words the user used.\n\n"
        f"Conversation:\n{transcript}"
    )
    data = _json_completion(prompt)
    if not data:
        return memory
    memory["profile"].update({k: v for k, v in data.get("profile", {}).items() if v})
    for e in data.get("frequent_errors", []):
        if e and e not in memory["frequent_errors"]:
            memory["frequent_errors"].append(e)
    for w in data.get("vocab_learned", []):
        if w and w not in memory["vocab_learned"]:
            memory["vocab_learned"].append(w)
    return memory


# ---------------------------------------------------------------------------
# INTERFACE
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SlothDrive", page_icon="🦥", layout="centered")

for key, val in {"memory": None, "history": [], "summary": None,
                 "last_audio_id": None, "gemini_exhausted": False}.items():
    if key not in st.session_state:
        st.session_state[key] = val
if st.session_state.memory is None:
    st.session_state.memory = load_memory()

with st.sidebar:
    st.header("⚙️ Réglages")
    level = st.selectbox("Ton niveau d'anglais", list(LEVELS.keys()), index=0)
    tts_on = st.toggle("Écouter les réponses (voix)", value=True)

    # État de la chaîne de secours
    g_ok = get_gemini() is not None
    q_ok = get_groq() is not None
    if g_ok and not st.session_state.gemini_exhausted:
        st.caption("🟢 Actif : Gemini (Groq en secours)")
    elif q_ok:
        st.caption("🟡 Actif : Groq" + (" (Gemini épuisé)" if st.session_state.gemini_exhausted else ""))
    else:
        st.caption("🔴 Aucune clé API détectée")

    st.divider()
    if st.button("🏁 Terminer la session") and st.session_state.history:
        with st.spinner("Je prépare ton bilan..."):
            st.session_state.summary = analyze_session(st.session_state.history)
            st.session_state.memory = update_memory(st.session_state.history, st.session_state.memory)
            if st.session_state.summary:
                st.session_state.memory["sessions"].append(
                    {"date": dt.date.today().isoformat(),
                     "topic": st.session_state.summary.get("topic", "")}
                )
            save_memory(st.session_state.memory)
        st.session_state.history = []

    with st.expander("🧠 Ce que je retiens de toi"):
        m = st.session_state.memory
        if m["profile"]:
            for k, v in m["profile"].items():
                st.write(f"• **{k}** : {v}")
        else:
            st.caption("Encore rien — on apprend à se connaître.")
        if m["frequent_errors"]:
            st.caption("À surveiller : " + ", ".join(m["frequent_errors"][:5]))

st.title("🦥 SlothDrive")
st.caption("Parle anglais chaque jour avec un compagnon qui se souvient de toi.")

if st.session_state.summary:
    s = st.session_state.summary
    with st.container(border=True):
        st.subheader("📊 Ton bilan")
        c1, c2 = st.columns(2)
        c1.metric("Fautes de grammaire", s.get("grammar_errors", "—"))
        c2.metric("Nouveaux mots", len(s.get("new_words", [])))
        if s.get("new_words"):
            st.write("**Mots à retenir :** " + ", ".join(s["new_words"]))
        if s.get("strengths"):
            st.write("**Points forts :** " + ", ".join(s["strengths"]))
        if s.get("next_focus"):
            st.write("**Prochain focus :** " + s["next_focus"])
        if s.get("encouragement"):
            st.success(s["encouragement"])
    if st.button("Nouvelle conversation"):
        st.session_state.summary = None
        st.rerun()

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

audio = st.audio_input("🎙️ Parle en anglais (ou écris ci-dessous)")
user_text = None
if audio is not None and audio.file_id != st.session_state.last_audio_id:
    st.session_state.last_audio_id = audio.file_id
    with st.spinner("Je t'écoute..."):
        user_text = transcribe(audio.getvalue())

typed = st.chat_input("Écris ton message en anglais...")
if typed:
    user_text = typed

if user_text:
    st.session_state.history.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    system_prompt = build_system_prompt(level, st.session_state.memory)
    with st.chat_message("assistant"):
        with st.spinner("..."):
            reply, provider = chat_reply(system_prompt, st.session_state.history)
        st.write(reply)
        st.caption(f"via {provider}")
        if tts_on:
            ab = synthesize(reply)
            if ab:
                st.audio(ab, format="audio/wav")

    st.session_state.history.append({"role": "assistant", "content": reply})
