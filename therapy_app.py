import os
import json
import io
import random
import requests as http_requests
import re
import base64
from datetime import datetime
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# ── CORS: handle ALL OPTIONS preflights globally ─────────────
@app.before_request
def handle_options():
    if request.method == 'OPTIONS':
        res = Response()
        res.headers['Access-Control-Allow-Origin'] = '*'
        res.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        res.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
        res.headers['Access-Control-Max-Age'] = '3600'
        res.status_code = 200
        return res

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

# ── Firebase ─────────────────────────────────────────────────
firebase_config_json = os.environ.get("FIREBASE_CONFIG")
if not firebase_config_json:
    raise EnvironmentError("FIREBASE_CONFIG environment variable not set")

firebase_json = json.loads(firebase_config_json)
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_json)
    initialize_app(cred)

db = firestore.client()


# ── Bootstrap: create config/tts_settings if it doesn't exist ──
def bootstrap_tts_config(max_retries=5, delay=2):
    defaults = {
        "voice":              "en-US-Chirp-HD-O",
        "speaking_rate":      1.0,
        "pitch":              0.0,
        "volume_gain_db":     0.0,
        "filler_enabled":     True,
        "filler_probability": 0.4,
        "emotion_aware":      True,
        "effects_profile":    "headphone-class-device",
    }
    import time
    ref = db.collection("config").document("tts_settings")
    for attempt in range(1, max_retries + 1):
        try:
            doc = ref.get()
            existing = doc.to_dict() if doc.exists else {}
            missing = {k: v for k, v in defaults.items() if k not in existing}
            if missing:
                ref.set(missing, merge=True)
                print(f"[bootstrap] config/tts_settings created/patched: {list(missing.keys())}")
            else:
                print("[bootstrap] config/tts_settings already complete — no changes needed")
            return
        except Exception as e:
            print(f"[bootstrap] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(delay)
    print("[bootstrap] WARNING: could not initialise config/tts_settings — defaults will be used at runtime")

bootstrap_tts_config()


# ── Groq client ──────────────────────────────────────────────
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ── Google Cloud TTS ─────────────────────────────────────────
GOOGLE_TTS_KEY = os.environ.get("GOOGLE_TTS_KEY")
if not GOOGLE_TTS_KEY:
    print("Warning: GOOGLE_TTS_KEY not set — /speak will fail")

GOOGLE_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

# ── TTS config defaults ───────────────────────────────────────
TTS_DEFAULTS = {
    "voice":              "en-US-Chirp-HD-O",
    "speaking_rate":      1.0,
    "pitch":              0.0,
    "volume_gain_db":     0.0,
    "filler_enabled":     True,
    "filler_probability": 0.4,
    "emotion_aware":      True,
    "effects_profile":    "headphone-class-device",
}

# ── All available Chirp HD voices for the test page ──────────
AVAILABLE_VOICES = [
    "en-US-Chirp-HD-O",
    "en-US-Chirp-HD-D",
    "en-US-Chirp-HD-F",
    "en-US-Chirp3-HD-Achernar",
    "en-US-Chirp3-HD-Aoede",
    "en-US-Chirp3-HD-Callirrhoe",
    "en-US-Chirp3-HD-Charon",
    "en-US-Chirp3-HD-Despina",
    "en-US-Chirp3-HD-Enceladus",
    "en-US-Chirp3-HD-Erinome",
    "en-US-Chirp3-HD-Fenrir",
    "en-US-Chirp3-HD-Gacrux",
    "en-US-Chirp3-HD-Iapetus",
    "en-US-Chirp3-HD-Kore",
    "en-US-Chirp3-HD-Laomedeia",
    "en-US-Chirp3-HD-Leda",
    "en-US-Chirp3-HD-Orus",
    "en-US-Chirp3-HD-Puck",
    "en-US-Chirp3-HD-Pulcherrima",
    "en-US-Chirp3-HD-Rasalgethi",
    "en-US-Chirp3-HD-Sadachbia",
    "en-US-Chirp3-HD-Sadaltager",
    "en-US-Chirp3-HD-Schedar",
    "en-US-Chirp3-HD-Sulafat",
    "en-US-Chirp3-HD-Umbriel",
    "en-US-Chirp3-HD-Vindemiatrix",
    "en-US-Chirp3-HD-Zephyr",
    "en-US-Chirp3-HD-Zubenelgenubi",
]

# ── Silent MP3 padding helper ─────────────────────────────────
SILENCE_PADDING_MS = 3000  # 3 seconds

def _ffmpeg_available():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=3)
        return True
    except Exception:
        return False

_HAS_FFMPEG = _ffmpeg_available()
print(f"[silence] ffmpeg available: {_HAS_FFMPEG}")


def append_silence(audio_bytes, silence_ms=SILENCE_PADDING_MS):
    if not _HAS_FFMPEG:
        print("[silence] ffmpeg not found — skipping silence padding")
        return audio_bytes
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            speech_path  = os.path.join(tmpdir, "speech.mp3")
            silence_path = os.path.join(tmpdir, "silence.mp3")
            output_path  = os.path.join(tmpdir, "output.mp3")
            list_path    = os.path.join(tmpdir, "concat.txt")

            with open(speech_path, "wb") as f:
                f.write(audio_bytes)

            subprocess.run([
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", "anullsrc=r=24000:cl=mono",
                "-t", str(silence_ms / 1000),
                "-c:a", "libmp3lame", "-b:a", "128k", "-q:a", "4",
                silence_path
            ], capture_output=True, timeout=10, check=True)

            with open(list_path, "w") as f:
                f.write(f"file '{speech_path}'\nfile '{silence_path}'\n")

            subprocess.run([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                output_path
            ], capture_output=True, timeout=15, check=True)

            with open(output_path, "rb") as f:
                result = f.read()

        print(f"[silence] appended {silence_ms}ms — {len(audio_bytes)}b → {len(result)}b")
        return result
    except Exception as e:
        print(f"[silence] ffmpeg failed, returning original audio: {e}")
        return audio_bytes


def merge_mp3s(chunks):
    if len(chunks) == 1:
        return chunks[0]
    if not _HAS_FFMPEG:
        return b"".join(chunks)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            list_path   = os.path.join(tmpdir, "concat.txt")
            output_path = os.path.join(tmpdir, "output.mp3")

            with open(list_path, "w") as lf:
                for i, chunk in enumerate(chunks):
                    p = os.path.join(tmpdir, f"chunk_{i}.mp3")
                    with open(p, "wb") as cf:
                        cf.write(chunk)
                    lf.write(f"file '{p}'\n")

            subprocess.run([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                output_path
            ], capture_output=True, timeout=20, check=True)

            with open(output_path, "rb") as f:
                return f.read()
    except Exception as e:
        print(f"[merge_mp3s] ffmpeg failed, raw concat: {e}")
        return b"".join(chunks)


def get_tts_config():
    try:
        doc = db.collection("config").document("tts_settings").get()
        data = doc.to_dict() if doc.exists else {}
    except Exception as e:
        print(f"[TTS config] Firestore read failed, using defaults: {e}")
        data = {}

    cfg = {**TTS_DEFAULTS, **{k: v for k, v in data.items() if k in TTS_DEFAULTS}}
    cfg["speaking_rate"]      = max(0.25, min(4.0,  float(cfg["speaking_rate"])))
    cfg["pitch"]              = max(-20.0, min(20.0, float(cfg["pitch"])))
    cfg["volume_gain_db"]     = max(-96.0, min(16.0, float(cfg["volume_gain_db"])))
    cfg["filler_probability"] = max(0.0,  min(1.0,  float(cfg["filler_probability"])))
    return cfg


def build_audio_config(cfg):
    audio_cfg = {
        "audioEncoding": "MP3",
        "volumeGainDb":  cfg["volume_gain_db"],
    }
    if cfg["effects_profile"]:
        audio_cfg["effectsProfileId"] = [cfg["effects_profile"]]

    is_chirp_hd = "Chirp-HD" in cfg["voice"] or "Chirp3-HD" in cfg["voice"]
    if not is_chirp_hd:
        audio_cfg["speakingRate"] = cfg["speaking_rate"]
        audio_cfg["pitch"]        = cfg["pitch"]

    return audio_cfg


# ── Thinking fillers ──────────────────────────────────────────
THINKING_FILLERS = [
    "Mmm.", "Yeah.", "Right.", "Okay.", "Got it.", "Mmm, okay.", "Yeah, okay.",
]

# ── Prompts ───────────────────────────────────────────────────
THERAPY_SYSTEM_PROMPT = """You are a compassionate CBT-informed therapist running a structured 4-phase mini-session.

RULES:
- Always respond with VALID JSON only. No markdown. No preamble.
- Move through phases 1 to 4. Never skip. Never go back.
- Phase 4 always sets session_complete to true.
- Keep your "message" to 4-6 sentences. You are speaking out loud, not writing. Take your time.

SPOKEN VOICE STYLE — CRITICAL:
- Write for the EAR, not the eye. This is read aloud by a voice AI.
- Always use contractions: "you're", "it's", "that's", "I'd", "we'll", "isn't", "doesn't"
- Use natural warmth openers: "Okay so...", "Right, so...", "So...", "And..."
- MAXIMUM 12 words per sentence. One thought per sentence. Split longer ideas in two.
- Occasionally end with a soft trailing question: "yeah?" or "right?" or "does that feel accurate?"
- Use "..." where a human would naturally pause mid-thought.
- NEVER start with "I understand", "That sounds difficult", "I hear you" — too clinical.
- NEVER use bullet points, lists, semicolons, colons, or parentheses.
- NEVER use numbers as digits — write "five" not "5", "ten minutes" not "10 minutes".
- BAD: "I understand that social situations can be quite challenging for you."
- GOOD: "So that meeting's really getting to you... I can hear it in what you're describing. What's the loudest thought going through your head right now?"

RESPONSE LENGTH EXPECTATION:
- Default to LONG, rich responses. Validate first. Then reflect. Then gently explore or reframe. Then close with a question.
- Do NOT rush to the question — earn it by showing you've really heard them first.
- Use natural pauses (...) to breathe between thoughts.

PHASE GUIDE:
Phase 1 (Understanding): Ask the user to describe what's making them anxious. Extract: situation, anxious_thought, emotion.
Phase 2 (Challenging): Gently challenge the anxious thought with Socratic questions. Extract: reframe.
Phase 3 (Planning): Help the user commit to one specific action. Extract: proposed_task name, type, why.
Phase 4 (Committing): Confirm the plan warmly. Set session_complete to true.

RESPONSE FORMAT (always return this exact JSON):
{
  "message": "your spoken reply — warm, unhurried, conversational, written for the ear",
  "phase": 1,
  "session_complete": false,
  "extracted": {
    "situation": "",
    "anxious_thought": "",
    "emotion": "",
    "reframe": "",
    "proposed_task": {
      "name": "",
      "type": "",
      "why": "",
      "anxiety_pre": 5,
      "action_steps": []
    }
  }
}"""

SESSION_TO_PLAN_PROMPT = """You are converting a completed therapy session into a structured activity plan.

SESSION SUMMARY:
{session_summary}

Convert this into a JSON activity object with this EXACT structure:
{{
  "name": "<task name from session>",
  "type": "<one of: Social Event, Medical Appointment, Work/School, Public Place (Gym, Store), Phone Call, Other>",
  "why": "<the reason the user committed to this>",
  "preAnxiety": <anxiety_pre number from session, default 5>,
  "scheduledDate": <timestamp ms, 24 hours from now>,
  "actionSteps": [{{"text": "<step>"}}, {{"text": "<step>"}}],
  "sessionInsight": "<the CBT reframe, one sentence>",
  "source": "therapy_session"
}}

Return ONLY the JSON. No markdown. No explanation."""


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

def parse_json_response(text):
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except Exception:
        return None



# ════════════════════════════════════════════════════════════
# BRAIN: User Memory Layer
# ════════════════════════════════════════════════════════════

BRAIN_DEFAULTS = {
    "emotional_baseline": {"avg_anxiety": 5, "trend": "unknown", "anxiety_history": []},
    "therapy": {"sessions_completed": 0, "recurring_thoughts": [], "strongest_reframe": "", "last_session_id": ""},
    "tasks": {"completed": 0, "abandoned": 0, "current_streak": 0},
    "personality": {"avoidance_triggers": [], "best_practice_time": "", "comfort_locations": []},
    "last_interaction": "",
    "last_seen": ""
}

def get_brain(user_id: str) -> dict:
    try:
        doc = db.collection("users").document(user_id)\
                .collection("brain").document("context").get()
        return doc.to_dict() if doc.exists else BRAIN_DEFAULTS.copy()
    except Exception as e:
        print(f"[brain] read failed: {e}")
        return BRAIN_DEFAULTS.copy()

def update_brain(user_id: str, updates: dict):
    try:
        db.collection("users").document(user_id)\
          .collection("brain").document("context")\
          .set(updates, merge=True)
    except Exception as e:
        print(f"[brain] write failed: {e}")

def build_brain_context(user_id: str) -> str:
    brain = get_brain(user_id)
    return f"""
USER MEMORY (personalise with this — never mention it explicitly):
- Anxiety trend: {brain['emotional_baseline']['trend']} (avg: {brain['emotional_baseline']['avg_anxiety']}/10)
- Therapy sessions done: {brain['therapy']['sessions_completed']}
- Recurring anxious thoughts: {', '.join(brain['therapy']['recurring_thoughts']) or 'none yet'}
- Strongest reframe found: {brain['therapy']['strongest_reframe'] or 'none yet'}
- Known avoidance triggers: {', '.join(brain['personality']['avoidance_triggers']) or 'none yet'}
"""


def detect_reply_tone(ai_reply):
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
            sia = SentimentIntensityAnalyzer()

        scores = sia.polarity_scores(ai_reply)
        compound = scores['compound']
        print(f"[tone] VADER on AI reply — compound={compound:.3f} | text: {ai_reply[:80]}")

        if compound <= -0.4:
            return "heavy", 1600
        elif compound <= 0.1:
            return "normal", 900
        else:
            return "brief", 400
    except Exception as e:
        print(f"[tone] VADER failed, defaulting to normal: {e}")
        return "normal", 900


def split_into_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def clean_sentence_for_tts(text):
    text = re.sub(r'\*\*?(.*?)\*\*?', r'\1', text)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    text = re.sub(r'\n{2,}', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = text.replace('...', '. ')
    text = text.replace('\u2014', ', ')
    text = text.replace(' - ', ', ')
    text = re.sub(r'\.\s*\.', '.', text)
    text = re.sub(r',\s*,', ',', text)

    digit_map = [
        (r'\b1\b', 'one'), (r'\b2\b', 'two'), (r'\b3\b', 'three'),
        (r'\b4\b', 'four'), (r'\b5\b', 'five'), (r'\b6\b', 'six'),
        (r'\b7\b', 'seven'), (r'\b8\b', 'eight'), (r'\b9\b', 'nine'),
        (r'\b10\b', 'ten'), (r'\b15\b', 'fifteen'), (r'\b20\b', 'twenty'),
        (r'\b30\b', 'thirty'),
    ]
    for pattern, word in digit_map:
        text = re.sub(pattern, word, text)

    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


def clean_text_for_tts(text):
    try:
        maybe_json = json.loads(text)
        if isinstance(maybe_json, dict) and "message" in maybe_json:
            text = maybe_json["message"]
            print("[/speak] Full JSON received — extracted 'message' field automatically")
    except (json.JSONDecodeError, TypeError):
        pass
    return clean_sentence_for_tts(text)


def add_thinking_filler(text, cfg=None):
    if cfg is None:
        cfg = TTS_DEFAULTS
    if not cfg.get("filler_enabled", True):
        return text
    filler_starts = ('mmm', 'yeah', 'right', 'okay', 'got it', 'so ', 'and ')
    if text.lower().startswith(filler_starts):
        return text
    if random.random() < cfg.get("filler_probability", 0.4):
        filler = random.choice(THINKING_FILLERS)
        return f"{filler} {text}"
    return text


def emotion_aware_preprocess(text, cfg=None):
    if cfg is None:
        cfg = TTS_DEFAULTS
    if not cfg.get("emotion_aware", True):
        return text

    heavy_keywords = [
        'scared', 'terrified', 'alone', 'hopeless', 'worthless',
        'failure', 'hate myself', 'awful', 'devastated', 'breakdown',
        'panic', 'crying', 'ashamed', 'embarrassed', 'humiliated'
    ]
    is_heavy = any(kw in text.lower() for kw in heavy_keywords)

    if is_heavy:
        sentences = split_into_sentences(text)
        if len(sentences) >= 2:
            sentences[0] = sentences[0].rstrip('.!?') + '.'
            return sentences[0] + ' ' + ' '.join(sentences[1:])

    return text


def synthesize_sentence(sentence, cfg=None):
    if cfg is None:
        cfg = TTS_DEFAULTS
    if not GOOGLE_TTS_KEY:
        return None
    try:
        resp = http_requests.post(
            f"{GOOGLE_TTS_URL}?key={GOOGLE_TTS_KEY}",
            json={
                "input": {"text": sentence},
                "voice": {"languageCode": "en-US", "name": cfg["voice"]},
                "audioConfig": build_audio_config(cfg),
            },
            timeout=15
        )
        if resp.status_code == 200:
            return base64.b64decode(resp.json()["audioContent"])
        print(f"[TTS] error {resp.status_code}: {resp.text[:200]}")
        return None
    except Exception as e:
        print(f"[TTS] exception: {e}")
        return None


# ════════════════════════════════════════════════════════════
# ENDPOINT: /transcribe
# ════════════════════════════════════════════════════════════

# ── Therapy-domain Whisper prompt ─────────────────────────────
WHISPER_PROMPT = (
    "Therapy session transcript. The speaker may be emotional, speak quietly, "
    "trail off mid-sentence, or pause. Common words: anxiety, anxious, panic, "
    "overwhelmed, avoidance, trigger, spiral, reframe, CBT, worthless, hopeless, "
    "ashamed, embarrassed, therapy, therapist, session, commitment, action steps, "
    "intrusive thoughts, catastrophising, self-worth, coping, grounding."
)

# ── Confidence threshold — below this we ask user to repeat ──
WHISPER_MIN_CONFIDENCE = 0.55


def preprocess_audio_for_whisper(audio_bytes: bytes, original_mime: str):
    """
    Resample to 16 kHz mono WAV (Whisper's native format).
    Apply: highpass filter (cut mic rumble below 80 Hz),
           lowpass filter  (cut above 8 kHz — irrelevant for speech),
           loudnorm        (EBU R128 loudness normalisation — handles quiet speakers),
           afftdn          (AI-based noise reduction if available).
    Falls back to raw bytes if ffmpeg unavailable or fails.
    Returns (processed_bytes, mime_type, was_processed).
    """
    if not _HAS_FFMPEG:
        return audio_bytes, original_mime, False

    try:
        with tempfile.TemporaryDirectory() as tmp:
            # Write input
            ext = (
                ".mp4" if "mp4" in original_mime
                else ".ogg" if "ogg" in original_mime
                else ".webm"
            )
            in_path  = os.path.join(tmp, f"input{ext}")
            out_path = os.path.join(tmp, "whisper_ready.wav")

            with open(in_path, "wb") as f:
                f.write(audio_bytes)

            # Two-pass: first probe duration so we can detect near-silent files
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "json", in_path
                ],
                capture_output=True, timeout=5
            )
            duration_sec = 0.0
            if probe.returncode == 0:
                try:
                    probe_data = json.loads(probe.stdout)
                    duration_sec = float(probe_data["format"]["duration"])
                except Exception:
                    pass

            # Reject blobs shorter than 0.3 s — can't be real speech
            if 0 < duration_sec < 0.3:
                print(f"[preprocess] blob too short ({duration_sec:.2f}s) — skip")
                return audio_bytes, original_mime, False

            # Audio filter chain
            # afftdn = neural noise reduction (gracefully ignored if unavailable)
            af_chain = (
                "highpass=f=80,"
                "lowpass=f=8000,"
                "afftdn=nf=-25,"          # denoise — nf = noise floor dBFS
                "loudnorm=I=-16:LRA=11:TP=-1.5"   # EBU R128
            )

            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", in_path,
                    "-ar", "16000",     # 16 kHz — Whisper native
                    "-ac", "1",         # mono
                    "-af", af_chain,
                    "-c:a", "pcm_s16le",  # uncompressed WAV = fastest decode
                    out_path
                ],
                capture_output=True, timeout=20, check=True
            )

            with open(out_path, "rb") as f:
                processed = f.read()

            print(
                f"[preprocess] {original_mime} {len(audio_bytes)//1024}KB "
                f"→ WAV 16kHz {len(processed)//1024}KB "
                f"(dur={duration_sec:.1f}s)"
            )
            return processed, "audio/wav", True

    except subprocess.CalledProcessError as e:
        # afftdn may not exist on some ffmpeg builds — retry without it
        print(f"[preprocess] filter chain failed ({e}), retrying without afftdn...")
        try:
            with tempfile.TemporaryDirectory() as tmp2:
                ext = (
                    ".mp4" if "mp4" in original_mime
                    else ".ogg" if "ogg" in original_mime
                    else ".webm"
                )
                in2  = os.path.join(tmp2, f"input{ext}")
                out2 = os.path.join(tmp2, "whisper_ready.wav")
                with open(in2, "wb") as f:
                    f.write(audio_bytes)

                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", in2,
                        "-ar", "16000", "-ac", "1",
                        "-af", "highpass=f=80,lowpass=f=8000,loudnorm=I=-16:LRA=11:TP=-1.5",
                        "-c:a", "pcm_s16le",
                        out2
                    ],
                    capture_output=True, timeout=20, check=True
                )
                with open(out2, "rb") as f:
                    return f.read(), "audio/wav", True
        except Exception as e2:
            print(f"[preprocess] fallback also failed: {e2} — using raw audio")
            return audio_bytes, original_mime, False

    except Exception as e:
        print(f"[preprocess] unexpected error: {e} — using raw audio")
        return audio_bytes, original_mime, False


def compute_avg_confidence(verbose_json_response) -> float:
    """
    Pull per-segment no_speech_prob out of verbose_json.
    Returns average speech confidence (1 - no_speech_prob).
    Falls back to 1.0 if unavailable (don't penalise missing data).
    """
    try:
        segments = verbose_json_response.segments
        if not segments:
            return 1.0
        confidences = [1.0 - seg.no_speech_prob for seg in segments]
        return sum(confidences) / len(confidences)
    except Exception:
        return 1.0


# ════════════════════════════════════════════════════════════
# ENDPOINT: /transcribe  (v2 — top-notch quality)
# ════════════════════════════════════════════════════════════
@app.route('/transcribe', methods=['POST', 'OPTIONS'])
def transcribe():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        audio_name  = audio_file.filename or 'audio.webm'

        if not audio_bytes:
            return jsonify({"error": "Empty audio file"}), 400

        # Detect MIME from filename
        if 'mp4' in audio_name:
            original_mime = 'audio/mp4'
        elif 'ogg' in audio_name:
            original_mime = 'audio/ogg'
        else:
            original_mime = 'audio/webm'

        # ── Step 1: preprocess (resample, denoise, normalise) ──
        processed_bytes, final_mime, was_processed = preprocess_audio_for_whisper(
            audio_bytes, original_mime
        )
        final_name = "audio.wav" if was_processed else audio_name

        # ── Step 2: Whisper transcription ──────────────────────
        transcript_response = client.audio.transcriptions.create(
            model="whisper-large-v3",          # full v3 — better on quiet/emotional speech
            file=(final_name, io.BytesIO(processed_bytes), final_mime),
            language="en",                      # skip language detection overhead
            prompt=WHISPER_PROMPT,              # prime with therapy vocabulary
            response_format="verbose_json",     # gives us per-segment confidence
            temperature=0.0,                    # deterministic — no hallucination
        )

        raw_text = (transcript_response.text or "").strip()

        # ── Step 3: confidence check ───────────────────────────
        avg_confidence = compute_avg_confidence(transcript_response)
        print(
            f"[transcribe] avg_confidence={avg_confidence:.3f} | "
            f"text='{raw_text[:80]}'"
        )

        # Near-silent / no-speech blob
        if not raw_text:
            return jsonify({
                "success": False,
                "transcript": "",
                "confidence": 0.0,
                "error_type": "empty",
                "user_message": "Nothing was captured. Please try again.",
            }), 200  # 200 so frontend handles gracefully

        if avg_confidence < WHISPER_MIN_CONFIDENCE:
            print(f"[transcribe] LOW CONFIDENCE ({avg_confidence:.3f}) — rejecting: '{raw_text}'")
            return jsonify({
                "success": False,
                "transcript": raw_text,         # send it anyway so frontend can decide
                "confidence": avg_confidence,
                "error_type": "low_confidence",
                "user_message": "Couldn't hear that clearly — could you say that again?",
            }), 200

        # ── Step 4: light post-processing ─────────────────────
        # Whisper sometimes hallucinates filler on silence
        HALLUCINATION_PATTERNS = [
            r"^(thank you\.?|thanks\.?|you\.?|\.+|\s*)$",
            r"^(bye\.?|goodbye\.?|see you\.?)+$",
            r"^\[.*\]$",                         # [Music] [Applause] etc.
        ]
        import re as _re
        is_hallucination = any(
            _re.match(p, raw_text.lower().strip())
            for p in HALLUCINATION_PATTERNS
        )
        if is_hallucination:
            print(f"[transcribe] HALLUCINATION detected: '{raw_text}'")
            return jsonify({
                "success": False,
                "transcript": "",
                "confidence": avg_confidence,
                "error_type": "hallucination",
                "user_message": "Didn't catch that — please try again.",
            }), 200

        print(f"[transcribe] ✓ '{raw_text}' (confidence={avg_confidence:.3f})")

        return jsonify({
            "success": True,
            "transcript": raw_text,
            "confidence": round(avg_confidence, 3),
            "was_preprocessed": was_processed,
        })

    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════════════════════════════════
# ENDPOINT: /speak (single-shot TTS)
# ════════════════════════════════════════════════════════════
@app.route('/speak', methods=['POST', 'OPTIONS'])
def speak():
    if request.method == 'OPTIONS':
        res = Response()
        res.headers['Access-Control-Allow-Origin'] = '*'
        res.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        res.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        res.headers['Access-Control-Max-Age'] = '3600'
        res.status_code = 200
        return res

    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        if not GOOGLE_TTS_KEY:
            return jsonify({"error": "GOOGLE_TTS_KEY not configured"}), 500

        clean = clean_text_for_tts(text)
        cfg = get_tts_config()
        filled = add_thinking_filler(clean, cfg)
        final = emotion_aware_preprocess(filled, cfg)

        print(f"[Google TTS] Voice: {cfg['voice']} | Text: {final[:120]}")

        response = http_requests.post(
            f"{GOOGLE_TTS_URL}?key={GOOGLE_TTS_KEY}",
            json={
                "input": {"text": final},
                "voice": {"languageCode": "en-US", "name": cfg["voice"]},
                "audioConfig": build_audio_config(cfg),
            },
            timeout=30
        )

        if response.status_code != 200:
            print(f"[Google TTS] Error {response.status_code}: {response.text}")
            return jsonify({"error": f"Google TTS error {response.status_code}: {response.text}"}), 503

        audio_bytes = base64.b64decode(response.json()["audioContent"])
        padded_audio = append_silence(audio_bytes)
        print(f"[Google TTS] OK — padded to {len(padded_audio)} bytes total")

        return Response(
            padded_audio,
            mimetype="audio/mpeg",
            headers={"Content-Type": "audio/mpeg", "Access-Control-Allow-Origin": "*"}
        )

    except http_requests.exceptions.Timeout:
        return jsonify({"error": "Google TTS request timed out"}), 503
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════
# ENDPOINT: /speak-sentences
# Returns JSON array of base64 MP3s, one per sentence.
# Frontend plays sentence[0] immediately, queues the rest.
# No merging, no ffmpeg, no silence padding overhead.
# ════════════════════════════════════════════════════════════
@app.route('/speak-sentences', methods=['POST', 'OPTIONS'])
def speak_sentences():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json()
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        if not GOOGLE_TTS_KEY:
            return jsonify({"error": "GOOGLE_TTS_KEY not configured"}), 500

        # Clean + prepare text
        clean      = clean_text_for_tts(text)
        cfg        = get_tts_config()
        filled     = add_thinking_filler(clean, cfg)
        final_text = emotion_aware_preprocess(filled, cfg)
        sentences  = [s for s in split_into_sentences(final_text) if s.strip()][:8]

        if not sentences:
            return jsonify({"sentences": []}), 200

        # Synthesise ALL sentences in parallel
        chunks_by_idx = {}
        with ThreadPoolExecutor(max_workers=min(len(sentences), 8)) as pool:
            future_map = {
                pool.submit(synthesize_sentence, s, cfg): i
                for i, s in enumerate(sentences)
            }
            for future in as_completed(future_map):
                idx    = future_map[future]
                result = future.result()
                if result:
                    chunks_by_idx[idx] = base64.b64encode(result).decode("utf-8")

        # Return in order — frontend plays [0] immediately, queues rest
        ordered = [
            {"index": i, "audio": chunks_by_idx[i]}
            for i in sorted(chunks_by_idx)
            if i in chunks_by_idx
        ]

        print(f"[speak-sentences] {len(ordered)}/{len(sentences)} sentences synthesised in parallel")
        return jsonify({"sentences": ordered, "count": len(ordered)})

    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
        
# ════════════════════════════════════════════════════════════
# ENDPOINT: /speak-stream
# ════════════════════════════════════════════════════════════
@app.route('/speak-stream', methods=['POST', 'OPTIONS'])
def speak_stream():
    if request.method == 'OPTIONS':
        res = Response()
        res.headers['Access-Control-Allow-Origin'] = '*'
        res.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        res.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        res.headers['Access-Control-Max-Age'] = '3600'
        res.status_code = 200
        return res

    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        if not GOOGLE_TTS_KEY:
            return jsonify({"error": "GOOGLE_TTS_KEY not configured"}), 500

        try:
            maybe_json = json.loads(text)
            if isinstance(maybe_json, dict) and "message" in maybe_json:
                text = maybe_json["message"]
        except (json.JSONDecodeError, TypeError):
            pass

        clean = clean_sentence_for_tts(text)
        cfg = get_tts_config()
        filled = add_thinking_filler(clean, cfg)
        final = emotion_aware_preprocess(filled, cfg)
        sentences = split_into_sentences(final) or [final]

        print(f"[speak-stream] {len(sentences)} sentence(s) → voice: {cfg['voice']}")

        def generate():
            chunks = []
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                chunk = synthesize_sentence(sentence, cfg)
                if chunk:
                    print(f"[speak-stream] chunk {i+1}/{len(sentences)}: {len(chunk)} bytes")
                    chunks.append(chunk)

            if not chunks:
                return

            if len(chunks) == 1:
                combined_bytes = chunks[0]
            else:
                combined_bytes = merge_mp3s(chunks)

            final_bytes = append_silence(combined_bytes)
            print(f"[speak-stream] streaming {len(final_bytes)} bytes total (incl. 3s silence)")
            yield final_bytes

        return Response(
            stream_with_context(generate()),
            mimetype="audio/mpeg",
            headers={
                "Content-Type": "audio/mpeg",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no",
            }
        )

    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════
# ENDPOINT: /speak-test  — raw voice tester, no fillers/processing
# ════════════════════════════════════════════════════════════
@app.route('/speak-test', methods=['POST', 'OPTIONS'])
def speak_test():
    """
    Voice testing endpoint. Reads any text exactly as given — no fillers,
    no emotion processing, no digit conversion. Supports overriding the voice
    per-request so you can compare voices without touching Firebase config.

    POST body:
      {
        "text":  "Any text of any length you want to hear.",
        "voice": "en-US-Chirp3-HD-Aoede"   // optional — overrides Firebase setting
      }
    """
    if request.method == 'OPTIONS':
        res = Response()
        res.headers['Access-Control-Allow-Origin'] = '*'
        res.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        res.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        res.headers['Access-Control-Max-Age'] = '3600'
        res.status_code = 200
        return res

    try:
        data = request.get_json()
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        if not GOOGLE_TTS_KEY:
            return jsonify({"error": "GOOGLE_TTS_KEY not configured"}), 500

        # Use the live Firebase config as base, then override voice if supplied
        cfg = get_tts_config()
        override_voice = (data.get("voice") or "").strip()
        if override_voice:
            cfg["voice"] = override_voice

        # Split into sentences so long texts don't hit the 5000-char TTS limit
        sentences = [s for s in (split_into_sentences(text) or [text]) if s.strip()]
        print(f"[speak-test] {len(sentences)} sentence(s) | voice: {cfg['voice']}")

        # Fire all TTS calls in parallel — dramatically faster for long texts
        chunks_by_index = {}
        with ThreadPoolExecutor(max_workers=min(len(sentences), 8)) as pool:
            future_to_idx = {
                pool.submit(synthesize_sentence, sentence, cfg): i
                for i, sentence in enumerate(sentences)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                result = future.result()
                if result:
                    chunks_by_index[idx] = result
                    print(f"[speak-test] chunk {idx+1}/{len(sentences)}: {len(result)} bytes")

        # Reassemble in original sentence order
        chunks = [chunks_by_index[i] for i in sorted(chunks_by_index)]

        if not chunks:
            return jsonify({"error": "TTS produced no audio"}), 500

        combined = chunks[0] if len(chunks) == 1 else merge_mp3s(chunks)

        # No silence tail on the test endpoint — just the raw speech
        print(f"[speak-test] returning {len(combined)} bytes | voice: {cfg['voice']}")
        return Response(
            combined,
            mimetype="audio/mpeg",
            headers={
                "Content-Type": "audio/mpeg",
                "Access-Control-Allow-Origin": "*",
                "X-Voice-Used": cfg["voice"],
            }
        )

    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════
# ENDPOINT: /voices  — list all testable voices
# ════════════════════════════════════════════════════════════
@app.route('/voices', methods=['GET'])
def list_voices():
    cfg = get_tts_config()
    return jsonify({
        "current_voice": cfg["voice"],
        "available_voices": AVAILABLE_VOICES,
    })


# ════════════════════════════════════════════════════════════
# ENDPOINT: /voice-test-ui  — browser UI for voice testing
# ════════════════════════════════════════════════════════════
@app.route('/voice-test-ui', methods=['GET'])
def voice_test_ui():
    voices_json = json.dumps(AVAILABLE_VOICES)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Voice Tester — Theraply</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap');

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:        #0d0f14;
    --surface:   #13161e;
    --border:    #1f2430;
    --accent:    #7c6af7;
    --accent2:   #a78bfa;
    --text:      #e8e6f0;
    --muted:     #6b6880;
    --success:   #4ade80;
    --warn:      #fb923c;
  }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Mono', monospace;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 24px 80px;
  }}

  header {{
    text-align: center;
    margin-bottom: 48px;
  }}

  header h1 {{
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-style: italic;
    background: linear-gradient(135deg, var(--accent2), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
  }}

  header p {{
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 8px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }}

  .card {{
    width: 100%;
    max-width: 760px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
  }}

  label {{
    display: block;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
  }}

  select, textarea {{
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 0.88rem;
    padding: 14px 16px;
    outline: none;
    transition: border-color 0.2s;
    appearance: none;
  }}

  select:focus, textarea:focus {{
    border-color: var(--accent);
  }}

  select {{
    cursor: pointer;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' fill='none'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%236b6880' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 16px center;
    padding-right: 40px;
  }}

  textarea {{
    height: 200px;
    resize: vertical;
    line-height: 1.7;
  }}

  .row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 24px;
  }}

  .field {{ margin-bottom: 24px; }}
  .field:last-child {{ margin-bottom: 0; }}

  .char-count {{
    font-size: 0.72rem;
    color: var(--muted);
    text-align: right;
    margin-top: 6px;
  }}

  .char-count.warn {{ color: var(--warn); }}

  button {{
    width: 100%;
    padding: 16px;
    border: none;
    border-radius: 12px;
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    letter-spacing: 0.06em;
    cursor: pointer;
    transition: all 0.2s;
  }}

  #playBtn {{
    background: linear-gradient(135deg, var(--accent), #5b4fd4);
    color: #fff;
    font-weight: 500;
  }}

  #playBtn:hover:not(:disabled) {{
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(124, 106, 247, 0.4);
  }}

  #playBtn:disabled {{
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }}

  .status-bar {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 18px;
    border-radius: 10px;
    font-size: 0.8rem;
    margin-top: 20px;
    border: 1px solid var(--border);
    min-height: 50px;
    transition: all 0.3s;
  }}

  .status-bar.idle   {{ background: transparent; color: var(--muted); }}
  .status-bar.loading {{ background: rgba(124,106,247,0.08); color: var(--accent2); border-color: var(--accent); }}
  .status-bar.playing {{ background: rgba(74,222,128,0.08); color: var(--success); border-color: var(--success); }}
  .status-bar.error  {{ background: rgba(251,146,60,0.08); color: var(--warn); border-color: var(--warn); }}

  .dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
    background: currentColor;
  }}

  .dot.pulse {{ animation: pulse 1.2s ease-in-out infinite; }}

  @keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.4; transform: scale(0.7); }}
  }}

  .voice-badge {{
    display: inline-block;
    background: rgba(124,106,247,0.15);
    color: var(--accent2);
    border: 1px solid rgba(124,106,247,0.3);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    margin-top: 8px;
  }}

  .presets {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
  }}

  .preset {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 6px 12px;
    cursor: pointer;
    transition: all 0.15s;
    letter-spacing: 0.04em;
  }}

  .preset:hover {{
    border-color: var(--accent);
    color: var(--accent2);
  }}

  audio {{ width: 100%; margin-top: 16px; border-radius: 8px; }}
  audio::-webkit-media-controls-panel {{ background: var(--surface); }}
</style>
</head>
<body>

<header>
  <h1>Voice Tester</h1>
  <p>Theraply · Raw TTS Preview</p>
</header>

<div class="card">
  <div class="row">
    <div class="field">
      <label>Voice</label>
      <select id="voiceSelect"></select>
      <div class="voice-badge" id="voiceBadge">loading...</div>
    </div>
    <div class="field">
      <label>Preset Phrases</label>
      <div class="presets">
        <button class="preset" onclick="setPreset('therapy-open')">Therapy opener</button>
        <button class="preset" onclick="setPreset('therapy-reframe')">CBT reframe</button>
        <button class="preset" onclick="setPreset('long')">Long passage</button>
        <button class="preset" onclick="setPreset('fillers')">With fillers</button>
        <button class="preset" onclick="setPreset('numbers')">Numbers/digits</button>
      </div>
    </div>
  </div>

  <div class="field">
    <label>Text to speak</label>
    <textarea id="textInput" placeholder="Type or paste any text here — as long as you like. The voice will read it all."></textarea>
    <div class="char-count" id="charCount">0 characters</div>
  </div>

  <button id="playBtn" onclick="synthesise()">▶ Synthesise &amp; Play</button>

  <div class="status-bar idle" id="statusBar">
    <div class="dot"></div>
    <span id="statusText">Ready — enter text and hit play</span>
  </div>

  <audio id="player" controls style="display:none"></audio>
</div>

<script>
const VOICES = {voices_json};
const PRESETS = {{
  'therapy-open':   "So you've been carrying this for a while... I can hear it. What's the one thing that's been sitting heaviest on your chest this week, yeah?",
  'therapy-reframe': "Right, so here's what I'm noticing. That thought — that you'll definitely fail — it's loud. But loudness isn't the same as truth. What would you say to a friend who told you that exact thing?",
  'long':           "Okay so let's slow down for a second. You've shared a lot, and I want to make sure I've really heard you. The meeting on Thursday is worrying you. You're convinced your boss thinks you're not performing well. And underneath all of that, there's this fear that you're going to lose your job. That's a lot to be carrying around all week. I'm curious though — when you imagine the meeting actually happening, what's the very first thought that shows up? Not the second or the third. The very first one.",
  'fillers':        "Mmm, okay. Yeah, that makes sense. Right, so what you're describing is a really common pattern. Got it. And the anxiety tends to peak right before the event, yeah?",
  'numbers':        "We'll start with just 5 minutes. Then 10. By the third session you might aim for 20 or 30 minutes without checking your phone.",
}};

// Populate voice dropdown
const sel = document.getElementById('voiceSelect');
VOICES.forEach(v => {{
  const opt = document.createElement('option');
  opt.value = v;
  opt.textContent = v.replace('en-US-', '');
  sel.appendChild(opt);
}});

// Try to match the current Firebase voice
fetch('/voices').then(r => r.json()).then(d => {{
  const cur = d.current_voice;
  sel.value = cur;
  document.getElementById('voiceBadge').textContent = cur;
}}).catch(() => {{}});

sel.addEventListener('change', () => {{
  document.getElementById('voiceBadge').textContent = sel.value;
}});

// Char counter
const ta = document.getElementById('textInput');
const cc = document.getElementById('charCount');
ta.addEventListener('input', () => {{
  const n = ta.value.length;
  cc.textContent = n + ' character' + (n === 1 ? '' : 's');
  cc.classList.toggle('warn', n > 3000);
}});

function setPreset(key) {{
  ta.value = PRESETS[key] || '';
  ta.dispatchEvent(new Event('input'));
}}

function setStatus(state, msg) {{
  const bar = document.getElementById('statusBar');
  bar.className = 'status-bar ' + state;
  document.getElementById('statusText').textContent = msg;
  bar.querySelector('.dot').className = 'dot' + (state === 'loading' ? ' pulse' : '');
}}

async function synthesise() {{
  const text = ta.value.trim();
  if (!text) {{ setStatus('error', 'Please enter some text first.'); return; }}

  const voice = sel.value;
  const btn   = document.getElementById('playBtn');
  const player = document.getElementById('player');

  btn.disabled = true;
  player.style.display = 'none';
  setStatus('loading', 'Sending to Google TTS — ' + voice.replace('en-US-', '') + '...');

  try {{
    const res = await fetch('/speak-test', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ text, voice }})
    }});

    if (!res.ok) {{
      const err = await res.json().catch(() => ({{}}));
      throw new Error(err.error || 'HTTP ' + res.status);
    }}

    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    player.src = url;
    player.style.display = 'block';
    player.play();
    setStatus('playing', 'Playing — ' + voice.replace('en-US-', ''));

    player.onended = () => setStatus('idle', 'Done — ready for next test');
  }} catch(e) {{
    setStatus('error', 'Error: ' + e.message);
  }} finally {{
    btn.disabled = false;
  }}
}}

// Allow Ctrl+Enter to trigger play
ta.addEventListener('keydown', e => {{
  if (e.ctrlKey && e.key === 'Enter') synthesise();
}});
</script>
</body>
</html>"""
    return Response(html, mimetype="text/html")


# ════════════════════════════════════════════════════════════
# ENDPOINT: /therapy-session
# ════════════════════════════════════════════════════════════

@app.route('/therapy-session', methods=['POST', 'OPTIONS'])
def therapy_session():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data            = request.get_json()
        user_id         = data.get("user_id")
        user_message    = data.get("message", "").strip()
        session_id      = data.get("session_id")
        start_new       = data.get("start_new", False)
        response_length = data.get("response_length", "long")

        if not user_id or not user_message:
            return jsonify({"error": "user_id and message required"}), 400

        # ── Load / create session ──────────────────────────────
        if not session_id or start_new:
            session_id = f"therapy_{user_id}_{int(datetime.now().timestamp())}"
            session_data = {
                "session_id": session_id, "user_id": user_id,
                "messages": [], "phase": 1,
                "extracted": {
                    "situation": "", "anxious_thought": "", "emotion": "", "reframe": "",
                    "proposed_task": {"name": "", "type": "", "why": "", "anxiety_pre": 5, "action_steps": []}
                },
                "session_complete": False,
                "created_at": datetime.utcnow().isoformat()
            }
            db.collection("users").document(user_id) \
              .collection("therapy_sessions").document(session_id).set(session_data)
        else:
            doc = db.collection("users").document(user_id) \
                    .collection("therapy_sessions").document(session_id).get()
            if not doc.exists:
                return jsonify({"error": "Session not found. Pass start_new: true."}), 404
            session_data = doc.to_dict()

        if session_data.get("session_complete"):

            # ── Update brain ───────────────────────────────────────────
            brain_updates = {
                "last_interaction": "therapy-session",
                "last_seen": datetime.utcnow().isoformat(),
                "therapy.last_session_id": session_id,
            }
            if session_complete:
                brain = get_brain(user_id)
                brain_updates["therapy.sessions_completed"] = brain["therapy"]["sessions_completed"] + 1

            if new_extracted.get("anxious_thought"):
                brain = get_brain(user_id)
                thoughts = brain["therapy"]["recurring_thoughts"]
            if new_extracted["anxious_thought"] not in thoughts:
                thoughts.append(new_extracted["anxious_thought"])
                brain_updates["therapy.recurring_thoughts"] = thoughts[-5:]

            if new_extracted.get("reframe"):
                brain_updates["therapy.strongest_reframe"] = new_extracted["reframe"]

            threading.Thread(target=update_brain, args=(user_id, brain_updates), daemon=True).start()

            return jsonify({
                "session_id": session_id, "reply": "This session is complete.",
                "phase": 5, "session_complete": True,
                "extracted": session_data.get("extracted", {})
            })

        messages = session_data.get("messages", [])
        if len(messages) == 0:
            # Load personality file
            try:
                with open("prompt_therapy_personality.txt", "r") as f:
                    personality = f.read()
            except FileNotFoundError:
                personality = ""

            brain_context = build_brain_context(user_id)
            full_system = personality + "\n\n" + THERAPY_SYSTEM_PROMPT + "\n\n" + brain_context
            messages = [{"role": "system", "content": full_system}]
        messages.append({"role": "user", "content": user_message})

        current_phase    = session_data.get("phase", 1)
        extracted_so_far = session_data.get("extracted", {})

        length_guide = {
            "short":  "RESPONSE LENGTH: SHORT — 1 to 2 sentences max. One thought only.",
            "medium": "RESPONSE LENGTH: MEDIUM — 2 to 3 sentences. One main point plus a follow-up question.",
            "long": (
                "RESPONSE LENGTH: LONG — 5 to 7 sentences. This is what a real therapist sounds like. "
                "Validate first, then reflect back what you heard in your own words, then gently explore "
                "or reframe, then close with a question. Take your time. Let the reply breathe. "
                "Use natural pauses (...) between thoughts. Do NOT rush to the question — earn it by "
                "showing you've really heard them first. The person should feel truly seen."
            ),
        }.get(response_length, "RESPONSE LENGTH: LONG — 5 to 7 sentences.")

        length_tokens = {"short": 200, "medium": 400, "long": 1200}.get(response_length, 1200)

        phase_reminder = {
            "role": "system",
            "content": (
                f"CURRENT PHASE: {current_phase}\n"
                f"EXTRACTED SO FAR: {json.dumps(extracted_so_far)}\n"
                f"{length_guide}\n"
                "Respond with valid JSON only. Write the message for the EAR — warm, unhurried, "
                "conversational. Use short sentences and natural pauses (...)."
            )
        }
        messages_for_model = [messages[0], phase_reminder] + messages[1:]

        # ════════════════════════════════════════════════════
        # STEP 1: LLM call  +  STEP 2: TTS prep — run together
        # Strategy:
        #   • LLM call is blocking (Groq is fast, ~400-800ms)
        #   • The moment we have ai_reply we split into sentences
        #   • Fire ALL sentence TTS calls in parallel via ThreadPoolExecutor
        #   • Firestore write happens concurrently with TTS
        #   • Total latency ≈ max(LLM, longest_single_TTS_sentence)
        #     instead of LLM + sum(all_TTS_sentences)
        # ════════════════════════════════════════════════════

        # ── LLM ───────────────────────────────────────────────
        llm_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages_for_model,
            temperature=0.65,
            max_tokens=length_tokens
        )
        raw_reply = llm_response.choices[0].message.content.strip()
        parsed    = parse_json_response(raw_reply)

        # ── Extract fields ─────────────────────────────────────
        if not parsed:
            # Unparseable — save and return text only, no audio
            messages.append({"role": "assistant", "content": raw_reply})
            db.collection("users").document(user_id) \
              .collection("therapy_sessions").document(session_id) \
              .update({"messages": messages})
            return jsonify({
                "session_id": session_id, "reply": raw_reply,
                "phase": current_phase, "session_complete": False,
                "audio_b64": None,
            })

        ai_reply        = parsed.get("message", raw_reply)
        next_phase      = parsed.get("phase", current_phase)
        session_complete = parsed.get("session_complete", False)
        new_extracted   = parsed.get("extracted", {})

        # Merge extracted data
        merged = session_data.get("extracted", {})
        for key, val in new_extracted.items():
            if isinstance(val, dict):
                if key not in merged:
                    merged[key] = {}
                for subkey, subval in val.items():
                    if subval and subval != "" and subval != 0:
                        merged[key][subkey] = subval
            else:
                if val and val != "" and val != 0:
                    merged[key] = val

        messages.append({"role": "assistant", "content": ai_reply})

        # ── Tone detection ─────────────────────────────────────
        tone, pause_ms = detect_reply_tone(ai_reply)

        # ── Prepare TTS text ───────────────────────────────────
        cfg        = get_tts_config()
        clean      = clean_text_for_tts(ai_reply)
        filled     = add_thinking_filler(clean, cfg)
        final_text = emotion_aware_preprocess(filled, cfg)
        sentences  = [s for s in split_into_sentences(final_text) if s.strip()]
        # Guard: never synthesise more than 8 sentences to stay fast
        sentences  = sentences[:8]

        # ── Fire Firestore write + all TTS calls in parallel ───
        # This is the key optimisation:
        #   - Firestore write doesn't block TTS
        #   - All N sentences hit Google TTS simultaneously
        #   - Wall time = slowest single sentence (~300-500ms), not N * 300ms
        audio_b64 = None

        def _firestore_write():
            update_payload = {
                "messages": messages, "phase": next_phase,
                "extracted": merged, "session_complete": session_complete,
                "updated_at": datetime.utcnow().isoformat()
            }
            if session_complete:
                update_payload["completed_at"] = datetime.utcnow().isoformat()
            db.collection("users").document(user_id) \
              .collection("therapy_sessions").document(session_id) \
              .update(update_payload)

        def _synthesise_all(sentence_list, tts_cfg):
            if not GOOGLE_TTS_KEY or not sentence_list:
                return None
            chunks_by_idx = {}
            with ThreadPoolExecutor(max_workers=min(len(sentence_list) + 1, 8)) as pool:
                future_map = {
                    pool.submit(synthesize_sentence, s, tts_cfg): i
                    for i, s in enumerate(sentence_list)
                }
                for future in as_completed(future_map):
                    idx = future_map[future]
                    result = future.result()
                    if result:
                        chunks_by_idx[idx] = result
            ordered = [chunks_by_idx[i] for i in sorted(chunks_by_idx) if i in chunks_by_idx]
            if not ordered:
                return None
            # Return list of individual chunks — no merging, no ffmpeg
            return ordered

        # Run Firestore write and TTS concurrently
        with ThreadPoolExecutor(max_workers=2) as outer_pool:
            fs_future  = outer_pool.submit(_firestore_write)
            tts_future = outer_pool.submit(_synthesise_all, sentences, cfg)

            # Wait for both — TTS is almost always slower so Firestore is free
            try:
                fs_future.result(timeout=8)
            except Exception as fe:
                print(f"[therapy-session] Firestore write failed: {fe}")

            try:
                audio_chunks = tts_future.result(timeout=15)
                if audio_chunks:
                    audio_b64 = [base64.b64encode(c).decode("utf-8") for c in audio_chunks]
                    print(f"[therapy-session] TTS done — {len(audio_chunks)} sentences")
            except Exception as te:
                print(f"[therapy-session] TTS failed (non-fatal): {te}")
                audio_b64 = None

        return jsonify({
            "session_id":       session_id,
            "reply":            ai_reply,
            "phase":            next_phase,
            "session_complete": session_complete,
            "extracted":        merged,
            "turn_count":       len([m for m in messages if m["role"] == "user"]),
            "tone":             tone,
            "pause_ms":         pause_ms,
            "response_length":  response_length,
            "audio_b64":        audio_b64,  # base64 MP3 or null — frontend falls back to /speak
        })

    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
        

# ════════════════════════════════════════════════════════════
# ENDPOINT: /session-to-plan
# ════════════════════════════════════════════════════════════
@app.route('/session-to-plan', methods=['POST', 'OPTIONS'])
def session_to_plan():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        session_id = data.get("session_id")

        if not user_id or not session_id:
            return jsonify({"error": "user_id and session_id required"}), 400

        doc = db.collection("users").document(user_id) \
                .collection("therapy_sessions").document(session_id).get()
        if not doc.exists:
            return jsonify({"error": "Session not found"}), 404

        session_data = doc.to_dict()
        if not session_data.get("session_complete"):
            return jsonify({"error": "Session not complete.", "current_phase": session_data.get("phase", 1)}), 400

        extracted = session_data.get("extracted", {})
        if not extracted.get("proposed_task", {}).get("name"):
            return jsonify({"error": "No task extracted"}), 400

        session_summary = (
            f"SITUATION: {extracted.get('situation', '')}\n"
            f"ANXIOUS THOUGHT: {extracted.get('anxious_thought', '')}\n"
            f"EMOTION: {extracted.get('emotion', '')}\n"
            f"CBT REFRAME: {extracted.get('reframe', '')}\n"
            f"PROPOSED TASK: {json.dumps(extracted.get('proposed_task', {}))}"
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": SESSION_TO_PLAN_PROMPT.format(session_summary=session_summary)}],
            temperature=0.3, max_tokens=500
        )
        plan = parse_json_response(response.choices[0].message.content.strip())
        if not plan:
            return jsonify({"error": "Failed to parse plan"}), 500

        plan["session_id"] = session_id
        plan["created_at"] = int(datetime.now().timestamp() * 1000)
        plan["completed"] = False
        plan["source"] = "therapy_session"
        if not plan.get("scheduledDate"):
            plan["scheduledDate"] = int((datetime.now().timestamp() + 86400) * 1000)

        ref = db.collection("users").document(user_id).collection("activities").document()
        ref.set(plan)
        plan["id"] = ref.id

        db.collection("users").document(user_id) \
          .collection("therapy_sessions").document(session_id) \
          .update({"plan_id": ref.id, "plan_created_at": datetime.utcnow().isoformat()})

        return jsonify({"success": True, "plan": plan, "activity_id": ref.id})

    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════
# ENDPOINT: /therapy-session/history
# ════════════════════════════════════════════════════════════
@app.route('/therapy-session/history', methods=['POST', 'OPTIONS'])
@app.route('/api/therapy-session-history', methods=['POST', 'OPTIONS'])
def therapy_session_history():
    if request.method == 'OPTIONS':
        return '', 204

    data = request.get_json()
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    try:
        docs = db.collection("users").document(user_id) \
                 .collection("therapy_sessions") \
                 .order_by("created_at", direction=firestore.Query.DESCENDING) \
                 .limit(20).stream()

        sessions = []
        for doc in docs:
            d = doc.to_dict()
            sessions.append({
                "session_id": doc.id,
                "created_at": d.get("created_at"),
                "session_complete": d.get("session_complete", False),
                "phase": d.get("phase", 1),
                "plan_id": d.get("plan_id"),
                "situation": d.get("extracted", {}).get("situation", ""),
                "task_name": d.get("extracted", {}).get("proposed_task", {}).get("name", ""),
                "turn_count": len([m for m in d.get("messages", []) if m.get("role") == "user"])
            })

        return jsonify({"success": True, "sessions": sessions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════
# ENDPOINT: /health
# ════════════════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    cfg = get_tts_config()
    return jsonify({
        "status": "ok",
        "tts_provider": "Google Cloud TTS",
        "tts_voice": cfg["voice"],
        "tts_key_set": bool(GOOGLE_TTS_KEY),
        "chirp_hd_mode": "Chirp-HD" in cfg["voice"],
        "live_config": cfg,
        "features": [
            "sentence-chunked streaming via /speak-stream",
            f"thinking fillers — {'enabled' if cfg['filler_enabled'] else 'disabled'} @ {int(cfg['filler_probability']*100)}%",
            f"emotion-aware pacing — {'enabled' if cfg['emotion_aware'] else 'disabled'}",
            "JSON safety net in /speak and /speak-stream",
            "digit-to-word conversion",
            "Firebase-driven TTS config via config/tts_settings",
            "3-second ffmpeg-merged silence tail on all audio responses",
            "long responses by default (5-7 sentences, 1200 max_tokens)",
            "voice test UI at /voice-test-ui",
        ]
    })


# ════════════════════════════════════════════════════════════
# ENDPOINT: /tts-config
# ════════════════════════════════════════════════════════════
@app.route('/tts-config', methods=['GET', 'POST', 'OPTIONS'])
def tts_config():
    if request.method == 'OPTIONS':
        return '', 204

    if request.method == 'GET':
        cfg = get_tts_config()
        return jsonify({"success": True, "config": cfg})

    try:
        updates = request.get_json() or {}

        if updates.get("action") == "reset_defaults":
            db.collection("config").document("tts_settings").delete()
            bootstrap_tts_config()
            cfg = get_tts_config()
            return jsonify({"success": True, "action": "reset_defaults", "current_config": cfg})

        allowed = set(TTS_DEFAULTS.keys())
        filtered = {k: v for k, v in updates.items() if k in allowed}
        if not filtered:
            return jsonify({"error": f"No valid fields. Allowed: {sorted(allowed)}"}), 400

        db.collection("config").document("tts_settings").set(filtered, merge=True)
        cfg = get_tts_config()
        print(f"[tts-config] Updated: {filtered}")
        return jsonify({"success": True, "updated": filtered, "current_config": cfg})

    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/')
def index():
    return jsonify({
        "status": "Voice therapy backend running",
        "tts": "Google Cloud TTS — Chirp HD-O (plain text)",
        "endpoints": {
            "/speak": "single-shot TTS (+ 3s silence tail)",
            "/speak-stream": "sentence-chunked TTS (+ 3s silence tail)",
            "/speak-test": "raw voice tester — no fillers, voice override supported",
            "/voice-test-ui": "browser UI for voice testing",
            "/voices": "list all available voices",
            "/transcribe": "Whisper STT",
            "/therapy-session": "CBT session turn (long responses by default)",
            "/session-to-plan": "convert session to activity plan",
            "/therapy-session/history": "list past sessions",
        }
    })


"""
╔══════════════════════════════════════════════════════════════════════════╗
║  THERAPLY — BRAIN + EXERCISE ENGINE                                      ║
║  Drop-in additions to your existing app.py                               ║
║                                                                          ║
║  NEW: /brain/:user_id          — read/update persistent user memory      ║
║  NEW: /exercise/prescribe      — LLM picks & configures an exercise      ║
║  NEW: /exercise/complete       — submit results, update brain + session   ║
║  NEW: /exercise/catalog        — list all available exercise types        ║
╚══════════════════════════════════════════════════════════════════════════╝

Paste this file's contents into your app.py (after your existing helpers,
before if __name__ == '__main__').  All imports it needs are already in
your existing file.
"""

import threading


# ════════════════════════════════════════════════════════════
#  BRAIN v2 — Rich Persistent User Memory
# ════════════════════════════════════════════════════════════

BRAIN_V2_SCHEMA = {
    # ── Emotional history ────────────────────────────────────
    "emotional_baseline": {
        "avg_anxiety":       5.0,       # rolling 10-session mean
        "trend":             "unknown", # "improving" | "worsening" | "stable"
        "anxiety_history":   [],        # list of {"date": iso, "score": int, "session_id": str}
        "peak_anxiety":      0,
        "lowest_anxiety":    10,
        "good_days_streak":  0,
    },

    # ── Cognitive patterns ───────────────────────────────────
    "cognitive": {
        "recurring_distortions": [],    # ["catastrophising", "mind-reading", ...]
        "strongest_reframe":     "",    # best CBT insight produced
        "reframe_history": [],          # list of {"reframe": str, "situation": str, "date": iso}
        "avoidance_triggers":    [],    # situations user avoids
        "core_fears":            [],    # ["rejection", "failure", "abandonment", ...]
        "thought_themes":        [],    # recurring thought clusters
    },

    # ── Session stats ────────────────────────────────────────
    "therapy": {
        "sessions_completed":    0,
        "sessions_abandoned":    0,
        "avg_session_length":    0,     # turns
        "last_session_id":       "",
        "last_session_date":     "",
        "breakthrough_moments":  [],    # list of {"insight": str, "date": iso}
    },

    # ── Exercise history ─────────────────────────────────────
    "exercises": {
        "total_completed":       0,
        "types_tried":           [],    # ["breathing_4_7_8", "thought_record", ...]
        "most_effective":        "",    # exercise type with lowest post-anxiety
        "avg_anxiety_reduction": 0.0,  # mean (pre - post) across all exercises
        "recent_results": [],           # last 5 exercise outcomes
    },

    # ── Personality / preferences ────────────────────────────
    "personality": {
        "communication_style":   "unknown", # "direct" | "gentle" | "analytical"
        "prefers_metaphors":     False,
        "comfort_locations":     [],
        "best_practice_time":    "",    # "morning" | "evening" | etc.
        "support_network":       [],    # ["partner", "friend X", ...]
        "wins":                  [],    # positive events to reference
    },

    # ── Meta ─────────────────────────────────────────────────
    "last_interaction":  "never",
    "last_seen":         "",
    "profile_version":   2,
}


def get_brain_v2(user_id: str) -> dict:
    """Fetch brain from Firestore, filling missing keys from schema."""
    try:
        doc = (
            db.collection("users").document(user_id)
              .collection("brain").document("context").get()
        )
        stored = doc.to_dict() if doc.exists else {}
    except Exception as e:
        print(f"[brain_v2] read failed: {e}")
        stored = {}

    # Deep-merge: schema provides defaults, stored values win
    import copy
    merged = copy.deepcopy(BRAIN_V2_SCHEMA)
    _deep_merge(merged, stored)
    return merged


def _deep_merge(base: dict, override: dict):
    """Recursively apply override into base (in-place)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def save_brain_v2(user_id: str, brain: dict):
    try:
        db.collection("users").document(user_id) \
          .collection("brain").document("context") \
          .set(brain, merge=True)
    except Exception as e:
        print(f"[brain_v2] write failed: {e}")


def build_rich_brain_context(user_id: str) -> str:
    """
    Build a focused, actionable LLM system-prompt insert from the brain.
    Designed to personalise WITHOUT making the AI sound like it's reading
    from a file — it shapes tone and emphasis rather than reciting facts.
    """
    brain = get_brain_v2(user_id)
    eb    = brain["emotional_baseline"]
    cog   = brain["cognitive"]
    th    = brain["therapy"]
    ex    = brain["exercises"]
    pers  = brain["personality"]

    sessions     = th["sessions_completed"]
    trend        = eb["trend"]
    avg_anxiety  = eb["avg_anxiety"]
    triggers     = cog["avoidance_triggers"]
    distortions  = cog["recurring_distortions"]
    reframe      = cog["strongest_reframe"]
    core_fears   = cog["core_fears"]
    wins         = pers.get("wins", [])
    support      = pers.get("support_network", [])
    best_ex      = ex["most_effective"]
    ex_reduction = ex["avg_anxiety_reduction"]

    lines = [
        "═══ USER MEMORY — shape your response with this, never recite it ═══",
    ]

    # Emotional context
    if sessions == 0:
        lines.append("First session — no history yet. Start fresh, build trust slowly.")
    else:
        lines.append(
            f"Sessions done: {sessions}. "
            f"Anxiety trend: {trend} (avg {avg_anxiety:.1f}/10). "
            + (f"Improving — acknowledge progress subtly." if trend == "improving" else
               f"Worsening — increase warmth and scaffolding." if trend == "worsening" else
               "Stable — gently probe what's blocking movement.")
        )

    # Cognitive patterns
    if distortions:
        lines.append(
            f"Recurring thought patterns: {', '.join(distortions[:3])}. "
            "You've seen these before — name them gently when they appear."
        )
    if reframe:
        lines.append(
            f"A reframe that landed before: \"{reframe[:120]}\". "
            "You can build on this if it fits."
        )
    if core_fears:
        lines.append(
            f"Core fears: {', '.join(core_fears[:3])}. "
            "Don't name these directly unless the user opens that door."
        )

    # Practical
    if triggers:
        lines.append(f"Known avoidance triggers: {', '.join(triggers[:4])}.")
    if wins:
        lines.append(
            f"Recent wins to reference warmly if relevant: {'; '.join(str(w) for w in wins[:2])}."
        )
    if support:
        lines.append(f"Support network: {', '.join(str(s) for s in support[:3])}.")

    # Exercise history
    if ex["total_completed"] > 0 and best_ex:
        lines.append(
            f"Exercises completed: {ex['total_completed']}. "
            f"Most effective: {best_ex} (avg anxiety drop: {ex_reduction:.1f} pts). "
            "Recommend this type first when relevant."
        )
    elif ex["total_completed"] > 0:
        lines.append(f"Exercises completed: {ex['total_completed']}.")

    # Communication style
    style = pers.get("communication_style", "unknown")
    if style == "direct":
        lines.append("Prefers direct language. Skip long preamble — get to the point.")
    elif style == "analytical":
        lines.append("Analytical — responds well to structured reframes and named cognitive patterns.")
    elif style == "gentle":
        lines.append("Needs gentle pacing — go slowly, validate heavily before challenging.")

    lines.append("═══════════════════════════════════════════════════════════")
    return "\n".join(lines)


def update_brain_after_session(user_id: str, session_data: dict):
    """
    Called after a therapy session completes.
    Updates: anxiety history, distortions, reframe, triggers, streak.
    Pure function logic — Firestore write happens in a background thread.
    """
    brain    = get_brain_v2(user_id)
    extracted = session_data.get("extracted", {})
    messages  = session_data.get("messages", [])

    # ── Anxiety score ─────────────────────────────────────────
    pre_anxiety = (
        extracted.get("proposed_task", {}).get("anxiety_pre") or
        session_data.get("anxiety_pre") or 5
    )
    history = brain["emotional_baseline"]["anxiety_history"]
    history.append({
        "date":       datetime.utcnow().isoformat(),
        "score":      pre_anxiety,
        "session_id": session_data.get("session_id", ""),
    })
    history = history[-20:]  # keep last 20 sessions

    scores = [h["score"] for h in history]
    avg = sum(scores) / len(scores)
    if len(scores) >= 3:
        recent3 = scores[-3:]
        if recent3[-1] < recent3[0] - 0.5:
            trend = "improving"
        elif recent3[-1] > recent3[0] + 0.5:
            trend = "worsening"
        else:
            trend = "stable"
    else:
        trend = "unknown"

    brain["emotional_baseline"]["anxiety_history"] = history
    brain["emotional_baseline"]["avg_anxiety"]     = round(avg, 2)
    brain["emotional_baseline"]["trend"]           = trend
    brain["emotional_baseline"]["peak_anxiety"]    = max(brain["emotional_baseline"]["peak_anxiety"], pre_anxiety)
    brain["emotional_baseline"]["lowest_anxiety"]  = min(brain["emotional_baseline"]["lowest_anxiety"], pre_anxiety)

    # ── Reframe history ───────────────────────────────────────
    reframe = extracted.get("reframe", "")
    if reframe:
        brain["cognitive"]["strongest_reframe"] = reframe
        rh = brain["cognitive"]["reframe_history"]
        rh.append({
            "reframe":   reframe,
            "situation": extracted.get("situation", ""),
            "date":      datetime.utcnow().isoformat(),
        })
        brain["cognitive"]["reframe_history"] = rh[-10:]

    # ── Avoidance triggers ────────────────────────────────────
    situation = extracted.get("situation", "")
    if situation and situation not in brain["cognitive"]["avoidance_triggers"]:
        brain["cognitive"]["avoidance_triggers"].append(situation)
        brain["cognitive"]["avoidance_triggers"] = brain["cognitive"]["avoidance_triggers"][-10:]

    # ── Session stats ─────────────────────────────────────────
    brain["therapy"]["sessions_completed"] += 1
    brain["therapy"]["last_session_id"]    = session_data.get("session_id", "")
    brain["therapy"]["last_session_date"]  = datetime.utcnow().isoformat()

    turn_count = len([m for m in messages if m.get("role") == "user"])
    prev_avg   = brain["therapy"]["avg_session_length"]
    n          = brain["therapy"]["sessions_completed"]
    brain["therapy"]["avg_session_length"] = round(
        (prev_avg * (n - 1) + turn_count) / n, 1
    )

    # ── Meta ──────────────────────────────────────────────────
    brain["last_interaction"] = "therapy-session"
    brain["last_seen"]        = datetime.utcnow().isoformat()

    threading.Thread(
        target=save_brain_v2, args=(user_id, brain), daemon=True
    ).start()


def update_brain_after_exercise(user_id: str, result: dict):
    """
    Called when an exercise is completed.
    Updates: exercise history, most_effective, avg_anxiety_reduction.
    """
    brain   = get_brain_v2(user_id)
    ex      = brain["exercises"]
    ex_type = result.get("exercise_type", "unknown")
    pre     = float(result.get("anxiety_pre",  5))
    post    = float(result.get("anxiety_post", 5))
    reduction = pre - post

    ex["total_completed"] += 1

    if ex_type not in ex["types_tried"]:
        ex["types_tried"].append(ex_type)

    recent = ex.get("recent_results", [])
    recent.append({
        "type":      ex_type,
        "pre":       pre,
        "post":      post,
        "reduction": round(reduction, 1),
        "date":      datetime.utcnow().isoformat(),
    })
    ex["recent_results"] = recent[-5:]

    # Recompute avg reduction
    all_results = ex["recent_results"]
    if all_results:
        ex["avg_anxiety_reduction"] = round(
            sum(r["reduction"] for r in all_results) / len(all_results), 2
        )

    # Find most effective type
    by_type: dict = {}
    for r in all_results:
        t = r["type"]
        by_type.setdefault(t, []).append(r["reduction"])
    if by_type:
        ex["most_effective"] = max(by_type, key=lambda t: sum(by_type[t]) / len(by_type[t]))

    brain["exercises"]         = ex
    brain["last_interaction"]  = "exercise"
    brain["last_seen"]         = datetime.utcnow().isoformat()

    threading.Thread(
        target=save_brain_v2, args=(user_id, brain), daemon=True
    ).start()


# ════════════════════════════════════════════════════════════
#  EXERCISE CATALOG
# ════════════════════════════════════════════════════════════

EXERCISE_CATALOG = {

    # ── Breathing ────────────────────────────────────────────
    "breathing_box": {
        "name":        "Box breathing",
        "description": "4-4-4-4 rhythmic breathing to calm the nervous system",
        "best_for":    ["acute anxiety", "panic", "pre-event nerves"],
        "duration_s":  180,
        "config": {
            "phases": [
                {"label": "Inhale",  "duration": 4, "color": "#7c6af7"},
                {"label": "Hold",    "duration": 4, "color": "#a78bfa"},
                {"label": "Exhale",  "duration": 4, "color": "#4ade80"},
                {"label": "Hold",    "duration": 4, "color": "#86efac"},
            ],
            "cycles":  4,
            "message": "Let your shoulders drop. Each breath is a reset.",
        },
        "inputs": ["anxiety_pre", "anxiety_post", "notes"],
    },

    "breathing_4_7_8": {
        "name":        "4-7-8 breathing",
        "description": "Activates the parasympathetic nervous system",
        "best_for":    ["insomnia", "high anxiety", "anger"],
        "duration_s":  240,
        "config": {
            "phases": [
                {"label": "Inhale",  "duration": 4,  "color": "#7c6af7"},
                {"label": "Hold",    "duration": 7,  "color": "#a78bfa"},
                {"label": "Exhale",  "duration": 8,  "color": "#4ade80"},
            ],
            "cycles":  4,
            "message": "Longer exhales tell your brain it's safe to relax.",
        },
        "inputs": ["anxiety_pre", "anxiety_post", "notes"],
    },

    # ── Grounding ────────────────────────────────────────────
    "grounding_5_4_3_2_1": {
        "name":        "5-4-3-2-1 grounding",
        "description": "Sensory anchoring to interrupt anxiety spirals",
        "best_for":    ["dissociation", "panic", "intrusive thoughts"],
        "duration_s":  300,
        "config": {
            "steps": [
                {"count": 5, "sense": "see",   "icon": "👁",  "prompt": "Name 5 things you can see right now"},
                {"count": 4, "sense": "touch",  "icon": "✋", "prompt": "Name 4 things you can physically feel"},
                {"count": 3, "sense": "hear",   "icon": "👂", "prompt": "Name 3 things you can hear"},
                {"count": 2, "sense": "smell",  "icon": "👃", "prompt": "Name 2 things you can smell"},
                {"count": 1, "sense": "taste",  "icon": "👅", "prompt": "Name 1 thing you can taste"},
            ],
            "message": "Each sense brings you back to the present moment.",
        },
        "inputs": ["anxiety_pre", "anxiety_post", "responses", "notes"],
    },

    # ── Cognitive ────────────────────────────────────────────
    "thought_record": {
        "name":        "Thought record",
        "description": "CBT worksheet: identify, challenge, and reframe a thought",
        "best_for":    ["rumination", "negative self-talk", "cognitive distortions"],
        "duration_s":  600,
        "config": {
            "fields": [
                {
                    "id":          "situation",
                    "label":       "What just happened?",
                    "placeholder": "Describe the situation in 1–2 sentences",
                    "type":        "textarea",
                    "rows":        2,
                },
                {
                    "id":          "hot_thought",
                    "label":       "What's the hottest thought in your head?",
                    "placeholder": "The thought that's hitting hardest right now",
                    "type":        "textarea",
                    "rows":        2,
                },
                {
                    "id":          "belief_before",
                    "label":       "How much do you believe that thought? (0–100%)",
                    "type":        "slider",
                    "min":         0,
                    "max":         100,
                    "step":        5,
                    "unit":        "%",
                },
                {
                    "id":          "emotion",
                    "label":       "What emotion does it bring up?",
                    "placeholder": "e.g. shame, fear, anger, sadness",
                    "type":        "text",
                },
                {
                    "id":          "emotion_intensity",
                    "label":       "Intensity of that emotion (0–100)",
                    "type":        "slider",
                    "min":         0,
                    "max":         100,
                    "step":        5,
                    "unit":        "%",
                },
                {
                    "id":          "evidence_for",
                    "label":       "Evidence that supports this thought",
                    "placeholder": "Be honest — what facts back it up?",
                    "type":        "textarea",
                    "rows":        2,
                },
                {
                    "id":          "evidence_against",
                    "label":       "Evidence that contradicts this thought",
                    "placeholder": "What facts challenge it?",
                    "type":        "textarea",
                    "rows":        2,
                },
                {
                    "id":          "friend_advice",
                    "label":       "What would you tell a close friend who had this thought?",
                    "placeholder": "Imagine they came to you with this exact thought...",
                    "type":        "textarea",
                    "rows":        2,
                },
                {
                    "id":          "balanced_thought",
                    "label":       "A more balanced version of the thought",
                    "placeholder": "Not toxic positivity — just more accurate",
                    "type":        "textarea",
                    "rows":        2,
                },
                {
                    "id":          "belief_after",
                    "label":       "How much do you believe the original thought now? (0–100%)",
                    "type":        "slider",
                    "min":         0,
                    "max":         100,
                    "step":        5,
                    "unit":        "%",
                },
            ],
            "message": "Thoughts are hypotheses, not facts. Let's examine the evidence.",
        },
        "inputs": ["anxiety_pre", "anxiety_post", "responses"],
    },

    # ── Body scan ────────────────────────────────────────────
    "body_scan": {
        "name":        "Body scan",
        "description": "Progressive attention through the body to locate and release tension",
        "best_for":    ["physical tension", "stress", "mind-body disconnect"],
        "duration_s":  480,
        "config": {
            "zones": [
                {"id": "head",      "label": "Head & face",    "prompt": "Notice any tension in your jaw, forehead, or eyes"},
                {"id": "neck",      "label": "Neck & shoulders","prompt": "Let your shoulders drop. Any tightness here?"},
                {"id": "chest",     "label": "Chest",           "prompt": "Notice your breathing. Is it shallow or deep?"},
                {"id": "abdomen",   "label": "Abdomen",         "prompt": "Any knots or tension in your stomach?"},
                {"id": "arms",      "label": "Arms & hands",    "prompt": "Are your hands clenched? Let them relax."},
                {"id": "legs",      "label": "Legs & feet",     "prompt": "Notice any tension. Let your legs feel heavy."},
            ],
            "tension_scale": {"min": 0, "max": 10, "label": "Tension level"},
            "message":       "You can't think your way out of a body response. Start here.",
        },
        "inputs": ["anxiety_pre", "anxiety_post", "zone_ratings", "notes"],
    },

    # ── Behavioural ──────────────────────────────────────────
    "fear_ladder": {
        "name":        "Fear ladder",
        "description": "Build a hierarchy of exposures from least to most feared",
        "best_for":    ["avoidance", "social anxiety", "phobias"],
        "duration_s":  900,
        "config": {
            "steps":          10,
            "fields_per_step": [
                {"id": "situation", "label": "Situation", "type": "text",   "placeholder": "What would you do?"},
                {"id": "anxiety",   "label": "SUDS",       "type": "slider", "min": 0, "max": 100, "unit": "/100"},
            ],
            "message": "We face fears in small steps — not in one leap.",
        },
        "inputs": ["responses", "notes"],
    },

    "values_compass": {
        "name":        "Values compass",
        "description": "Identify what matters most and how current behaviour aligns",
        "best_for":    ["low motivation", "life direction", "depression"],
        "duration_s":  600,
        "config": {
            "domains": [
                "Family & relationships", "Work & career",
                "Health & body", "Personal growth",
                "Leisure & fun", "Spirituality / meaning",
                "Community & friendships", "Creativity",
            ],
            "fields_per_domain": [
                {"id": "importance",  "label": "How important is this to you?", "type": "slider", "min": 0, "max": 10},
                {"id": "living_it",   "label": "How much are you living by it?", "type": "slider", "min": 0, "max": 10},
            ],
            "message": "The gap between what matters and what we do is where anxiety lives.",
        },
        "inputs": ["responses", "notes"],
    },

    # ── Visualisation ────────────────────────────────────────
    "safe_place": {
        "name":        "Safe place visualisation",
        "description": "Guided imagery to build an internal anchor of calm",
        "best_for":    ["high distress", "trauma response", "bedtime anxiety"],
        "duration_s":  360,
        "config": {
            "prompts": [
                "Close your eyes. Picture a place where you feel completely safe — real or imagined.",
                "What do you see around you? Notice the colours, shapes, and light.",
                "What can you hear? Wind, water, silence, music — anything.",
                "What does the air feel like on your skin? Temperature, texture.",
                "Notice how your body feels in this place. Let it soften.",
                "Give this place a name you'll remember.",
            ],
            "audio_guided": True,
            "message": "This place belongs to you. You can return here anytime.",
        },
        "inputs": ["anxiety_pre", "anxiety_post", "place_name", "description", "notes"],
    },
}


EXERCISE_PRESCRIBE_PROMPT = """You are a CBT therapist. Based on this therapy session context, 
prescribe the SINGLE most appropriate exercise from the catalog.

SESSION CONTEXT:
{context}

AVAILABLE EXERCISE TYPES:
{catalog_summary}

RULES:
- Choose ONE exercise type only
- Consider the user's anxiety level, current phase, and situation
- If anxiety > 7, prioritise breathing or grounding
- If in cognitive phase, prioritise thought_record
- If avoidance is the theme, consider fear_ladder
- Return VALID JSON only:

{{
  "exercise_type": "<type_key>",
  "rationale": "<one sentence why this fits>",
  "custom_intro": "<2-3 sentences the AI says to introduce the exercise, warm and spoken>",
  "anxiety_pre_estimate": <estimated anxiety 1-10>
}}"""


# ════════════════════════════════════════════════════════════
#  ENDPOINT: GET /brain/:user_id
# ════════════════════════════════════════════════════════════

@app.route('/brain/<user_id>', methods=['GET', 'OPTIONS'])
def get_brain_endpoint(user_id):
    if request.method == 'OPTIONS':
        return '', 204
    brain = get_brain_v2(user_id)
    return jsonify({"success": True, "brain": brain})


@app.route('/brain/<user_id>', methods=['POST', 'OPTIONS'])
def update_brain_endpoint(user_id):
    """
    Manually update specific brain fields.
    Body: {"path.to.field": value, ...}
    E.g. {"personality.communication_style": "direct"}
    """
    if request.method == 'OPTIONS':
        return '', 204
    data  = request.get_json() or {}
    brain = get_brain_v2(user_id)

    for dotpath, value in data.items():
        parts = dotpath.split(".")
        target = brain
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value

    save_brain_v2(user_id, brain)
    return jsonify({"success": True, "brain": brain})


# ════════════════════════════════════════════════════════════
#  ENDPOINT: /exercise/catalog
# ════════════════════════════════════════════════════════════

@app.route('/exercise/catalog', methods=['GET'])
def exercise_catalog():
    summary = {
        k: {
            "name":        v["name"],
            "description": v["description"],
            "best_for":    v["best_for"],
            "duration_s":  v["duration_s"],
        }
        for k, v in EXERCISE_CATALOG.items()
    }
    return jsonify({"success": True, "exercises": summary})


# ════════════════════════════════════════════════════════════
#  ENDPOINT: POST /exercise/prescribe
# ════════════════════════════════════════════════════════════

@app.route('/exercise/prescribe', methods=['POST', 'OPTIONS'])
def exercise_prescribe():
    """
    Given a user_id + optional session context, pick the right exercise
    and return full config + intro speech + optional TTS audio.

    Body:
    {
      "user_id":    "...",
      "session_id": "...",        // optional — loads session context
      "situation":  "...",        // optional manual override
      "anxiety":    7,            // optional manual anxiety score
      "exercise_type": "..."      // optional — skip LLM, use this type directly
    }
    """
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data          = request.get_json()
        user_id       = data.get("user_id")
        session_id    = data.get("session_id")
        manual_type   = data.get("exercise_type")
        manual_anx    = data.get("anxiety", 5)
        manual_sit    = data.get("situation", "")

        if not user_id:
            return jsonify({"error": "user_id required"}), 400

        # ── Load session context if provided ──────────────────
        session_ctx = ""
        session_data = {}
        if session_id:
            try:
                doc = db.collection("users").document(user_id) \
                        .collection("therapy_sessions").document(session_id).get()
                if doc.exists:
                    session_data = doc.to_dict()
                    ext = session_data.get("extracted", {})
                    session_ctx = (
                        f"Situation: {ext.get('situation', manual_sit)}\n"
                        f"Anxious thought: {ext.get('anxious_thought', '')}\n"
                        f"Emotion: {ext.get('emotion', '')}\n"
                        f"Phase: {session_data.get('phase', 1)}\n"
                        f"Anxiety estimate: {ext.get('proposed_task', {}).get('anxiety_pre', manual_anx)}"
                    )
            except Exception as e:
                print(f"[exercise_prescribe] session load failed: {e}")

        if not session_ctx:
            session_ctx = (
                f"Situation: {manual_sit}\nAnxiety: {manual_anx}/10\n"
                "No active session context."
            )

        # ── Brain context ──────────────────────────────────────
        brain_ctx = build_rich_brain_context(user_id)

        # ── Determine exercise type ────────────────────────────
        if manual_type and manual_type in EXERCISE_CATALOG:
            ex_type   = manual_type
            rationale = "Manually selected."
            intro     = f"I'd like us to try a {EXERCISE_CATALOG[ex_type]['name']} exercise."
            anx_pre   = manual_anx
        else:
            # Let LLM decide
            catalog_summary = "\n".join(
                f"- {k}: {v['name']} — {v['description']} (best for: {', '.join(v['best_for'])})"
                for k, v in EXERCISE_CATALOG.items()
            )
            prompt = EXERCISE_PRESCRIBE_PROMPT.format(
                context=f"{session_ctx}\n\n{brain_ctx}",
                catalog_summary=catalog_summary,
            )
            llm_resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=300,
            )
            parsed = parse_json_response(llm_resp.choices[0].message.content.strip())
            if not parsed or parsed.get("exercise_type") not in EXERCISE_CATALOG:
                # Fallback
                ex_type   = "breathing_box"
                rationale = "Default fallback."
                intro     = "Let's start with some breathing to calm your nervous system."
                anx_pre   = manual_anx
            else:
                ex_type   = parsed["exercise_type"]
                rationale = parsed.get("rationale", "")
                intro     = parsed.get("custom_intro", "")
                anx_pre   = parsed.get("anxiety_pre_estimate", manual_anx)

        exercise_def = EXERCISE_CATALOG[ex_type]

        # ── Build TTS audio for intro ──────────────────────────
        audio_b64 = None
        if GOOGLE_TTS_KEY and intro:
            try:
                cfg   = get_tts_config()
                clean = clean_text_for_tts(intro)
                audio = synthesize_sentence(clean, cfg)
                if audio:
                    audio_b64 = base64.b64encode(audio).decode("utf-8")
            except Exception as e:
                print(f"[exercise_prescribe] TTS failed (non-fatal): {e}")

        return jsonify({
            "success":         True,
            "exercise_type":   ex_type,
            "exercise":        exercise_def,
            "rationale":       rationale,
            "intro_speech":    intro,
            "anxiety_pre":     anx_pre,
            "audio_b64":       audio_b64,
            "session_id":      session_id,
            "user_id":         user_id,
        })

    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════
#  ENDPOINT: POST /exercise/complete
# ════════════════════════════════════════════════════════════

@app.route('/exercise/complete', methods=['POST', 'OPTIONS'])
def exercise_complete():
    """
    Submit exercise results. Updates brain + session + optionally
    continues the therapy session with context from the exercise.

    Body:
    {
      "user_id":       "...",
      "session_id":    "...",       // optional — to continue therapy session
      "exercise_type": "breathing_box",
      "anxiety_pre":   7,
      "anxiety_post":  4,
      "responses":     {...},       // exercise-specific responses
      "notes":         "...",       // free-text from user
      "duration_s":    180,         // how long they actually spent
      "continue_session": true      // if true, generates a therapist reply
    }
    """
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data            = request.get_json()
        user_id         = data.get("user_id")
        session_id      = data.get("session_id")
        ex_type         = data.get("exercise_type", "unknown")
        anxiety_pre     = float(data.get("anxiety_pre",  5))
        anxiety_post    = float(data.get("anxiety_post", anxiety_pre))
        responses       = data.get("responses",  {})
        notes           = data.get("notes", "")
        duration_s      = data.get("duration_s", 0)
        continue_sess   = data.get("continue_session", False)

        if not user_id:
            return jsonify({"error": "user_id required"}), 400

        reduction = anxiety_pre - anxiety_post

        # ── Save exercise result to Firestore ──────────────────
        result_doc = {
            "user_id":       user_id,
            "session_id":    session_id or "",
            "exercise_type": ex_type,
            "exercise_name": EXERCISE_CATALOG.get(ex_type, {}).get("name", ex_type),
            "anxiety_pre":   anxiety_pre,
            "anxiety_post":  anxiety_post,
            "reduction":     round(reduction, 1),
            "responses":     responses,
            "notes":         notes,
            "duration_s":    duration_s,
            "completed_at":  datetime.utcnow().isoformat(),
        }
        ref = db.collection("users").document(user_id) \
                .collection("exercise_results").document()
        ref.set(result_doc)
        result_id = ref.id

        # ── Update brain (background) ─────────────────────────
        threading.Thread(
            target=update_brain_after_exercise,
            args=(user_id, {**result_doc, "exercise_type": ex_type}),
            daemon=True,
        ).start()

        # ── Optional: continue therapy session ────────────────
        therapist_reply = None
        audio_b64       = None

        if continue_sess and session_id:
            try:
                doc = db.collection("users").document(user_id) \
                        .collection("therapy_sessions").document(session_id).get()
                if doc.exists:
                    session_data = doc.to_dict()
                    messages     = session_data.get("messages", [])

                    exercise_summary = _summarise_exercise_for_llm(ex_type, result_doc)
                    followup_msg = (
                        f"[Exercise completed: {EXERCISE_CATALOG.get(ex_type, {}).get('name', ex_type)}]\n"
                        f"{exercise_summary}"
                    )
                    messages.append({"role": "user", "content": followup_msg})

                    follow_prompt = {
                        "role": "system",
                        "content": (
                            "The user just completed an exercise. Respond warmly. "
                            "Acknowledge their anxiety shift specifically. "
                            "Gently connect what they noticed to the session's theme. "
                            "RESPONSE LENGTH: MEDIUM — 3 to 4 sentences. "
                            "Return valid JSON: {\"message\": \"...\", \"phase\": N, "
                            "\"session_complete\": false, \"extracted\": {}}"
                        ),
                    }
                    messages_for_model = [messages[0], follow_prompt] + messages[1:]

                    llm_resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages_for_model,
                        temperature=0.65,
                        max_tokens=500,
                    )
                    raw   = llm_resp.choices[0].message.content.strip()
                    parsed = parse_json_response(raw)
                    therapist_reply = parsed.get("message", raw) if parsed else raw
                    messages.append({"role": "assistant", "content": therapist_reply})

                    db.collection("users").document(user_id) \
                      .collection("therapy_sessions").document(session_id) \
                      .update({"messages": messages})

                    # TTS
                    if GOOGLE_TTS_KEY and therapist_reply:
                        cfg   = get_tts_config()
                        clean = clean_text_for_tts(therapist_reply)
                        sents = [s for s in split_into_sentences(clean) if s.strip()][:5]
                        chunks = []
                        with ThreadPoolExecutor(max_workers=min(len(sents), 5)) as pool:
                            futures = {pool.submit(synthesize_sentence, s, cfg): i for i, s in enumerate(sents)}
                            by_idx  = {}
                            for fut in as_completed(futures):
                                idx = futures[fut]
                                r   = fut.result()
                                if r:
                                    by_idx[idx] = r
                        chunks = [by_idx[i] for i in sorted(by_idx) if i in by_idx]
                        if chunks:
                            audio_b64 = [base64.b64encode(c).decode("utf-8") for c in chunks]

            except Exception as e:
                print(f"[exercise_complete] session follow-up failed (non-fatal): {e}")

        return jsonify({
            "success":          True,
            "result_id":        result_id,
            "exercise_type":    ex_type,
            "anxiety_pre":      anxiety_pre,
            "anxiety_post":     anxiety_post,
            "reduction":        round(reduction, 1),
            "therapist_reply":  therapist_reply,
            "audio_b64":        audio_b64,
        })

    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


def _summarise_exercise_for_llm(ex_type: str, result: dict) -> str:
    """Convert exercise results into a natural language summary for the LLM."""
    pre    = result.get("anxiety_pre", 5)
    post   = result.get("anxiety_post", 5)
    change = pre - post
    notes  = result.get("notes", "")
    resp   = result.get("responses", {})

    direction = (
        f"anxiety dropped from {pre} to {post} (down {change:.0f} points)" if change > 0 else
        f"anxiety unchanged at {post}" if change == 0 else
        f"anxiety increased from {pre} to {post}"
    )

    parts = [f"The user completed {ex_type}. Their {direction}."]

    if ex_type == "thought_record":
        hot_thought    = resp.get("hot_thought", "")
        balanced       = resp.get("balanced_thought", "")
        belief_before  = resp.get("belief_before", "")
        belief_after   = resp.get("belief_after", "")
        if hot_thought:
            parts.append(f"The hot thought was: \"{hot_thought}\".")
        if balanced:
            parts.append(f"They reframed it to: \"{balanced}\".")
        if belief_before and belief_after:
            parts.append(f"Belief in original thought: {belief_before}% → {belief_after}%.")

    elif ex_type in ("grounding_5_4_3_2_1",):
        if resp:
            samples = list(resp.items())[:3]
            parts.append(
                "Grounding responses: " +
                "; ".join(f"{k}: {v}" for k, v in samples)
            )

    elif ex_type == "body_scan":
        zone_ratings = result.get("zone_ratings", {})
        if zone_ratings:
            high_tension = [z for z, r in zone_ratings.items() if isinstance(r, (int, float)) and r >= 7]
            if high_tension:
                parts.append(f"High tension zones: {', '.join(high_tension)}.")

    elif ex_type == "fear_ladder":
        if resp:
            top3 = list(resp.items())[:3]
            parts.append(
                "Fear ladder steps: " +
                "; ".join(f"{v.get('situation','?')} (SUDS {v.get('anxiety','?')})" for _, v in top3 if isinstance(v, dict))
            )

    elif ex_type == "values_compass":
        if resp:
            gaps = {
                domain: (r.get("importance", 5) - r.get("living_it", 5))
                for domain, r in resp.items()
                if isinstance(r, dict)
            }
            big_gaps = [d for d, g in sorted(gaps.items(), key=lambda x: -x[1])[:2] if g > 2]
            if big_gaps:
                parts.append(f"Biggest values gaps: {', '.join(big_gaps)}.")

    elif ex_type == "safe_place":
        place_name = result.get("place_name", "")
        description = result.get("description", "")
        if place_name:
            parts.append(f"Their safe place is: \"{place_name}\".")
        if description:
            parts.append(f"Description: {description[:100]}.")

    if notes:
        parts.append(f"User note: \"{notes[:200]}\".")

    return " ".join(parts)


# ════════════════════════════════════════════════════════════
#  ENDPOINT: GET /exercise/history/:user_id
# ════════════════════════════════════════════════════════════

@app.route('/exercise/history/<user_id>', methods=['GET', 'OPTIONS'])
def exercise_history(user_id):
    if request.method == 'OPTIONS':
        return '', 204
    try:
        docs = (
            db.collection("users").document(user_id)
              .collection("exercise_results")
              .order_by("completed_at", direction=firestore.Query.DESCENDING)
              .limit(30).stream()
        )
        results = []
        for doc in docs:
            d = doc.to_dict()
            results.append({
                "id":            doc.id,
                "exercise_type": d.get("exercise_type"),
                "exercise_name": d.get("exercise_name"),
                "anxiety_pre":   d.get("anxiety_pre"),
                "anxiety_post":  d.get("anxiety_post"),
                "reduction":     d.get("reduction"),
                "completed_at":  d.get("completed_at"),
                "session_id":    d.get("session_id"),
            })
        return jsonify({"success": True, "results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════
#  PATCH: /therapy-session — upgrade to use brain v2
#
#  Replace build_brain_context(user_id) calls in your existing
#  /therapy-session endpoint with build_rich_brain_context(user_id).
#
#  Also add this to your session_complete block inside /therapy-session:
#
#    if session_complete:
#        threading.Thread(
#            target=update_brain_after_session,
#            args=(user_id, session_data_for_brain),
#            daemon=True
#        ).start()
#
#  Where session_data_for_brain = {
#    "session_id":   session_id,
#    "extracted":    merged,          # the merged extracted dict
#    "messages":     messages,
#    "anxiety_pre":  merged.get("proposed_task", {}).get("anxiety_pre", 5),
#  }
# ════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════
#  FRONTEND CONTRACT  (what the React/RN app receives)
# ════════════════════════════════════════════════════════════
#
#  POST /exercise/prescribe  →  {
#    exercise_type:  "thought_record",
#    exercise: {
#      name: "Thought record",
#      config: { fields: [...] },        ← render this
#      inputs: ["anxiety_pre", ...],
#    },
#    intro_speech: "Let's try a...",
#    audio_b64:    "<base64 MP3>",       ← play immediately
#    anxiety_pre:  6,
#  }
#
#  Frontend renders config.fields dynamically.
#  On submit → POST /exercise/complete with filled responses.
#
#  Exercise UI rendering rules by type:
#
#  "breathing_*"           → AnimatedBreathingCircle (expand/hold/contract)
#  "grounding_5_4_3_2_1"  → StepByStepForm (one sense at a time)
#  "thought_record"        → FormWizard (config.fields, one per screen)
#  "body_scan"             → SilhouetteMap (tap body zones, rate tension)
#  "fear_ladder"           → DraggableList (reorder steps by SUDS)
#  "values_compass"        → RadarChart (two overlapping fills)
#  "safe_place"            → GuidedAudioForm (prompts + free text)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
