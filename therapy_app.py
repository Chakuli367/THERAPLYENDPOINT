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
            return jsonify({
                "session_id": session_id, "reply": "This session is complete.",
                "phase": 5, "session_complete": True,
                "extracted": session_data.get("extracted", {})
            })

        messages = session_data.get("messages", [])
        if len(messages) == 0:
            messages = [{"role": "system", "content": THERAPY_SYSTEM_PROMPT}]
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


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
