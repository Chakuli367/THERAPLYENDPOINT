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
    """
    Append silence to an MP3 by writing it via ffmpeg concat so the output
    is a single valid MP3 file (no header/frame corruption).
    Falls back to returning the original bytes unchanged if ffmpeg is absent.
    """
    if not _HAS_FFMPEG:
        print("[silence] ffmpeg not found — skipping silence padding")
        return audio_bytes

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            speech_path  = os.path.join(tmpdir, "speech.mp3")
            silence_path = os.path.join(tmpdir, "silence.mp3")
            output_path  = os.path.join(tmpdir, "output.mp3")
            list_path    = os.path.join(tmpdir, "concat.txt")

            # Write the speech MP3
            with open(speech_path, "wb") as f:
                f.write(audio_bytes)

            # Generate silence MP3 with ffmpeg
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"anullsrc=r=24000:cl=mono",
                "-t", str(silence_ms / 1000),
                "-c:a", "libmp3lame", "-b:a", "128k", "-q:a", "4",
                silence_path
            ], capture_output=True, timeout=10, check=True)

            # Concat list
            with open(list_path, "w") as f:
                f.write(f"file '{speech_path}'\nfile '{silence_path}'\n")

            # Merge both into one MP3
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
    """
    Merge a list of MP3 byte-strings into one valid MP3 using ffmpeg concat.
    Falls back to raw concatenation if ffmpeg is absent (works for most players
    when all chunks share the same bitrate/samplerate).
    """
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
    "Mmm.",
    "Yeah.",
    "Right.",
    "Okay.",
    "Got it.",
    "Mmm, okay.",
    "Yeah, okay.",
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
    """Call Google TTS for a single sentence. Returns MP3 bytes or None."""
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
@app.route('/transcribe', methods=['POST', 'OPTIONS'])
def transcribe():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_bytes = request.files['audio'].read()
        audio_name = request.files['audio'].filename or 'audio.webm'

        if 'mp4' in audio_name:
            mime = 'audio/mp4'
        elif 'ogg' in audio_name:
            mime = 'audio/ogg'
        else:
            mime = 'audio/webm'

        transcript = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=(audio_name, io.BytesIO(audio_bytes), mime),
        )
        return jsonify({"success": True, "transcript": transcript.text.strip()})

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

        print(f"[Google TTS] Voice: {cfg['voice']} | Chirp-HD: {'Chirp-HD' in cfg['voice']} | Text: {final[:120]}")

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
        # Append 3 seconds of proper silence using ffmpeg so the MP3 stays valid.
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
            # Collect all sentence audio first, then merge with silence via ffmpeg
            # so the final MP3 is a single valid file (no frame corruption).
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
                # Concatenate sentence MP3s properly with ffmpeg
                combined_bytes = merge_mp3s(chunks)

            # Append 3 seconds of proper silence
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
# ENDPOINT: /therapy-session
# ════════════════════════════════════════════════════════════
@app.route('/therapy-session', methods=['POST', 'OPTIONS'])
def therapy_session():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id")
        start_new = data.get("start_new", False)
        # Default changed to "long" so responses are richer out of the box
        response_length = data.get("response_length", "long")  # "short" | "medium" | "long"

        if not user_id or not user_message:
            return jsonify({"error": "user_id and message required"}), 400

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

        current_phase = session_data.get("phase", 1)
        extracted_so_far = session_data.get("extracted", {})

        length_guide = {
            "short": (
                "RESPONSE LENGTH: SHORT — 1 to 2 sentences max. One thought only. "
                "Get straight to the point. No elaboration."
            ),
            "medium": (
                "RESPONSE LENGTH: MEDIUM — 2 to 3 sentences. One main point plus a follow-up question. "
                "Warm but concise."
            ),
            "long": (
                "RESPONSE LENGTH: LONG — 5 to 7 sentences. This is what a real therapist sounds like. "
                "Validate first, then reflect back what you heard in your own words, then gently explore "
                "or reframe, then close with a question. Take your time. Let the reply breathe. "
                "Use natural pauses (...) between thoughts. Do NOT rush to the question — earn it by "
                "showing you've really heard them first. The person should feel truly seen."
            ),
        }.get(response_length, "RESPONSE LENGTH: LONG — 5 to 7 sentences.")

        # Increased max_tokens for long responses to give the model room to breathe
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

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages_for_model,
            temperature=0.65,
            max_tokens=length_tokens
        )
        raw_reply = response.choices[0].message.content.strip()
        parsed = parse_json_response(raw_reply)

        if not parsed:
            messages.append({"role": "assistant", "content": raw_reply})
            db.collection("users").document(user_id) \
              .collection("therapy_sessions").document(session_id) \
              .update({"messages": messages})
            return jsonify({
                "session_id": session_id, "reply": raw_reply,
                "phase": current_phase, "session_complete": False
            })

        ai_reply = parsed.get("message", raw_reply)
        next_phase = parsed.get("phase", current_phase)
        session_complete = parsed.get("session_complete", False)
        new_extracted = parsed.get("extracted", {})

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

        update_payload = {
            "messages": messages, "phase": next_phase,
            "extracted": merged, "session_complete": session_complete,
            "updated_at": datetime.utcnow().isoformat()
        }
        if session_complete:
            update_payload["completed_at"] = datetime.utcnow().isoformat()

        db.collection("users").document(user_id) \
          .collection("therapy_sessions").document(session_id).update(update_payload)

        tone, pause_ms = detect_reply_tone(ai_reply)
        print(f"[tone] → {tone} | pause {pause_ms}ms before speaking")

        return jsonify({
            "session_id": session_id, "reply": ai_reply,
            "phase": next_phase, "session_complete": session_complete,
            "extracted": merged,
            "turn_count": len([m for m in messages if m["role"] == "user"]),
            "tone": tone,
            "pause_ms": pause_ms,
            "response_length": response_length,
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
        ]
    })


# ════════════════════════════════════════════════════════════
# ENDPOINT: /tts-config  — read & write TTS settings live
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
            "/speak-stream": "sentence-chunked TTS — faster + more natural (+ 3s silence tail)",
            "/transcribe": "Whisper STT",
            "/therapy-session": "CBT session turn (long responses by default)",
            "/session-to-plan": "convert session to activity plan",
            "/therapy-session/history": "list past sessions",
        }
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
