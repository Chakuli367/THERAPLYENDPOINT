import os
import json
import io
import copy
import random
import threading
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


# ── CORS: handle ALL OPTIONS preflights globally ──────────────
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


# ── Firebase ──────────────────────────────────────────────────
firebase_config_json = os.environ.get("FIREBASE_CONFIG")
if not firebase_config_json:
    raise EnvironmentError("FIREBASE_CONFIG environment variable not set")

firebase_json = json.loads(firebase_config_json)
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_json)
    initialize_app(cred)

db = firestore.client()


# ════════════════════════════════════════════════════════════
#  TTS CONFIG & BOOTSTRAP
# ════════════════════════════════════════════════════════════

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

AVAILABLE_VOICES = [
    "en-US-Chirp-HD-O", "en-US-Chirp-HD-D", "en-US-Chirp-HD-F",
    "en-US-Chirp3-HD-Achernar", "en-US-Chirp3-HD-Aoede", "en-US-Chirp3-HD-Callirrhoe",
    "en-US-Chirp3-HD-Charon", "en-US-Chirp3-HD-Despina", "en-US-Chirp3-HD-Enceladus",
    "en-US-Chirp3-HD-Erinome", "en-US-Chirp3-HD-Fenrir", "en-US-Chirp3-HD-Gacrux",
    "en-US-Chirp3-HD-Iapetus", "en-US-Chirp3-HD-Kore", "en-US-Chirp3-HD-Laomedeia",
    "en-US-Chirp3-HD-Leda", "en-US-Chirp3-HD-Orus", "en-US-Chirp3-HD-Puck",
    "en-US-Chirp3-HD-Pulcherrima", "en-US-Chirp3-HD-Rasalgethi", "en-US-Chirp3-HD-Sadachbia",
    "en-US-Chirp3-HD-Sadaltager", "en-US-Chirp3-HD-Schedar", "en-US-Chirp3-HD-Sulafat",
    "en-US-Chirp3-HD-Umbriel", "en-US-Chirp3-HD-Vindemiatrix", "en-US-Chirp3-HD-Zephyr",
    "en-US-Chirp3-HD-Zubenelgenubi",
]


def bootstrap_tts_config(max_retries=5, delay=2):
    import time
    ref = db.collection("config").document("tts_settings")
    for attempt in range(1, max_retries + 1):
        try:
            doc = ref.get()
            existing = doc.to_dict() if doc.exists else {}
            missing = {k: v for k, v in TTS_DEFAULTS.items() if k not in existing}
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


# ── Groq client ───────────────────────────────────────────────
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

GOOGLE_TTS_KEY = os.environ.get("GOOGLE_TTS_KEY")
if not GOOGLE_TTS_KEY:
    print("Warning: GOOGLE_TTS_KEY not set — /speak will fail")

GOOGLE_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
SILENCE_PADDING_MS = 3000


# ════════════════════════════════════════════════════════════
#  8-WEEK PROGRAM DEFINITION
# ════════════════════════════════════════════════════════════
#
#  Structure:
#    Program → 8 Weeks → each week has 2–3 sessions
#    Each session: 5 Parts
#      Part 1: Check-in (voice + anxiety slider)
#      Part 2: Last-session review (voice companion)
#      Part 3: Psychoeducation (concept card — static content, rendered by frontend)
#      Part 4: Interactive exercise (exercise_type key → EXERCISE_CATALOG)
#      Part 5: Commit & close (voice companion proposes action, user sets reminder)
#
# ════════════════════════════════════════════════════════════

PROGRAM = [
    # ── Week 1: Awareness ────────────────────────────────────
    {
        "week": 1,
        "theme": "Awareness",
        "tagline": "What is my anxiety actually doing?",
        "sessions": [
            {
                "session_number": 1,
                "title": "Meeting your anxiety",
                "goal": "Understand how social anxiety shows up across different areas of your life.",
                "psychoeducation": {
                    "title": "What social anxiety actually is",
                    "body": (
                        "Social anxiety isn't shyness. It's your brain's threat-detection system "
                        "misfiring in social situations — treating embarrassment like physical danger. "
                        "The fight-or-flight response activates. Your heart races. Your mind predicts "
                        "disaster. None of this means something is wrong with you. It means your brain "
                        "learned to be very cautious. That learning can change."
                    ),
                    "key_insight": "Anxiety is a false alarm, not a forecast.",
                },
                "exercise_type": "anxiety_mapping",
                "commit_prompt": (
                    "This week, notice one moment when anxiety shows up — "
                    "just notice it, don't fight it. What situation was it?"
                ),
            },
            {
                "session_number": 2,
                "title": "Where you feel it",
                "goal": "Connect anxiety to physical sensations in the body.",
                "psychoeducation": {
                    "title": "Anxiety lives in the body",
                    "body": (
                        "Before you ever have a conscious anxious thought, your body is already "
                        "responding. Tight chest. Shallow breathing. Tense shoulders. Knotted stomach. "
                        "These are your early warning signals. Learning to read them gives you a head "
                        "start — you can intervene before the spiral takes hold."
                    ),
                    "key_insight": "Your body knows before your mind does.",
                },
                "exercise_type": "body_scan",
                "commit_prompt": (
                    "Once today, when you feel tension in your body, pause and name where it is. "
                    "That's the whole task."
                ),
            },
        ],
    },

    # ── Week 2: Triggers ─────────────────────────────────────
    {
        "week": 2,
        "theme": "Triggers",
        "tagline": "What sets it off?",
        "sessions": [
            {
                "session_number": 1,
                "title": "Mapping your triggers",
                "goal": "Identify the specific situations that reliably spike anxiety.",
                "psychoeducation": {
                    "title": "Triggers aren't random",
                    "body": (
                        "Social anxiety tends to cluster around predictable themes: "
                        "being evaluated, being the centre of attention, "
                        "saying something wrong, rejection. Your triggers are specific — "
                        "a meeting with your boss hits differently than a chat with a friend. "
                        "Knowing your triggers turns a vague dread into a concrete map you can work with."
                    ),
                    "key_insight": "What you can name, you can change.",
                },
                "exercise_type": "trigger_swipe",
                "commit_prompt": (
                    "This week, when a trigger situation comes up, don't avoid it yet — "
                    "just stay one minute longer than you normally would."
                ),
            },
            {
                "session_number": 2,
                "title": "Avoidance patterns",
                "goal": "See how avoidance maintains and strengthens anxiety over time.",
                "psychoeducation": {
                    "title": "Why avoidance feels like relief but isn't",
                    "body": (
                        "When you avoid a feared situation, anxiety drops immediately. "
                        "That relief is real — and it's a trap. Your brain learns: "
                        "'We escaped danger.' Next time, the alarm fires louder. "
                        "Avoidance is short-term relief, long-term growth of the fear. "
                        "Every avoided situation adds one brick to the wall."
                    ),
                    "key_insight": "Avoidance feeds the fear it's trying to escape.",
                },
                "exercise_type": "avoidance_diary",
                "commit_prompt": (
                    "Identify one small thing you've been avoiding. "
                    "You don't have to do it yet — just write it down."
                ),
            },
        ],
    },

    # ── Week 3: Thoughts ─────────────────────────────────────
    {
        "week": 3,
        "theme": "Thoughts",
        "tagline": "What am I actually telling myself?",
        "sessions": [
            {
                "session_number": 1,
                "title": "Catching the thought",
                "goal": "Learn to identify the automatic thoughts that fuel anxiety.",
                "psychoeducation": {
                    "title": "Automatic thoughts",
                    "body": (
                        "Thoughts happen so fast they feel like facts. "
                        "'They think I'm boring.' 'I said the wrong thing.' 'Everyone noticed.' "
                        "These are automatic thoughts — fast, reflexive, and almost always negative "
                        "when anxiety is running the show. The first step isn't to challenge them. "
                        "It's just to notice them. You can't change what you can't see."
                    ),
                    "key_insight": "A thought is not a fact. It's just a thought.",
                },
                "exercise_type": "thought_catch",
                "commit_prompt": (
                    "Today, when you notice an anxious thought, write it down word for word. "
                    "No analysis. Just capture it."
                ),
            },
            {
                "session_number": 2,
                "title": "The full thought record",
                "goal": "Complete a structured CBT thought record from start to finish.",
                "psychoeducation": {
                    "title": "The thought-feeling-behaviour loop",
                    "body": (
                        "CBT is built on one observation: thoughts, feelings, and behaviours "
                        "are all connected and all influence each other. "
                        "Change the thought, and the feeling shifts. "
                        "Change the behaviour, and the thought updates. "
                        "The thought record is the classic tool for interrupting this loop "
                        "at the cognitive level — examining the thought before it becomes a spiral."
                    ),
                    "key_insight": "You don't have to believe every thought you have.",
                },
                "exercise_type": "thought_record",
                "commit_prompt": (
                    "Use one anxious thought this week to fill out a thought record — "
                    "even informally, in your head."
                ),
            },
        ],
    },

    # ── Week 4: Evidence ─────────────────────────────────────
    {
        "week": 4,
        "theme": "Evidence",
        "tagline": "Is that thought actually accurate?",
        "sessions": [
            {
                "session_number": 1,
                "title": "Prosecution vs defence",
                "goal": "Learn to examine evidence for and against an anxious prediction.",
                "psychoeducation": {
                    "title": "Treating your thoughts like a trial",
                    "body": (
                        "Anxious minds act as prosecution only — gathering every shred of "
                        "evidence that the feared thing is true and ignoring anything that "
                        "contradicts it. This is confirmation bias in action. "
                        "In this exercise you become the defence lawyer for yourself — "
                        "finding the counter-evidence your brain keeps filing away."
                    ),
                    "key_insight": "Your brain is not an impartial witness.",
                },
                "exercise_type": "evidence_court",
                "commit_prompt": (
                    "Next time an anxious prediction appears, ask: "
                    "what's one piece of evidence that this isn't certain?"
                ),
            },
            {
                "session_number": 2,
                "title": "How likely is it really?",
                "goal": "Recalibrate probability estimates for feared outcomes.",
                "psychoeducation": {
                    "title": "Probability distortion",
                    "body": (
                        "Social anxiety makes unlikely things feel inevitable. "
                        "The chance that everyone noticed your stumble feels like a certainty. "
                        "The chance of being humiliated feels like 80%. "
                        "When you examine these predictions over time, the actual rate of "
                        "feared outcomes is almost always far lower than the anxiety predicted. "
                        "This gap is where your freedom lives."
                    ),
                    "key_insight": "Anxiety is a bad statistician.",
                },
                "exercise_type": "probability_pie",
                "commit_prompt": (
                    "Before a situation you're anxious about this week, "
                    "write your feared outcome and give it a probability. "
                    "Then check what actually happened."
                ),
            },
        ],
    },

    # ── Week 5: Exposure Prep ─────────────────────────────────
    {
        "week": 5,
        "theme": "Exposure Prep",
        "tagline": "What am I actually avoiding?",
        "sessions": [
            {
                "session_number": 1,
                "title": "Building the ladder",
                "goal": "Create a personalised hierarchy of feared situations.",
                "psychoeducation": {
                    "title": "Gradual exposure",
                    "body": (
                        "The most effective treatment for social anxiety is exposure — "
                        "deliberately entering feared situations and staying until anxiety naturally "
                        "decreases. But not all at once. A fear ladder takes the most feared "
                        "situation at the top and builds a staircase of smaller steps downward. "
                        "You start at the bottom, where anxiety is manageable, and work up. "
                        "The brain learns, one step at a time: this is survivable."
                    ),
                    "key_insight": "The ladder is climbed one rung at a time.",
                },
                "exercise_type": "fear_ladder",
                "commit_prompt": (
                    "Look at the bottom rung of your ladder. "
                    "What would it take to attempt it this week?"
                ),
            },
            {
                "session_number": 2,
                "title": "Before and after",
                "goal": "Use prediction logging to track the gap between fear and reality.",
                "psychoeducation": {
                    "title": "The prediction gap",
                    "body": (
                        "Social anxiety makes powerful predictions: 'It will be awful.' "
                        "'I'll freeze.' 'They'll notice everything.' "
                        "Most of these predictions never come true — but because we avoid the "
                        "situation, we never get to see that. Prediction logging closes that loop: "
                        "you write what you expect, then record what actually happened. "
                        "Over time the gap becomes undeniable."
                    ),
                    "key_insight": "Reality is almost always kinder than anxiety predicts.",
                },
                "exercise_type": "prediction_logger",
                "commit_prompt": (
                    "Choose one upcoming situation. "
                    "Write your prediction now — and we'll debrief it next session."
                ),
            },
        ],
    },

    # ── Week 6: Exposure ─────────────────────────────────────
    {
        "week": 6,
        "theme": "Exposure",
        "tagline": "Taking the first real steps.",
        "sessions": [
            {
                "session_number": 1,
                "title": "Planning the exposure",
                "goal": "Set a specific, time-bound exposure commitment with a coping plan.",
                "psychoeducation": {
                    "title": "How to do an exposure",
                    "body": (
                        "A good exposure has three qualities: it's specific, it's slightly "
                        "uncomfortable (SUDS 40–60 to start), and you stay until anxiety "
                        "naturally reduces — not until you escape. "
                        "You don't need to feel confident going in. "
                        "You need to go in anyway and let your nervous system learn "
                        "that the threat was never what it said it was."
                    ),
                    "key_insight": "Courage is acting despite anxiety, not the absence of it.",
                },
                "exercise_type": "exposure_commitment",
                "commit_prompt": (
                    "You've planned the exposure. "
                    "The only task now is to do it before our next session."
                ),
            },
            {
                "session_number": 2,
                "title": "Debrief",
                "goal": "Process the exposure experience — what happened vs. what was predicted.",
                "psychoeducation": {
                    "title": "What the data says",
                    "body": (
                        "After an exposure, the brain needs help updating. "
                        "It will try to explain away the success: 'I got lucky.' "
                        "'It was easier than usual.' 'They were being nice.' "
                        "The debrief is where you explicitly compare your prediction to reality "
                        "and make the learning stick. This is where transformation actually happens."
                    ),
                    "key_insight": "The debrief is as important as the exposure itself.",
                },
                "exercise_type": "exposure_debrief",
                "commit_prompt": (
                    "What's the next rung? "
                    "Name the next situation you'll approach before we meet again."
                ),
            },
        ],
    },

    # ── Week 7: Resilience ────────────────────────────────────
    {
        "week": 7,
        "theme": "Resilience",
        "tagline": "What to do when it spikes again.",
        "sessions": [
            {
                "session_number": 1,
                "title": "Safety behaviours",
                "goal": "Identify subtle avoidance behaviours that prevent full learning.",
                "psychoeducation": {
                    "title": "Safety behaviours — the sneaky saboteur",
                    "body": (
                        "Safety behaviours are the subtle things you do to reduce anxiety "
                        "in a feared situation: over-preparing, rehearsing what to say, "
                        "staying near the exit, avoiding eye contact, speaking quietly. "
                        "They feel helpful. They're not. They prevent your brain from "
                        "learning the situation is actually safe — and they keep you "
                        "dependent on the crutch."
                    ),
                    "key_insight": "Safety behaviours are avoidance in disguise.",
                },
                "exercise_type": "safety_behaviors_audit",
                "commit_prompt": (
                    "In your next social situation, drop one safety behaviour. "
                    "Just one. Notice what happens."
                ),
            },
            {
                "session_number": 2,
                "title": "Your coping cards",
                "goal": "Build a personalised set of go-to responses for high-anxiety moments.",
                "psychoeducation": {
                    "title": "When the brain goes offline",
                    "body": (
                        "In a high-anxiety moment, your prefrontal cortex — the rational part — "
                        "partially shuts down. You can't recall the insights from therapy. "
                        "Coping cards solve this: they're short, specific, pre-prepared responses "
                        "you can pull up when the system is overloaded. "
                        "Like having your own personalised anxiety first-aid kit."
                    ),
                    "key_insight": "Prepare your calm for the moments when calm is hardest.",
                },
                "exercise_type": "coping_cards",
                "commit_prompt": (
                    "Save your coping cards somewhere accessible — your phone home screen, "
                    "your wallet. You'll use one before our last session."
                ),
            },
        ],
    },

    # ── Week 8: Independence ──────────────────────────────────
    {
        "week": 8,
        "theme": "Independence",
        "tagline": "Life without the training wheels.",
        "sessions": [
            {
                "session_number": 1,
                "title": "Your values compass",
                "goal": "Reconnect action with what genuinely matters — not what anxiety permits.",
                "psychoeducation": {
                    "title": "Living by values, not rules",
                    "body": (
                        "Social anxiety narrows your life. It builds rules: "
                        "'Don't speak unless spoken to.' 'Stay in the background.' "
                        "'Don't risk embarrassment.' "
                        "Values point in the opposite direction. "
                        "What kind of friend do you want to be? What experiences do you want? "
                        "What does a full life look like for you? "
                        "When you move toward values instead of away from anxiety, "
                        "the fear becomes smaller by comparison."
                    ),
                    "key_insight": "A values-led life is the antidote to an anxiety-led one.",
                },
                "exercise_type": "values_compass",
                "commit_prompt": (
                    "Pick the value with the biggest gap. "
                    "What's one thing you could do this week that serves it?"
                ),
            },
            {
                "session_number": 2,
                "title": "Your progress story",
                "goal": "Review the full journey — patterns changed, moments won, next steps.",
                "psychoeducation": {
                    "title": "What you've built",
                    "body": (
                        "Progress with anxiety isn't linear and it's rarely dramatic. "
                        "It looks like: stayed in the room a little longer. "
                        "Sent the message instead of drafting it for three days. "
                        "Spoke in a meeting even though my voice shook. "
                        "These small wins are the real data. "
                        "They're evidence that the story your anxiety tells about you "
                        "is not the whole story."
                    ),
                    "key_insight": "You are not your anxiety.",
                },
                "exercise_type": "progress_story",
                "commit_prompt": (
                    "What's the one thing you want to keep doing after this program ends?"
                ),
            },
        ],
    },
]


def get_program_session(week: int, session_number: int) -> dict | None:
    """Return the program session definition for a given week + session number."""
    for w in PROGRAM:
        if w["week"] == week:
            for s in w["sessions"]:
                if s["session_number"] == session_number:
                    return {**s, "week": w["week"], "theme": w["theme"], "tagline": w["tagline"]}
    return None


def get_next_program_position(week: int, session_number: int) -> dict:
    """Return the next (week, session_number) in the program, or None if complete."""
    for wi, w in enumerate(PROGRAM):
        if w["week"] == week:
            sessions = w["sessions"]
            for si, s in enumerate(sessions):
                if s["session_number"] == session_number:
                    # next session in same week?
                    if si + 1 < len(sessions):
                        return {"week": week, "session_number": sessions[si + 1]["session_number"]}
                    # next week?
                    if wi + 1 < len(PROGRAM):
                        next_week = PROGRAM[wi + 1]
                        return {"week": next_week["week"], "session_number": next_week["sessions"][0]["session_number"]}
                    # program complete
                    return {"week": None, "session_number": None, "program_complete": True}
    return {"week": 1, "session_number": 1}


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
                    "id": "situation", "label": "What just happened?",
                    "placeholder": "Describe the situation in 1–2 sentences",
                    "type": "textarea", "rows": 2,
                },
                {
                    "id": "hot_thought", "label": "What's the hottest thought in your head?",
                    "placeholder": "The thought that's hitting hardest right now",
                    "type": "textarea", "rows": 2,
                },
                {
                    "id": "belief_before", "label": "How much do you believe that thought? (0–100%)",
                    "type": "slider", "min": 0, "max": 100, "step": 5, "unit": "%",
                },
                {
                    "id": "emotion", "label": "What emotion does it bring up?",
                    "placeholder": "e.g. shame, fear, anger, sadness",
                    "type": "text",
                },
                {
                    "id": "emotion_intensity", "label": "Intensity of that emotion (0–100)",
                    "type": "slider", "min": 0, "max": 100, "step": 5, "unit": "%",
                },
                {
                    "id": "evidence_for", "label": "Evidence that supports this thought",
                    "placeholder": "Be honest — what facts back it up?",
                    "type": "textarea", "rows": 2,
                },
                {
                    "id": "evidence_against", "label": "Evidence that contradicts this thought",
                    "placeholder": "What facts challenge it?",
                    "type": "textarea", "rows": 2,
                },
                {
                    "id": "friend_advice", "label": "What would you tell a close friend who had this thought?",
                    "placeholder": "Imagine they came to you with this exact thought...",
                    "type": "textarea", "rows": 2,
                },
                {
                    "id": "balanced_thought", "label": "A more balanced version of the thought",
                    "placeholder": "Not toxic positivity — just more accurate",
                    "type": "textarea", "rows": 2,
                },
                {
                    "id": "belief_after", "label": "How much do you believe the original thought now? (0–100%)",
                    "type": "slider", "min": 0, "max": 100, "step": 5, "unit": "%",
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
                {"id": "head",     "label": "Head & face",     "prompt": "Notice any tension in your jaw, forehead, or eyes"},
                {"id": "neck",     "label": "Neck & shoulders", "prompt": "Let your shoulders drop. Any tightness here?"},
                {"id": "chest",    "label": "Chest",            "prompt": "Notice your breathing. Is it shallow or deep?"},
                {"id": "abdomen",  "label": "Abdomen",          "prompt": "Any knots or tension in your stomach?"},
                {"id": "arms",     "label": "Arms & hands",     "prompt": "Are your hands clenched? Let them relax."},
                {"id": "legs",     "label": "Legs & feet",      "prompt": "Notice any tension. Let your legs feel heavy."},
            ],
            "tension_scale": {"min": 0, "max": 10, "label": "Tension level"},
            "message": "You can't think your way out of a body response. Start here.",
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
            "steps":           10,
            "fields_per_step": [
                {"id": "situation", "label": "Situation", "type": "text",   "placeholder": "What would you do?"},
                {"id": "anxiety",   "label": "SUDS",      "type": "slider", "min": 0, "max": 100, "unit": "/100"},
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
                {"id": "importance", "label": "How important is this to you?", "type": "slider", "min": 0, "max": 10},
                {"id": "living_it",  "label": "How much are you living by it?", "type": "slider", "min": 0, "max": 10},
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

    # ── Program-specific exercises (new) ─────────────────────

    "anxiety_mapping": {
        "name":        "Anxiety mapping",
        "description": "Map where anxiety shows up across different life domains",
        "best_for":    ["first session", "awareness", "psychoeducation"],
        "duration_s":  300,
        "config": {
            "domains": [
                {"id": "work",        "label": "Work / school",         "icon": "💼"},
                {"id": "social",      "label": "Social situations",     "icon": "👥"},
                {"id": "family",      "label": "Family",                "icon": "🏠"},
                {"id": "romance",     "label": "Romantic relationships","icon": "❤️"},
                {"id": "health",      "label": "Health",                "icon": "🫀"},
                {"id": "performance", "label": "Performance / speaking","icon": "🎤"},
                {"id": "strangers",   "label": "Strangers / public",    "icon": "🌍"},
                {"id": "authority",   "label": "Authority figures",     "icon": "🏛️"},
            ],
            "scale": {"min": 0, "max": 10, "label": "Anxiety in this area (0 = none, 10 = severe)"},
            "message": "There are no right answers. This is just your map.",
        },
        "inputs": ["domain_scores", "notes"],
    },

    "trigger_swipe": {
        "name":        "Trigger sorter",
        "description": "Swipe situations to rate how much they trigger anxiety",
        "best_for":    ["trigger identification", "awareness"],
        "duration_s":  300,
        "config": {
            "situations": [
                "Making a phone call to a stranger",
                "Attending a party where I know few people",
                "Speaking up in a meeting",
                "Eating alone in public",
                "Starting a conversation with someone new",
                "Being the centre of attention",
                "Disagreeing with someone",
                "Asking for help",
                "Being evaluated or assessed",
                "Making a mistake in front of others",
                "Using public transport at busy times",
                "Returning something to a shop",
                "Being on a video call",
                "Meeting someone's friends or family for the first time",
                "Going to a job interview",
            ],
            "scale": {"low": "No anxiety", "high": "Very anxious"},
            "ui_hint": "swipe_cards",
            "message": "Rate each situation honestly. This becomes your personal trigger map.",
        },
        "inputs": ["situation_ratings", "notes"],
    },

    "avoidance_diary": {
        "name":        "Avoidance diary",
        "description": "Track situations avoided and the anxiety behind each",
        "best_for":    ["avoidance patterns", "awareness"],
        "duration_s":  300,
        "config": {
            "fields": [
                {"id": "situation",      "label": "What did you avoid?",               "type": "text"},
                {"id": "anxiety_level",  "label": "Anxiety level if you had done it",  "type": "slider", "min": 0, "max": 10},
                {"id": "what_you_did",   "label": "What did you do instead?",          "type": "text"},
                {"id": "relief_level",   "label": "Short-term relief felt",            "type": "slider", "min": 0, "max": 10},
                {"id": "cost",           "label": "What did avoiding cost you?",       "type": "textarea", "rows": 2},
            ],
            "max_entries": 5,
            "message": "Avoidance always has a price. Naming it is the first step.",
        },
        "inputs": ["avoidance_entries", "notes"],
    },

    "thought_catch": {
        "name":        "Thought catch",
        "description": "Identify and write down a single automatic anxious thought",
        "best_for":    ["thought awareness", "first cognitive work"],
        "duration_s":  180,
        "config": {
            "fields": [
                {"id": "situation",    "label": "What was happening?",                 "type": "text"},
                {"id": "thought",      "label": "The exact thought that appeared",     "type": "textarea", "rows": 2,
                 "placeholder": "Write it word for word, even if it sounds extreme"},
                {"id": "emotion",      "label": "What emotion came with it?",          "type": "text"},
                {"id": "body_signal",  "label": "Where did you feel it in your body?", "type": "text"},
                {"id": "believed",     "label": "How much did you believe it? (0–100%)", "type": "slider", "min": 0, "max": 100},
            ],
            "message": "You can't change what you haven't noticed.",
        },
        "inputs": ["responses", "notes"],
    },

    "evidence_court": {
        "name":        "Evidence court",
        "description": "Examine evidence for and against an anxious belief",
        "best_for":    ["challenging thoughts", "cognitive restructuring"],
        "duration_s":  480,
        "config": {
            "fields": [
                {"id": "belief",           "label": "The thought on trial",                    "type": "textarea", "rows": 2},
                {"id": "belief_strength",  "label": "How strongly do you believe it? (0–100%)", "type": "slider", "min": 0, "max": 100},
                {"id": "evidence_for",     "label": "Evidence FOR (prosecution)",               "type": "textarea_list", "rows": 3,
                 "placeholder": "Add each piece of evidence on its own line"},
                {"id": "evidence_against", "label": "Evidence AGAINST (defence)",              "type": "textarea_list", "rows": 3,
                 "placeholder": "What facts challenge this belief?"},
                {"id": "verdict",          "label": "A more balanced verdict",                  "type": "textarea", "rows": 2},
                {"id": "belief_after",     "label": "How strongly do you believe it now? (0–100%)", "type": "slider", "min": 0, "max": 100},
            ],
            "ui_hint": "split_screen",
            "message": "Your brain is not an impartial witness.",
        },
        "inputs": ["responses", "notes"],
    },

    "probability_pie": {
        "name":        "Probability pie",
        "description": "Recalibrate feared outcome probabilities",
        "best_for":    ["catastrophising", "probability distortion"],
        "duration_s":  360,
        "config": {
            "fields": [
                {"id": "feared_outcome",    "label": "What's the worst outcome you're imagining?", "type": "textarea", "rows": 2},
                {"id": "initial_prob",      "label": "How likely does it feel? (0–100%)",          "type": "slider", "min": 0, "max": 100},
                {"id": "alternative_outcomes", "label": "Other possible outcomes",                 "type": "outcome_builder",
                 "hint": "Add at least 3 other things that could realistically happen"},
                {"id": "realistic_prob",    "label": "After considering alternatives, how likely is the feared outcome?", "type": "slider", "min": 0, "max": 100},
                {"id": "insight",           "label": "What did you notice?",                       "type": "textarea", "rows": 2},
            ],
            "ui_hint": "pie_chart",
            "message": "Anxiety is a bad statistician.",
        },
        "inputs": ["responses", "notes"],
    },

    "prediction_logger": {
        "name":        "Prediction logger",
        "description": "Record a prediction before a situation and debrief it after",
        "best_for":    ["exposure prep", "prediction testing"],
        "duration_s":  240,
        "config": {
            "pre_fields": [
                {"id": "situation",    "label": "What's the upcoming situation?",                 "type": "text"},
                {"id": "prediction",   "label": "What do you predict will happen?",              "type": "textarea", "rows": 2},
                {"id": "anxiety_pre",  "label": "Anxiety going in (0–10)",                       "type": "slider", "min": 0, "max": 10},
                {"id": "worst_case",   "label": "Absolute worst case?",                           "type": "text"},
                {"id": "best_case",    "label": "Most realistic positive outcome?",               "type": "text"},
            ],
            "post_fields": [
                {"id": "what_happened",  "label": "What actually happened?",                    "type": "textarea", "rows": 2},
                {"id": "anxiety_post",   "label": "Anxiety level during (0–10)",                "type": "slider", "min": 0, "max": 10},
                {"id": "prediction_met", "label": "Did your prediction come true?",             "type": "select",
                 "options": ["Yes, exactly", "Partially", "No, it was different", "Better than predicted"]},
                {"id": "learning",       "label": "What did this teach you?",                   "type": "textarea", "rows": 2},
            ],
            "two_stage": True,
            "message": "Reality is almost always kinder than anxiety predicts.",
        },
        "inputs": ["pre_responses", "post_responses", "notes"],
    },

    "exposure_commitment": {
        "name":        "Exposure commitment",
        "description": "Plan a specific exposure with a coping strategy",
        "best_for":    ["exposure prep", "action planning"],
        "duration_s":  480,
        "config": {
            "fields": [
                {"id": "situation",      "label": "The exposure situation",                    "type": "text",
                 "placeholder": "Be specific — where, when, with whom"},
                {"id": "suds_estimate",  "label": "Expected anxiety (SUDS 0–100)",             "type": "slider", "min": 0, "max": 100},
                {"id": "date",           "label": "When will you do it?",                      "type": "date"},
                {"id": "goal",           "label": "What counts as success?",                   "type": "textarea", "rows": 2,
                 "placeholder": "Not 'feel no anxiety' — what would 'did the thing' look like?"},
                {"id": "safety_drop",    "label": "Which safety behaviour will you drop?",     "type": "text"},
                {"id": "coping_plan",    "label": "If anxiety spikes, I will...",              "type": "textarea", "rows": 2},
                {"id": "reminder_time",  "label": "Set a reminder",                           "type": "datetime_local"},
            ],
            "message": "Courage is acting despite anxiety, not the absence of it.",
        },
        "inputs": ["responses", "notes"],
    },

    "exposure_debrief": {
        "name":        "Exposure debrief",
        "description": "Process what happened during an exposure",
        "best_for":    ["post-exposure processing", "learning consolidation"],
        "duration_s":  360,
        "config": {
            "fields": [
                {"id": "situation",       "label": "What was the exposure?",                    "type": "text"},
                {"id": "anxiety_before",  "label": "Anxiety before (0–10)",                    "type": "slider", "min": 0, "max": 10},
                {"id": "anxiety_peak",    "label": "Anxiety at peak (0–10)",                   "type": "slider", "min": 0, "max": 10},
                {"id": "anxiety_after",   "label": "Anxiety at end (0–10)",                    "type": "slider", "min": 0, "max": 10},
                {"id": "what_happened",   "label": "What actually happened?",                  "type": "textarea", "rows": 2},
                {"id": "prediction_vs",   "label": "How did reality compare to your prediction?", "type": "textarea", "rows": 2},
                {"id": "brain_excuse",    "label": "Is your brain explaining away the success? How?", "type": "textarea", "rows": 2,
                 "placeholder": "'I got lucky', 'they were being nice', etc."},
                {"id": "real_learning",   "label": "What's the honest learning here?",         "type": "textarea", "rows": 2},
            ],
            "anxiety_chart": True,
            "message": "The debrief is as important as the exposure itself.",
        },
        "inputs": ["responses", "notes"],
    },

    "safety_behaviors_audit": {
        "name":        "Safety behaviours audit",
        "description": "Identify subtle avoidance tactics used in social situations",
        "best_for":    ["advanced exposure", "subtle avoidance"],
        "duration_s":  300,
        "config": {
            "behaviors": [
                {"id": "over_prepare",    "label": "Over-prepare or rehearse what I'll say"},
                {"id": "speak_quietly",   "label": "Speak quietly or quickly to draw less attention"},
                {"id": "avoid_eye",       "label": "Avoid eye contact"},
                {"id": "stay_near_exit",  "label": "Position myself near the exit"},
                {"id": "check_phone",     "label": "Check my phone to look busy"},
                {"id": "avoid_pause",     "label": "Fill silences immediately"},
                {"id": "agree_always",    "label": "Agree to avoid conflict"},
                {"id": "joke_deflect",    "label": "Use humour to deflect personal questions"},
                {"id": "arrive_late",     "label": "Arrive late / leave early"},
                {"id": "bring_someone",   "label": "Only go if I have someone with me"},
                {"id": "alcohol",         "label": "Use alcohol to feel more comfortable"},
                {"id": "mental_review",   "label": "Replay conversations for mistakes afterward"},
            ],
            "fields_per_behavior": [
                {"id": "use_it",      "label": "I do this", "type": "checkbox"},
                {"id": "frequency",   "label": "How often?", "type": "select",
                 "options": ["Rarely", "Sometimes", "Often", "Almost always"]},
            ],
            "message": "Safety behaviours are avoidance in disguise.",
        },
        "inputs": ["audit_results", "notes"],
    },

    "coping_cards": {
        "name":        "Coping cards builder",
        "description": "Create personalised go-to responses for high-anxiety moments",
        "best_for":    ["relapse prevention", "in-the-moment coping"],
        "duration_s":  480,
        "config": {
            "card_count": 3,
            "fields_per_card": [
                {"id": "trigger",   "label": "When I feel...",                  "type": "text",
                 "placeholder": "e.g. panic before a meeting"},
                {"id": "thought",   "label": "My brain tells me...",            "type": "textarea", "rows": 2,
                 "placeholder": "e.g. everyone will think I'm incompetent"},
                {"id": "response",  "label": "But the truth is...",             "type": "textarea", "rows": 2,
                 "placeholder": "e.g. I have prepared well. Anxiety is not evidence."},
                {"id": "action",    "label": "So I will...",                    "type": "text",
                 "placeholder": "e.g. take 3 slow breaths, then go in"},
            ],
            "ui_hint": "card_builder",
            "message": "Prepare your calm for when calm is hardest.",
        },
        "inputs": ["cards", "notes"],
    },

    "progress_story": {
        "name":        "Progress story",
        "description": "Auto-generated review of the full 8-week journey",
        "best_for":    ["final session", "reflection", "consolidation"],
        "duration_s":  600,
        "config": {
            "sections": [
                {"id": "wins",       "label": "Moments I'm proud of",         "auto_populated": True},
                {"id": "patterns",   "label": "Patterns I've noticed",        "auto_populated": True},
                {"id": "anxiety",    "label": "How my anxiety has shifted",   "auto_populated": True, "chart": True},
                {"id": "forward",    "label": "What I want to keep doing",    "type": "textarea", "rows": 3},
                {"id": "letter",     "label": "A message to future me",       "type": "textarea", "rows": 4},
            ],
            "uses_brain_data": True,
            "message": "You are not your anxiety.",
        },
        "inputs": ["reflections", "notes"],
    },
}


# ════════════════════════════════════════════════════════════
#  BRAIN v2 — Rich Persistent User Memory
# ════════════════════════════════════════════════════════════

BRAIN_V2_SCHEMA = {
    "emotional_baseline": {
        "avg_anxiety":      5.0,
        "trend":            "unknown",
        "anxiety_history":  [],
        "peak_anxiety":     0,
        "lowest_anxiety":   10,
        "good_days_streak": 0,
    },
    "cognitive": {
        "recurring_distortions": [],
        "strongest_reframe":     "",
        "reframe_history":       [],
        "avoidance_triggers":    [],
        "core_fears":            [],
        "thought_themes":        [],
    },
    "therapy": {
        "sessions_completed":   0,
        "sessions_abandoned":   0,
        "avg_session_length":   0,
        "last_session_id":      "",
        "last_session_date":    "",
        "breakthrough_moments": [],
    },
    "exercises": {
        "total_completed":       0,
        "types_tried":           [],
        "most_effective":        "",
        "avg_anxiety_reduction": 0.0,
        "recent_results":        [],
    },
    "personality": {
        "communication_style": "unknown",
        "prefers_metaphors":   False,
        "comfort_locations":   [],
        "best_practice_time":  "",
        "support_network":     [],
        "wins":                [],
    },
    "last_interaction": "never",
    "last_seen":        "",
    "profile_version":  2,
}


def _deep_merge(base: dict, override: dict):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def get_brain_v2(user_id: str) -> dict:
    try:
        doc = db.collection("users").document(user_id) \
                .collection("brain").document("context").get()
        stored = doc.to_dict() if doc.exists else {}
    except Exception as e:
        print(f"[brain_v2] read failed: {e}")
        stored = {}
    merged = copy.deepcopy(BRAIN_V2_SCHEMA)
    _deep_merge(merged, stored)
    return merged


def save_brain_v2(user_id: str, brain: dict):
    try:
        db.collection("users").document(user_id) \
          .collection("brain").document("context") \
          .set(brain, merge=True)
    except Exception as e:
        print(f"[brain_v2] write failed: {e}")


def build_rich_brain_context(user_id: str) -> str:
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

    lines = ["═══ USER MEMORY — shape your response with this, never recite it ═══"]

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
    if distortions:
        lines.append(
            f"Recurring thought patterns: {', '.join(distortions[:3])}. "
            "You've seen these before — name them gently when they appear."
        )
    if reframe:
        lines.append(f"A reframe that landed before: \"{reframe[:120]}\". You can build on this if it fits.")
    if core_fears:
        lines.append(f"Core fears: {', '.join(core_fears[:3])}. Don't name these directly unless the user opens that door.")
    if triggers:
        lines.append(f"Known avoidance triggers: {', '.join(triggers[:4])}.")
    if wins:
        lines.append(f"Recent wins to reference warmly if relevant: {'; '.join(str(w) for w in wins[:2])}.")
    if support:
        lines.append(f"Support network: {', '.join(str(s) for s in support[:3])}.")
    if ex["total_completed"] > 0 and best_ex:
        lines.append(
            f"Exercises completed: {ex['total_completed']}. "
            f"Most effective: {best_ex} (avg anxiety drop: {ex_reduction:.1f} pts)."
        )
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
    brain    = get_brain_v2(user_id)
    extracted = session_data.get("extracted", {})
    messages  = session_data.get("messages", [])

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
    history = history[-20:]
    scores  = [h["score"] for h in history]
    avg     = sum(scores) / len(scores)
    if len(scores) >= 3:
        recent3 = scores[-3:]
        trend = ("improving" if recent3[-1] < recent3[0] - 0.5
                 else "worsening" if recent3[-1] > recent3[0] + 0.5
                 else "stable")
    else:
        trend = "unknown"

    brain["emotional_baseline"]["anxiety_history"] = history
    brain["emotional_baseline"]["avg_anxiety"]     = round(avg, 2)
    brain["emotional_baseline"]["trend"]           = trend
    brain["emotional_baseline"]["peak_anxiety"]    = max(brain["emotional_baseline"]["peak_anxiety"], pre_anxiety)
    brain["emotional_baseline"]["lowest_anxiety"]  = min(brain["emotional_baseline"]["lowest_anxiety"], pre_anxiety)

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

    situation = extracted.get("situation", "")
    if situation and situation not in brain["cognitive"]["avoidance_triggers"]:
        brain["cognitive"]["avoidance_triggers"].append(situation)
        brain["cognitive"]["avoidance_triggers"] = brain["cognitive"]["avoidance_triggers"][-10:]

    brain["therapy"]["sessions_completed"] += 1
    brain["therapy"]["last_session_id"]    = session_data.get("session_id", "")
    brain["therapy"]["last_session_date"]  = datetime.utcnow().isoformat()

    turn_count = len([m for m in messages if m.get("role") == "user"])
    prev_avg   = brain["therapy"]["avg_session_length"]
    n          = brain["therapy"]["sessions_completed"]
    brain["therapy"]["avg_session_length"] = round((prev_avg * (n - 1) + turn_count) / n, 1)

    brain["last_interaction"] = "therapy-session"
    brain["last_seen"]        = datetime.utcnow().isoformat()

    threading.Thread(target=save_brain_v2, args=(user_id, brain), daemon=True).start()


def update_brain_after_exercise(user_id: str, result: dict):
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

    all_results = ex["recent_results"]
    if all_results:
        ex["avg_anxiety_reduction"] = round(
            sum(r["reduction"] for r in all_results) / len(all_results), 2
        )
    by_type: dict = {}
    for r in all_results:
        t = r["type"]
        by_type.setdefault(t, []).append(r["reduction"])
    if by_type:
        ex["most_effective"] = max(by_type, key=lambda t: sum(by_type[t]) / len(by_type[t]))

    brain["exercises"]        = ex
    brain["last_interaction"] = "exercise"
    brain["last_seen"]        = datetime.utcnow().isoformat()

    threading.Thread(target=save_brain_v2, args=(user_id, brain), daemon=True).start()


# ════════════════════════════════════════════════════════════
#  PROGRAM PROGRESS STORAGE HELPERS
# ════════════════════════════════════════════════════════════

def get_program_progress(user_id: str) -> dict:
    try:
        doc = db.collection("users").document(user_id) \
                .collection("program").document("progress").get()
        if doc.exists:
            return doc.to_dict()
    except Exception as e:
        print(f"[program] progress read failed: {e}")
    return {
        "current_week":           1,
        "current_session_number": 1,
        "current_part":           1,    # 1=check-in 2=review 3=psychoeducation 4=exercise 5=commit
        "started_at":             datetime.utcnow().isoformat(),
        "program_complete":       False,
        "active_session_id":      None,
    }


def save_program_progress(user_id: str, progress: dict):
    try:
        db.collection("users").document(user_id) \
          .collection("program").document("progress") \
          .set(progress, merge=True)
    except Exception as e:
        print(f"[program] progress write failed: {e}")


# ════════════════════════════════════════════════════════════
#  STRUCTURED SESSION DOCUMENT HELPERS
# ════════════════════════════════════════════════════════════

def create_structured_session(user_id: str, week: int, session_number: int) -> dict:
    """Create and persist a new structured session document."""
    session_id = f"s_{user_id}_w{week}_s{session_number}_{int(datetime.now().timestamp())}"
    program_session = get_program_session(week, session_number)
    session_doc = {
        "session_id":      session_id,
        "user_id":         user_id,
        "week":            week,
        "session_number":  session_number,
        "title":           program_session.get("title", "") if program_session else "",
        "theme":           program_session.get("theme", "") if program_session else "",
        "current_part":    1,
        # Part 1 — check-in
        "checkin_score":   None,
        "checkin_note":    None,
        # Part 2 — review
        "review_complete": False,
        "review_messages": [],
        # Part 3 — psychoeducation (frontend renders; we just track completion)
        "psychoeducation_read": False,
        # Part 4 — exercise
        "exercise_type":        program_session.get("exercise_type") if program_session else None,
        "exercise_result_id":   None,
        "exercise_complete":    False,
        # Part 5 — commit
        "commitment":           None,
        "commitment_date":      None,
        "session_complete":     False,
        # Conversation history (used across parts 2 + 5)
        "messages":             [],
        "created_at":           datetime.utcnow().isoformat(),
        "updated_at":           datetime.utcnow().isoformat(),
    }
    db.collection("users").document(user_id) \
      .collection("structured_sessions").document(session_id).set(session_doc)
    return session_doc


def get_structured_session(user_id: str, session_id: str) -> dict | None:
    try:
        doc = db.collection("users").document(user_id) \
                .collection("structured_sessions").document(session_id).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        print(f"[session] read failed: {e}")
        return None


def update_structured_session(user_id: str, session_id: str, updates: dict):
    updates["updated_at"] = datetime.utcnow().isoformat()
    db.collection("users").document(user_id) \
      .collection("structured_sessions").document(session_id) \
      .update(updates)


# ════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS
# ════════════════════════════════════════════════════════════

def load_personality() -> str:
    try:
        with open("prompt_therapy_personality.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""


# Part-specific system prompt injected alongside the personality
PART_SYSTEM_PROMPTS = {
    # Part 2: review last session's commitment
    2: """You are in PART 2 — LAST SESSION REVIEW.
Your job: ask about the commitment from last session, then reflect on what happened.
If this is week 1, session 1: skip the review and move straight to warmly welcoming the user.

Rules:
- ONE question only.
- Reference the specific commitment from last session if available in context.
- If the user avoided the commitment, don't shame — name the pattern gently.
- If they completed it, don't over-celebrate — acknowledge what it actually meant.
- Keep to 3–4 sentences.
- Always respond with valid JSON: {"message": "...", "part_complete": false}
- Set part_complete to true when review is done (after 1–2 turns).""",

    # Part 5: commit and close
    5: """You are in PART 5 — COMMIT AND CLOSE.
Your job: help the user commit to one specific real-world action before next session.

The commitment must be:
- Specific (when, where, with whom)
- Slightly uncomfortable but achievable
- Connected to what came up in today's session

Rules:
- Propose a specific commitment based on the session context.
- If the user pushes back, adjust — but don't drop to zero.
- Confirm the commitment, then close the session warmly.
- 3–5 sentences.
- Always respond with valid JSON:
  {"message": "...", "commitment": "...", "part_complete": false}
- Set part_complete to true and populate commitment string when user agrees.""",
}

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

RESPONSE FORMAT (always return this exact JSON):
{
  "message": "your spoken reply",
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

WHISPER_PROMPT = (
    "Therapy session transcript. The speaker may be emotional, speak quietly, "
    "trail off mid-sentence, or pause. Common words: anxiety, anxious, panic, "
    "overwhelmed, avoidance, trigger, spiral, reframe, CBT, worthless, hopeless, "
    "ashamed, embarrassed, therapy, therapist, session, commitment, action steps, "
    "intrusive thoughts, catastrophising, self-worth, coping, grounding."
)

WHISPER_MIN_CONFIDENCE = 0.55
THINKING_FILLERS = [
    "Mmm.", "Yeah.", "Right.", "Okay.", "Got it.", "Mmm, okay.", "Yeah, okay.",
]


# ════════════════════════════════════════════════════════════
#  AUDIO HELPERS
# ════════════════════════════════════════════════════════════

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
                "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                "-t", str(silence_ms / 1000), "-c:a", "libmp3lame", "-b:a", "128k",
                "-q:a", "4", silence_path
            ], capture_output=True, timeout=10, check=True)
            with open(list_path, "w") as f:
                f.write(f"file '{speech_path}'\nfile '{silence_path}'\n")
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_path, "-c", "copy", output_path
            ], capture_output=True, timeout=15, check=True)
            with open(output_path, "rb") as f:
                return f.read()
    except Exception as e:
        print(f"[silence] ffmpeg failed: {e}")
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
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_path, "-c", "copy", output_path
            ], capture_output=True, timeout=20, check=True)
            with open(output_path, "rb") as f:
                return f.read()
    except Exception as e:
        print(f"[merge_mp3s] failed: {e}")
        return b"".join(chunks)


def get_tts_config():
    try:
        doc  = db.collection("config").document("tts_settings").get()
        data = doc.to_dict() if doc.exists else {}
    except Exception as e:
        print(f"[TTS config] Firestore read failed: {e}")
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


def _synthesise_all(sentence_list, tts_cfg):
    if not GOOGLE_TTS_KEY or not sentence_list:
        return None
    chunks_by_idx = {}
    with ThreadPoolExecutor(max_workers=min(len(sentence_list) + 1, 8)) as pool:
        future_map = {pool.submit(synthesize_sentence, s, tts_cfg): i for i, s in enumerate(sentence_list)}
        for future in as_completed(future_map):
            idx = future_map[future]
            result = future.result()
            if result:
                chunks_by_idx[idx] = result
    return [chunks_by_idx[i] for i in sorted(chunks_by_idx) if i in chunks_by_idx] or None


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
    return re.sub(r'\s{2,}', ' ', text).strip()


def clean_text_for_tts(text):
    try:
        maybe_json = json.loads(text)
        if isinstance(maybe_json, dict) and "message" in maybe_json:
            text = maybe_json["message"]
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
        return f"{random.choice(THINKING_FILLERS)} {text}"
    return text


def emotion_aware_preprocess(text, cfg=None):
    if cfg is None:
        cfg = TTS_DEFAULTS
    if not cfg.get("emotion_aware", True):
        return text
    heavy_keywords = [
        'scared', 'terrified', 'alone', 'hopeless', 'worthless', 'failure',
        'hate myself', 'awful', 'devastated', 'breakdown', 'panic', 'crying',
        'ashamed', 'embarrassed', 'humiliated'
    ]
    if any(kw in text.lower() for kw in heavy_keywords):
        sentences = split_into_sentences(text)
        if len(sentences) >= 2:
            sentences[0] = sentences[0].rstrip('.!?') + '.'
            return sentences[0] + ' ' + ' '.join(sentences[1:])
    return text


def detect_reply_tone(ai_reply):
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
            sia = SentimentIntensityAnalyzer()
        scores   = sia.polarity_scores(ai_reply)
        compound = scores['compound']
        if compound <= -0.4:
            return "heavy", 1600
        elif compound <= 0.1:
            return "normal", 900
        else:
            return "brief", 400
    except Exception as e:
        print(f"[tone] VADER failed: {e}")
        return "normal", 900


def build_tts_response(text: str) -> list[str] | None:
    """Return list of base64-encoded MP3 chunks, or None."""
    if not GOOGLE_TTS_KEY or not text:
        return None
    cfg        = get_tts_config()
    clean      = clean_text_for_tts(text)
    filled     = add_thinking_filler(clean, cfg)
    final_text = emotion_aware_preprocess(filled, cfg)
    sentences  = [s for s in split_into_sentences(final_text) if s.strip()][:8]
    chunks     = _synthesise_all(sentences, cfg)
    if not chunks:
        return None
    return [base64.b64encode(c).decode("utf-8") for c in chunks]


def parse_json_response(text):
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except Exception:
        return None


def preprocess_audio_for_whisper(audio_bytes: bytes, original_mime: str):
    if not _HAS_FFMPEG:
        return audio_bytes, original_mime, False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ext = (
                ".mp4" if "mp4" in original_mime
                else ".ogg" if "ogg" in original_mime
                else ".webm"
            )
            in_path  = os.path.join(tmp, f"input{ext}")
            out_path = os.path.join(tmp, "whisper_ready.wav")
            with open(in_path, "wb") as f:
                f.write(audio_bytes)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "json", in_path],
                capture_output=True, timeout=5
            )
            duration_sec = 0.0
            if probe.returncode == 0:
                try:
                    duration_sec = float(json.loads(probe.stdout)["format"]["duration"])
                except Exception:
                    pass
            if 0 < duration_sec < 0.3:
                return audio_bytes, original_mime, False
            af_chain = (
                "highpass=f=80,lowpass=f=8000,"
                "afftdn=nf=-25,loudnorm=I=-16:LRA=11:TP=-1.5"
            )
            result = subprocess.run([
                "ffmpeg", "-y", "-i", in_path,
                "-ar", "16000", "-ac", "1",
                "-af", af_chain, "-c:a", "pcm_s16le", out_path
            ], capture_output=True, timeout=20, check=True)
            with open(out_path, "rb") as f:
                processed = f.read()
            return processed, "audio/wav", True
    except subprocess.CalledProcessError:
        try:
            with tempfile.TemporaryDirectory() as tmp2:
                ext = (".mp4" if "mp4" in original_mime else ".ogg" if "ogg" in original_mime else ".webm")
                in2  = os.path.join(tmp2, f"input{ext}")
                out2 = os.path.join(tmp2, "whisper_ready.wav")
                with open(in2, "wb") as f:
                    f.write(audio_bytes)
                subprocess.run([
                    "ffmpeg", "-y", "-i", in2, "-ar", "16000", "-ac", "1",
                    "-af", "highpass=f=80,lowpass=f=8000,loudnorm=I=-16:LRA=11:TP=-1.5",
                    "-c:a", "pcm_s16le", out2
                ], capture_output=True, timeout=20, check=True)
                with open(out2, "rb") as f:
                    return f.read(), "audio/wav", True
        except Exception as e2:
            print(f"[preprocess] fallback failed: {e2}")
            return audio_bytes, original_mime, False
    except Exception as e:
        print(f"[preprocess] unexpected error: {e}")
        return audio_bytes, original_mime, False


def compute_avg_confidence(verbose_json_response) -> float:
    try:
        segments = verbose_json_response.segments
        if not segments:
            return 1.0
        return sum(1.0 - seg.no_speech_prob for seg in segments) / len(segments)
    except Exception:
        return 1.0


# ════════════════════════════════════════════════════════════
#  ██████╗ ██████╗  ██████╗  ██████╗ ██████╗  █████╗ ███╗   ███╗
#  ██╔══██╗██╔══██╗██╔═══██╗██╔════╝ ██╔══██╗██╔══██╗████╗ ████║
#  ██████╔╝██████╔╝██║   ██║██║  ███╗██████╔╝███████║██╔████╔██║
#  ██╔═══╝ ██╔══██╗██║   ██║██║   ██║██╔══██╗██╔══██║██║╚██╔╝██║
#  ██║     ██║  ██║╚██████╔╝╚██████╔╝██║  ██║██║  ██║██║ ╚═╝ ██║
#  ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝
#
#  STRUCTURED PROGRAM ENDPOINTS
# ════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────
#  GET /program/status/:user_id
#  Returns: current position in program, full session definition,
#  psychoeducation content, exercise config.
# ────────────────────────────────────────────────────────────
@app.route('/program/status/<user_id>', methods=['GET', 'OPTIONS'])
def program_status(user_id):
    if request.method == 'OPTIONS':
        return '', 204
    try:
        progress       = get_program_progress(user_id)
        week           = progress.get("current_week", 1)
        session_number = progress.get("current_session_number", 1)
        current_part   = progress.get("current_part", 1)
        session_def    = get_program_session(week, session_number)

        # Week definition
        week_def = next((w for w in PROGRAM if w["week"] == week), None)

        # Exercise config
        ex_type   = session_def.get("exercise_type") if session_def else None
        ex_config = EXERCISE_CATALOG.get(ex_type) if ex_type else None

        # Active session document
        active_session_id = progress.get("active_session_id")
        active_session    = None
        if active_session_id:
            active_session = get_structured_session(user_id, active_session_id)

        return jsonify({
            "success":        True,
            "week":           week,
            "session_number": session_number,
            "current_part":   current_part,
            "program_complete": progress.get("program_complete", False),
            "week_definition":  week_def,
            "session_definition": session_def,
            "exercise_type":    ex_type,
            "exercise_config":  ex_config,
            "active_session":   active_session,
            "total_weeks":      len(PROGRAM),
        })
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  POST /program/start-session
#  Starts a new structured session at the user's current position.
#  Returns: session_id, session definition, opening voice message.
# ────────────────────────────────────────────────────────────
@app.route('/program/start-session', methods=['POST', 'OPTIONS'])
def program_start_session():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data    = request.get_json()
        user_id = data.get("user_id")
        if not user_id:
            return jsonify({"error": "user_id required"}), 400

        progress       = get_program_progress(user_id)
        week           = data.get("week") or progress.get("current_week", 1)
        session_number = data.get("session_number") or progress.get("current_session_number", 1)
        session_def    = get_program_session(week, session_number)

        if not session_def:
            return jsonify({"error": f"No session found for week {week}, session {session_number}"}), 404

        # Create session document
        session_doc = create_structured_session(user_id, week, session_number)
        session_id  = session_doc["session_id"]

        # Update program progress
        progress["active_session_id"] = session_id
        progress["current_part"]      = 1
        save_program_progress(user_id, progress)

        # Generate opening voice message for Part 1 (check-in)
        brain_ctx = build_rich_brain_context(user_id)
        personality = load_personality()
        week_def  = next((w for w in PROGRAM if w["week"] == week), {})

        opening_prompt = f"""
{personality}

{brain_ctx}

You are opening Week {week}, Session {session_number}: "{session_def.get('title', '')}".
Week theme: {week_def.get('theme', '')} — {week_def.get('tagline', '')}

This is PART 1 — CHECK-IN.

Greet the user warmly (this is session {progress.get('therapy', {}).get('sessions_completed', 0) + 1} overall).
Then ask them to rate their anxiety from 0 to 10 right now, and what's on their mind coming in today.
Keep it to 3–4 sentences. Natural. Unhurried.

Respond ONLY with valid JSON: {{"message": "..."}}
"""
        llm_resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": opening_prompt}],
            temperature=0.65,
            max_tokens=300,
        )
        parsed = parse_json_response(llm_resp.choices[0].message.content.strip())
        opening_msg = parsed.get("message", "Welcome. How are you feeling right now, on a scale of zero to ten?") if parsed else "Welcome. How are you feeling right now, on a scale of zero to ten?"

        audio_b64 = build_tts_response(opening_msg)

        return jsonify({
            "success":          True,
            "session_id":       session_id,
            "week":             week,
            "session_number":   session_number,
            "current_part":     1,
            "session_definition": session_def,
            "psychoeducation":  session_def.get("psychoeducation"),
            "exercise_type":    session_def.get("exercise_type"),
            "exercise_config":  EXERCISE_CATALOG.get(session_def.get("exercise_type", ""), {}),
            "opening_message":  opening_msg,
            "audio_b64":        audio_b64,
        })
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  POST /program/checkin
#  Saves Part 1 check-in score + note. Returns transition to Part 2.
#
#  Body: { user_id, session_id, score (0–10), note (optional) }
# ────────────────────────────────────────────────────────────
@app.route('/program/checkin', methods=['POST', 'OPTIONS'])
def program_checkin():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data       = request.get_json()
        user_id    = data.get("user_id")
        session_id = data.get("session_id")
        score      = data.get("score", 5)
        note       = data.get("note", "")

        if not user_id or not session_id:
            return jsonify({"error": "user_id and session_id required"}), 400

        session_doc = get_structured_session(user_id, session_id)
        if not session_doc:
            return jsonify({"error": "Session not found"}), 404

        # Save check-in
        update_structured_session(user_id, session_id, {
            "checkin_score": score,
            "checkin_note":  note,
            "current_part":  2,
        })
        save_program_progress(user_id, {"current_part": 2})

        # Build Part 2 opening — review of last commitment
        progress = get_program_progress(user_id)
        week     = session_doc.get("week", 1)
        sess_num = session_doc.get("session_number", 1)
        session_def = get_program_session(week, sess_num)
        brain_ctx   = build_rich_brain_context(user_id)
        personality = load_personality()

        # Find last session's commitment if any
        last_commitment = ""
        try:
            past = (
                db.collection("users").document(user_id)
                  .collection("structured_sessions")
                  .order_by("created_at", direction=firestore.Query.DESCENDING)
                  .limit(5).stream()
            )
            for doc in past:
                d = doc.to_dict()
                if d.get("session_id") != session_id and d.get("commitment"):
                    last_commitment = d["commitment"]
                    break
        except Exception:
            pass

        is_first_session = (week == 1 and sess_num == 1)

        part2_prompt = f"""
{personality}

{brain_ctx}

{PART_SYSTEM_PROMPTS[2]}

CONTEXT:
- Week {week}, Session {sess_num}: {session_def.get('title', '') if session_def else ''}
- User's check-in anxiety: {score}/10
- Check-in note: "{note}"
- Last session commitment: "{last_commitment}"
- First session ever: {is_first_session}
"""
        llm_resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": part2_prompt}],
            temperature=0.65,
            max_tokens=400,
        )
        raw    = llm_resp.choices[0].message.content.strip()
        parsed = parse_json_response(raw)
        review_msg = parsed.get("message", raw) if parsed else raw

        # Store as first message in session
        messages = [{"role": "assistant", "content": review_msg, "part": 2}]
        update_structured_session(user_id, session_id, {"review_messages": messages})

        audio_b64 = build_tts_response(review_msg)

        return jsonify({
            "success":       True,
            "current_part":  2,
            "review_message": review_msg,
            "audio_b64":     audio_b64,
        })
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  POST /program/review-turn
#  Continues the Part 2 review conversation (max 2 turns typically).
#  When part_complete, returns transition info to Part 3.
#
#  Body: { user_id, session_id, message }
# ────────────────────────────────────────────────────────────
@app.route('/program/review-turn', methods=['POST', 'OPTIONS'])
def program_review_turn():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data          = request.get_json()
        user_id       = data.get("user_id")
        session_id    = data.get("session_id")
        user_message  = data.get("message", "").strip()

        if not user_id or not session_id or not user_message:
            return jsonify({"error": "user_id, session_id, and message required"}), 400

        session_doc = get_structured_session(user_id, session_id)
        if not session_doc:
            return jsonify({"error": "Session not found"}), 404

        week        = session_doc.get("week", 1)
        sess_num    = session_doc.get("session_number", 1)
        session_def = get_program_session(week, sess_num)
        personality = load_personality()
        brain_ctx   = build_rich_brain_context(user_id)

        messages = session_doc.get("review_messages", [])
        messages.append({"role": "user", "content": user_message, "part": 2})

        # Build model messages (strip part tag for LLM)
        model_msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]

        system = f"""{personality}

{brain_ctx}

{PART_SYSTEM_PROMPTS[2]}

CONTEXT: Week {week}, Session {sess_num}: {session_def.get('title', '') if session_def else ''}
Turn count: {len([m for m in messages if m['role'] == 'user'])}
"""
        llm_resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system}] + model_msgs,
            temperature=0.65,
            max_tokens=400,
        )
        raw    = llm_resp.choices[0].message.content.strip()
        parsed = parse_json_response(raw)
        reply  = parsed.get("message", raw) if parsed else raw
        part_complete = (parsed or {}).get("part_complete", False)

        messages.append({"role": "assistant", "content": reply, "part": 2})

        updates = {"review_messages": messages}
        if part_complete:
            updates["review_complete"] = True
            updates["current_part"]    = 3
            save_program_progress(user_id, {"current_part": 3})

        update_structured_session(user_id, session_id, updates)

        audio_b64 = build_tts_response(reply)

        return jsonify({
            "success":      True,
            "reply":        reply,
            "part_complete": part_complete,
            "next_part":    3 if part_complete else 2,
            "audio_b64":    audio_b64,
            # If moving to Part 3, return psychoeducation content
            "psychoeducation": session_def.get("psychoeducation") if part_complete else None,
        })
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  POST /program/psychoeducation-complete
#  Frontend signals user has read/watched the psychoeducation card.
#  Returns: exercise config for Part 4.
#
#  Body: { user_id, session_id }
# ────────────────────────────────────────────────────────────
@app.route('/program/psychoeducation-complete', methods=['POST', 'OPTIONS'])
def program_psychoeducation_complete():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data       = request.get_json()
        user_id    = data.get("user_id")
        session_id = data.get("session_id")

        if not user_id or not session_id:
            return jsonify({"error": "user_id and session_id required"}), 400

        session_doc = get_structured_session(user_id, session_id)
        if not session_doc:
            return jsonify({"error": "Session not found"}), 404

        update_structured_session(user_id, session_id, {
            "psychoeducation_read": True,
            "current_part":         4,
        })
        save_program_progress(user_id, {"current_part": 4})

        week        = session_doc.get("week", 1)
        sess_num    = session_doc.get("session_number", 1)
        session_def = get_program_session(week, sess_num)
        ex_type     = session_def.get("exercise_type") if session_def else None
        ex_config   = EXERCISE_CATALOG.get(ex_type) if ex_type else None

        # Generate exercise intro speech
        intro_msg = ""
        audio_b64 = None
        if ex_config:
            personality = load_personality()
            brain_ctx   = build_rich_brain_context(user_id)
            intro_prompt = f"""
{personality}

{brain_ctx}

You are transitioning the user from the psychoeducation card to the interactive exercise.
Exercise: {ex_config['name']} — {ex_config['description']}
Message from exercise catalog: "{ex_config['config'].get('message', '')}"
Check-in anxiety score was: {session_doc.get('checkin_score', 5)}/10

Write a 2–3 sentence spoken introduction to the exercise. Warm. Calm. Specific to what they're about to do.
Respond ONLY with valid JSON: {{"message": "..."}}
"""
            llm_resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": intro_prompt}],
                temperature=0.6,
                max_tokens=200,
            )
            parsed    = parse_json_response(llm_resp.choices[0].message.content.strip())
            intro_msg = parsed.get("message", ex_config["config"].get("message", "")) if parsed else ""
            audio_b64 = build_tts_response(intro_msg) if intro_msg else None

        return jsonify({
            "success":       True,
            "current_part":  4,
            "exercise_type": ex_type,
            "exercise":      ex_config,
            "intro_message": intro_msg,
            "audio_b64":     audio_b64,
        })
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  POST /program/exercise-complete
#  User submits exercise results. Saves to exercise_results,
#  updates brain, generates Part 4 debrief, transitions to Part 5.
#
#  Body: {
#    user_id, session_id,
#    anxiety_pre, anxiety_post,
#    responses (dict), notes (str), duration_s (int)
#  }
# ────────────────────────────────────────────────────────────
@app.route('/program/exercise-complete', methods=['POST', 'OPTIONS'])
def program_exercise_complete():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data         = request.get_json()
        user_id      = data.get("user_id")
        session_id   = data.get("session_id")
        anxiety_pre  = float(data.get("anxiety_pre", 5))
        anxiety_post = float(data.get("anxiety_post", anxiety_pre))
        responses    = data.get("responses", {})
        notes        = data.get("notes", "")
        duration_s   = data.get("duration_s", 0)

        if not user_id or not session_id:
            return jsonify({"error": "user_id and session_id required"}), 400

        session_doc = get_structured_session(user_id, session_id)
        if not session_doc:
            return jsonify({"error": "Session not found"}), 404

        week        = session_doc.get("week", 1)
        sess_num    = session_doc.get("session_number", 1)
        session_def = get_program_session(week, sess_num)
        ex_type     = session_doc.get("exercise_type") or (session_def.get("exercise_type") if session_def else "unknown")
        reduction   = anxiety_pre - anxiety_post

        # Save exercise result
        result_doc = {
            "user_id":       user_id,
            "session_id":    session_id,
            "exercise_type": ex_type,
            "exercise_name": EXERCISE_CATALOG.get(ex_type, {}).get("name", ex_type),
            "anxiety_pre":   anxiety_pre,
            "anxiety_post":  anxiety_post,
            "reduction":     round(reduction, 1),
            "responses":     responses,
            "notes":         notes,
            "duration_s":    duration_s,
            "week":          week,
            "session_number": sess_num,
            "completed_at":  datetime.utcnow().isoformat(),
        }
        ref = db.collection("users").document(user_id) \
                .collection("exercise_results").document()
        ref.set(result_doc)
        result_id = ref.id

        # Update brain (background)
        threading.Thread(
            target=update_brain_after_exercise,
            args=(user_id, {**result_doc, "exercise_type": ex_type}),
            daemon=True,
        ).start()

        # Update session doc
        update_structured_session(user_id, session_id, {
            "exercise_result_id": result_id,
            "exercise_complete":  True,
            "current_part":       5,
        })
        save_program_progress(user_id, {"current_part": 5})

        # Generate debrief + Part 5 opening (commit prompt)
        personality  = load_personality()
        brain_ctx    = build_rich_brain_context(user_id)
        ex_name      = EXERCISE_CATALOG.get(ex_type, {}).get("name", ex_type)
        commit_prompt_text = session_def.get("commit_prompt", "What's one small step you'll take before we next meet?") if session_def else ""

        debrief_prompt = f"""
{personality}

{brain_ctx}

{PART_SYSTEM_PROMPTS[5]}

The user just completed the exercise: {ex_name}
- Anxiety before: {anxiety_pre}/10
- Anxiety after: {anxiety_post}/10
- Reduction: {reduction:.1f} points
- Notes: "{notes}"

First, acknowledge the exercise result specifically (don't just say "great job").
Then transition naturally into Part 5 — helping them commit to a real-world action.
The suggested commit prompt for this session is: "{commit_prompt_text}"

Respond with valid JSON:
{{"message": "...", "commitment": "", "part_complete": false}}
"""
        llm_resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": debrief_prompt}],
            temperature=0.65,
            max_tokens=500,
        )
        raw    = llm_resp.choices[0].message.content.strip()
        parsed = parse_json_response(raw)
        debrief_msg = parsed.get("message", raw) if parsed else raw

        # Store as first Part 5 message
        commit_messages = [{"role": "assistant", "content": debrief_msg, "part": 5}]
        update_structured_session(user_id, session_id, {"commit_messages": commit_messages})

        audio_b64 = build_tts_response(debrief_msg)

        return jsonify({
            "success":       True,
            "result_id":     result_id,
            "exercise_type": ex_type,
            "anxiety_pre":   anxiety_pre,
            "anxiety_post":  anxiety_post,
            "reduction":     round(reduction, 1),
            "current_part":  5,
            "debrief_message": debrief_msg,
            "audio_b64":     audio_b64,
        })
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  POST /program/commit-turn
#  Continues the Part 5 commit conversation.
#  When part_complete, finalises session and advances program position.
#
#  Body: { user_id, session_id, message }
# ────────────────────────────────────────────────────────────
@app.route('/program/commit-turn', methods=['POST', 'OPTIONS'])
def program_commit_turn():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data         = request.get_json()
        user_id      = data.get("user_id")
        session_id   = data.get("session_id")
        user_message = data.get("message", "").strip()

        if not user_id or not session_id or not user_message:
            return jsonify({"error": "user_id, session_id, and message required"}), 400

        session_doc = get_structured_session(user_id, session_id)
        if not session_doc:
            return jsonify({"error": "Session not found"}), 404

        week        = session_doc.get("week", 1)
        sess_num    = session_doc.get("session_number", 1)
        session_def = get_program_session(week, sess_num)
        personality = load_personality()
        brain_ctx   = build_rich_brain_context(user_id)

        messages = session_doc.get("commit_messages", [])
        messages.append({"role": "user", "content": user_message, "part": 5})

        model_msgs = [{"role": m["role"], "content": m["content"]} for m in messages]

        system = f"""{personality}

{brain_ctx}

{PART_SYSTEM_PROMPTS[5]}

CONTEXT: Week {week}, Session {sess_num}
Suggested commitment for this session: "{session_def.get('commit_prompt', '') if session_def else ''}"
"""
        llm_resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system}] + model_msgs,
            temperature=0.65,
            max_tokens=400,
        )
        raw    = llm_resp.choices[0].message.content.strip()
        parsed = parse_json_response(raw)
        reply           = parsed.get("message", raw) if parsed else raw
        part_complete   = (parsed or {}).get("part_complete", False)
        commitment_text = (parsed or {}).get("commitment", "")

        messages.append({"role": "assistant", "content": reply, "part": 5})

        updates = {"commit_messages": messages}

        if part_complete:
            updates["session_complete"] = True
            updates["commitment"]       = commitment_text
            updates["commitment_date"]  = datetime.utcnow().isoformat()
            updates["current_part"]     = 5

            # Advance program position
            progress    = get_program_progress(user_id)
            next_pos    = get_next_program_position(week, sess_num)
            prog_updates = {
                "active_session_id": None,
                "current_part":      1,
            }
            if next_pos.get("program_complete"):
                prog_updates["program_complete"] = True
            else:
                prog_updates["current_week"]           = next_pos["week"]
                prog_updates["current_session_number"] = next_pos["session_number"]
            save_program_progress(user_id, prog_updates)

            # Update brain (background)
            threading.Thread(
                target=update_brain_after_session,
                args=(user_id, {
                    "session_id": session_id,
                    "extracted":  {},
                    "messages":   messages,
                    "anxiety_pre": session_doc.get("checkin_score", 5),
                }),
                daemon=True,
            ).start()

        update_structured_session(user_id, session_id, updates)

        audio_b64 = build_tts_response(reply)

        return jsonify({
            "success":        True,
            "reply":          reply,
            "part_complete":  part_complete,
            "commitment":     commitment_text,
            "session_complete": part_complete,
            "program_complete": next_pos.get("program_complete", False) if part_complete else False,
            "next_week":      next_pos.get("week") if part_complete else None,
            "next_session":   next_pos.get("session_number") if part_complete else None,
            "audio_b64":      audio_b64,
        })
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  GET /program/sessions/:user_id
#  List all completed structured sessions for a user.
# ────────────────────────────────────────────────────────────
@app.route('/program/sessions/<user_id>', methods=['GET', 'OPTIONS'])
def program_session_list(user_id):
    if request.method == 'OPTIONS':
        return '', 204
    try:
        docs = (
            db.collection("users").document(user_id)
              .collection("structured_sessions")
              .order_by("created_at", direction=firestore.Query.DESCENDING)
              .limit(30).stream()
        )
        sessions = []
        for doc in docs:
            d = doc.to_dict()
            sessions.append({
                "session_id":       doc.id,
                "week":             d.get("week"),
                "session_number":   d.get("session_number"),
                "title":            d.get("title"),
                "theme":            d.get("theme"),
                "current_part":     d.get("current_part", 1),
                "session_complete": d.get("session_complete", False),
                "checkin_score":    d.get("checkin_score"),
                "commitment":       d.get("commitment"),
                "exercise_type":    d.get("exercise_type"),
                "exercise_complete": d.get("exercise_complete", False),
                "created_at":       d.get("created_at"),
            })
        return jsonify({"success": True, "sessions": sessions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  GET /program/definition
#  Returns the full 8-week program structure (for frontend navigation)
# ────────────────────────────────────────────────────────────
@app.route('/program/definition', methods=['GET'])
def program_definition():
    return jsonify({
        "success":      True,
        "total_weeks":  len(PROGRAM),
        "program":      PROGRAM,
        "exercise_catalog_keys": list(EXERCISE_CATALOG.keys()),
    })


# ════════════════════════════════════════════════════════════
#  EXISTING ENDPOINTS — FULLY PRESERVED
# ════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────
#  POST /transcribe
# ────────────────────────────────────────────────────────────
@app.route('/transcribe', methods=['POST', 'OPTIONS'])
def transcribe():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        audio_file  = request.files['audio']
        audio_bytes = audio_file.read()
        audio_name  = audio_file.filename or 'audio.webm'
        if not audio_bytes:
            return jsonify({"error": "Empty audio file"}), 400

        original_mime = (
            'audio/mp4' if 'mp4' in audio_name
            else 'audio/ogg' if 'ogg' in audio_name
            else 'audio/webm'
        )
        processed_bytes, final_mime, was_processed = preprocess_audio_for_whisper(audio_bytes, original_mime)
        final_name = "audio.wav" if was_processed else audio_name

        transcript_response = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=(final_name, io.BytesIO(processed_bytes), final_mime),
            language="en",
            prompt=WHISPER_PROMPT,
            response_format="verbose_json",
            temperature=0.0,
        )
        raw_text       = (transcript_response.text or "").strip()
        avg_confidence = compute_avg_confidence(transcript_response)

        if not raw_text:
            return jsonify({"success": False, "transcript": "", "confidence": 0.0,
                            "error_type": "empty", "user_message": "Nothing was captured. Please try again."}), 200

        if avg_confidence < WHISPER_MIN_CONFIDENCE:
            return jsonify({"success": False, "transcript": raw_text, "confidence": avg_confidence,
                            "error_type": "low_confidence",
                            "user_message": "Couldn't hear that clearly — could you say that again?"}), 200

        HALLUCINATION_PATTERNS = [
            r"^(thank you\.?|thanks\.?|you\.?|\.+|\s*)$",
            r"^(bye\.?|goodbye\.?|see you\.?)+$",
            r"^\[.*\]$",
        ]
        if any(re.match(p, raw_text.lower().strip()) for p in HALLUCINATION_PATTERNS):
            return jsonify({"success": False, "transcript": "", "confidence": avg_confidence,
                            "error_type": "hallucination",
                            "user_message": "Didn't catch that — please try again."}), 200

        return jsonify({"success": True, "transcript": raw_text,
                        "confidence": round(avg_confidence, 3), "was_preprocessed": was_processed})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  POST /speak  (single-shot TTS)
# ────────────────────────────────────────────────────────────
@app.route('/speak', methods=['POST', 'OPTIONS'])
def speak():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        if not GOOGLE_TTS_KEY:
            return jsonify({"error": "GOOGLE_TTS_KEY not configured"}), 500

        clean = clean_text_for_tts(text)
        cfg   = get_tts_config()
        final = emotion_aware_preprocess(add_thinking_filler(clean, cfg), cfg)

        response = http_requests.post(
            f"{GOOGLE_TTS_URL}?key={GOOGLE_TTS_KEY}",
            json={
                "input":       {"text": final},
                "voice":       {"languageCode": "en-US", "name": cfg["voice"]},
                "audioConfig": build_audio_config(cfg),
            },
            timeout=30
        )
        if response.status_code != 200:
            return jsonify({"error": f"Google TTS error {response.status_code}"}), 503

        audio_bytes   = base64.b64decode(response.json()["audioContent"])
        padded_audio  = append_silence(audio_bytes)
        return Response(padded_audio, mimetype="audio/mpeg",
                        headers={"Content-Type": "audio/mpeg", "Access-Control-Allow-Origin": "*"})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  POST /speak-sentences
# ────────────────────────────────────────────────────────────
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

        cfg        = get_tts_config()
        final_text = emotion_aware_preprocess(
            add_thinking_filler(clean_text_for_tts(text), cfg), cfg
        )
        sentences = [s for s in split_into_sentences(final_text) if s.strip()][:8]
        if not sentences:
            return jsonify({"sentences": []}), 200

        chunks_by_idx = {}
        with ThreadPoolExecutor(max_workers=min(len(sentences), 8)) as pool:
            future_map = {pool.submit(synthesize_sentence, s, cfg): i for i, s in enumerate(sentences)}
            for future in as_completed(future_map):
                idx    = future_map[future]
                result = future.result()
                if result:
                    chunks_by_idx[idx] = base64.b64encode(result).decode("utf-8")

        ordered = [{"index": i, "audio": chunks_by_idx[i]}
                   for i in sorted(chunks_by_idx) if i in chunks_by_idx]
        return jsonify({"sentences": ordered, "count": len(ordered)})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  POST /speak-stream
# ────────────────────────────────────────────────────────────
@app.route('/speak-stream', methods=['POST', 'OPTIONS'])
def speak_stream():
    if request.method == 'OPTIONS':
        return '', 204
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

        cfg       = get_tts_config()
        final     = emotion_aware_preprocess(add_thinking_filler(clean_sentence_for_tts(text), cfg), cfg)
        sentences = split_into_sentences(final) or [final]

        def generate():
            chunks = []
            for sentence in sentences:
                if not sentence.strip():
                    continue
                chunk = synthesize_sentence(sentence, cfg)
                if chunk:
                    chunks.append(chunk)
            if not chunks:
                return
            combined   = chunks[0] if len(chunks) == 1 else merge_mp3s(chunks)
            final_bytes = append_silence(combined)
            yield final_bytes

        return Response(
            stream_with_context(generate()),
            mimetype="audio/mpeg",
            headers={"Content-Type": "audio/mpeg", "Access-Control-Allow-Origin": "*",
                     "X-Accel-Buffering": "no"}
        )
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  POST /speak-test  (raw voice tester)
# ────────────────────────────────────────────────────────────
@app.route('/speak-test', methods=['POST', 'OPTIONS'])
def speak_test():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data           = request.get_json()
        text           = (data.get("text") or "").strip()
        override_voice = (data.get("voice") or "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        if not GOOGLE_TTS_KEY:
            return jsonify({"error": "GOOGLE_TTS_KEY not configured"}), 500

        cfg = get_tts_config()
        if override_voice:
            cfg["voice"] = override_voice

        sentences = [s for s in (split_into_sentences(text) or [text]) if s.strip()]
        chunks_by_index = {}
        with ThreadPoolExecutor(max_workers=min(len(sentences), 8)) as pool:
            future_to_idx = {pool.submit(synthesize_sentence, sentence, cfg): i
                             for i, sentence in enumerate(sentences)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                result = future.result()
                if result:
                    chunks_by_index[idx] = result

        chunks = [chunks_by_index[i] for i in sorted(chunks_by_index)]
        if not chunks:
            return jsonify({"error": "TTS produced no audio"}), 500

        combined = chunks[0] if len(chunks) == 1 else merge_mp3s(chunks)
        return Response(combined, mimetype="audio/mpeg",
                        headers={"Content-Type": "audio/mpeg",
                                 "Access-Control-Allow-Origin": "*",
                                 "X-Voice-Used": cfg["voice"]})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  GET /voices
# ────────────────────────────────────────────────────────────
@app.route('/voices', methods=['GET'])
def list_voices():
    cfg = get_tts_config()
    return jsonify({"current_voice": cfg["voice"], "available_voices": AVAILABLE_VOICES})


# ────────────────────────────────────────────────────────────
#  GET /voice-test-ui
# ────────────────────────────────────────────────────────────
@app.route('/voice-test-ui', methods=['GET'])
def voice_test_ui():
    voices_json = json.dumps(AVAILABLE_VOICES)
    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Voice Tester — Theraply</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap');
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#0d0f14;--surface:#13161e;--border:#1f2430;--accent:#7c6af7;--accent2:#a78bfa;--text:#e8e6f0;--muted:#6b6880;--success:#4ade80;--warn:#fb923c}}
body{{background:var(--bg);color:var(--text);font-family:'DM Mono',monospace;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:48px 24px 80px}}
header{{text-align:center;margin-bottom:48px}}
header h1{{font-family:'DM Serif Display',serif;font-size:clamp(2rem,5vw,3.2rem);font-style:italic;background:linear-gradient(135deg,var(--accent2),var(--accent));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-0.02em}}
header p{{color:var(--muted);font-size:0.8rem;margin-top:8px;letter-spacing:0.08em;text-transform:uppercase}}
.card{{width:100%;max-width:760px;background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:32px;margin-bottom:24px}}
label{{display:block;font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}}
select,textarea{{width:100%;background:var(--bg);border:1px solid var(--border);border-radius:10px;color:var(--text);font-family:'DM Mono',monospace;font-size:0.88rem;padding:14px 16px;outline:none;transition:border-color 0.2s;appearance:none}}
select:focus,textarea:focus{{border-color:var(--accent)}}
select{{cursor:pointer;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' fill='none'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%236b6880' stroke-width='1.5' stroke-linecap='round'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 16px center;padding-right:40px}}
textarea{{height:200px;resize:vertical;line-height:1.7}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px}}
.field{{margin-bottom:24px}}.field:last-child{{margin-bottom:0}}
button{{width:100%;padding:16px;border:none;border-radius:12px;font-family:'DM Mono',monospace;font-size:0.9rem;letter-spacing:0.06em;cursor:pointer;transition:all 0.2s}}
#playBtn{{background:linear-gradient(135deg,var(--accent),#5b4fd4);color:#fff;font-weight:500}}
#playBtn:hover:not(:disabled){{transform:translateY(-2px);box-shadow:0 8px 32px rgba(124,106,247,0.4)}}
#playBtn:disabled{{opacity:0.5;cursor:not-allowed;transform:none}}
.status-bar{{display:flex;align-items:center;gap:10px;padding:14px 18px;border-radius:10px;font-size:0.8rem;margin-top:20px;border:1px solid var(--border);min-height:50px;transition:all 0.3s}}
.status-bar.idle{{background:transparent;color:var(--muted)}}.status-bar.loading{{background:rgba(124,106,247,0.08);color:var(--accent2);border-color:var(--accent)}}.status-bar.playing{{background:rgba(74,222,128,0.08);color:var(--success);border-color:var(--success)}}.status-bar.error{{background:rgba(251,146,60,0.08);color:var(--warn);border-color:var(--warn)}}
.dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0;background:currentColor}}.dot.pulse{{animation:pulse 1.2s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:0.4;transform:scale(0.7)}}}}
audio{{width:100%;margin-top:16px;border-radius:8px}}
</style></head><body>
<header><h1>Voice Tester</h1><p>Theraply · Raw TTS Preview</p></header>
<div class="card">
  <div class="row">
    <div class="field"><label>Voice</label><select id="voiceSelect"></select></div>
    <div class="field"><label>Preset Phrases</label>
      <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:12px">
        <button style="width:auto;padding:6px 12px;background:var(--bg);border:1px solid var(--border);color:var(--muted);border-radius:8px;font-size:0.72rem" onclick="setPreset('open')">Therapy opener</button>
        <button style="width:auto;padding:6px 12px;background:var(--bg);border:1px solid var(--border);color:var(--muted);border-radius:8px;font-size:0.72rem" onclick="setPreset('reframe')">CBT reframe</button>
      </div>
    </div>
  </div>
  <div class="field"><label>Text to speak</label><textarea id="textInput" placeholder="Type or paste any text here."></textarea></div>
  <button id="playBtn" onclick="synthesise()">▶ Synthesise &amp; Play</button>
  <div class="status-bar idle" id="statusBar"><div class="dot"></div><span id="statusText">Ready</span></div>
  <audio id="player" controls style="display:none"></audio>
</div>
<script>
const VOICES={voices_json};
const PRESETS={{
  open:"So you've been carrying this for a while... I can hear it. What's the one thing that's been sitting heaviest on your chest this week, yeah?",
  reframe:"Right, so here's what I'm noticing. That thought — that you'll definitely fail — it's loud. But loudness isn't the same as truth. What would you say to a friend who told you that exact thing?"
}};
const sel=document.getElementById('voiceSelect');
VOICES.forEach(v=>{{const o=document.createElement('option');o.value=v;o.textContent=v.replace('en-US-','');sel.appendChild(o)}});
fetch('/voices').then(r=>r.json()).then(d=>{{sel.value=d.current_voice}}).catch(()=>{{}});
function setPreset(k){{document.getElementById('textInput').value=PRESETS[k]||''}}
function setStatus(s,m){{const b=document.getElementById('statusBar');b.className='status-bar '+s;document.getElementById('statusText').textContent=m;b.querySelector('.dot').className='dot'+(s==='loading'?' pulse':'')}}
async function synthesise(){{
  const text=document.getElementById('textInput').value.trim();
  if(!text){{setStatus('error','Please enter some text first.');return}}
  const voice=sel.value;const btn=document.getElementById('playBtn');const player=document.getElementById('player');
  btn.disabled=true;player.style.display='none';setStatus('loading','Sending to Google TTS...');
  try{{
    const res=await fetch('/speak-test',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{text,voice}})}});
    if(!res.ok)throw new Error('HTTP '+res.status);
    const blob=await res.blob();const url=URL.createObjectURL(blob);
    player.src=url;player.style.display='block';player.play();setStatus('playing','Playing — '+voice.replace('en-US-',''));
    player.onended=()=>setStatus('idle','Done');
  }}catch(e){{setStatus('error','Error: '+e.message)}}finally{{btn.disabled=false}}
}}
document.getElementById('textInput').addEventListener('keydown',e=>{{if(e.ctrlKey&&e.key==='Enter')synthesise()}});
</script></body></html>"""
    return Response(html, mimetype="text/html")


# ────────────────────────────────────────────────────────────
#  POST /therapy-session  (original free-form session, preserved)
# ────────────────────────────────────────────────────────────
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
            personality  = load_personality()
            brain_context = build_rich_brain_context(user_id)
            full_system  = personality + "\n\n" + THERAPY_SYSTEM_PROMPT + "\n\n" + brain_context
            messages     = [{"role": "system", "content": full_system}]
        messages.append({"role": "user", "content": user_message})

        current_phase    = session_data.get("phase", 1)
        extracted_so_far = session_data.get("extracted", {})

        length_guide = {
            "short":  "RESPONSE LENGTH: SHORT — 1 to 2 sentences max. One thought only.",
            "medium": "RESPONSE LENGTH: MEDIUM — 2 to 3 sentences.",
            "long":   (
                "RESPONSE LENGTH: LONG — 5 to 7 sentences. Validate first, then reflect, "
                "then gently explore or reframe, then close with a question. Take your time."
            ),
        }.get(response_length, "RESPONSE LENGTH: LONG — 5 to 7 sentences.")
        length_tokens = {"short": 200, "medium": 400, "long": 1200}.get(response_length, 1200)

        phase_reminder = {
            "role": "system",
            "content": (
                f"CURRENT PHASE: {current_phase}\n"
                f"EXTRACTED SO FAR: {json.dumps(extracted_so_far)}\n"
                f"{length_guide}\n"
                "Respond with valid JSON only."
            )
        }
        messages_for_model = [messages[0], phase_reminder] + messages[1:]

        llm_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages_for_model,
            temperature=0.65,
            max_tokens=length_tokens
        )
        raw_reply = llm_response.choices[0].message.content.strip()
        parsed    = parse_json_response(raw_reply)

        if not parsed:
            messages.append({"role": "assistant", "content": raw_reply})
            db.collection("users").document(user_id) \
              .collection("therapy_sessions").document(session_id) \
              .update({"messages": messages})
            return jsonify({
                "session_id": session_id, "reply": raw_reply,
                "phase": current_phase, "session_complete": False, "audio_b64": None,
            })

        ai_reply         = parsed.get("message", raw_reply)
        next_phase       = parsed.get("phase", current_phase)
        session_complete = parsed.get("session_complete", False)
        new_extracted    = parsed.get("extracted", {})

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
        tone, pause_ms = detect_reply_tone(ai_reply)

        cfg        = get_tts_config()
        clean      = clean_text_for_tts(ai_reply)
        filled     = add_thinking_filler(clean, cfg)
        final_text = emotion_aware_preprocess(filled, cfg)
        sentences  = [s for s in split_into_sentences(final_text) if s.strip()][:8]

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

        with ThreadPoolExecutor(max_workers=2) as outer_pool:
            fs_future  = outer_pool.submit(_firestore_write)
            tts_future = outer_pool.submit(_synthesise_all, sentences, cfg)
            try:
                fs_future.result(timeout=8)
            except Exception as fe:
                print(f"[therapy-session] Firestore write failed: {fe}")
            try:
                audio_chunks = tts_future.result(timeout=15)
                if audio_chunks:
                    audio_b64 = [base64.b64encode(c).decode("utf-8") for c in audio_chunks]
            except Exception as te:
                print(f"[therapy-session] TTS failed: {te}")

        if session_complete:
            threading.Thread(
                target=update_brain_after_session,
                args=(user_id, {
                    "session_id": session_id,
                    "extracted":  merged,
                    "messages":   messages,
                }),
                daemon=True,
            ).start()

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
            "audio_b64":        audio_b64,
        })
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  POST /session-to-plan
# ────────────────────────────────────────────────────────────
@app.route('/session-to-plan', methods=['POST', 'OPTIONS'])
def session_to_plan():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data       = request.get_json()
        user_id    = data.get("user_id")
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

        plan.update({
            "session_id": session_id,
            "created_at": int(datetime.now().timestamp() * 1000),
            "completed":  False,
            "source":     "therapy_session",
        })
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


# ────────────────────────────────────────────────────────────
#  POST /therapy-session/history
# ────────────────────────────────────────────────────────────
@app.route('/therapy-session/history', methods=['POST', 'OPTIONS'])
@app.route('/api/therapy-session-history', methods=['POST', 'OPTIONS'])
def therapy_session_history():
    if request.method == 'OPTIONS':
        return '', 204
    data    = request.get_json()
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
                "session_id":       doc.id,
                "created_at":       d.get("created_at"),
                "session_complete": d.get("session_complete", False),
                "phase":            d.get("phase", 1),
                "plan_id":          d.get("plan_id"),
                "situation":        d.get("extracted", {}).get("situation", ""),
                "task_name":        d.get("extracted", {}).get("proposed_task", {}).get("name", ""),
                "turn_count":       len([m for m in d.get("messages", []) if m.get("role") == "user"])
            })
        return jsonify({"success": True, "sessions": sessions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  Brain endpoints
# ────────────────────────────────────────────────────────────
@app.route('/brain/<user_id>', methods=['GET', 'OPTIONS'])
def get_brain_endpoint(user_id):
    if request.method == 'OPTIONS':
        return '', 204
    return jsonify({"success": True, "brain": get_brain_v2(user_id)})


@app.route('/brain/<user_id>', methods=['POST', 'OPTIONS'])
def update_brain_endpoint(user_id):
    if request.method == 'OPTIONS':
        return '', 204
    data  = request.get_json() or {}
    brain = get_brain_v2(user_id)
    for dotpath, value in data.items():
        parts  = dotpath.split(".")
        target = brain
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    save_brain_v2(user_id, brain)
    return jsonify({"success": True, "brain": brain})


# ────────────────────────────────────────────────────────────
#  Exercise endpoints (standalone — for ad-hoc use outside program)
# ────────────────────────────────────────────────────────────
@app.route('/exercise/catalog', methods=['GET'])
def exercise_catalog_endpoint():
    summary = {
        k: {"name": v["name"], "description": v["description"],
            "best_for": v["best_for"], "duration_s": v["duration_s"]}
        for k, v in EXERCISE_CATALOG.items()
    }
    return jsonify({"success": True, "exercises": summary})


@app.route('/exercise/prescribe', methods=['POST', 'OPTIONS'])
def exercise_prescribe():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data        = request.get_json()
        user_id     = data.get("user_id")
        session_id  = data.get("session_id")
        manual_type = data.get("exercise_type")
        manual_anx  = data.get("anxiety", 5)
        manual_sit  = data.get("situation", "")
        if not user_id:
            return jsonify({"error": "user_id required"}), 400

        session_ctx = ""
        if session_id:
            try:
                doc = db.collection("users").document(user_id) \
                        .collection("therapy_sessions").document(session_id).get()
                if doc.exists:
                    ext = doc.to_dict().get("extracted", {})
                    session_ctx = (
                        f"Situation: {ext.get('situation', manual_sit)}\n"
                        f"Anxious thought: {ext.get('anxious_thought', '')}\n"
                        f"Emotion: {ext.get('emotion', '')}\n"
                        f"Phase: {doc.to_dict().get('phase', 1)}\n"
                        f"Anxiety estimate: {ext.get('proposed_task', {}).get('anxiety_pre', manual_anx)}"
                    )
            except Exception:
                pass

        if not session_ctx:
            session_ctx = f"Situation: {manual_sit}\nAnxiety: {manual_anx}/10\nNo active session context."

        brain_ctx = build_rich_brain_context(user_id)

        if manual_type and manual_type in EXERCISE_CATALOG:
            ex_type   = manual_type
            rationale = "Manually selected."
            intro     = f"I'd like us to try a {EXERCISE_CATALOG[ex_type]['name']} exercise."
            anx_pre   = manual_anx
        else:
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
                temperature=0.4, max_tokens=300,
            )
            parsed = parse_json_response(llm_resp.choices[0].message.content.strip())
            if not parsed or parsed.get("exercise_type") not in EXERCISE_CATALOG:
                ex_type = "breathing_box"; rationale = "Default fallback."
                intro   = "Let's start with some breathing."; anx_pre = manual_anx
            else:
                ex_type   = parsed["exercise_type"]
                rationale = parsed.get("rationale", "")
                intro     = parsed.get("custom_intro", "")
                anx_pre   = parsed.get("anxiety_pre_estimate", manual_anx)

        audio_b64 = None
        if GOOGLE_TTS_KEY and intro:
            try:
                cfg   = get_tts_config()
                audio = synthesize_sentence(clean_text_for_tts(intro), cfg)
                if audio:
                    audio_b64 = base64.b64encode(audio).decode("utf-8")
            except Exception:
                pass

        return jsonify({
            "success":       True,
            "exercise_type": ex_type,
            "exercise":      EXERCISE_CATALOG[ex_type],
            "rationale":     rationale,
            "intro_speech":  intro,
            "anxiety_pre":   anx_pre,
            "audio_b64":     audio_b64,
            "session_id":    session_id,
            "user_id":       user_id,
        })
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/exercise/complete', methods=['POST', 'OPTIONS'])
def exercise_complete():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data          = request.get_json()
        user_id       = data.get("user_id")
        session_id    = data.get("session_id")
        ex_type       = data.get("exercise_type", "unknown")
        anxiety_pre   = float(data.get("anxiety_pre",  5))
        anxiety_post  = float(data.get("anxiety_post", anxiety_pre))
        responses     = data.get("responses",  {})
        notes         = data.get("notes", "")
        duration_s    = data.get("duration_s", 0)
        continue_sess = data.get("continue_session", False)
        if not user_id:
            return jsonify({"error": "user_id required"}), 400

        reduction  = anxiety_pre - anxiety_post
        result_doc = {
            "user_id": user_id, "session_id": session_id or "",
            "exercise_type": ex_type,
            "exercise_name": EXERCISE_CATALOG.get(ex_type, {}).get("name", ex_type),
            "anxiety_pre": anxiety_pre, "anxiety_post": anxiety_post,
            "reduction": round(reduction, 1),
            "responses": responses, "notes": notes, "duration_s": duration_s,
            "completed_at": datetime.utcnow().isoformat(),
        }
        ref = db.collection("users").document(user_id).collection("exercise_results").document()
        ref.set(result_doc)
        result_id = ref.id

        threading.Thread(
            target=update_brain_after_exercise,
            args=(user_id, {**result_doc, "exercise_type": ex_type}),
            daemon=True,
        ).start()

        therapist_reply = None
        audio_b64       = None

        if continue_sess and session_id:
            try:
                doc = db.collection("users").document(user_id) \
                        .collection("therapy_sessions").document(session_id).get()
                if doc.exists:
                    session_data = doc.to_dict()
                    messages     = session_data.get("messages", [])
                    ex_name      = EXERCISE_CATALOG.get(ex_type, {}).get("name", ex_type)

                    def _summarise(et, r):
                        pre = r.get("anxiety_pre", 5); post = r.get("anxiety_post", 5)
                        change = pre - post
                        direction = (f"anxiety dropped from {pre} to {post}" if change > 0
                                     else f"anxiety unchanged at {post}" if change == 0
                                     else f"anxiety increased from {pre} to {post}")
                        return f"The user completed {et}. Their {direction}."

                    followup_msg = f"[Exercise completed: {ex_name}]\n{_summarise(ex_type, result_doc)}"
                    messages.append({"role": "user", "content": followup_msg})

                    follow_prompt = {
                        "role": "system",
                        "content": (
                            "The user just completed an exercise. Respond warmly. "
                            "Acknowledge their anxiety shift specifically. "
                            "RESPONSE LENGTH: MEDIUM — 3 to 4 sentences. "
                            "Return valid JSON: {\"message\": \"...\", \"phase\": N, "
                            "\"session_complete\": false, \"extracted\": {}}"
                        ),
                    }
                    llm_resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[messages[0], follow_prompt] + messages[1:],
                        temperature=0.65, max_tokens=500,
                    )
                    raw    = llm_resp.choices[0].message.content.strip()
                    parsed = parse_json_response(raw)
                    therapist_reply = parsed.get("message", raw) if parsed else raw
                    messages.append({"role": "assistant", "content": therapist_reply})
                    db.collection("users").document(user_id) \
                      .collection("therapy_sessions").document(session_id) \
                      .update({"messages": messages})

                    audio_b64 = build_tts_response(therapist_reply)
            except Exception as e:
                print(f"[exercise_complete] follow-up failed: {e}")

        return jsonify({
            "success": True, "result_id": result_id,
            "exercise_type": ex_type, "anxiety_pre": anxiety_pre,
            "anxiety_post": anxiety_post, "reduction": round(reduction, 1),
            "therapist_reply": therapist_reply, "audio_b64": audio_b64,
        })
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


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


# ────────────────────────────────────────────────────────────
#  GET /tts-config  /  POST /tts-config
# ────────────────────────────────────────────────────────────
@app.route('/tts-config', methods=['GET', 'POST', 'OPTIONS'])
def tts_config():
    if request.method == 'OPTIONS':
        return '', 204
    if request.method == 'GET':
        return jsonify({"success": True, "config": get_tts_config()})
    try:
        updates = request.get_json() or {}
        if updates.get("action") == "reset_defaults":
            db.collection("config").document("tts_settings").delete()
            bootstrap_tts_config()
            return jsonify({"success": True, "action": "reset_defaults", "current_config": get_tts_config()})
        allowed  = set(TTS_DEFAULTS.keys())
        filtered = {k: v for k, v in updates.items() if k in allowed}
        if not filtered:
            return jsonify({"error": f"No valid fields. Allowed: {sorted(allowed)}"}), 400
        db.collection("config").document("tts_settings").set(filtered, merge=True)
        return jsonify({"success": True, "updated": filtered, "current_config": get_tts_config()})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  GET /health
# ────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    cfg = get_tts_config()
    return jsonify({
        "status": "ok",
        "tts_provider": "Google Cloud TTS",
        "tts_voice":    cfg["voice"],
        "tts_key_set":  bool(GOOGLE_TTS_KEY),
        "program_weeks": len(PROGRAM),
        "exercise_types": len(EXERCISE_CATALOG),
        "features": [
            "8-week structured CBT program",
            "5-part session arc (check-in → review → psychoeducation → exercise → commit)",
            "17 interactive exercise types",
            "brain v2 persistent memory",
            "Google Cloud TTS — Chirp HD",
            "Whisper STT with confidence gating",
        ]
    })


# ────────────────────────────────────────────────────────────
#  GET /
# ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return jsonify({
        "status": "Theraply backend running",
        "endpoints": {
            # ── Program (new) ──────────────────────────────────────
            "GET  /program/status/:user_id":          "Current position + full session config",
            "POST /program/start-session":            "Create session, get opening voice message",
            "POST /program/checkin":                  "Save Part 1 score, get Part 2 review opener",
            "POST /program/review-turn":              "Continue Part 2 conversation",
            "POST /program/psychoeducation-complete": "Mark Part 3 done, get exercise intro",
            "POST /program/exercise-complete":        "Save exercise results, get Part 5 commit opener",
            "POST /program/commit-turn":              "Continue Part 5, finalise session on completion",
            "GET  /program/sessions/:user_id":        "List structured sessions",
            "GET  /program/definition":               "Full 8-week program definition",
            # ── Legacy / standalone ───────────────────────────────
            "POST /therapy-session":                  "Free-form CBT session (legacy)",
            "POST /session-to-plan":                  "Convert session to activity",
            "POST /therapy-session/history":          "Past free-form sessions",
            "GET  /brain/:user_id":                   "User memory",
            "POST /brain/:user_id":                   "Update user memory",
            "GET  /exercise/catalog":                 "All exercise types",
            "POST /exercise/prescribe":               "LLM picks exercise",
            "POST /exercise/complete":                "Submit standalone exercise",
            "GET  /exercise/history/:user_id":        "Past exercise results",
            # ── Audio ─────────────────────────────────────────────
            "POST /transcribe":                       "Whisper STT",
            "POST /speak":                            "Single-shot TTS",
            "POST /speak-sentences":                  "Parallel sentence TTS",
            "POST /speak-stream":                     "Streaming TTS",
            "POST /speak-test":                       "Raw voice tester",
            "GET  /voice-test-ui":                    "Browser voice tester UI",
            "GET  /voices":                           "Available voices",
            "GET  /tts-config":                       "TTS settings",
            "POST /tts-config":                       "Update TTS settings",
        }
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
