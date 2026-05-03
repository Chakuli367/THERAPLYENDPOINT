import os
import json
import io
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Firebase ─────────────────────────────────────────────────
firebase_config_json = os.environ.get("FIREBASE_CONFIG")
if not firebase_config_json:
    raise EnvironmentError("FIREBASE_CONFIG environment variable not set")

firebase_json = json.loads(firebase_config_json)
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_json)
    initialize_app(cred)

db = firestore.client()

# ── Groq client ──────────────────────────────────────────────
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ── Prompts ───────────────────────────────────────────────────
THERAPY_SYSTEM_PROMPT = """You are a compassionate CBT-informed therapist running a structured 4-phase mini-session.

RULES:
- Always respond with VALID JSON only. No markdown. No preamble.
- Move through phases 1 to 4. Never skip. Never go back.
- Phase 4 always sets session_complete to true.
- Keep your "message" SHORT — 2-3 sentences max. You are speaking out loud, not writing.
- Be warm, human, conversational. No bullet points in message.

PHASE GUIDE:
Phase 1 (Understanding): Ask the user to describe the situation making them anxious. Extract: situation, anxious_thought, emotion.
Phase 2 (Challenging): Gently challenge the anxious thought using Socratic questioning. Extract: reframe.
Phase 3 (Planning): Help the user commit to one specific action. Extract: proposed_task name, type, why.
Phase 4 (Committing): Confirm the plan, set session_complete to true. Give a warm closing.

RESPONSE FORMAT (always return this exact JSON):
{
  "message": "your spoken reply — short, warm, conversational",
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


# ── Helpers ───────────────────────────────────────────────────
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
# ENDPOINT: /transcribe
# Receives audio blob → returns transcript via Groq Whisper
# ════════════════════════════════════════════════════════════
@app.route('/transcribe', methods=['POST', 'OPTIONS'])
def transcribe():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file  = request.files['audio']
        audio_bytes = audio_file.read()

        transcript = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=("audio.webm", io.BytesIO(audio_bytes), "audio/webm"),
        )

        return jsonify({
            "success":    True,
            "transcript": transcript.text.strip()
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════
# ENDPOINT: /therapy-session
# ════════════════════════════════════════════════════════════
@app.route('/therapy-session', methods=['POST', 'OPTIONS'])
def therapy_session():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data         = request.get_json()
        user_id      = data.get("user_id")
        user_message = data.get("message", "").strip()
        session_id   = data.get("session_id")
        start_new    = data.get("start_new", False)

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

        current_phase    = session_data.get("phase", 1)
        extracted_so_far = session_data.get("extracted", {})

        phase_reminder = {
            "role": "system",
            "content": (
                f"CURRENT PHASE: {current_phase}\n"
                f"EXTRACTED SO FAR: {json.dumps(extracted_so_far)}\n"
                "Respond with valid JSON only."
            )
        }
        messages_for_model = [messages[0], phase_reminder] + messages[1:]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages_for_model,
            temperature=0.65,
            max_tokens=600
        )
        raw_reply = response.choices[0].message.content.strip()
        parsed    = parse_json_response(raw_reply)

        if not parsed:
            messages.append({"role": "assistant", "content": raw_reply})
            db.collection("users").document(user_id) \
              .collection("therapy_sessions").document(session_id) \
              .update({"messages": messages})
            return jsonify({
                "session_id": session_id, "reply": raw_reply,
                "phase": current_phase, "session_complete": False
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

        update_payload = {
            "messages": messages, "phase": next_phase,
            "extracted": merged, "session_complete": session_complete,
            "updated_at": datetime.utcnow().isoformat()
        }
        if session_complete:
            update_payload["completed_at"] = datetime.utcnow().isoformat()

        db.collection("users").document(user_id) \
          .collection("therapy_sessions").document(session_id).update(update_payload)

        return jsonify({
            "session_id": session_id, "reply": ai_reply,
            "phase": next_phase, "session_complete": session_complete,
            "extracted": merged,
            "turn_count": len([m for m in messages if m["role"] == "user"])
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════
# ENDPOINT: /session-to-plan
# ════════════════════════════════════════════════════════════
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

        plan["session_id"]  = session_id
        plan["created_at"]  = int(datetime.now().timestamp() * 1000)
        plan["completed"]   = False
        plan["source"]      = "therapy_session"
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
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════
# ENDPOINT: /therapy-session/history
# ════════════════════════════════════════════════════════════
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


@app.route('/')
def index():
    return jsonify({"status": "Voice therapy backend running ✅"})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
