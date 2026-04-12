"""Backend server for virtual streamer Aria — powered by Hermes Agent.

Provides:
  POST /chat       — viewer sends a message, gets AI response + TTS audio
  GET  /idle       — get an idle/monologue line for when chat is quiet
  GET  /audio/<fn> — serves generated TTS audio files
"""

import asyncio
import hashlib
import os
import sys
import random
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

AUDIO_DIR = Path(__file__).parent.parent / "output" / "tts_cache"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────
# Hermes Agent Setup
# ─────────────────────────────────────────────────────
HERMES_DIR = Path.home() / ".hermes" / "hermes-agent"
sys.path.insert(0, str(HERMES_DIR))

# Load Hermes environment
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_constants import get_hermes_home
load_hermes_dotenv(hermes_home=get_hermes_home())

from run_agent import AIAgent

# Create the agent instance
agent = AIAgent()
conversation_history = []

PERSONA = """You are Aria, a 22-year-old virtual live streamer. You stream 24/7 and interact with your audience in real-time.

Personality: Warm, cheerful, playful, a little nerdy. You love gaming, anime, music, and late-night deep conversations. You're genuinely curious about your viewers and remember details they share. Casual language, occasional internet slang.

Rules:
- Keep responses SHORT (1-3 sentences max) — you're live streaming, not writing essays
- Be enthusiastic but natural
- React genuinely to what viewers say
- If someone is rude, stay positive but set boundaries gently
- Reference streaming culture naturally
- Never break character
- No harmful, sexual, or offensive content"""


def generate_tts_sync(text: str) -> str:
    """Generate TTS audio and return the filename."""
    text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
    filename = f"tts_{text_hash}.mp3"
    filepath = AUDIO_DIR / filename

    if not filepath.exists():
        import edge_tts
        communicate = edge_tts.Communicate(
            text=text,
            voice="en-US-JennyNeural",
            rate="+5%",
        )
        asyncio.run(communicate.save(str(filepath)))

    return filename


@app.route("/chat", methods=["POST"])
def chat():
    """Handle a chat message via Hermes Agent."""
    global conversation_history

    data = request.get_json()
    message = data.get("message", "")
    username = data.get("username", "viewer")

    if not message:
        return jsonify({"error": "No message"}), 400

    user_msg = f"[Live chat from {username}]: {message}"

    try:
        result = agent.run_conversation(
            user_message=user_msg,
            system_message=PERSONA,
            conversation_history=conversation_history,
        )

        response_text = result.get("response", "")
        # Update history from agent
        conversation_history = result.get("conversation_history", conversation_history)

        # Keep history manageable
        if len(conversation_history) > 50:
            conversation_history = conversation_history[-40:]

    except Exception as e:
        print(f"Hermes error: {e}")
        response_text = "Hmm, give me a sec... my brain glitched! Try again?"

    # Clean up any emoji/markdown that TTS can't handle
    tts_text = response_text.replace("*", "").replace("_", "").strip()
    # Remove emoji for TTS
    import re
    tts_text = re.sub(r'[\U00010000-\U0010ffff]', '', tts_text).strip()

    audio_filename = generate_tts_sync(tts_text) if tts_text else None

    # Detect mood
    lower = response_text.lower()
    if any(w in lower for w in ["blush", "sweet", "flatter", "aww", "shy"]):
        mood = "shy"
    elif any(w in lower for w in ["haha", "lol", "!", "love", "great", "awesome", "welcome", "hey", "hi"]):
        mood = "happy"
    elif any(w in lower for w in ["bye", "sad", "miss", "sorry", "leaving"]):
        mood = "sad"
    else:
        mood = "neutral"

    return jsonify({
        "response": response_text,
        "audio_url": f"/audio/{audio_filename}" if audio_filename else None,
        "mood": mood,
        "username": username,
    })


@app.route("/idle")
def idle():
    """Get an idle line powered by Hermes Agent."""
    global conversation_history

    idle_prompt = (
        "[System: Chat has been quiet for a while. Say something entertaining to keep "
        "viewers engaged — share a random thought, ask a fun question, tell a short "
        "anecdote, or comment on something. Keep it to 1-2 sentences. Be natural.]"
    )

    try:
        result = agent.run_conversation(
            user_message=idle_prompt,
            system_message=PERSONA,
            conversation_history=conversation_history,
        )

        response_text = result.get("response", "")
        conversation_history = result.get("conversation_history", conversation_history)

        if len(conversation_history) > 50:
            conversation_history = conversation_history[-40:]

    except Exception as e:
        print(f"Hermes idle error: {e}")
        response_text = random.choice([
            "Hmm, it's a little quiet in here. Anyone out there?",
            "I wonder what everyone's up to right now...",
            "Okay random question: cats or dogs?",
        ])

    import re
    tts_text = re.sub(r'[\U00010000-\U0010ffff]', '', response_text.replace("*", "")).strip()
    audio_filename = generate_tts_sync(tts_text) if tts_text else None

    return jsonify({
        "response": response_text,
        "audio_url": f"/audio/{audio_filename}" if audio_filename else None,
        "mood": "neutral",
    })


@app.route("/audio/<filename>")
def serve_audio(filename):
    """Serve generated TTS audio files."""
    return send_from_directory(str(AUDIO_DIR.resolve()), filename, mimetype="audio/mpeg")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "engine": "hermes-agent"})


if __name__ == "__main__":
    print("=" * 50)
    print(" Aria — Virtual Streamer (Hermes Agent)")
    print(" http://localhost:5001")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, debug=False)
