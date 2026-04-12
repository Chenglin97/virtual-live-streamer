#!/usr/bin/env python3
"""Bridge to Hermes Agent — runs in Hermes's venv, exposes HTTP API.

Start this with Hermes's Python:
  ~/.hermes/hermes-agent/venv/bin/python src/hermes_bridge.py
"""

import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Add Hermes to path
HERMES_DIR = Path.home() / ".hermes" / "hermes-agent"
sys.path.insert(0, str(HERMES_DIR))

# Load Hermes env
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_constants import get_hermes_home
load_hermes_dotenv(hermes_home=get_hermes_home())

from run_agent import AIAgent

# Flask is not in Hermes venv, use stdlib http server instead
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

AUDIO_DIR = Path(__file__).parent.parent / "output" / "tts_cache"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

agent = AIAgent(
    base_url="https://open.palebluedot.ai/v1",
    model="anthropic/claude-opus-4.6",
    api_key="sk-YuA06jNFuzJoyaJha8C4CYOrvECdGQ1LtkEEESfUkuT6UY6d",
)
conversation_history = []

PERSONA = """You are Aria, a 22-year-old virtual live streamer. You stream 24/7 and interact with your audience in real-time.

Personality: Warm, cheerful, playful, a little nerdy. You love gaming, anime, music, and late-night deep conversations. You're genuinely curious about your viewers and remember details they share. Casual language, occasional internet slang.

Rules:
- Keep responses SHORT (1-3 sentences max) — you're live streaming
- Be enthusiastic but natural
- React genuinely to what viewers say
- Reference streaming culture naturally
- Never break character"""


def generate_tts(text: str) -> str | None:
    """Generate TTS audio, return filename."""
    try:
        import edge_tts
    except ImportError:
        return None

    clean = re.sub(r'[\U00010000-\U0010ffff]', '', text.replace("*", "")).strip()
    if not clean:
        return None

    text_hash = hashlib.md5(clean.encode()).hexdigest()[:12]
    filename = f"tts_{text_hash}.mp3"
    filepath = AUDIO_DIR / filename

    if not filepath.exists():
        communicate = edge_tts.Communicate(text=clean, voice="en-US-JennyNeural", rate="+5%")
        asyncio.run(communicate.save(str(filepath)))

    return filename


def detect_mood(text: str) -> str:
    lower = text.lower()
    if any(w in lower for w in ["blush", "sweet", "flatter", "aww"]):
        return "shy"
    elif any(w in lower for w in ["haha", "lol", "!", "love", "great", "awesome", "welcome"]):
        return "happy"
    elif any(w in lower for w in ["bye", "sad", "sorry", "leaving"]):
        return "sad"
    return "neutral"


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, data, status=200):
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._json_response({"status": "ok", "engine": "hermes-agent"})

        elif self.path == "/idle":
            self._handle_idle()

        elif self.path.startswith("/audio/"):
            self._serve_audio(self.path[7:])

        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._handle_chat(body)
        else:
            self._json_response({"error": "not found"}, 404)

    def _handle_chat(self, body):
        global conversation_history
        message = body.get("message", "")
        username = body.get("username", "viewer")

        if not message:
            self._json_response({"error": "no message"}, 400)
            return

        user_msg = f"[Live chat from {username}]: {message}"

        try:
            result = agent.run_conversation(
                user_message=user_msg,
                system_message=PERSONA,
                conversation_history=conversation_history,
            )
            response_text = result.get("final_response") or result.get("response") or ""
            if response_text == "None":
                response_text = ""
            conversation_history = result.get("conversation_history", conversation_history)
            if len(conversation_history) > 50:
                conversation_history = conversation_history[-40:]
        except Exception as e:
            print(f"Hermes error: {e}")
            response_text = "Hmm, give me a sec... my brain glitched!"

        audio_filename = generate_tts(response_text)
        mood = detect_mood(response_text)

        self._json_response({
            "response": response_text,
            "audio_url": f"/audio/{audio_filename}" if audio_filename else None,
            "mood": mood,
            "username": username,
        })

    def _handle_idle(self):
        global conversation_history

        try:
            result = agent.run_conversation(
                user_message="[System: Chat is quiet. Say something entertaining — a thought, fun question, or anecdote. 1-2 sentences max.]",
                system_message=PERSONA,
                conversation_history=conversation_history,
            )
            response_text = result.get("final_response") or result.get("response") or ""
            if response_text == "None":
                response_text = ""
            conversation_history = result.get("conversation_history", conversation_history)
            if len(conversation_history) > 50:
                conversation_history = conversation_history[-40:]
        except Exception as e:
            print(f"Hermes idle error: {e}")
            response_text = "Hmm, it's quiet in here... anyone out there?"

        audio_filename = generate_tts(response_text)

        self._json_response({
            "response": response_text,
            "audio_url": f"/audio/{audio_filename}" if audio_filename else None,
            "mood": "neutral",
        })

    def _serve_audio(self, filename):
        filepath = AUDIO_DIR / filename
        if not filepath.exists():
            self._json_response({"error": "not found"}, 404)
            return

        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Content-Length", str(filepath.stat().st_size))
        self.end_headers()
        self.wfile.write(filepath.read_bytes())

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


if __name__ == "__main__":
    port = 5001
    print("=" * 50)
    print(" Aria — Virtual Streamer (Hermes Agent)")
    print(f" http://localhost:{port}")
    print("=" * 50)
    server = HTTPServer(("0.0.0.0", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
