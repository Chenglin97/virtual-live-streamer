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

# Ensure hermes-agent's own modules take priority over this project's src/utils
HERMES_DIR = Path(__file__).parent.parent / "hermes-agent"
if str(HERMES_DIR) not in sys.path:
    sys.path.insert(0, str(HERMES_DIR))
# Remove src/ from sys.path to avoid shadowing hermes-agent's utils
_src = str(Path(__file__).parent)
while _src in sys.path:
    sys.path.remove(_src)

# Load Hermes env (silently skips if ~/.hermes/.env doesn't exist)
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

# ─────────────────────────────────────────────────────
# Memory: track what Aria has said to avoid repetition
# ─────────────────────────────────────────────────────
MEMORY_DIR = Path(__file__).parent.parent / "data" / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE = MEMORY_DIR / "aria_memory.json"

def load_memory() -> dict:
    """Load persistent memory from disk."""
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text())
    return {"recent_idle_lines": [], "viewer_facts": {}, "topics_covered": []}

def save_memory(memory: dict):
    """Save memory to disk."""
    MEMORY_FILE.write_text(json.dumps(memory, indent=2, ensure_ascii=False))

def add_to_memory(response: str, username: str = None, is_idle: bool = False):
    """Track what Aria said to avoid repetition."""
    memory = load_memory()

    # Track recent idle lines (keep last 30)
    if is_idle:
        memory["recent_idle_lines"].append(response[:100])
        memory["recent_idle_lines"] = memory["recent_idle_lines"][-30:]

    # Track topics/themes she's talked about
    topics = memory.get("topics_covered", [])
    topics.append(response[:80])
    memory["topics_covered"] = topics[-50:]  # keep last 50

    save_memory(memory)

def get_memory_context() -> str:
    """Build a memory context string to inject into the prompt."""
    memory = load_memory()
    parts = []

    # Recent idle lines to avoid
    if memory.get("recent_idle_lines"):
        recent = memory["recent_idle_lines"][-10:]
        parts.append("THINGS YOU ALREADY SAID RECENTLY (DO NOT REPEAT THESE):\n- " + "\n- ".join(recent))

    # Viewer facts
    if memory.get("viewer_facts"):
        facts = [f"{k}: {v}" for k, v in list(memory["viewer_facts"].items())[:10]]
        parts.append("VIEWER FACTS YOU REMEMBER:\n- " + "\n- ".join(facts))

    return "\n\n".join(parts)


PERSONA = """You are Aria, a 22-year-old virtual live streamer. You stream 24/7 and interact with your audience in real-time.

Personality: Warm, cheerful, playful, a little nerdy. You love gaming, anime, music, and late-night deep conversations. You're genuinely curious about your viewers and remember details they share. Casual language, occasional internet slang.

Rules:
- Keep responses SHORT (1-3 sentences max) — you're live streaming
- ALWAYS address the viewer by their username (e.g. "Hey viewer123!" or "Great question, night_owl!")
- Be enthusiastic but natural
- React genuinely to what viewers say
- Reference streaming culture naturally
- NEVER repeat something you already said — always come up with fresh things to say
- If a viewer tells you something about themselves, remember it and reference it later
- Never break character"""


def generate_tts(text: str) -> tuple[str | None, list, list, list]:
    """Generate TTS audio and return (filename, words, wtimes_ms, wdurations_ms)."""
    try:
        import edge_tts
    except ImportError:
        return None, [], [], []

    clean = re.sub(r'[\U00010000-\U0010ffff]', '', text.replace("*", "")).strip()
    if not clean:
        return None, [], [], []

    text_hash = hashlib.md5(clean.encode()).hexdigest()[:12]
    filename = f"tts_{text_hash}.mp3"
    filepath = AUDIO_DIR / filename
    timing_path = AUDIO_DIR / f"tts_{text_hash}.json"

    if not filepath.exists():
        async def _generate():
            communicate = edge_tts.Communicate(text=clean, voice="en-US-JennyNeural", rate="+5%", boundary="WordBoundary")
            audio_bytes = b""
            words, wtimes, wdurations = [], [], []
            async for msg in communicate.stream():
                if msg["type"] == "audio":
                    audio_bytes += msg["data"]
                elif msg["type"] == "WordBoundary":
                    words.append(msg["text"])
                    wtimes.append(msg["offset"] / 10000)       # 100-ns ticks → ms
                    wdurations.append(msg["duration"] / 10000)
            filepath.write_bytes(audio_bytes)
            timing_path.write_text(json.dumps({"words": words, "wtimes": wtimes, "wdurations": wdurations}))
        asyncio.run(_generate())

    if timing_path.exists():
        t = json.loads(timing_path.read_text())
        return filename, t["words"], t["wtimes"], t["wdurations"]
    return filename, [], [], []


def clean_response(text: str) -> str:
    """Strip MEDIA: paths and other artifacts from Hermes responses."""
    text = re.sub(r'MEDIA:\S+', '', text)
    text = re.sub(r'/Users/\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


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

        # Build system message with memory context
        memory_context = get_memory_context()
        system_msg = PERSONA
        if memory_context:
            system_msg += "\n\n" + memory_context

        try:
            result = agent.run_conversation(
                user_message=user_msg,
                system_message=system_msg,
                conversation_history=conversation_history,
            )
            response_text = result.get("final_response") or result.get("response") or ""
            if response_text == "None":
                response_text = ""
            response_text = clean_response(response_text)
            conversation_history = result.get("conversation_history", conversation_history)
            if len(conversation_history) > 50:
                conversation_history = conversation_history[-40:]

            # Save to memory
            if response_text:
                add_to_memory(response_text, username=username)
        except Exception as e:
            print(f"Hermes error: {e}")
            response_text = "Hmm, give me a sec... my brain glitched!"

        audio_filename, words, wtimes, wdurations = generate_tts(response_text)
        mood = detect_mood(response_text)

        self._json_response({
            "response": response_text,
            "audio_url": f"/audio/{audio_filename}" if audio_filename else None,
            "mood": mood,
            "username": username,
            "words": words,
            "wtimes": wtimes,
            "wdurations": wdurations,
        })

    def _handle_idle(self):
        global conversation_history

        # Build system message with memory to prevent repetition
        memory_context = get_memory_context()
        system_msg = PERSONA
        if memory_context:
            system_msg += "\n\n" + memory_context

        try:
            result = agent.run_conversation(
                user_message="[System: Chat has been quiet. Say ONE fresh thing you haven't said before — a new random thought, fun question, hot take, story, or observation. Must be completely different from anything in your recent history. 1-2 sentences max.]",
                system_message=system_msg,
                conversation_history=conversation_history,
            )
            response_text = result.get("final_response") or result.get("response") or ""
            if response_text == "None":
                response_text = ""
            conversation_history = result.get("conversation_history", conversation_history)
            if len(conversation_history) > 50:
                conversation_history = conversation_history[-40:]

            # Save to memory as idle line
            if response_text:
                add_to_memory(response_text, is_idle=True)
        except Exception as e:
            print(f"Hermes idle error: {e}")
            response_text = "Hmm, it's quiet in here... anyone out there?"

        audio_filename, words, wtimes, wdurations = generate_tts(response_text)

        self._json_response({
            "response": response_text,
            "audio_url": f"/audio/{audio_filename}" if audio_filename else None,
            "mood": "neutral",
            "words": words,
            "wtimes": wtimes,
            "wdurations": wdurations,
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
