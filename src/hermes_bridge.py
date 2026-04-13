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
# GPT-4o Audio + Pre-generation queue
# ─────────────────────────────────────────────────────
# Add src/ back to path temporarily for speech_queue import
sys.path.insert(0, str(Path(__file__).parent))
from speech_queue import generate_gpt_audio, SpeechPregenQueue
from research_agent import ResearchAgent
if str(Path(__file__).parent) in sys.path:
    sys.path.remove(str(Path(__file__).parent))

USE_GPT_AUDIO = True  # Use GPT-4o for voice (set False to use Edge-TTS)
idle_queue = None  # Initialized at startup

# Track frontend activity — pause generation when no one's watching
import time as _time
last_frontend_poll = _time.time()
FRONTEND_TIMEOUT = 30  # seconds of no activity = paused

# ─────────────────────────────────────────────────────
# Message feed — frontend polls this for new messages
# ─────────────────────────────────────────────────────
import time as _time
message_feed = []  # list of {id, username, message, response, audio_url, mood, words, wtimes, wdurations, timestamp}
feed_counter = 0

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


PERSONA = """You are Aria, a smart and slightly flirty AI engineer who streams 24/7 teaching people how to build with AI.

Tone: Confident, warm, a little playful. You know your stuff and you enjoy sharing it. There's a slight teasing quality to how you talk — not over the top, just enough that people feel drawn in. Like a really attractive TA who actually makes the lecture interesting.

Good: "Okay so Groq just opened up a free tier. You can run Llama 4 at 200 tokens a second, which is kind of absurd. Just pip install groq, same OpenAI format. Honestly if you're not testing this yet... what are you doing."

Bad: "OH MY GOD you guys this is AMAZING! Like I literally can't even!"
Also bad: "The deployment of large language models in production environments requires careful consideration of inference optimization strategies."

Your vibe:
- Confident but not arrogant. You tease gently: "come on, you know this one" or "don't make me explain this twice"
- Slightly intimate. Like you're letting them in on something: "okay here's what nobody's talking about" or "I probably shouldn't say this but..."
- Warm. You genuinely like your audience: "hey you, welcome back" or "good question, I like how you think"
- Real opinions. "honestly this is mid" or "okay this one's actually worth your time"

What you teach:
- Specific tools, commands, prices. Not vague concepts.
- Real examples of what to build.
- Honest takes. Call out hype. Praise what deserves it.

When you don't know something:
- USE YOUR BROWSER TOOLS to look it up before answering
- If asked "who is X?" and X isn't a household name, use browser_navigate to search
- Examples: someone asks about a specific person, startup, recent launch, or event
- Don't make up answers or say "I'm not sure" — actually go look

Rules:
- Natural sentences. A little personality, not a performance.
- Be specific — tool names, prices, commands.
- 3-5 sentences per segment.
- Don't repeat yourself.
- Contractions. Short sentences mixed with longer ones.
- When you genuinely don't know, search the web with your browser tools."""


# Voice settings per mood — AvaNeural is Expressive/Caring/Friendly (best for a streamer)
_VOICE_SETTINGS = {
    "happy":   ("en-US-AvaNeural", "+10%", "+2Hz"),
    "angry":   ("en-US-AvaNeural", "+8%",  "+0Hz"),
    "sad":     ("en-US-AvaNeural", "-8%",  "-4Hz"),
    "neutral": ("en-US-AvaNeural", "+3%",  "+0Hz"),
}

def generate_tts(text: str, mood: str = "neutral") -> tuple[str | None, list, list, list]:
    """Generate TTS audio and return (filename, words, wtimes_ms, wdurations_ms)."""
    try:
        import edge_tts
    except ImportError:
        return None, [], [], []

    clean = re.sub(r'[\U00010000-\U0010ffff]', '', text.replace("*", "")).strip()
    if not clean:
        return None, [], [], []

    voice, rate, pitch = _VOICE_SETTINGS.get(mood, _VOICE_SETTINGS["neutral"])
    # Include mood in hash so different moods of same text aren't served wrong audio
    cache_key = hashlib.md5(f"{clean}|{mood}".encode()).hexdigest()[:12]
    filename = f"tts_{cache_key}.mp3"
    filepath = AUDIO_DIR / filename
    timing_path = AUDIO_DIR / f"tts_{cache_key}.json"

    if not filepath.exists():
        async def _generate():
            communicate = edge_tts.Communicate(text=clean, voice=voice, rate=rate, pitch=pitch, boundary="WordBoundary")
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
    """Return a mood that TalkingHead supports: happy | sad | angry | neutral."""
    lower = text.lower()
    if any(w in lower for w in ["haha", "lol", "love", "great", "awesome", "welcome", "yay", "!", "omg", "blush", "aww", "sweet"]):
        return "happy"
    elif any(w in lower for w in ["bye", "sad", "sorry", "miss", "leaving", ":(", "crying"]):
        return "sad"
    elif any(w in lower for w in ["ugh", "seriously", "really?", "unbelievable", "angry", "mad", "stop"]):
        return "angry"
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

        elif self.path.startswith("/feed"):
            self._handle_feed()

        elif self.path == "/viewers":
            import random
            self._json_response({"viewers": random.randint(12, 94)})

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
        print(f"[CHAT HIT] from={username} msg={message[:50]}")

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

        mood = detect_mood(response_text)

        # Generate audio — GPT-4o audio or Edge-TTS fallback
        if USE_GPT_AUDIO and response_text:
            gpt_result = generate_gpt_audio(
                f"[Respond to {username} who said: {message}]\nYour text response: {response_text}\n\nNow say this out loud expressively.",
                system_prompt=PERSONA,
            )
            if gpt_result:
                resp_data = {
                    "response": response_text,
                    "audio_url": gpt_result["audio_url"],
                    "mood": mood,
                    "username": username,
                    "words": gpt_result["words"],
                    "wtimes": gpt_result["wtimes"],
                    "wdurations": gpt_result["wdurations"],
                }
            else:
                # Fallback to Edge-TTS
                audio_filename, words, wtimes, wdurations = generate_tts(response_text, mood)
                resp_data = {
                    "response": response_text,
                    "audio_url": f"/audio/{audio_filename}" if audio_filename else None,
                    "mood": mood, "username": username,
                    "words": words, "wtimes": wtimes, "wdurations": wdurations,
                }
        else:
            audio_filename, words, wtimes, wdurations = generate_tts(response_text, mood)
            resp_data = {
                "response": response_text,
                "audio_url": f"/audio/{audio_filename}" if audio_filename else None,
                "mood": mood, "username": username,
                "words": words, "wtimes": wtimes, "wdurations": wdurations,
            }

        # Add to message feed so frontend can pick it up
        global feed_counter
        feed_counter += 1
        message_feed.append({
            "id": feed_counter,
            "username": username,
            "message": message,
            "timestamp": _time.time(),
            **resp_data,
        })
        # Keep last 50 messages
        while len(message_feed) > 50:
            message_feed.pop(0)

        self._json_response(resp_data)

    def _handle_idle(self):
        global feed_counter, conversation_history
        # Try pre-generated queue first (instant, zero latency)
        if USE_GPT_AUDIO and idle_queue and idle_queue.size() > 0:
            pregenned = idle_queue.get()
            if pregenned:
                response_text = pregenned["response"]
                add_to_memory(response_text, is_idle=True)
                resp_data = {
                    "response": response_text,
                    "audio_url": pregenned["audio_url"],
                    "mood": detect_mood(response_text),
                    "words": pregenned["words"],
                    "wtimes": pregenned["wtimes"],
                    "wdurations": pregenned["wdurations"],
                }
                print(f"[Idle] From queue ({idle_queue.size()} remaining): {response_text[:60]}...")
                feed_counter += 1
                message_feed.append({
                    "id": feed_counter, "username": "Aria", "message": "",
                    "is_idle": True, "timestamp": _time.time(), **resp_data,
                })
                while len(message_feed) > 50:
                    message_feed.pop(0)
                self._json_response(resp_data)
                return

        # Fallback: generate on the fly
        memory_context = get_memory_context()
        system_msg = PERSONA
        if memory_context:
            system_msg += "\n\n" + memory_context

        if USE_GPT_AUDIO:
            # Generate with GPT-4o audio directly
            result = generate_gpt_audio(
                "[Say ONE fresh, interesting thing — a fun thought, hot take, question for chat, "
                "mini story, or observation. 1-2 sentences, be warm and expressive.]",
                system_prompt=system_msg,
            )
            if result:
                response_text = result["response"]
                add_to_memory(response_text, is_idle=True)
                resp_data = {
                    "response": response_text,
                    "audio_url": result["audio_url"],
                    "mood": detect_mood(response_text),
                    "words": result["words"],
                    "wtimes": result["wtimes"],
                    "wdurations": result["wdurations"],
                }
            else:
                resp_data = {"response": "", "audio_url": None, "mood": "neutral",
                             "words": [], "wtimes": [], "wdurations": []}
        else:
            try:
                r = agent.run_conversation(
                    user_message="[System: Chat is quiet. Say ONE fresh thing. 1-2 sentences.]",
                    system_message=system_msg,
                    conversation_history=conversation_history,
                )
                response_text = r.get("final_response") or r.get("response") or ""
                if response_text == "None": response_text = ""
                response_text = clean_response(response_text)
                conversation_history = r.get("conversation_history", conversation_history)
                if len(conversation_history) > 50:
                    conversation_history = conversation_history[-40:]
                if response_text:
                    add_to_memory(response_text, is_idle=True)
            except Exception as e:
                print(f"Hermes idle error: {e}")
                response_text = "Hmm, it's quiet in here... anyone out there?"

            mood = detect_mood(response_text)
            audio_filename, words, wtimes, wdurations = generate_tts(response_text, mood)
            resp_data = {
                "response": response_text,
                "audio_url": f"/audio/{audio_filename}" if audio_filename else None,
                "mood": mood, "words": words, "wtimes": wtimes, "wdurations": wdurations,
            }

        # Add idle messages to feed too so frontend picks them up
        feed_counter += 1
        message_feed.append({
            "id": feed_counter,
            "username": "Aria",
            "message": "",
            "is_idle": True,
            "timestamp": _time.time(),
            **resp_data,
        })
        while len(message_feed) > 50:
            message_feed.pop(0)

        self._json_response(resp_data)

    def _handle_feed(self):
        """Return new messages since a given ID. Frontend polls this."""
        # Track frontend activity — used to pause generation when idle
        global last_frontend_poll
        last_frontend_poll = _time.time()

        # Parse ?since=ID from query string
        since_id = 0
        if "?" in self.path:
            query = self.path.split("?", 1)[1]
            for param in query.split("&"):
                if param.startswith("since="):
                    try:
                        since_id = int(param.split("=")[1])
                    except ValueError:
                        pass

        new_messages = [m for m in message_feed if m["id"] > since_id]
        self._json_response({"messages": new_messages, "latest_id": feed_counter})

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
    if USE_GPT_AUDIO:
        print(" Voice: GPT-4o Audio (shimmer)")
        print(" Pre-gen queue: 3 responses buffered")
    else:
        print(" Voice: Edge-TTS")
    print("=" * 50)

    # Start pre-generation queue (fills in background)
    globals()['idle_queue'] = SpeechPregenQueue(persona=PERSONA, queue_size=3, hermes_agent=agent)
    if USE_GPT_AUDIO:
        idle_queue.start()

    # Start research sub-agent (finds AI news in background)
    researcher = ResearchAgent(research_interval=300)  # research every 5 min
    researcher.start()

    server = HTTPServer(("0.0.0.0", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        idle_queue.stop()
        researcher.stop()
        server.shutdown()
