#!/usr/bin/env python3
"""Pre-generation queue — always keeps N responses ready so Aria never goes silent.

Runs as a background thread inside hermes_bridge.py.
While Aria speaks, this fills the queue with the next idle lines.
"""

import base64
import io
import json
import threading
import time
import wave
from collections import deque
from pathlib import Path

import httpx

AUDIO_DIR = Path(__file__).parent.parent / "output" / "tts_cache"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_old_audio(max_age_seconds: int = 3600):
    """Delete audio files older than max_age_seconds. Called periodically."""
    import os
    now = time.time()
    count = 0
    for f in AUDIO_DIR.iterdir():
        if f.suffix in ('.wav', '.mp3', '.json') and (now - f.stat().st_mtime) > max_age_seconds:
            f.unlink()
            count += 1
    if count:
        print(f"[Cleanup] Deleted {count} old audio files")

# PaleBlueDot GPT-4o Audio config
GPT_AUDIO_URL = "https://open.palebluedot.ai/v1/chat/completions"
GPT_AUDIO_KEY = "sk-YsLWFSaNOGBy8S3ZKDKWwpiALlSCCBv2TEGPxanBAT0Jhiof"
GPT_AUDIO_MODEL = "openai/gpt-audio"
GPT_AUDIO_VOICE = "nova"


def generate_gpt_audio(text: str, system_prompt: str = "", voice: str = None, speak_only: str = None) -> dict | None:
    """Call GPT-4o audio to generate speech.

    If speak_only is provided, GPT-4o reads that exact text aloud (TTS mode).
    Otherwise it generates its own response based on the text/system prompt.

    Returns {response, audio_url, words, wtimes, wdurations} or None.
    """
    voice = voice or GPT_AUDIO_VOICE

    messages = []
    if speak_only:
        # TTS mode: just have GPT-4o read the exact text aloud
        messages.append({
            "role": "system",
            "content": (
                "You are a text-to-speech engine. Read the user's text aloud EXACTLY as written. "
                "Speak at a RELAXED, unhurried pace — like you're chatting on a cozy late-night stream. "
                "Pause briefly between clauses. Warm, natural intonation. "
                "Do not add anything. Do not change anything. Just speak it."
            )
        })
        messages.append({"role": "user", "content": speak_only})
    else:
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

    # Voice style
    if messages and messages[0]["role"] != "system":
        messages.insert(0, {
            "role": "system",
            "content": (
                "You are a 22-year-old woman streaming on Twitch. Bright, youthful voice. "
                "Speak at a relaxed, medium pace — not rushed. Take your time with each sentence. "
                "Warm and natural.\n"
                "CRITICAL: Vary your opening every single time. NEVER start with 'okay so', 'so', 'alright', or 'let me'. "
                "Instead start with the actual content — a fact, a question, a name, a number, a bold claim. "
                "Examples of good openings: 'Vector databases cost...', 'Here is something wild...', "
                "'Most people get this wrong...', 'The thing about RAG is...', 'Groq just dropped...'\n"
                "3-5 sentences. Finish every sentence completely."
            )
        })

    body = {
        "model": GPT_AUDIO_MODEL,
        "modalities": ["text", "audio"],
        "audio": {"voice": voice, "format": "pcm16"},
        "stream": True,
        "messages": messages,
        "max_tokens": 2000,
    }

    audio_chunks = []
    transcript = ""

    try:
        # Use raw bytes streaming to avoid any line-buffering issues
        buffer = ""
        with httpx.stream(
            "POST", GPT_AUDIO_URL,
            json=body,
            headers={
                "Authorization": f"Bearer {GPT_AUDIO_KEY}",
                "Content-Type": "application/json",
            },
            timeout=60,
        ) as resp:
            for raw_chunk in resp.iter_bytes():
                buffer += raw_chunk.decode("utf-8", errors="replace")
                # Process complete SSE events (separated by \n\n or just \n)
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    # Accept both "data: " and "data:" (with/without space)
                    if line.startswith("data:"):
                        data = line[5:].lstrip()
                    else:
                        continue
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    # Extract audio + transcript from ALL possible locations
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    for choice in choices:
                        # Try delta first (streaming)
                        delta = choice.get("delta", {}) or choice.get("message", {})
                        audio_obj = delta.get("audio") or {}
                        if isinstance(audio_obj, dict):
                            d = audio_obj.get("data", "")
                            if d:
                                try:
                                    audio_chunks.append(base64.b64decode(d))
                                except Exception:
                                    pass
                            t = audio_obj.get("transcript", "")
                            if t:
                                transcript += t

        if not audio_chunks:
            return None

        # Convert PCM16 to WAV
        pcm = b"".join(audio_chunks)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm)

        # Save to file
        import hashlib
        file_hash = hashlib.md5(transcript.encode()).hexdigest()[:12]
        filename = f"gpt_{file_hash}.wav"
        filepath = AUDIO_DIR / filename
        filepath.write_bytes(buf.getvalue())

        # Build word timings from transcript
        words = transcript.split()
        duration_ms = (len(pcm) / 2) / 24000 * 1000  # PCM16 = 2 bytes per sample
        ms_per_word = duration_ms / max(len(words), 1)

        return {
            "response": transcript,
            "audio_url": f"/audio/{filename}",
            "words": words,
            "wtimes": [i * ms_per_word for i in range(len(words))],
            "wdurations": [ms_per_word * 0.9] * len(words),
            "duration_ms": duration_ms,
        }

    except Exception as e:
        print(f"GPT audio error: {e}")
        return None


class SpeechPregenQueue:
    """Background thread that pre-generates idle responses so Aria never goes silent."""

    def __init__(self, persona: str, queue_size: int = 3, hermes_agent=None):
        self.persona = persona
        self.queue_size = queue_size
        self.queue = deque()
        self.lock = threading.Lock()
        self._running = False
        self._thread = None
        self.recent_topics = []
        self.hermes_agent = hermes_agent
        self.hermes_history = []
        self._current_fallback_topic = None  # Stick with one topic for multiple segments
        self._fallback_segments_on_topic = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._fill_loop, daemon=True)
        self._thread.start()
        print(f"[PregenQueue] Started (target={self.queue_size} buffered)")

    def stop(self):
        self._running = False

    def get(self) -> dict | None:
        """Get next pre-generated response, or None if queue empty."""
        with self.lock:
            if self.queue:
                return self.queue.popleft()
        return None

    def size(self) -> int:
        with self.lock:
            return len(self.queue)

    def _fill_loop(self):
        """Background loop: keep queue full — pauses if no frontend connected."""
        while self._running:
            # Only generate if frontend is active (saves tokens when no viewers)
            try:
                import sys
                import importlib
                bridge = importlib.import_module("hermes_bridge")
                idle_secs = time.time() - bridge.last_frontend_poll
                if idle_secs > bridge.FRONTEND_TIMEOUT:
                    time.sleep(5)
                    continue
            except Exception:
                pass

            with self.lock:
                current_size = len(self.queue)

            if current_size < self.queue_size:
                # Priority 1: Get next segment from curriculum (long-form lesson)
                curriculum_segment = None
                try:
                    from curriculum_agent import get_next_segment
                    curriculum_segment = get_next_segment()
                except ImportError:
                    pass

                # Priority 2: Get a researched current-events topic
                topic = None
                if not curriculum_segment:
                    try:
                        from research_agent import pop_topic
                        topic = pop_topic()
                        if topic:
                            print(f"[PregenQueue] Speaking about: {topic.get('topic', '?')}")
                    except ImportError:
                        pass

                # Simple prompt — let Hermes decide what to talk about based on its conversation history.
                # If there's a curriculum segment or research topic, provide it as context. Otherwise just let it continue.
                if curriculum_segment:
                    user_msg = (
                        f"[Say this in one sentence, your own voice:] {curriculum_segment['text']}"
                    )
                    log_label = f"[course] {curriculum_segment['lesson'][:40]}"
                elif topic:
                    user_msg = (
                        f"[One sentence about this news:] {topic.get('topic', '')}: {topic.get('summary', '')}"
                    )
                    log_label = topic.get('topic', '?')[:50]
                else:
                    user_msg = "[Next sentence. ONE sentence only. Then stop.]"
                    log_label = "next"

                # Step 1: Hermes Agent generates the text (with personality, memory, tools)
                response_text = ""
                if self.hermes_agent:
                    try:
                        # Reuse same task_id so all idle calls log to ONE session file
                        result = self.hermes_agent.run_conversation(
                            user_message=user_msg,
                            system_message=self.persona,
                            conversation_history=self.hermes_history,
                            task_id="aria_idle_monologue",
                        )
                        response_text = result.get("final_response") or result.get("response") or ""
                        if response_text == "None":
                            response_text = ""
                        # Strip MEDIA: paths Hermes injects
                        import re
                        response_text = re.sub(r'MEDIA:\S+', '', response_text)
                        response_text = re.sub(r'/Users/\S+', '', response_text).strip()
                        # Update Hermes history (keep recent context)
                        self.hermes_history = result.get("messages", result.get("conversation_history", self.hermes_history))
                        if len(self.hermes_history) > 60:
                            self.hermes_history = self.hermes_history[-50:]
                    except Exception as e:
                        print(f"[PregenQueue] Hermes error: {e}")

                if not response_text:
                    audio_result = generate_gpt_audio(user_msg, system_prompt=self.persona)
                else:
                    audio_result = generate_gpt_audio(text="", speak_only=response_text)

                if audio_result:
                    with self.lock:
                        self.queue.append(audio_result)
                    self.recent_topics.append(audio_result["response"][:100])
                    if len(self.recent_topics) > 30:
                        self.recent_topics = self.recent_topics[-20:]
                    print(f"[PregenQueue] Buffered [{log_label}] ({len(self.queue)}): {audio_result['response'][:60]}...")

                if not self._running:
                    return

            # Cleanup old audio files every loop
            try: cleanup_old_audio(3600)
            except: pass

            # Wait before checking again
            time.sleep(2 if current_size < self.queue_size else 5)
