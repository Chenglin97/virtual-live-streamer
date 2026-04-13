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

# PaleBlueDot GPT-4o Audio config
GPT_AUDIO_URL = "https://open.palebluedot.ai/v1/chat/completions"
GPT_AUDIO_KEY = "sk-YsLWFSaNOGBy8S3ZKDKWwpiALlSCCBv2TEGPxanBAT0Jhiof"
GPT_AUDIO_MODEL = "openai/gpt-audio"
GPT_AUDIO_VOICE = "shimmer"


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
                "You are a text-to-speech engine. Read the user's text aloud EXACTLY as written, "
                "with natural intonation, warmth, and a slightly playful edge. "
                "Do not add anything. Do not change anything. Do not summarize. "
                "Just speak the text exactly as given."
            )
        })
        messages.append({"role": "user", "content": speak_only})
    else:
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

    # Voice style — confident, warm, slight edge
    if messages and messages[0]["role"] != "system":
        messages.insert(0, {
            "role": "system",
            "content": (
                "Spoken audio for a live stream. You're confident and warm with a slight playful edge.\n"
                "Not a performance. Not a lecture. Just a smart, attractive person talking.\n"
                "Short sentences. Natural rhythm. A little teasing. A little intimate.\n"
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
        self.hermes_agent = hermes_agent  # Hermes Agent for text generation
        self.hermes_history = []  # Aria's running monologue history

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
                # Try to get a researched topic first
                topic = None
                try:
                    from research_agent import pop_topic, queue_size as rq_size
                    topic = pop_topic()
                    if topic:
                        print(f"[PregenQueue] Speaking about: {topic.get('topic', '?')}")
                except ImportError:
                    pass

                # Build the prompt for Hermes
                if topic:
                    talking_points = "\n".join(f"- {p}" for p in topic.get("talking_points", []))
                    how_to = topic.get('how_to_use', '')
                    user_msg = (
                        f"[Talk about this on stream — speak naturally:]\n\n"
                        f"TOPIC: {topic.get('topic', '')}\n"
                        f"KEY INFO: {topic.get('summary', '')}\n"
                        f"WHY COOL: {topic.get('why_interesting', '')}\n"
                        f"HOW TO USE: {how_to}\n"
                        f"DETAILS:\n{talking_points}\n\n"
                        f"Weave in specific tools and numbers naturally. "
                        f"Sound like a real person, not a blog post. "
                        f"3-5 sentences. Finish every sentence."
                    )
                    log_label = topic.get('topic', '?')[:50]
                else:
                    # Pick a fresh fallback topic, avoiding what we just said
                    import random
                    fallback_topics = [
                        "vector databases (pinecone vs chroma vs weaviate, real prices and tradeoffs)",
                        "fine-tuning with LoRA — when it actually beats RAG and when it doesn't",
                        "the cheapest way to run inference at scale right now",
                        "prompt caching tricks that cut OpenAI bills by 90%",
                        "why most RAG systems are bad and the 3 things that fix them",
                        "the latest open-source coding model worth using",
                        "evaluating LLMs without spending hundreds on labeled data",
                        "structured outputs: JSON mode, function calling, or constrained decoding?",
                        "long context windows — when 1M tokens is a trap",
                        "embedding models compared (text-embedding-3 vs Cohere vs open-source)",
                        "real cost of running Llama 70B vs just paying OpenAI",
                        "agent observability — Langfuse vs Helicone vs build-your-own",
                        "voice AI stack right now — best STT, best TTS, real latency numbers",
                        "image generation in 2026 — Flux vs SDXL vs commercial APIs",
                        "synthetic data for training small models that beat GPT-4",
                        "the AI tooling nobody talks about but everyone should use",
                        "MCP servers — what they do, which ones are actually useful",
                        "deploying LLMs on Modal vs Replicate vs RunPod (real costs)",
                        "why guardrails are mostly theater and what actually works",
                        "the surprising thing about Claude vs GPT vs Gemini for coding",
                    ]
                    avoid = set()
                    for t in self.recent_topics[-15:]:
                        for word in t.lower().split():
                            if len(word) > 4:
                                avoid.add(word)
                    fresh = [t for t in fallback_topics if not any(w in t.lower() for w in avoid if len(w) > 4)]
                    chosen = random.choice(fresh) if fresh else random.choice(fallback_topics)

                    avoid_list = "\n".join(f"- {t[:80]}" for t in self.recent_topics[-10:])
                    user_msg = (
                        f"[Live streaming. Pick a SPECIFIC angle on this AI topic and talk about it:]\n"
                        f"SUGGESTED TOPIC: {chosen}\n\n"
                        f"DO NOT repeat anything from these recent things you said:\n{avoid_list}\n\n"
                        f"Talk about something COMPLETELY DIFFERENT. Switch topics — don't continue the previous thread. "
                        f"Be specific with tool names, prices, real numbers. "
                        f"3-5 sentences. Sound like a real person."
                    )
                    log_label = f"fallback: {chosen[:40]}"

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
                        self.hermes_history = result.get("conversation_history", self.hermes_history)
                        if len(self.hermes_history) > 30:
                            self.hermes_history = self.hermes_history[-20:]
                    except Exception as e:
                        print(f"[PregenQueue] Hermes error: {e}")

                if not response_text:
                    # Fallback: GPT-4o generates its own text
                    audio_result = generate_gpt_audio(user_msg, system_prompt=self.persona)
                else:
                    # Step 2: GPT-4o reads Hermes's text aloud (TTS only)
                    audio_result = generate_gpt_audio(text="", speak_only=response_text)

                if audio_result:
                    with self.lock:
                        self.queue.append(audio_result)
                    self.recent_topics.append(audio_result["response"][:100])
                    if len(self.recent_topics) > 30:
                        self.recent_topics = self.recent_topics[-20:]
                    print(f"[PregenQueue] Buffered [{log_label}] ({len(self.queue)}/{self.queue_size}): {audio_result['response'][:60]}...")

                if not self._running:
                    return

            # Wait before checking again
            time.sleep(2 if current_size < self.queue_size else 5)
