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


def generate_gpt_audio(text: str, system_prompt: str = "", voice: str = None) -> dict | None:
    """Call GPT-4o audio to generate speech. Returns {response, audio_url, transcript, words, wtimes, wdurations} or None."""
    voice = voice or GPT_AUDIO_VOICE

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})

    # System instruction to prevent mid-sentence cuts
    if messages and messages[0]["role"] != "system":
        messages.insert(0, {
            "role": "system",
            "content": "CRITICAL: Always speak in complete sentences. Never stop mid-word or mid-sentence. Finish every thought completely before stopping."
        })

    body = {
        "model": GPT_AUDIO_MODEL,
        "modalities": ["text", "audio"],
        "audio": {"voice": voice, "format": "pcm16"},
        "stream": True,
        "messages": messages,
        "max_tokens": 800,
    }

    audio_chunks = []
    transcript = ""

    try:
        with httpx.stream(
            "POST", GPT_AUDIO_URL,
            json=body,
            headers={
                "Authorization": f"Bearer {GPT_AUDIO_KEY}",
                "Content-Type": "application/json",
            },
            timeout=60,
        ) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if delta.get("audio"):
                    d = delta["audio"].get("data", "")
                    if d:
                        audio_chunks.append(base64.b64decode(d))
                    t = delta["audio"].get("transcript", "")
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

    def __init__(self, persona: str, queue_size: int = 3):
        self.persona = persona
        self.queue_size = queue_size
        self.queue = deque()
        self.lock = threading.Lock()
        self._running = False
        self._thread = None
        self.recent_topics = []  # Track to avoid repetition

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
        """Background loop: keep queue full from researched topics or continuation."""
        while self._running:
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

                if topic:
                    # Generate speech about the researched topic
                    talking_points = "\n".join(f"- {p}" for p in topic.get("talking_points", []))
                    recent_said = "\n".join(f"- {t}" for t in self.recent_topics[-5:])

                    prompt = (
                        f"[You're live streaming about AI news. Present this topic to your audience:]\n\n"
                        f"TOPIC: {topic.get('topic', '')}\n"
                        f"SUMMARY: {topic.get('summary', '')}\n"
                        f"WHY IT MATTERS: {topic.get('why_interesting', '')}\n"
                        f"TALKING POINTS:\n{talking_points}\n\n"
                        f"[Explain this excitedly like you just discovered it. "
                        f"Break it down so anyone can understand. "
                        f"2-3 sentences per segment. Be enthusiastic and educational. "
                        f"Use analogies if helpful.]"
                    )
                    if recent_said:
                        prompt += f"\n\nYou already said (transition smoothly, don't repeat):\n{recent_said}"

                    # Generate multiple self-contained segments for this topic
                    segment_prompts = [
                        prompt,  # Segment 1: introduce the topic
                        (f"[You just introduced '{topic.get('topic', '')}' to your audience. "
                         f"Now go DEEPER — explain WHY this matters, give a concrete example, "
                         f"or use an analogy to make it click. "
                         f"Start with a complete new sentence. 2-3 full sentences.]"),
                        (f"[Wrap up your take on '{topic.get('topic', '')}'. "
                         f"Share your personal opinion, a prediction, or ask chat what they think. "
                         f"Start with a complete new sentence. End cleanly. 2-3 full sentences.]"),
                    ]
                    for segment, seg_prompt in enumerate(segment_prompts):
                        result = generate_gpt_audio(seg_prompt, system_prompt=self.persona)
                        if result:
                            with self.lock:
                                self.queue.append(result)
                            self.recent_topics.append(result["response"][:100])
                            print(f"[PregenQueue] Buffered ({len(self.queue)}/{self.queue_size}): {result['response'][:60]}...")
                        if not self._running:
                            return

                else:
                    # No researched topic — continue natural monologue
                    recent_context = ""
                    if self.recent_topics:
                        recent_context = "\n\nWhat you've said so far (CONTINUE naturally):\n"
                        recent_context += "\n".join(f"- {t}" for t in self.recent_topics[-5:])

                    prompt = (
                        f"[You are live streaming about AI. Continue your monologue — "
                        f"share a thought about AI, teach a concept, give a hot take, "
                        f"or tell a relatable story about tech. "
                        f"2-3 sentences. Be warm and educational.]"
                        f"{recent_context}"
                    )

                    result = generate_gpt_audio(prompt, system_prompt=self.persona)
                    if result:
                        with self.lock:
                            self.queue.append(result)
                        self.recent_topics.append(result["response"][:100])
                        if len(self.recent_topics) > 30:
                            self.recent_topics = self.recent_topics[-20:]
                        print(f"[PregenQueue] Buffered ({len(self.queue)}/{self.queue_size}): {result['response'][:60]}...")

            # Wait before checking again
            time.sleep(2 if current_size < self.queue_size else 5)
