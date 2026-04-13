#!/usr/bin/env python3
"""Curriculum Agent — writes Aria a full structured course she works through.

Background agent that:
1. Generates a full multi-week curriculum (modules → lessons → segments)
2. Each segment is a 30-60 second talking-points block
3. Aria works through them sequentially, never running out of material
4. New modules are written ahead while she's talking through current ones
5. Periodically refreshes with current AI news as new modules

Layout:
  data/curriculum/
    curriculum.json          — overview & progress pointer
    module_001_llm_basics.json
    module_002_rag.json
    module_003_agents.json
    ...

Each module file:
  {
    "module": "Building Production RAG Systems",
    "intro": "...what we'll cover...",
    "lessons": [
      {
        "title": "Why RAG instead of fine-tuning",
        "segments": [
          {"text": "...what to say...", "duration_sec": 45, "spoken": false},
          ...
        ]
      },
      ...
    ]
  }
"""

import json
import sys
import threading
import time
from pathlib import Path

HERMES_DIR = Path(__file__).parent.parent / "hermes-agent"
if str(HERMES_DIR) not in sys.path:
    sys.path.insert(0, str(HERMES_DIR))
_src = str(Path(__file__).parent)
while _src in sys.path:
    sys.path.remove(_src)

from run_agent import AIAgent

CURRICULUM_DIR = Path(__file__).parent.parent / "data" / "curriculum"
CURRICULUM_DIR.mkdir(parents=True, exist_ok=True)
INDEX_FILE = CURRICULUM_DIR / "curriculum.json"


# ─────────────────────────────────────────────────────
# Initial topic outline — agent expands each into a full module
# ─────────────────────────────────────────────────────
DEFAULT_OUTLINE = [
    "LLM Fundamentals: What's actually happening when you call GPT-4",
    "Prompt Engineering: Patterns that work in production",
    "RAG Systems: From toy demo to real deployment",
    "Vector Databases: Pinecone vs Chroma vs Weaviate, real costs",
    "Embeddings Deep Dive: What models to use and why",
    "Fine-tuning: When LoRA beats RAG (and when it doesn't)",
    "AI Agents: What's actually different from prompt chains",
    "Tool Use: Function calling, MCP, structured outputs",
    "Agent Frameworks Reviewed: LangChain, CrewAI, AutoGen, Hermes",
    "Memory in Agents: Conversation history, semantic memory, knowledge graphs",
    "Open Source LLMs: Llama 4, Mistral, Qwen — when local beats API",
    "LLM Inference Optimization: vLLM, TGI, llama.cpp, batching tricks",
    "Cost Engineering: How to cut your AI bill 90%",
    "Voice AI Stack: STT → LLM → TTS pipelines that actually work",
    "Image Generation: Flux, SDXL, ComfyUI workflows",
    "Multi-modal Models: GPT-4o, Gemini, what they're actually good at",
    "AI Evals: How to measure if your LLM app is working",
    "Production AI Pipelines: Monitoring, observability, debugging",
    "AI Safety in Practice: Guardrails, jailbreak prevention, content filters",
    "The Business of AI: Real revenue, real moats, real failures",
    "Building AI Products: Idea to launch in 30 days",
    "The State of AI 2026: What's shipping, what's hype",
]


def load_index() -> dict:
    if INDEX_FILE.exists():
        try:
            return json.loads(INDEX_FILE.read_text())
        except Exception:
            pass
    return {
        "modules": [],          # [{"file": "module_001_xxx.json", "title": "...", "completed": false}]
        "current_module": 0,
        "current_lesson": 0,
        "current_segment": 0,
        "outline_used": [],     # outline topics already turned into modules
        "version": 1,
    }


def save_index(idx: dict):
    INDEX_FILE.write_text(json.dumps(idx, indent=2, ensure_ascii=False))


def load_module(filename: str) -> dict | None:
    path = CURRICULUM_DIR / filename
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def save_module(filename: str, data: dict):
    (CURRICULUM_DIR / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False))


def get_next_segment() -> dict | None:
    """Get the next un-spoken segment from the curriculum.

    Returns: {"text": "...", "module": "...", "lesson": "..."} or None if curriculum empty.
    Marks segment as spoken and advances the pointer.
    """
    idx = load_index()
    if not idx["modules"]:
        return None

    while idx["current_module"] < len(idx["modules"]):
        mod_meta = idx["modules"][idx["current_module"]]
        mod = load_module(mod_meta["file"])
        if not mod:
            idx["current_module"] += 1
            continue

        lessons = mod.get("lessons", [])
        while idx["current_lesson"] < len(lessons):
            lesson = lessons[idx["current_lesson"]]
            segments = lesson.get("segments", [])
            while idx["current_segment"] < len(segments):
                seg = segments[idx["current_segment"]]
                if not seg.get("spoken"):
                    seg["spoken"] = True
                    save_module(mod_meta["file"], mod)
                    idx["current_segment"] += 1
                    save_index(idx)
                    return {
                        "text": seg["text"],
                        "module": mod.get("module", ""),
                        "lesson": lesson.get("title", ""),
                    }
                idx["current_segment"] += 1
            idx["current_lesson"] += 1
            idx["current_segment"] = 0

        # Module complete
        mod_meta["completed"] = True
        idx["current_module"] += 1
        idx["current_lesson"] = 0
        idx["current_segment"] = 0
        save_index(idx)

    return None


def remaining_segments() -> int:
    """Count un-spoken segments across all modules."""
    idx = load_index()
    total = 0
    for mod_meta in idx["modules"]:
        mod = load_module(mod_meta["file"])
        if not mod:
            continue
        for lesson in mod.get("lessons", []):
            for seg in lesson.get("segments", []):
                if not seg.get("spoken"):
                    total += 1
    return total


# ─────────────────────────────────────────────────────
# Curriculum agent
# ─────────────────────────────────────────────────────
MODULE_PROMPT = """Write a full streaming module for an AI educator named Aria.

Topic: {topic}

Structure: 5-8 lessons, each with 4-6 segments. Each segment is what Aria says out loud — natural spoken language, 2-4 sentences, completely self-contained.

Each segment should:
- Be standalone — make sense without the previous segment
- Be SPECIFIC: real tool names, real prices, real benchmarks, real examples
- Sound like she's talking to a friend, not reading a textbook
- Use contractions, natural rhythm, occasional opinions
- Take about 30-45 seconds to say out loud (~80-120 words)

Return ONLY a JSON object:
{{
  "module": "Topic title",
  "intro": "1-2 sentence module overview Aria says first",
  "lessons": [
    {{
      "title": "Lesson title",
      "segments": [
        {{"text": "What Aria says (2-4 sentences, specific, conversational)", "spoken": false}},
        ...
      ]
    }},
    ...
  ]
}}

Make at least 5 lessons with 4+ segments each = 20+ segments minimum.
"""


class CurriculumAgent:
    """Background agent that writes new curriculum modules ahead of time."""

    def __init__(self, write_interval: int = 600, target_buffer: int = 100):
        """
        Args:
            write_interval: seconds between checking if we need new modules
            target_buffer: keep at least this many un-spoken segments queued
        """
        self.write_interval = write_interval
        self.target_buffer = target_buffer
        self.agent = AIAgent(
            base_url="https://open.palebluedot.ai/v1",
            model="anthropic/claude-opus-4.6",
            api_key="sk-YuA06jNFuzJoyaJha8C4CYOrvECdGQ1LtkEEESfUkuT6UY6d",
        )
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[Curriculum] Agent started (target buffer: {self.target_buffer} segments)")

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                remaining = remaining_segments()
                if remaining < self.target_buffer:
                    print(f"[Curriculum] Buffer low ({remaining} segments). Writing next module...")
                    self._write_next_module()
                else:
                    print(f"[Curriculum] Buffer healthy: {remaining} segments queued")
            except Exception as e:
                print(f"[Curriculum] Error: {e}")

            for _ in range(self.write_interval):
                if not self._running:
                    return
                time.sleep(1)

    def _write_next_module(self):
        idx = load_index()
        # Pick next outline topic not yet used
        used = set(idx.get("outline_used", []))
        remaining_topics = [t for t in DEFAULT_OUTLINE if t not in used]
        if not remaining_topics:
            # Restart from beginning, mark all as fresh
            idx["outline_used"] = []
            remaining_topics = DEFAULT_OUTLINE
        topic = remaining_topics[0]

        print(f"[Curriculum] Writing module: {topic}")
        result = self.agent.run_conversation(
            user_message=MODULE_PROMPT.format(topic=topic),
            system_message=(
                "You write streaming curriculum for an AI educator. "
                "Output ONLY valid JSON — no markdown, no commentary. "
                "Be specific with real tools, real prices, real examples."
            ),
            conversation_history=[],
            task_id="aria_curriculum_writer",
        )

        response = result.get("final_response") or result.get("response") or ""
        module = self._parse_json(response)

        if module and "lessons" in module and module["lessons"]:
            # Save module
            slug = topic.split(":")[0].lower().replace(" ", "_")[:30]
            num = len(idx["modules"]) + 1
            filename = f"module_{num:03d}_{slug}.json"
            save_module(filename, module)

            # Update index
            idx["modules"].append({
                "file": filename,
                "title": module.get("module", topic),
                "completed": False,
            })
            idx.setdefault("outline_used", []).append(topic)
            save_index(idx)

            seg_count = sum(len(l.get("segments", [])) for l in module["lessons"])
            print(f"[Curriculum] Wrote {filename}: {len(module['lessons'])} lessons, {seg_count} segments")
        else:
            print(f"[Curriculum] Failed to parse module for: {topic}")

    def _parse_json(self, response: str) -> dict | None:
        import re
        try:
            return json.loads(response)
        except Exception:
            pass
        # Fenced
        matches = re.findall(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
        for m in matches:
            try:
                d = json.loads(m)
                if isinstance(d, dict):
                    return d
            except Exception:
                continue
        # First {...}
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            try:
                d = json.loads(match.group(0))
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
        return None


if __name__ == "__main__":
    print("Starting curriculum agent (standalone test)...")
    agent = CurriculumAgent(write_interval=30, target_buffer=20)
    agent.start()
    try:
        while True:
            time.sleep(15)
            print(f"[Status] {remaining_segments()} segments queued")
    except KeyboardInterrupt:
        agent.stop()
