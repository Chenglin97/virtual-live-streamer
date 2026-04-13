#!/usr/bin/env python3
"""Learning Agent — Aria gets better at streaming over time.

Background agent that:
1. Researches what makes streamers/educators/AI personalities good
2. Distills findings into a structured knowledge base
3. Periodically reviews and consolidates knowledge
4. The knowledge is injected into Aria's persona so she improves

Knowledge categories:
  tone        — how the best ones speak (cadence, vocabulary, energy)
  hooks       — opening lines, attention-grabbers, retention tricks
  engagement  — how to make viewers feel seen and respond
  teaching    — how to explain technical things simply
  avoid       — common streamer mistakes / things that turn viewers off
  examples    — concrete quotes/examples from real streamers
"""

import json
import sys
import threading
import time
from pathlib import Path

# Hermes Agent setup (same as research_agent)
HERMES_DIR = Path(__file__).parent.parent / "hermes-agent"
if str(HERMES_DIR) not in sys.path:
    sys.path.insert(0, str(HERMES_DIR))

_src = str(Path(__file__).parent)
while _src in sys.path:
    sys.path.remove(_src)

from run_agent import AIAgent

KNOWLEDGE_FILE = Path(__file__).parent.parent / "data" / "streamer_knowledge.json"
KNOWLEDGE_FILE.parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────
# Default starter knowledge (replaced by learning over time)
# ─────────────────────────────────────────────────────
DEFAULT_KNOWLEDGE = {
    "tone": [
        "Confident but not preachy — share opinions like you're chatting with friends",
        "Short sentences. Mix in longer ones for rhythm.",
    ],
    "hooks": [
        "Open with a specific number or claim that surprises ('this thing costs 20 cents and replaces a $50k team')",
    ],
    "engagement": [
        "Address viewers by name when they chat",
        "Ask questions that have easy answers viewers can type fast",
    ],
    "teaching": [
        "Lead with the result, then explain how it works",
        "Name actual tools and prices, never vague generalities",
    ],
    "avoid": [
        "Don't say 'AI is amazing' or other empty hype",
        "Don't drill into the same topic for more than 2 segments",
        "Don't lecture — keep energy conversational",
    ],
    "examples": [],
    "_meta": {"version": 0, "last_updated": 0, "research_count": 0},
}


def load_knowledge() -> dict:
    if KNOWLEDGE_FILE.exists():
        try:
            return json.loads(KNOWLEDGE_FILE.read_text())
        except Exception:
            pass
    save_knowledge(DEFAULT_KNOWLEDGE)
    return DEFAULT_KNOWLEDGE.copy()


def save_knowledge(data: dict):
    KNOWLEDGE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def knowledge_as_prompt() -> str:
    """Format the knowledge for injection into Aria's persona."""
    k = load_knowledge()
    parts = []
    for category, items in k.items():
        if category.startswith("_") or not items:
            continue
        if isinstance(items, list):
            parts.append(f"\n{category.upper()}:")
            for item in items[:8]:  # cap per category
                parts.append(f"  - {item}")
    if parts:
        return "STREAMING WISDOM (refined from research):\n" + "\n".join(parts)
    return ""


# ─────────────────────────────────────────────────────
# Research prompts
# ─────────────────────────────────────────────────────
RESEARCH_PROMPTS = [
    {
        "topic": "What makes the best Twitch streamers great",
        "prompt": (
            "Use browser_navigate to visit https://twitchtracker.com/games/all "
            "and learn what categories are popular. Then visit one or two top "
            "streamer pages and observe their style. "
            "Also search for articles about top streamer techniques. "
            "Return a JSON object with keys: tone, hooks, engagement, avoid. "
            "Each is a list of 3-5 specific actionable tips. Be concrete with examples."
        ),
    },
    {
        "topic": "How great tech educators on YouTube/Twitch teach complex things simply",
        "prompt": (
            "Use browser_navigate to look at top tech YouTubers like Fireship, "
            "ThePrimeagen, and similar AI educators. Visit https://www.youtube.com "
            "and search 'AI explained'. "
            "Observe how they hook viewers, use analogies, pace their content. "
            "Return JSON with keys: teaching, hooks, examples (3-5 each)."
        ),
    },
    {
        "topic": "How AI streamers (like Neuro-sama, Vedal) keep viewers engaged",
        "prompt": (
            "Use browser_navigate to visit https://www.twitch.tv/vedal987 and "
            "https://www.twitch.tv/directory/category/just-chatting "
            "Read articles about Neuro-sama and AI VTubers. "
            "What makes them captivating? Return JSON with keys: tone, engagement, hooks."
        ),
    },
    {
        "topic": "Common streamer mistakes that lose viewers",
        "prompt": (
            "Search the web for 'why streamers lose viewers' and "
            "'common streaming mistakes'. Use browser_navigate to read 2-3 articles. "
            "Return JSON with key 'avoid' containing 5-8 specific mistakes."
        ),
    },
    {
        "topic": "Best opening lines and attention hooks for live content",
        "prompt": (
            "Search for 'best stream opening lines' and 'YouTube hook techniques 2026'. "
            "Read a few articles. Return JSON with key 'hooks' containing 5-8 "
            "specific opening line patterns or attention-grabbing techniques."
        ),
    },
]


class LearningAgent:
    """Background agent that improves Aria's streaming skills over time."""

    def __init__(self, research_interval: int = 1800, review_interval: int = 3600):
        """
        Args:
            research_interval: seconds between research cycles (default 30min)
            review_interval: seconds between knowledge consolidation (default 60min)
        """
        self.research_interval = research_interval
        self.review_interval = review_interval
        self.agent = AIAgent(
            base_url="https://open.palebluedot.ai/v1",
            model="anthropic/claude-opus-4.6",
            api_key="sk-YuA06jNFuzJoyaJha8C4CYOrvECdGQ1LtkEEESfUkuT6UY6d",
        )
        self._running = False
        self._thread = None
        self._last_research = 0
        self._last_review = 0
        self._prompt_idx = 0

        # Initialize knowledge file if missing
        load_knowledge()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[Learning] Agent started (research every {self.research_interval}s, review every {self.review_interval}s)")

    def stop(self):
        self._running = False

    def _loop(self):
        """Main loop: research and review on intervals."""
        while self._running:
            now = time.time()

            # Research
            if now - self._last_research >= self.research_interval:
                try:
                    self._do_research()
                except Exception as e:
                    print(f"[Learning] Research error: {e}")
                self._last_research = time.time()

            # Review/consolidate
            if now - self._last_review >= self.review_interval:
                try:
                    self._do_review()
                except Exception as e:
                    print(f"[Learning] Review error: {e}")
                self._last_review = time.time()

            for _ in range(60):
                if not self._running:
                    return
                time.sleep(1)

    def _do_research(self):
        """Run one research cycle on a rotating prompt."""
        prompt_obj = RESEARCH_PROMPTS[self._prompt_idx % len(RESEARCH_PROMPTS)]
        self._prompt_idx += 1
        print(f"[Learning] Researching: {prompt_obj['topic']}")

        result = self.agent.run_conversation(
            user_message=prompt_obj["prompt"],
            system_message=(
                "You are a research assistant studying what makes streamers great. "
                "Use browser_navigate, browser_snapshot, and other browser tools to "
                "actually visit pages and observe. Return ONLY a JSON object — no markdown, "
                "no commentary. Every tip should be specific and actionable, not vague."
            ),
            conversation_history=[],
            task_id="aria_learning_research",
        )

        response = result.get("final_response") or result.get("response") or ""
        new_findings = self._parse_json(response)

        if new_findings:
            self._merge_knowledge(new_findings)
            print(f"[Learning] Added findings to knowledge base")
        else:
            print(f"[Learning] Could not parse findings from response")

    def _do_review(self):
        """Consolidate the knowledge — remove duplicates, refine, prioritize."""
        knowledge = load_knowledge()
        knowledge_text = json.dumps({k: v for k, v in knowledge.items() if not k.startswith("_")}, indent=2)

        prompt = (
            f"Here is Aria's current streaming knowledge base. Consolidate it:\n\n"
            f"{knowledge_text}\n\n"
            f"Tasks:\n"
            f"1. Remove duplicates and contradictions\n"
            f"2. Combine similar items into stronger single statements\n"
            f"3. Cut anything vague (e.g. 'be authentic' is too vague)\n"
            f"4. Keep the 6 BEST items per category — these will guide a real AI streamer\n"
            f"5. Make each item specific and actionable\n\n"
            f"Return ONLY a JSON object with the same structure (tone, hooks, engagement, "
            f"teaching, avoid, examples). No markdown, no commentary."
        )

        result = self.agent.run_conversation(
            user_message=prompt,
            system_message="You are an editor refining a knowledge base. Output only JSON.",
            conversation_history=[],
            task_id="aria_learning_review",
        )

        response = result.get("final_response") or result.get("response") or ""
        consolidated = self._parse_json(response)

        if consolidated:
            # Preserve _meta
            consolidated["_meta"] = knowledge.get("_meta", {})
            consolidated["_meta"]["version"] = consolidated["_meta"].get("version", 0) + 1
            consolidated["_meta"]["last_updated"] = int(time.time())
            save_knowledge(consolidated)
            print(f"[Learning] Knowledge consolidated to v{consolidated['_meta']['version']}")
        else:
            print(f"[Learning] Could not parse consolidated knowledge")

    def _merge_knowledge(self, new_findings: dict):
        """Merge new findings into existing knowledge."""
        knowledge = load_knowledge()
        for category, items in new_findings.items():
            if category.startswith("_") or not isinstance(items, list):
                continue
            existing = knowledge.get(category, [])
            # Add new items (dedup by lowercase comparison)
            existing_lower = {str(e).lower() for e in existing}
            for item in items:
                if str(item).lower() not in existing_lower:
                    existing.append(item)
            knowledge[category] = existing[-30:]  # keep last 30 per category

        meta = knowledge.setdefault("_meta", {})
        meta["research_count"] = meta.get("research_count", 0) + 1
        meta["last_updated"] = int(time.time())
        save_knowledge(knowledge)

    def _parse_json(self, response: str) -> dict | None:
        """Extract JSON dict from agent response."""
        import re
        try:
            return json.loads(response)
        except Exception:
            pass

        # Try fenced blocks
        matches = re.findall(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
        for match in matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue

        # Try standalone {...}
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        return None


if __name__ == "__main__":
    print("Starting learning agent (standalone test)...")
    agent = LearningAgent(research_interval=60, review_interval=120)
    agent.start()
    try:
        while True:
            time.sleep(30)
            k = load_knowledge()
            print(f"\n[Knowledge] v{k.get('_meta', {}).get('version', 0)}, "
                  f"researches={k.get('_meta', {}).get('research_count', 0)}")
            for cat in ['tone', 'hooks', 'engagement', 'teaching', 'avoid']:
                items = k.get(cat, [])
                print(f"  {cat}: {len(items)} items")
    except KeyboardInterrupt:
        agent.stop()
