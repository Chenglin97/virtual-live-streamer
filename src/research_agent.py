#!/usr/bin/env python3
"""Research sub-agent — continuously finds AI news and feeds topics to the streamer.

Runs as a background thread. Uses Hermes Agent with web tools to research
AI news, papers, launches, and trending topics. Writes findings to a
topic queue that the speech pre-gen system reads from.

The main streamer (Aria) then talks about these researched topics like an
AI educator going through the latest news.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

# Hermes Agent setup
HERMES_DIR = Path(__file__).parent.parent / "hermes-agent"
if str(HERMES_DIR) not in sys.path:
    sys.path.insert(0, str(HERMES_DIR))

_src = str(Path(__file__).parent)
while _src in sys.path:
    sys.path.remove(_src)

from run_agent import AIAgent

TOPIC_QUEUE_FILE = Path(__file__).parent.parent / "data" / "topic_queue.json"
TOPIC_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)

RESEARCH_PROMPT = """You are a research assistant for Aria, an AI educator streamer who teaches people how to BUILD with AI.

Find the LATEST AI news that is ACTIONABLE — things people can use, build with, or profit from.

Search the web for:
1. New AI model releases — with benchmark numbers, pricing, what they're best for
2. New AI tools/frameworks — with install commands and what problems they solve
3. Specific AI tutorials or techniques going viral — methods people are actually using
4. AI business opportunities — real companies making money with AI, how much, how
5. Open source projects — specific repos, star counts, what they do, how to use them
6. Cost changes — API price drops, free tiers, cheaper alternatives

For EACH finding, provide:
- topic: Specific title (e.g. "Llama 4 Scout 109B runs free on Groq at 200 tok/s" NOT "New AI model released")
- summary: What it is + specific numbers (price, speed, accuracy)
- why_interesting: What can people BUILD with this? What value does it create?
- how_to_use: Step by step how a developer would actually use this (commands, URLs, code patterns)
- talking_points: 4-5 specific concrete points — real numbers, real tool names, real comparisons

Return as JSON array. Find 5-8 topics. Be SPECIFIC — no generalities.
"""


def load_topic_queue() -> list:
    if TOPIC_QUEUE_FILE.exists():
        try:
            return json.loads(TOPIC_QUEUE_FILE.read_text())
        except Exception:
            return []
    return []


def save_topic_queue(topics: list):
    TOPIC_QUEUE_FILE.write_text(json.dumps(topics, indent=2, ensure_ascii=False))


def pop_topic() -> dict | None:
    """Get the next topic from the queue (FIFO). Returns None if empty."""
    topics = load_topic_queue()
    if not topics:
        return None
    topic = topics.pop(0)
    save_topic_queue(topics)
    return topic


def queue_size() -> int:
    return len(load_topic_queue())


class ResearchAgent:
    """Background agent that researches AI news and fills the topic queue."""

    def __init__(self, research_interval: int = 600):
        """
        Args:
            research_interval: Seconds between research runs (default 10 min)
        """
        self.research_interval = research_interval
        self.agent = AIAgent(
            base_url="https://open.palebluedot.ai/v1",
            model="anthropic/claude-opus-4.6",
            api_key="sk-YuA06jNFuzJoyaJha8C4CYOrvECdGQ1LtkEEESfUkuT6UY6d",
        )
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._research_loop, daemon=True)
        self._thread.start()
        print(f"[Research] Agent started (interval={self.research_interval}s)")

    def stop(self):
        self._running = False

    def _research_loop(self):
        """Continuously research and fill the topic queue."""
        while self._running:
            current_size = queue_size()

            if current_size < 5:
                print(f"[Research] Queue low ({current_size} topics), researching...")
                try:
                    self._do_research()
                except Exception as e:
                    print(f"[Research] Error: {e}")

            # Wait before next research cycle
            for _ in range(self.research_interval):
                if not self._running:
                    return
                time.sleep(1)

    def _do_research(self):
        """Run one research cycle using Hermes Agent."""
        result = self.agent.run_conversation(
            user_message=RESEARCH_PROMPT,
            system_message=(
                "You are a research assistant. Use your web search and browsing tools "
                "to find the very latest AI news. Return structured JSON. "
                "Be thorough — check multiple sources."
            ),
            conversation_history=[],
        )

        response = result.get("final_response") or result.get("response") or ""

        # Try to extract JSON from the response
        topics = self._parse_topics(response)

        if topics:
            existing = load_topic_queue()
            # Deduplicate by topic title
            existing_titles = {t.get("topic", "").lower() for t in existing}
            new_topics = [t for t in topics if t.get("topic", "").lower() not in existing_titles]

            if new_topics:
                existing.extend(new_topics)
                save_topic_queue(existing)
                print(f"[Research] Added {len(new_topics)} new topics (total: {len(existing)})")
                for t in new_topics:
                    print(f"  + {t.get('topic', 'untitled')}")
            else:
                print(f"[Research] No new topics found")
        else:
            print(f"[Research] Could not parse topics from response")

    def _parse_topics(self, response: str) -> list:
        """Extract topic list from agent response."""
        # Try direct JSON parse
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return data
        except Exception:
            pass

        # Try to find JSON array in the response
        import re
        matches = re.findall(r'\[[\s\S]*?\]', response)
        for match in matches:
            try:
                data = json.loads(match)
                if isinstance(data, list) and len(data) > 0:
                    return data
            except Exception:
                continue

        # Try to find JSON between code fences
        fenced = re.findall(r'```(?:json)?\s*([\s\S]*?)```', response)
        for block in fenced:
            try:
                data = json.loads(block)
                if isinstance(data, list):
                    return data
            except Exception:
                continue

        return []


# ── For testing standalone ──
if __name__ == "__main__":
    print("Starting research agent (standalone test)...")
    agent = ResearchAgent(research_interval=60)
    agent.start()

    try:
        while True:
            time.sleep(10)
            print(f"[Queue] {queue_size()} topics available")
            topics = load_topic_queue()
            for t in topics[:3]:
                print(f"  - {t.get('topic', '?')}")
    except KeyboardInterrupt:
        agent.stop()
        print("Stopped.")
