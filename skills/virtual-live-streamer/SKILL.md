---
name: virtual-live-streamer
description: Deploy a 24/7 AI-powered virtual live streamer with 3D avatar, real-time lip sync, TTS voice, and Twitch chat integration. Powered by Hermes Agent.
version: 1.0.0
author: Chenglin97
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Streaming, VTuber, TTS, 3D Avatar, Twitch, Live Stream, AI Streamer, Lip Sync]
    related_skills: [songwriting-and-ai-music, youtube-content]
prerequisites:
  commands: [ffmpeg, node]
  python_packages: [edge-tts, httpx, websockets]
---

# Virtual Live Streamer

Deploy a fully autonomous 24/7 AI virtual streamer with a 3D animated avatar that speaks, lip-syncs, responds to Twitch chat, and generates idle conversation — all powered by Hermes Agent.

## What It Does

- **3D Avatar** — Full-body animated character (TalkingHead/Three.js) with lip sync, gestures, mood expressions, and idle animations rendered in-browser
- **AI Brain** — Hermes Agent generates personality-driven responses via any LLM (Claude, GPT, Hermes, etc.)
- **Voice** — Edge-TTS with word-level timestamps for accurate lip sync
- **Twitch Chat** — Reads viewer messages via IRC, responds in chat AND through the avatar with voice
- **Idle Talk** — When chat is quiet, the streamer shares thoughts, asks questions, tells stories
- **Memory** — Persistent memory prevents repetition; remembers viewer details across sessions
- **Persona** — Fully customizable personality via SOUL.md

## Architecture

```
Twitch Chat ──→ twitch_bridge.py ──→ hermes_bridge.py ──→ Hermes Agent ──→ LLM
                                            │
                                      Edge-TTS + Word Timings
                                            │
                                      /feed endpoint
                                            │
Browser (TalkingHead 3D) ←── polls ─────────┘
   • Lip-synced speech
   • Mood expressions
   • Hand gestures
   • Chat overlay
```

## Quick Start

```bash
# 1. Clone
git clone --recursive https://github.com/Chenglin97/virtual-live-streamer.git
cd virtual-live-streamer

# 2. Install Hermes Agent (if not already installed)
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

# 3. Configure LLM in ~/.hermes/config.yaml
#    Set provider, model, api_key, base_url

# 4. Install TTS
~/.local/bin/uv pip install edge-tts --python ~/.hermes/hermes-agent/venv/bin/python

# 5. Clone TalkingHead frontend
git clone https://github.com/met4citizen/TalkingHead.git talkinghead_3d
cp streamer.html talkinghead_3d/streamer.html

# 6. Configure Twitch (interactive on first run)
# Creates config/twitch.json with channel + OAuth token

# 7. Start (3 terminals)
~/.hermes/hermes-agent/venv/bin/python src/hermes_bridge.py     # AI + TTS backend
cd talkinghead_3d && python3 -m http.server 8080                 # 3D avatar frontend
~/.hermes/hermes-agent/venv/bin/python src/twitch_bridge.py      # Twitch chat bridge
```

Open http://localhost:8080/streamer.html — capture in OBS — stream to Twitch.

## When to Use

- User wants to create an AI virtual streamer or VTuber
- User wants a 24/7 autonomous Twitch/YouTube stream
- User wants a 3D avatar that talks with lip sync
- User wants to build an AI companion with voice and visual presence

## When NOT to Use

- Face-swapping on real webcam video → use Deep-Live-Cam directly
- Simple chatbot without voice/visual → use Hermes chat or gateway
- Pre-recorded video production → use video editing tools

## Customization

| What | Where |
|------|-------|
| Personality | `~/.hermes/SOUL.md` or `PERSONA` in `src/hermes_bridge.py` |
| Voice | `voice` param in `generate_tts()` — any Edge-TTS voice |
| Avatar | `showAvatar({ url: ... })` in `streamer.html` — any GLB with ARKit blend shapes |
| LLM | `AIAgent(base_url=..., model=...)` in `src/hermes_bridge.py` |
| Stream platform | `config/twitch.json` — Twitch IRC; extendable to YouTube/Bilibili |

## Key Files

| File | Purpose |
|------|---------|
| `src/hermes_bridge.py` | Backend: Hermes Agent + TTS + memory + /feed API |
| `src/twitch_bridge.py` | Twitch IRC chat bridge |
| `streamer.html` | 3D avatar frontend with lip sync + chat UI |
| `data/memory/aria_memory.json` | Persistent memory (auto-created) |
| `config/twitch.json` | Twitch credentials (auto-created) |
