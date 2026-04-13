# Streaming Setup Knowledge

## Architecture Overview

The virtual live streamer runs as 3 processes:

1. **hermes_bridge.py** — HTTP API server that wraps Hermes Agent
   - POST /chat — viewer message → AI response + TTS audio with word timings
   - GET /idle — generate unprompted monologue
   - GET /feed?since=N — poll for new messages (used by frontend)
   - GET /audio/<file> — serve generated MP3 files

2. **Frontend (streamer.html)** — Browser-based 3D avatar
   - TalkingHead library (Three.js + WebGL)
   - Polls /feed every 2s for new messages
   - Decodes MP3 via Web Audio API → AudioBuffer
   - Generates visemes from word timings for lip sync
   - Renders mood expressions, hand gestures, idle animations

3. **twitch_bridge.py** — Twitch IRC WebSocket client
   - Connects to Twitch chat via wss://irc-ws.chat.twitch.tv
   - Forwards viewer messages to hermes_bridge /chat endpoint
   - Posts AI responses back to Twitch chat
   - Runs idle loop when chat is quiet

## TTS Pipeline

Edge-TTS with WordBoundary events provides:
- MP3 audio bytes
- Per-word timestamps (offset in 100ns ticks → ms)
- Per-word durations

These are cached to disk (output/tts_cache/) keyed by text+mood hash.

## Memory System

- Recent responses saved to data/memory/aria_memory.json
- Injected into system prompt as "DO NOT REPEAT THESE"
- Viewer facts structure ready for personalization
- Survives server restarts
