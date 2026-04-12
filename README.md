# Virtual Live Streamer

A 24/7 AI-powered virtual live streamer with a 3D animated avatar, real-time lip sync, and an autonomous AI brain powered by [Hermes Agent](https://github.com/NousResearch/hermes-agent).

## Demo

The streamer features:
- 3D animated avatar with lip-synced speech
- AI-powered personality that responds to chat
- Idle monologues when chat is quiet (never repeats itself)
- Mood-based expressions and gestures
- Persistent memory across sessions

## Architecture

```
Browser (TalkingHead 3D)  ←→  HTTP API  ←→  Hermes Agent  ←→  LLM (Claude/etc)
       ↑                          ↑
   Lip sync               Edge-TTS audio
   Animations              generation
   Chat UI
```

## Quick Start (macOS/Linux)

### 1. Clone the repo

```bash
git clone --recursive https://github.com/Chenglin97/virtual-live-streamer.git
cd virtual-live-streamer
```

### 2. Install Hermes Agent

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

This installs Hermes to `~/.hermes/hermes-agent/` with its own Python 3.11 venv.

### 3. Configure the LLM

Edit `~/.hermes/config.yaml`:

```yaml
model:
  default: "anthropic/claude-opus-4.6"   # or any model your provider supports
  provider: "custom"
  api_key: "your-api-key-here"
  base_url: "https://open.palebluedot.ai/v1"  # or openrouter.ai, etc
```

Or use OpenRouter (easiest — supports 200+ models):

```yaml
model:
  default: "anthropic/claude-opus-4.6"
  provider: "openrouter"
```

Then set your key in `~/.hermes/.env`:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 4. Install edge-tts (for voice)

```bash
~/.local/bin/uv pip install edge-tts --python ~/.hermes/hermes-agent/venv/bin/python
```

### 5. Clone TalkingHead (3D avatar frontend)

```bash
git clone https://github.com/met4citizen/TalkingHead.git talkinghead_3d
cp talkinghead_3d/streamer.html talkinghead_3d/streamer.html 2>/dev/null  # already included if cloned with --recursive
```

> Note: `talkinghead_3d/streamer.html` is our custom streamer UI — it should already be in the repo.

### 6. Start the backend

```bash
~/.hermes/hermes-agent/venv/bin/python src/hermes_bridge.py
```

You should see:
```
==================================================
 Aria — Virtual Streamer (Hermes Agent)
 http://localhost:5001
==================================================
```

### 7. Start the frontend

In a second terminal:

```bash
cd talkinghead_3d
python3 -m http.server 8080
```

### 8. Open the stream

Open http://localhost:8080/streamer.html in Chrome.

- Type a message in the chat box and click **Chat**
- The avatar will respond with voice + lip sync
- When chat is quiet, she'll talk on her own

## Customization

### Change the personality

Edit `~/.hermes/SOUL.md` — this defines the streamer's persona, loaded fresh on every message.

Or edit the `PERSONA` string in `src/hermes_bridge.py` for the streaming-specific personality.

### Change the voice

In `src/hermes_bridge.py`, change the `voice` parameter:

```python
edge_tts.Communicate(text=clean, voice="en-US-JennyNeural", rate="+5%")
```

Available voices: run `edge-tts --list-voices` or see [Edge TTS voices](https://speech.platform.bing.com/consumer/speech/synthesize/readaloud/voices/list).

### Change the 3D avatar

In `talkinghead_3d/streamer.html`, change the avatar URL in `showAvatar()`:

```javascript
await head.showAvatar({
  url: './avatars/brunette.glb',  // Change this to any GLB avatar
  body: 'F',                      // 'F' or 'M'
  avatarMood: 'happy',
  lipsyncLang: 'en'
});
```

Supports:
- [Ready Player Me](https://readyplayer.me/) avatars (GLB with ARKit blend shapes)
- [VRoid](https://vroid.com/) models
- Any GLB with Mixamo-compatible rig + ARKit/Oculus viseme blend shapes

### Change the LLM model

Edit `src/hermes_bridge.py`:

```python
agent = AIAgent(
    base_url="https://open.palebluedot.ai/v1",
    model="anthropic/claude-opus-4.6",
    api_key="your-key",
)
```

Works with any OpenAI-compatible API (OpenRouter, Nous Portal, local Ollama, etc).

## Project Structure

```
├── src/
│   ├── hermes_bridge.py     # Backend: Hermes Agent + TTS → HTTP API
│   ├── server.py            # Standalone backend (no Hermes, simpler)
│   ├── face_engine/         # Deep-Live-Cam face swap (earlier approach)
│   ├── talking_head/        # Wav2Lip talking head (earlier approach)
│   ├── agent/               # Direct LLM agent (without Hermes)
│   ├── tts/                 # TTS engine wrapper
│   ├── chat/                # Chat platform readers
│   ├── stream/              # RTMP streaming pipeline
│   └── orchestrator/        # Main control loop
├── talkinghead_3d/
│   └── streamer.html        # 3D avatar streamer UI (our custom page)
├── hermes-agent/            # Git submodule → Hermes Agent
├── config/                  # Configuration files
├── data/memory/             # Aria's persistent memory (auto-created)
├── output/tts_cache/        # Cached TTS audio files (auto-created)
└── scripts/                 # Test and utility scripts
```

## How It Works

1. **Viewer sends chat message** → Frontend (`streamer.html`) POSTs to backend
2. **Hermes Agent** receives the message with Aria's persona + memory context
3. **Claude** (or any LLM) generates a response in character
4. **Edge-TTS** converts the response to speech audio (MP3)
5. **Frontend** receives response text + audio URL
6. **TalkingHead** decodes the audio, generates visemes from text, and plays lip-synced speech
7. **Memory** saves what Aria said to prevent repetition

For idle mode: every 30 seconds of silence, the frontend calls `/idle` and Aria says something unprompted.

## Requirements

- macOS or Linux
- Python 3.11+ (installed by Hermes installer)
- Node.js (for TalkingHead — installed by Hermes installer)
- Chrome browser (WebGL for 3D rendering)
- An LLM API key (OpenRouter, PaleBlueDot, Anthropic, OpenAI, etc)

## License

MIT
