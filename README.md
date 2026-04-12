# Virtual Live Streamer

A 24/7 AI-powered virtual live streamer that combines real-time face swapping with an autonomous AI agent.

## Architecture

```
                    +-------------------+
                    |   Orchestrator    |
                    | (main loop/ctrl)  |
                    +--------+----------+
                             |
          +------------------+------------------+
          |                  |                  |
+---------v-----+  +---------v-----+  +---------v-----+
|  Face Engine  |  |   AI Agent    |  |    Stream      |
| (Deep-Live-   |  |  (Hermes +    |  |   Pipeline     |
|    Cam)       |  |   Chat/TTS)   |  |  (RTMP/OBS)    |
+---------------+  +---------------+  +---------------+
```

### Components

- **Face Engine** — Real-time face swapping powered by [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam). Takes a source face image and applies it to a base video/webcam feed frame-by-frame.
- **AI Agent** — Autonomous conversational agent powered by [Hermes Agent](https://github.com/nousresearch/hermes-agent). Reads chat, generates responses, maintains personality and memory across sessions.
- **TTS** — Text-to-speech synthesis to give the virtual streamer a voice (edge-tts / VITS / other).
- **Chat** — Reads live chat from streaming platforms (YouTube, Twitch, Bilibili) and feeds messages to the AI agent.
- **Stream Pipeline** — Composites face-swapped video + audio and pushes to RTMP endpoints via ffmpeg.
- **Orchestrator** — Main control loop that coordinates all components, handles scheduling, health checks, and auto-restart for 24/7 operation.

## Prerequisites

- Python 3.11+
- ffmpeg
- CUDA-capable GPU (recommended) or Apple Silicon
- OBS Studio (optional, for advanced scene management)

## Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/Chenglin97/virtual-live-streamer.git
cd virtual-live-streamer

# Install dependencies
pip install -e ".[all]"

# Download required models
python scripts/download_models.py

# Configure
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your settings (stream key, LLM provider, face image, etc.)

# Run
python -m src.orchestrator.main
```

## Configuration

All configuration lives in `config/config.yaml`. Key sections:

| Section | Description |
|---------|-------------|
| `face_engine` | Source face image, execution provider (cuda/coreml), quality settings |
| `agent` | LLM provider, model, personality prompt, memory settings |
| `tts` | Voice engine, voice ID, speed/pitch |
| `stream` | RTMP URL, stream key, resolution, bitrate, encoder |
| `chat` | Platform (twitch/youtube/bilibili), channel ID, auth tokens |
| `orchestrator` | Health check interval, auto-restart policy, schedule |

## Project Structure

```
src/
  face_engine/     # Deep-Live-Cam integration — frame-level face swap
  agent/           # Hermes Agent wrapper — personality, conversation, memory
  tts/             # Text-to-speech synthesis
  chat/            # Live chat readers (Twitch/YouTube/Bilibili)
  stream/          # RTMP streaming pipeline via ffmpeg
  orchestrator/    # Main control loop, scheduling, health monitoring
  utils/           # Shared helpers (logging, metrics, config loader)
config/            # Configuration files
assets/
  faces/           # Source face images
  overlays/        # Stream overlay assets
scripts/           # Setup and utility scripts
tests/             # Test suite
```

## License

MIT
