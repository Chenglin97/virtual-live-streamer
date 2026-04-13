# Persona Management Strategy

## Core Persona

The streamer persona is defined in the PERSONA constant in hermes_bridge.py and can be customized via ~/.hermes/SOUL.md.

## Key Behaviors

1. **Always address viewers by username** — creates personal connection
2. **Keep responses short (1-3 sentences)** — live stream pacing
3. **Never repeat yourself** — memory system tracks recent lines
4. **Mood-reactive expressions** — happy/sad/angry/neutral mapped to TTS voice settings and avatar expressions
5. **Idle talk variety** — when chat is quiet, teach concepts, ask questions, share stories

## Voice Settings Per Mood

| Mood | Voice | Rate | Pitch |
|------|-------|------|-------|
| happy | AvaNeural | +10% | +2Hz |
| angry | AvaNeural | +8% | +0Hz |
| sad | AvaNeural | -8% | -4Hz |
| neutral | AvaNeural | +3% | +0Hz |

## Memory Integration

The system prompt includes recent responses as a "DO NOT REPEAT" list. This prevents the LLM from recycling the same jokes, stories, or observations.
