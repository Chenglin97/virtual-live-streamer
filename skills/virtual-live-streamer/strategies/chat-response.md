# Chat Response Strategy

## Message Flow

1. Viewer sends message on Twitch
2. twitch_bridge.py receives via IRC WebSocket
3. POST to hermes_bridge /chat with {username, message}
4. Hermes Agent generates response with persona + memory context
5. Edge-TTS generates audio with word-level timestamps
6. Response + audio URL added to /feed
7. Frontend polls /feed, displays message, speaks with lip sync
8. twitch_bridge posts response text to Twitch chat

## Rate Limiting

- Minimum 5 seconds between responses (twitch_bridge)
- Speech queue ensures one clip at a time (frontend)
- Idle talk blocked while speaking

## Error Handling

- If Hermes fails: fallback response "my brain glitched"
- If TTS fails: response text shown without audio
- If frontend audio fails: fallback to HTML Audio element
- If Twitch IRC disconnects: auto-reconnect on next message
