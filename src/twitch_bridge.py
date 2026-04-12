#!/usr/bin/env python3
"""Twitch Chat Bridge — reads Twitch chat, sends to Aria, posts replies back.

Uses twitchio v3 Client API (not Bot/commands).

Usage:
  ~/.hermes/hermes-agent/venv/bin/python src/twitch_bridge.py
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path

import httpx

# ─────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────
CONFIG_FILE = Path(__file__).parent.parent / "config" / "twitch.json"
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5001")
MIN_RESPONSE_INTERVAL = 5
IDLE_CHECK_INTERVAL = 45


def load_config() -> dict:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {}


def setup_interactive():
    """Interactive first-time setup."""
    print("\nNo Twitch config found. Let's set it up.\n")
    channel = input("Enter your Twitch channel name: ").strip()
    token = input("Enter your OAuth token (from twitchtokengenerator.com): ").strip()

    if not token.startswith("oauth:"):
        token = "oauth:" + token

    config = {"channel": channel, "token": token, "bot_name": channel}
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))
    print(f"\nSaved to {CONFIG_FILE}")
    return config


class AriaTwitchBridge:
    """Bridges Twitch chat to Aria's AI backend using twitchio v3 Client."""

    def __init__(self, channel: str, token: str, bot_name: str = ""):
        self.channel_name = channel
        self.bot_name = bot_name or channel
        self.token = token if token.startswith("oauth:") else f"oauth:{token}"
        self.last_response_time = 0
        self.http_client = httpx.AsyncClient(timeout=30)
        self.client = None

    async def start(self):
        """Connect to Twitch and start listening."""
        print(f"Connecting to Twitch IRC #{self.channel_name}...")
        await self._run_irc()

    async def _run_irc(self):
        """Connect to Twitch IRC directly (most reliable approach)."""
        import websockets

        uri = "wss://irc-ws.chat.twitch.tv:443"

        async with websockets.connect(uri) as ws:
            # Authenticate
            await ws.send(f"PASS {self.token}")
            await ws.send(f"NICK {self.bot_name}")
            await ws.send(f"JOIN #{self.channel_name}")
            await ws.send("CAP REQ :twitch.tv/tags twitch.tv/commands")

            print(f"Connected to Twitch IRC as {self.bot_name}")
            print(f"Listening to #{self.channel_name}")
            print(f"Backend: {BACKEND_URL}")
            print("=" * 50)

            # Start idle talk task
            idle_task = asyncio.create_task(self._idle_loop(ws))

            try:
                async for raw in ws:
                    # Handle PING/PONG
                    if raw.startswith("PING"):
                        await ws.send("PONG :tmi.twitch.tv")
                        continue

                    # Parse PRIVMSG (chat messages)
                    parsed = self._parse_irc(raw)
                    if parsed:
                        username, message = parsed
                        await self._handle_message(ws, username, message)

            except Exception as e:
                print(f"IRC error: {e}")
            finally:
                idle_task.cancel()

    def _parse_irc(self, raw: str) -> tuple[str, str] | None:
        """Parse a Twitch IRC PRIVMSG into (username, message)."""
        if "PRIVMSG" not in raw:
            return None

        try:
            # Format: @tags :user!user@user.tmi.twitch.tv PRIVMSG #channel :message
            # or: :user!user@user.tmi.twitch.tv PRIVMSG #channel :message
            parts = raw.split(" ")
            username = ""
            message = ""

            for i, part in enumerate(parts):
                if part == "PRIVMSG":
                    # Username is before PRIVMSG
                    user_part = parts[i - 1]
                    if "!" in user_part:
                        username = user_part.split("!")[0].lstrip(":")
                    # Message is after #channel :
                    msg_start = raw.index("PRIVMSG")
                    msg_parts = raw[msg_start:].split(":", 1)
                    if len(msg_parts) > 1:
                        message = msg_parts[1].strip()
                    break

            if username and message:
                # Ignore our own messages
                if username.lower() == self.bot_name.lower():
                    return None
                return username, message

        except Exception:
            pass

        return None

    async def _handle_message(self, ws, username: str, message: str):
        """Process a chat message and send Aria's response."""
        print(f"[Chat] {username}: {message}")

        # Rate limit
        now = time.time()
        if now - self.last_response_time < MIN_RESPONSE_INTERVAL:
            print(f"  (rate limited)")
            return

        try:
            resp = await self.http_client.post(
                f"{BACKEND_URL}/chat",
                json={"username": username, "message": message},
            )
            data = resp.json()
            response_text = data.get("response", "")

            if response_text:
                self.last_response_time = time.time()
                chat_text = self._clean_for_twitch(response_text)

                # Send to Twitch chat
                await ws.send(f"PRIVMSG #{self.channel_name} :{chat_text}")
                print(f"[Aria] {chat_text}")

        except Exception as e:
            print(f"  Error: {e}")

    async def _idle_loop(self, ws):
        """Send idle messages when chat is quiet."""
        while True:
            await asyncio.sleep(IDLE_CHECK_INTERVAL)

            silence = time.time() - self.last_response_time
            if silence >= IDLE_CHECK_INTERVAL:
                try:
                    resp = await self.http_client.get(f"{BACKEND_URL}/idle")
                    data = resp.json()
                    response_text = data.get("response", "")

                    if response_text:
                        self.last_response_time = time.time()
                        chat_text = self._clean_for_twitch(response_text)
                        await ws.send(f"PRIVMSG #{self.channel_name} :{chat_text}")
                        print(f"[Aria idle] {chat_text}")

                except Exception as e:
                    print(f"  Idle error: {e}")

    def _clean_for_twitch(self, text: str) -> str:
        """Clean text for Twitch chat (500 char limit)."""
        # Remove MEDIA: file paths that Hermes injects
        text = re.sub(r'MEDIA:\S+', '', text)
        # Remove markdown
        text = text.replace("*", "").replace("_", "").replace("~", "")
        # Remove emoji
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        # Remove file paths that might leak
        text = re.sub(r'/Users/\S+', '', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Twitch limit is 500 chars
        if len(text) > 490:
            text = text[:487] + "..."
        return text


async def main():
    print("=" * 50)
    print(" Aria — Twitch Chat Bridge")
    print("=" * 50)

    config = load_config()
    if not config.get("channel") or not config.get("token"):
        config = setup_interactive()

    bridge = AriaTwitchBridge(
        channel=config["channel"],
        token=config["token"],
        bot_name=config.get("bot_name", config["channel"]),
    )

    await bridge.start()


if __name__ == "__main__":
    asyncio.run(main())
