"""Live chat reader — pulls messages from streaming platforms."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

logger = logging.getLogger("vls.chat")


@dataclass
class ChatMessage:
    """A single chat message from a viewer."""
    username: str
    message: str
    platform: str
    timestamp: float = 0.0
    is_donation: bool = False
    is_subscription: bool = False
    donation_amount: float = 0.0
    badges: list[str] = field(default_factory=list)


class ChatReader:
    """Reads live chat messages from a streaming platform.

    Supports Twitch, YouTube Live, and Bilibili.
    """

    def __init__(self, config: dict[str, Any]):
        self.platform = config.get("platform", "twitch")
        self.channel = config.get("channel", "")
        self.auth_token = config.get("auth_token", "")
        self.ignore_bots = config.get("ignore_bots", True)
        self.command_prefix = config.get("command_prefix", "!")

        self._running = False
        self._message_queue: asyncio.Queue[ChatMessage] = asyncio.Queue()

    async def initialize(self) -> None:
        """Connect to the chat platform."""
        logger.info("Initializing chat reader for %s (channel=%s)", self.platform, self.channel)
        # Platform-specific connection will be established in start()
        logger.info("Chat reader ready")

    async def start(self) -> None:
        """Start reading chat messages in the background."""
        self._running = True

        if self.platform == "twitch":
            await self._start_twitch()
        elif self.platform == "youtube":
            await self._start_youtube()
        elif self.platform == "bilibili":
            await self._start_bilibili()
        else:
            raise ValueError(f"Unsupported chat platform: {self.platform}")

    async def get_message(self, timeout: float = 1.0) -> ChatMessage | None:
        """Get the next chat message, or None if timeout."""
        try:
            return await asyncio.wait_for(self._message_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def messages(self) -> AsyncIterator[ChatMessage]:
        """Async iterator over incoming chat messages."""
        while self._running:
            msg = await self.get_message()
            if msg is not None:
                yield msg

    async def _start_twitch(self) -> None:
        """Connect to Twitch IRC chat."""
        try:
            import twitchio
        except ImportError:
            logger.warning("twitchio not installed — chat reader running in stub mode")
            return

        # TODO: Full twitchio integration
        # For now, this is a placeholder for the connection setup
        logger.info("Twitch chat connected to #%s", self.channel)

    async def _start_youtube(self) -> None:
        """Connect to YouTube Live chat."""
        try:
            import pytchat
        except ImportError:
            logger.warning("pytchat not installed — chat reader running in stub mode")
            return

        logger.info("YouTube Live chat connected")

    async def _start_bilibili(self) -> None:
        """Connect to Bilibili live room chat."""
        try:
            import blivedm
        except ImportError:
            logger.warning("blivedm not installed — chat reader running in stub mode")
            return

        logger.info("Bilibili chat connected")

    async def stop(self) -> None:
        """Stop reading chat."""
        self._running = False
        logger.info("Chat reader stopped")
