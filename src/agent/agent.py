"""AI agent for the virtual streamer — handles conversation, personality, and memory.

Can use Hermes Agent as the backend or fall back to direct LLM API calls.
"""

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger("vls.agent")


class StreamerAgent:
    """Autonomous conversational agent for the virtual streamer.

    Reads chat messages, generates responses with a consistent personality,
    and produces idle chatter when chat is quiet.
    """

    def __init__(self, config: dict[str, Any]):
        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-4o")
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url")
        self.personality = config.get("personality", "You are a friendly virtual streamer.")
        self.max_response_length = config.get("max_response_length", 200)
        self.response_cooldown = config.get("response_cooldown_seconds", 5)
        self.idle_talk_interval = config.get("idle_talk_interval_seconds", 60)

        self._client = None
        self._conversation_history: list[dict[str, str]] = []
        self._last_response_time: float = 0

    async def initialize(self) -> None:
        """Set up the LLM client."""
        logger.info("Initializing streamer agent (provider=%s, model=%s)", self.provider, self.model)

        if self.provider in ("openai", "openrouter", "nous"):
            from openai import AsyncOpenAI

            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            elif self.provider == "openrouter":
                kwargs["base_url"] = "https://openrouter.ai/api/v1"

            self._client = AsyncOpenAI(**kwargs)
            self._api_type = "openai"

        elif self.provider == "anthropic":
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.api_key)
            self._api_type = "anthropic"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        self._conversation_history = [
            {"role": "system", "content": self.personality}
        ]
        logger.info("Streamer agent initialized")

    async def respond_to_chat(self, username: str, message: str) -> str | None:
        """Generate a response to a chat message.

        Returns None if on cooldown.
        """
        now = time.time()
        if now - self._last_response_time < self.response_cooldown:
            return None

        self._conversation_history.append(
            {"role": "user", "content": f"[Chat from {username}]: {message}"}
        )

        response = await self._generate_response()
        self._last_response_time = time.time()
        return response

    async def idle_talk(self) -> str:
        """Generate idle chatter when chat is quiet."""
        self._conversation_history.append(
            {
                "role": "user",
                "content": (
                    "[System: Chat has been quiet. Say something entertaining to keep "
                    "the audience engaged — share a thought, tell a short story, "
                    "comment on what you're doing, or ask the audience a question.]"
                ),
            }
        )
        return await self._generate_response()

    async def react_to_event(self, event_description: str) -> str:
        """Generate a reaction to a stream event (new follower, donation, etc)."""
        self._conversation_history.append(
            {"role": "user", "content": f"[Stream event: {event_description}]"}
        )
        return await self._generate_response()

    async def _generate_response(self) -> str:
        """Call the LLM and return the response text."""
        try:
            if self._api_type == "openai":
                completion = await self._client.chat.completions.create(
                    model=self.model,
                    messages=self._conversation_history,
                    max_tokens=self.max_response_length,
                    temperature=0.9,
                )
                text = completion.choices[0].message.content or ""

            elif self._api_type == "anthropic":
                # Anthropic expects system message separately
                system_msg = self._conversation_history[0]["content"]
                messages = [
                    m for m in self._conversation_history[1:]
                    if m["role"] in ("user", "assistant")
                ]
                response = await self._client.messages.create(
                    model=self.model,
                    system=system_msg,
                    messages=messages,
                    max_tokens=self.max_response_length,
                    temperature=0.9,
                )
                text = response.content[0].text
            else:
                text = ""

            # Track in history
            self._conversation_history.append({"role": "assistant", "content": text})

            # Trim history to avoid token overflow (keep system + last 50 exchanges)
            if len(self._conversation_history) > 102:
                self._conversation_history = (
                    self._conversation_history[:1] + self._conversation_history[-100:]
                )

            return text.strip()

        except Exception as e:
            logger.error("Agent response failed: %s", e)
            return "Hmm, let me think about that..."

    async def shutdown(self) -> None:
        """Clean up agent resources."""
        self._conversation_history.clear()
        logger.info("Streamer agent shut down")
