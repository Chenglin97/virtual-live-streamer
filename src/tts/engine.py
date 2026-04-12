"""Text-to-speech engine for the virtual streamer's voice."""

import asyncio
import io
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("vls.tts")


class TTSEngine:
    """Converts agent text responses into audio.

    Default backend is edge-tts (Microsoft Azure free voices).
    """

    def __init__(self, config: dict[str, Any]):
        self.engine_name = config.get("engine", "edge-tts")
        self.voice = config.get("voice", "en-US-JennyNeural")
        self.rate = config.get("rate", "+0%")
        self.volume = config.get("volume", "+0%")

    async def initialize(self) -> None:
        """Verify the TTS engine is available."""
        logger.info("Initializing TTS engine: %s (voice=%s)", self.engine_name, self.voice)

        if self.engine_name == "edge-tts":
            try:
                import edge_tts
            except ImportError:
                raise RuntimeError("edge-tts is required. Install with: pip install edge-tts")
        logger.info("TTS engine ready")

    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (MP3 format).

        Returns raw MP3 bytes that can be piped to ffmpeg or played directly.
        """
        if not text.strip():
            return b""

        if self.engine_name == "edge-tts":
            return await self._synthesize_edge_tts(text)
        else:
            raise ValueError(f"Unsupported TTS engine: {self.engine_name}")

    async def synthesize_to_file(self, text: str, output_path: Path) -> Path:
        """Convert text to audio and save to file."""
        audio_bytes = await self.synthesize(text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio_bytes)
        return output_path

    async def _synthesize_edge_tts(self, text: str) -> bytes:
        """Generate speech using Microsoft Edge TTS."""
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            volume=self.volume,
        )

        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        return b"".join(audio_chunks)

    async def shutdown(self) -> None:
        """Clean up TTS resources."""
        logger.info("TTS engine shut down")
