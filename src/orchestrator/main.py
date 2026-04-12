"""Main orchestrator — ties all components together for 24/7 streaming."""

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.face_engine import FaceEngine
from src.agent import StreamerAgent
from src.tts import TTSEngine
from src.chat import ChatReader
from src.stream import StreamPipeline

logger = logging.getLogger("vls.orchestrator")


class Orchestrator:
    """Main control loop that coordinates all streamer components.

    Manages the lifecycle of face engine, AI agent, TTS, chat reader,
    and stream pipeline. Handles health checks, auto-restart, and
    graceful shutdown.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        orch_config = config.get("orchestrator", {})

        self.health_check_interval = orch_config.get("health_check_interval", 30)
        self.auto_restart = orch_config.get("auto_restart", True)
        self.max_restart_attempts = orch_config.get("max_restart_attempts", 5)
        self.restart_cooldown = orch_config.get("restart_cooldown_seconds", 30)

        # Components
        self.face_engine = FaceEngine(config.get("face_engine", {}))
        self.agent = StreamerAgent(config.get("agent", {}))
        self.tts = TTSEngine(config.get("tts", {}))
        self.chat_reader = ChatReader(config.get("chat", {}))
        self.stream = StreamPipeline(config.get("stream", {}))

        self._running = False
        self._restart_count = 0

    async def start(self) -> None:
        """Initialize all components and start the main loop."""
        logger.info("=" * 60)
        logger.info("Virtual Live Streamer starting up")
        logger.info("=" * 60)

        # Initialize components
        self.face_engine.initialize()

        face_config = self.config.get("face_engine", {})
        resolution = face_config.get("resolution", {"width": 1280, "height": 720})
        fps = face_config.get("fps", 30)

        await asyncio.gather(
            self.agent.initialize(),
            self.tts.initialize(),
            self.chat_reader.initialize(),
            self.stream.initialize(
                width=resolution["width"],
                height=resolution["height"],
                fps=fps,
            ),
        )

        # Start streaming
        self.stream.start()
        await self.chat_reader.start()

        self._running = True
        logger.info("All components initialized — entering main loop")

        # Run main loops concurrently
        await asyncio.gather(
            self._video_loop(fps),
            self._chat_loop(),
            self._idle_talk_loop(),
            self._health_check_loop(),
        )

    async def _video_loop(self, fps: int) -> None:
        """Capture video, apply face swap, and write to stream."""
        base_video = self.config.get("face_engine", {}).get("base_video", "webcam")

        if base_video == "webcam":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(base_video)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {base_video}")

        frame_interval = 1.0 / fps
        logger.info("Video loop started (source=%s, fps=%d)", base_video, fps)

        try:
            while self._running:
                start_time = time.monotonic()

                ret, frame = cap.read()
                if not ret:
                    if base_video != "webcam":
                        # Loop video file
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        logger.error("Webcam read failed")
                        await asyncio.sleep(0.1)
                        continue

                # Face swap
                swapped_frame = self.face_engine.swap_face(frame)

                # Write to stream
                if not self.stream.write_frame(swapped_frame):
                    logger.error("Failed to write frame to stream")
                    if self.auto_restart:
                        await self._restart_stream()

                # Frame rate control
                elapsed = time.monotonic() - start_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        finally:
            cap.release()

    async def _chat_loop(self) -> None:
        """Process incoming chat messages and generate responses."""
        logger.info("Chat loop started")

        async for msg in self.chat_reader.messages():
            if not self._running:
                break

            logger.debug("Chat [%s]: %s", msg.username, msg.message)

            # Handle special events
            if msg.is_donation:
                response = await self.agent.react_to_event(
                    f"{msg.username} donated ${msg.donation_amount}!"
                )
            elif msg.is_subscription:
                response = await self.agent.react_to_event(
                    f"{msg.username} just subscribed!"
                )
            else:
                response = await self.agent.respond_to_chat(msg.username, msg.message)

            if response:
                logger.info("Agent says: %s", response)
                # Synthesize and play TTS
                audio = await self.tts.synthesize(response)
                # TODO: Mix audio into the stream pipeline

    async def _idle_talk_loop(self) -> None:
        """Generate idle chatter when chat is quiet."""
        interval = self.config.get("agent", {}).get("idle_talk_interval_seconds", 60)
        logger.info("Idle talk loop started (interval=%ds)", interval)

        while self._running:
            await asyncio.sleep(interval)
            if not self._running:
                break

            response = await self.agent.idle_talk()
            if response:
                logger.info("Agent (idle): %s", response)
                audio = await self.tts.synthesize(response)
                # TODO: Mix audio into the stream pipeline

    async def _health_check_loop(self) -> None:
        """Periodically check that all components are healthy."""
        while self._running:
            await asyncio.sleep(self.health_check_interval)
            if not self._running:
                break

            if not self.stream.is_alive():
                logger.warning("Stream pipeline is dead!")
                if self.auto_restart:
                    await self._restart_stream()

    async def _restart_stream(self) -> None:
        """Attempt to restart the stream pipeline."""
        if self._restart_count >= self.max_restart_attempts:
            logger.error("Max restart attempts reached — shutting down")
            self._running = False
            return

        self._restart_count += 1
        logger.warning(
            "Restarting stream (attempt %d/%d)",
            self._restart_count, self.max_restart_attempts,
        )

        self.stream.stop()
        await asyncio.sleep(self.restart_cooldown)
        self.stream.start()

    async def stop(self) -> None:
        """Gracefully shut down all components."""
        logger.info("Shutting down...")
        self._running = False

        self.stream.stop()
        await self.chat_reader.stop()
        await self.agent.shutdown()
        await self.tts.shutdown()
        self.face_engine.release()

        logger.info("Virtual Live Streamer stopped")


def main():
    """Entry point."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    config = load_config(config_path)

    orch_config = config.get("orchestrator", {})
    setup_logging(
        level=orch_config.get("log_level", "INFO"),
        log_file=orch_config.get("log_file"),
    )

    orchestrator = Orchestrator(config)

    loop = asyncio.new_event_loop()

    # Handle Ctrl+C gracefully
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: loop.create_task(orchestrator.stop()))

    try:
        loop.run_until_complete(orchestrator.start())
    except KeyboardInterrupt:
        loop.run_until_complete(orchestrator.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
