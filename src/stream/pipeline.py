"""Streaming pipeline — composites video + audio and pushes to RTMP."""

import asyncio
import logging
import shutil
import subprocess
from typing import Any

import numpy as np

logger = logging.getLogger("vls.stream")

Frame = np.ndarray


class StreamPipeline:
    """Manages the ffmpeg process that streams video+audio to an RTMP endpoint.

    Frames are written to ffmpeg's stdin as raw video. Audio is mixed in
    from TTS output via a named pipe or secondary input.
    """

    def __init__(self, config: dict[str, Any]):
        self.rtmp_url = config.get("rtmp_url", "")
        self.stream_key = config.get("stream_key", "")
        self.encoder = config.get("encoder", "libx264")
        self.bitrate = config.get("bitrate", "4500k")
        self.preset = config.get("preset", "veryfast")
        self.audio_bitrate = config.get("audio_bitrate", "160k")
        self.sample_rate = config.get("sample_rate", 44100)

        self._ffmpeg_process: subprocess.Popen | None = None
        self._width: int = 1280
        self._height: int = 720
        self._fps: int = 30

    async def initialize(self, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        """Verify ffmpeg is available and store stream dimensions."""
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg first.")

        self._width = width
        self._height = height
        self._fps = fps
        logger.info(
            "Stream pipeline ready (%dx%d @ %dfps -> %s)",
            width, height, fps, self.rtmp_url,
        )

    def start(self) -> None:
        """Start the ffmpeg streaming process."""
        full_url = f"{self.rtmp_url}/{self.stream_key}" if self.stream_key else self.rtmp_url

        cmd = [
            "ffmpeg",
            "-y",
            # Video input: raw frames from stdin
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self._width}x{self._height}",
            "-r", str(self._fps),
            "-i", "-",
            # Audio: silent audio track (TTS audio will be mixed in separately)
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=stereo:sample_rate={self.sample_rate}",
            # Output encoding
            "-c:v", self.encoder,
            "-b:v", self.bitrate,
            "-preset", self.preset,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", self.audio_bitrate,
            "-ar", str(self.sample_rate),
            "-shortest",
            "-f", "flv",
            full_url,
        ]

        logger.info("Starting ffmpeg stream to %s", self.rtmp_url)
        self._ffmpeg_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write_frame(self, frame: Frame) -> bool:
        """Write a single BGR frame to the ffmpeg process.

        Returns False if the stream is dead.
        """
        if self._ffmpeg_process is None or self._ffmpeg_process.stdin is None:
            return False

        try:
            # Resize if needed
            if frame.shape[1] != self._width or frame.shape[0] != self._height:
                import cv2
                frame = cv2.resize(frame, (self._width, self._height))

            self._ffmpeg_process.stdin.write(frame.tobytes())
            return True
        except BrokenPipeError:
            logger.error("ffmpeg pipe broken — stream may have died")
            return False

    def is_alive(self) -> bool:
        """Check if the ffmpeg process is still running."""
        if self._ffmpeg_process is None:
            return False
        return self._ffmpeg_process.poll() is None

    def stop(self) -> None:
        """Stop the streaming process gracefully."""
        if self._ffmpeg_process:
            if self._ffmpeg_process.stdin:
                self._ffmpeg_process.stdin.close()
            self._ffmpeg_process.wait(timeout=10)
            logger.info("Stream pipeline stopped")
            self._ffmpeg_process = None
