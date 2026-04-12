"""Talking head engine — generates video of a face speaking from audio.

Uses Wav2Lip to generate lip-synced video from a single face image + audio.
No webcam required — fully autonomous.
"""

import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

logger = logging.getLogger("vls.talking_head")

WAV2LIP_DIR = Path(__file__).parent.parent.parent / "wav2lip_model"


class TalkingHeadEngine:
    """Generates talking head video from a face image and audio.

    Pipeline: face image + audio → Wav2Lip → video frames
    """

    def __init__(self, config: dict[str, Any]):
        self.face_image_path = Path(config.get("source_face", "assets/faces/avatar.png"))
        self.resolution = config.get("resolution", {"width": 1280, "height": 720})
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self._model = None
        self._face_image = None
        self._face_det = None

    def initialize(self) -> None:
        """Load Wav2Lip model and prepare the source face."""
        logger.info("Initializing talking head engine (device=%s)", self.device)

        # Add wav2lip to path
        sys.path.insert(0, str(WAV2LIP_DIR))

        # Load face image
        if not self.face_image_path.exists():
            raise FileNotFoundError(f"Face image not found: {self.face_image_path}")

        self._face_image = cv2.imread(str(self.face_image_path))
        if self._face_image is None:
            raise ValueError(f"Could not load face image: {self.face_image_path}")

        logger.info("Face image loaded: %s", self._face_image.shape)

        # Load Wav2Lip model
        checkpoint_path = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Wav2Lip model not found: {checkpoint_path}")

        self._load_model(checkpoint_path)
        logger.info("Talking head engine ready")

    def _load_model(self, checkpoint_path: Path) -> None:
        """Load the Wav2Lip GAN model."""
        from models import Wav2Lip as Wav2LipModel

        model = Wav2LipModel()
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Clean up state dict keys
        cleaned = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            cleaned[k] = v

        model.load_state_dict(cleaned)
        model = model.to(self.device)
        model.eval()
        self._model = model
        logger.info("Wav2Lip model loaded on %s", self.device)

    def generate_video_from_audio(
        self,
        audio_path: str | Path,
        output_path: str | Path,
        fps: int = 25,
    ) -> Path:
        """Generate a talking head video from audio.

        Uses Wav2Lip's inference script as a subprocess for reliability.
        """
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Generating talking head video: %s -> %s", audio_path, output_path)

        # Use Wav2Lip inference as subprocess
        cmd = [
            sys.executable,
            str(WAV2LIP_DIR / "inference.py"),
            "--checkpoint_path", str(WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"),
            "--face", str(self.face_image_path),
            "--audio", str(audio_path),
            "--outfile", str(output_path),
            "--resize_factor", "1",
            "--fps", str(fps),
            "--pads", "0", "10", "0", "0",
            "--nosmooth",
        ]

        t0 = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(WAV2LIP_DIR),
            timeout=300,
        )

        elapsed = time.time() - t0

        if result.returncode != 0:
            logger.error("Wav2Lip failed: %s", result.stderr[-500:] if result.stderr else "no error")
            raise RuntimeError(f"Wav2Lip inference failed: {result.stderr[-200:]}")

        if not output_path.exists():
            raise RuntimeError("Wav2Lip produced no output file")

        logger.info("Video generated in %.1fs: %s", elapsed, output_path)
        return output_path

    def generate_idle_frames(self, num_frames: int = 150, fps: int = 25) -> list[np.ndarray]:
        """Generate idle animation frames (subtle head movement).

        Returns a list of frames that can be looped when the agent isn't talking.
        """
        face = self._face_image.copy()
        h, w = face.shape[:2]
        target_w = self.resolution["width"]
        target_h = self.resolution["height"]

        # Resize face to fit target resolution
        scale = min(target_w / w, target_h / h) * 0.6
        new_w, new_h = int(w * scale), int(h * scale)
        face_resized = cv2.resize(face, (new_w, new_h))

        frames = []
        for i in range(num_frames):
            # Create frame with dark background
            frame = np.full((target_h, target_w, 3), (25, 25, 30), dtype=np.uint8)

            # Subtle breathing / idle movement
            t = i / fps
            dx = int(3 * np.sin(2 * np.pi * t / 4))  # gentle sway
            dy = int(2 * np.sin(2 * np.pi * t / 3))  # gentle nod

            # Center the face
            x = (target_w - new_w) // 2 + dx
            y = (target_h - new_h) // 2 + dy

            # Paste face onto frame
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(target_w, x + new_w), min(target_h, y + new_h)
            fx1, fy1 = x1 - x, y1 - y
            fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)

            frame[y1:y2, x1:x2] = face_resized[fy1:fy2, fx1:fx2]
            frames.append(frame)

        return frames

    def release(self) -> None:
        """Clean up resources."""
        self._model = None
        self._face_image = None
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        logger.info("Talking head engine released")
