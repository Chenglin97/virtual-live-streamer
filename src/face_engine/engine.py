"""Face swapping engine — wraps Deep-Live-Cam's face swap pipeline.

This module provides a clean interface to Deep-Live-Cam's frame processors,
operating on raw numpy frames so it can plug into any video pipeline.
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("vls.face_engine")

# Type alias matching Deep-Live-Cam convention
Frame = np.ndarray


class FaceEngine:
    """Real-time face swapping engine.

    Wraps Deep-Live-Cam's InsightFace + ONNX pipeline to process individual
    video frames. Call `swap_face(frame)` to get a face-swapped frame back.
    """

    def __init__(self, config: dict[str, Any]):
        self.source_face_path = Path(config["source_face"])
        self.execution_provider = config.get("execution_provider", "cpu")
        self.face_enhancer_name = config.get("face_enhancer")
        self.resolution = config.get("resolution", {"width": 1280, "height": 720})
        self.fps = config.get("fps", 30)

        self._face_analyser = None
        self._face_swapper = None
        self._face_enhancer = None
        self._source_face = None

    def initialize(self) -> None:
        """Load models and prepare the source face embedding."""
        logger.info("Initializing face engine (provider=%s)", self.execution_provider)

        # These imports come from Deep-Live-Cam (added as a submodule or installed)
        # We defer imports so the module can be loaded without the heavy deps present
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise RuntimeError(
                "insightface is required. Install with: pip install insightface==0.7.3"
            )

        providers = self._get_execution_providers()

        # Face analyser
        self._face_analyser = FaceAnalysis(
            name="buffalo_l", providers=providers
        )
        self._face_analyser.prepare(ctx_id=0, det_size=(640, 640))

        # Face swapper model
        import onnxruntime

        model_path = Path("models/inswapper_128_fp16.onnx")
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face swap model not found at {model_path}. "
                "Run: python scripts/download_models.py"
            )

        self._face_swapper = insightface.model_zoo.get_model(
            str(model_path), providers=providers
        )

        # Face enhancer (optional)
        if self.face_enhancer_name:
            self._init_enhancer(providers)

        # Load and analyse source face
        if not self.source_face_path.exists():
            raise FileNotFoundError(f"Source face image not found: {self.source_face_path}")

        source_img = cv2.imread(str(self.source_face_path))
        faces = self._face_analyser.get(source_img)
        if not faces:
            raise ValueError(f"No face detected in source image: {self.source_face_path}")

        self._source_face = sorted(faces, key=lambda f: f.bbox[0])[0]
        logger.info("Face engine initialized — source face loaded")

    def swap_face(self, frame: Frame) -> Frame:
        """Detect faces in frame and swap with the source face.

        Returns the processed frame (or original if no face detected).
        """
        if self._face_analyser is None:
            raise RuntimeError("Face engine not initialized. Call initialize() first.")

        faces = self._face_analyser.get(frame)
        if not faces:
            return frame

        result = frame.copy()
        for face in faces:
            result = self._face_swapper.get(result, face, self._source_face, paste_back=True)

        if self._face_enhancer is not None:
            result = self._enhance_faces(result, faces)

        return result

    def _get_execution_providers(self) -> list[str]:
        provider_map = {
            "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
            "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
            "openvino": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
            "cpu": ["CPUExecutionProvider"],
        }
        return provider_map.get(self.execution_provider, ["CPUExecutionProvider"])

    def _init_enhancer(self, providers: list[str]) -> None:
        """Initialize face enhancement model (GFPGAN or GPEN)."""
        enhancer_path = Path(f"models/GFPGANv1.4.onnx")
        if enhancer_path.exists():
            import onnxruntime

            self._face_enhancer = onnxruntime.InferenceSession(
                str(enhancer_path), providers=providers
            )
            logger.info("Face enhancer loaded: %s", self.face_enhancer_name)
        else:
            logger.warning("Enhancer model not found at %s — skipping", enhancer_path)

    def _enhance_faces(self, frame: Frame, faces: list) -> Frame:
        """Apply face enhancement (stub — full implementation uses GFPGAN pipeline)."""
        # TODO: Implement full GFPGAN enhancement pipeline
        # For now, returns frame as-is when enhancer model loading is more complex
        return frame

    def release(self) -> None:
        """Clean up resources."""
        self._face_analyser = None
        self._face_swapper = None
        self._face_enhancer = None
        self._source_face = None
        logger.info("Face engine released")
