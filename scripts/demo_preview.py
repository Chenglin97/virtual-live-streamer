#!/usr/bin/env python3
"""Demo script — captures webcam, applies face swap, shows a live preview window.

This is for local testing without needing an RTMP stream.
Usage: python scripts/demo_preview.py [--source assets/faces/avatar.png]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def create_sample_face(output_path: Path) -> None:
    """Generate a simple placeholder face image for testing."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Draw a simple face outline
    cv2.circle(img, (256, 256), 200, (200, 180, 160), -1)  # head
    cv2.circle(img, (190, 210), 25, (255, 255, 255), -1)   # left eye white
    cv2.circle(img, (322, 210), 25, (255, 255, 255), -1)   # right eye white
    cv2.circle(img, (190, 210), 12, (60, 40, 20), -1)      # left pupil
    cv2.circle(img, (322, 210), 12, (60, 40, 20), -1)      # right pupil
    cv2.ellipse(img, (256, 310), (60, 30), 0, 0, 180, (100, 60, 60), 3)  # mouth
    cv2.imwrite(str(output_path), img)
    print(f"  Created placeholder face: {output_path}")


def run_preview_no_swap() -> None:
    """Run webcam preview without face swapping (test video pipeline only)."""
    print("\n--- Webcam Preview (no face swap) ---")
    print("Press 'q' to quit\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        print("  On macOS, check System Settings > Privacy & Security > Camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_counter = 0
    fps_time = time.time()
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # FPS counter
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_time = time.time()

        # Add info overlay
        cv2.putText(frame, f"FPS: {fps_display}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Virtual Live Streamer - Preview", (10, 70),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "No face swap (models loading or source not set)", (10, 110),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

        cv2.imshow("Virtual Live Streamer - Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_preview_with_swap(source_face_path: str) -> None:
    """Run webcam preview with face swapping."""
    print("\n--- Face Swap Preview ---")
    print(f"Source face: {source_face_path}")
    print("Press 'q' to quit\n")

    # Import face engine
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.face_engine.engine import FaceEngine

    config = {
        "source_face": source_face_path,
        "execution_provider": "coreml",
        "face_enhancer": None,
        "resolution": {"width": 1280, "height": 720},
        "fps": 30,
    }

    engine = FaceEngine(config)
    print("Loading face engine (this may take a moment)...")
    try:
        engine.initialize()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run: python scripts/download_models.py")
        return
    except Exception as e:
        print(f"ERROR initializing face engine: {e}")
        print("Falling back to webcam-only preview...")
        run_preview_no_swap()
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_counter = 0
    fps_time = time.time()
    fps_display = 0
    process_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Face swap
        t0 = time.time()
        swapped = engine.swap_face(frame)
        process_time = (time.time() - t0) * 1000
        process_times.append(process_time)
        if len(process_times) > 30:
            process_times.pop(0)

        # FPS counter
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_time = time.time()

        avg_ms = sum(process_times) / len(process_times)

        # Info overlay
        cv2.putText(swapped, f"FPS: {fps_display}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(swapped, f"Swap: {avg_ms:.1f}ms", (10, 70),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(swapped, "Virtual Live Streamer - Face Swap Active", (10, 110),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Side-by-side: original | swapped
        h, w = frame.shape[:2]
        small_orig = cv2.resize(frame, (w // 4, h // 4))
        swapped[10:10 + h // 4, w - w // 4 - 10:w - 10] = small_orig

        cv2.imshow("Virtual Live Streamer - Face Swap", swapped)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    engine.release()


def main():
    parser = argparse.ArgumentParser(description="Demo preview for Virtual Live Streamer")
    parser.add_argument("--source", "-s", type=str, default=None,
                        help="Path to source face image")
    parser.add_argument("--no-swap", action="store_true",
                        help="Run webcam preview without face swapping")
    args = parser.parse_args()

    print("=" * 50)
    print(" Virtual Live Streamer — Demo Preview")
    print("=" * 50)

    if args.no_swap:
        run_preview_no_swap()
        return

    # Check for source face
    source = args.source
    if source is None:
        source = "assets/faces/avatar.png"
        if not Path(source).exists():
            print(f"\nNo source face found at {source}")
            print("Generating a placeholder...")
            Path(source).parent.mkdir(parents=True, exist_ok=True)
            create_sample_face(Path(source))

    if not Path(source).exists():
        print(f"Source face not found: {source}")
        print("Running webcam-only preview...")
        run_preview_no_swap()
        return

    # Check for models
    model_path = Path("models/inswapper_128_fp16.onnx")
    if not model_path.exists():
        print(f"\nFace swap model not found at {model_path}")
        print("Run: python scripts/download_models.py")
        print("\nFalling back to webcam-only preview...")
        run_preview_no_swap()
        return

    run_preview_with_swap(source)


if __name__ == "__main__":
    main()
