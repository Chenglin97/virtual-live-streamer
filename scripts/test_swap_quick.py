#!/usr/bin/env python3
"""Quick face swap test — captures webcam frames, swaps face, saves results."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    source_path = "assets/faces/avatar.png"

    print("Loading face engine...")
    from src.face_engine.engine import FaceEngine

    config = {
        "source_face": source_path,
        "execution_provider": "cpu",
        "face_enhancer": None,
    }

    engine = FaceEngine(config)
    engine.initialize()
    print("Face engine ready!")

    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Warm up camera
    print("Warming up camera...")
    for _ in range(15):
        cap.read()

    # Capture and swap multiple frames
    print("Capturing and swapping faces...")
    results = []

    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            print(f"  Frame {i}: capture failed")
            continue

        t0 = time.time()
        swapped = engine.swap_face(frame)
        elapsed = (time.time() - t0) * 1000

        diff = np.mean(np.abs(frame.astype(float) - swapped.astype(float)))
        face_detected = diff > 1.0

        print(f"  Frame {i}: swap={elapsed:.0f}ms, face_detected={face_detected}")

        if i == 0 or face_detected:
            results.append((frame, swapped, face_detected, elapsed))

        # Small delay between captures for varied expressions
        time.sleep(0.3)

    cap.release()

    if not results:
        print("No frames captured!")
        return

    # Save best result (one with face detected, or first)
    best = next((r for r in results if r[2]), results[0])
    frame, swapped, detected, elapsed = best

    # Save original
    cv2.imwrite(str(OUTPUT_DIR / "swap_original.jpg"), frame)

    # Save swapped
    cv2.imwrite(str(OUTPUT_DIR / "swap_result.jpg"), swapped)

    # Save source face for reference
    source_img = cv2.imread(source_path)
    if source_img is not None:
        # Resize source to match frame height
        h = frame.shape[0]
        scale = h / source_img.shape[0]
        sw = int(source_img.shape[1] * scale)
        source_resized = cv2.resize(source_img, (sw, h))

        # Create comparison: source | original | swapped
        comparison = np.hstack([source_resized, frame, swapped])
        # Labels
        cv2.putText(comparison, "Source Face", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Your Webcam", (sw + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Face Swapped", (sw + frame.shape[1] + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(str(OUTPUT_DIR / "swap_comparison.jpg"), comparison)

    # Also save a "stream preview" with overlays
    preview = swapped.copy()
    h, w = preview.shape[:2]

    # Top bar
    overlay = preview.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 25), -1)
    preview = cv2.addWeighted(overlay, 0.7, preview, 0.3, 0)

    cv2.circle(preview, (30, 28), 10, (0, 0, 255), -1)
    cv2.putText(preview, "LIVE", (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(preview, "Virtual Live Streamer", (130, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(preview, f"Swap: {elapsed:.0f}ms", (w - 200, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

    cv2.imwrite(str(OUTPUT_DIR / "stream_preview.jpg"), preview)

    print(f"\nDone! Results saved to output/")
    print(f"  output/swap_original.jpg    — your webcam")
    print(f"  output/swap_result.jpg      — face swapped")
    print(f"  output/swap_comparison.jpg  — side by side")
    print(f"  output/stream_preview.jpg   — how stream would look")

    engine.release()


if __name__ == "__main__":
    main()
