#!/usr/bin/env python3
"""Full stream preview test — captures webcam, swaps face, saves video + screenshots."""

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

    # ── Load face engine ──
    print("Loading face engine...")
    from src.face_engine.engine import FaceEngine

    config = {
        "source_face": source_path,
        "execution_provider": "cpu",
        "face_enhancer": None,
    }
    engine = FaceEngine(config)
    engine.initialize()
    print("Face engine ready!\n")

    # ── Open webcam ──
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Warm up
    for _ in range(20):
        cap.read()

    ret, first = cap.read()
    if not ret:
        print("Cannot read webcam")
        cap.release()
        return

    h, w = first.shape[:2]
    print(f"Webcam: {w}x{h}\n")

    # ── Record a 3-second clip ──
    fps = 5  # Lower fps since CPU swap is ~1s/frame
    duration = 3
    total = fps * duration

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_path = OUTPUT_DIR / "stream_clip.mp4"
    writer = cv2.VideoWriter(str(vid_path), fourcc, fps, (w, h))

    best_frame = None
    best_original = None
    swap_times = []

    print(f"Recording {duration}s stream preview ({total} frames)...")
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        swapped = engine.swap_face(frame)
        ms = (time.time() - t0) * 1000
        swap_times.append(ms)

        # Check if face was swapped
        diff = np.mean(np.abs(frame.astype(float) - swapped.astype(float)))
        detected = diff > 1.0

        # Add stream overlay
        overlay = swapped.copy()
        # Semi-transparent top bar
        bar = overlay.copy()
        cv2.rectangle(bar, (0, 0), (w, 55), (15, 15, 20), -1)
        overlay = cv2.addWeighted(bar, 0.6, overlay, 0.4, 0)
        overlay[55:, :] = swapped[55:, :]

        # LIVE indicator
        cv2.circle(overlay, (28, 28), 10, (0, 0, 255), -1)
        cv2.putText(overlay, "LIVE", (48, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(overlay, "Aria  |  Just Chatting", (130, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Viewer count
        cv2.putText(overlay, "1,847 viewers", (w - 220, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # Bottom chat area
        chat_h = 160
        chat_overlay = overlay.copy()
        cv2.rectangle(chat_overlay, (0, h - chat_h), (w, h), (10, 10, 15), -1)
        overlay[h - chat_h:, :] = cv2.addWeighted(
            chat_overlay[h - chat_h:, :], 0.5, overlay[h - chat_h:, :], 0.5, 0
        )

        chats = [
            ("ModBot", "[MOD]", "Welcome to the stream!", (128, 128, 255)),
            ("night_owl_22", "", "hiii aria!", (200, 200, 200)),
            ("sparkle_fan", "", "you look amazing today", (200, 200, 200)),
            ("gamer_pro99", "", "first time here, loving this!", (200, 200, 200)),
            ("luna_dreams", "", "can you say hi to me??", (200, 200, 200)),
        ]
        for j, (user, badge, msg, color) in enumerate(chats):
            y = h - chat_h + 25 + j * 28
            text = f"{badge} {user}: {msg}" if badge else f"{user}: {msg}"
            cv2.putText(overlay, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        writer.write(overlay)

        if detected and best_frame is None:
            best_frame = overlay.copy()
            best_original = frame.copy()

        print(f"  Frame {i+1}/{total}: {ms:.0f}ms (face: {'yes' if detected else 'no'})")

    writer.release()
    cap.release()

    avg_ms = sum(swap_times) / len(swap_times) if swap_times else 0
    print(f"\nAvg swap time: {avg_ms:.0f}ms/frame")

    # ── Save screenshots ──
    if best_frame is not None:
        cv2.imwrite(str(OUTPUT_DIR / "stream_screenshot.jpg"), best_frame)
        cv2.imwrite(str(OUTPUT_DIR / "stream_original.jpg"), best_original)

        # Side-by-side comparison
        source_img = cv2.imread(source_path)
        src_h, src_w = source_img.shape[:2]
        scale = h / src_h
        source_resized = cv2.resize(source_img, (int(src_w * scale), h))

        comparison = np.hstack([source_resized, best_original, best_frame])
        cv2.putText(comparison, "Character", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(comparison, "Your Camera", (source_resized.shape[1] + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(comparison, "Stream Output", (source_resized.shape[1] + w + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imwrite(str(OUTPUT_DIR / "stream_comparison.jpg"), comparison)
        print("Saved: output/stream_comparison.jpg")

    print("Saved: output/stream_screenshot.jpg")
    print("Saved: output/stream_clip.mp4")

    engine.release()
    print("\nDone!")


if __name__ == "__main__":
    main()
