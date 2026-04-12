#!/usr/bin/env python3
"""Test the full pipeline — generate test frames, face swap, TTS, save output.

Works without a webcam by downloading a sample face video.

Generates:
  - output/test_original.jpg   — input frame
  - output/test_swapped.jpg    — face-swapped frame (if face detected)
  - output/test_sidebyside.jpg — comparison
  - output/test_speech.mp3     — TTS sample
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

OUTPUT_DIR = Path("output")


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_source_face(output_path: Path) -> None:
    """Generate a synthetic source face image."""
    img = np.full((512, 512, 3), (210, 190, 175), dtype=np.uint8)
    cv2.ellipse(img, (256, 280), (180, 220), 0, 0, 360, (220, 200, 185), -1)
    cv2.ellipse(img, (190, 230), (28, 18), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (322, 230), (28, 18), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (190, 230), 10, (50, 30, 20), -1)
    cv2.circle(img, (322, 230), 10, (50, 30, 20), -1)
    pts = np.array([[256, 260], [240, 310], [272, 310]], np.int32)
    cv2.polylines(img, [pts], False, (170, 150, 140), 2)
    cv2.ellipse(img, (256, 350), (50, 20), 0, 10, 170, (140, 100, 100), 3)
    cv2.line(img, (155, 200), (225, 195), (120, 90, 70), 3)
    cv2.line(img, (287, 195), (357, 200), (120, 90, 70), 3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"  Created source face: {output_path}")


def get_test_frame():
    """Try webcam first, fall back to generating a frame with a photo-like face."""
    # Try webcam
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        for _ in range(10):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            print("  Using webcam frame")
            return frame

    # Fall back: create a realistic test frame with a face-like pattern
    print("  Webcam unavailable — generating synthetic test frame")
    frame = np.full((720, 1280, 3), (40, 35, 30), dtype=np.uint8)

    # Background gradient
    for y in range(720):
        shade = int(30 + 40 * (y / 720))
        frame[y, :] = (shade, shade - 5, shade - 10)

    # Draw a "person" shape in center
    cx, cy = 640, 360
    # Head (skin-colored ellipse)
    cv2.ellipse(frame, (cx, cy - 40), (90, 120), 0, 0, 360, (180, 170, 160), -1)
    # Body
    cv2.rectangle(frame, (cx - 120, cy + 80), (cx + 120, 720), (60, 60, 80), -1)
    # Eyes
    cv2.ellipse(frame, (cx - 30, cy - 60), (18, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(frame, (cx + 30, cy - 60), (18, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(frame, (cx - 30, cy - 60), 7, (40, 30, 20), -1)
    cv2.circle(frame, (cx + 30, cy - 60), 7, (40, 30, 20), -1)
    # Nose
    cv2.line(frame, (cx, cy - 40), (cx - 8, cy - 5), (160, 150, 140), 2)
    # Mouth
    cv2.ellipse(frame, (cx, cy + 15), (30, 12), 0, 10, 170, (150, 100, 100), 2)

    # "Stream" overlay elements
    cv2.rectangle(frame, (0, 0), (1280, 50), (20, 20, 25), -1)
    cv2.putText(frame, "LIVE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.circle(frame, (100, 25), 8, (0, 0, 255), -1)
    cv2.putText(frame, "Virtual Live Streamer", (130, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "1,234 viewers", (1050, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Chat overlay
    cv2.rectangle(frame, (900, 500), (1270, 710), (30, 30, 35), -1)
    cv2.putText(frame, "Chat", (920, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    chat_msgs = [
        ("user42", "hello!!", (100, 200, 255)),
        ("streamer_fan", "great stream!", (255, 200, 100)),
        ("newbie_123", "first time here", (100, 255, 100)),
        ("pro_gamer", "lets gooo", (255, 100, 200)),
    ]
    for i, (user, msg, color) in enumerate(chat_msgs):
        y = 560 + i * 35
        cv2.putText(frame, f"{user}:", (920, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(frame, msg, (920 + len(user) * 10 + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return frame


def test_webcam_capture():
    """Test frame capture."""
    print("\n[1/4] Capturing test frame...")
    frame = get_test_frame()
    h, w = frame.shape[:2]
    print(f"  OK: Frame {w}x{h}")

    output = OUTPUT_DIR / "test_original.jpg"
    cv2.imwrite(str(output), frame)
    print(f"  Saved: {output}")
    return frame


def test_face_swap(frame):
    """Test face swapping on a captured frame."""
    print("\n[2/4] Testing face swap engine...")

    source_path = Path("assets/faces/avatar.png")
    if not source_path.exists():
        print("  No source face found, generating placeholder...")
        create_source_face(source_path)

    model_path = Path("models/inswapper_128_fp16.onnx")
    if not model_path.exists():
        print("  SKIP: Model not found — run: python scripts/download_models.py")
        return None

    from src.face_engine.engine import FaceEngine

    config = {
        "source_face": str(source_path),
        "execution_provider": "cpu",  # Most compatible
        "face_enhancer": None,
    }

    engine = FaceEngine(config)
    try:
        print("  Loading models (this takes ~10s on first run)...")
        engine.initialize()
    except Exception as e:
        print(f"  FAIL: {e}")
        return None

    t0 = time.time()
    swapped = engine.swap_face(frame)
    elapsed = (time.time() - t0) * 1000

    # Check if any faces were detected and swapped
    diff = np.mean(np.abs(frame.astype(float) - swapped.astype(float)))
    faces_found = diff > 1.0

    output = OUTPUT_DIR / "test_swapped.jpg"
    cv2.imwrite(str(output), swapped)
    print(f"  Face swap completed in {elapsed:.0f}ms (faces detected: {faces_found})")
    print(f"  Saved: {output}")

    if not faces_found:
        print("  NOTE: No real face detected in synthetic frame — this is expected.")
        print("        Use a real photo/webcam to see actual face swapping.")

    # Side by side
    h, w = frame.shape[:2]
    sidebyside = np.hstack([frame, swapped])
    cv2.putText(sidebyside, "Original", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(sidebyside, "Face Swapped", (w + 10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    output_sbs = OUTPUT_DIR / "test_sidebyside.jpg"
    cv2.imwrite(str(output_sbs), sidebyside)
    print(f"  Saved: {output_sbs}")

    engine.release()
    return swapped


def test_tts():
    """Test TTS synthesis."""
    print("\n[3/4] Testing TTS engine...")
    import asyncio
    from src.tts.engine import TTSEngine

    config = {
        "engine": "edge-tts",
        "voice": "en-US-JennyNeural",
        "rate": "+0%",
        "volume": "+0%",
    }

    async def _test():
        tts = TTSEngine(config)
        await tts.initialize()
        text = "Hello everyone! Welcome to my stream. I'm so happy to see you all here today!"
        t0 = time.time()
        audio = await tts.synthesize(text)
        elapsed = (time.time() - t0) * 1000

        output = OUTPUT_DIR / "test_speech.mp3"
        output.write_bytes(audio)
        duration_est = len(audio) / 16000  # rough estimate
        print(f"  OK: Generated {len(audio)/1024:.1f} KB of audio in {elapsed:.0f}ms")
        print(f"  Saved: {output}")
        return audio

    return asyncio.run(_test())


def test_agent_response():
    """Test the AI agent (dry run — no real API call unless key is set)."""
    print("\n[4/4] Testing AI agent...")
    import asyncio
    import os
    from src.agent.agent import StreamerAgent

    api_key = os.environ.get("AGENT_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

    if not api_key:
        print("  SKIP: No API key set (set OPENAI_API_KEY or AGENT_API_KEY)")
        print("  The agent module is importable and ready — just needs an API key.")
        return

    config = {
        "provider": "openai",
        "model": "gpt-4o",
        "api_key": api_key,
        "personality": "You are a friendly virtual streamer named Aria. Keep responses under 2 sentences.",
        "max_response_length": 100,
        "response_cooldown_seconds": 0,
        "idle_talk_interval_seconds": 60,
    }

    async def _test():
        agent = StreamerAgent(config)
        await agent.initialize()

        response = await agent.respond_to_chat("test_user", "Hey! How's it going?")
        print(f"  Agent response: {response}")

        idle = await agent.idle_talk()
        print(f"  Idle talk: {idle}")

        await agent.shutdown()

    asyncio.run(_test())


def main():
    ensure_output_dir()

    print("=" * 55)
    print(" Virtual Live Streamer — Pipeline Test")
    print("=" * 55)

    # Test 1: Frame capture
    frame = test_webcam_capture()

    # Test 2: Face swap
    test_face_swap(frame)

    # Test 3: TTS
    test_tts()

    # Test 4: Agent
    test_agent_response()

    print("\n" + "=" * 55)
    print(" Test complete! Check the output/ directory:")
    print("=" * 55)

    for f in sorted(OUTPUT_DIR.glob("test_*")):
        size = f.stat().st_size
        if size > 1_000_000:
            print(f"  {f}  ({size / 1_000_000:.1f} MB)")
        elif size > 1_000:
            print(f"  {f}  ({size / 1_000:.1f} KB)")
        else:
            print(f"  {f}  ({size} bytes)")


if __name__ == "__main__":
    main()
