#!/usr/bin/env python3
"""Test the autonomous pipeline: face image + TTS audio → talking head video.

No webcam needed. This is the 24/7 streamer pipeline.
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def test_tts():
    """Generate test speech audio."""
    print("\n[1/3] Generating TTS audio...")
    from src.tts.engine import TTSEngine

    tts = TTSEngine({
        "engine": "edge-tts",
        "voice": "en-US-JennyNeural",
        "rate": "+0%",
        "volume": "+0%",
    })
    await tts.initialize()

    text = "Hey everyone! Welcome to my stream. I'm so happy to see you all here today. How is everyone doing?"
    audio = await tts.synthesize(text)

    # Save as mp3 and convert to wav for wav2lip
    mp3_path = OUTPUT_DIR / "test_tts.mp3"
    wav_path = OUTPUT_DIR / "test_tts.wav"
    mp3_path.write_bytes(audio)

    # Convert mp3 to wav using ffmpeg
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", str(mp3_path),
        "-ar", "16000", "-ac", "1",
        str(wav_path)
    ], capture_output=True)

    print(f"  Audio saved: {wav_path} ({wav_path.stat().st_size // 1024} KB)")
    return wav_path


def test_wav2lip(audio_path: Path):
    """Run Wav2Lip to generate talking head video."""
    print("\n[2/3] Generating talking head video with Wav2Lip...")

    face_path = Path("assets/faces/avatar.png")
    output_path = OUTPUT_DIR / "talking_head.avi"

    wav2lip_dir = Path("wav2lip_model")

    # Run wav2lip inference
    import subprocess
    cmd = [
        sys.executable,
        str(wav2lip_dir / "inference.py"),
        "--checkpoint_path", str(wav2lip_dir / "checkpoints" / "wav2lip_gan.pth"),
        "--face", str(face_path),
        "--audio", str(audio_path),
        "--outfile", str(output_path),
        "--resize_factor", "1",
        "--pads", "0", "10", "0", "0",
        "--nosmooth",
    ]

    print(f"  Running: {' '.join(cmd[-8:])}")
    t0 = time.time()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-1000:]}")
        print(f"  STDOUT: {result.stdout[-500:]}")
        return None

    if output_path.exists():
        size = output_path.stat().st_size
        print(f"  Video generated in {elapsed:.1f}s: {output_path} ({size // 1024} KB)")
        return output_path
    else:
        print(f"  No output file generated")
        # Check if it ended up in wav2lip results dir
        alt = wav2lip_dir / "results" / "result_voice.mp4"
        if alt.exists():
            import shutil
            final = OUTPUT_DIR / "talking_head.mp4"
            shutil.copy(alt, final)
            print(f"  Found output at: {alt} -> {final}")
            return final
        return None


def test_idle_frames():
    """Generate idle animation frames."""
    print("\n[3/3] Generating idle animation...")
    import cv2
    import numpy as np

    face = cv2.imread("assets/faces/avatar.png")
    if face is None:
        print("  FAIL: Could not load face image")
        return

    h, w = face.shape[:2]
    target_w, target_h = 1280, 720

    # Scale face to fit
    scale = min(target_w / w, target_h / h) * 0.55
    new_w, new_h = int(w * scale), int(h * scale)
    face_resized = cv2.resize(face, (new_w, new_h))

    fps = 25
    duration = 4
    total = fps * duration
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_path = OUTPUT_DIR / "idle_animation.mp4"
    writer = cv2.VideoWriter(str(vid_path), fourcc, fps, (target_w, target_h))

    for i in range(total):
        frame = np.full((target_h, target_w, 3), (25, 25, 30), dtype=np.uint8)

        # Subtle idle movement
        t = i / fps
        dx = int(4 * np.sin(2 * np.pi * t / 5))
        dy = int(3 * np.sin(2 * np.pi * t / 3.5))

        x = (target_w - new_w) // 2 + dx
        y = (target_h - new_h) // 2 + dy - 30

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(target_w, x + new_w), min(target_h, y + new_h)
        fx1, fy1 = x1 - x, y1 - y
        fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)
        frame[y1:y2, x1:x2] = face_resized[fy1:fy2, fx1:fx2]

        # Stream overlay
        bar = frame.copy()
        cv2.rectangle(bar, (0, 0), (target_w, 50), (15, 15, 20), -1)
        frame = cv2.addWeighted(bar, 0.6, frame, 0.4, 0)
        frame[50:, :] = np.full((target_h - 50, target_w, 3), (25, 25, 30), dtype=np.uint8)
        # Re-paste face below bar
        frame2 = np.full((target_h, target_w, 3), (25, 25, 30), dtype=np.uint8)
        frame2[y1:y2, x1:x2] = face_resized[fy1:fy2, fx1:fx2]
        cv2.rectangle(frame2, (0, 0), (target_w, 50), (15, 15, 20), -1)

        cv2.circle(frame2, (28, 28), 8, (0, 0, 255), -1)
        cv2.putText(frame2, "LIVE", (45, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame2, "Aria  |  Just Chatting", (110, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(frame2)

    writer.release()
    print(f"  Idle animation saved: {vid_path} ({total} frames, {duration}s)")
    return vid_path


async def main():
    print("=" * 55)
    print(" Autonomous Streamer — Pipeline Test")
    print(" (No webcam needed)")
    print("=" * 55)

    # Step 1: TTS
    audio_path = await test_tts()

    # Step 2: Wav2Lip talking head
    video_path = test_wav2lip(audio_path)

    # Step 3: Idle animation
    idle_path = test_idle_frames()

    print("\n" + "=" * 55)
    print(" Results:")
    print("=" * 55)
    for f in sorted(OUTPUT_DIR.glob("*")):
        if f.name.startswith(("test_tts", "talking_head", "idle_")):
            size = f.stat().st_size
            label = "MB" if size > 1_000_000 else "KB"
            val = size / 1_000_000 if size > 1_000_000 else size / 1_000
            print(f"  {f.name:30s} {val:.1f} {label}")

    if video_path:
        print(f"\nOpening results...")
        import subprocess
        subprocess.run(["open", str(video_path)])
    if idle_path:
        import subprocess
        subprocess.run(["open", str(idle_path)])


if __name__ == "__main__":
    asyncio.run(main())
