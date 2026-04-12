#!/usr/bin/env python3
"""Minimal test: make the character speak using Wav2Lip.

Generates TTS audio, then runs Wav2Lip to create a talking video.
Opens the result automatically.
"""

import asyncio
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT = Path("output")
OUTPUT.mkdir(parents=True, exist_ok=True)
WAV2LIP = Path("wav2lip_model")


async def generate_speech(text: str) -> Path:
    """Generate speech audio from text."""
    from src.tts.engine import TTSEngine

    tts = TTSEngine({"engine": "edge-tts", "voice": "en-US-JennyNeural"})
    await tts.initialize()
    audio = await tts.synthesize(text)

    mp3 = OUTPUT / "speech.mp3"
    wav = OUTPUT / "speech.wav"
    mp3.write_bytes(audio)

    # Convert to 16kHz mono WAV (required by Wav2Lip)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp3), "-ar", "16000", "-ac", "1", str(wav)],
        capture_output=True,
    )
    print(f"  TTS: {wav} ({wav.stat().st_size // 1024} KB)")
    return wav


def run_wav2lip(face_path: str, audio_path: Path) -> Path | None:
    """Run Wav2Lip inference to generate talking head video."""
    # Ensure temp/results dirs exist inside wav2lip dir
    (WAV2LIP / "temp").mkdir(exist_ok=True)
    (WAV2LIP / "results").mkdir(exist_ok=True)

    outfile = WAV2LIP / "results" / "talking.mp4"

    # Use relative paths from wav2lip dir, absolute for external files
    face_abs = str(Path(face_path).resolve())
    audio_abs = str(audio_path.resolve())
    outfile_abs = str(outfile.resolve())

    cmd = [
        sys.executable, "inference.py",
        "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
        "--face", face_abs,
        "--audio", audio_abs,
        "--outfile", outfile_abs,
        "--static", "True",
        "--fps", "25",
        "--pads", "0", "10", "0", "0",
        "--nosmooth",
        "--wav2lip_batch_size", "1",
    ]

    print(f"  Running Wav2Lip...")
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(WAV2LIP.resolve()),  # Run from wav2lip dir so temp/ paths work
        timeout=600,
    )
    elapsed = time.time() - t0

    print(f"  STDOUT: {result.stdout[-300:]}" if result.stdout else "  (no stdout)")
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}" if result.stderr else "  (no stderr)")
        return None

    if outfile.exists():
        final = OUTPUT / "talking_head.mp4"
        shutil.copy(outfile, final)
        print(f"  Video generated in {elapsed:.1f}s: {final} ({final.stat().st_size // 1024} KB)")
        return final

    # Check temp
    temp_avi = WAV2LIP / "temp" / "result.avi"
    if temp_avi.exists():
        final = OUTPUT / "talking_head.avi"
        shutil.copy(temp_avi, final)
        print(f"  Raw video (no audio mux): {final}")
        return final

    print(f"  No output found")
    return None


async def main():
    print("=" * 50)
    print(" Making the character SPEAK")
    print("=" * 50)

    face = "assets/faces/avatar.png"
    text = "Hey everyone! Welcome to my stream. How is everyone doing tonight?"

    print(f"\n[1] Generating speech...")
    wav = await generate_speech(text)

    print(f"\n[2] Generating talking head video...")
    video = run_wav2lip(face, wav.resolve())

    if video:
        print(f"\n[3] Opening result...")
        subprocess.run(["open", str(video)])
    else:
        print("\n  Wav2Lip failed. Check output above for errors.")
        print("  You can also try running manually:")
        print(f"    cd wav2lip_model")
        print(f"    python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth \\")
        print(f"      --face ../assets/faces/avatar.png --audio ../output/speech.wav \\")
        print(f"      --static True --outfile results/talking.mp4")


if __name__ == "__main__":
    asyncio.run(main())
