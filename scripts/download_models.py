#!/usr/bin/env python3
"""Download required ONNX models for the face engine."""

import hashlib
import sys
from pathlib import Path

# Model registry — add new models here
MODELS = [
    {
        "name": "inswapper_128_fp16.onnx",
        "url": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx",
        "dest": "models/inswapper_128_fp16.onnx",
        "size_mb": 264,
    },
    {
        "name": "GFPGANv1.4.onnx",
        "url": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx",
        "dest": "models/GFPGANv1.4.onnx",
        "size_mb": 348,
    },
]


def download_model(url: str, dest: str, name: str, size_mb: int) -> None:
    """Download a model file with progress bar."""
    dest_path = Path(dest)

    if dest_path.exists():
        print(f"  [skip] {name} already exists at {dest}")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {name} (~{size_mb} MB)...")

    try:
        import urllib.request

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
                print(f"\r  [{bar}] {pct}%", end="", flush=True)

        urllib.request.urlretrieve(url, str(dest_path), reporthook=_progress)
        print(f"\n  [done] {name} saved to {dest}")

    except Exception as e:
        print(f"\n  [error] Failed to download {name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        print(f"  You can download manually from: {url}")


def main():
    print("Downloading required models...\n")

    for model in MODELS:
        download_model(
            url=model["url"],
            dest=model["dest"],
            name=model["name"],
            size_mb=model["size_mb"],
        )

    print("\nDone! All models are ready.")


if __name__ == "__main__":
    main()
