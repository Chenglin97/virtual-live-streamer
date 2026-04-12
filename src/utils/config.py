"""Configuration loader for the virtual live streamer."""

import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("config/config.yaml")


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file with environment variable overrides."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Copy config/config.example.yaml to {path} and fill in your values."
        )

    with open(path) as f:
        config = yaml.safe_load(f)

    # Environment variable overrides
    _apply_env_overrides(config)

    return config


def _apply_env_overrides(config: dict[str, Any]) -> None:
    """Override config values with environment variables where set."""
    env_mappings = {
        "AGENT_API_KEY": ("agent", "api_key"),
        "STREAM_KEY": ("stream", "stream_key"),
        "CHAT_AUTH_TOKEN": ("chat", "auth_token"),
        "AGENT_MODEL": ("agent", "model"),
        "AGENT_PROVIDER": ("agent", "provider"),
        "AGENT_BASE_URL": ("agent", "base_url"),
        "FACE_SOURCE": ("face_engine", "source_face"),
        "RTMP_URL": ("stream", "rtmp_url"),
        "TTS_VOICE": ("tts", "voice"),
    }

    for env_var, key_path in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            section = config
            for key in key_path[:-1]:
                section = section.setdefault(key, {})
            section[key_path[-1]] = value
