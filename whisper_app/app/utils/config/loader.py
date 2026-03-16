"""
Configuration loader for WhisperSign application.

Loads and validates app_config.yaml into a structured dictionary.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

# Default config values (used when keys are missing from YAML)
_DEFAULTS = {
    "hardware": {
        "device": "leap_motion",
        "target_fps": 60,
        "buffer_duration": 3.0,
    },
    "model": {
        "checkpoint_path": "../Whisper_modification/checkpoints/final_model.pt",
        "device": "cuda",
        "window_duration": 3.0,
        "inference_interval": 0.5,
    },
    "preprocessing": {
        "smoothing_window": 5,
        "spatial_normalization": True,
        "scale_normalization": True,
    },
    "ui": {
        "window_width": 1200,
        "window_height": 800,
        "visualization_fps": 30,
        "show_skeleton": True,
        "show_confidence": True,
    },
    "logging": {
        "level": "INFO",
        "file": "logs/app.log",
    },
}


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file, applying defaults for missing keys.

    Args:
        config_path: Path to the YAML config file. If None, tries
                     the default location (whisper_app/config/app_config.yaml).

    Returns:
        Merged configuration dictionary.
    """
    if config_path is None:
        # Try default path relative to this file
        config_path = str(
            Path(__file__).resolve().parent.parent.parent / "config" / "app_config.yaml"
        )

    config = {}
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.warning(f"Config file not found at {config_path}; using defaults.")

    # Deep merge with defaults
    merged = _deep_merge(_DEFAULTS, config)
    return merged


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """Recursively merge overrides into defaults."""
    result = defaults.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
