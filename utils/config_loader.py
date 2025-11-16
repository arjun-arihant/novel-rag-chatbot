"""
Configuration loader for RAG chatbot
"""
import os
import yaml
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def get_config_value(config: Dict[str, Any], *keys, default=None):
    """
    Safely get nested configuration value.

    Args:
        config: Configuration dictionary
        *keys: Nested keys to access
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.

    Args:
        config: Original configuration
        updates: Updates to apply

    Returns:
        Updated configuration
    """
    def deep_update(base, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                deep_update(base[key], value)
            else:
                base[key] = value
        return base

    return deep_update(config.copy(), updates)
