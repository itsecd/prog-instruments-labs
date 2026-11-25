"""Load and validate layered configuration (base -> environment)."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class ConfigurationError(ValueError):
    """Invalid or unsafe application configuration."""


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; the environment layer has precedence."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as source:
            loaded = yaml.safe_load(source)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot read configuration {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"Configuration must be a mapping: {path}")
    return loaded


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate types, paths and rule names before touching the filesystem."""
    expected = {"paths", "behavior", "rules", "logging"}
    if set(config) != expected:
        raise ConfigurationError(f"Top-level keys must be {sorted(expected)}")

    paths, behavior, rules, logging = (
        config["paths"], config["behavior"], config["rules"], config["logging"]
    )
    if not isinstance(paths, dict) or set(paths) != {"source", "destination"}:
        raise ConfigurationError("paths must contain source and destination")
    for key, value in paths.items():
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"paths.{key} must be a non-empty string")

    if not isinstance(behavior, dict) or set(behavior) != {"dry_run", "on_conflict"}:
        raise ConfigurationError("behavior must contain dry_run and on_conflict")
    if type(behavior["dry_run"]) is not bool:
        raise ConfigurationError("behavior.dry_run must be boolean")
    if behavior["on_conflict"] not in ("skip", "rename"):
        raise ConfigurationError("behavior.on_conflict must be skip or rename")

    if not isinstance(rules, dict) or not rules:
        raise ConfigurationError("rules must be a non-empty mapping")
    for category, extensions in rules.items():
        if not isinstance(category, str) or not category.isidentifier():
            raise ConfigurationError("rule categories must be valid directory names")
        if not isinstance(extensions, list) or not extensions:
            raise ConfigurationError(f"rules.{category} must be a non-empty list")
        for ext in extensions:
            if not isinstance(ext, str) or not ext.startswith(".") or len(ext) < 2 or not ext[1:].isalnum():
                raise ConfigurationError(f"Invalid extension in rules.{category}: {ext!r}")

    if not isinstance(logging, dict) or set(logging) != {"level"}:
        raise ConfigurationError("logging must contain level")
    if logging["level"] not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise ConfigurationError("Invalid logging.level")
    return config


def load_config(environment: str = "dev", config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load base.yaml then dev.yaml or prod.yaml, validating the result."""
    if environment not in {"dev", "prod"}:
        raise ConfigurationError("Environment must be 'dev' or 'prod'")
    base = _read_yaml(Path(config_dir) / "base.yaml")
    env = _read_yaml(Path(config_dir) / f"{environment}.yaml")
    return validate_config(merge_config(base, env))
