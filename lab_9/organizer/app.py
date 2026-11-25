"""Organize files into folders, driven entirely by YAML configuration."""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from .config import ConfigurationError, load_config

LOGGER = logging.getLogger(__name__)


def target_category(filename: str, rules: dict[str, list[str]]) -> str:
    """Find a category based on a case-insensitive filename extension."""
    suffix = Path(filename).suffix.lower()
    for category, extensions in rules.items():
        if suffix in (ext.lower() for ext in extensions):
            return category
    return "other"


def _target_with_suffix(target: Path) -> Path:
    """Find a free path, without overwriting existing files."""
    index = 1
    while target.exists():
        target = target.with_name(f"{target.stem}_{index}{target.suffix}")
        index += 1
    return target


def organize(config: dict, *, root: Path | None = None) -> dict[str, int]:
    """Organize direct child files; return a small execution summary."""
    root_dir = Path.cwd() if root is None else Path(root)
    source = (root_dir / config["paths"]["source"]).resolve()
    destination = (root_dir / config["paths"]["destination"]).resolve()
    if source == destination:
        raise ConfigurationError("source and destination must differ")
    if not source.is_dir():
        raise ConfigurationError(f"Source directory does not exist: {source}")
    if destination == source or source in destination.parents:
        raise ConfigurationError("Destination cannot be inside source")

    summary = {"moved": 0, "skipped": 0, "planned": 0}
    dry_run = config["behavior"]["dry_run"]
    for entry in sorted(source.iterdir()):
        if not entry.is_file() or entry.is_symlink():
            continue
        category = target_category(entry.name, config["rules"])
        target = destination / category / entry.name
        if target.exists():
            if config["behavior"]["on_conflict"] == "skip":
                LOGGER.info("Skipping existing file: %s", target)
                summary["skipped"] += 1
                continue
            target = _target_with_suffix(target)
        if dry_run:
            LOGGER.info("DRY RUN: %s -> %s", entry, target)
            summary["planned"] += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(entry), str(target))
        LOGGER.info("Moved: %s -> %s", entry, target)
        summary["moved"] += 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Organize files with layered YAML config")
    parser.add_argument("--env", choices=("dev", "prod"), default="dev")
    args = parser.parse_args()
    try:
        config = load_config(args.env)
        logging.basicConfig(level=config["logging"]["level"], format="%(levelname)s %(message)s")
        result = organize(config)
    except ConfigurationError as exc:
        parser.exit(2, f"Configuration error: {exc}\n")
    print(f"Result: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
