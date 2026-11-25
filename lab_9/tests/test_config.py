"""Demonstrate layered YAML configuration and real file organizer behavior."""
from copy import deepcopy
from pathlib import Path

import pytest

from lab_9.organizer.app import organize, target_category
from lab_9.organizer.config import ConfigurationError, load_config, merge_config, validate_config


def test_deep_merge_preserves_base_settings():
    merged = merge_config({"nested": {"one": 1, "two": 2}}, {"nested": {"two": 9}})
    assert merged == {"nested": {"one": 1, "two": 9}}


def test_dev_and_prod_layers_have_different_behaviors():
    dev, prod = load_config("dev"), load_config("prod")
    assert dev["behavior"]["dry_run"] is True
    assert prod["behavior"]["dry_run"] is False
    assert dev["rules"] == prod["rules"]
    assert dev["logging"]["level"] == "DEBUG"


def test_reject_unknown_environment():
    with pytest.raises(ConfigurationError, match="Environment"):
        load_config("stage")


def test_reject_unsafe_rule_name():
    config = load_config()
    config["rules"]["../escape"] = [".txt"]
    with pytest.raises(ConfigurationError, match="categories"):
        validate_config(config)


def test_reject_invalid_yaml(tmp_path):
    (tmp_path / "base.yaml").write_text("invalid: [", encoding="utf-8")
    (tmp_path / "dev.yaml").write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Cannot read"):
        load_config("dev", tmp_path)


def test_case_insensitive_extensions():
    assert target_category("PHOTO.JPG", {"images": [".jpg"]}) == "images"
    assert target_category("archive.zip", {"images": [".jpg"]}) == "other"


def test_dev_dry_run_does_not_move_files(tmp_path):
    source = tmp_path / "inbox"
    source.mkdir()
    (source / "a.pdf").write_text("hello", encoding="utf-8")
    config = load_config("dev")
    config["paths"] = {"source": "inbox", "destination": "out"}
    assert organize(config, root=tmp_path) == {"moved": 0, "skipped": 0, "planned": 1}
    assert (source / "a.pdf").exists()
    assert not (tmp_path / "out").exists()


def test_prod_moves_files_by_type(tmp_path):
    source = tmp_path / "inbox"
    source.mkdir()
    (source / "a.pdf").write_text("hello", encoding="utf-8")
    (source / "b.png").write_text("image", encoding="utf-8")
    (source / "c.bin").write_text("other", encoding="utf-8")
    config = load_config("prod")
    config["paths"] = {"source": "inbox", "destination": "out"}
    assert organize(config, root=tmp_path) == {"moved": 3, "skipped": 0, "planned": 0}
    for category, filename in (("documents", "a.pdf"), ("images", "b.png"), ("other", "c.bin")):
        assert (tmp_path / "out" / category / filename).exists()


def test_prod_renames_conflicting_file(tmp_path):
    source = tmp_path / "inbox"
    source.mkdir()
    (source / "a.txt").write_text("new", encoding="utf-8")
    existing = tmp_path / "out" / "documents"
    existing.mkdir(parents=True)
    (existing / "a.txt").write_text("old", encoding="utf-8")
    config = load_config("prod")
    config["paths"] = {"source": "inbox", "destination": "out"}
    assert organize(config, root=tmp_path)["moved"] == 1
    assert (existing / "a.txt").read_text(encoding="utf-8") == "old"
    assert (existing / "a_1.txt").read_text(encoding="utf-8") == "new"


def test_skip_existing_file(tmp_path):
    source = tmp_path / "inbox"
    source.mkdir()
    (source / "a.txt").write_text("new", encoding="utf-8")
    existing = tmp_path / "out" / "documents"
    existing.mkdir(parents=True)
    (existing / "a.txt").write_text("old", encoding="utf-8")
    config = load_config("prod")
    config["behavior"]["on_conflict"] = "skip"
    config["paths"] = {"source": "inbox", "destination": "out"}
    assert organize(config, root=tmp_path)["skipped"] == 1
    assert (source / "a.txt").exists()


def test_reject_nested_destination(tmp_path):
    source = tmp_path / "inbox"
    source.mkdir()
    config = load_config("dev")
    config["paths"] = {"source": "inbox", "destination": "inbox/out"}
    with pytest.raises(ConfigurationError, match="inside source"):
        organize(config, root=tmp_path)
