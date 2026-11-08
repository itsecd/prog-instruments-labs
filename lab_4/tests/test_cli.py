"""Smoke test for the CLI entry point."""

from pathlib import Path

from main import main

DATA = Path(__file__).resolve().parent.parent / "26.csv"


def test_cli_writes_result_json(tmp_path):
    out = tmp_path / "result.json"
    code = main([str(DATA), "--output", str(out), "--variant", "26"])
    assert code == 0
    assert out.exists()
    assert '"variant": "26"' in out.read_text(encoding="utf-8")
