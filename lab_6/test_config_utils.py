import json

import pytest

from config_utils import load_config, save_config


def test_loads_utf8_settings(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"input_file": "пример.txt", "aes_key_size": 192}', encoding="utf-8")

    assert load_config(path) == {"input_file": "пример.txt", "aes_key_size": 192}


def test_saves_utf8_json(tmp_path):
    path = tmp_path / "config.json"
    settings = {"input_file": "данные.bin", "aes_key_size": 256}

    save_config(path, settings)

    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == settings
    assert "данные.bin" in text


def test_missing_config_reports_path(tmp_path):
    path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError) as error:
        load_config(path)

    assert str(path) in str(error.value)


@pytest.mark.parametrize("contents", ["", "{", '{"aes_key_size":}'])
def test_rejects_invalid_json(tmp_path, contents):
    path = tmp_path / "invalid.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="Ошибка чтения JSON"):
        load_config(path)


def test_reports_config_write_failure(tmp_path):
    path = tmp_path / "missing_directory" / "config.json"

    with pytest.raises(OSError, match="Ошибка сохранения конфигурации"):
        save_config(path, {"aes_key_size": 128})

    assert not path.exists()
