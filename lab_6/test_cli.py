import sys

import pytest

import cli


@pytest.fixture
def settings():
    return {
        "input_file": "input.txt",
        "encrypted_file": "encrypted.bin",
        "decrypted_file": "output.txt",
        "encrypted_key_file": "aes_key.enc",
        "public_key_file": "public.pem",
        "private_key_file": "private.pem",
        "aes_key_size": "192",
    }


@pytest.mark.parametrize(
    ("option", "operation", "arguments", "message"),
    [
        (
            "--generation",
            "generate_keys",
            ("aes_key.enc", "public.pem", "private.pem", 192),
            "Готово: ключи созданы",
        ),
        (
            "--encryption",
            "encrypt_file",
            ("input.txt", "public.pem", "aes_key.enc", "encrypted.bin", 192),
            "Готово: файл зашифрован",
        ),
        (
            "--decryption",
            "decrypt_file",
            ("encrypted.bin", "private.pem", "aes_key.enc", "output.txt"),
            "Готово: файл расшифрован",
        ),
    ],
)
def test_cli_dispatches_selected_command(
    monkeypatch, capsys, settings, option, operation, arguments, message
):
    calls = []

    def load_config(path):
        calls.append(("load_config", path))
        return settings

    def record_call(name):
        def command(*args):
            calls.append((name, args))

        return command

    monkeypatch.setattr(sys, "argv", ["cli.py", "--config", "custom.json", option])
    monkeypatch.setattr(cli, "load_config", load_config)
    for name in ("generate_keys", "encrypt_file", "decrypt_file"):
        monkeypatch.setattr(cli, name, record_call(name))

    cli.main()

    assert calls == [("load_config", "custom.json"), (operation, arguments)]
    assert message in capsys.readouterr().out


@pytest.mark.parametrize(
    ("options", "reported_option"),
    [
        pytest.param([], "--generation", id="missing"),
        pytest.param(["-enc", "-dec"], "--decryption", id="conflicting"),
    ],
)
def test_cli_rejects_invalid_command_selection(monkeypatch, capsys, options, reported_option):
    def unexpected_config_load(path):
        pytest.fail(f"Invalid arguments triggered a config read: {path}")

    monkeypatch.setattr(sys, "argv", ["cli.py", *options])
    monkeypatch.setattr(cli, "load_config", unexpected_config_load)

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 2
    assert reported_option in capsys.readouterr().err


def test_cli_reports_missing_default_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["cli.py", "-gen"])

    with pytest.raises(FileNotFoundError, match=r"config\.json"):
        cli.main()
