import pytest
from PyQt5.QtWidgets import QApplication, QLineEdit

import app
from config_utils import save_config


@pytest.fixture(scope="module")
def qt_app():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("QT_QPA_PLATFORM", "offscreen")
        application = QApplication.instance() or QApplication([])
        yield application


@pytest.fixture
def window(qt_app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    widget = app.CryptoWindow()
    yield widget
    widget.close()


def test_gui_loads_saved_settings_and_keeps_defaults(qt_app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_config(tmp_path / "config.json", {"input_file": "пример.txt", "aes_key_size": 256})
    widget = app.CryptoWindow()

    try:
        settings = widget.collect_settings()
        assert settings["input_file"] == "пример.txt"
        assert settings["aes_key_size"] == 256
        assert settings["public_key_file"] == str(tmp_path / "public.pem")
    finally:
        widget.close()


@pytest.mark.parametrize(
    ("save_mode", "dialog", "selected", "expected"),
    [
        (False, "getOpenFileName", "input.txt", "input.txt"),
        (True, "getSaveFileName", "output.txt", "output.txt"),
        (True, "getSaveFileName", "", "original.txt"),
    ],
)
def test_gui_updates_path_only_when_selected(window, monkeypatch, save_mode, dialog, selected, expected):
    monkeypatch.setattr(app.QFileDialog, dialog, lambda *_: (selected, ""))
    edit = QLineEdit("original.txt", window)

    window.choose_path(edit, save_mode)

    assert edit.text() == expected


@pytest.mark.parametrize(
    ("action", "operation", "keys", "message"),
    [
        ("save_config_action", None, (), "Настройки сохранены"),
        (
            "generate_action",
            "generate_keys",
            ("encrypted_key_file", "public_key_file", "private_key_file", "aes_key_size"),
            "Ключи созданы",
        ),
        (
            "encrypt_action",
            "encrypt_file",
            ("input_file", "public_key_file", "encrypted_key_file", "encrypted_file", "aes_key_size"),
            "Файл зашифрован",
        ),
        (
            "decrypt_action",
            "decrypt_file",
            ("encrypted_file", "private_key_file", "encrypted_key_file", "decrypted_file"),
            "Файл расшифрован",
        ),
    ],
)
def test_gui_runs_action_and_saves_current_settings(window, monkeypatch, action, operation, keys, message):
    calls = []
    window.aes_box.setCurrentText("192")
    settings = window.collect_settings()
    if operation is not None:
        monkeypatch.setattr(app, operation, lambda *args: calls.append((operation, args)))
    monkeypatch.setattr(app, "save_config", lambda *args: calls.append(("save", args)))
    monkeypatch.setattr(
        app.QMessageBox, "information", lambda _, title, text: calls.append(("info", title, text))
    )

    getattr(window, action)()

    expected = [(operation, tuple(settings[key] for key in keys))] if operation else []
    expected += [("save", ("config.json", settings)), ("info", "Готово", message)]
    assert calls == expected
    log_message = message if operation else "config.json сохранён"
    assert log_message in window.log_box.toPlainText()


@pytest.mark.parametrize(
    ("action", "operation"),
    [
        ("save_config_action", "save_config"),
        ("generate_action", "generate_keys"),
        ("encrypt_action", "encrypt_file"),
        ("decrypt_action", "decrypt_file"),
    ],
)
def test_gui_reports_operation_error(window, monkeypatch, action, operation):
    messages = []

    def fail(*_):
        raise OSError("test failure")

    def unexpected_success(*_):
        pytest.fail("Failed operation reported success")

    monkeypatch.setattr(app, "save_config", unexpected_success)
    monkeypatch.setattr(app, operation, fail)
    monkeypatch.setattr(app.QMessageBox, "information", unexpected_success)
    monkeypatch.setattr(
        app.QMessageBox, "critical", lambda _, title, text: messages.append((title, text))
    )

    getattr(window, action)()

    assert messages == [("Ошибка", "test failure")]
    assert "Ошибка: test failure" in window.log_box.toPlainText()
