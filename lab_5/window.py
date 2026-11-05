import sys
import json
import logging

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QLabel,
    QFileDialog,
    QRadioButton,
    QGroupBox,
    QMessageBox,
)

from Camellia import encrypt, decrypt
from key_gen import key_gen
from logging_config import setup_logging

logger = logging.getLogger(__name__)


class CryptoApp(QMainWindow):
    def __init__(self):
        """Constructor object of class"""
        super().__init__()
        logger.info("CryptoApp initialization started")
        self.settings = self.load_settings()
        self.init_ui()
        self.apply_settings()
        logger.info("CryptoApp initialization completed")

    def load_settings(self):
        """Loads the settings from settings.json"""
        logger.debug("Loading settings from settings.json")
        settings = {
            "initial_file": "",
            "encrypted_file": "",
            "decrypted_file": "",
            "symmetric_key": "",
            "public_key": "",
            "secret_key": "",
        }
        try:
            with open("settings.json", "r", encoding="utf-8") as f:
                loaded = json.load(f)
                logger.debug("Settings loaded: keys=%s", list(loaded.keys()))
                for key, val in loaded.items():
                    try:
                        with open(val, "rb") as _:
                            pass
                    except Exception:
                        logger.warning(
                            "Settings path for '%s' is invalid or file not found: %s",
                            key,
                            val,
                        )
                        loaded[key] = ""
                settings.update(loaded)
        except FileNotFoundError:
            logger.warning("settings.json not found; creating default settings")
            with open("settings.json", "w", encoding="utf-8") as f:
                json.dump(settings, f)
        except json.JSONDecodeError:
            logger.exception("settings.json is not valid JSON; using defaults")
        return settings

    def init_ui(self):
        """Creates and fills in the user interface"""
        logger.debug("Initializing UI")
        self.setWindowTitle("Шифрователь 2000 (RSA + Camellia)")
        self.setMinimumSize(600, 350)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        mode_group = QGroupBox("Режим работы")
        mode_layout = QHBoxLayout()
        self.radio_encrypt = QRadioButton("Шифрование")
        self.radio_decrypt = QRadioButton("Расшифрование")
        self.radio_encrypt.setChecked(True)
        mode_layout.addWidget(self.radio_encrypt)
        mode_layout.addWidget(self.radio_decrypt)
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)

        self.file_widgets = {}
        file_labels = [
            ("initial_file", "Исходный файл:"),
            ("symmetric_key", "Зашифрованный симм. ключ:"),
            ("public_key", "Публичный ключ RSA (собеседника):"),
            ("secret_key", "Приватный ключ RSA (Ваш):"),
            ("encrypted_file", "Зашифрованный файл:"),
            ("decrypted_file", "Расшифрованный файл:"),
        ]
        for key, label_text in file_labels:
            layout = QHBoxLayout()
            label = QLabel(label_text)
            label.setFixedWidth(200)
            line_edit = QLineEdit()
            line_edit.editingFinished.connect(self.update_file_settings)
            button = QPushButton("Обзор...")
            button.clicked.connect(lambda checked, k=key: self.browse_file(k))
            layout.addWidget(label)
            layout.addWidget(line_edit)
            layout.addWidget(button)
            main_layout.addLayout(layout)
            self.file_widgets[key] = line_edit

        bottom_layout = QHBoxLayout()

        self.start_button = QPushButton("Старт")
        self.start_button.clicked.connect(self.start_process)
        bottom_layout.addWidget(self.start_button)

        gen_group = QGroupBox("Генерация ключей")
        gen_layout = QVBoxLayout()

        key_len_layout = QHBoxLayout()
        key_len_layout.addWidget(QLabel("Длина ключа:"))
        self.len_128 = QRadioButton("128")
        self.len_192 = QRadioButton("192")
        self.len_256 = QRadioButton("256")
        self.len_256.setChecked(True)
        key_len_layout.addWidget(self.len_128)
        key_len_layout.addWidget(self.len_192)
        key_len_layout.addWidget(self.len_256)
        gen_layout.addLayout(key_len_layout)

        self.gen_button = QPushButton("Сгенерировать ключи")
        self.gen_button.clicked.connect(self.generate_keys)
        gen_layout.addWidget(self.gen_button)

        gen_group.setLayout(gen_layout)
        bottom_layout.addWidget(gen_group)

        main_layout.addLayout(bottom_layout)

    def browse_file(self, key):
        """Opens the file selection dialog and inserts path to field."""
        logger.debug("Browse file dialog opened for: %s", key)
        if key in (
            "public_key",
            "secret_key",
            "symmetric_key",
            "encrypted_file",
            "decrypted_file",
        ):
            if key in ("encrypted_file", "decrypted_file"):
                path, _ = QFileDialog.getSaveFileName(self, f"Сохранить как {key}")
            else:
                path, _ = QFileDialog.getOpenFileName(self, f"Выберите файл {key}")
        else:
            path, _ = QFileDialog.getOpenFileName(self, f"Выберите файл {key}")

        if path:
            logger.info("User selected file for '%s': %s", key, path)
            self.file_widgets[key].setText(path)
            self.update_file_settings()

    def update_file_settings(self):
        """Updates file settings.json when change value in UI."""
        logger.debug("Updating settings.json from UI fields")
        self.settings.update(self.get_current_paths())
        with open("settings.json", "w", encoding="utf-8") as f:
            json.dump(self.settings, f)

    def generate_keys(self):
        """Generate public private and symmetric keys"""
        if self.len_128.isChecked():
            length = 128
        elif self.len_192.isChecked():
            length = 192
        else:
            length = 256

        logger.info("User requested key generation: symmetric_key_bits=%s", length)

        paths = self.get_current_paths()
        if (
            not paths["public_key"]
            or not paths["secret_key"]
            or not paths["symmetric_key"]
        ):
            logger.warning(
                "Key generation validation failed: public/private/symmetric key path is empty"
            )
            QMessageBox.warning(
                self,
                "Предупреждение",
                "Укажите пути для сохранения публичного, приватного и симметричного ключей.",
            )
            return

        try:
            key_gen(
                length,
                paths["public_key"],
                paths["secret_key"],
                paths["symmetric_key"],
            )
            logger.info("Key generation finished successfully")
            QMessageBox.information(self, "Успех", "Ключи успешно сгенерированы.")
        except Exception as e:
            logger.exception("Key generation failed")
            QMessageBox.critical(self, "Ошибка", f"Ошибка генерации ключей:\n{str(e)}")

    def apply_settings(self):
        """Load fields from settings.json."""
        logger.debug("Applying settings to UI fields")
        for key, widget in self.file_widgets.items():
            if key in self.settings:
                widget.setText(self.settings[key])

    def get_current_paths(self):
        """Get paths from UI fields.
        :return: dict {key: path}"""
        return {key: widget.text().strip() for key, widget in self.file_widgets.items()}

    def start_process(self):
        """Start decrypt or encrypt data."""
        paths = self.get_current_paths()
        mode = "encrypt" if self.radio_encrypt.isChecked() else "decrypt"
        logger.info("User started process: mode=%s", mode)

        try:
            if mode == "encrypt":
                required = [
                    "initial_file",
                    "symmetric_key",
                    "secret_key",
                    "encrypted_file",
                ]
                for req in required:
                    if not paths[req]:
                        logger.warning("Validation failed: field '%s' is empty", req)
                        raise ValueError(f"Поле '{req}' не заполнено.")
                    if req in required[:3]:
                        try:
                            with open(paths[req], "rb") as _:
                                pass
                        except Exception:
                            logger.error(
                                "Required file is missing or unreadable: %s", paths[req]
                            )
                            raise FileNotFoundError(f"Не найден файл '{paths[req]}'")

                logger.info(
                    "Encryption parameters validated: input=%s, output=%s",
                    paths["initial_file"],
                    paths["encrypted_file"],
                )
                encrypt(
                    paths["initial_file"],
                    paths["symmetric_key"],
                    paths["secret_key"],
                    paths["encrypted_file"],
                )

                logger.info("Encryption process finished successfully")
                QMessageBox.information(
                    self,
                    "Успех",
                    f"Файл зашифрован.\nРезультат сохранён в:\n{paths['encrypted_file']}",
                )
            else:
                required = [
                    "encrypted_file",
                    "symmetric_key",
                    "secret_key",
                    "decrypted_file",
                ]
                for req in required:
                    if not paths[req]:
                        logger.warning("Validation failed: field '%s' is empty", req)
                        raise ValueError(f"Поле '{req}' не заполнено.")
                    if req in required[:3]:
                        try:
                            with open(paths[req], "rb") as _:
                                pass
                        except Exception:
                            logger.error(
                                "Required file is missing or unreadable: %s", paths[req]
                            )
                            raise FileNotFoundError(f"Не найден файл '{paths[req]}'")

                logger.info(
                    "Decryption parameters validated: input=%s, output=%s",
                    paths["encrypted_file"],
                    paths["decrypted_file"],
                )
                decrypt(
                    paths["encrypted_file"],
                    paths["symmetric_key"],
                    paths["secret_key"],
                    paths["decrypted_file"],
                )

                logger.info("Decryption process finished successfully")
                QMessageBox.information(
                    self,
                    "Успех",
                    f"Файл расшифрован.\nРезультат сохранён в:\n{paths['decrypted_file']}",
                )

        except Exception as e:
            logger.exception("Process failed: mode=%s", mode)
            QMessageBox.critical(self, "Ошибка", str(e))


def main():
    """Entry point. Main function"""
    setup_logging()
    logger.info("Application starting")
    app = QApplication(sys.argv)
    window = CryptoApp()
    window.show()
    exit_code = app.exec_()
    logger.info("Application stopped with exit code=%s", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
