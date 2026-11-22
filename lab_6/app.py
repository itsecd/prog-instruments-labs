import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit, QFileDialog,
    QComboBox, QTextEdit, QVBoxLayout, QHBoxLayout, QGroupBox, QMessageBox
)
from config_utils import load_config, save_config
from crypto_utils import generate_keys, encrypt_file, decrypt_file


class CryptoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная №3 — RSA + AES")
        self.resize(780, 560)
        self.config_path = "config.json"
        self.settings = self.get_default_settings()
        if os.path.exists(self.config_path):
            self.settings.update(load_config(self.config_path))
        self.make_ui()
        self.log("Программа запущена")
        self.log("Вариант 1: AES, ключ 128/192/256 бит")

    def get_default_settings(self):
        base = os.getcwd()
        return {
            "input_file": os.path.join(base, "input.txt"),
            "encrypted_file": os.path.join(base, "encrypted.bin"),
            "decrypted_file": os.path.join(base, "output.txt"),
            "encrypted_key_file": os.path.join(base, "aes_key.enc"),
            "public_key_file": os.path.join(base, "public.pem"),
            "private_key_file": os.path.join(base, "private.pem"),
            "aes_key_size": 128
        }

    def make_ui(self):
        main = QVBoxLayout()

        title = QLabel("Гибридная криптосистема RSA + AES")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        main.addWidget(title)
        main.addWidget(QLabel("PyQt-интерфейс, JSON-настройки и CLI через argparse"))

        keys = QGroupBox("1. Ключи")
        keys_layout = QVBoxLayout()
        self.encrypted_key_edit = QLineEdit(self.settings["encrypted_key_file"])
        self.public_key_edit = QLineEdit(self.settings["public_key_file"])
        self.private_key_edit = QLineEdit(self.settings["private_key_file"])
        keys_layout.addLayout(self.path_row("Зашифрованный AES ключ:", self.encrypted_key_edit, True))
        keys_layout.addLayout(self.path_row("Открытый RSA ключ:", self.public_key_edit, True))
        keys_layout.addLayout(self.path_row("Закрытый RSA ключ:", self.private_key_edit, True))

        aes_row = QHBoxLayout()
        aes_row.addWidget(QLabel("Размер AES ключа:"))
        self.aes_box = QComboBox()
        self.aes_box.addItems(["128", "192", "256"])
        self.aes_box.setCurrentText(str(self.settings["aes_key_size"]))
        aes_row.addWidget(self.aes_box)
        aes_row.addWidget(QLabel("бит"))
        aes_row.addStretch()
        keys_layout.addLayout(aes_row)

        gen_btn = QPushButton("Сгенерировать ключи")
        gen_btn.clicked.connect(self.generate_action)
        keys_layout.addWidget(gen_btn)
        keys.setLayout(keys_layout)
        main.addWidget(keys)

        files = QGroupBox("2. Файлы")
        files_layout = QVBoxLayout()
        self.input_edit = QLineEdit(self.settings["input_file"])
        self.encrypted_edit = QLineEdit(self.settings["encrypted_file"])
        self.decrypted_edit = QLineEdit(self.settings["decrypted_file"])
        files_layout.addLayout(self.path_row("Исходный файл:", self.input_edit, False))
        files_layout.addLayout(self.path_row("Зашифрованный файл:", self.encrypted_edit, True))
        files_layout.addLayout(self.path_row("Расшифрованный файл:", self.decrypted_edit, True))
        files.setLayout(files_layout)
        main.addWidget(files)

        buttons = QHBoxLayout()
        enc_btn = QPushButton("Зашифровать")
        dec_btn = QPushButton("Расшифровать")
        save_btn = QPushButton("Сохранить config.json")
        enc_btn.clicked.connect(self.encrypt_action)
        dec_btn.clicked.connect(self.decrypt_action)
        save_btn.clicked.connect(self.save_config_action)
        buttons.addWidget(enc_btn)
        buttons.addWidget(dec_btn)
        buttons.addWidget(save_btn)
        main.addLayout(buttons)

        log_group = QGroupBox("Журнал работы")
        log_layout = QVBoxLayout()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        log_layout.addWidget(self.log_box)
        log_group.setLayout(log_layout)
        main.addWidget(log_group)

        self.setLayout(main)

    def path_row(self, text, edit, save_mode):
        row = QHBoxLayout()
        label = QLabel(text)
        label.setFixedWidth(190)
        btn = QPushButton("Обзор")
        btn.clicked.connect(lambda: self.choose_path(edit, save_mode))
        row.addWidget(label)
        row.addWidget(edit)
        row.addWidget(btn)
        return row

    def choose_path(self, edit, save_mode):
        if save_mode:
            path, _ = QFileDialog.getSaveFileName(self, "Выберите путь")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Выберите файл")
        if path:
            edit.setText(path)

    def collect_settings(self):
        return {
            "input_file": self.input_edit.text(),
            "encrypted_file": self.encrypted_edit.text(),
            "decrypted_file": self.decrypted_edit.text(),
            "encrypted_key_file": self.encrypted_key_edit.text(),
            "public_key_file": self.public_key_edit.text(),
            "private_key_file": self.private_key_edit.text(),
            "aes_key_size": int(self.aes_box.currentText())
        }

    def log(self, text):
        self.log_box.append(text)

    def save_config_action(self):
        try:
            save_config(self.config_path, self.collect_settings())
            self.log("config.json сохранён")
            QMessageBox.information(self, "Готово", "Настройки сохранены")
        except Exception as e:
            self.show_error(e)

    def generate_action(self):
        try:
            s = self.collect_settings()
            self.log("Генерирую AES и RSA ключи...")
            generate_keys(s["encrypted_key_file"], s["public_key_file"], s["private_key_file"], s["aes_key_size"])
            save_config(self.config_path, s)
            self.log("Ключи созданы")
            QMessageBox.information(self, "Готово", "Ключи созданы")
        except Exception as e:
            self.show_error(e)

    def encrypt_action(self):
        try:
            s = self.collect_settings()
            self.log("Запущено шифрование файла...")
            encrypt_file(s["input_file"], s["public_key_file"], s["encrypted_key_file"], s["encrypted_file"], s["aes_key_size"])
            save_config(self.config_path, s)
            self.log("Файл зашифрован: " + s["encrypted_file"])
            QMessageBox.information(self, "Готово", "Файл зашифрован")
        except Exception as e:
            self.show_error(e)

    def decrypt_action(self):
        try:
            s = self.collect_settings()
            self.log("Запущено дешифрование файла...")
            decrypt_file(s["encrypted_file"], s["private_key_file"], s["encrypted_key_file"], s["decrypted_file"])
            save_config(self.config_path, s)
            self.log("Файл расшифрован: " + s["decrypted_file"])
            QMessageBox.information(self, "Готово", "Файл расшифрован")
        except Exception as e:
            self.show_error(e)

    def show_error(self, error):
        self.log("Ошибка: " + str(error))
        QMessageBox.critical(self, "Ошибка", str(error))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CryptoWindow()
    window.show()
    sys.exit(app.exec_())
