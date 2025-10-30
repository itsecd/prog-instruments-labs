import sys

from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QDialog,
    QFileDialog,
    QMessageBox,
)

from constant import DEFAULT_DIRECTORY, FILTER, IconTypes
from filehandler import FileHandler
from hybrid_crypto_system.hybrid_crypto_system import HybridCryptoSystem
from hybrid_crypto_system.logger.logger_config import logger


class SelectLength(QDialog):
    """Dialog box with key length selection"""

    def __init__(self):
        """Initializing the dialog box"""
        super().__init__()
        self.setWindowTitle("Select key length")
        self.setFixedSize(700, 100)

        self.button64 = QPushButton("64 bits")
        self.button128 = QPushButton("128 bits")
        self.button192 = QPushButton("192 bits")

        self.button64.setStyleSheet("height: 130px; font-size: 18px;")
        self.button128.setStyleSheet("height: 130px; font-size: 18px;")
        self.button192.setStyleSheet("height: 130px; font-size: 18px;")

        layout = QHBoxLayout()
        layout.addWidget(self.button64)
        layout.addWidget(self.button128)
        layout.addWidget(self.button192)
        self.setLayout(layout)

        self.button64.clicked.connect(lambda: self.__select_length(64))
        self.button128.clicked.connect(lambda: self.__select_length(128))
        self.button192.clicked.connect(lambda: self.__select_length(192))

        self.__key_length = None

    def __select_length(self, length):
        """Length selection method"""
        self.__key_length = length
        self.accept()

    def get_key_length(self):
        return self.__key_length


class MainWindow(QMainWindow):
    """Hybrid Crypto System Application Window"""

    def __init__(self):
        """Initializing the application window"""
        logger.info("Hybrid Crypto System started")
        super().__init__()
        self.setWindowTitle("Hybrid CryptoSystem")
        self.setFixedSize(1280, 320)
        (container := QWidget()).setLayout(layout := QVBoxLayout())

        self.open_settings_button = QPushButton("Open settings file")
        self.select_key_length_button = QPushButton("Select key length")
        self.generator_button = QPushButton("Generate keys")
        self.encrypt_button = QPushButton("Encrypt data")
        self.decrypt_button = QPushButton("Decrypt data")

        self.open_settings_button.setStyleSheet("height: 130px; font-size: 18px;")
        self.select_key_length_button.setStyleSheet("height: 130px; font-size: 18px;")
        self.generator_button.setStyleSheet("height: 130px; font-size: 18px;")
        self.encrypt_button.setStyleSheet("height: 130px; font-size: 18px;")
        self.decrypt_button.setStyleSheet("height: 130px; font-size: 18px;")

        self.open_settings_button.clicked.connect(self.open_settings)
        self.select_key_length_button.clicked.connect(self.select_key_length)
        self.generator_button.clicked.connect(self.generate_keys)
        self.encrypt_button.clicked.connect(self.encrypt_data)
        self.decrypt_button.clicked.connect(self.decrypt_data)

        layout1 = QHBoxLayout()
        layout2 = QHBoxLayout()

        layout1.addWidget(self.generator_button)
        layout1.addWidget(self.encrypt_button)
        layout1.addWidget(self.decrypt_button)
        layout2.addWidget(self.open_settings_button)
        layout2.addWidget(self.select_key_length_button)

        layout.addLayout(layout1)
        layout.addLayout(layout2)

        self.setCentralWidget(container)

        self.__crypto_system = None
        self.__settings = None
        self.__dialog = None

    def show_message(self, title: str, text: str, icon_type: IconTypes):
        """Message output for the user"""
        msg = QMessageBox()
        msg.setStyleSheet("font-size: 14px;")
        msg.setWindowTitle(title)
        msg.setText(text)
        match icon_type:
            case IconTypes.Critical:
                icon = QMessageBox.Icon.Critical
            case IconTypes.Warning:
                icon = QMessageBox.Icon.Warning
            case IconTypes.Information:
                icon = QMessageBox.Icon.Information
            case IconTypes.Question:
                icon = QMessageBox.Icon.Question
            case _:
                icon = QMessageBox.Icon.NoIcon
        msg.setIcon(icon)
        msg.exec()

    def select_key_length(self):
        """Launch a dialog box to select the key length"""
        logger.info("Select key length dialog box started")
        self.__dialog = SelectLength()
        self.__dialog.exec()
        self.__crypto_system = HybridCryptoSystem(
            TripleDES,
            self.__dialog.get_key_length()
        )
        logger.debug("Selected key length: %d", self.__dialog.get_key_length())

    def open_settings(self):
        """Upload the settings file"""
        try:
            file, _ = QFileDialog.getOpenFileName(
                parent=QApplication.activeWindow(),
                caption="Select json file with settings",
                directory=DEFAULT_DIRECTORY,
                filter=FILTER,
            )
            if file:
                self.__settings = FileHandler.read_data(file, "r")
                self.show_message(
                    "Success", "Settings have been loaded", IconTypes.Information
                )
                logger.info("Settings have been loaded")
                if not QtCore.QFile.exists(file):
                    self.show_message("Error", "File not found", IconTypes.Critical)
            else:
                self.show_message(
                    "Error", "Please select a valid json file", IconTypes.Critical
                )
        except Exception as e:
            self.show_message(
                "Error",
                f"An error occurred while opening the file: {e}",
                IconTypes.Critical,
            )

    def generate_keys(self):
        """Launch key generation"""
        if not self.__settings:
            self.show_message(
                "Settings was not found", "Load settings first", IconTypes.Warning
            )
            return
        if not self.__crypto_system:
            self.show_message(
                "Key length not found", "Select key length first", IconTypes.Warning
            )
            return
        try:
            self.__crypto_system.generate_keys(
                self.__settings["symmetric_key"],
                self.__settings["private_key"],
                self.__settings["public_key"],
            )
            self.show_message(
                "Success", "Keys were saved to files", IconTypes.Information
            )
        except Exception as e:
            self.show_message(
                "Error!",
                f"An error occurred when generating keys: {e}",
                IconTypes.Critical,
            )

    def encrypt_data(self):
        """Launch data encryption"""
        if not self.__settings:
            self.show_message(
                "Settings was not found", "Load settings first", IconTypes.Warning
            )
            return
        if not self.__crypto_system:
            self.__crypto_system = HybridCryptoSystem()
        try:
            self.__crypto_system.encrypt_data(
                self.__settings["plain_text"],
                self.__settings["private_key"],
                self.__settings["symmetric_key"],
                self.__settings["encrypted_text"],
            )
            self.show_message("Success", "Data was encrypted", IconTypes.Information)
        except ValueError as ve:
            self.show_message(
                "Error", f"Something wrong in data: {ve}", IconTypes.Critical
            )
        except Exception as e:
            self.show_message(
                "Error",
                f"An error occurred while encrypting the data: {e}",
                IconTypes.Critical,
            )

    def decrypt_data(self):
        """Launch data decryption"""
        if not self.__settings:
            self.show_message(
                "Settings was not found", "Load settings first", IconTypes.Warning
            )
            return
        if not self.__crypto_system:
            self.__crypto_system = HybridCryptoSystem()
        try:
            self.__crypto_system.decrypt_data(
                self.__settings["encrypted_text"],
                self.__settings["private_key"],
                self.__settings["symmetric_key"],
                self.__settings["decrypted_text"],
            )
            self.show_message("Success", "Data was decrypted", IconTypes.Information)
        except ValueError as ve:
            self.show_message(
                "Error", f"Something wrong in data: {ve}", IconTypes.Critical
            )
        except Exception as e:
            self.show_message(
                "Error",
                f"An error occurred while decrypting the data: {e}",
                IconTypes.Critical,
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
