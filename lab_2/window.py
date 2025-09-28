import sys

from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget
)

from function import *


class HybridCryptosystem(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.init_ui()

    def init_ui(self) -> None:
        self.setWindowTitle("Hybrid cryptosystem")

        layout = QVBoxLayout()

        self.load_text_button = QPushButton("Load text", self)
        self.load_text_button.clicked.connect(self.load_txt)
        layout.addWidget(self.load_text_button)

        self.load_key_button = QPushButton("Load SM4 key", self)
        self.load_key_button.clicked.connect(self.generate_key_sm4)
        layout.addWidget(self.load_key_button)

        self.load_key_button = QPushButton("Load SM4 key: Not generated", self)
        self.load_key_button.clicked.connect(self.load_key_sm4)
        layout.addWidget(self.load_key_button)

        self.encrypt_text_button = QPushButton("Encrypt text using SM4", self)
        self.encrypt_text_button.clicked.connect(self.encrypt_text)
        layout.addWidget(self.encrypt_text_button)

        self.save_encrypt_text_button = QPushButton("Save encrypt text", self)
        self.save_encrypt_text_button.clicked.connect(self.save_encrypt_text)
        layout.addWidget(self.save_encrypt_text_button)

        self.decrypt_text_button = QPushButton("Decrypt text using SM4", self)
        self.decrypt_text_button.clicked.connect(self.decrypt_text)
        layout.addWidget(self.decrypt_text_button)

        self.save_decrypt_text_button = QPushButton("Save decrypt text", self)
        self.save_decrypt_text_button.clicked.connect(self.save_decrypt_text)
        layout.addWidget(self.save_decrypt_text_button)

        self.generate_keys_button = QPushButton("Generate RSA Keys", self)
        self.generate_keys_button.clicked.connect(self.generate_rsa_keys)
        layout.addWidget(self.generate_keys_button)

        self.load_public_key_button = QPushButton("Load public key", self)
        self.load_public_key_button.clicked.connect(self.load_pub_key)
        layout.addWidget(self.load_public_key_button)

        self.load_private_key_button = QPushButton("Load private key", self)
        self.load_private_key_button.clicked.connect(self.load_priv_key)
        layout.addWidget(self.load_private_key_button)

        self.encrypt_button = QPushButton("Encrypt symmetric key", self)
        self.encrypt_button.clicked.connect(self.encrypt_symmetric_key)
        layout.addWidget(self.encrypt_button)

        self.save_encrypt_key = QPushButton("Save encrypt symmetric key", self)
        self.save_encrypt_key.clicked.connect(self.save_encrypted_key)
        layout.addWidget(self.save_encrypt_key)

        self.decrypt_button = QPushButton("Decrypt symmetric key", self)
        self.decrypt_button.clicked.connect(self.decrypt_symmetric_key)
        layout.addWidget(self.decrypt_button)

        self.save_decrypt_key = QPushButton("Save decrypt symmetric key", self)
        self.save_decrypt_key.clicked.connect(self.save_decrypted_key)
        layout.addWidget(self.save_decrypt_key)

        self.setLayout(layout)
        self.encrypted_key = None
        self.decrypted_key = None
        self.key_sm4 = None
        self.public_key = None
        self.private_key = None
        self.decrypted_text = None
        self.encrypted_text = None
        self.text = None

    def load_txt(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            try:
                self.text = FileHandler.read_txt_file(file_path)
                QMessageBox.information(self, "Success", "Loaded successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load: {str(e)}")

    def generate_key_sm4(self) -> None:
        try:
            self.key_sm4 = SymmetricEncryption.generate_sm4_key()
            filename, _ = QFileDialog.getSaveFileName(self, "Save SM4 Key", "", "Text Files (*.txt);;All Files (*)")
            if filename:
                FileHandler.write_txt_file(self.key_sm4, filename)
            QMessageBox.information(self, "Success", "SM4 key generated and saved successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to generate SM4 key: {str(e)}")

    def load_key_sm4(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File With SM4 Key", "",
                                                   "Text Files (*.txt);;All Files (*)")
        if file_path:
            try:
                self.key_sm4 = FileHandler.read_txt_file(file_path)
                QMessageBox.information(self, "Success", "Loaded key for SM4 successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load key for SM4: {str(e)}")

    def encrypt_text(self) -> None:
        if self.text is None or self.key_sm4 is None:
            QMessageBox.warning(self, "Warning", "Please load text and key for SM4.")
            return
        try:
            self.encrypted_text = SymmetricEncryption.sm4_encrypt(self.key_sm4, self.text)
            QMessageBox.information(self, "Success", "Text encrypted successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to encrypt text: {str(e)}")

    def save_encrypt_text(self) -> None:
        if not (self.encrypted_text is None):
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Encrypted Key", "",
                                                       "Text Files (*.txt);;All Files (*)")
            if file_path:
                try:
                    FileHandler.write_txt_file(self.encrypted_text, file_path)
                    QMessageBox.information(self, "Success", "Encrypted text saved successfully.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to save encrypted text: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please encrypt text.")
            return

    def decrypt_text(self) -> None:
        if self.encrypted_text is None or self.key_sm4 is None:
            QMessageBox.warning(self, "Warning", "Please load text and key for SM4.")
            return
        try:
            self.decrypted_text = SymmetricEncryption.sm4_decrypt(self.key_sm4, self.encrypted_text)
            QMessageBox.information(self, "Success", "Text decrypted successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to decrypt text: {str(e)}")

    def save_decrypt_text(self) -> None:
        if not (self.decrypted_text is None):
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Encrypted Key", "",
                                                       "Text Files (*.txt);;All Files (*)")
            if file_path:
                try:
                    FileHandler.write_txt_file(self.decrypted_text, file_path)
                    QMessageBox.information(self, "Success", "Decrypted text saved successfully.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to save decrypted text: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please decrypt text.")
            return

    def generate_rsa_keys(self) -> None:
        try:
            private_key, public_key = AsymmetricEncryption.generate_rsa_keys()
            private_filename, _ = QFileDialog.getSaveFileName(self, "Save Private Key", "",
                                                              "Text Files (*.txt);;All Files (*)")
            if private_filename:
                AsymmetricEncryption.serialize_private_key(private_key, private_filename)

            public_filename, _ = QFileDialog.getSaveFileName(self, "Save Public Key", "",
                                                             "Text Files (*.txt);;All Files (*)")
            if public_filename:
                AsymmetricEncryption.serialize_public_key(public_key, public_filename)
            QMessageBox.information(self, "Success", "RSA keys generated and saved successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to generate RSA keys: {str(e)}")

    def load_pub_key(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            try:
                self.public_key = AsymmetricEncryption.load_public_key(file_path)
                QMessageBox.information(self, "Success", "Loaded public key successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load public key: {str(e)}")

    def load_priv_key(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            try:
                self.private_key = AsymmetricEncryption.load_private_key(file_path)
                QMessageBox.information(self, "Success", "Loaded public key successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load public key: {str(e)}")

    def encrypt_symmetric_key(self) -> None:
        if self.public_key is None or self.key_sm4 is None:
            QMessageBox.warning(self, "Warning", "Please load a public key and key for SM4.")
            return
        try:
            self.encrypted_key = AsymmetricEncryption.encrypt_symmetric_key(self.key_sm4, self.public_key)
            QMessageBox.information(self, "Success", "Symmetric key encrypted successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to encrypt symmetric key: {str(e)}")

    def save_encrypted_key(self) -> None:
        if not (self.encrypted_key is None):
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Encrypted Key", "",
                                                       "Text Files (*.txt);;All Files (*)")
            if file_path:
                try:
                    FileHandler.write_txt_file(self.encrypted_key, file_path)
                    QMessageBox.information(self, "Success", "Encrypted symmetric key saved successfully.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to save encrypted key: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please encrypt symmetric key.")
            return

    def decrypt_symmetric_key(self) -> None:
        if not (self.encrypted_key is None):
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Private Key File", "",
                                                       "Text Files (*.txt);;All Files (*)")
            if file_path:
                try:
                    private_key = AsymmetricEncryption.load_private_key(file_path)
                    self.decrypted_key = AsymmetricEncryption.decrypt_symmetric_key(self.encrypted_key, private_key)
                    QMessageBox.information(self, "Success", f"Decrypted symmetric key saved successfully")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to decrypt symmetric key: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please load encrypt symmetric key.")
            return

    def save_decrypted_key(self) -> None:
        if not (self.decrypted_key is None):
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Decrypted Key", "",
                                                       "Text Files (*.txt);;All Files (*)")
            if file_path:
                try:
                    FileHandler.write_txt_file(self.decrypted_key, file_path)
                    QMessageBox.information(self, "Success", "Decrypted symmetric key saved successfully.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to save decrypted key: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please decrypt symmetric key.")
            return


if __name__ == "__main__":
    app = QApplication(sys.argv)
    encryptor = HybridCryptosystem()
    encryptor.resize(400, 200)
    encryptor.show()
    sys.exit(app.exec_())