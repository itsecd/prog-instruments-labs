from crypt_system.files import *
from crypt_system.de_setialization import *
from crypt_system.asymmetric import Asymmetric
from crypt_system.symmetric import Symmetric
from loguru import logger as log

class Cryptosys:
    def __init__(self):
        try:
            self._settings = json_file_open("settings.json")
            self._flag = True
        except Exception as e:
            print("Файл с настройками не найден")

    def menu(self) -> None:
        """
        Меню выбора вариантов (консольное приложение)
        :return: None
        """
        print("Выберите действие")
        print("1 - Сгенерировать ключи ")
        print("2 - Зашифровать текст при помощи симметричного ключа")
        print("3 - Расшифровать зашифрованный текст")
        print("4 - Завершить работу программы")
        print("Выберите номер 1-4")

    def console_app(self) -> None:
        """
        Реализация консольного приложения
        :return: None
        """
        while self._flag:
            self.menu()
            action = input()
            match action:
                case "1":
                    try:
                        log.info("init key generation")
                        private_key, public_key = Asymmetric.asymmetrical_keygen()
                        symmetric_key = Symmetric.symmetrical_keygen()
                        serialization_public_key(
                            self._settings["public_key"], public_key
                        )
                        serialization_private_key(
                            self._settings["private_key"], private_key
                        )
                        d_sym_key = Asymmetric.symmetrical_key_encryptor(
                            symmetric_key, public_key
                        )
                        serialization_symmetrical(
                            self._settings["encrypt_symmetric_key"], d_sym_key
                        )
                        log.info("keys generated")
                        print("Ключи сгенерированы")
                    except Exception as e:
                        print(f"Failed to generate keys: {e}")
                case "2":
                    try:
                        log.info("init encrypt the text using a symmetric key")
                        text = txt_file_open(self._settings["text"])
                        encrypted_symmetrical_key = deserialization_symmetrical(
                            self._settings["encrypt_symmetric_key"]
                        )
                        decrypted_symmetrical_key = (
                            Asymmetric.symmetrical_key_decryptor(
                                encrypted_symmetrical_key,
                                deserialization_private_key(
                                    self._settings["private_key"]
                                ),
                            )
                        )
                        c_text = Symmetric.text_encrypter(
                            text, decrypted_symmetrical_key
                        )
                        bytes_file_save(self._settings["encrypted_text"], c_text)
                        log.info("text encrypted by symmetric key")
                        print("Текст зашифрован")
                    except Exception as e:
                        print(f"Couldn't encrypt the text: {e}")
                case "3":
                    try:
                        log.info("init decrypt the encrypted text")
                        encrypted_text = txt_file_open(self._settings["encrypted_text"])
                        encrypted_symmetrical_key = deserialization_symmetrical(
                            self._settings["encrypt_symmetric_key"]
                        )
                        decrypted_symmetrical_key = (
                            Asymmetric.symmetrical_key_decryptor(
                                encrypted_symmetrical_key,
                                deserialization_private_key(
                                    self._settings["private_key"]
                                ),
                            )
                        )
                        decrypted_text = Symmetric.text_decrypter(
                            encrypted_text, decrypted_symmetrical_key
                        )
                        txt_file_save(
                            decrypted_text.decode("UTF-8"),
                            self._settings["decrypted_text"],
                        )
                        log.info("text decrypted")
                        print("Текст расшифрован")
                    except Exception as e:
                        print(f"Couldn't decipher the text: {e}")
                case "4":
                    log.info("program is completed")
                    self._flag = False
                case _:
                    log.info("user input incorrect number")
                    print("Пожалуйста, выберите номер от 1 до 5")

