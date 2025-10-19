import json
import os
import shutil

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding




class FileManager:

    @staticmethod
    def save_keys(encrypted_sym_key_path, public_key_path, private_key_path,
                  public_key, private_key, symmetric_key):

        print("Сохранение ключей...")

        with open(public_key_path, 'wb') as pub_out:
            pub_out.write(public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))

        with open(private_key_path, 'wb') as priv_out:
            priv_out.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

        encrypted_key = public_key.encrypt(
            symmetric_key,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        with open(encrypted_sym_key_path, 'wb') as sym_out:
            sym_out.write(encrypted_key)

        print(f"Публичный ключ сохранен в {public_key_path}")
        print(f"Приватный ключ сохранен в {private_key_path}")
        print(f"Зашифрованный симметричный ключ сохранен в {encrypted_sym_key_path}")

    @staticmethod
    def encrypt_file(input_file, output_file, encryptor):

        print(f"Шифрование файла {input_file}...")

        with open(input_file, 'rb') as f:
            plaintext = f.read()

        ciphertext = encryptor.encrypt(plaintext)

        with open(output_file, 'wb') as f:
            f.write(encryptor.iv)
            f.write(ciphertext)

        print("\nHex-дамп зашифрованного файла:")
        with open(output_file, 'rb') as f:
            print(f.read().hex())

        print(f"Файл зашифрован и сохранен в {output_file}")

    @staticmethod
    def decrypt_file(input_file, output_file, decryptor):

        print(f"Расшифровка файла {input_file}...")

        with open(input_file, 'rb') as f:
            iv = f.read(8)
            ciphertext = f.read()

        decrypted = decryptor.decrypt(ciphertext, iv)

        with open(output_file, 'wb') as f:
            f.write(decrypted)

        print(f"Файл расшифрован и сохранен в {output_file}")

    @staticmethod
    def load_settings_file(settings_path):

        try:
            with open(settings_path) as f:
                return json.load(f)
        except:
            print("Ошибка загрузки настроек")
            return {}

    def backup_keys(self, settings):
        print("Создание резервной копии ключей...")

        backup_dir = "/backup/keys/"
        paths = {
            'private': "/backup/keys/private_backup.pem",
            'public': "/backup/keys/public_backup.pem",
            'symmetric': "/backup/keys/symmetric_backup.key"
        }

        try:
            # Создаем директорию если не существует
            os.makedirs(backup_dir, exist_ok=True)

            # Копируем файлы ключей
            if os.path.exists(settings['private_key']):
                shutil.copy2(settings['private_key'], paths['private'])
            if os.path.exists(settings['public_key']):
                shutil.copy2(settings['public_key'], paths['public'])
            if os.path.exists(settings['symmetric_key']):
                shutil.copy2(settings['symmetric_key'], paths['symmetric'])

            print(f"Резервные копии созданы в: {backup_dir}")
            return True

        except Exception as e:
            print(f"Ошибка при создании бэкапа: {e}")
            return False