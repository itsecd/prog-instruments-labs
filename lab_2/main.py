import argparse
import os

from sup_functions import SupportFunctions
from works_with_files import WorkWithFiles
from asymmetrical_crypt import Asymmetrical
from symmetrical_crypt import Symmetrical


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-gen', '--generation',
                       help='Запускает режим генерации ключей')
    group.add_argument('-enc', '--encryption',
                       help='Запускает режим шифрования')
    group.add_argument('-dec', '--decryption',
                       help='Запускает режим дешифрования')
    args = parser.parse_args()
    json_path = None
    if args.generation is not None:
        json_path = args.generation
    elif args.encryption is not None:
        json_path = args.encryption
    elif args.decryption is not None:
        json_path = args.decryption

    settings = WorkWithFiles.load_config_settings(json_path if isinstance(json_path, str) else None)

    if args.generation is not None:
        public_key, private_key, encrypted_symmetric_key = SupportFunctions.generate_keys(settings)
        WorkWithFiles.save_public_key(public_key, settings['public_key'])
        WorkWithFiles.save_private_key(private_key, settings['secret_key'])
        WorkWithFiles.save_encrypt_symmetric_key(encrypted_symmetric_key, settings)
    elif args.encryption is not None:
        print("\n||Шифрование информации с помощью алгоритма ChaCha20||")
        path_to_initial = settings['initial_file']
        path_to_private_key = settings['secret_key']
        path_to_encrypted_sym_key = settings['encrypted_symmetric_key_file']
        encrypted_file_path = settings['encrypted_file']
        if not path_to_encrypted_sym_key:
            path_to_encrypted_sym_key = settings['symmetric_key']
        if not all([path_to_initial, path_to_private_key,
                    path_to_encrypted_sym_key, encrypted_file_path]):
            print(
                "Error: Не указаны все необходимые пути в настройках для шифрования.")
            exit(1)
        if not os.path.exists(path_to_initial):
            print(f"Error: Исходный файл не найден по пути {path_to_initial}")
            exit(1)
        if not os.path.exists(path_to_private_key):
            print(
                f"Error: Файл приватного ключа не найден по пути {path_to_private_key}")
            exit(1)
        if not os.path.exists(path_to_encrypted_sym_key):
            print(
                f"Error: Файл зашифрованного симметричного ключа не найден по пути {path_to_encrypted_sym_key}")
            exit(1)
        private_key = WorkWithFiles.read_private_key(path_to_private_key)
        encrypted_sym_key_data = WorkWithFiles.read_file(path_to_encrypted_sym_key)
        symmetric_key = Asymmetrical.decrypt_symmetric_key(private_key, encrypted_sym_key_data)
        print(f"Чтение файла {path_to_initial}...")
        content = WorkWithFiles.read_file(path_to_initial)

        ciphertext, nonce = Symmetrical.symmetric_encrypt_chacha20(content, symmetric_key)

        print(f"Сохранение зашифрованных данных в файл {encrypted_file_path}...")
        WorkWithFiles.write_file(encrypted_file_path, ciphertext)
        print(f"Сохранение nonce в файл {settings['nonce']}...")
        WorkWithFiles.write_file(settings['nonce'], nonce)
        print("||Шифрование и сохранение завершено успешно!||")
    elif args.decryption is not None:
        print("\n||Дешифрование информации с помощью алгоритма ChaCha20||")
        path_to_encrypt_file = settings['encrypted_file']
        path_to_private_key = settings['secret_key']
        path_to_encrypted_sym_key = settings['encrypted_symmetric_key_file'] or \
                                    settings['symmetric_key']
        path_to_decrypted_key = settings['decrypted_file']

        if not all([path_to_encrypt_file, path_to_private_key,
                    path_to_encrypted_sym_key, path_to_decrypted_key]):
            print(
                "Error: Не указаны все необходимые пути в настройках для дешифрования.")
            exit(1)
        if not os.path.exists(path_to_encrypt_file):
            print(
                f"Error: Зашифрованный файл не найден по пути {path_to_encrypt_file}")
            exit(1)
        if not os.path.exists(path_to_private_key):
            print(
                f"Error: Файл приватного ключа не найден по пути {path_to_private_key}")
            exit(1)
        if not os.path.exists(path_to_encrypted_sym_key):
            print(
                f"Error: Файл зашифрованного симметричного ключа не найден по пути {path_to_encrypted_sym_key}")
            exit(1)

        private_key = WorkWithFiles.read_private_key(path_to_private_key)
        encrypted_sym_key_data = WorkWithFiles.read_file(path_to_encrypted_sym_key)
        symmetric_key = Asymmetrical.decrypt_symmetric_key(private_key, encrypted_sym_key_data)
        print(f"Чтение зашифрованного файла {path_to_encrypt_file}...")
        encrypted_content = WorkWithFiles.read_file(path_to_encrypt_file)
        nonce = WorkWithFiles.read_file(settings['nonce'])

        plaintext = Symmetrical.symmetric_decrypt_chacha20(encrypted_content, symmetric_key, nonce)

        print(f"Сохранение расшифрованных данных в {path_to_decrypted_key}...")
        WorkWithFiles.write_file(path_to_decrypted_key, plaintext)
        print("||Дешифрование и сохранение завершено успешно!||")

if __name__ == "__main__":
    main()
