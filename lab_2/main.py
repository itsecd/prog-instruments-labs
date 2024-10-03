import argparse
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives import padding as padding2
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
import yaml


def key_generation(symmetric_key_path: str, public_key_path: str, secret_key_path: str) -> None:
    """
    :param symmetric_key_path:  путь, по которому сериализовать зашифрованный симметричный ключ
    :param public_key_path: путь, по которому сериализовать открытый ключ
    :param secret_key_path: путь, по которому сериализовать закрытый ключ
    :return: ничего не возращает
    """
    with tqdm(100, desc='Generation keys') as prograssbar:
        # генерация ключа симметричного алгоритма шифрования
        symmetric_key = os.urandom(16)

        # генерация пары ключей для асимметричного алгоритма шифрования
        keys = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        private_key = keys
        public_key = keys.public_key()
        prograssbar.update(20)

        # сериализация открытого ключа в файл
        public_pem = public_key_path + '\\key.pem'
        with open(public_pem, 'wb') as public_out:
            public_out.write(public_key.public_bytes(encoding=serialization.Encoding.PEM,
                                                     format=serialization.PublicFormat.SubjectPublicKeyInfo))
        prograssbar.update(20)

        # сериализация закрытого ключа в файл
        private_pem = secret_key_path + '\\key.pem'
        with open(private_pem, 'wb') as private_out:
            private_out.write(private_key.private_bytes(encoding=serialization.Encoding.PEM,
                                                        format=serialization.PrivateFormat.TraditionalOpenSSL,
                                                        encryption_algorithm=serialization.NoEncryption()))
        prograssbar.update(20)

        # шифрование симметричного ключа открытым ключом при помощи RSA-OAEP
        encrypted_symmetric_key = public_key.encrypt(symmetric_key,
                                                    padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                                                algorithm=hashes.SHA256(),
                                                                label=None))
        prograssbar.update(20)

        # сериализация ключа симмеричного алгоритма в файл
        symmetric_file = symmetric_key_path + '\\key.txt'
        with open(symmetric_file, 'wb') as key_file:
            key_file.write(encrypted_symmetric_key)
        prograssbar.update(20)


def encrypt_data(initial_file_path: str, secret_key_path: str, symmetric_key_path: str,
                 encrypted_file_path: str) -> None:
    """
    :param initial_file_path: путь к шифруемому текстовому файлу
    :param secret_key_path: путь к закрытому ключу ассиметричного алгоритма
    :param symmetric_key_path: путь к зашифрованному ключу симметричного алгоритма
    :param encrypted_file_path: путь, по которому сохранить зашифрованный текстовый файл
    :return: ничего не возвращает
    """
    # десериализация ключа симметричного алгоритма
    symmetric_file = symmetric_key_path + '\\key.txt'
    with open(symmetric_file, mode='rb') as key_file:
        encrypted_symmetric_key = key_file.read()

    # десериализация закрытого ключа
    private_pem = secret_key_path + '\\key.pem'
    with open(private_pem, 'rb') as pem_in:
        private_bytes = pem_in.read()
    private_key = load_pem_private_key(private_bytes, password=None)

    # дешифрование симметричного ключа асимметричным алгоритмом
    d_symmetric_key = private_key.decrypt(encrypted_symmetric_key,
                                          padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                                       algorithm=hashes.SHA256(),
                                                       label=None))

    # паддинг данных для работы блочного шифра - делаем длину сообщения кратной длине шифруемого блока (64 бита)
    initial_file = initial_file_path + '\\text.txt'
    with open(initial_file, 'r') as _file:
        initial_content = _file.read()
    padder = padding2.ANSIX923(64).padder()
    text = bytes(initial_content, 'UTF-8')
    padded_text = padder.update(text) + padder.finalize()

    # шифрование текста симметричным алгоритмом
    # iv - случайное значение для инициализации блочного режима,должно быть размером с блок и каждый раз новым
    iv = os.urandom(8)
    cipher = Cipher(algorithms.IDEA(d_symmetric_key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    c_text = encryptor.update(padded_text) + encryptor.finalize()

    # зашифрованный текст хранится в виде словаря, где под ключом 'text' хранится сам зашифрованный текст,
    # a 'iv' - случайное значение для инициализации блочного режима, которое нужно для декодирования текста
    dict_t = {'text': c_text, 'iv': iv}
    encrypted_file = encrypted_file_path + '\\file.yaml'
    with open(encrypted_file, 'w') as _file:
        yaml.dump(dict_t, _file)


def decrypting_data(encrypted_file_path: str, secret_key_path: str, symmetric_key_path: str,
                    decrypted_file_path: str) -> None:
    """
    :param encrypted_file_path: путь к зашифрованному текстовому файлу
    :param secret_key_path: путь к закрытому ключу ассиметричного алгоритма
    :param symmetric_key_path: путь к зашифрованному ключу симметричного алгоритма
    :param decrypted_file_path: путь, по которому сохранить расшифрованный текстовый файл
    :return: ничего не возвращает
    """
    # десериализация ключа симметричного алгоритма
    symmetric_file = symmetric_key_path + '\\key.txt'
    with open(symmetric_file, mode='rb') as key_file:
        encrypted_symmetric_key = key_file.read()

    # десериализация закрытого ключа
    private_pem = secret_key_path + '\\key.pem'
    with open(private_pem, 'rb') as pem_in:
        private_bytes = pem_in.read()
    private_key = load_pem_private_key(private_bytes, password=None)

    # дешифрование симметричного ключа асимметричным алгоритмом
    dsymmetric_key = private_key.decrypt(encrypted_symmetric_key,
                                         padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                                      algorithm=hashes.SHA256(),
                                                      label=None))

    # десериализация шифрованного файла
    encrypted_file = encrypted_file_path + '\\file.yaml'
    with open(encrypted_file) as _file:
        content_encrypted = yaml.safe_load(_file)

    text_enc = content_encrypted["text"]
    iv_enc = content_encrypted["iv"]

    # дешифрование и депаддинг текста симметричным алгоритмом
    cipher = Cipher(algorithms.IDEA(dsymmetric_key), modes.CBC(iv_enc))
    decryptor = cipher.decryptor()
    dc_text = decryptor.update(text_enc) + decryptor.finalize()

    unpadder = padding2.ANSIX923(64).unpadder()
    unpadded_dc_text = unpadder.update(dc_text) + unpadder.finalize()

    decrypted_file = decrypted_file_path + '\\file.txt'
    with open(decrypted_file, 'w') as _file:
        _file.write(str(unpadded_dc_text))


parser = argparse.ArgumentParser(description="main")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-gen', '--generation', help='Запускает режим генерации ключей')
group.add_argument('-enc', '--encryption', help='Запускает режим шифрования')
group.add_argument('-dec', '--decryption', help='Запускает режим дешифрования')
parser.add_argument('-symkey',
                    type=str,
                    help='Это обязательный строковый позиционный аргумент,'
                         'который указывает, путь к папке, в которую сериализуется зашифрованный симметричный ключ',
                    dest='symmetric_key_path')
parser.add_argument('-pubkey',
                    type=str,
                    help='Это обязательный строковый позиционный аргумент,'
                         'который указывает, путь к папке, в которую сериализуется открытый ключ',
                    dest='public_key_path')
parser.add_argument('-seckey',
                    type=str,
                    help='Это обязательный строковый позиционный аргумент,'
                         'который указывает, путь к папке, в которую сериализуется закрытый ключ',
                    dest='secret_key_path')
parser.add_argument('-initial',
                    type=str,
                    help='Это обязательный строковый позиционный аргумент,'
                         'который указывает, путь к папке, в которой хранится начальный файл',
                    dest='initial_file_path')
parser.add_argument('-encrypted',
                    type=str,
                    help='Это обязательный строковый позиционный аргумент,'
                         'который указывает, путь к папке, в которую сохраняется шифрованный файл',
                    dest='encrypted_file_path')
parser.add_argument('-dencrypted',
                    type=str,
                    help='Это обязательный строковый позиционный аргумент,'
                         'который указывает, путь к папке, в которую сохраняется дешифрованный файл',
                    dest='decrypted_file_path')
args = parser.parse_args()

if args.generation:
    key_generation(args.symmetric_key_path, args.public_key_path, args.secret_key_path)
if args.encryption:
    with tqdm(100, desc='Encryption mode') as prograssbar:
        encrypt_data(args.initial_file_path, args.secret_key_path, args.symmetric_key_path, args.encrypted_file_path)
        prograssbar.update(100)
if args.decryption:
    with tqdm(100, desc='Decryption mode') as prograssbar:
        decrypting_data(args.encrypted_file_path, args.secret_key_path, args.symmetric_key_path,
                        args.decrypted_file_path)
        prograssbar.update(100)