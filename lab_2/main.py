import argparse
import json
import os
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import load_pem_public_key, load_pem_private_key
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def hybrid_key_generation(settings: dict) -> None:

    # генерация ключа симметричного алгоритма шифрования
    key = os.urandom(32)
    print('ключ')
    print(key)

    # генерация пары ключей для асимметричного алгоритма шифрования
    keys = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    private_key = keys
    public_key = keys.public_key()

    # сериализация открытого ключа в файл
    public_pem = 'public.pem'
    with open(public_pem, mode='wb') as public_out:
        public_out.write(public_key.public_bytes(encoding=serialization.Encoding.PEM,
                                                 format=serialization.PublicFormat.SubjectPublicKeyInfo))
    settings['public_key'] = public_pem

    # сериализация закрытого ключа в файл
    private_pem = 'private.pem'
    with open(private_pem, mode='wb') as private_out:
        private_out.write(private_key.private_bytes(encoding=serialization.Encoding.PEM,
                                                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                                                    encryption_algorithm=serialization.NoEncryption()))
    settings['secret_key'] = private_pem

    # зашифрование ключа симметричного шифрования открытым ключом
    from cryptography.hazmat.primitives.asymmetric import padding

    symmetric_key_enc = public_key.encrypt(key, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                                             algorithm=hashes.SHA256(), label=None))
    print('зашифрованный ключ')
    print(symmetric_key_enc)

    # сериализация ключа симмеричного алгоритма в файл
    file_name = 'symmetric.txt'
    with open(file_name, mode='wb') as key_file:
        key_file.write(symmetric_key_enc)
    settings['symmetric_key'] = file_name


def hybrid_data_encryption(settings: dict) -> None:

    # десериализация ключа симметричного алгоритма
    with open(settings['symmetric_key'], mode='rb') as key_file:
        content = key_file.read()

    # десериализация закрытого ключа
    with open(settings['secret_key'], mode='rb') as pem_in:
        private_bytes = pem_in.read()
    d_private_key = load_pem_private_key(private_bytes, password=None, )

    # дешифрование ключа симметричного шифрования асимметричным алгоритмом
    from cryptography.hazmat.primitives.asymmetric import padding

    dc_symmetric_key = d_private_key.decrypt(content, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                                                   algorithm=hashes.SHA256(), label=None))
    print('расшифрованный ключ')
    print(dc_symmetric_key)

    # чтение исходных данных
    with open(settings['initial_file'], mode='r') as source_text:
        tmp = source_text.read()
    text = bytes(tmp, 'UTF-8')
    print('исходный текст')
    print(text)

    # паддинг данных для работы блочного шифра - делаем длину сообщения кратной длине шифркуемого блока
    from cryptography.hazmat.primitives import padding

    padder = padding.ANSIX923(128).padder()
    padded_text = padder.update(text) + padder.finalize()
    print('исходный текст с добавкой')
    print(padded_text)

    # шифрование текста симметричным алгоритмом
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(dc_symmetric_key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    enc_text = encryptor.update(padded_text) + encryptor.finalize()

    print('шифрованый текст')
    print(enc_text)

    # запись зашифрованного текста в файл
    file_name = 'encrypted_file.txt'
    with open(file_name, mode='wb') as f:
        f.write(enc_text)
    settings['encrypted_file'] = file_name


def hybrid_data_decryption(settings: dict) -> None:

    # десериализация текста
    with open(settings['encrypted_file'], mode='rb') as dc_file:
        dc_file_content = dc_file.read()
    print('чтение зашифрованного текста из файла')
    print(dc_file_content)

    # десериализация ключа симметричного алгоритма
    with open(settings['symmetric_key'], mode='rb') as key_file:
        content = key_file.read()

    # десериализация закрытого ключа
    with open(settings['secret_key'], mode='rb') as pem_in:
        private_bytes = pem_in.read()
    d_private_key = load_pem_private_key(private_bytes, password=None, )

    from cryptography.hazmat.primitives.asymmetric import padding
    # дешифрование ключа симметричного шифрования асимметричным алгоритмом
    dc_symmetric_key = d_private_key.decrypt(content, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                                                   algorithm=hashes.SHA256(), label=None))
    print('расшифрованный ключ')
    print(dc_symmetric_key)

    # дешифрование и депаддинг текста симметричным алгоритмом
    from cryptography.hazmat.primitives import padding

    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(dc_symmetric_key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    dc_text = decryptor.update(dc_file_content) + decryptor.finalize()

    unpadder = padding.ANSIX923(128).unpadder()
    unpadded_dc_text = unpadder.update(dc_text) + unpadder.finalize()

    print('расшифрованный текст')
    print(dc_text)
    print(dc_text.decode('utf-8'))
    print('расшифрованный текст')
    print(unpadded_dc_text)
    print(unpadded_dc_text.decode('utf-8'))


if __name__ == '__main__':
    settings = {
        'initial_file': 'Hello.txt',
        'encrypted_file': 'path/to/encrypted/file.txt',
        'decrypted_file': 'path/to/decrypted/file.txt',
        'symmetric_key': 'path/to/symmetric/key.txt',
        'public_key': 'path/to/public/key.pem',
        'secret_key': 'path/to/secret/key.pem',
    }

    hybrid_key_generation(settings)
    hybrid_data_encryption(settings)
    hybrid_data_decryption(settings)

    # пишем в файл
    with open('settings.json', 'w') as fp:
        json.dump(settings, fp)
    # читаем из файла
    with open('settings.json') as json_file:
        json_data = json.load(json_file)
    print(json_data)
