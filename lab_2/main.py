import os
import json
import argparse
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import load_pem_public_key, load_pem_private_key
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


with open('settings.json') as _file:
    json_data = json.load(_file)

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-gen', '--generation', help='Запускает режим генерации ключей')
group.add_argument('-enc', '--encryption', help='Запускает режим шифрования')
group.add_argument('-dec', '--decryption', help='Запускает режим дешифрования')
args = parser.parse_args()

if args.generation is not None:
    print('Симметричный ключ: ')
    key = os.urandom(16)
    print(key)

    keys = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    private_key = keys
    public_key = keys.public_key()
    print('Закрытый ключ: ')
    print(private_key)
    print('Открытый ключ: ')
    print(public_key)

    with open(json_data['public_key'], 'wb') as public_file:
        public_file.write(public_key.public_bytes(encoding=serialization.Encoding.PEM,
                                                  format=serialization.PublicFormat.SubjectPublicKeyInfo))

    with open(json_data['private_key'], 'wb') as private_file:
        private_file.write(private_key.private_bytes(encoding=serialization.Encoding.PEM,
                                                     format=serialization.PrivateFormat.TraditionalOpenSSL,
                                                     encryption_algorithm=serialization.NoEncryption()))

    print('Открытый и закрытый ключ были сериализованы в файлы secret.pem и public.pem')


    with open(json_data['symmetric_key'], "wb") as _file:
        _file.write(public_key.encrypt(key,
    padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
    algorithm=hashes.SHA256(), label=None)))
    print('Зашифрованно и записанно в файл symmetric.txt')


elif args.encryption is not None:
    print('Второе задание: ')
    with open(json_data['public_key'], 'rb') as pem_in:
        public_bytes = pem_in.read()
    d_public_key = load_pem_public_key(public_bytes)
    with open(json_data['private_key'], 'rb') as pem_in:
        private_bytes = pem_in.read()
    d_private_key = load_pem_private_key(private_bytes, password=None, )
    with open(json_data['symmetric_key'], 'rb') as symmetric_file:
        d_symmetric_key = symmetric_file.read()
    print('Десериализованны: Открытый, закрытый, симметричные ключи ')

    dcs_key = d_private_key.decrypt(d_symmetric_key,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None))
    print(dcs_key)


    with open(json_data['initial_file'], 'r', encoding='utf-8') as _text:
        text = _text.read()
    text = bytes(text, 'UTF-8')

    from cryptography.hazmat.primitives import padding

    padder=padding.ANSIX923(1024).padder()
    padded_text = padder.update(text) + padder.finalize()


    iv = os.urandom(16)

    with open(json_data['random_value'], 'wb') as _value:
        _value.write(iv)

    cipher = Cipher(algorithms.SEED(dcs_key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    cs_text = encryptor.update(padded_text) + encryptor.finalize()

    with open(json_data['encrypted_file'], 'wb') as e_file:
        e_file.write(cs_text)
    print('Зашифровали и сохранили текст в encrypted_file.txt')

else:
    print('Tретье задание: ')
    with open(json_data['public_key'], 'rb') as pem_in:
        public_bytes = pem_in.read()
    d_public_key = load_pem_public_key(public_bytes)
    with open(json_data['private_key'], 'rb') as pem_in:
        private_bytes = pem_in.read()
    d_private_key = load_pem_private_key(private_bytes, password=None, )
    with open(json_data['symmetric_key'], 'rb') as symmetric_file:
        cim_symmetric_key = symmetric_file.read()

    print("выполнена десериализация ключей")

    dcs_key = d_private_key.decrypt(cim_symmetric_key, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                                                           algorithm=hashes.SHA256(), label=None))
    print("Выполнена дешифровка")
    print(dcs_key)

    with open(json_data['encrypted_file'], 'rb') as en_file:
        c_text = en_file.read()
    with open(json_data['random_value'], 'rb') as file_value:
        iv = file_value.read()
    print("Выполнена десериализация")

    from cryptography.hazmat.primitives import padding

    cipher = Cipher(algorithms.SEED(dcs_key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    dc_text = decryptor.update(c_text) + decryptor.finalize()
    unpadder = padding.ANSIX923(1024).unpadder()
    unpadded_dc_text = unpadder.update(dc_text) + unpadder.finalize()
    decode_text = unpadded_dc_text.decode('UTF-8')
    with open(json_data['decrypted_file'], 'w') as dec_file:
        dec_file.write(decode_text)
    print('Текст расшифрован и сохранен в файл decrypted_file.txt')