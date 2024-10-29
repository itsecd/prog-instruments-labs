import argparse
import json
import os
from const import WAY, SIZE
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import (
    load_pem_public_key,
    load_pem_private_key,
)
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def create_sym_key() -> bytes:
    return os.urandom(16)


def serialize_sym(key: bytes, path: str) -> None:
    try:
        with open(path, "wb") as file:
            file.write(key)
    except Exception as e:
        print("Возникла ошибка при сиреализации симетричного ключа: ", e)


def deserialize_sym(path: str) -> bytes:
    try:
        with open(path, "rb") as file:
            key = file.read()
        return key
    except Exception as e:
        print("Возникла ошибка при десиреализации симетричного ключа: ", e)


def encrypt_text(text: str, key: bytes) -> bytes:
    padder = padding.PKCS7(64).padder()
    bi_text = bytes(text, "UTF-8")
    iv = os.urandom(8)
    cipher = Cipher(algorithms.IDEA(key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    padded_text = padder.update(bi_text) + padder.finalize()
    c_text = iv + encryptor.update(padded_text) + encryptor.finalize()
    return c_text


def decode_text(c_text: bytes, key: bytes) -> str:
    iv = c_text[:8]
    cipher_text = c_text[8:]
    cipher = Cipher(algorithms.IDEA(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    dc_text = decryptor.update(cipher_text) + decryptor.finalize()
    unpadder = padding.PKCS7(64).unpadder()
    unpadded_dc_text = unpadder.update(dc_text) + unpadder.finalize()
    return unpadded_dc_text.decode("UTF-8")


def create_asym_key() -> dict:
    keys = rsa.generate_private_key(public_exponent=65537, key_size=1024)
    asym = {"private": keys, "public": keys.public_key()}
    return asym


def serialize_private(key: dict, path: str) -> None:
    try:
        with open(path, "wb") as file:
            file.write(
                key["private"].private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )
    except Exception as e:
        print("Возникла ошибка при сиреализации публичного ключа: ", e)


def serialize_public(key: dict, path: str) -> None:
    try:
        with open(path, "wb") as file:
            file.write(
                key["public"].public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
    except Exception as e:
        print("Возникла ошибка при сиреализации приватного ключа: ", e)


def encrypt_sym_key(asym_key: dict, sym_key: bytes) -> bytes:
    key = asym_key["public"].encrypt(
        sym_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return key


def decrypt_sym_key(asym_key: dict, key: bytes) -> bytes:
    de_key = asym_key["private"].decrypt(
        key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return de_key


def deserylie_asym(public_path: str, private_path: str) -> dict:
    with open(public_path, "rb") as pem_in:
        public_bytes = pem_in.read()
    d_public_key = load_pem_public_key(public_bytes)
    with open(private_path, "rb") as pem_in:
        private_bytes = pem_in.read()
    d_private_key = load_pem_private_key(
        private_bytes,
        password=None,
    )
    asym = {"private": d_private_key, "public": d_public_key}
    return asym

def generation_proc(private_way: str, public_way: str, symm_way: str):
    print("Запуск процедкры генерации и сериализации ключей")
    sym_key = create_sym_key()
    print("Произошло создание симметричного ключа")
    asym_key = create_asym_key()
    print("Произогло создание асимметричного ключа")
    cyph_sym_key = encrypt_sym_key(asym_key, sym_key)
    print("Произошла шифровка симметричного ключа")
    serialize_private(asym_key, private_way)
    print("Произошла сериализация приватного ключа")
    serialize_public(asym_key, public_way)
    print("Произошла сериализация пкбличчного ключа")
    serialize_sym(cyph_sym_key, symm_way)
    print("Произошла сериализация симметричного ключа")


def encryption_proc(
    encr_way: str, orig_way: str, private_way: str, public_way: str, symm_way: str
):
    print("Запуск процедуры шифровки текста")
    asym_key = deserylie_asym(public_way, private_way)
    print("Произошла десериализация асимметричного ключа")
    sym_key = deserialize_sym(symm_way)
    print("Произошла десериализация симметричного ключа")
    sym_key = decrypt_sym_key(asym_key, sym_key)
    print("Произошло дешифрование симметричного ключа")
    try:
        with open(orig_way, "r", encoding="utf-8") as file:
            text = file.read()
        print("Произошло чтение оригинального текста")
    except Exception as e:
        print("Возникла ошибка при чтении файла с текста: ", e)
    enc_text = encrypt_text(text, sym_key)
    print("Произошло шифрование оригинального текста")
    try:
        with open(encr_way, "wb") as file:
            file.write(enc_text)
        print("Произошла запись зашифрованного текста")
    except Exception as e:
        print("Возникла ошибка при чтении файла с текста: ", e)


def decryption_proc(
    uncyph_way: str, encr_way: str, private_way: str, public_way: str, symm_way: str
):
    print("Запуск процедуры дешифровки текста")
    asym_key = deserylie_asym(public_way, private_way)
    print("Произошла десериализация асимметричного ключа")
    sym_key = deserialize_sym(symm_way)
    print("Произошла десериализация симметричного ключа")
    sym_key = decrypt_sym_key(asym_key, sym_key)
    print("Произошло дешифрование симметричного ключа")
    try:
        with open(encr_way, "rb") as file:
            text = file.read()
        print("Произошло чтение зашифрованного текста")
    except Exception as e:
        print("Возникла ошибка при чтении файла с текста: ", e)
    text = decode_text(text, sym_key)
    print("Произошла дешифровка текста")
    try:
        with open(uncyph_way, "w", encoding="utf-8") as file:
            file.write(text)
        print("Произошла запись расшифрованного текста")
    except Exception as e:
        print("Возникла ошибка записи расшифрованного текста: ", e)


if __name__ == "__main__":

    print("start program")
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-gen", "--generation", help="Запускает режим генерации ключей")
    group.add_argument("-enc", "--encryption", help="Запускает режим шифрования")
    group.add_argument("-dec", "--decryption", help="Запускает режим дешифрования")
    args = parser.parse_args()

    try:
        with open(WAY, "r", encoding="utf-8") as file:
            ways = json.load(file)
    except Exception as e:
        print("Возникла ошибка открытия файла с путями")

    match args:
        case args if args.generation:
            generation_proc(ways[2], ways[1], ways[0])
        case args if  args.encryption:
            encryption_proc(ways[4], ways[3], ways[2], ways[1], ways[0])
        case args if args.decryption:
            decryption_proc(ways[5], ways[4], ways[2], ways[1], ways[0])


        
        
