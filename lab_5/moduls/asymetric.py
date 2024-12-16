from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import (
    load_pem_public_key,
    load_pem_private_key,
)


class Asymetric:

    def create_asym_key() -> dict:
        """Функция создаёт асиметричный ключ, не принимает значения и возвращает словарь с публичным и приватным ключами"""
        keys = rsa.generate_private_key(public_exponent=65537, key_size=1024)
        asym = {"private": keys, "public": keys.public_key()}
        return asym

    def encrypt_sym_key(asym_key: dict, sym_key: bytes) -> bytes:
        """Функция шифрует симетричный ключ, принимает асимметричный и симметричный, возвращает зашифрованный симетричный"""
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
        """Функция дешифрует симетричный ключ, принимает асимметричный и зашифрованный симметричный, возвращает симетричный"""
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
        """Производит десериализацию асимтричного ключа, принимает путь публичной и приватной частей возвращает словарь с асиметричным ключом"""
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
