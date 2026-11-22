import os
from cryptography.hazmat.primitives import hashes, padding as sym_padding, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def read_bytes(path):
    with open(path, "rb") as file:
        return file.read()


def write_bytes(path, data):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "wb") as file:
        file.write(data)


def generate_symmetric_key(key_size):
    if key_size not in (128, 192, 256):
        raise ValueError("Размер ключа AES должен быть 128, 192 или 256 бит")
    return os.urandom(key_size // 8)


def generate_rsa_keys():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return private_key, public_key


def save_private_key(private_key, path):
    data = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    )
    write_bytes(path, data)


def save_public_key(public_key, path):
    data = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    write_bytes(path, data)


def load_private_key(path):
    return serialization.load_pem_private_key(read_bytes(path), password=None)


def load_public_key(path):
    return serialization.load_pem_public_key(read_bytes(path))


def encrypt_symmetric_key(aes_key, public_key):
    return public_key.encrypt(
        aes_key,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )


def decrypt_symmetric_key(encrypted_key, private_key):
    return private_key.decrypt(
        encrypted_key,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )


def generate_keys(encrypted_key_path, public_key_path, private_key_path, aes_key_size):
    aes_key = generate_symmetric_key(aes_key_size)
    private_key, public_key = generate_rsa_keys()

    save_private_key(private_key, private_key_path)
    save_public_key(public_key, public_key_path)

    encrypted_key = encrypt_symmetric_key(aes_key, public_key)
    write_bytes(encrypted_key_path, encrypted_key)


def encrypt_file(input_path, public_key_path, encrypted_key_path, output_path, aes_key_size=256):
    public_key = load_public_key(public_key_path)
    aes_key = generate_symmetric_key(aes_key_size)

    encrypted_key = encrypt_symmetric_key(aes_key, public_key)
    write_bytes(encrypted_key_path, encrypted_key)

    data = read_bytes(input_path)
    iv = os.urandom(16)

    padder = sym_padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()

    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    write_bytes(output_path, iv + encrypted_data)


def decrypt_file(input_path, private_key_path, encrypted_key_path, output_path):
    private_key = load_private_key(private_key_path)
    encrypted_key = read_bytes(encrypted_key_path)
    aes_key = decrypt_symmetric_key(encrypted_key, private_key)

    data = read_bytes(input_path)
    iv = data[:16]
    encrypted_data = data[16:]

    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

    unpadder = sym_padding.PKCS7(128).unpadder()
    decrypted_data = unpadder.update(padded_data) + unpadder.finalize()

    write_bytes(output_path, decrypted_data)
