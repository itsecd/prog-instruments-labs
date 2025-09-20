import os
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import load_pem_private_key


def read_file(file_path):
    """
    Читает содержимое файла в бинарном режиме с обработкой ошибок.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    try:
        with open(file_path, 'rb') as f:
            return f.read()
    except IOError as e:
        print(f"Ошибка при чтении файла {file_path}: {str(e)}")
        raise


def write_file(file_path, data):
    """
    Записывает данные в файл в бинарном режиме с обработкой ошибок, используя match/case.
    """
    output_dir = os.path.dirname(file_path)
    try:
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(file_path, 'wb') as f:
            f.write(data)
    except OSError as e:
        match str(e).lower().find("mkdir") >= 0:
            case True:
                print(f"Ошибка при создании директории {output_dir}: {str(e)}")
            case False:
                print(f"Ошибка при записи файла {file_path}: {str(e)}")
        raise
    except IOError as e:
        print(f"Ошибка при записи файла {file_path}: {str(e)}")
        raise
    except Exception as e:
        print(f"Неизвестная ошибка при записи файла {file_path}: {str(e)}")
        raise

def generate_asymmetric_keys():
    """
    Генерирует пару ключей RSA.
    """
    print("Генерация пары ключей RSA...")
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return private_key, public_key


def encrypt_symmetric_key(sym_key, public_key, path):
    """
    Шифрует симметричный ключ с помощью RSA-OAEP.
    """
    print("Шифрование симметричного ключа...")
    encrypted_key = public_key.encrypt(
        sym_key,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    write_file(path, encrypted_key)
    print(f"Зашифрованный симметричный ключ сохранен в {path}")


def decrypt_symmetric_key(encrypted_key_path, private_key_path):
    """
    Расшифровывает симметричный ключ.
    """
    print("Расшифровка симметричного ключа...")
    encrypted_key = read_file(encrypted_key_path)
    private_key_data = read_file(private_key_path)
    private_key = load_pem_private_key(private_key_data, password=None)

    sym_key = private_key.decrypt(
        encrypted_key,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return sym_key
