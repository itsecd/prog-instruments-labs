from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import logging

logger = logging.getLogger(__name__)

def generate_rsa_keys():
    """
    Создает пару RSA ключей (2048 бит).
    :return: Приватный и публичный ключи
    """
    logger.debug("Генерация RSA ключей")
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    logger.debug("RSA ключи успешно сгенерированы")
    return private_key, private_key.public_key()

def save_key_to_pem(key, key_path: str, is_private: bool) -> None:
    """
    Сохраняет ключ в PEM формате.
    :param key: Ключ для сохранения
    :param key_path: Путь для сохранения
    :param is_private: Флаг типа ключа
    """
    try:
        key_type = "приватный" if is_private else "публичный"
        logger.debug(f"Сохранение {key_type} ключа в {key_path}")
        with open(key_path, 'wb') as f:
            if is_private:
                f.write(key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            else:
                f.write(key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
        logger.debug(f"{key_type.capitalize()} ключ успешно сохранен")
    except Exception as e:
        logger.error(f"Ошибка сохранения ключа: {e}")
        print(f"Ошибка сохранения ключа: {e}")
        raise

def load_key_from_pem(key_path: str, is_private: bool):
    """
    Загружает ключ из PEM файла.
    :param key_path: Путь к файлу
    :param is_private: Флаг типа ключа
    :return: Загруженный ключ
    """
    try:
        key_type = "приватный" if is_private else "публичный"
        logger.debug(f"Загрузка {key_type} ключа из {key_path}")
        with open(key_path, 'rb') as f:
            data = f.read()
            if is_private:
                key = serialization.load_pem_private_key(data, password=None)
            else:
                key = serialization.load_pem_public_key(data)
        logger.debug(f"{key_type.capitalize()} ключ успешно загружен")
        return key
    except Exception as e:
        logger.error(f"Ошибка загрузки ключа: {e}")
        print(f"Ошибка загрузки ключа: {e}")
        raise

def rsa_encrypt(public_key, data: bytes) -> bytes:
    """
    Шифрует данные публичным ключом RSA.
    :param public_key: Публичный ключ
    :param data: Данные для шифрования
    :return: Зашифрованные данные
    """
    logger.debug(f"RSA шифрование, размер данных: {len(data)} байт")
    result = public_key.encrypt(
        data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    logger.debug(f"RSA шифрование завершено, размер результата: {len(result)} байт")
    return result

def rsa_decrypt(private_key, encrypted_data: bytes) -> bytes:
    """
    Расшифровывает данные приватным ключом RSA.
    :param private_key: Приватный ключ
    :param encrypted_data: Зашифрованные данные
    :return: Расшифрованные данные
    """
    logger.debug(f"RSA дешифрование, размер данных: {len(encrypted_data)} байт")
    result = private_key.decrypt(
        encrypted_data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    logger.debug(f"RSA дешифрование завершено, размер результата: {len(result)} байт")
    return result