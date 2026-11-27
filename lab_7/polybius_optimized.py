import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

POLYBIUS = {
    'А': '11', 'Б': '12', 'В': '13', 'Г': '14', 'Д': '15', 'Е': '16',
    'Ж': '21', 'З': '22', 'И': '23', 'Й': '24', 'К': '25', 'Л': '26',
    'М': '31', 'Н': '32', 'О': '33', 'П': '34', 'Р': '35', 'С': '36',
    'Т': '41', 'У': '42', 'Ф': '43', 'Х': '44', 'Ц': '45', 'Ч': '46',
    'Ш': '51', 'Щ': '52', 'Ъ': '53', 'Ы': '54', 'Ь': '55', 'Э': '56',
    'Ю': '61', 'Я': '62', ' ': '63',
}

REVERSE = {v: k for k, v in POLYBIUS.items()}


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


def encrypt(text):
    """Оптимизированное шифрование: используем list и join."""
    logger.info("Начало шифрования...")
    encrypted_parts = []
    for char in text.upper():
        if char in POLYBIUS:
            encrypted_parts.append(POLYBIUS[char])
        else:
            encrypted_parts.append(char)
    encrypted = ''.join(encrypted_parts)
    logger.info(f"Шифрование завершено. Длина: {len(encrypted)}")
    return encrypted


def decrypt(encrypted):
    """Оптимизированное дешифрование: без срезов строк и с кэшированием len."""
    logger.info("Начало дешифрования...")
    decrypted_parts = []
    i = 0
    length = len(encrypted)
    while i < length:
        if i + 1 < length:
            pair = encrypted[i] + encrypted[i+1]  # без среза
            if pair in REVERSE:
                decrypted_parts.append(REVERSE[pair])
                i += 2
                continue
        decrypted_parts.append(encrypted[i])
        i += 1
    decrypted = ''.join(decrypted_parts)
    logger.info(f"Дешифрование завершено. Длина: {len(decrypted)}")
    return decrypted

def main():
    logger.info("Запуск программы...")
    text = read_file("input_large.txt")
    logger.info(f"Прочитано символов: {len(text)}")

    encrypted = encrypt(text)
    decrypted = decrypt(encrypted)

    if decrypted == text.upper():
        logger.info("УСПЕХ: текст совпадает!")
    else:
        logger.error("ОШИБКА: текст не совпадает!")


if __name__ == "__main__":
    main()
