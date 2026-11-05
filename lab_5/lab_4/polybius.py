import logging
import sys

def setup_logger():
    logger = logging.getLogger("PolybiusLogger")
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler('polybius.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()

def save(filename, content):
    logger.debug(f"Попытка сохранить данные в файл: {filename}")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f"Файл {filename} успешно сохранен. Размер: {len(content)} символов.")

def read(filename):
    logger.debug(f"Попытка прочитать файл: {filename}")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Файл {filename} прочитан. Длина текста: {len(content)} символов.")
        return content
    except FileNotFoundError:
        logger.error(f"Критическая ошибка: файл {filename} не найден!")
        raise

POLYBIUS = {
    'А': '11', 'Б': '12', 'В': '13', 'Г': '14', 'Д': '15', 'Е': '16',
    'Ж': '21', 'З': '22', 'И': '23', 'Й': '24', 'К': '25', 'Л': '26',
    'М': '31', 'Н': '32', 'О': '33', 'П': '34', 'Р': '35', 'С': '36',
    'Т': '41', 'У': '42', 'Ф': '43', 'Х': '44', 'Ц': '45', 'Ч': '46',
    'Ш': '51', 'Щ': '52', 'Ъ': '53', 'Ы': '54', 'Ь': '55', 'Э': '56',
    'Ю': '61', 'Я': '62', ' ': '63',
}

REVERSE = {v: k for k, v in POLYBIUS.items()}

def encrypt(text):
    logger.info("Начало процесса шифрования.")
    encrypted = ''
    for char in text.upper():
        if char in POLYBIUS:
            encrypted += POLYBIUS[char]
        else:
            logger.warning(f"Символ '{char}' отсутствует в квадрате Полибия. Оставлен без изменений.")
            encrypted += char
    logger.info(f"Шифрование завершено. Результат: {len(encrypted)} символов.")
    return encrypted

def decrypt(encrypted):
    logger.info("Начало процесса дешифрования.")
    decrypted = ''
    i = 0
    while i < len(encrypted):
        if i + 1 < len(encrypted) and encrypted[i:i + 2] in REVERSE:
            decrypted += REVERSE[encrypted[i:i + 2]]
            i += 2
        else:
            decrypted += encrypted[i]
            i += 1
    logger.info(f"Дешифрование завершено. Результат: {len(decrypted)} символов.")
    return decrypted

if __name__ == "__main__":
    logger.info("Запуск программы шифра Полибия.")
    
    try:
        text = read('input.txt')
    except FileNotFoundError:
        logger.critical("Программа завершена, так как входной файл не найден.")
        sys.exit(1)

    encrypted = encrypt(text)
    print(f"Зашифрованный текст: {encrypted}")

    decrypted = decrypt(encrypted)
    print(f"Расшифрованный текст: {decrypted}")

    if decrypted == text.upper():
        logger.info("Проверка пройдена: исходный и расшифрованный текст совпадают.")
    else:
        logger.error("Ошибка проверки: текст не совпадает после дешифровки!")

    save('encrypted_polybius.txt', encrypted)
    save('decrypted_polybius.txt', decrypted)
    
    logger.info("Программа успешно завершена.")
