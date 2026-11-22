class PolybiusCipher:
    POLYBIUS = {
        'А': '11', 'Б': '12', 'В': '13', 'Г': '14', 'Д': '15', 'Е': '16',
        'Ж': '21', 'З': '22', 'И': '23', 'Й': '24', 'К': '25', 'Л': '26',
        'М': '31', 'Н': '32', 'О': '33', 'П': '34', 'Р': '35', 'С': '36',
        'Т': '41', 'У': '42', 'Ф': '43', 'Х': '44', 'Ц': '45', 'Ч': '46',
        'Ш': '51', 'Щ': '52', 'Ъ': '53', 'Ы': '54', 'Ь': '55', 'Э': '56',
        'Ю': '61', 'Я': '62', ' ': '63',
    }
    
    REVERSE = {v: k for k, v in POLYBIUS.items()}

    def encrypt(self, text: str) -> str:
        encrypted = ''
        for char in text.upper():
            if char in self.POLYBIUS:
                encrypted += self.POLYBIUS[char]
            else:
                encrypted += char
        return encrypted

    def _decode_char(self, encrypted: str, index: int) -> tuple[str, int]:
        if index + 1 < len(encrypted):
            pair = encrypted[index:index + 2]
            if pair in self.REVERSE:
                return self.REVERSE[pair], index + 2
        return encrypted[index], index + 1

    def decrypt(self, encrypted: str) -> str:
        decrypted = ''
        i = 0
        while i < len(encrypted):
            char, i = self._decode_char(encrypted, i)
            decrypted += char
        return decrypted


def save(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

def read(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


if __name__ == "__main__":
    cipher = PolybiusCipher()
    
    text = read('input.txt')
    print(f"Исходный текст: {len(text)} символов")

    encrypted = cipher.encrypt(text)
    print(f"Зашифрованный: {len(encrypted)} символов")

    decrypted = cipher.decrypt(encrypted)
    print(f"Расшифрованный: {len(decrypted)} символов")

    if decrypted == text.upper():
        print("✓ УСПЕХ! Всё работает")
    else:
        print("✗ Ошибка")

    save('encrypted_polybius.txt', encrypted)
    save('decrypted_polybius.txt', decrypted)

    key_content = (
        "КВАДРАТ ПОЛИБИЯ 6x6:\n"
        "   1 2 3 4 5 6\n"
        "1  А Б В Г Д Е\n"
        "2  Ж З И Й К Л\n"
        "3  М Н О П Р С\n"
        "4  Т У Ф Х Ц Ч\n"
        "5  Ш Щ Ъ Ы Ь Э\n"
        "6  Ю Я   (пробел)\n"
    )
    save('key_polybius.txt', key_content)

    print("\nФайлы сохранены:")
    print("- encrypted_polybius.txt")
    print("- decrypted_polybius.txt")
    print("- key_polybius.txt")
