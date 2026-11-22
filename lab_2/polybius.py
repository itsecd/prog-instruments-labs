def save(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)


def read(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


POLYBIUS = {
    'А': '11', 'Б': '12', 'В': '13', 'Г': '14', 'Д': '15', 'Е': '16',
    'Ж': '21', 'З': '22', 'И': '23', 'Й': '24', 'К': '25', 'Л': '26',
    'М': '31', 'Н': '32', 'О': '33', 'П': '34', 'Р': '35', 'С': '36',
    'Т': '41', 'У': '42', 'Ф': '43', 'Х': '44', 'Ц': '45', 'Ч': '46',
    'Ш': '51', 'Щ': '52', 'Ъ': '53', 'Ы': '54', 'Ь': '55', 'Э': '56',
    'Ю': '61', 'Я': '62', ' ': '63',
}

REVERSE = {v: k for k, v in POLYBIUS.items()}

text = read('input.txt')
print(f"Исходный текст: {len(text)} символов")

encrypted = ''
for char in text.upper():
    if char in POLYBIUS:
        encrypted += POLYBIUS[char]
    else:
        encrypted += char

print(f"Зашифрованный: {len(encrypted)} символов")

decrypted = ''
i = 0
while i < len(encrypted):
    if i + 1 < len(encrypted) and encrypted[i:i + 2] in REVERSE:
        decrypted += REVERSE[encrypted[i:i + 2]]
        i += 2
    else:
        decrypted += encrypted[i]
        i += 1

print(f"Расшифрованный: {len(decrypted)} символов")

if decrypted == text.upper():
    print("✓ УСПЕХ! Всё работает")
else:
    print("✗ Ошибка")

save('encrypted_polybius.txt', encrypted)
save('decrypted_polybius.txt', decrypted)

with open('key_polybius.txt', 'w', encoding='utf-8') as f:
    f.write("КВАДРАТ ПОЛИБИЯ 6x6:\n")
    f.write("   1 2 3 4 5 6\n")
    f.write("1  А Б В Г Д Е\n")
    f.write("2  Ж З И Й К Л\n")
    f.write("3  М Н О П Р С\n")
    f.write("4  Т У Ф Х Ц Ч\n")
    f.write("5  Ш Щ Ъ Ы Ь Э\n")
    f.write("6  Ю Я   (пробел)\n")

print("\nФайлы сохранены:")
print("- encrypted_polybius.txt")
print("- decrypted_polybius.txt")
print("- key_polybius.txt")
