import json
from collections import Counter


def load_text(filename):
    """Загружает текст из файла."""
    try:
        with open(filename, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Ошибка: файл {filename} не найден")
        raise


def print_stats(text):
    """Выводит частотную статистику символов."""
    total = len(text)
    chars = Counter(text)

    print("Статистика символов\n")
    print(f"Всего: {total}, Уникальных: {len(chars)}\n")
    print("Символ           | Кол-во | Частота")

    special = {"\n": "ПЕРЕВОД СТРОКИ", " ": "ПРОБЕЛ", "\t": "ТАБУЛЯЦИЯ"}

    for c, cnt in sorted(chars.items(), key=lambda x: -x[1]):
        name = special.get(c, c if c.isprintable() else f"\\x{ord(c):02x}")
        print(f"{name:<16} | {cnt:6} | {cnt / total:.4f}")


def replace_interactive(text):
    """Интерактивно заменяет символы в тексте."""
    reps = {}
    while True:
        print(f"\nТекст:\n\n{text[:200]}\n")
        if reps:
            print("Замены:", reps)

        old = input("\nСимвол для замены (Enter - выход): ")
        if not old:
            break
        if len(old) != 1:
            continue

        new = input(f"Заменить '{old}' на: ")
        if len(new) != 1:
            continue

        reps[old] = new
        text = text.replace(old, new)

    return reps, text


def save_results(reps, text):
    """Сохраняет результат и ключ замен в файлы."""
    if not reps:
        return

    with open("decrypted.txt", "w", encoding="utf-8") as f:
        f.write(text)
    with open("key.json", "w", encoding="utf-8") as f:
        json.dump(reps, f, ensure_ascii=False, indent=2)

    print("\nСохранено: decrypted.txt и key.json")


def main():
    """Основная функция программы."""
    text = load_text("original.txt")
    print_stats(text)
    print("\n\nЗамена Символов")
    reps, new_text = replace_interactive(text)
    save_results(reps, new_text)


if __name__ == "__main__":
    main()
