import argparse


def intput_file(filename: str) -> str:
    """Функция для считывания файла
    На вход принимается имя необходимого файла
    Если файл не найден будет выброшено исключение
    """
    try:
        with open(filename, encoding="utf-8") as file:
            print(f"File {filename} ready to work")
            return file.read()
    except FileNotFoundError:
        print("Sorry, this file impossible to detect")
        return ""


def output_file(filename: str, text: str) -> None:
    """Функция для переноса данных в необходимый файл
    На вход принимается имя файла и данные (ожидается строка)
    """
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)
        file.write("\n")


def parse_alfa(text: str) -> dict[str, float]:
    """Функция для парсинга стандартного алфавита
    с коэффициентами встречаемости.
    """
    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        left, _, right = line.partition("=")
        left = left.strip()
        try:
            value = float(right.strip())
        except ValueError:
            continue
        key = " " if left == "" else left
        result[key] = value
    return result


def freq_liter(start_string: str, total: int) -> dict[str, float]:
    """Функция для подсчета доли встречаемости букв
    в исходном тексте
    """
    freq_abs: dict[str, int] = {}
    for ch in start_string:
        freq_abs[ch] = freq_abs.get(ch, 0) + 1
    freq_percent = {ch: count / total for ch, count in freq_abs.items()}
    return freq_percent


def normalize_list(alfa_text: str) -> dict[str, float]:
    """Функция для нормализации словаря с пробелом как символом."""
    return parse_alfa(alfa_text)


def frequency_analysis_dict(
    sorted_text: list[tuple[str, float]], sorted_ref: list[tuple[str, float]]
) -> None:
    """Функция для создания и вывода результатов сравнения
    частотного анализа
    """
    dictionary = []
    for i in range(min(len(sorted_text), len(sorted_ref))):
        text_char, text_freq = sorted_text[i]
        ref_letter, ref_freq_val = sorted_ref[i]
        dictionary.append([ref_letter, text_char])
    print("Словарь (буква -> символ):")
    print(dictionary, "\n")


def decrypt_with_key(key_content: list[str], start_string: str) -> str:
    """Функция для расшифровки текста при помощи ключа"""
    replace_dict = {}
    for line in key_content:
        if ":" in line:
            parts = line.split(":", 1)
            src_char = parts[0].strip()
            dst_char = parts[1]
            if src_char:
                replace_dict[src_char] = dst_char
    trans_table = str.maketrans(replace_dict)
    decrypted_text = start_string.translate(trans_table)
    return decrypted_text


def parse_command_line():
    """Разбор аргументов командной строки
    с помощью argparse."""
    parser = argparse.ArgumentParser(
        description="Частотный анализ и расшифровка текста по ключу замены."
    )
    parser.add_argument("input_text_file", help="Файл с исходным зашифрованным текстом")
    parser.add_argument("alfa_file", help="Файл с эталонными частотами букв (alfa.txt)")
    parser.add_argument("key_file", help="Файл с ключом замены (формат: буква:замена)")
    parser.add_argument("output_file", help="Файл для записи расшифрованного текста")
    return parser.parse_args()


def main():
    """Функция для вызова всех функций и совокупности действий"""
    args = parse_command_line()

    start_string = intput_file(args.input_text_file)
    alfa_text = intput_file(args.alfa_file)
    total = len(start_string)
    freq_percent = freq_liter(start_string, total)
    ref_freq = normalize_list(alfa_text)

    sorted_ref = sorted(ref_freq.items(), key=lambda x: x[1], reverse=True)
    sorted_text = sorted(freq_percent.items(), key=lambda x: x[1], reverse=True)
    frequency_analysis_dict(sorted_text, sorted_ref)

    key_content = intput_file(args.key_file).strip().splitlines()
    decrypted_text = decrypt_with_key(key_content, start_string)

    print("Результат расшифровки:")
    print(decrypted_text)
    output_file(args.output_file, decrypted_text)


if __name__ == "__main__":
    main()
