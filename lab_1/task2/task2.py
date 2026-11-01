import importlib.util

FREQ_FILE = "task2/frequency_of_letters_task_2.txt"
SOURCE_FILE = "task2/cod3.txt"
KEY_FILE = "task2/task_2_key.py"
RESULT_FILE = "task2/cod3_result.txt"

def get_freq_file(text: str, filename: str = FREQ_FILE) -> None:
    freq = {}
    total_chars = len(text)
    
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Всего символов: {total_chars}\n\n")
        for char, count in sorted_freq:
            percentage = (count / total_chars) * 100
            f.write(f"{repr(char)}: {count} ({percentage:.2f}%)\n")


def load_key_from_python(filepath: str) -> dict:
    spec = importlib.util.spec_from_file_location("key_module", filepath)
    key_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(key_module)
    return key_module.KEY


def decode_text(text: str, key: dict[str, str], filename: str = RESULT_FILE) -> None:
    replace_table = load_key_from_python(KEY_FILE)
    decoded = "".join(key.get(char, char) for char in text)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(decoded)
    print(f"Расшифровано {len(decoded)} символов")


def main() -> None:
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Загружено {len(text)} символов из {SOURCE_FILE}")
    
    get_freq_file(text)
    print(f"Частотный анализ сохранен в {FREQ_FILE}")
    
    decode_text(text)
    print(f"Результат сохранен в {RESULT_FILE}")


if __name__ == "__main__":
    main()