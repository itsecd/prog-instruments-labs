from settings import RUSFREQ, DECTEXT, ENCTEXT2, GENKEY, FINALDICT
from collections import Counter

def read(filename: str) -> str:
    """
        Читает содержимое текстового файла.

        :param filename: Путь к файлу, который нужно прочитать.
        :return: Строка с содержимым файла или сообщение об ошибке.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Ошибка: {e}"

def save(filename: str, text: str) -> None:
    """
        Сохраняет текст в файл.

        :param filename: Путь к файлу, в который будет записан текст.
        :param text: Строка, которая будет записана в файл.
        :return: None
    """
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(text)
    except Exception as e:
        print(f"Ошибка: {e}")

def frequency_index(text: str) -> dict:
    """
        Вычисляет частотный индекс символов в тексте.

        :param text: Входной текст для анализа.
        :return: Словарь, где ключ — символ, значение — его частота в тексте.
        """
    try:
        total_chars = len(text)
        if total_chars == 0:
            return {}

        frequencies = Counter(text)
        return {char: round(freq / total_chars, 5) for char, freq in frequencies.items()}
    except Exception as e:
        print(f"Ошибка при вычислении частотного индекса: {e}")
        return {}

def replace(text, replacements):
    """
       Выполняет замену символов в тексте согласно переданному словарю замен.

       :param text: Исходный текст.
       :param replacements: Словарь замен, где ключ — символ в зашифрованном тексте,
                            а значение — предполагаемый оригинальный символ.
       :return: Текст после замены символов.
   """
    try:
        inverted_replacements = {v: k for k, v in replacements.items()}
        return ''.join(inverted_replacements.get(char, char) for char in text)
    except Exception as e:
        print(f"Ошибка: {e}")
        return

def replacementdict(freq_dict1: dict, freq_dict2: dict) -> dict:
    """
        Создаёт словарь замен символов на основе частотного анализа двух текстов.

        :param freq_dict1: Частотный словарь языка.
        :param freq_dict2: Частотный словарь зашифрованного текста.
        :return: Словарь замен, в котором символы зашифрованного текста
                 сопоставляются с наиболее вероятными оригинальными символами.
    """
    try:
        sorted_chars1 = sorted(freq_dict1.keys(), key=lambda c: freq_dict1[c], reverse=True)
        sorted_chars2 = sorted(freq_dict2.keys(), key=lambda c: freq_dict2[c], reverse=True)

        replacement_map = {c1: c2 for c1, c2 in zip(sorted_chars1, sorted_chars2)}

        return replacement_map
    except Exception as e:
        print(f"Ошибка при создании словаря замен: {e}")
        return {}

def main() -> None:
    """
        Главная функция программы.

        Выполняет частотный анализ зашифрованного текста, создаёт словарь замен,
        выполняет замену символов и дешифрует текст. Затем сохраняет результат
        и ключ расшифровки в файлы.

        :return: None
    """
    try:
        print("Зашифрованный текст:\n")
        text = read(ENCTEXT2).upper()
        if not text:
            print("Ошибка: текст не загружен.")
            return
        print(text)
        freqdict = frequency_index(text)
        if not freqdict:
            print("\nОшибка: не удалось вычислить частотный индекс.")
            return
        print("\nИндекс частоты появления букв в зашифрованном тексте:\n")
        print(dict(sorted(freqdict.items(), key=lambda item: item[1], reverse=True)))
        replacedict = replacementdict(RUSFREQ, freqdict)
        print("\nСловарь замен первая стадия\n")
        print(replacedict)
        dectext = replace(text, replacedict)
        print("\nТекст после первой стадии дешифрования\n")
        print(dectext)
        dectext = replace(text, FINALDICT)
        print("\nДешифрованный текст\n")
        print(dectext)
        key = {v: k for k, v in FINALDICT.items()}
        print("\nКлюч:\n")
        if key:
            print(key)
        save(DECTEXT, str(dectext))
        save(GENKEY,str(key))

    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()
