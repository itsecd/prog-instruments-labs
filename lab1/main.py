import datetime
import json
import math
import os
import re
import string
import sys
from collections import Counter

class TextFileAnalyzer:
    def __init__(self):
        self.analysis_history = []
        self.current_file_path = ""
    
    def read_file_content(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            print(f"Ошибка: Файл '{file_path}' не найден.")
            return None
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")
            return None

    def calculate_basic_stats(self, content):
        if not content:
            return None
        
        total_chars = len(content)
        lines = content.split('\n')
        total_lines = len(lines)
        words = content.split()
        total_words = len(words)
        
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        total_sentences = len(sentences)
        
        space_count = content.count(' ')
        punctuation_count = sum(1 for char in content if char in string.punctuation)
        
        print(
            f"Рассчитываем сложную статистику для текста длиной {total_chars} символов."
            "Это может занять некоторое время для больших файлов."
        )
        
        avg_word_length = sum(len(word) for word in words) / total_words if total_words > 0 else 0
        avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
        avg_line_length = sum(len(line) for line in lines) / total_lines if total_lines > 0 else 0
        
        unique_words = len(set(word.lower().strip(string.punctuation) for word in words))
        vocabulary_richness = (unique_words / total_words) * 100 if total_words > 0 else 0
        
        return {
            'total_chars': total_chars,
            'total_lines': total_lines,
            'total_words': total_words,
            'total_sentences': total_sentences,
            'space_count': space_count,
            'punctuation_count': punctuation_count,
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_line_length': round(avg_line_length, 2),
            'unique_words': unique_words,
            'vocabulary_richness': round(vocabulary_richness, 2)
        }

    def analyze_word_frequency(self, content, top_n=10):
        if not content:
            return None
        
        words = content.split()
        cleaned_words = []
        
        for word in words:
            clean_word = word.strip(string.punctuation).lower()
            if clean_word and clean_word not in ['', ' ']:
                cleaned_words.append(clean_word)
        
        word_freq = Counter(cleaned_words)
        return word_freq.most_common(top_n)

    def analyze_character_frequency(self, content, top_n=15):
        if not content:
            return None
        
        printable_chars = [char for char in content if char.isprintable() and char not in [' ', '\n', '\t']]
        char_freq = Counter(printable_chars)
        return char_freq.most_common(top_n)

    def calculate_readability_score(self, stats):
        if not stats or stats['total_words'] == 0:
            return 0
        
        avg_sentence_len = stats['avg_sentence_length']
        avg_word_len = stats['avg_word_length']
        
        readability = 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_word_len)
        return round(readability, 2)

    def generate_text_complexity_report(self, stats, word_freq, char_freq):
        report = "=" * 80 + "\n"
        report += "КОМПЛЕКСНЫЙ АНАЛИЗ ТЕКСТОВОГО ФАЙЛА\n"
        report += "=" * 80 + "\n\n"
        
        report += "ОСНОВНАЯ СТАТИСТИКА:\n"
        report += "-" * 40 + "\n"
        report += f"Общее количество символов: {stats['total_chars']:,}\n"
        report += f"Общее количество слов: {stats['total_words']:,}\n"
        report += f"Общее количество строк: {stats['total_lines']:,}\n"
        report += f"Общее количество предложений: {stats['total_sentences']:,}\n"
        report += f"Количество пробелов: {stats['space_count']:,}\n"
        report += f"Количество знаков препинания: {stats['punctuation_count']:,}\n\n"
        
        report += "СРЕДНИЕ ЗНАЧЕНИЯ:\n"
        report += "-" * 40 + "\n"
        report += f"Средняя длина слова: {stats['avg_word_length']} символов\n"
        report += f"Средняя длина предложения: {stats['avg_sentence_length']} слов\n"
        report += f"Средняя длина строки: {stats['avg_line_length']} символов\n\n"
        
        report += "АНАЛИЗ СЛОЖНОСТИ ТЕКСТА:\n"
        report += "-" * 40 + "\n"
        report += f"Уникальных слов: {stats['unique_words']:,}\n"
        report += f"Богатство словаря: {stats['vocabulary_richness']}%\n"
        
        readability = self.calculate_readability_score(stats)
        report += f"Оценка удобочитаемости: {readability}\n"
        
        if readability > 80:
            report += "Уровень сложности: Легкий (понятен широкой аудитории)\n"
        elif readability > 60:
            report += "Уровень сложности: Средний (требует базового образования)\n"
        elif readability > 40:
            report += "Уровень сложности: Сложный (требует хорошего образования)\n"
        else:
            report += "Уровень сложности: Очень сложный (требует специальных знаний)\n"
        
        report += "\n"
        
        report += "ТОП-10 САМЫХ ЧАСТЫХ СЛОВ:\n"
        report += "-" * 40 + "\n"
        for i, (word, count) in enumerate(word_freq, 1):
            percentage = (count / stats['total_words']) * 100
            report += f"{i:2d}. {word:<15} - {count:>6} раз ({percentage:5.2f}%)\n"
        
        report += "\n"
        
        report += "ТОП-15 САМЫХ ЧАСТЫХ СИМВОЛОВ:\n"
        report += "-" * 40 + "\n"
        for i, (char, count) in enumerate(char_freq, 1):
            percentage = (count / stats['total_chars']) * 100
            display_char = char if char.isprintable() else f"\\x{ord(char):02x}"
            report += f"{i:2d}. '{display_char}' - {count:>6} раз ({percentage:5.2f}%)\n"
        
        return report

    def save_analysis_to_file(self, report, filename_prefix):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_analysis_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(report)
            print(f"Отчет сохранен в файл: {filename}")
            return True
        except Exception as e:
            print(f"Ошибка при сохранении отчета: {e}")
            return False

    def save_history_to_json(self, filename="analysis_history.json"):
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(self.analysis_history, file, indent=2, ensure_ascii=False)
            print(f"История анализов сохранена в файл: {filename}")
            return True
        except Exception as e:
            print(f"Ошибка при сохранении истории: {e}")
            return False

    def load_history_from_json(self, filename="analysis_history.json"):
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as file:
                    self.analysis_history = json.load(file)
                print(f"История анализов загружена из файла: {filename}")
                return True
        except Exception as e:
            print(f"Ошибка при загрузке истории: {e}")
        return False

    def analyze_file(self, file_path):
        print(f"\nНачинаем анализ файла: {file_path}")
        
        content = self.read_file_content(file_path)
        if not content:
            return None
        
        basic_stats = self.calculate_basic_stats(content)
        if not basic_stats:
            return None
        
        word_frequency = self.analyze_word_frequency(content)
        char_frequency = self.analyze_character_frequency(content)
        
        report = self.generate_text_complexity_report(basic_stats, word_frequency, char_frequency)
        
        history_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'stats': basic_stats,
            'readability_score': self.calculate_readability_score(basic_stats)
        }
        self.analysis_history.append(history_entry)
        
        return report

    def show_history_summary(self):
        if not self.analysis_history:
            print("История анализов пуста.")
            return
        
        print("\n" + "=" * 60)
        print("ИСТОРИЯ АНАЛИЗОВ ТЕКСТОВЫХ ФАЙЛОВ")
        print("=" * 60)
        
        for i, entry in enumerate(self.analysis_history[-10:], 1): 
            date = datetime.datetime.fromisoformat(entry['timestamp']).strftime("%Y-%m-%d %H:%M")
            filename = os.path.basename(entry['file_path'])
            
            print(f"\n{i}. {date} - {filename}")
            print(f"   Размер: {entry['file_size']:,} байт")
            print(f"   Символов: {entry['stats']['total_chars']:,}")
            print(f"   Слов: {entry['stats']['total_words']:,}")
            print(f"   Удобочитаемость: {entry['readability_score']}")

    def compare_files(self, file_path1, file_path2):
        print(f"\nСравниваем файлы: {file_path1} и {file_path2}")
        
        content1 = self.read_file_content(file_path1)
        if not content1:
            return None
        
        stats1 = self.calculate_basic_stats(content1)
        if not stats1:
            return None
        
        content2 = self.read_file_content(file_path2)
        if not content2:
            return None
        
        stats2 = self.calculate_basic_stats(content2)
        if not stats2:
            return None
        
        report = "=" * 80 + "\n"
        report += "СРАВНИТЕЛЬНЫЙ АНАЛИЗ ДВУХ ТЕКСТОВЫХ ФАЙЛОВ\n"
        report += "=" * 80 + "\n\n"
        
        report += f"Файл 1: {file_path1}\n"
        report += f"Файл 2: {file_path2}\n\n"
        
        report += "ПОКАЗАТЕЛЬ\t\tФАЙЛ 1\tФАЙЛ 2\tРАЗНИЦА\n"
        report += "-" * 60 + "\n"
        
        metrics = [
            ('Символы', 'total_chars'),
            ('Слова', 'total_words'),
            ('Строки', 'total_lines'),
            ('Предложения', 'total_sentences'),
            ('Уникальные слова', 'unique_words')
        ]
        
        for name, key in metrics:
            val1 = stats1[key]
            val2 = stats2[key]
            diff = val2 - val1
            diff_sign = "+" if diff > 0 else ""
            
            report += f"{name:<20}{val1:>10,}{val2:>10,}{diff_sign:>5}{diff:>10,}\n"
        
        read1 = self.calculate_readability_score(stats1)
        read2 = self.calculate_readability_score(stats2)
        read_diff = read2 - read1
        read_sign = "+" if read_diff > 0 else ""
        
        report += f"{'Удобочитаемость':<20}{read1:>10.1f}{read2:>10.1f}{read_sign:>5}{read_diff:>10.1f}\n"
        
        if read1 > read2:
            report += f"\nВывод: Файл 1 более удобочитаем на {read1-read2:.1f} пунктов\n"
        else:
            report += f"\nВывод: Файл 2 более удобочитаем на {read2-read1:.1f} пунктов\n"
        
        return report

def main():
    analyzer = TextFileAnalyzer()
    
    analyzer.load_history_from_json()
    
    print("ТЕКСТОВЫЙ АНАЛИЗАТОР ФАЙЛОВ")
    print("=" * 50)
    
    while True:
        print("\nВыберите действие:")
        print("1. Анализ одного файла")
        print("2. Сравнение двух файлов")
        print("3. Показать историю анализов")
        print("4. Сохранить историю в файл")
        print("5. Выход")
        
        choice = input("Ваш выбор (1-5): ").strip()
        
        if choice == '1':
            file_path = input("Введите путь к файлу для анализа: ").strip()
            if os.path.exists(file_path):
                report = analyzer.analyze_file(file_path)
                if report:
                    print("\n" + report)
                    
                    save_choice = input("Сохранить отчет в файл? (y/n): ").lower()
                    if save_choice == 'y':
                        filename_base = os.path.splitext(os.path.basename(file_path))[0]
                        analyzer.save_analysis_to_file(report, filename_base)
            else:
                print("Файл не существует!")
        
        elif choice == '2':
            file1 = input("Введите путь к первому файлу: ").strip()
            file2 = input("Введите путь ко второму файлу: ").strip()
            
            if os.path.exists(file1) and os.path.exists(file2):
                report = analyzer.compare_files(file1, file2)
                if report:
                    print("\n" + report)
            else:
                print("Один или оба файла не существуют!")
        
        elif choice == '3':
            analyzer.show_history_summary()
        
        elif choice == '4':
            analyzer.save_history_to_json()
        
        elif choice == '5':
            print("Выход из программы...")
            break
        
        else:
            print("Неверный выбор! Попробуйте снова.")

if __name__ == "__main__":

    main()
    

