import argparse
from hash import *

def main():
    parser = argparse.ArgumentParser(description="Хеш-функции: вычисление SHA-256, проверка целостности и лавинного эффекта")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    hash_parser = subparsers.add_parser('hash', help='Вычислить SHA-256 хеш')
    group = hash_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', help='Путь к файлу')
    group.add_argument('-t', '--text', help='Текст')
    hash_parser.add_argument('-s', '--save', help='Сохранить хеш в файл')
    
    check_parser = subparsers.add_parser('check', help='Проверить целостность файла')
    check_parser.add_argument('-f', '--file', required=True, help='Путь к файлу')
    check_group = check_parser.add_mutually_exclusive_group(required=True)
    check_group.add_argument('-hf', '--hash-file', help='Файл с сохранённым хешем')
    check_group.add_argument('-hv', '--hash-value', help='Хеш для сравнения')

    av_parser = subparsers.add_parser('avalanche', help='Показать лавинный эффект')
    av_parser.add_argument('text1', help='Первый текст')
    av_parser.add_argument('text2', help='Второй текст')
    
    args = parser.parse_args()
    
    match args.command:
        case 'hash':
            """
            Обработка команды hash: вычисляет хеш
            Аргументы: Текст, либо файл с текстом 
            Возвращает: Вычисленный хеш; сохранение хеша в файл
            """
            if args.file:
                result = hash_file(args.file)
                if result is None:
                    print(f"Ошибка: не удалось прочитать файл '{args.file}'")
                    return
                print(f"Хеш файла: {result}")
            else:
                result = hash_text(args.text)
                print(f"Хеш текста: {result}")
            
            if args.save:
                if save_hash(result, args.save):
                    print(f"Хеш сохранён в '{args.save}'")
                else:
                    print(f"Ошибка: не удалось сохранить хеш")
        
        case 'check':
            """
            Обработка команды check: проверяет хеш на целостность
            Аргументы: Файл с проверяемым хешем; файл с ожидаемым хешом, либо ввод вручную
            Возвращает: Результат проверка: Файл изменён - нецелостный ; Файл не изменён - целостный
            """
            if args.hash_file:
                expected = load_hash(args.hash_file)
                if expected is None:
                    print(f"Ошибка: не удалось прочитать хеш из '{args.hash_file}'")
                    return
            else:
                expected = args.hash_value
            
            if check_integrity(args.file, expected):
                print(f"Файл '{args.file}' не изменён")
            else:
                print(f"Файл '{args.file}' изменён!")
        
        case 'avalanche':
            """
            Обработка команды avalanche: проверяет изменение хеша от двух разных текстов
            Аргументы: Два текста для вычисления хеша
            Возвращает: Хеш от двух текстов; Различие бит; % различия хеша
            """
            h1, h2, diff, percent = avalanche(args.text1, args.text2)
            print(f"\nТекст 1: {args.text1}")
            print(f"Хеш 1: {h1}")
            print(f"\nТекст 2: {args.text2}")
            print(f"Хеш 2: {h2}")
            print(f"\nРазличается бит: {diff} из 256 ({percent:.1f}%)")


if __name__ == "__main__":
    main()