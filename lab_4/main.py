"""
main.py - Точка входа с поддержкой командной строки
"""
import argparse
import sys
from auth_core import PasswordHasher, UserDatabase, CollisionDemo
from gui import run_gui


def cli_register(args):
    """Регистрация через CLI"""
    db = UserDatabase(args.db)
    if db.user_exists(args.username):
        print(f"Ошибка: пользователь '{args.username}' уже существует")
        return
    
    if args.safe:
        pwd_hash, salt = PasswordHasher.hash_with_salt(args.password)
        db.add_user(args.username, pwd_hash, salt, "SHA-256 with salt")
        print(f" Пользователь '{args.username}' зарегистрирован (с солью)")
    else:
        pwd_hash = PasswordHasher.hash_unsafe(args.password)
        db.add_user(args.username, pwd_hash, None, "SHA-256 without salt")
        print(f" Пользователь '{args.username}' зарегистрирован БЕЗ соли")


def cli_login(args):
    """Авторизация через CLI"""
    db = UserDatabase(args.db)
    user = db.get_user(args.username)
    
    if not user:
        print(f"Ошибка: пользователь '{args.username}' не найден")
        return
    
    if args.safe:
        if not user.get("salt"):
            print("Ошибка: этот пользователь зарегистрирован без соли")
            return
        success = PasswordHasher.verify_with_salt(args.password, user["salt"], user["hash"])
    else:
        success = PasswordHasher.verify_unsafe(args.password, user["hash"])
    
    if success:
        print(f" Добро пожаловать, {args.username}!")
    else:
        print(" Неверный пароль")


def cli_list(args):
    """Список пользователей"""
    db = UserDatabase(args.db)
    users = db.get_all_users()
    
    if not users:
        print("Нет зарегистрированных пользователей")
        return
    
    print("\n Список юзеров:")
    print("-" * 50)
    for username, data in users.items():
        unsafe = " Уязвимо(ай ай ай)" if not data.get("salt") else " Безопасно"
        print(f"  {username}: {data['method']} - {unsafe}")
        print(f"    Хеш: {data['hash'][:32]}...")


def cli_collision(args):
    """Поиск коллизии"""
    target_hash = args.hash or PasswordHasher.hash_unsafe("test_password")
    result = CollisionDemo.find_collision_demo(target_hash, args.max_attempts)
    
    if result:
        print(f"\n Найден пароль: {result}")
    else:
        print(f"\n Коллизия не найдена за {args.max_attempts} попыток")


def cli_analyze(args):
    """Анализ безопасности"""
    db = UserDatabase(args.db)
    unsafe = db.get_unsafe_users()
    
    print("Анализ безопастности")
    
    if unsafe:
        print(f"\n Найдено {len(unsafe)} уязвимых пользователей:")
        for u in unsafe:
            print(f"  - {u}")
        print("\n Уязвимости:")
        print("  - Подвержены rainbow table атакам")
        print("  - Одинаковые пароли дают одинаковые хеши")
        print("  - Высокая скорость перебора")
    else:
        print("\n Все пользователи используют соль!")


def main():
    parser = argparse.ArgumentParser(
        description=" Система аутентификации с хешированием паролей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --gui                           # Запуск GUI
  python main.py register -u alice -p 123 -s     # Регистрация с солью
  python main.py login -u alice -p 123 -s        # Авторизация
  python main.py list                            # Список пользователей
  python main.py collision --max 50000           # Поиск коллизии
  python main.py analyze                         # Анализ безопасности
        """
    )
    
    parser.add_argument("--gui", action="store_true", help="Запустить графический интерфейс")
    parser.add_argument("--db", default="users.json", help="Файл базы данных")
    
    subparsers = parser.add_subparsers(dest="command", help="Команды")
    
    reg_parser = subparsers.add_parser("register", help="Регистрация пользователя")
    reg_parser.add_argument("-u", "--username", required=True)
    reg_parser.add_argument("-p", "--password", required=True)
    reg_parser.add_argument("-s", "--safe", action="store_true", help="Использовать соль")
    
    login_parser = subparsers.add_parser("login", help="Авторизация пользователя")
    login_parser.add_argument("-u", "--username", required=True)
    login_parser.add_argument("-p", "--password", required=True)
    login_parser.add_argument("-s", "--safe", action="store_true", help="Использовать соль")
    
    subparsers.add_parser("list", help="Показать всех пользователей")
    
    coll_parser = subparsers.add_parser("collision", help="Поиск коллизии (демо с tqdm)")
    coll_parser.add_argument("--hash", help="Целевой хеш")
    coll_parser.add_argument("--max", dest="max_attempts", type=int, default=100000, 
                             help="Максимум попыток")
    
    subparsers.add_parser("analyze", help="Анализ безопасности")
    
    args = parser.parse_args()
    
    try:
        if args.gui:
            run_gui()
        elif args.command == "register":
            cli_register(args)
        elif args.command == "login":
            cli_login(args)
        elif args.command == "list":
            cli_list(args)
        elif args.command == "collision":
            cli_collision(args)
        elif args.command == "analyze":
            cli_analyze(args)
        else:
            parser.print_help()
    except Exception as e:
        print(f" Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()