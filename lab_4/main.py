"""
main.py - Точка входа с поддержкой командной строки
"""
import argparse
import sys
from tqdm import tqdm
from auth_core import (
    AuthService, UserDatabase, CollisionDemo, 
    AuthError, UserExistsError, UserNotFoundError, InvalidPasswordError
)
from gui import run_gui


def cli_register(args):
    db = UserDatabase(args.db)
    service = AuthService(db)
    try:
        method = service.register(args.username, args.password, args.safe)
        print(f"✅ Пользователь '{args.username}' зарегистрирован ({method})")
    except UserExistsError as e:
        print(f"❌ Ошибка: {e}")


def cli_login(args):
    db = UserDatabase(args.db)
    service = AuthService(db)
    try:
        service.login(args.username, args.password, args.safe)
        print(f"✅ Добро пожаловать, {args.username}!")
    except (UserNotFoundError, InvalidPasswordError, AuthError) as e:
        print(f"❌ Ошибка: {e}")


def cli_list(args):
    db = UserDatabase(args.db)
    users = db.get_all_users()
    
    if not users:
        print("Нет зарегистрированных пользователей")
        return
    
    print("\n👥 Список юзеров:")
    print("-" * 50)
    for username, data in users.items():
        unsafe = "⚠️ Уязвимо(ай ай ай)" if not data.get("salt") else "🔒 Безопасно"
        print(f"  {username}: {data['method']} - {unsafe}")
        print(f"    Хеш: {data['hash'][:32]}...")


def cli_collision(args):
    target_hash = args.hash or CollisionDemo.hash_unsafe("test_password") if hasattr(CollisionDemo, 'hash_unsafe') else args.hash
    # Т.к. мы убрали хеширование из CollisionDemo, используем PasswordHasher напрямую, если хеш не задан
    from auth_core import PasswordHasher
    if not target_hash:
         target_hash = PasswordHasher.hash_unsafe("test_password")

    print(f"\n🔎 Поиск пароля для хеша: {target_hash[:16]}...")
    
    # tqdm теперь в CLI
    with tqdm(total=args.max_attempts, desc="Перебор паролей", unit="попытка") as pbar:
        result = None
        for i in range(args.max_attempts):
            test_password = f"password_{i}"
            if PasswordHasher.hash_unsafe(test_password) == target_hash:
                result = test_password
                pbar.set_description("🎯 Коллизия найдена!")
                break
            pbar.update(1)
    
    if result:
        print(f"\n✅ Найден пароль: {result}")
    else:
        print(f"\n❌ Коллизия не найдена за {args.max_attempts} попыток")


def cli_analyze(args):
    db = UserDatabase(args.db)
    unsafe = db.get_unsafe_users()
    
    print("📊 Анализ безопасности")
    
    if unsafe:
        print(f"\n⚠️ Найдено {len(unsafe)} уязвимых пользователей:")
        for u in unsafe:
            print(f"  - {u}")
        print("\nУязвимости:")
        print("  - Подвержены rainbow table атакам")
        print("  - Одинаковые пароли дают одинаковые хеши")
        print("  - Высокая скорость перебора")
    else:
        print("\n✅ Все пользователи используют соль!")


def main():
    parser = argparse.ArgumentParser(
        description="🔐 Система аутентификации с хешированием паролей",
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
        print(f"❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()