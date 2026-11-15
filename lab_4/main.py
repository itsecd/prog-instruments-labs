"""
Главный модуль - точка входа в приложение.
Связующий файл между всеми модулями.
"""

import sys
import os


def check_dependencies():
    """Проверяет наличие необходимых библиотек."""
    missing = []
    
    try:
        import PyQt5
    except ImportError:
        missing.append("PyQt5")
    
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    
    try:
        import tqdm
    except ImportError:
        missing.append("tqdm")
    
    if missing:
        print(" Отсутствуют необходимые библиотеки:")
        for lib in missing:
            print(f"   - {lib}")
        print("\nУстановите их командой:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    print(" Все зависимости установлены")
    return True


def check_files():
    """Проверяет наличие необходимых файлов."""
    required_files = ["collision_finder.py", "gui.py"]
    missing = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(os.path.dirname(__file__), file)):
            missing.append(file)
    
    if missing:
        print(" Отсутствуют необходимые файлы:")
        for file in missing:
            print(f"   - {file}")
        return False
    
    print(" Все необходимые файлы найдены")
    return True


def main():
    """Основная функция запуска."""
    
    print(" ПОИСК КОЛЛИЗИЙ УКОРОЧЕННЫХ ХЕШЕЙ")
    
    # Проверяем файлы
    print("\n Проверка файлов...")
    if not check_files():
        print("\n Невозможно запустить приложение")
        input("Нажмите Enter для выхода...")
        return 1
    
    # Проверяем зависимости
    print("\n Проверка зависимостей...")
    if not check_dependencies():
        print("\nНевозможно запустить приложение")
        input("Нажмите Enter для выхода...")
        return 1
    
    # Запускаем GUI
    print("\n Запуск графического интерфейса...\n")
    try:
        # Импортируем класс CollisionGUI из gui.py
        from gui import CollisionGUI
        from PyQt5.QtWidgets import QApplication
        
        # Создаём и запускаем приложение
        app = QApplication(sys.argv)
        window = CollisionGUI()
        window.show()
        print("Приложение успешно запущено!")
        print("Для выхода закройте окно приложения\n")
        sys.exit(app.exec_())
        
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("\nПроверьте, что в файле gui.py определён класс CollisionGUI")
        import traceback
        traceback.print_exc()
        input("\nНажмите Enter для выхода...")
        return 1
        
    except Exception as e:
        print(f"Ошибка при запуске GUI: {e}")
        import traceback
        traceback.print_exc()
        input("\nНажмите Enter для выхода...")
        return 1


if __name__ == "__main__":
    sys.exit(main())