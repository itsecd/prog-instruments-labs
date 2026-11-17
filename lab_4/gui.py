"""
gui.py - Графический интерфейс на PyQt6 (ИСПРАВЛЕННАЯ ВЕРСИЯ)
"""
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTextEdit, QTabWidget, QMessageBox, QGroupBox,
                             QRadioButton, QProgressBar, QComboBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from auth_core import PasswordHasher, UserDatabase, CollisionDemo


class CollisionThread(QThread):
    """Поток для поиска коллизий"""
    progress = pyqtSignal(int)
    result = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def __init__(self, target_hash: str, max_attempts: int = 100000):
        super().__init__()
        self.target_hash = target_hash
        self.max_attempts = max_attempts
    
    def run(self):
        found = CollisionDemo.find_collision_demo(self.target_hash, self.max_attempts)
        self.result.emit(found if found else "Не найдено")
        self.finished_signal.emit()


class AuthGUI(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.db = UserDatabase()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(" Система аутентификации - Лабораторная №4")
        self.setGeometry(100, 100, 650, 550)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Вкладки
        tabs = QTabWidget()
        tabs.addTab(self.create_auth_tab(), " Авторизация")
        tabs.addTab(self.create_register_tab(), " Регистрация")
        tabs.addTab(self.create_users_tab(), " Пользователи")
        tabs.addTab(self.create_collision_tab(), " Поиск коллизий")
        tabs.addTab(self.create_analysis_tab(), " Анализ")
        
        layout.addWidget(tabs)
        
        # Статусная строка
        self.statusBar().showMessage("Готов к работе")
    
    def create_auth_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        group = QGroupBox("Вход в систему")
        group_layout = QVBoxLayout()
        
        self.auth_login = QLineEdit()
        self.auth_login.setPlaceholderText("Логин")
        self.auth_password = QLineEdit()
        self.auth_password.setPlaceholderText("Пароль")
        self.auth_password.setEchoMode(QLineEdit.EchoMode.Password)
        
        self.auth_method_safe = QRadioButton("С солью (безопасно)")
        self.auth_method_unsafe = QRadioButton("Без соли (уязвимо)")
        self.auth_method_safe.setChecked(True)
        
        auth_btn = QPushButton("Войти")
        auth_btn.clicked.connect(self.do_login)
        
        group_layout.addWidget(QLabel("Логин:"))
        group_layout.addWidget(self.auth_login)
        group_layout.addWidget(QLabel("Пароль:"))
        group_layout.addWidget(self.auth_password)
        group_layout.addWidget(QLabel("Метод проверки:"))
        group_layout.addWidget(self.auth_method_safe)
        group_layout.addWidget(self.auth_method_unsafe)
        group_layout.addWidget(auth_btn)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        layout.addStretch()
        
        return tab
    
    def create_register_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        group = QGroupBox("Регистрация нового пользователя")
        group_layout = QVBoxLayout()
        
        self.reg_login = QLineEdit()
        self.reg_login.setPlaceholderText("Логин")
        self.reg_password = QLineEdit()
        self.reg_password.setPlaceholderText("Пароль")
        self.reg_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.reg_password_confirm = QLineEdit()
        self.reg_password_confirm.setPlaceholderText("Подтвердите пароль")
        self.reg_password_confirm.setEchoMode(QLineEdit.EchoMode.Password)
        
        self.reg_method_safe = QRadioButton("С солью (рекомендуется)")
        self.reg_method_unsafe = QRadioButton("Без соли (не рекомендуется)")
        self.reg_method_safe.setChecked(True)
        
        reg_btn = QPushButton("Зарегистрироваться")
        reg_btn.clicked.connect(self.do_register)
        
        group_layout.addWidget(QLabel("Логин:"))
        group_layout.addWidget(self.reg_login)
        group_layout.addWidget(QLabel("Пароль:"))
        group_layout.addWidget(self.reg_password)
        group_layout.addWidget(QLabel("Подтверждение:"))
        group_layout.addWidget(self.reg_password_confirm)
        group_layout.addWidget(QLabel("Метод хранения:"))
        group_layout.addWidget(self.reg_method_safe)
        group_layout.addWidget(self.reg_method_unsafe)
        group_layout.addWidget(reg_btn)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        layout.addStretch()
        
        return tab
    
    def create_users_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        refresh_btn = QPushButton("Обновить список")
        refresh_btn.clicked.connect(self.refresh_users_list)
        
        self.users_text = QTextEdit()
        self.users_text.setReadOnly(True)
        
        clear_btn = QPushButton("Очистить всех пользователей")
        clear_btn.setStyleSheet("background-color: #ff4444; color: white;")
        clear_btn.clicked.connect(self.clear_all_users)
        
        layout.addWidget(refresh_btn)
        layout.addWidget(self.users_text)
        layout.addWidget(clear_btn)
        
        self.refresh_users_list()
        return tab
    
    def create_collision_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        group = QGroupBox("Демонстрация поиска коллизий")
        group_layout = QVBoxLayout()
        
        self.collision_user_select = QComboBox()
        self.collision_user_select.addItem("-- Выберите пользователя для атаки --")
        self.refresh_user_list_for_collision()
        
        refresh_users_btn = QPushButton("Обновить список пользователей")
        refresh_users_btn.clicked.connect(self.refresh_user_list_for_collision)
        
        self.collision_hash = QLineEdit()
        self.collision_hash.setPlaceholderText("Или введите хеш вручную (оставьте пустым для демо-режима)")
        
        self.collision_progress = QProgressBar()
        self.collision_progress.setVisible(False)
        
        search_btn = QPushButton("Начать поиск коллизии")
        search_btn.clicked.connect(self.start_collision_search)
        
        self.collision_result = QLabel("Результат будет здесь")
        
        group_layout.addWidget(QLabel("1. Выберите пользователя (реальная атака):"))
        group_layout.addWidget(self.collision_user_select)
        group_layout.addWidget(refresh_users_btn)
        group_layout.addWidget(QLabel("2. ИЛИ введите хеш вручную:"))
        group_layout.addWidget(self.collision_hash)
        group_layout.addWidget(QLabel("3. Начать поиск:"))
        group_layout.addWidget(search_btn)
        group_layout.addWidget(self.collision_progress)
        group_layout.addWidget(self.collision_result)
        
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        layout.addStretch()
        
        return tab
    
    def create_analysis_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        analyze_btn = QPushButton("Выполнить анализ безопасности")
        analyze_btn.clicked.connect(self.do_analysis)
        
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        
        layout.addWidget(analyze_btn)
        layout.addWidget(self.analysis_text)
        
        return tab
    
    def refresh_user_list_for_collision(self):
        """Обновить список пользователей для выбора атаки"""
        self.collision_user_select.clear()
        self.collision_user_select.addItem("-- Выберите пользователя для атаки --")
        
        users = self.db.get_all_users()
        for username, data in users.items():
            if not data.get("salt"):
                self.collision_user_select.addItem(f" {username} (без соли)")
            else:
                self.collision_user_select.addItem(f" {username} (с солью)")
    
    def do_login(self):
        username = self.auth_login.text().strip()
        password = self.auth_password.text().strip()
        use_salt = self.auth_method_safe.isChecked()
        
        if not username or not password:
            QMessageBox.warning(self, "Ошибка", "Введите логин и пароль")
            return
        
        user = self.db.get_user(username)
        if not user:
            QMessageBox.warning(self, "Ошибка", "Пользователь не найден")
            return
        
        try:
            if use_salt:
                if not user.get("salt"):
                    QMessageBox.warning(self, "Ошибка", 
                                         "Этот пользователь зарегистрирован без соли. "
                                         "Используйте метод 'Без соли'")
                    return
                success = PasswordHasher.verify_with_salt(password, user["salt"], user["hash"])
            else:
                success = PasswordHasher.verify_unsafe(password, user["hash"])
            
            if success:
                QMessageBox.information(self, "Успех", f"Добро пожаловать, {username}!")
                self.statusBar().showMessage(f"Пользователь {username} вошёл в систему")
            else:
                QMessageBox.critical(self, "Ошибка", "Неверный пароль")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {e}")
    
    def do_register(self):
        username = self.reg_login.text().strip()
        password = self.reg_password.text().strip()
        password_confirm = self.reg_password_confirm.text().strip()
        use_salt = self.reg_method_safe.isChecked()
        
        if not username or not password:
            QMessageBox.warning(self, "Ошибка", "Заполните все поля")
            return
        
        if password != password_confirm:
            QMessageBox.warning(self, "Ошибка", "Пароли не совпадают")
            return
        
        if self.db.user_exists(username):
            QMessageBox.warning(self, "Ошибка", "Пользователь уже существует")
            return
        
        try:
            if use_salt:
                password_hash, salt = PasswordHasher.hash_with_salt(password)
                method = "SHA-256 with salt"
                self.db.add_user(username, password_hash, salt, method)
                msg = f"Пользователь {username} зарегистрирован с солью (безопасно)"
            else:
                password_hash = PasswordHasher.hash_unsafe(password)
                method = "SHA-256 without salt (UNSAFE)"
                self.db.add_user(username, password_hash, None, method)
                msg = f" Пользователь {username} зарегистрирован БЕЗ соли (Очень плохо!)"
            
            QMessageBox.information(self, "Успех", msg)
            self.statusBar().showMessage(msg)
            self.reg_login.clear()
            self.reg_password.clear()
            self.reg_password_confirm.clear()
            self.refresh_users_list()
            self.refresh_user_list_for_collision()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось зарегистрировать: {e}")
    
    def refresh_users_list(self):
        users = self.db.get_all_users()
        if not users:
            self.users_text.setText(" Нет зарегистрированных пользователей\n\n"
                                   "Нажмите на вкладку 'Регистрация', чтобы добавить пользователя.")
            return
        
        text = " ЗАРЕГИСТРИРОВАННЫЕ ПОЛЬЗОВАТЕЛИ:\n" + "="*50 + "\n\n"
        for username, data in users.items():
            if data.get('salt'):
                text += f" {username}\n"
                text += f"    Хеш: {data['hash'][:32]}...\n"
                text += f"    Метод: {data['method']} - БЕЗОПАСНО\n"
                text += f"    Соль: {data['salt'][:16]}...\n"
            else:
                text += f" {username} (УЯЗВИМЫЙ!)\n"
                text += f"    Хеш: {data['hash'][:32]}...\n"
                text += f"    Метод: {data['method']} - НЕБЕЗОПАСНО!\n"
                text += f"    НЕТ СОЛИ! Подвержен rainbow table атакам!\n"
            text += "\n"
        
        self.users_text.setText(text)
    
    def clear_all_users(self):
        """Очистка всех пользователей с подтверждением"""
        reply = QMessageBox.question(self, "Подтверждение", 
                                     "ВНИМАНИЕ! Это удалит ВСЕХ пользователей.\n"
                                     "Продолжить?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.db.delete_all()
            self.refresh_users_list()
            self.refresh_user_list_for_collision()
            QMessageBox.information(self, "Готово", "База пользователей очищена")
            self.statusBar().showMessage("База пользователей очищена")
    
    def start_collision_search(self):
        """Начать поиск коллизии с корректной проверкой"""
        
        target_hash = None
        
        selected = self.collision_user_select.currentText()
        
        if selected and not selected.startswith("-- Выберите"):
            username = selected.split(" ", 1)[-1].split(" (")[0]
            user = self.db.get_user(username)
            
            if user:
                if user.get("salt"):
                    reply = QMessageBox.question(self, "Предупреждение",
                                                "Этот пользователь использует соль!\n"
                                                "Поиск пароля для хеша с солью практически невозможен.\n"
                                                "Продолжить? (это займёт очень много времени)",
                                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    if reply == QMessageBox.StandardButton.No:
                        return
                
                target_hash = user["hash"]
                self.statusBar().showMessage(f"Поиск пароля для пользователя {username}...")
        
        if not target_hash:
            hash_input = self.collision_hash.text().strip()
            if hash_input:
                target_hash = hash_input
                self.statusBar().showMessage("Поиск по введённому хешу...")
            else:
                reply = QMessageBox.question(self, "Демо-режим",
                                            "Вы не выбрали пользователя и не ввели хеш.\n"
                                            "Запустить ДЕМО-режим (поиск тестового пароля 'password_12345')?\n\n"
                                            "Это не имеет отношения к вашей базе пользователей!",
                                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    target_hash = PasswordHasher.hash_unsafe("password_12345")
                    self.statusBar().showMessage("ДЕМО-режим: поиск тестового пароля...")
                else:
                    return
        
        # Запускаем поиск
        self.collision_progress.setVisible(True)
        self.collision_progress.setRange(0, 0)  
        self.collision_result.setText(" Поиск коллизии... Смотрите консоль")
        
        self.collision_thread = CollisionThread(target_hash, 100000)
        self.collision_thread.result.connect(self.on_collision_result)
        self.collision_thread.finished_signal.connect(self.on_collision_finished)
        self.collision_thread.start()
    
    def on_collision_result(self, result):
        if result and result != "Не найдено":
            self.collision_result.setText(f" Найден пароль: {result}")
            self.statusBar().showMessage(f"Коллизия найдена! Пароль: {result}")
        else:
            self.collision_result.setText(" Коллизия не найдена за допустимое время")
            self.statusBar().showMessage("Коллизия не найдена")
    
    def on_collision_finished(self):
        self.collision_progress.setVisible(False)
    
    def do_analysis(self):
        analysis_text = "=" * 60 + "\n"
        analysis_text += "Анализ безопастности паролей\n"
        analysis_text += "=" * 60 + "\n\n"
        
        users = self.db.get_all_users()
        
        if not users:
            analysis_text += " В базе нет пользователей!\n\n"
            analysis_text += "Сначала зарегистрируйте хотя бы одного пользователя\n"
            analysis_text += "для демонстрации уязвимостей.\n"
        else:
            unsafe_users = self.db.get_unsafe_users()
            
            if unsafe_users:
                analysis_text += f" Найдены уязвимые пользователи ({len(unsafe_users)} шт.):\n"
                for user in unsafe_users:
                    analysis_text += f"  - {user}\n"
                
                
            else:
                analysis_text += " Все супер!\n\n"

        
        
        self.analysis_text.setText(analysis_text)


def run_gui():
    app = QApplication(sys.argv)
    window = AuthGUI()
    window.show()
    sys.exit(app.exec())