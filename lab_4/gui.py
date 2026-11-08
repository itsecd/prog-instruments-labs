import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                             QFileDialog, QMessageBox, QLineEdit, QHBoxLayout, QTextEdit)
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from crypto import calculate_sha256, find_partial_collision, save_file_hash, verify_file_hash


class LogSignaler(QObject):
    """
    Класс-посредник для передачи строк между потоками.
    """
    text_written = pyqtSignal(str)


class QtLogStream:
    """
    Класс, перехватывающий вывод tqdm
    """

    def __init__(self, signaler : LogSignaler) -> None:
        """
        Инициализирует поток перехвата логов
        :param signaler: Объект для межпоточного взаимодействия
        """
        self.signaler = signaler

    def write(self, text : str) -> None:
        """
        Перехватывает текстовый вывод от tqdm, очищает его и отправляет в GUI
        :param text: текстовая строка индикатора прогресса
        """

        clean_text = text.replace('\r', '').strip()
        if clean_text:
            self.signaler.text_written.emit(clean_text)

    def flush(self):
        """
        Заглушка
        """
        pass


class CollisionWorker(QThread):
    """
    Фоновый поток для поиска коллизии
    """
    finished = pyqtSignal(str, str)

    def __init__(self, prefix : str, signaler : LogSignaler) -> None:
        """
        Инициализирует рабочий поток подбора коллизии
        :param prefix: целевой хеш для поиска
        :param signaler: сигнальный класс для трансляции шагов tqdm
        """
        super().__init__()
        self.prefix = prefix
        self.signaler = signaler
        self.stream = QtLogStream(self.signaler)

    def run(self):
        """
        Запускает алгоритм поиска коллизии в отдельном потоке
        """

        candidate, col_hash = find_partial_collision(self.prefix, out_file=self.stream)
        self.finished.emit(candidate or "", col_hash or "")


class AppGui(QWidget):
    """
    Главное графическое окно приложения для контроля целостности файлов и поиска коллизий
    """

    def __init__(self):
        """
        Инициализирует базовые параметры графического окна и подписки на сигналы логирования
        """
        super().__init__()
        self.selected_file_path = None

        self.signaler = LogSignaler()
        self.signaler.text_written.connect(self.update_log_display)

        self.init_ui()

    def _show_error(self, message: str) -> None:
        """
        Показывает диалоговое окно с сообщением об ошибке
        :param message: текст ошибки для отображения пользователю
        """
        QMessageBox.critical(self, "Ошибка", message)

    def _show_warning(self, message: str) -> None:
        """
        Показывает диалоговое окно с предупреждением
        :param message: текст предупреждения для отображения пользователю
        """
        QMessageBox.warning(self, "Внимание", message)

    def _show_info(self, title: str, message: str) -> None:
        """
        Показывает информационное диалоговое окно
        :param title: заголовок окна
        :param message: текст сообщения для отображения пользователю
        """
        QMessageBox.information(self, title, message)

    def _ensure_file_selected(self) -> bool:
        """
        Проверяет, что пользователь выбрал файл для работы
        :return: True, если файл выбран, иначе False
        """
        if not self.selected_file_path:
            self._show_warning("Сначала выберите файл!")
            return False
        
        return True

    def init_ui(self) -> None:
        """
        Создает и размещает основные секции графического интерфейса приложения
        """
        self.setWindowTitle("Контроль целостности файлов (SHA-256)")
        self.resize(550, 500)

        layout = QVBoxLayout()
        self._build_file_section(layout)
        self._build_collision_section(layout)
        self._build_log_section(layout)
        self.setLayout(layout)

    def _build_file_section(self, layout) -> None:
        """
        Создает секцию выбора файла и работы с его хешем
        :param layout: вертикальный слой, в который добавляются виджеты секции
        """
        self.label_file = QLabel("Файл не выбран", self)
        layout.addWidget(self.label_file)

        btn_select = QPushButton("Выбрать файл для проверки", self)
        btn_select.clicked.connect(self.select_file)
        layout.addWidget(btn_select)

        self.label_hash = QLabel("Текущий хеш: -", self)
        self.label_hash.setWordWrap(True)
        layout.addWidget(self.label_hash)

        btn_save = QPushButton("Сохранить хеш", self)
        btn_save.clicked.connect(self.handle_save)
        layout.addWidget(btn_save)

        btn_verify = QPushButton("Проверить сохранённый хеш", self)
        btn_verify.clicked.connect(self.handle_verify)
        layout.addWidget(btn_verify)

    def _build_collision_section(self, layout) -> None:
        """
        Создает секцию поиска коллизии
        :param layout: вертикальный слой, в который добавляются виджеты секции
        """
        layout.addWidget(QLabel("\nПоиск коллизии"))
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Хеш:"))
        self.entry_prefix = QLineEdit("abc", self)
        hbox.addWidget(self.entry_prefix)

        hbox_widget = QWidget()
        hbox_widget.setLayout(hbox)
        layout.addWidget(hbox_widget)

        self.btn_collision = QPushButton("Запустить поиск коллизии", self)
        self.btn_collision.clicked.connect(self.handle_collision)
        layout.addWidget(self.btn_collision)

    def _build_log_section(self, layout) -> None:
        """
        Создает секцию вывода результатов работы
        :param layout: вертикальный слой, в который добавляются виджеты секции
        """
        layout.addWidget(QLabel("Результаты:"))
        self.log_display = QTextEdit(self)
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("background-color: #1e1e1e; color: #a9dc76; font-family: Consolas, Monaco, monospace;")
        layout.addWidget(self.log_display)

    def select_file(self):
        """
        Вызывает стандартное диалоговое окно ОС для выбора файла и мгновенно рассчитывает его текущий хеш
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл")
        if file_path:
            self.selected_file_path = file_path
            self.label_file.setText(f"Файл: {os.path.basename(file_path)}")
            try:
                self.label_hash.setText(f"Текущий хеш: {calculate_sha256(file_path)}")
            except Exception as e:
                self._show_error(str(e))

    def handle_save(self):
        """
        Обработчик нажатия кнопки сохранения
        """
        if not self._ensure_file_selected():
            return
        try:
            save_path = save_file_hash(self.selected_file_path)
            self._show_info("Успех", f"Хеш сохранен в файл:\n{os.path.basename(save_path)}")
        except Exception as e:
            self._show_error(str(e))

    def handle_verify(self):
        """
        Обработчик нажатия кнопки проверки
        """
        if not self._ensure_file_selected():

            return

        hash_file_path, _ = QFileDialog.getOpenFileName(self, "Выберите файл с хешем", filter="*.sha256")
        if not hash_file_path:
            return

        try:
            is_intact, _, _ = verify_file_hash(self.selected_file_path, hash_file_path)
            if is_intact:
                self._show_info("Результат", "Целостность подтверждена.\nФайл не изменялся.")
            else:
                self._show_error("Хеши не совпадают\nФайл был изменен.")
        except Exception as e:
            self._show_error(str(e))

    def handle_collision(self):
        """
        Инициализирует процесс подбора коллизии
        """
        prefix = self.entry_prefix.text().strip()
        if not prefix:
            self._show_warning("Введите префикс")
            return

        self.log_display.clear()
        self.log_display.append(f">> Запуск перебора для префикса: '{prefix}'\n")
        self.btn_collision.setEnabled(False)


        self.worker = CollisionWorker(prefix, self.signaler)
        self.worker.finished.connect(self.on_collision_finished)
        self.worker.start()

    def update_log_display(self, text : str) -> None:
        """
        Ловит строчки от tqdm и выводит в текстовое поле
        :param text: очищенная текстовая строка состояния прогресс-бара для отображения
        """

        self.log_display.append(text)

        self.log_display.ensureCursorVisible()

    def on_collision_finished(self, candidate : str, col_hash :  str) -> None:
        """
        Разблокирует интерфейс и публикует итоги работы
        :param candidate: найденная исходная строка, породившая коллизию или пустая строка при неудаче
        :param col_hash: результирующий хеш SHA-256, содержащий нужный префикс или пустая строка при неудаче
        """
        self.btn_collision.setEnabled(True)

        if candidate:
            self.log_display.append("\nКоллизия найдена")
            self.log_display.append(f"Исходная строка (оригинал): {candidate}")
            self.log_display.append(f"Результирующий SHA-256 хеш: {col_hash}")
            self._show_info("Успех!", f"Найдено.\nСтрока: {candidate}")
        else:
            self.log_display.append("\nПревышен лимит итераций. Коллизия не найдена.")
            self._show_warning("Коллизия не найдена.")