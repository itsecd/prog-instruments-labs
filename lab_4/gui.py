import sys
from typing import Optional
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QTextEdit, QLabel, QSpinBox,
    QComboBox, QHBoxLayout, QMessageBox, QProgressBar,
    QTabWidget, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QCheckBox, QStatusBar, QGridLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QPalette, QColor

from collision_finder import ShortHashCollisionFinder

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class CollisionWorker(QThread):
    finished = pyqtSignal(object, object, str, int, str)
    
    def __init__(self, trunc_bits: int, max_attempts: int):
        super().__init__()
        self.trunc_bits = trunc_bits
        self.max_attempts = max_attempts
        self._is_cancelled = False
    
    def cancel(self):
        self._is_cancelled = True
    
    def run(self):
        try:
            finder = ShortHashCollisionFinder(trunc_bits=self.trunc_bits)
            msg1, msg2, sh, attempts = finder.find_collision(max_attempts=self.max_attempts)
            
            if self._is_cancelled:
                self.finished.emit(None, None, None, 0, "Поиск отменён")
            else:
                self.finished.emit(msg1, msg2, sh, attempts, None)
        except Exception as e:
            self.finished.emit(None, None, None, 0, str(e))


class ExperimentsWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict, str)
    status = pyqtSignal(str)
    
    def __init__(self, bits_list, num_experiments, max_attempts):
        super().__init__()
        self.bits_list = bits_list
        self.num_experiments = num_experiments
        self.max_attempts = max_attempts
        self._is_cancelled = False
    
    def cancel(self):
        self._is_cancelled = True
    
    def run(self):
        results = {}
        total = len(self.bits_list) * self.num_experiments
        current = 0
        
        for bits in self.bits_list:
            if self._is_cancelled:
                self.finished.emit({}, "Эксперименты отменены")
                return
            
            self.status.emit(f"Проведение экспериментов для {bits} бит...")
            finder = ShortHashCollisionFinder(trunc_bits=bits)
            exp_results = {}
            
            for i in range(self.num_experiments):
                if self._is_cancelled:
                    self.finished.emit({}, "Эксперименты отменены")
                    return
                
                _, _, _, attempts = finder.find_collision(max_attempts=self.max_attempts, show_progress=False)
                exp_results[i + 1] = attempts if attempts < self.max_attempts else None
                current += 1
                self.progress.emit(current, total)
            
            results[bits] = exp_results
        
        self.finished.emit(results, None)


class CollisionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Поиск коллизий укороченных хешей (SHA-256)")
        self.setGeometry(100, 100, 950, 700)
        
        # Установка стилей
        self.setup_styles()
        
        self.worker: Optional[CollisionWorker] = None
        self.exp_worker: Optional[ExperimentsWorker] = None
        
        self.setup_ui()
        self.update_theory_label()
    
    def setup_styles(self):
        """Настройка нежных фиолетово-зелёных тонов"""
        
        # Основная цветовая палитра
        self.colors = {
            'bg_main': '#2D2B3A',           # тёмно-филалетовый фон
            'bg_secondary': '#3A3750',       # средний филалетовый
            'bg_tertiary': '#48456A',        # светлый филалетовый
            'accent_lavender': '#9B8BB5',    # нежная лаванда
            'accent_mint': '#8BB59B',        # нежная мята
            'accent_soft_lavender': '#B8A9C9', # мягкая лаванда
            'accent_soft_mint': '#A9C9B8',   # мягкая мята
            'text_light': '#E8E6F0',         # светлый текст
            'text_muted': '#B8B5C4',         # приглушённый текст
            'success': '#8BB59B',            # мятный для успеха
            'error': '#C9A9B8',              # розовато-филалетовый для ошибок
            'warning': '#D4C9A9',            # мягкий жёлтый
            'border': '#5B5080'              # филалетовая граница
        }
        
        # Общий стиль приложения
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {self.colors['bg_main']};
            }}
            
            QTabWidget::pane {{
                border: 1px solid {self.colors['border']};
                background-color: {self.colors['bg_secondary']};
                border-radius: 5px;
            }}
            
            QTabBar::tab {{
                background-color: {self.colors['bg_tertiary']};
                color: {self.colors['text_light']};
                padding: 8px 16px;
                margin: 2px;
                border-radius: 4px;
                font-weight: bold;
            }}
            
            QTabBar::tab:selected {{
                background-color: {self.colors['accent_lavender']};
                color: {self.colors['bg_main']};
            }}
            
            QTabBar::tab:hover {{
                background-color: {self.colors['accent_soft_lavender']};
            }}
            
            QGroupBox {{
                color: {self.colors['accent_soft_mint']};
                font-weight: bold;
                border: 1px solid {self.colors['border']};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: {self.colors['accent_mint']};
            }}
            
            QLabel {{
                color: {self.colors['text_light']};
            }}
            
            QPushButton {{
                background-color: {self.colors['accent_lavender']};
                color: {self.colors['bg_main']};
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }}
            
            QPushButton:hover {{
                background-color: {self.colors['accent_soft_lavender']};
            }}
            
            QPushButton:pressed {{
                background-color: {self.colors['accent_mint']};
            }}
            
            QPushButton:disabled {{
                background-color: {self.colors['text_muted']};
                color: {self.colors['bg_secondary']};
            }}
            
            QTextEdit, QLineEdit {{
                background-color: {self.colors['bg_tertiary']};
                color: {self.colors['text_light']};
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                padding: 5px;
                font-family: 'Courier New', monospace;
            }}
            
            QSpinBox, QComboBox {{
                background-color: {self.colors['bg_tertiary']};
                color: {self.colors['text_light']};
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                padding: 3px;
            }}
            
            QSpinBox::up-button, QSpinBox::down-button,
            QComboBox::drop-down {{
                background-color: {self.colors['accent_lavender']};
                border: none;
                width: 20px;
            }}
            
            QProgressBar {{
                border: 1px solid {self.colors['border']};
                border-radius: 5px;
                text-align: center;
                color: {self.colors['text_light']};
                background-color: {self.colors['bg_tertiary']};
            }}
            
            QProgressBar::chunk {{
                background-color: {self.colors['accent_mint']};
                border-radius: 4px;
            }}
            
            QTableWidget {{
                background-color: {self.colors['bg_tertiary']};
                color: {self.colors['text_light']};
                border: 1px solid {self.colors['border']};
                gridline-color: {self.colors['border']};
            }}
            
            QHeaderView::section {{
                background-color: {self.colors['bg_secondary']};
                color: {self.colors['accent_mint']};
                padding: 5px;
                border: 1px solid {self.colors['border']};
                font-weight: bold;
            }}
            
            QCheckBox {{
                color: {self.colors['text_light']};
                spacing: 5px;
            }}
            
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid {self.colors['border']};
                background-color: {self.colors['bg_tertiary']};
            }}
            
            QCheckBox::indicator:checked {{
                background-color: {self.colors['accent_mint']};
                border-color: {self.colors['accent_mint']};
            }}
            
            QStatusBar {{
                background-color: {self.colors['bg_secondary']};
                color: {self.colors['text_muted']};
                border-top: 1px solid {self.colors['border']};
            }}
            
            QScrollBar:vertical {{
                background-color: {self.colors['bg_secondary']};
                width: 12px;
                border-radius: 6px;
            }}
            
            QScrollBar::handle:vertical {{
                background-color: {self.colors['accent_lavender']};
                border-radius: 6px;
                min-height: 20px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background-color: {self.colors['accent_mint']};
            }}
        """)
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Заголовок с градиентным эффектом
        title_label = QLabel("🔐 Поиск коллизий укороченных хешей")
        title_label.setStyleSheet(f"""
            font-size: 20px;
            font-weight: bold;
            color: {self.colors['accent_soft_mint']};
            padding: 10px;
            background-color: {self.colors['bg_secondary']};
            border-radius: 8px;
            margin-bottom: 10px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        self.setup_search_tab()
        self.setup_experiments_tab()
    
    def setup_search_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Настройки
        settings_group = QGroupBox("⚙️ Настройки поиска")
        settings_layout = QVBoxLayout()
        
        bits_layout = QHBoxLayout()
        bits_label = QLabel("Длина хеша (бит):")
        bits_label.setStyleSheet(f"color: {self.colors['accent_soft_lavender']}; font-weight: bold;")
        bits_layout.addWidget(bits_label)
        
        self.bits_combo = QComboBox()
        self.bits_combo.addItems(["8", "12", "16"])
        bits_layout.addWidget(self.bits_combo)
        
        self.bits_info = QLabel("(8 бит = 1 байт = 256 значений, 12 бит = 4096, 16 бит = 65536)")
        self.bits_info.setStyleSheet(f"color: {self.colors['text_muted']}; font-size: 10px;")
        bits_layout.addWidget(self.bits_info)
        settings_layout.addLayout(bits_layout)
        
        attempts_layout = QHBoxLayout()
        attempts_label = QLabel("Макс. попыток:")
        attempts_label.setStyleSheet(f"color: {self.colors['accent_soft_lavender']}; font-weight: bold;")
        attempts_layout.addWidget(attempts_label)
        
        self.attempts_spin = QSpinBox()
        self.attempts_spin.setRange(1000, 200000)
        self.attempts_spin.setValue(50000)
        attempts_layout.addWidget(self.attempts_spin)
        
        self.theory_label = QLabel("")
        self.theory_label.setStyleSheet(f"color: {self.colors['accent_mint']}; font-weight: bold;")
        attempts_layout.addWidget(self.theory_label)
        
        settings_layout.addLayout(attempts_layout)
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        self.btn_search = QPushButton("✨ Начать поиск коллизии")
        self.btn_search.clicked.connect(self.start_collision_search)
        buttons_layout.addWidget(self.btn_search)
        
        self.btn_cancel = QPushButton("❌ Отменить")
        self.btn_cancel.clicked.connect(self.cancel_search)
        self.btn_cancel.setEnabled(False)
        buttons_layout.addWidget(self.btn_cancel)
        layout.addLayout(buttons_layout)
        
        # Прогресс
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Результат
        result_group = QGroupBox("📋 Результат поиска коллизии")
        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Courier New", 10))
        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        self.tabs.addTab(tab, "🔍 Поиск коллизии")
        
        self.bits_combo.currentTextChanged.connect(self.update_theory_label)
    
    def update_theory_label(self):
        bits = int(self.bits_combo.currentText())
        finder = ShortHashCollisionFinder(trunc_bits=bits)
        expected = finder.theoretical_expected_attempts()
        self.theory_label.setText(f"📊 Теория (атака дней рождения): ~{expected} попыток")
    
    def start_collision_search(self):
        bits = int(self.bits_combo.currentText())
        attempts = self.attempts_spin.value()
        
        self.btn_search.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.result_text.clear()
        self.result_text.append("🔍 Поиск коллизии... (может занять время)")
        self.result_text.append(f"📋 Параметры: длина={bits} бит, попыток={attempts}")
        self.result_text.append("")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        
        self.worker = CollisionWorker(bits, attempts)
        self.worker.finished.connect(self.on_collision_found)
        self.worker.start()
    
    def cancel_search(self):
        if self.worker:
            self.worker.cancel()
            self.result_text.append("⏸️ Отмена поиска...")
    
    def on_collision_found(self, msg1, msg2, short_hash, attempts, error):
        self.progress.setVisible(False)
        self.btn_search.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        
        if error:
            self.result_text.append(f" Ошибка: {error}")
            return
        
        bits = int(self.bits_combo.currentText())
        
        if msg1 is None:
            self.result_text.append(f" Коллизия не найдена за {attempts} попыток")
            self.result_text.append(f" Совет: увеличьте количество попыток или уменьшите длину хеша")
            return
        
        finder = ShortHashCollisionFinder(trunc_bits=bits)
        
               # Вычисляем хеши для обоих сообщений
        hash1_int = finder._compute_hash(msg1)
        hash2_int = finder._compute_hash(msg2)

        # HEX представление (логика инкапсулирована в finder)
        hash1_hex = finder.short_hash_hex(msg1)
        hash2_hex = finder.short_hash_hex(msg2)
        # Полные хеши для справки
        import hashlib
        full_hash1 = hashlib.sha256(msg1).hexdigest()
        full_hash2 = hashlib.sha256(msg2).hexdigest()
        
        is_valid, _, _ = finder.verify_collision(msg1, msg2)
        prob = finder.theoretical_probability(attempts)
        expected = finder.theoretical_expected_attempts()
        
        # Красивый вывод результата
        self.result_text.append("-_" * 35)
        self.result_text.append("✅ КОЛЛИЗИЯ НАЙДЕНА!")
        self.result_text.append("-_" * 35)
        
        self.result_text.append("")
        self.result_text.append("📊 ОСНОВНАЯ ИНФОРМАЦИЯ:")
        self.result_text.append(f"  • Длина хеша: {bits} бит")
        self.result_text.append(f"  • Количество попыток: {attempts}")
        self.result_text.append(f"  • Теоретическое ожидание: ~{expected} попыток")
        self.result_text.append(f"  • Вероятность за {attempts} попыток: {prob:.6f} ({prob*100:.4f}%)")
        
        self.result_text.append("")
        self.result_text.append("🔢 СРАВНЕНИЕ УКОРОЧЕННЫХ ХЕШЕЙ:")
        self.result_text.append(f"  • Хеш сообщения 1 (HEX):  {hash1_hex}")
        self.result_text.append(f"  • Хеш сообщения 2 (HEX):  {hash2_hex}")
        self.result_text.append(f"  • Хеш сообщения 1 (DEC):  {hash1_int}")
        self.result_text.append(f"  • Хеш сообщения 2 (DEC):  {hash2_int}")
        
        if hash1_hex == hash2_hex:
            self.result_text.append("  • Результат: ХЕШИ СОВПАДАЮТ!")
        else:
            self.result_text.append("  • Результат: ХЕШИ НЕ СОВПАДАЮТ (ошибка!)")
        
        self.result_text.append("")
        self.result_text.append("📝 СРАВНЕНИЕ САМИХ СООБЩЕНИЙ:")
        self.result_text.append(f"  • Сообщение 1: {msg1.decode('latin1', errors='replace')}")
        self.result_text.append(f"  • Сообщение 2: {msg2.decode('latin1', errors='replace')}")
        self.result_text.append(f"  • Сообщения разные? {'✅ ДА' if msg1 != msg2 else '❌ НЕТ'}")
        
        self.result_text.append("")
        self.result_text.append("🔬 ПОЛНЫЕ ХЕШИ SHA-256 (для справки):")
        self.result_text.append(f"  • Полный хеш 1: {full_hash1}")
        self.result_text.append(f"  • Полный хеш 2: {full_hash2}")
        self.result_text.append(f"  • Полные хеши разные? {' ДА' if full_hash1 != full_hash2 else ' НЕТ'}")
        
        if is_valid:
            self.result_text.append("")
            self.result_text.append(" ПРОВЕРКА ПРОЙДЕНА: коллизия подтверждена!")
        else:
            self.result_text.append("")
            self.result_text.append(" ОШИБКА: коллизия не подтверждена!")
    
    def setup_experiments_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        settings_group = QGroupBox("🧪 Параметры экспериментов")
        settings_layout = QGridLayout()
        
        bits_label = QLabel("Длины хешей:")
        bits_label.setStyleSheet(f"color: {self.colors['accent_soft_lavender']}; font-weight: bold;")
        settings_layout.addWidget(bits_label, 0, 0)
        
        bits_selection = QHBoxLayout()
        self.bits_8 = QCheckBox("8 бит")
        self.bits_8.setChecked(True)
        self.bits_12 = QCheckBox("12 бит")
        self.bits_12.setChecked(True)
        self.bits_16 = QCheckBox("16 бит")
        self.bits_16.setChecked(True)
        bits_selection.addWidget(self.bits_8)
        bits_selection.addWidget(self.bits_12)
        bits_selection.addWidget(self.bits_16)
        settings_layout.addLayout(bits_selection, 0, 1)
        
        exp_label = QLabel("Кол-во экспериментов:")
        exp_label.setStyleSheet(f"color: {self.colors['accent_soft_lavender']}; font-weight: bold;")
        settings_layout.addWidget(exp_label, 1, 0)
        self.exp_count = QSpinBox()
        self.exp_count.setRange(1, 50)
        self.exp_count.setValue(10)
        settings_layout.addWidget(self.exp_count, 1, 1)
        
        attempts_label = QLabel("Макс. попыток:")
        attempts_label.setStyleSheet(f"color: {self.colors['accent_soft_lavender']}; font-weight: bold;")
        settings_layout.addWidget(attempts_label, 2, 0)
        self.exp_attempts = QSpinBox()
        self.exp_attempts.setRange(1000, 200000)
        self.exp_attempts.setValue(50000)
        settings_layout.addWidget(self.exp_attempts, 2, 1)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        buttons_layout = QHBoxLayout()
        self.exp_btn = QPushButton("🚀 Запустить эксперименты")
        self.exp_btn.clicked.connect(self.start_experiments)
        buttons_layout.addWidget(self.exp_btn)
        
        self.exp_cancel_btn = QPushButton("⏹️ Отменить")
        self.exp_cancel_btn.clicked.connect(self.cancel_experiments)
        self.exp_cancel_btn.setEnabled(False)
        buttons_layout.addWidget(self.exp_cancel_btn)
        layout.addLayout(buttons_layout)
        
        self.exp_progress = QProgressBar()
        self.exp_progress.setVisible(False)
        layout.addWidget(self.exp_progress)
        
        self.exp_status = QLabel("")
        self.exp_status.setStyleSheet(f"color: {self.colors['accent_soft_lavender']};")
        layout.addWidget(self.exp_status)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(["Длина (бит)", "Успешно", "Ср. попыток", "Теория", "Отклонение"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.results_table)
        
        # Настройка цветов для таблицы
        self.results_table.setStyleSheet(f"""
            QTableWidget::item:selected {{
                background-color: {self.colors['accent_lavender']};
                color: {self.colors['bg_main']};
            }}
        """)
        
        self.figure = Figure(figsize=(5, 4), facecolor=self.colors['bg_secondary'])
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.tabs.addTab(tab, "📊 Эксперименты")
    
    def start_experiments(self):
        bits_list = []
        if self.bits_8.isChecked():
            bits_list.append(8)
        if self.bits_12.isChecked():
            bits_list.append(12)
        if self.bits_16.isChecked():
            bits_list.append(16)
        
        if not bits_list:
            QMessageBox.warning(self, "Ошибка", "Выберите хотя бы одну длину хеша")
            return
        
        self.exp_btn.setEnabled(False)
        self.exp_cancel_btn.setEnabled(True)
        self.exp_progress.setVisible(True)
        self.results_table.setRowCount(0)
        
        self.exp_worker = ExperimentsWorker(bits_list, self.exp_count.value(), self.exp_attempts.value())
        self.exp_worker.progress.connect(self.update_exp_progress)
        self.exp_worker.status.connect(self.exp_status.setText)
        self.exp_worker.finished.connect(self.on_experiments_complete)
        self.exp_worker.start()
    
    def cancel_experiments(self):
        if self.exp_worker:
            self.exp_worker.cancel()
    
    def update_exp_progress(self, current, total):
        self.exp_progress.setRange(0, total)
        self.exp_progress.setValue(current)
    
    def on_experiments_complete(self, results, error):
        self.exp_progress.setVisible(False)
        self.exp_btn.setEnabled(True)
        self.exp_cancel_btn.setEnabled(False)
        
        if error:
            self.exp_status.setText(f"Ошибка: {error}")
            return
        
        row = 0
        bits_data = []
        practical = []
        theoretical = []
        
        for bits, exp_results in results.items():
            successful = [v for v in exp_results.values() if v is not None]
            finder = ShortHashCollisionFinder(trunc_bits=bits)
            theory = finder.theoretical_expected_attempts()
            
            avg = sum(successful) / len(successful) if successful else 0
            deviation = ((avg - theory) / theory * 100) if theory > 0 and avg > 0 else 0
            
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(str(bits)))
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{len(successful)}/{len(exp_results)}"))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{avg:.0f}" if avg > 0 else "Нет"))
            self.results_table.setItem(row, 3, QTableWidgetItem(str(theory)))
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{deviation:+.1f}%" if avg > 0 else "N/A"))
            
            bits_data.append(bits)
            practical.append(avg if avg > 0 else 0)
            theoretical.append(theory)
            row += 1
        
        # График в нежных тонах
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(self.colors['bg_tertiary'])
        self.figure.patch.set_facecolor(self.colors['bg_secondary'])
        
        x_pos = range(len(bits_data))
        bars1 = ax.bar([i - 0.2 for i in x_pos], practical, 0.4, 
                       label='Практика', color=self.colors['accent_mint'], alpha=0.8)
        bars2 = ax.bar([i + 0.2 for i in x_pos], theoretical, 0.4,
                       label='Теория', color=self.colors['accent_lavender'], alpha=0.8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{b} бит" for b in bits_data])
        ax.set_ylabel('Количество попыток', color=self.colors['text_light'])
        ax.set_title('Сравнение практических и теоретических результатов\n(атака "дней рождения")', 
                    color=self.colors['accent_soft_mint'])
        ax.legend(facecolor=self.colors['bg_secondary'], labelcolor=self.colors['text_light'])
        ax.grid(True, alpha=0.2, color=self.colors['border'])
        ax.tick_params(colors=self.colors['text_muted'])
        
        # Добавление значений на столбцы
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9,
                           color=self.colors['text_light'])
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9,
                           color=self.colors['text_light'])
        
        self.canvas.draw()
        self.exp_status.setText("Эксперименты завершены")


def run_gui():
    app = QApplication(sys.argv)
    window = CollisionGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()