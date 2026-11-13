import sys
import time
from typing import Optional

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QComboBox, QSpinBox, QPushButton, QTableWidget,
        QTableWidgetItem, QHeaderView, QProgressBar, QTextEdit, QGroupBox,
        QSplitter, QStatusBar, QMessageBox, QFrame, QTabWidget,
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QFont, QColor, QPalette, QLinearGradient, QBrush
except ImportError:
    print("Ошибка: PyQt6 не установлен. Выполните: pip install PyQt6")
    sys.exit(1)

from config_loader import (
    EXCELLENT_AVALANCHE_PERCENT,
    GUI_COLORS,
    GUI_DEFAULT_TEXT,
    GUI_MIN_HEIGHT,
    GUI_MIN_WIDTH,
    GUI_SPLITTER_SIZES,
    GUI_WINDOW_TITLE,
    HASH_PREVIEW_LENGTH,
    IDEAL_AVALANCHE_PERCENT,
    MAX_EXPERIMENT_COUNT,
    MIN_EXPERIMENT_COUNT,
    MODERATE_AVALANCHE_PERCENT,
    MODIFICATION_TYPES,
    SUPPORTED_ALGORITHMS,
    WARNING_AVALANCHE_PERCENT,
    TABLE_HASH_PREVIEW_LENGTH,
    TEXT_PREVIEW_LENGTH,
)
from hash_core import (
    AvalancheResult,
    compute_hash,
    get_avalanche_quality,
    run_experiments,
    summarize_results,
)



class ExperimentWorker(QThread):
    progress = pyqtSignal(int, int)         
    result_ready = pyqtSignal(list)       
    error_occurred = pyqtSignal(str)

    def __init__(self, text: str, count: int, algorithm: str):
        super().__init__()
        self.text = text
        self.count = count
        self.algorithm = algorithm

    def run(self):
        try:
            results = run_experiments(
                self.text,
                count=self.count,
                algorithm=self.algorithm,
                progress_callback=lambda c, t: self.progress.emit(c, t),
            )
            self.result_ready.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))



DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Consolas', 'Courier New', monospace;
}

QGroupBox {
    border: 1px solid #16213e;
    border-radius: 6px;
    margin-top: 8px;
    padding: 8px;
    font-size: 11px;
    font-weight: bold;
    color: #4fc3f7;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}

QLineEdit, QComboBox, QSpinBox {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 4px;
    padding: 5px 8px;
    color: #e0e0e0;
    font-size: 12px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1px solid #4fc3f7;
}

QPushButton {
    background-color: #0f3460;
    color: #e0e0e0;
    border: 1px solid #4fc3f7;
    border-radius: 5px;
    padding: 7px 20px;
    font-size: 12px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #1a5276;
}
QPushButton:pressed {
    background-color: #4fc3f7;
    color: #1a1a2e;
}
QPushButton:disabled {
    background-color: #2c2c3e;
    color: #666;
    border-color: #444;
}

QProgressBar {
    border: 1px solid #0f3460;
    border-radius: 4px;
    background-color: #16213e;
    text-align: center;
    color: #e0e0e0;
    height: 18px;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #0f3460, stop:1 #4fc3f7);
    border-radius: 3px;
}

QTableWidget {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 4px;
    gridline-color: #0f3460;
    color: #e0e0e0;
    font-size: 11px;
}
QTableWidget::item:selected {
    background-color: #1a5276;
}
QHeaderView::section {
    background-color: #0f3460;
    color: #4fc3f7;
    padding: 4px;
    border: 1px solid #16213e;
    font-size: 11px;
    font-weight: bold;
}

QTextEdit {
    background-color: #0d0d1a;
    border: 1px solid #0f3460;
    border-radius: 4px;
    padding: 6px;
    color: #4fc3f7;
    font-family: 'Consolas', monospace;
    font-size: 11px;
}

QTabWidget::pane {
    border: 1px solid #0f3460;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #16213e;
    color: #aaa;
    padding: 6px 16px;
    border-radius: 4px 4px 0 0;
}
QTabBar::tab:selected {
    background-color: #0f3460;
    color: #4fc3f7;
}

QStatusBar {
    background-color: #0f3460;
    color: #aaa;
    font-size: 10px;
}

QSplitter::handle {
    background-color: #0f3460;
    width: 3px;
}

QLabel#title {
    font-size: 18px;
    font-weight: bold;
    color: #4fc3f7;
}
QLabel#subtitle {
    font-size: 11px;
    color: #888;
}
QLabel#hashValue {
    font-family: 'Consolas', monospace;
    font-size: 10px;
    color: #a8d8a8;
    padding: 3px;
    background-color: #0d0d1a;
    border-radius: 3px;
}
"""



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(GUI_WINDOW_TITLE)
        self.setMinimumSize(GUI_MIN_WIDTH, GUI_MIN_HEIGHT)
        self.worker: Optional[ExperimentWorker] = None
        self._results: list[AvalancheResult] = []
        self._setup_ui()
        self.setStyleSheet(DARK_STYLE)

    # ── UI ─────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(12, 10, 12, 10)

        # Заголовок
        header = QVBoxLayout()
        title = QLabel("Лавинный эффект хеш-функций")
        title.setObjectName("title")
        subtitle = QLabel(f"Лабораторная работа №4 · {' / '.join(a.upper() for a in SUPPORTED_ALGORITHMS)}")
        subtitle.setObjectName("subtitle")
        header.addWidget(title)
        header.addWidget(subtitle)
        root.addLayout(header)

        # Разделитель
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #0f3460;")
        root.addWidget(line)

        # Основная часть
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Левая панель: параметры + хеш ──────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(8)

        # Параметры
        params_group = QGroupBox("⚙  Параметры эксперимента")
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(6)

        params_layout.addWidget(QLabel("Исходная строка:"))
        self.text_input = QLineEdit(GUI_DEFAULT_TEXT)
        self.text_input.setPlaceholderText("Введите текст для хеширования…")
        self.text_input.textChanged.connect(self._update_hash_preview)
        params_layout.addWidget(self.text_input)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Алгоритм:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(SUPPORTED_ALGORITHMS)
        self.algo_combo.currentTextChanged.connect(self._update_hash_preview)
        row1.addWidget(self.algo_combo)
        params_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Экспериментов:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(MIN_EXPERIMENT_COUNT, MAX_EXPERIMENT_COUNT)
        self.count_spin.setValue(MIN_EXPERIMENT_COUNT * 10)
        row2.addWidget(self.count_spin)
        params_layout.addLayout(row2)

        left_layout.addWidget(params_group)

        # Хеш-превью
        hash_group = QGroupBox("Хеш исходной строки")
        hash_layout = QVBoxLayout(hash_group)
        self.hash_preview = QLabel("—")
        self.hash_preview.setObjectName("hashValue")
        self.hash_preview.setWordWrap(True)
        hash_layout.addWidget(self.hash_preview)
        left_layout.addWidget(hash_group)

        # Кнопки
        self.run_btn = QPushButton("Запустить эксперименты")
        self.run_btn.clicked.connect(self._run_experiments)
        left_layout.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        self.cancel_btn = QPushButton("Остановить")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel_experiments)
        left_layout.addWidget(self.cancel_btn)

        left_layout.addStretch()

        #Правая панель: результаты
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(8)

        tabs = QTabWidget()

        # Таблица результатов
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "№", "Тип", "Изменённый текст", "Хеш до", "Хеш после", "% разл."
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table_layout.addWidget(self.results_table)
        tabs.addTab(table_widget, "Таблица")

        # Статистика
        stat_widget = QWidget()
        stat_layout = QVBoxLayout(stat_widget)
        self.stat_text = QTextEdit()
        self.stat_text.setReadOnly(True)
        self.stat_text.setPlaceholderText("Здесь появится статистика после запуска…")
        stat_layout.addWidget(self.stat_text)
        tabs.addTab(stat_widget, "Статистика")

        right_layout.addWidget(tabs)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes(GUI_SPLITTER_SIZES)

        root.addWidget(splitter)

        # Статус-бар
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Готово к работе")

        self._update_hash_preview()


    def _update_hash_preview(self):
        text = self.text_input.text()
        algo = self.algo_combo.currentText()
        if not text:
            self.hash_preview.setText("—")
            return
        try:
            h = compute_hash(text, algo)
            # Переносим для читаемости
            self.hash_preview.setText(h[:HASH_PREVIEW_LENGTH] + "\n" + h[HASH_PREVIEW_LENGTH:])
        except Exception as e:
            self.hash_preview.setText(f"Ошибка: {e}")

    def _run_experiments(self):
        text = self.text_input.text().strip()
        if not text:
            QMessageBox.warning(self, "Ошибка", "Введите исходную строку.")
            return

        count = self.count_spin.value()
        algo = self.algo_combo.currentText()

        self.run_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.progress_bar.setVisible(True)
        total_experiments = count * len(MODIFICATION_TYPES)
        self.progress_bar.setRange(0, total_experiments)
        self.progress_bar.setValue(0)
        self.results_table.setRowCount(0)
        self.stat_text.clear()
        self._results = []
        self.status.showMessage(f"Запуск {total_experiments} экспериментов…")

        self.worker = ExperimentWorker(text, count, algo)
        self.worker.progress.connect(self._on_progress)
        self.worker.result_ready.connect(self._on_results)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()

    def _cancel_experiments(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self._finish_ui()
            self.status.showMessage("Остановлено пользователем")

    def _on_progress(self, current: int, total: int):
        self.progress_bar.setValue(current)
        self.status.showMessage(f"Эксперимент {current} из {total}…")

    def _on_results(self, results: list):
        self._results = results
        self._populate_table(results)
        self._populate_stats(results)
        self._finish_ui()
        self.status.showMessage(
            f"Готово — {len(results)} экспериментов завершено"
        )

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "Ошибка выполнения", msg)
        self._finish_ui()
        self.status.showMessage("Ошибка!")

    def _finish_ui(self):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.progress_bar.setVisible(False)
        if self.worker:
            self.worker = None

    def _populate_table(self, results: list[AvalancheResult]):
        self.results_table.setRowCount(len(results))
        for row, r in enumerate(results):
            items = [
                str(row + 1),
                r.modification_type,
                repr(r.modified_text)[:TEXT_PREVIEW_LENGTH],
                r.modified_hash[:TABLE_HASH_PREVIEW_LENGTH] + "…",
                r.original_hash[:TABLE_HASH_PREVIEW_LENGTH] + "…",
                f"{r.diff_percent:.1f}%",
            ]
            for col, val in enumerate(items):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                match r.diff_percent:
                    case p if p >= EXCELLENT_AVALANCHE_PERCENT:
                        item.setForeground(QColor(GUI_COLORS["excellent"]))
                    case p if p >= WARNING_AVALANCHE_PERCENT:
                        item.setForeground(QColor(GUI_COLORS["warning"]))
                    case _:
                        item.setForeground(QColor(GUI_COLORS["weak"]))
                self.results_table.setItem(row, col, item)

    def _populate_stats(self, results: list[AvalancheResult]):
        s = summarize_results(results)
        if not s:
            return

        lines = [
            "╔══════════════════════════════════════════╗",
            "║         СВОДНАЯ СТАТИСТИКА               ║",
            "╚══════════════════════════════════════════╝",
            "",
            f"  Всего экспериментов : {s['total_experiments']}",
            f"  Среднее % различий  : {s['avg_diff_percent']:.2f}%",
            f"  Минимум             : {s['min_diff_percent']:.2f}%",
            f"  Максимум            : {s['max_diff_percent']:.2f}%",
            f"  Средн. изм. бит     : {s['avg_changed_bits']} / {s['total_bits']}",
            "",
            "  По типу модификации:",
            "  " + "─" * 44,
        ]
        for mod, pct in s["by_modification"].items():
            lines.append(f"  {mod:<32} {pct:.2f}%")

        lines += [
            "",
            "  " + "─" * 44,
        ]
        quality = get_avalanche_quality(s["avg_diff_percent"])
        match quality.level:
            case "excellent":
                verdict = (
                    f"ОТЛИЧНЫЙ лавинный эффект (~{IDEAL_AVALANCHE_PERCENT}%) "
                    f"- {quality.description}"
                )
            case "moderate":
                verdict = f"УМЕРЕННЫЙ лавинный эффект - {quality.description}"
            case _:
                verdict = f"СЛАБЫЙ лавинный эффект - {quality.description}"
        lines.append(f"  Вывод: {verdict}")
        lines.append("")
        lines.append("  Теория: для идеальной хеш-функции изменение одного")
        lines.append("  бита входа должно изменять ~50% выходных бит.")

        self.stat_text.setPlainText("\n".join(lines))



def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Критическая ошибка: {e}", file=sys.stderr)
        sys.exit(1)
