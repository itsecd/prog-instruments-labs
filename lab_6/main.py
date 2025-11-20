import os
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
							 QHBoxLayout, QPushButton, QLabel, QSpinBox,
							 QTextEdit, QMessageBox, QGroupBox, QLineEdit)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from time_test import PerformanceBenchmark
from hash_card import CardSearcher
from const import *
import lun


class CompactCardSearchApp(QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("Поиск номера карты и проверка Луна")
		self.setFixedSize(600, 600)

		self.central_widget = QWidget()
		self.setCentralWidget(self.central_widget)

		self.layout = QVBoxLayout()
		self.central_widget.setLayout(self.layout)

		self.init_ui()
		self.benchmark = None

	def init_ui(self):
		# Группа параметров поиска
		params_group = QGroupBox("Параметры поиска")
		params_layout = QVBoxLayout()

		self.hash_label = QLabel(f"Хеш: {CARD_HASH[:15]}...")
		self.hash_label.setToolTip(CARD_HASH)
		params_layout.addWidget(self.hash_label)

		self.digits_label = QLabel(f"Последние цифры: {LAST_4_DIGITS}")
		params_layout.addWidget(self.digits_label)

		self.bins_label = QLabel(f"BIN-коды: {len(BINS)} шт")
		self.bins_label.setToolTip(", ".join(BINS))
		params_layout.addWidget(self.bins_label)

		params_group.setLayout(params_layout)
		self.layout.addWidget(params_group)

		# Группа проверки Луна
		luhn_group = QGroupBox("Проверка алгоритма Луна")
		luhn_layout = QVBoxLayout()

		self.number_input = QLineEdit()
		self.number_input.setPlaceholderText("Введите номер карты")
		luhn_layout.addWidget(self.number_input)

		buttons_layout = QHBoxLayout()

		self.check_btn = QPushButton("Проверить")
		self.check_btn.clicked.connect(self.check_luhn)
		buttons_layout.addWidget(self.check_btn)

		self.compute_btn = QPushButton("Вычислить контрольную цифру")
		self.compute_btn.clicked.connect(self.compute_luhn)
		buttons_layout.addWidget(self.compute_btn)

		luhn_layout.addLayout(buttons_layout)

		self.luhn_result = QLabel()
		self.luhn_result.setAlignment(Qt.AlignCenter)
		luhn_layout.addWidget(self.luhn_result)

		luhn_group.setLayout(luhn_layout)
		self.layout.addWidget(luhn_group)

		# Кнопки действий
		buttons_layout = QHBoxLayout()

		self.benchmark_btn = QPushButton("Тест скорости")
		self.benchmark_btn.clicked.connect(self.run_benchmark)
		buttons_layout.addWidget(self.benchmark_btn)

		self.search_btn = QPushButton("Поиск")
		self.search_btn.clicked.connect(self.show_search_options)
		buttons_layout.addWidget(self.search_btn)

		self.layout.addLayout(buttons_layout)

		# Компактная область результатов
		self.result_area = QTextEdit()
		self.result_area.setMaximumHeight(150)
		self.result_area.setReadOnly(True)
		self.layout.addWidget(self.result_area)

		# Мини-график
		self.figure = Figure(figsize=(5, 3))
		self.canvas = FigureCanvas(self.figure)
		self.canvas.setMaximumHeight(200)
		self.canvas.setVisible(False)
		self.layout.addWidget(self.canvas)

		# Компактные настройки поиска
		self.search_group = QGroupBox("Настройки поиска")
		self.search_layout = QVBoxLayout()

		spin_layout = QHBoxLayout()
		spin_layout.addWidget(QLabel("Процессы:"))

		self.process_spin = QSpinBox()
		self.process_spin.setRange(1, os.cpu_count() * 2)
		self.process_spin.setValue(os.cpu_count())
		spin_layout.addWidget(self.process_spin)

		self.search_layout.addLayout(spin_layout)

		self.start_search_btn = QPushButton("Начать поиск")
		self.start_search_btn.clicked.connect(self.run_search)
		self.search_layout.addWidget(self.start_search_btn)

		self.search_group.setLayout(self.search_layout)
		self.search_group.setVisible(False)
		self.layout.addWidget(self.search_group)

	def check_luhn(self):
		number = self.number_input.text().strip()
		if not number:
			self.luhn_result.setText("Введите номер карты")
			self.luhn_result.setStyleSheet("color: red")
			return

		if not number.isdigit():
			self.luhn_result.setText("Номер должен содержать только цифры")
			self.luhn_result.setStyleSheet("color: red")
			return

		# Используем функции из модуля lun
		is_valid = lun.luhn_algorithm_check(number)
		if is_valid:
			self.luhn_result.setText("Номер корректен (алгоритм Луна)")
			self.luhn_result.setStyleSheet("color: green")
		else:
			self.luhn_result.setText("Номер НЕ корректен (алгоритм Луна)")
			self.luhn_result.setStyleSheet("color: red")

	def compute_luhn(self):
		number = self.number_input.text().strip()
		if not number:
			self.luhn_result.setText("Введите номер карты")
			self.luhn_result.setStyleSheet("color: red")
			return

		if not number.isdigit():
			self.luhn_result.setText("Номер должен содержать только цифры")
			self.luhn_result.setStyleSheet("color: red")
			return

		# Используем функции из модуля lun
		full_number = lun.luhn_algorithm_compute(number)
		self.luhn_result.setText(f"Полный номер с контрольной цифрой: {full_number}")
		self.luhn_result.setStyleSheet("color: blue")
		self.number_input.setText(full_number)

	def run_benchmark(self):
		self.clear_results()
		self.search_group.setVisible(False)
		self.canvas.setVisible(True)

		self.result_area.append("Тестирование скорости...")
		QApplication.processEvents()

		self.benchmark = PerformanceBenchmark(BINS)
		max_proc = min(24, int(os.cpu_count() * 1.5))
		self.benchmark.run_tests(max_proc)

		self.figure.clear()
		ax = self.figure.add_subplot(111)

		processes, durations = zip(*self.benchmark.results)
		ax.plot(processes, durations, 'bo-', linewidth=1, markersize=4)

		min_duration = min(durations)
		opt_process = processes[durations.index(min_duration)]
		ax.plot(opt_process, min_duration, 'ro', label=f'Оптимум: {opt_process}')

		ax.set_title("Скорость поиска", fontsize=10)
		ax.set_xlabel("Процессы", fontsize=8)
		ax.set_ylabel("Время (с)", fontsize=8)
		ax.tick_params(axis='both', which='major', labelsize=8)
		ax.grid(True, linestyle=':', alpha=0.7)
		ax.legend(fontsize=8)

		self.figure.tight_layout()
		self.canvas.draw()
		self.result_area.append(f"Оптимально: {opt_process} процессов")

	def show_search_options(self):
		self.clear_results()
		self.canvas.setVisible(False)
		self.search_group.setVisible(True)

	def run_search(self):
		self.clear_results()
		nproc = self.process_spin.value()

		self.result_area.append(f"Поиск с {nproc} процессами...")
		QApplication.processEvents()

		searcher = CardSearcher()
		result, duration = searcher.search_card_number(nproc)

		if result:
			self.result_area.append(f"\nНайден номер:\n{result}")
			self.result_area.append(f"Время: {duration:.2f} сек.")

			reply = QMessageBox.question(
				self, 'Сохранение', 'Сохранить результат?',
				QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
			)

			if reply == QMessageBox.Yes:
				searcher.save_result(result)
				self.result_area.append("(Сохранено в result.json)")
		else:
			self.result_area.append("\nНомер не найден")

	def clear_results(self):
		self.result_area.clear()

	def closeEvent(self, event):
		reply = QMessageBox.question(
			self, 'Выход', 'Закрыть программу?',
			QMessageBox.Yes | QMessageBox.No, QMessageBox.No
		)
		event.accept() if reply == QMessageBox.Yes else event.ignore()


def main():
	app = QApplication(sys.argv)
	app.setStyle('Fusion')
	window = CompactCardSearchApp()
	window.show()
	sys.exit(app.exec_())


if __name__ == "__main__":
	main()
