import os
import sys
from hash import *

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QFileDialog, QMessageBox, QTabWidget
    )
from PyQt6.QtCore import Qt


class HashWindow(QMainWindow):
    "Конструктор окна: задаёт заголовок, размер, создаёт вкладки"
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Хеш-функции (SHA-256)")
        self.setGeometry(100, 100, 800, 500)
        
        central = QWidget()
        self.setCentralWidget(central)
        
        tabs = QTabWidget()
        tabs.addTab(self.create_hash_tab(), "Вычислить хеш")
        tabs.addTab(self.create_check_tab(), "Проверить целостность")
        tabs.addTab(self.create_avalanche_tab(), "Лавинный эффект")
        
        layout = QVBoxLayout(central)
        layout.addWidget(tabs)
    
    def create_hash_tab(self):
        "Создаёт вкладку «Вычислить хеш»"
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        layout.addWidget(QLabel("Текст или выберите файл:"))
        self.input_text = QTextEdit()
        layout.addWidget(self.input_text)
        
        btn_layout = QHBoxLayout()
        btn_file = QPushButton("Выбрать файл")
        btn_file.clicked.connect(self.load_file)
        btn_hash = QPushButton("Вычислить хеш")
        btn_hash.clicked.connect(self.calc_hash)
        btn_layout.addWidget(btn_file)
        btn_layout.addWidget(btn_hash)
        layout.addLayout(btn_layout)
        
        btn_save_layout = QHBoxLayout()
        btn_save = QPushButton("Сохранить хеш в файл")
        btn_save.clicked.connect(self.save_hash_to_file)
        btn_save_layout.addWidget(btn_save)
        layout.addLayout(btn_save_layout)

        layout.addWidget(QLabel("Результат (SHA-256):"))
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)
        
        return widget
    
    def create_check_tab(self):
        "Создаёт вкладку «Проверить целостность»."
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        layout.addWidget(QLabel("Файл для проверки:"))
        self.check_file = QTextEdit()
        layout.addWidget(self.check_file)
        
        btn_file = QPushButton("Выбрать файл")
        btn_file.clicked.connect(self.select_check_file)
        layout.addWidget(btn_file)
        
        layout.addWidget(QLabel("Ожидаемый хеш:"))
        self.check_hash = QTextEdit()
        layout.addWidget(self.check_hash)
        
        btn_verify = QPushButton("Проверить")
        btn_verify.clicked.connect(self.verify)
        layout.addWidget(btn_verify)
        
        self.check_result = QLabel("")
        layout.addWidget(self.check_result)
        
        return widget
    
    def create_avalanche_tab(self):
        "Создаёт вкладку «Лавинный эффект»."
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        layout.addWidget(QLabel("Текст 1:"))
        self.av_text1 = QTextEdit()
        layout.addWidget(self.av_text1)
        
        layout.addWidget(QLabel("Текст 2:"))
        self.av_text2 = QTextEdit()
        layout.addWidget(self.av_text2)
        
        btn_comp = QPushButton("Сравнить")
        btn_comp.clicked.connect(self.compare_avalanche)
        layout.addWidget(btn_comp)
        
        self.av_result = QTextEdit()
        self.av_result.setReadOnly(True)
        layout.addWidget(self.av_result)
        
        return widget
    
    def load_file(self):
        "Открывает диалог выбора файла для чтения"
        filepath, _ = QFileDialog.getOpenFileName(self, "Выберите файл")
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.input_text.setText(f.read())
            except:
                QMessageBox.warning(self, "Ошибка", "Не удалось прочитать файл")
    
    def calc_hash(self):
        "Вычисляет хеш"
        text = self.input_text.toPlainText()
        if text:
            h = hash_text(text)
            self.result_text.setText(h)
        else:
            QMessageBox.warning(self, "Ошибка", "Введите текст или выберите файл")
    
    def select_check_file(self):
        "Открывает диалог выбора файла для записи"
        filepath, _ = QFileDialog.getOpenFileName(self, "Выберите файл")
        if filepath:
            self.check_file.setText(filepath)
    
    def save_hash_to_file(self):
        "Сохраняет вычисленный хеш в файл"
        hash_value = self.result_text.toPlainText().strip()
        if not hash_value:
            QMessageBox.warning(self, "Ошибка", "Нет хеша для сохранения. Сначала вычислите хеш.")
            return
    
        filepath, _ = QFileDialog.getSaveFileName(self, "Сохранить хеш", "", "Хеш файлы (*.sha256);;Все файлы (*)")
        if filepath:
            if save_hash(hash_value, filepath):
                QMessageBox.information(self, "Успех", f"Хеш сохранён в {filepath}")
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось сохранить хеш")

    def verify(self):
        "Проверяет соответсвие ожидаемого хеша"
        filepath = self.check_file.toPlainText().strip()
        expected = self.check_hash.toPlainText().strip()
        
        if not filepath or not expected:
            QMessageBox.warning(self, "Ошибка", "Укажите файл и ожидаемый хеш")
            return
        
        if check_integrity(filepath, expected):
            self.check_result.setText("✔ Файл не изменён")
            self.check_result.setStyleSheet("color: green")
        else:
            self.check_result.setText("❌ Файл изменён")
            self.check_result.setStyleSheet("color: red")
    
    def compare_avalanche(self):
        "Проверка лавинного эффекта"
        t1 = self.av_text1.toPlainText()
        t2 = self.av_text2.toPlainText()
        
        if not t1 or not t2:
            QMessageBox.warning(self, "Ошибка", "Введите оба текста")
            return
        
        h1, h2, diff, percent = avalanche(t1, t2)
        self.av_result.setText(
            f"Хеш 1: {h1}\n\n"
            f"Хеш 2: {h2}\n\n"
            f"Различается бит: {diff} из 256 ({percent:.1f}%)"
        )


def run_gui():
    app = QApplication(sys.argv)
    window = HashWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()