from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import QThread, QObject, Qt

from PyPDF2 import PdfMerger

import sys
import os
import typing
import time

import help_defenition as hd


class MergePDF(QThread):
    def __init__(self, parent: typing.Optional[QObject], to_merge: tuple) -> None:
        super().__init__(parent)

        self.list_file = to_merge[0]
        self.path_to_save = to_merge[1][0]
        self.path_to_get_origin = to_merge[1][1]

    def run(self):
        merger = PdfMerger()
        for file in self.list_file:
            if str(file)[-4:].find("pdf") != -1:
                merger.append(f"{self.path_to_get_origin}/{file}")
        merger.write(f"{self.path_to_save}/result_merger.pdf")
        self.parent().label_file_merged.setText(
            "Объединены в result_merger.pdf")
        self.parent().label_file_merged.adjustSize()


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()

        self.path_to_get_origin = "Не выбрано"
        self.path_to_save = ""
        self.file_was_checked = False
        self.list_file = []
        self.font_in_label = QFont("Times", 12)
        self.button_size_x = 280
        self.button_size_y = self.button_size_x / 4
        self.border_style = "border: 2px solid; border-radius: 10px;"

        while self.path_to_save == "":
            if not os.path.isfile("config.txt"):
                while self.path_to_save == "":
                    self.path_to_save = QtWidgets.QFileDialog.getExistingDirectory(
                        self, 'Выберите путь сохранения')
                    with open("config.txt", "w", encoding="utf-8") as f:
                        f.write(str(self.path_to_save))
            else:
                with open("config.txt", "r", encoding="utf-8") as f:
                    self.path_to_save = f.read()
            if self.path_to_save == "":
                os.remove("config.txt")

        self.setWindowTitle("Объединение PDF")
        self.setFixedSize(1000, 500)
        self.move(320, 180)

        self.label_path_to_save = QLabel(self)
        self.label_path_to_save.setFont(self.font_in_label)
        self.label_path_to_save.setText(
            f"Папка для сохранения: {self.path_to_save}")
        self.x_coord = int(340 - (len(self.path_to_save) + 22) / 2)
        if self.x_coord % 2 == 1:
            self.x_coord -= 1
        self.label_path_to_save.move(self.x_coord, 10)
        self.label_path_to_save.adjustSize()

        self.label_path_origin = QLabel(self)
        self.label_path_origin.setFont(self.font_in_label)
        self.label_path_origin.setText(
            f"Папка откуда брать: {self.path_to_get_origin}")
        self.x_coord = int(340 - (len(self.path_to_save) + 22) / 2)
        if self.x_coord % 2 == 1:
            self.x_coord -= 1
        self.label_path_origin.move(self.x_coord, 40)
        self.label_path_origin.adjustSize()

        self.label_list_file = QtWidgets.QListWidget(self)
        self.label_list_file.setFont(QFont("Times", 12))
        self.label_list_file.addItem(
            "\n\t     *здесь появиться список файлов*")
        self.label_list_file.setGeometry(100, 200, 450, 50)
        self.label_list_file.setStyleSheet(
            f"background-color: rgb(240, 230, 140); border-radius: 10px;")
        self.label_file_merged = QLabel(self)
        self.label_file_merged.setFont(self.font_in_label)
        self.label_file_merged.move(600, 80)

        self.lable_attention = QLabel(self)
        self.lable_attention.setFont(QFont("Times", 14))
        self.lable_attention.setText(
            '<p style="color: rgb(250, 55, 55);">!! Внимание !! В каком порядке расположены файлы, в таком они и будут в итоговом')
        self.lable_attention.move(10, 465)
        self.lable_attention.adjustSize()

        self.button_check_file = QtWidgets.QPushButton(self)
        self.button_check_file.setFont(self.font_in_label)
        self.button_check_file.setStyleSheet(
            f"background-image: url(image/Проверить_папку.png); {self.border_style}")
        self.button_check_file.setGeometry(
            100, 100, self.button_size_x, int(self.button_size_y))

        self.button_start_merge = QtWidgets.QPushButton(self)
        self.button_start_merge.setFont(self.font_in_label)
        self.button_start_merge.setStyleSheet(
            f"background-image: url(image/Объединить_ПДФ_2.png); {self.border_style}")
        self.button_start_merge.setGeometry(
            600, 100, self.button_size_x, int(self.button_size_y))

        self.button_repeat = QtWidgets.QPushButton(self)
        self.button_repeat.setFont(self.font_in_label)
        self.button_repeat.setStyleSheet(
            f"background-image: url(image/Сбросить_2.png); {self.border_style}")
        self.button_repeat.setGeometry(
            600, 180, self.button_size_x, int(self.button_size_y))

        self.button_set_new_path_to_save = QtWidgets.QPushButton(self)
        self.button_set_new_path_to_save.setFont(self.font_in_label)
        self.button_set_new_path_to_save.setStyleSheet(
            f"background-image: url(image/Новый_путь_для_сохранения.png); {self.border_style};")
        self.button_set_new_path_to_save.setGeometry(
            10, 5, self.button_size_x, 30)

        self.button_set_new_path_origin = QtWidgets.QPushButton(self)
        self.button_set_new_path_origin.setFont(self.font_in_label)
        self.button_set_new_path_origin.setStyleSheet(
            f"background-image: url(image/Изменить_папку_откуда_брать.png); {self.border_style};")
        self.button_set_new_path_origin.setGeometry(
            10, 36, self.button_size_x, 30)

        self.button_check_file.clicked.connect(self.check_file)
        self.button_start_merge.clicked.connect(self.start_merge)
        self.button_repeat.clicked.connect(self.clear_and_repeat)
        self.button_set_new_path_to_save.clicked.connect(
            self.set_new_path_to_save)
        self.button_set_new_path_origin.clicked.connect(self.set_path_origin)

    def check_file(self) -> None:
        if not self.path_to_get_origin == "Не выбрано":
            self.list_file = os.listdir(self.path_to_get_origin)
            if len(self.list_file) != 0 and hd.search_pdf_in_list(self.list_file) == True:
                i = 1
                if len(self.list_file) != 0 and hd.search_pdf_in_list(self.list_file) == True:
                    self.label_list_file.clear()
                    for elem in self.list_file:
                        if str(elem)[-4:].find("pdf") != -1:
                            self.label_list_file.addItem(f"{i}) {str(elem)}")
                            i += 1
                self.file_was_checked = True
                self.button_start_merge.setStyleSheet(
                    f"background-image: url(image/Объединить_ПДФ_1.png); {self.border_style}")
                self.button_repeat.setStyleSheet(
                    f"background-image: url(image/Сбросить_1.png); {self.border_style}")
            else:
                self.label_list_file.addItem("Папка с файлами пустая(")
            self.label_list_file.setGeometry(100, 200, 450, 250)
        else:
            pass

    def clear_and_repeat(self) -> None:
        if not self.path_to_get_origin == "Не выбрано":
            self.list_file.clear()
            self.button_start_merge.setStyleSheet(
                f"background-image: url(image/Объединить_ПДФ_2.png); {self.border_style}")
            self.button_repeat.setStyleSheet(
                f"background-image: url(image/Сбросить_2.png); {self.border_style}")
            self.label_list_file.setGeometry(100, 200, 450, 50)
            self.label_list_file.clear()
            self.label_file_merged.clear()
            self.label_list_file.addItem(
                "\n\t     *здесь появиться список файлов*")
            self.file_was_checked = False
        else:
            pass

    def set_new_path_to_save(self) -> None:
        self.path_to_save = ""
        while self.path_to_save == "":
            self.path_to_save = QtWidgets.QFileDialog.getExistingDirectory(
                self, 'Выберите новый путь сохранения')
            with open("config.txt", "w", encoding="utf-8") as f:
                f.write(str(self.path_to_save))
        self.label_path_to_save.setText(
            "Папка для сохранения: " + self.path_to_save)
        self.x_coord = int(350 - (len(self.path_to_save) + 22) / 2)
        self.label_path_to_save.adjustSize()

    def start_merge(self) -> None:
        if not self.path_to_get_origin == "Не выбрано":
            if self.file_was_checked:
                to_merge = (
                    self.list_file,
                    [
                        self.path_to_save,
                        self.path_to_get_origin
                    ]
                )
                self.merge_file = MergePDF(self, to_merge)
                self.merge_file.start()
        else:
            pass

    def set_path_origin(self) -> None:
        if self.file_was_checked == True:
            self.clear_and_repeat()
        self.path_to_get_origin = ""
        while self.path_to_get_origin == "Не выбрано" or self.path_to_get_origin == "":
            self.path_to_get_origin = QtWidgets.QFileDialog.getExistingDirectory(
                self, 'Выберите папку откуда взять файлы .pdf')
        self.label_path_origin.setText(
            "Папка для сохранения: " + self.path_to_get_origin)
        self.label_path_origin.adjustSize()


def application():
    app = QApplication(sys.argv)
    window = Window()
    window.setObjectName("MainWindow")
    window.setStyleSheet(
        "#MainWindow{border-image:url(image/background.png)}")
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    application()