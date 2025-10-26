# Стандартные библиотеки
import re
import sys

# Сторонние библиотеки
import docx
import pymorphy2
from PyQt5 import QtCore, QtWidgets, QtMultimedia
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QWidget,
)


SCREEN_SIZE = [300, 300, 335, 300]


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Text manager Instruction')
        self.load_mp3('media.mp3')
        self.playBtn = QPushButton('Воспроизвести', self)
        self.playBtn.move(70, 180)
        self.playBtn.clicked.connect(self.player.play)
        self.pauseBtn = QPushButton('Пауза', self)
        self.pauseBtn.move(210, 180)
        self.pauseBtn.clicked.connect(self.player.pause)
        self.stopBtn = QPushButton('Стоп', self)
        self.stopBtn.move(70, 220)
        self.stopBtn.clicked.connect(self.player.stop)
        self.btn = QPushButton('Дальше', self)
        self.btn.move(210, 220)
        self.btn.clicked.connect(self.open_types_of_files_form)
        self.label = QLabel(self)
        self.label.move(40, 30)
        self.text_label = QLabel(self)
        self.text_label.setFont(QFont('Times New Roman', 10))
        welcome_text = (
            "Вас приветствует Text manager - это программа для анализа "
            "содержания текста и поиска ответа на заданный вами вопрос."
        )
        self.text_label.setText(welcome_text)
        self.text_label.move(10, 40)

    def open_types_of_files_form(self):
        self.types_of_files_form = TypesOfFilesForm(
            self, "Данные для второй формы")
        self.types_of_files_form.show()

    def load_mp3(self, filename):
        media = QtCore.QUrl.fromLocalFile(filename)
        content = QtMultimedia.QMediaContent(media)
        self.player = QtMultimedia.QMediaPlayer()
        self.player.setMedia(content)


class TypesOfFilesForm(QWidget):
    def __init__(self, *args):
        super().__init__()
        self.init_ui(args)

    def init_ui(self, args):
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Text manager')
        self.btn_word_file = QPushButton(self)
        self.btn_word_file.clicked.connect(self.word_form)
        self.btn_word_file.setIcon(QIcon('word.png'))
        self.btn_word_file.move(170, 140)
        self.btn_word_file.setIconSize(QSize(100, 100))
        self.btn_text_file = QPushButton(self)
        self.btn_text_file.clicked.connect(self.open_text_form)
        self.btn_text_file.setIcon(QIcon('text.png'))
        self.btn_text_file.move(30, 140)
        self.btn_text_file.setIconSize(QSize(100, 100))
        self.name_label = QLabel(self)
        self.name_label.setText("Выберите формат файла,")
        self.name_label.move(60, 90)
        self.name_label.setFont(QFont('Times New Roman', 10))
        self.show()

    def word_form(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Load file', '', "Word File (*.docx)")
        self.text_input = ''
        if filename:
            doc = docx.Document(filename)
            for par in doc.paragraphs:
                self.text_input += par.text
            self.open_question_form()
        else:
            QMessageBox.warning(self, 'Error', "Файл не выбран.")

    def open_question_form(self):
        self.question_form = QuestionForm(self, "", self.text_input)
        self.question_form.show()

    def open_text_form(self):
        self.text_form = TextForm(self, "")
        self.text_form.show()


class TextForm(QWidget):
    def __init__(self, *args):
        super().__init__()
        self.init_ui(args)

    def init_ui(self, args):
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Text manager Text file')
        self.btn_download1 = QPushButton('Отправить', self)
        self.btn_download1.move(100, 150)
        self.btn_download1.clicked.connect(self.open_question_form)
        self.label = QLabel(self)
        self.label.move(40, 30)
        self.text_label = QLabel(self)
        self.text_label.setText("Пожалуйста введите текст.")
        self.text_label.move(100, 90)
        self.text_input = QLineEdit(self)
        self.text_input.move(100, 110)

    def open_question_form(self):
        self.question_form = QuestionForm(self, "", self.text_input.text())
        self.question_form.show()


class QuestionForm(QWidget):
    def __init__(self, *args):
        super().__init__()
        self.text1 = str(args[2])
        self.init_ui(args)

    def init_ui(self, args):
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Question Form')
        self.btn_download2 = QPushButton('Отправить', self)
        self.btn_download2.move(100, 150)
        self.btn_download2.clicked.connect(self.open_analysis_form)
        self.label = QLabel(self)
        self.label.move(40, 30)
        self.text_label = QLabel(self)
        self.text_label.setText("Пожалуйста введите текст вопроса.")
        self.text_label.move(70, 90)
        self.question_input = QLineEdit(self)
        self.question_input.move(100, 110)

    def open_analysis_form(self):
        self.question = self.question_input.text()
        self.analysis_form = AnalysisForm(self, "", self.text1, self.question)
        self.analysis_form.show()


class AnalysisForm(QMainWindow):
    def __init__(self, *args):
        super().__init__()
        self.text_analysis(args)

    def text_analysis(self, args):
        text, question = args[2], args[3]
        morph = pymorphy2.MorphAnalyzer()
        text = text.replace(',', '')
        analysis_text, analysis_question, suggestions = [], [], []
        split_regex = re.compile(r'[.|!|?|…]')
        for content_type, content in enumerate([text, question]):
            sentences = filter(
                None, [s.strip() for s in split_regex.split(content)])
            current_analysis = []
            for s in sentences:
                suggestions.append(s)
                words_analysis = []
                for t in s.split():
                    res = morph.parse(t)[0]
                    exclude_tags = ["CONJ", "NPRO", "PREP", "PRCL"]
                    if all(tag not in res.tag for tag in exclude_tags):
                        words_analysis.append(res.normal_form.lower())
                current_analysis.append(words_analysis)
            if content_type == 0:
                analysis_text = current_analysis
            else:
                analysis_question = [
                    word for sent in current_analysis for word in sent
                ]

        numbers = [
            analysis_text.index(i) for i in analysis_text
            for j in i if j in analysis_question
        ]
        numbers1 = [i for i in numbers if numbers.count(i) >= 2]
        numbers2 = sorted(list(set(numbers1)))
        self.text_output = [suggestions[i] for i in numbers2]

        with open("text manager answer.txt", 'w') as f:
            for item in self.text_output:
                f.write(item + '\n')

        if not self.text_output:
            self.open_bad_result_form()
        else:
            self.open_result_form()

    def open_result_form(self):
        self.result_form = ResultForm(self, "", self.text_output)
        self.result_form.show()

    def open_bad_result_form(self):
        self.bad_result_form = BadResultForm(self, "")
        self.bad_result_form.show()


class ResultForm(QMainWindow):
    def __init__(self, *args):
        super().__init__()
        self.text_output = args[2]
        self.init_ui(args)

    def init_ui(self, args):
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle('Text manager Good Answer')
        self.centralwidget = QtWidgets.QWidget(self)
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(0, 30, 600, 500))
        self.listWidget.addItems(self.text_output)
        self.setCentralWidget(self.centralwidget)


class BadResultForm(QWidget):
    def __init__(self, *args):
        super().__init__()
        self.init_ui(args)

    def init_ui(self, args):
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Text manager Bad Answer')
        self.text_label1 = QLabel(self)
        self.text_label1.setText(
            "К сожалению результатов по вашему запросу не найдено.")
        self.text_label1.move(20, 90)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec())