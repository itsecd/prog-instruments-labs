import docx
import pymorphy2
import re
import sys

from PyQt5 import QtCore, QtMultimedia, QtWidgets
from PyQt5.QtCore import QSize, QUrl
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QWidget
)

SCREEN_SIZE = [300, 300, 335, 300]


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Text manager Instruction')

        # Прикрепление аудио файла
        self.load_mp3('media.mp3')

        # Создание кнопки для воспроизведения аудио файла
        self.playBtn = QPushButton('Воспроизвести', self)
        self.playBtn.resize(self.playBtn.sizeHint())
        self.playBtn.move(70, 180)
        self.playBtn.clicked.connect(self.player.play)

        # Создание кнопки для приостановки аудио файла
        self.pauseBtn = QPushButton('Пауза', self)
        self.pauseBtn.resize(self.pauseBtn.sizeHint())
        self.pauseBtn.move(210, 180)
        self.pauseBtn.clicked.connect(self.player.pause)

        # Создание кнопки для выключения аудио файла
        self.stopBtn = QPushButton('Стоп', self)
        self.stopBtn.resize(self.stopBtn.sizeHint())
        self.stopBtn.move(70, 220)
        self.stopBtn.clicked.connect(self.player.stop)

        # Создание кнопки для перехода на следующую страницу
        self.btn = QPushButton('Дальше', self)
        self.btn.resize(self.btn.sizeHint())
        self.btn.move(210, 220)
        self.btn.clicked.connect(self.open_types_of_files_form)

        self.label = QLabel(self)
        self.label.move(40, 30)
        self.text_label = QLabel(self)
        self.text_label.setFont(QFont('Times New Roman', 10))
        self.text_label.setText("Вас приветствует Text manager.")
        self.text_label.move(70, 20)
        self.text_label = QLabel(self)
        self.text_label.setFont(QFont('Times New Roman', 10))
        self.text_label.setText("Text manager - это программа для анализа содержания")
        self.text_label.move(10, 40)
        self.text_label = QLabel(self)
        self.text_label.setFont(QFont('Times New Roman', 10))
        self.text_label.setText("текста и поиска ответа на заданный вами вопрос.")
        self.text_label.move(10, 60)
        self.text_label = QLabel(self)
        self.text_label.setFont(QFont('Times New Roman', 10))
        self.text_label.setText("Чтобы получше познакомится с функциями Text manager,")
        self.text_label.move(10, 80)
        self.text_label = QLabel(self)
        self.text_label.setFont(QFont('Times New Roman', 10))
        self.text_label.setText(" вы можите прослушать аудиоинструкцию,")
        self.text_label.move(40, 100)
        self.text_label = QLabel(self)
        self.text_label.setFont(QFont('Times New Roman', 10))
        self.text_label.setText(" нажав на кнопку 'Воспроизвести'.")
        self.text_label.move(60, 120)

    def open_types_of_files_form(self):
        self.types_of_files_form = TypesOfFilesForm(self, "Данные для второй формы")
        self.types_of_files_form.show()

    def load_mp3(self, filename):
        media = QtCore.QUrl.fromLocalFile(filename)
        content = QtMultimedia.QMediaContent(media)
        self.player = QtMultimedia.QMediaPlayer()
        self.player.setMedia(content)


class TypesOfFilesForm(QWidget):
    def __init__(self, *args):
        super().__init__()
        self.initUI(args)

    def initUI(self, args):
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Text manager')

        # Создание кнопки для прикрепления word файла
        self.btn_word_file = QPushButton(self)
        self.btn_word_file.clicked.connect(self.wordForm)
        self.btn_word_file.setIcon(QIcon('word_file.jpg'))
        self.btn_word_file.move(170, 140)
        self.btn_word_file.setIconSize(QSize(100, 100))

        # Создание кнопки для прикрепления text файла
        self.btn_text_file = QPushButton(self)
        self.btn_text_file.clicked.connect(self.open_text_form)
        self.btn_text_file.setIcon(QIcon('text_file.jpg'))
        self.btn_text_file.move(30, 140)
        self.btn_text_file.setIconSize(QSize(100, 100))

        self.name_label = QLabel(self)
        self.name_label.setText("Выберите формат файла,")
        self.name_label.move(60, 90)
        self.name_label.setFont(QFont('Times New Roman', 10))

        self.name_label1 = QLabel(self)
        self.name_label1.setText("который хотите прикрепить.")
        self.name_label1.move(60, 105)
        self.name_label1.setFont(QFont('Times New Roman', 10))

        self.show()

    def wordForm(self):
        filename = QFileDialog.getOpenFileName(self, 'Load file', '', "Word File (*.docx)")
        name = filename[0]
        # Получение имени файла

        self.text_input = ''

        if filename:
            doc = docx.Document(name)
            for par in doc.paragraphs:
                self.text_input += par.text
        else:
            QMessageBox.warning(self, 'Error', "Файл не выбран.")
        self.open_question_form()

    def open_question_form(self):
        self.question_form = QuestionForm(self, "", self.text_input)
        self.question_form.show()

    def open_text_form(self):
        self.text_form = TextForm(self, "")
        self.text_form.show()


class TextForm(QWidget):
    def __init__(self, *args):
        super().__init__()
        self.initUI(args)
        self.length = ''

    def initUI(self, args):
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Text manager Text file')

        # Создание кнопки для загрузки введенного текста
        self.btn_download1 = QPushButton('Отправить', self)
        self.btn_download1.resize(self.btn_download1.sizeHint())
        self.btn_download1.move(100, 150)
        self.btn_download1.clicked.connect(self.open_question_form)

        self.label = QLabel(self)
        self.label.move(40, 30)

        self.text_label = QLabel(self)
        self.text_label.setText("Пожалуйста введите текст.")
        self.text_label.move(100, 90)
        self.text_label.setFont(QFont('Times New Roman', 10))

        self.text_input = QLineEdit(self)
        self.text_input.move(100, 110)

        self.lbl = QLabel(args[-1], self)
        self.lbl.adjustSize()

    def open_question_form(self):
        self.question_form = QuestionForm(self, "", self.text_input.text())
        self.question_form.show()


class QuestionForm(QWidget):
    def __init__(self, *args):
        super().__init__()
        self.initUI(args)
        self.length = ''
        self.text1 = ''

    def initUI(self, args):
        length = args[2]
        self.text1 = str(length)
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Question Form')

        # Создание кнопки для загрузки введенного текста
        self.btn_download2 = QPushButton('Отправить', self)
        self.btn_download2.resize(self.btn_download2.sizeHint())
        self.btn_download2.move(100, 150)

        # Если есть ответ на вопрос то есть длина списка с ответами не равно 0
        # тогда можно открывать класс вывода ответов иначе открыть класс
        # в котором программа говорит о том что совпадений нет
        self.btn_download2.clicked.connect(self.open_analysis_form)

        self.label = QLabel(self)
        self.label.move(40, 30)

        self.text_label = QLabel(self)
        self.text_label.setText("Пожалуйста введите текст вопроса.")
        self.text_label.move(100, 90)
        self.text_label.setFont(QFont('Times New Roman', 10))

        self.question_input = QLineEdit(self)
        self.question_input.move(100, 110)

        self.lbl = QLabel(args[-1], self)
        self.lbl.adjustSize()

    def open_analysis_form(self):
        self.question = self.question_input.text()
        self.analysis_form = AnalysisForm(self, "", self.lbl.text(), self.question)
        self.analysis_form.show()


class AnalysisForm(QMainWindow):
    def __init__(self, *args):
        super().__init__()
        self.text_analysis(args)
        self.text = ''
        self.question = ''

    def text_analysis(self, args):
        self.text = args[2]
        self.question = args[3]
        morph = pymorphy2.MorphAnalyzer()
        self.text = self.text.replace(',', '')
        a = []
        b = []
        c = [self.text, self.question]
        analysis_text = []
        analysis_question = []
        analysis_question1 = []
        suggestions = []

        # Разбиение введенного текста для анализа
        # и текста вопроса на предложения и слова путем создания вложенных списков
        split_regex = re.compile(r'[.|!|?|…]')
        for i in c:
            sentences = filter(lambda t: t, [t.strip() for t in split_regex.split(i)])
            for s in sentences:
                # Создание списка с предложениями для
                # дальнейшего вывода слов в той форме в которой они были изначально
                suggestions.append(s)
                g = s.split()
                for t in g:
                    res = morph.parse(t)[0]
                    # Во вложенные списки попадают только те части речи,
                    # которые не являются частицами, местоимениями, местоимениями, союзами, предлогами
                    if ("CONJ" not in res.tag) and ("NPRO" not in res.tag) and ("PREP" not in res.tag) and (
                            "PRCL" not in res.tag):
                        # Причем слова изменяются и попадают в список в начальной форме и в нижнем регистре
                        b.append((morph.parse(t)[0].normal_form).lower())
                        a.append(b)
                        b = []
                if i == self.text:
                    # Вложенный список с текстом для анализа
                    analysis_text.append(a)
                if i == self.question:
                    # Вложенный список с вопросами
                    # Можно будет добавить возможность ввода сразу нескольких вопросов вместо одного
                    analysis_question1.append(a)
                a = []
        numbers = []
        for i in analysis_question1:
            for j in i:
                analysis_question.append(''.join(j))

        # Сравниваются слова из введенного текста и слова из текста вопроса
        for i in analysis_text:
            for j in i:
                e = ''.join(j)
                for k in analysis_question:
                    if k == e:
                        # И добавляются индексы предложений в новый список
                        numbers.append(analysis_text.index(i))
                        continue
        numbers1 = []
        for i in numbers:
            # Если в предложении и вопросе 2 и больше одинаковых слов,
            # то индексы этих предложений попадают в новый список
            if numbers.count(i) >= 2:
                numbers1.append(i)
        numbers2 = []
        for i in numbers1:
            if i not in numbers2:
                numbers2.append(i)
        self.text_output = []
        for i in numbers2:
            text_output1 = (
                (((str((suggestions[int(i)]))).replace("['", "")).replace("'],", "")).replace("']]", "")).replace(
                "[", "")
            self.text_output.append(text_output1)

        # Запись результатов в файл
        f = open("text manager answer.txt", 'w')
        for i in self.text_output:
            f.write(i + '\n')
        f.close()
        f = open("text manager answer.txt", 'r')
        print(f.read())
        f.close()

        # Добавление предложений в список результатов
        if self.text_output == []:
            # Если результаты есть то открывается одна форма если нет то другая
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
        self.initUI(args)
        self.text_output = ""

    def initUI(self, args):
        self.text_output = args[2]
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle('Text manager Good Answer')
        self.pixmap = QPixmap('text.jpg')
        self.image = QLabel(self)
        self.image.move(30, 0)
        self.image.resize(400, 30)
        self.image.setPixmap(self.pixmap)
        self.centralwidget = QtWidgets.QWidget(self)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect())
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(0, 30, 600, 500))
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        self.setCentralWidget(self.centralwidget)
        self.listWidget.addItems(self.text_output)


class BadResultForm(QWidget):
    def __init__(self, *args):
        super().__init__()
        self.initUI(args)

    def initUI(self, args):
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Text manager Bad Answer')
        self.lbl = QLabel(args[-1], self)
        self.lbl.adjustSize()
        self.text_label1 = QLabel(self)
        self.text_label2 = QLabel(self)
        self.text_label3 = QLabel(self)
        self.text_label4 = QLabel(self)
        self.text_label1.setText("К сожалению результатов по вашему запросу ")
        self.text_label2.setText("не найдено. Попробуйте еще раз сформулировав")
        self.text_label3.setText("более точный вопрос или приложите больше")
        self.text_label4.setText("информации для поиска.")
        self.text_label1.move(40, 90)
        self.text_label2.move(40, 105)
        self.text_label3.move(40, 120)
        self.text_label4.move(40, 135)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec())