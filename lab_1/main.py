"""
Лабораторная работа №1 по курсу "Технологии и методы программирования".
Приведение кода к стандартам PEP8.

Группа: 6311
Студент: Ладыгин Денис

Программа "Text Manager" для семантического анализа текста
и поиска ответов на вопросы пользователя.
"""

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
    QMainWindow,
    QMessageBox,
    QPushButton,
    QWidget,
)

# Константы
SCREEN_SIZE = [300, 300, 335, 300]


class Example(QWidget):
    """Начальное окно с инструкцией и аудио-плеером."""

    def __init__(self):
        super().__init__()
        # Объявление всех атрибутов экземпляра в __init__
        self.player = None
        self.play_btn = None
        self.pause_btn = None
        self.stop_btn = None
        self.next_btn = None
        self.info_label = None
        self.title_label = None
        self.types_of_files_form = None

        self.init_ui()

    def init_ui(self):
        """Инициализирует пользовательский интерфейс окна."""
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Text manager Instruction')
        self.load_mp3('media.mp3')

        self.play_btn = QPushButton('Воспроизвести', self)
        self.play_btn.move(70, 180)
        self.play_btn.clicked.connect(self.player.play)

        self.pause_btn = QPushButton('Пауза', self)
        self.pause_btn.move(210, 180)
        self.pause_btn.clicked.connect(self.player.pause)

        self.stop_btn = QPushButton('Стоп', self)
        self.stop_btn.move(70, 220)
        self.stop_btn.clicked.connect(self.player.stop)

        self.next_btn = QPushButton('Дальше', self)
        self.next_btn.move(210, 220)
        self.next_btn.clicked.connect(self.open_types_of_files_form)

        self.title_label = QLabel(self)
        self.title_label.setFont(QFont('Times New Roman', 10))
        welcome_text = (
            "Вас приветствует Text manager - программа для анализа "
            "содержания текста и поиска ответа на заданный вами вопрос."
        )
        self.title_label.setText(welcome_text)
        self.title_label.setWordWrap(True)  # Автоматический перенос строк
        self.title_label.setGeometry(10, 20, 315, 100)

    def open_types_of_files_form(self):
        """Открывает окно выбора типа файла."""
        self.types_of_files_form = TypesOfFilesForm()
        self.types_of_files_form.show()

    def load_mp3(self, filename):
        """Загружает медиафайл в плеер."""
        media = QtCore.QUrl.fromLocalFile(filename)
        content = QtMultimedia.QMediaContent(media)
        self.player = QtMultimedia.QMediaPlayer()
        self.player.setMedia(content)


class TypesOfFilesForm(QWidget):
    """Окно для выбора источника текста (TXT или DOCX)."""

    def __init__(self):
        super().__init__()
        self.btn_word_file = None
        self.btn_text_file = None
        self.name_label = None
        self.text_input = ""
        self.question_form = None
        self.text_form = None

        self.init_ui()

    def init_ui(self):
        """Инициализирует пользовательский интерфейс окна."""
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Text manager')

        self.btn_word_file = QPushButton(self)
        self.btn_word_file.clicked.connect(self.word_form)
        self.btn_word_file.setIcon(QIcon('word_file.jpg'))
        self.btn_word_file.move(170, 140)
        self.btn_word_file.setIconSize(QSize(100, 100))

        self.btn_text_file = QPushButton(self)
        self.btn_text_file.clicked.connect(self.open_text_form)
        self.btn_text_file.setIcon(QIcon('text_file.jpg'))
        self.btn_text_file.move(30, 140)
        self.btn_text_file.setIconSize(QSize(100, 100))

        self.name_label = QLabel(self)
        self.name_label.setText("Выберите формат файла,\nкоторый хотите прикрепить.")
        self.name_label.setAlignment(QtCore.Qt.AlignCenter)
        self.name_label.move(60, 90)
        self.name_label.setFont(QFont('Times New Roman', 10))
        self.show()

    def word_form(self):
        """Открывает диалог выбора .docx файла и считывает текст."""
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Load file', '', "Word File (*.docx)")
        if filename:
            doc = docx.Document(filename)
            self.text_input = "\n".join([par.text for par in doc.paragraphs])
            self.open_question_form()
        else:
            QMessageBox.warning(self, 'Error', "Файл не выбран.")

    def open_question_form(self):
        """Открывает окно для ввода вопроса."""
        self.question_form = QuestionForm(text_to_analyze=self.text_input)
        self.question_form.show()

    def open_text_form(self):
        """Открывает окно для ручного ввода текста."""
        self.text_form = TextForm()
        self.text_form.show()


class TextForm(QWidget):
    """Окно для ручного ввода анализируемого текста."""

    def __init__(self):
        super().__init__()
        self.btn_download = None
        self.text_label = None
        self.text_input = None
        self.question_form = None

        self.init_ui()

    def init_ui(self):
        """Инициализирует пользовательский интерфейс окна."""
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Text manager Text file')

        self.btn_download = QPushButton('Отправить', self)
        self.btn_download.move(100, 150)
        self.btn_download.clicked.connect(self.open_question_form)

        self.text_label = QLabel(self)
        self.text_label.setText("Пожалуйста введите текст.")
        self.text_label.move(100, 90)

        self.text_input = QLineEdit(self)
        self.text_input.move(100, 110)

    def open_question_form(self):
        """Открывает окно для ввода вопроса, передавая введенный текст."""
        self.question_form = QuestionForm(text_to_analyze=self.text_input.text())
        self.question_form.show()


class QuestionForm(QWidget):
    """Окно для ввода текста вопроса."""

    def __init__(self, parent=None, text_to_analyze=""):
        super().__init__(parent)
        self.text_to_analyze = text_to_analyze
        self.btn_send = None
        self.text_label = None
        self.question_input = None
        self.analysis_form = None

        self.init_ui()

    def init_ui(self):
        """Инициализирует пользовательский интерфейс окна."""
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Question Form')

        self.btn_send = QPushButton('Отправить', self)
        self.btn_send.move(100, 150)
        self.btn_send.clicked.connect(self.open_analysis_form)

        self.text_label = QLabel(self)
        self.text_label.setText("Пожалуйста введите текст вопроса.")
        self.text_label.move(70, 90)

        self.question_input = QLineEdit(self)
        self.question_input.move(100, 110)

    def open_analysis_form(self):
        """Запускает форму анализа, передавая текст и вопрос."""
        question = self.question_input.text()
        self.analysis_form = AnalysisForm(
            text=self.text_to_analyze, question=question
        )
        self.analysis_form.show()


class AnalysisForm(QMainWindow):
    """Класс, выполняющий семантический анализ текста."""

    def __init__(self, parent=None, text="", question=""):
        super().__init__(parent)
        self.text_output = None
        self.result_form = None
        self.bad_result_form = None

        self.text_analysis(text, question)

    def text_analysis(self, text, question):
        """Основной метод анализа текста и поиска ответов."""
        morph = pymorphy2.MorphAnalyzer()
        text = text.replace(',', '')
        suggestions = []

        split_regex = re.compile(r'[\.!?…]')

        text_sentences = [s.strip() for s in split_regex.split(text) if s.strip()]
        question_sentences = [s.strip() for s in split_regex.split(question) if s.strip()]

        suggestions.extend(text_sentences)

        exclude_tags = {"CONJ", "NPRO", "PREP", "PRCL"}

        def normalize_words(sentence):
            words = []
            for word in sentence.split():
                parsed_word = morph.parse(word)[0]
                if not any(tag in parsed_word.tag for tag in exclude_tags):
                    words.append(parsed_word.normal_form.lower())
            return words

        analysis_text = [normalize_words(s) for s in text_sentences]
        analysis_question = [word for s in question_sentences for word in normalize_words(s)]

        found_indices = []
        for i, sentence_words in enumerate(analysis_text):
            common_words = set(sentence_words) & set(analysis_question)
            if len(common_words) >= 2:
                found_indices.append(i)

        self.text_output = [suggestions[i] for i in found_indices]

        try:
            with open("text_manager_answer.txt", 'w', encoding='utf-8') as f:
                for item in self.text_output:
                    f.write(item + '\n')
        except IOError as e:
            print(f"Ошибка записи в файл: {e}")


        if not self.text_output:
            self.open_bad_result_form()
        else:
            self.open_result_form()

    def open_result_form(self):
        """Открывает окно с результатами поиска."""
        self.result_form = ResultForm(self.text_output)
        self.result_form.show()

    def open_bad_result_form(self):
        """Открывает окно с сообщением об отсутствии результатов."""
        self.bad_result_form = BadResultForm()
        self.bad_result_form.show()


class ResultForm(QMainWindow):
    """Окно для отображения найденных предложений."""

    def __init__(self, text_output):
        super().__init__()
        self.text_output = text_output
        self.central_widget = None
        self.list_widget = None

        self.init_ui()

    def init_ui(self):
        """Инициализирует пользовательский интерфейс окна."""
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle('Text manager Good Answer')
        self.central_widget = QWidget(self)
        self.list_widget = QListWidget(self.central_widget)
        self.list_widget.setGeometry(QtCore.QRect(0, 0, 600, 400))
        self.list_widget.addItems(self.text_output)
        self.setCentralWidget(self.central_widget)


class BadResultForm(QWidget):
    """Окно с сообщением об отсутствии результатов."""

    def __init__(self):
        super().__init__()
        self.text_label = None
        self.init_ui()

    def init_ui(self):
        """Инициализирует пользовательский интерфейс окна."""
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Text manager Bad Answer')
        self.text_label = QLabel(self)
        self.text_label.setText(
            "К сожалению, результатов по вашему запросу не найдено."
        )
        self.text_label.setWordWrap(True)
        self.text_label.setAlignment(QtCore.Qt.AlignCenter)
        self.text_label.setGeometry(20, 90, 295, 50)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec())