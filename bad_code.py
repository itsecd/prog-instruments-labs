# Стандартные библиотеки
import re
import sys

# Сторонние библиотеки
import docx
import pymorphy2
from PyQt5 import QtCore, QtMultimedia, QtWidgets
from PyQt5.QtCore import QSize, QUrl
from PyQt5.QtGui import QFont, QIcon, QPixmap
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


# Константы приложения
SCREEN_SIZE = [300, 300, 335, 300]
ANSWER_FILE_NAME = 'text_manager_answer.txt'
AUDIO_FILE_NAME = 'media.mp3'
WORD_ICON_FILE = 'word_file.jpg'
TEXT_ICON_FILE = 'text_file.jpg'
BACKGROUND_IMAGE_FILE = 'text.jpg'

FONT_FAMILY = 'Times New Roman'
FONT_SIZE = 10

WORD_FILE_FILTER = 'Word File (*.docx)'
FILE_ENCODING = 'utf-8'


class Example(QWidget):
    """Главное окно приложения с инструкцией и аудиоплеером."""

    def __init__(self):
        """Инициализирует главное окно приложения."""
        super().__init__()
        self.init_ui()
        self.player = None

    def init_ui(self):
        """Настраивает пользовательский интерфейс главного окна."""
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Text manager Instruction')

        self.load_mp3(AUDIO_FILE_NAME)
        self.create_buttons()
        self.create_instruction_labels()

    def create_buttons(self):
        """Создает кнопки управления для главного окна."""
        button_configs = [
            ('Воспроизвести', 70, 180, self.player.play),
            ('Пауза', 210, 180, self.player.pause),
            ('Стоп', 70, 220, self.player.stop),
            ('Дальше', 210, 220, self.open_types_of_files_form),
        ]

        for text, x, y, slot in button_configs:
            button = QPushButton(text, self)
            button.resize(button.sizeHint())
            button.move(x, y)
            button.clicked.connect(slot)

    def create_instruction_labels(self):
        """Создает текстовые метки с инструкцией для пользователя."""
        labels_data = [
            ('Вас приветствует Text manager.', 70, 20),
            ('Text manager - это программа для анализа содержания', 10, 40),
            ('текста и поиска ответа на заданный вами вопрос.', 10, 60),
            ('Чтобы получше познакомится с функциями Text manager,', 10, 80),
            (' вы можите прослушать аудиоинструкцию,', 40, 100),
            (' нажав на кнопку "Воспроизвести".', 60, 120),
        ]

        for text, x, y in labels_data:
            label = QLabel(self)
            label.setFont(QFont(FONT_FAMILY, FONT_SIZE))
            label.setText(text)
            label.move(x, y)

    def open_types_of_files_form(self):
        """Открывает форму выбора типа файла."""
        self.types_of_files_form = TypesOfFilesForm(
            self, 'Данные для второй формы'
        )
        self.types_of_files_form.show()

    def load_mp3(self, filename):
        """
        Загружает аудиофайл для воспроизведения.

        Args:
            filename (str): Путь к аудиофайлу
        """
        media = QtCore.QUrl.fromLocalFile(filename)
        content = QtMultimedia.QMediaContent(media)
        self.player = QtMultimedia.QMediaPlayer()
        self.player.setMedia(content)


class TypesOfFilesForm(QWidget):
    """Форма для выбора типа загружаемого файла."""

    def __init__(self, *args):
        """Инициализирует форму выбора типа файла."""
        super().__init__()
        self.init_ui(args)
        self.text_input = ''

    def init_ui(self, args):
        """Настраивает пользовательский интерфейс формы выбора файла."""
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Text manager')

        self.create_file_buttons()
        self.create_info_labels()
        self.show()

    def create_file_buttons(self):
        """Создает кнопки для выбора типа файла."""
        button_configs = [
            (WORD_ICON_FILE, 170, 140, self.process_word_file),
            (TEXT_ICON_FILE, 30, 140, self.open_text_form),
        ]

        for icon_file, x, y, slot in button_configs:
            button = QPushButton(self)
            button.clicked.connect(slot)
            button.setIcon(QIcon(icon_file))
            button.move(x, y)
            button.setIconSize(QSize(100, 100))

    def create_info_labels(self):
        """Создает информационные метки формы."""
        label_texts = [
            ('Выберите формат файла,', 60, 90),
            ('который хотите прикрепить.', 60, 105),
        ]

        for text, x, y in label_texts:
            label = QLabel(self)
            label.setText(text)
            label.move(x, y)
            label.setFont(QFont(FONT_FAMILY, FONT_SIZE))

    def process_word_file(self):
        """Обрабатывает выбор и загрузку Word файла."""
        filename = QFileDialog.getOpenFileName(
            self, 'Load file', '', WORD_FILE_FILTER
        )
        file_path = filename[0]

        self.text_input = ''

        if file_path:
            doc = docx.Document(file_path)
            for paragraph in doc.paragraphs:
                self.text_input += paragraph.text
        else:
            QMessageBox.warning(self, 'Error', 'Файл не выбран.')

        self.open_question_form()

    def open_question_form(self):
        """Открывает форму для ввода вопроса."""
        self.question_form = QuestionForm(self, '', self.text_input)
        self.question_form.show()

    def open_text_form(self):
        """Открывает форму для ввода текста."""
        self.text_form = TextForm(self, '')
        self.text_form.show()


class TextForm(QWidget):
    """Форма для ввода текста вручную."""

    def __init__(self, *args):
        """Инициализирует форму ввода текста."""
        super().__init__()
        self.init_ui(args)
        self.text_length = ''

    def init_ui(self, args):
        """Настраивает пользовательский интерфейс формы ввода текста."""
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Text manager Text file')

        self.submit_btn = QPushButton('Отправить', self)
        self.submit_btn.resize(self.submit_btn.sizeHint())
        self.submit_btn.move(100, 150)
        self.submit_btn.clicked.connect(self.open_question_form)

        self.text_label = QLabel(self)
        self.text_label.setText('Пожалуйста введите текст.')
        self.text_label.move(100, 90)
        self.text_label.setFont(QFont(FONT_FAMILY, FONT_SIZE))

        self.text_input = QLineEdit(self)
        self.text_input.move(100, 110)

        self.additional_label = QLabel(args[-1], self)
        self.additional_label.adjustSize()

    def open_question_form(self):
        """Открывает форму для ввода вопроса с переданным текстом."""
        self.question_form = QuestionForm(self, '', self.text_input.text())
        self.question_form.show()


class QuestionForm(QWidget):
    """Форма для ввода вопроса к тексту."""

    def __init__(self, *args):
        """Инициализирует форму ввода вопроса."""
        super().__init__()
        self.init_ui(args)
        self.text_length = ''
        self.text_content = ''

    def init_ui(self, args):
        """Настраивает пользовательский интерфейс формы ввода вопроса."""
        text_length = args[2]
        self.text_content = str(text_length)
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Question Form')

        self.submit_btn = QPushButton('Отправить', self)
        self.submit_btn.resize(self.submit_btn.sizeHint())
        self.submit_btn.move(100, 150)
        self.submit_btn.clicked.connect(self.open_analysis_form)

        self.text_label = QLabel(self)
        self.text_label.setText('Пожалуйста введите текст вопроса.')
        self.text_label.move(100, 90)
        self.text_label.setFont(QFont(FONT_FAMILY, FONT_SIZE))

        self.question_input = QLineEdit(self)
        self.question_input.move(100, 110)

        self.additional_label = QLabel(args[-1], self)
        self.additional_label.adjustSize()

    def open_analysis_form(self):
        """Открывает форму анализа текста с вопросом."""
        question_text = self.question_input.text()
        self.analysis_form = AnalysisForm(
            self, '', self.additional_label.text(), question_text
        )
        self.analysis_form.show()


class AnalysisForm(QMainWindow):
    """Форма для анализа текста и поиска ответов на вопрос."""

    def __init__(self, *args):
        """Инициализирует форму анализа текста."""
        super().__init__()
        self.text_analysis(args)
        self.text_content = ''
        self.question_text = ''

    def text_analysis(self, args):
        """
        Анализирует текст и вопрос, находя релевантные предложения.

        Args:
            args: Аргументы, содержащие текст и вопрос для анализа
        """
        self.text_content = args[2]
        self.question_text = args[3]
        morph = pymorphy2.MorphAnalyzer()

        # Очистка текста от запятых
        self.text_content = self.text_content.replace(',', '')

        analysis_data = self._prepare_analysis_data(morph)
        matching_sentences = self._find_matching_sentences(analysis_data)

        self._save_results(matching_sentences)

        if matching_sentences:
            self.open_result_form(matching_sentences)
        else:
            self.open_bad_result_form()

    def _prepare_analysis_data(self, morph):
        """
        Подготавливает данные для анализа.

        Нормализует слова и разбивает на предложения.

        Args:
            morph: Морфологический анализатор pymorphy2

        Returns:
            dict: Подготовленные данные для анализа
        """
        split_regex = re.compile(r'[.|!|?|…]')
        texts_to_analyze = [self.text_content, self.question_text]

        analysis_text = []
        analysis_question = []
        original_sentences = []

        for text in texts_to_analyze:
            sentences = filter(
                lambda t: t, [t.strip() for t in split_regex.split(text)]
            )

            for sentence in sentences:
                original_sentences.append(sentence)
                normalized_words = self._normalize_sentence_words(
                    sentence, morph
                )

                if text == self.text_content:
                    analysis_text.append(normalized_words)
                elif text == self.question_text:
                    analysis_question.append(normalized_words)

        return {
            'analysis_text': analysis_text,
            'analysis_question': analysis_question,
            'original_sentences': original_sentences,
        }

    def _normalize_sentence_words(self, sentence, morph):
        """
        Нормализует слова в предложении.

        Args:
            sentence (str): Предложение для обработки
            morph: Морфологический анализатор

        Returns:
            list: Список нормализованных слов
        """
        words = sentence.split()
        normalized_words = []

        for word in words:
            parsed_word = morph.parse(word)[0]
            if self._is_content_word(parsed_word):
                normalized_form = parsed_word.normal_form.lower()
                normalized_words.append([normalized_form])

        return normalized_words

    def _is_content_word(self, parsed_word):
        """
        Проверяет, является ли слово значимым.

        Args:
            parsed_word: Результат морфологического разбора слова

        Returns:
            bool: True если слово значимое, False если служебное
        """
        excluded_tags = {'CONJ', 'NPRO', 'PREP', 'PRCL'}
        return not any(tag in parsed_word.tag for tag in excluded_tags)

    def _find_matching_sentences(self, analysis_data):
        """
        Находит предложения, релевантные вопросу.

        Args:
            analysis_data (dict): Подготовленные данные для анализа

        Returns:
            list: Список релевантных предложений
        """
        analysis_text = analysis_data['analysis_text']
        analysis_question = analysis_data['analysis_question']
        original_sentences = analysis_data['original_sentences']

        question_words = self._extract_question_words(analysis_question)
        matching_indices = self._find_matching_indices(
            analysis_text, question_words
        )
        significant_matches = self._filter_significant_matches(matching_indices)
        return self._build_result_sentences(
            significant_matches, original_sentences
        )

    def _extract_question_words(self, analysis_question):
        """
        Извлекает слова из вопроса.

        Args:
            analysis_question (list): Обработанные данные вопроса

        Returns:
            list: Список слов вопроса
        """
        question_words = []
        for sentence_words in analysis_question:
            for word_list in sentence_words:
                question_words.append(''.join(word_list))
        return question_words

    def _find_matching_indices(self, analysis_text, question_words):
        """
        Находит индексы предложений с совпадениями.

        Args:
            analysis_text (list): Обработанные данные текста
            question_words (list): Слова вопроса

        Returns:
            list: Индексы совпадающих предложений
        """
        matching_indices = []
        for i, sentence_words in enumerate(analysis_text):
            for word_list in sentence_words:
                word = ''.join(word_list)
                if word in question_words:
                    matching_indices.append(i)
        return matching_indices

    def _filter_significant_matches(self, matching_indices):
        """
        Фильтрует значимые совпадения.

        Args:
            matching_indices (list): Индексы совпадающих предложений

        Returns:
            list: Уникальные индексы значимых совпадений
        """
        significant_matches = []
        for index in matching_indices:
            if matching_indices.count(index) >= 2:
                significant_matches.append(index)
        return list(set(significant_matches))

    def _build_result_sentences(self, significant_matches, original_sentences):
        """
        Формирует итоговые предложения.

        Args:
            significant_matches (list): Индексы значимых предложений
            original_sentences (list): Оригинальные предложения

        Returns:
            list: Очищенные релевантные предложения
        """
        result_sentences = []
        for index in significant_matches:
            if index < len(original_sentences):
                cleaned_sentence = self._clean_sentence_text(
                    str(original_sentences[index])
                )
                result_sentences.append(cleaned_sentence)
        return result_sentences

    def _clean_sentence_text(self, sentence):
        """
        Очищает текст предложения от лишних символов.

        Args:
            sentence (str): Исходное предложение

        Returns:
            str: Очищенное предложение
        """
        replacements = [
            ("['", ''),
            ("'],", ''),
            ("']]", ''),
            ('[', ''),
        ]

        for old, new in replacements:
            sentence = sentence.replace(old, new)
        return sentence

    def _save_results(self, sentences):
        """
        Сохраняет результаты анализа в файл.

        Args:
            sentences (list): Список релевантных предложений
        """
        with open(ANSWER_FILE_NAME, 'w', encoding=FILE_ENCODING) as file:
            for sentence in sentences:
                file.write(sentence + '\n')

        with open(ANSWER_FILE_NAME, 'r', encoding=FILE_ENCODING) as file:
            print(file.read())

    def open_result_form(self, sentences):
        """Открывает форму с результатами поиска."""
        self.result_form = ResultForm(self, '', sentences)
        self.result_form.show()

    def open_bad_result_form(self):
        """Открывает форму с сообщением об отсутствии результатов."""
        self.bad_result_form = BadResultForm(self, '')
        self.bad_result_form.show()


class ResultForm(QMainWindow):
    """Форма для отображения результатов поиска."""

    def __init__(self, *args):
        """Инициализирует форму результатов."""
        super().__init__()
        self.init_ui(args)
        self.text_output = ''

    def init_ui(self, args):
        """Настраивает пользовательский интерфейс формы результатов."""
        self.text_output = args[2]
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle('Text manager Good Answer')

        # Установка фонового изображения
        self.pixmap = QPixmap(BACKGROUND_IMAGE_FILE)
        self.image = QLabel(self)
        self.image.move(30, 0)
        self.image.resize(400, 30)
        self.image.setPixmap(self.pixmap)

        # Настройка списка для отображения результатов
        self.central_widget = QtWidgets.QWidget(self)
        self.list_widget = QtWidgets.QListWidget(self.central_widget)
        self.list_widget.setGeometry(QtCore.QRect(0, 30, 600, 500))

        # Добавление элементов в список
        for _ in range(4):
            item = QtWidgets.QListWidgetItem()
            self.list_widget.addItem(item)

        self.setCentralWidget(self.central_widget)
        self.list_widget.addItems(self.text_output)


class BadResultForm(QWidget):
    """Форма для отображения сообщения об отсутствии результатов."""

    def __init__(self, *args):
        """Инициализирует форму отсутствия результатов."""
        super().__init__()
        self.init_ui(args)

    def init_ui(self, args):
        """Настраивает пользовательский интерфейс формы отсутствия результатов."""
        self.setGeometry(*SCREEN_SIZE)
        self.setWindowTitle('Text manager Bad Answer')

        messages = [
            'К сожалению результатов по вашему запросу ',
            'не найдено. Попробуйте еще раз сформулировав',
            'более точный вопрос или приложите больше',
            'информации для поиска.',
        ]

        y_position = 90
        for message in messages:
            label = QLabel(self)
            label.setText(message)
            label.move(40, y_position)
            y_position += 15


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec())