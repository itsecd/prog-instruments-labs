import sys

from PyQt5.QtWidgets import (
    QPushButton,
    QApplication,
    QMainWindow,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QDesktopWidget,
    QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont

from questions import Questions

class MainWindow(QMainWindow):
    """Class of main window of the app.
    """

    def __init__(self):
        super().__init__()
        self.PROGRAM_NAME = "Studiyng helper"
        self.ICON_PATH = "src/images/icon.png"

        self.X = 300
        self.Y = 300
        self.WINDOW_WIDTH = 600
        self.WINDOW_HEIGHT_MENU = 200
        self.WINDOW_HEIGHT_QUESTIONS = 400
        self.QUESTION_LABEL_WIDTH = 500
        self.QUESTION_LABEL_HEIGHT = 200

        self.BUTTON_COLOR = '#A99AEA'
        self.GOOD_BUTTON_COLOR = "#95D7AE"
        self.BAD_BUTTON_COLOR = "#FA824C"
        self.QUESTION_TEXT_COLOR = '#DCE3F9'
        self.PLAIN_TEXT_COLOR = '#342E37'
        self.STATISTIC_TEXT_COLOR = '#735CDD'
        self.BACKGROUND_COLOR = '#C5BBF1'

        self.MAIN_FONT = "Consolas"
        self.ACCENT_FONT = "FreeMono, monospace"
        self.CHECKBOX_FONT_SIZE = 12
        self.FONT_SIZE = 14
        self.QUESTION_FONT_SIZE = 20
        self.BUTTON_FONT_SIZE = 15
        
        self.teach_old = self.create_button("Повторить выученное", self.repeat_old_button_action)
        self.teach_new = self.create_button("Выучить новое", self.learn_new_button_action)
        self.teach_all = self.create_button("Учить все вопросы", self.learn_all_button_action)
        self.menu_button = self.create_button("Вернуться в меню", self.set_main_menu)
        self.good_button = self.create_button('Ответил', self.good_answer_button_action, self.GOOD_BUTTON_COLOR)
        self.bad_button = self.create_button('Не ответил', self.bad_answer_button_action, self.BAD_BUTTON_COLOR)
        self.skip_button = self.create_button('Отложить вопрос', self.skip_question_button_action)
        
        self.check_box = QCheckBox()
        self.question_label = QLabel()
        self.learned_label = QLabel()
        self.learned_label.setFont(QFont(self.MAIN_FONT, self.FONT_SIZE))
        self.learned_label.setStyleSheet(f"color: {self.STATISTIC_TEXT_COLOR};")
        self.inprocess_label = QLabel()
        self.inprocess_label.setFont(QFont(self.MAIN_FONT, self.FONT_SIZE))
        self.inprocess_label.setStyleSheet(f"color: {self.STATISTIC_TEXT_COLOR};")
        self.unlearned_label = QLabel()
        self.unlearned_label.setFont(QFont(self.MAIN_FONT, self.FONT_SIZE))
        self.unlearned_label.setStyleSheet(f"color: {self.STATISTIC_TEXT_COLOR};")

        self.init_ui()

    def init_ui(self):
        """initializing starting ui with button for chosing mode
        """
        self.setGeometry(self.X, self.Y, self.WINDOW_WIDTH, self.WINDOW_HEIGHT_MENU)
        self.setStyleSheet(f"background-color: {self.BACKGROUND_COLOR}")
        self.setWindowTitle(self.PROGRAM_NAME)
        self.setWindowIcon(QIcon(self.ICON_PATH))

        widget = QWidget()
        vbox = QVBoxLayout(widget)
        vbox.addStretch(1)
        vbox.addWidget(self.teach_old)
        vbox.addWidget(self.teach_new)
        vbox.addWidget(self.teach_all)

        self.check_box.stateChanged.connect(Questions.set_random_state)
        self.check_box.setText('Показывать вопросы в случайном порядке')
        self.check_box.setChecked(True)
        self.check_box.setFont(QFont(self.ACCENT_FONT, self.CHECKBOX_FONT_SIZE))

        vbox.addWidget(self.check_box)
        vbox.addStretch(1)
        self.setCentralWidget(widget)
        self.move_to_center()

        self.show()

    def center_window(self):
        """Move the window to the center of the screen."""
        window_geometry = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft())

    def change_ui(self):
        """Changes UI after choose of type to learning
        """
        widget = QWidget()
        widget.setStyleSheet(f"background-color: {self.BACKGROUND_COLOR};")

        self.good_button.setEnabled(True)
        self.bad_button.setEnabled(True)

        self.question_label.setStyleSheet(
            f"padding :15px; color: {self.PLAIN_TEXT_COLOR};"
        )
        self.question_label.setFont(QFont(self.MAIN_FONT, self.QUESTION_FONT_SIZE))
        self.question_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.question_label.setWordWrap(True)
        self.question_label.resize(self.QUESTION_LABEL_WIDTH, self.QUESTION_LABEL_HEIGHT)

        text = Questions.get_question()
        if text is not None:
            self.question_label.setText(Questions.get_question())
        else:
            self.question_label.setText("Похоже, таких вопросов еще нет...")
            self.good_button.setEnabled(False)
            self.bad_button.setEnabled(False)
        
        container = QWidget(self)
        container.setStyleSheet(f"background-color: {self.QUESTION_TEXT_COLOR};")
        upper_vbox = QVBoxLayout()
        upper_vbox.addWidget(container)
        in_widget_vbox = QVBoxLayout(container)

        text_hbox = QHBoxLayout()
        text_hbox.addWidget(self.question_label)

        buttons_hbox = QHBoxLayout()
        buttons_hbox.addWidget(self.good_button)
        buttons_hbox.addWidget(self.skip_button)
        buttons_hbox.addWidget(self.bad_button)

        bottom_hbox = QHBoxLayout()

        self.update_statistic_labels()
        bottom_statistics_vbox = QVBoxLayout()
        bottom_statistics_vbox.addWidget(self.learned_label)
        bottom_statistics_vbox.addWidget(self.inprocess_label)
        bottom_statistics_vbox.addWidget(self.unlearned_label)

        bottom_backbutton_vbox = QVBoxLayout()

        bottom_backbutton_vbox.addWidget(self.menu_button)

        in_widget_vbox.addLayout(text_hbox)
        in_widget_vbox.addStretch()
        in_widget_vbox.addLayout(buttons_hbox)

        bottom_hbox.addLayout(bottom_statistics_vbox)
        bottom_hbox.addStretch()
        bottom_hbox.addLayout(bottom_backbutton_vbox)

        main_vbox = QVBoxLayout(widget)
        main_vbox.addLayout(upper_vbox)
        main_vbox.addLayout(bottom_hbox)

        self.setGeometry(self.X, self.Y, self.WINDOW_WIDTH, self.WINDOW_HEIGHT_QUESTIONS)
        self.setCentralWidget(widget)
        self.move_to_center()
        self.show()

    def create_button(self, text: str, function, button_color: str = None) -> QPushButton:
        """Create a button with custom styles.

    Args:
        text (str): Text to display on the button.
        function (Callable[[None], None]): Function to connect to the button's clicked signal.
        button_color (str, optional): Background color for the button. Defaults to self.BUTTON_COLOR.

    Returns:
        QPushButton: A styled QPushButton instance.
    """
        if button_color is None:
            button_color = self.BUTTON_COLOR
        button = QPushButton(text)
        button.clicked.connect(function)
        button.resize(button.minimumSizeHint())
        button.setStyleSheet(
            f"""
            background-color: {button_color};
            color: {self.PLAIN_TEXT_COLOR};
            padding: 10px;
            border-radius: 5px;
            """
        )
        button.setFont(QFont(self.ACCENT_FONT, self.BUTTON_FONT_SIZE))
        return button

    def repeat_old_button_action(self):
        """ Action for button for repeat old questions.
            Changes ui for learning.
        """
        Questions.repeat_old()
        self.change_ui()

    def learn_all_button_action(self):
        """ Action for button for repeat all questions.
            Changes ui for learning.
        """
        Questions.repeat_all()
        self.change_ui()

    def learn_new_button_action(self):
        """ Action for button for repeat new questions.
            Changes ui for learning.
        """
        Questions.learn_new()
        self.change_ui()

    def good_answer_button_action(self):
        """Action for button for good answer to the question.
            Updates statistics.
        """
        Questions.question_accept()
        text = Questions.get_question()
        if text is not None:
            self.question_label.setText(text)
        else:
            self.question_label.setText("Поздравляю! Ты смог ответить на все вопросы)")
            self.good_button.setEnabled(False)
            self.bad_button.setEnabled(False)
        self.update_statistic_labels()

    def bad_answer_button_action(self):
        """Action for button for bad answer to the question.
            Updates statistics.
        """
        Questions.question_failed()
        text = Questions.get_question()
        self.question_label.setText("Повтори и попробуй еще раз...\n" + text)
        self.update_statistic_labels()

    def update_statistic_labels(self):
        """ Updates labels with statistics
        """
        self.learned_label.setText(f'Точно выучено вопросов: {Questions.get_learned()}')
        self.inprocess_label.setText(f'В процессе запоминания: {Questions.get_inprocess()}')
        self.unlearned_label.setText(f'Осталось невыученных: {Questions.get_unlearned()}')


    def set_main_menu(self):
        """ Opens window with main menu.
        """
        self.close()
        self.__init__()

    def skip_question_button_action(self):
        """ Action for button for skip question.
            Updates statistics.
        """
        Questions.skip_question()
        self.update_statistic_labels()
        text = Questions.get_question()
        self.question_label.setText(text)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
