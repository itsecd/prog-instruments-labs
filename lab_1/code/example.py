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
        self.BUTTON_COLOR = '#A99AEA'
        self.GOOD_BUTTON_COLOR = "#95D7AE"
        self.BAD_BUTTON_COLOR = "#FA824C"
        self.QUESTION_TEXT_COLOR = '#DCE3F9'
        self.PLAIN_TEXT_COLOR = '#342E37'
        self.STATISTIC_TEXT_COLOR = '#735CDD'
        self.BACKGROUND_COLOR = '#C5BBF1'
        self.FONT = "Consolas"
        self.FONT_SIZE = 14
        
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
        self.learned_label.setFont(QFont(self.FONT, self.FONT_SIZE))
        self.learned_label.setStyleSheet(f"color: {self.STATISTIC_TEXT_COLOR};")
        self.inprocess_label = QLabel()
        self.inprocess_label.setFont(QFont(self.FONT, self.FONT_SIZE))
        self.inprocess_label.setStyleSheet(f"color: {self.STATISTIC_TEXT_COLOR};")
        self.unlearned_label = QLabel()
        self.unlearned_label.setFont(QFont(self.FONT, self.FONT_SIZE))
        self.unlearned_label.setStyleSheet(f"color: {self.STATISTIC_TEXT_COLOR};")

        self.init_ui()

    def init_ui(self):
        """initializing starting ui with button for chosing mode
        """
        self.setGeometry(300, 300, 600, 200)
        self.setStyleSheet(f"background-color: {self.BACKGROUND_COLOR}")
        self.setWindowTitle("Studiyng helper")
        self.setWindowIcon(QIcon("src/images/icon.png"))

        widget = QWidget()
        vbox = QVBoxLayout(widget)
        vbox.addStretch(1)
        vbox.addWidget(self.teach_old)
        vbox.addWidget(self.teach_new)
        vbox.addWidget(self.teach_all)
        self.check_box.stateChanged.connect(Questions.set_random_state)
        self.check_box.setText('Показывать вопросы в случайном порядке')
        self.check_box.setChecked(True)
        self.check_box.setFont(QFont("FreeMono, monospace", 12))
        vbox.addWidget(self.check_box)
        vbox.addStretch(1)
        self.setCentralWidget(widget)
        self.move_to_center()
        self.show()

    def move_to_center(self):
        """move window to center of screen
        """
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def change_ui(self):
        """Changes UI after choose of type to learning
        """
        widget = QWidget()
        widget.setStyleSheet(f"background-color: {self.BACKGROUND_COLOR};")

        self.good_button.setEnabled(True)
        self.bad_button.setEnabled(True)

        # scroll_area.verticalScrollBar().

        self.question_label.setStyleSheet(
            f"padding :15px; color: {self.PLAIN_TEXT_COLOR};"
        )
        self.question_label.setFont(QFont("Consolas", 20))
        self.question_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.question_label.setWordWrap(True)
        self.question_label.resize(500, 200)
        text = Questions.get_question()
        if text is not None:
            self.question_label.setText(Questions.get_question())
        else:
            self.question_label.setText("Похоже, таких вопросов еще нет...")
            self.good_button.setEnabled(False)
            self.bad_button.setEnabled(False)
        # self.label.setGeometry(
        # 0, 0, 400, scroll_area.height())
        container = QWidget(self)
        container.setStyleSheet(f"background-color: {self.QUESTION_TEXT_COLOR};")
        upper_vbox = QVBoxLayout()
        upper_vbox.addWidget(container)
        in_widget_vbox = QVBoxLayout(container)

        # textbox = QVBoxLayout()
        # textbox.addWidget(scroll_area)
        text_hbox = QHBoxLayout()
        text_hbox.addWidget(self.question_label)
        # vrbox.addStretch()

        btns_hbox = QHBoxLayout()
        # btns_hbox.addStretch(1)
        btns_hbox.addWidget(self.good_button)
        # btns_hbox.addStretch(1)
        btns_hbox.addWidget(self.skip_button)
        btns_hbox.addWidget(self.bad_button)

        # btns_hbox.addStretch(1)

        bottom_hbox = QHBoxLayout()

        self.update_statistic_labels()
        bottom_stats_vbox = QVBoxLayout()
        bottom_stats_vbox.addWidget(self.learned_label)
        bottom_stats_vbox.addWidget(self.inprocess_label)
        bottom_stats_vbox.addWidget(self.unlearned_label)

        bottom_backbutton_vbox = QVBoxLayout()

        bottom_backbutton_vbox.addWidget(self.menu_button)

        in_widget_vbox.addLayout(text_hbox)
        in_widget_vbox.addStretch()
        in_widget_vbox.addLayout(btns_hbox)

        bottom_hbox.addLayout(bottom_stats_vbox)
        bottom_hbox.addStretch()
        bottom_hbox.addLayout(bottom_backbutton_vbox)

        main_vbox = QVBoxLayout(widget)
        main_vbox.addLayout(upper_vbox)
        main_vbox.addLayout(bottom_hbox)

        self.setGeometry(300, 300, 600, 400)
        self.setCentralWidget(widget)
        self.move_to_center()
        self.show()

    def create_button(self, text: str, func, btn_color: str = None) -> QPushButton:
        """Creates a button with styles

        Args:
            text (str): text of button
            func (_type_): function for button

        Returns:
            QPushButton: button with styles
        """
        if btn_color is None:
            btn_color = self.BUTTON_COLOR
        button = QPushButton(text)
        button.clicked.connect(func)
        button.resize(button.minimumSizeHint())
        button.setStyleSheet(
            f"""
                             background-color: {btn_color};
                             color: {self.PLAIN_TEXT_COLOR};
                             font-size: 20px;
                             padding: 10px 10px 10px 10px;
                             border-radius: 5px;
                             """
        )
        # shadow = QGraphicsDropShadowEffect(
        #     blurRadius=7, xOffset=3, yOffset=3, color=QColor("#31021F")
        # )
        # button.setGraphicsEffect(shadow)
        button.setFont(QFont("FreeMono, monospace", 15))
        return button

    def repeat_old_button_action(self):
        """ Action for button for repeat old questions.
            Changed ui for learning.
        """
        Questions.repeat_old()
        self.change_ui()

    def learn_all_button_action(self):
        """ Action for button for repeat all questions.
            Changed ui for learning.
        """
        Questions.repeat_all()
        self.change_ui()

    def learn_new_button_action(self):
        """ Action for button for repeat new questions.
            Changed ui for learning.
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
