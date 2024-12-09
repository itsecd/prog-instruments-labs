import os
import ctypes
import tkinter as tk
import sqlite3
from typing import Optional, List

import words_api
from PIL import Image, ImageTk

import settings as st
from config import WIDTH, HEIGHT, FIRST_RIGHT, BG, MAX_SCORE


ctypes.windll.shcore.SetProcessDpiAwareness(1)


class Wordle:
    """
    Класс для реализации игры Wordle с графическим интерфейсом на Tkinter.

    Атрибуты:
        FIRST_RIGHT (int): Константа, устанавливающая количество очков за правильный ответ.
        BG (str): Цвет фона интерфейса.
        MAX_SCORE (int): Максимальное количество очков, которое можно получить за игру.
    """

    def __init__(self) -> None:
        """
        Инициализация класса Wordle.

        Создает главное окно игры, инициализирует необходимые параметры и элементы интерфейса.
        Загружает настройки из базы данных и загружает изображения для интерфейса.
        """
        self.root = tk.Tk()

        self.width = WIDTH
        self.height = HEIGHT
        self.x_co = int(self.root.winfo_screenwidth() / 2) - int(self.width / 2)
        self.y_co = 50

        self.root.geometry(f"{self.width}x{self.height}+{self.x_co}+{self.y_co}")

        self.root.configure(background=BG)
        self.root.title("Wordle")
        self.root.wm_iconbitmap("images/icon.ico")

        self.guess = ""
        self.won = False
        self.guess_count = 0
        self.score = 0
        self.word_size = 5
        self.high_score = 0
        self.word_api = None
        self.get_from_db()

        self.setting = Image.open("images/setting.png")
        self.setting = self.setting.resize((40, 40), Image.Resampling.LANCZOS)
        self.setting = ImageTk.PhotoImage(self.setting)

        self.setting_dark = Image.open("images/setting_dark.png")
        self.setting_dark = self.setting_dark.resize((40, 40), Image.Resampling.LANCZOS)
        self.setting_dark = ImageTk.PhotoImage(self.setting_dark)

        label = Image.open("images/head.png")
        label = ImageTk.PhotoImage(label)

        top_frame = tk.Frame(self.root, bg=BG)
        top_frame.pack(fill="x")

        sett = tk.Button(
            top_frame,
            image=self.setting,
            command=self.open_setting,
            bd=0,
            bg=BG,
            cursor="hand2",
            activebackground=BG,
        )
        sett.pack(side="right")
        sett.bind("<Enter>", self.on_hover)
        sett.bind("<Leave>", self.off_hover)

        head = tk.Label(self.root, image=label, bd=0, bg=BG)
        head.pack()

        # word buttons

        main_btn_frame = tk.Frame(self.root, bg=BG)
        main_btn_frame.pack(pady=15)

        f1 = tk.Frame(main_btn_frame, bg=BG)
        f2 = tk.Frame(main_btn_frame, bg=BG)
        f3 = tk.Frame(main_btn_frame, bg=BG)
        f4 = tk.Frame(main_btn_frame, bg=BG)
        f5 = tk.Frame(main_btn_frame, bg=BG)
        f6 = tk.Frame(main_btn_frame, bg=BG)
        self.button_frames = [f1, f2, f3, f4, f5, f6]

        self.b_row1 = self.b_row2 = self.b_row3 = self.b_row4 = self.b_row5 = (
            self.b_row6
        ) = []
        self.buttons = []

        self.current_B_row = 0
        self.current_b = 0

        self.show_buttons()

        # keypad buttons

        keyboard_frame = tk.Frame(self.root, bg=BG)
        keyboard_frame.pack(pady=5)

        c = 65

        f1 = tk.Frame(keyboard_frame, bg=BG)
        f1.pack(side="top", pady=2)
        f2 = tk.Frame(keyboard_frame, bg=BG)
        f2.pack(side="top", pady=2)
        f3 = tk.Frame(keyboard_frame, bg=BG)
        f3.pack(side="top", pady=2)

        f = [f1, f2, f3]
        step = 6
        self.keypad_buttons = [[], [], []]

        self.keypad_btn_pos = {
            0: [chr(i) for i in range(65, 71)],
            1: [chr(i) for i in range(71, 81)],
            2: [chr(i) for i in range(81, 91)],
        }

        key_pad_color = "#ff7700"
        index = 0
        for i in range(3):
            for _ in range(step):
                b = tk.Button(
                    f[index],
                    text=chr(c),
                    font="cambria 13 bold",
                    bg=BG,
                    fg=key_pad_color,
                    cursor="hand2",
                    padx=3,
                )
                b.pack(side="left", padx=2)
                self.keypad_buttons[i].append(b)
                b.bind("<Button-1>", lambda e: self.key_press(keyboard=e))
                b.bind("<Enter>", lambda e: on_hover(e, "#575656"))
                b.bind("<Leave>", lambda e: off_hover(e, BG))
                c += 1
            if i == 0:
                b = tk.Button(
                    f[index],
                    text="Enter",
                    font="cambria 13 bold",
                    bg=BG,
                    fg=key_pad_color,
                    cursor="hand2",
                )
                b.pack(side="left", padx=2)
                b.bind("<Button-1>", lambda e: self.key_press(keyboard=e))
                b.bind("<Enter>", lambda e: on_hover(e, "#575656"))
                b.bind("<Leave>", lambda e: off_hover(e, BG))
            if i == 0:
                b = tk.Button(
                    f[index],
                    text="←",
                    font="cambria 13 bold",
                    bg=BG,
                    fg=key_pad_color,
                    cursor="hand2",
                )
                b.pack(side="left", padx=2)
                b.bind("<Button-1>", lambda e: self.key_press(keyboard=e))
                b.bind("<Enter>", lambda e: on_hover(e, "#575656"))
                b.bind("<Leave>", lambda e: off_hover(e, BG))
            index += 1
            step = 10

        self.status_bar = tk.Label(
            self.root,
            text=f"Score : {self.score}",
            font="cambria 10 bold",
            anchor="w",
            padx=10,
            background="#242424",
            fg="white",
        )
        self.status_bar.pack(fill="x", side="bottom")

        self.root.bind("<KeyRelease>", self.key_press)

        self.root.mainloop()

    def show_buttons(self) -> None:
        """
        Отображает кнопки для ввода слов в игре.

        Удаляет старые кнопки, если они существуют, и создает новые для текущей игры.
        """
        if self.buttons:
            for b in self.buttons:
                if b:
                    for i in b:
                        i.destroy()

        self.b_row1 = self.b_row2 = self.b_row3 = self.b_row4 = self.b_row5 = (
            self.b_row6
        ) = []
        self.buttons = []

        self.current_B_row = 0
        self.current_b = 0

        for i in range(6):
            row_btn = []
            self.button_frames[i].pack(pady=4)
            for j in range(self.word_size):
                b = tk.Button(
                    self.button_frames[i],
                    text="",
                    fg="white",
                    bd=2,
                    font="lucida 18",
                    bg=BG,
                    width=3,
                    height=1,
                )
                b.pack(side="left", padx=2)

                row_btn.append(b)
            self.buttons.append(row_btn)

    def key_press(
        self, e: Optional[tk.Event] = None, keyboard: Optional[tk.Button] = None
    ) -> None:
        """
        Обрабатывает нажатия клавиш на клавиатуре или кнопках интерфейса.

        Args:
            e (Optional[tk.Event]): Событие нажатия клавиши.
            keyboard (Optional[tk.Button]): Кнопка, на которую нажали на клавиатуре.
        """
        if e:
            if e.keysym == "BackSpace":
                self.erase_character()

            elif e.keysym == "Return":
                self.check_for_match()

            elif 65 <= e.keycode <= 90:
                key = e.char
                if self.current_b == self.word_size:
                    self.current_b = self.word_size - 1

                    characters = list(self.guess)
                    characters[self.current_b] = ""
                    self.guess = "".join(characters)

                self.buttons[self.current_B_row][self.current_b]["text"] = key.upper()
                self.buttons[self.current_B_row][self.current_b]["bg"] = "#3d3d3d"
                self.guess += key.upper()
                self.current_b += 1
            else:
                print(e.keysym)
        else:
            key_press = keyboard.widget
            if key_press["text"] == "Enter":
                self.check_for_match()
            elif key_press["text"] == "←":
                self.erase_character()
            else:
                if self.current_b == self.word_size:
                    self.current_b = self.word_size - 1

                    characters = list(self.guess)
                    characters[self.current_b] = ""
                    self.guess = "".join(characters)

                self.buttons[self.current_B_row][self.current_b]["text"] = key_press[
                    "text"
                ]
                self.guess += key_press["text"]
                self.current_b += 1

    def erase_character(self) -> None:
        """
        Удаляет последний введенный символ из текущего ввода.
        """
        if self.current_b > 0:
            self.current_b -= 1
            self.guess = self.guess[0 : self.current_b]

            self.buttons[self.current_B_row][self.current_b]["bg"] = BG
            self.buttons[self.current_B_row][self.current_b]["text"] = ""

    def check_for_match(self) -> None:
        """
        Проверяет введенное слово на совпадение с загаданным словом.

        Обновляет интерфейс в зависимости от результатов проверки (цвета кнопок и т.д.).
        """
        print("guess = ", self.guess)
        if len(self.guess) == self.word_size:
            self.guess_count += 1

            if self.word_api.is_valid_guess(self.guess):
                for button in self.buttons[self.current_B_row]:
                    button["bg"] = "green"

                # changing the keypad color
                self.change_keypad_color("#00ff2a", self.guess)

                self.won = True
                self.score += MAX_SCORE - 2 * (self.guess_count - 1)

                self.status_bar["text"] = f"Score : {self.score}"

                if self.score > self.high_score:
                    self.update_high_score()

                print("You won !!!")
                self.word_api.select_word()
                self.show_popup()
            else:
                if self.guess_count == 6:
                    print("You Lost !!!")
                    self.show_popup()
                    self.word_api.select_word()
                    return
                for i in range(self.word_size):
                    if self.word_api.is_at_right_position(i, self.guess[i]):
                        self.buttons[self.current_B_row][i]["bg"] = "green"

                        # changing the keypad color
                        self.change_keypad_color(
                            "#0fd630", self.guess[i], "#239436", "#0fd630"
                        )

                        characters = list(self.guess)
                        for index, char in enumerate(characters):
                            if self.word_api.is_at_right_position(i, char):
                                characters[index] = "/"

                        self.guess = "".join(characters)

                    elif self.word_api.is_in_word(self.guess[i]):
                        self.buttons[self.current_B_row][i]["bg"] = "#d0d925"

                        # changing the keypad color
                        self.change_keypad_color(
                            "#d0d925", self.guess[i], "#9ba128", "#d0d925"
                        )

                        characters = list(self.guess)
                        for index, char in enumerate(characters):
                            if char == self.guess[i] and index != i:
                                characters[index] = "/"

                        self.guess = "".join(characters)
                    else:
                        self.change_keypad_color(
                            "#4d4a4a", self.guess[i], "#3d3b3b", "#4d4a4a"
                        )

            self.current_b = 0
            self.current_B_row += 1
            self.guess = ""

    def reset(
        self, popup: Optional[tk.Toplevel] = None, keypad: Optional[tk.Button] = None
    ) -> None:
        """
        Сбрасывает состояние игры.

        Args:
            popup (Optional[tk.Toplevel]): Окно всплывающего сообщения о завершении игры.
            keypad (Optional[tk.Button]): Кнопка на клавиатуре, вызвавшая сброс.
        """
        if not keypad:
            for buttons_list in self.buttons:
                for button in buttons_list:
                    button["text"] = ""
                    button["bg"] = BG

        for buttons_list in self.keypad_buttons:
            for button in buttons_list:
                button["bg"] = BG
                button.bind("<Enter>", lambda e: on_hover(e, "#575656"))
                button.bind("<Leave>", lambda e: off_hover(e, BG))

        self.current_b = self.current_B_row = 0
        if not self.won:
            self.score = 0

        self.status_bar["text"] = f"Score : {self.score}"

        self.won = False
        self.guess_count = 0
        self.guess = ""

        self.root.attributes("-disabled", False)
        self.root.focus_get()
        if popup:
            popup.destroy()

    def show_popup(self) -> None:
        """
        Отображает всплывающее окно с результатами игры (выигрыш или проигрыш).
        """
        popup = tk.Toplevel()
        popup.title("Game Over")

        x_co = int(self.width / 2 - (450 / 2)) + self.x_co
        y_co = self.y_co + int(self.height / 2 - (250 / 2))

        popup.geometry(f"450x250+{x_co}+{y_co}")
        popup.configure(background="black")
        popup.wm_iconbitmap("images/icon.ico")
        popup.focus_force()

        status = "You Lost :("

        if self.won:
            status = "You Won !!!"

        status_label = tk.Label(
            popup, text=status, font="cambria 20 bold", fg="#14f41f", bg="black"
        )
        status_label.pack(pady=10)

        if not self.won:
            right_word = tk.Label(
                popup,
                text=f"The word was {self.word_api.word}",
                font="cambria 15 bold",
                fg="#14f41f",
                bg="black",
            )
            right_word.pack(pady=3)

        score_label = tk.Label(
            popup,
            text=f"Score : {self.score}",
            font="lucida 15 bold",
            fg="white",
            bg="black",
        )
        score_label.pack(pady=4)

        high_score_label = tk.Label(
            popup,
            text=f"High Score : {self.high_score}",
            font="lucida 15 bold",
            fg="white",
            bg="black",
        )
        high_score_label.pack(pady=4)

        button = tk.Button(
            popup,
            text="Okay",
            font="lucida 12 bold",
            fg="#00d0ff",
            cursor="hand2",
            bg="#252525",
            padx=10,
            command=lambda: self.reset(popup),
        )
        button.pack(pady=4)

        self.root.attributes("-disabled", True)

        def close() -> None:
            self.reset(popup)

        popup.protocol("WM_DELETE_WINDOW", close)

    def change_keypad_color(
        self,
        color: str,
        guess: str,
        on_hover_color: Optional[str] = None,
        off_hover_color: Optional[str] = None,
    ) -> None:
        """
        Изменяет цвет кнопок на клавиатуре в зависимости от состояния игры.

        Args:
            color (str): Цвет, который нужно установить для кнопок.
            guess (str): Введенное слово для проверки.
            on_hover_color (Optional[str]): Цвет кнопки при наведении.
            off_hover_color (Optional[str]): Цвет кнопки при отсутствии наведения.
        """
        for char in guess:
            if 65 <= ord(char) <= 70:
                btn_frame_index = 0
                btn_index = ord(char) - 65
            elif 71 <= ord(char) <= 80:
                btn_frame_index = 1
                btn_index = ord(char) - 71
            else:
                btn_frame_index = 2
                btn_index = ord(char) - 81

            if char == "/":
                return

            self.keypad_buttons[btn_frame_index][btn_index]["bg"] = color

            if on_hover_color:
                self.keypad_buttons[btn_frame_index][btn_index].bind(
                    "<Enter>", lambda e: on_hover(e, on_hover_color)
                )
                self.keypad_buttons[btn_frame_index][btn_index].bind(
                    "<Leave>", lambda e: off_hover(e, off_hover_color)
                )

    def get_from_db(self) -> None:
        """
        Получает настройки игры из базы данных или создает новую, если она не существует.
        """
        if not os.path.exists("settings.db"):
            connection = sqlite3.connect("settings.db")
            cursor = connection.cursor()
            cursor.execute(
                "CREATE TABLE info(id integer, word_length integer, high_score integer)"
            )
            cursor.execute("INSERT INTO info VALUES(?,?,?)", (0, 5, 0))

            self.word_api = words_api.Words(self.word_size)

            connection.commit()
            cursor.execute("SELECT * FROM info")
            connection.close()
        else:
            connection = sqlite3.connect("settings.db")
            cursor = connection.cursor()

            cursor.execute("SELECT * FROM info")

            data = cursor.fetchall()
            self.word_size = data[0][1]
            self.high_score = data[0][2]

            self.word_api = words_api.Words(self.word_size)

            connection.close()

    def update_high_score(self) -> None:
        """
        Обновляет рекордный счет в базе данных.
        """
        connection = sqlite3.connect("settings.db")
        cursor = connection.cursor()

        self.high_score = self.score
        print("update score = ", self.high_score)
        cursor.execute(f"UPDATE info SET high_score={self.score} WHERE id=0")
        connection.commit()

        connection.close()

    def open_setting(self) -> None:
        """
        Открывает окно настроек игры.
        """
        setting = st.Settings(self)

    def on_hover(self, e: tk.Event) -> None:
        """
        Обрабатывает событие наведения мыши на элемент интерфейса.

        Args:
            e (tk.Event): Событие наведения мыши.
        """
        widget = e.widget
        widget["image"] = self.setting_dark

    def off_hover(self, e: tk.Event) -> None:
        """
        Обрабатывает событие ухода мыши с элемента интерфейса.

        Args:
            e (tk.Event): Событие ухода мыши.
        """
        widget = e.widget
        widget["image"] = self.setting


def on_hover(e: tk.Event, color: str) -> None:
    """
    Изменяет цвет кнопки при наведении.

    Args:
        e (tk.Event): Событие наведения мыши.
        color (str): Цвет, который нужно установить для кнопки.
    """
    button = e.widget
    button["bg"] = color


def off_hover(e: tk.Event, color: str) -> None:
    """
    Возвращает цвет кнопки при уходе мыши.

    Args:
        e (tk.Event): Событие ухода мыши.
        color (str): Цвет, который нужно установить для кнопки.
    """
    button = e.widget
    button["bg"] = color


if __name__ == "__main__":
    Wordle()
