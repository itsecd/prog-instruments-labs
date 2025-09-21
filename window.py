import subprocess
import sys
import json
from multiprocessing import Process
from threading import Thread
import time
from tkinter import Tk, Label, Frame, Button, Scrollbar, Text, WORD, BOTH, RIGHT
from PIL import Image, ImageTk

import CONST
import functions
from functions import CardFinder


class App:
    def __init__(self, class_card_finder_instance):
        self.root = Tk()
        self.setup_ui()

        self.console = None
        self.setup_console()

        self.awaiting_input = False
        self.input_callback = None

        self.card_finder = class_card_finder_instance


    def setup_ui(self):
        self.root.title("01001000 011䷄䷅䷆䷇0001 0101䷁01011 ䷀䷂1䷃0䷈䷉䷊䷋ █▒▒0011 01▒▒▒▒▒▒▒ 10%")
        self.root.geometry("900x650")
        self.root.resizable(False, False)

        img = Image.open("icon.jpg").resize((32,32))
        icon = ImageTk.PhotoImage(img)
        self.root.iconphoto(True, icon)

        bg_image = Image.open("background.jpg").resize((900, 650), Image.Resampling.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_image)

        bg_label = Label(self.root, image=bg_photo)
        bg_label.image = bg_photo
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        button_frame = Frame(self.root, bg='', padx=10, pady=10)
        button_frame.place(relx=0.5, rely=0.5, anchor="center")

        Frame(self.root, height=40).pack()
        Button(self.root, text="Find a card number", width=30, height=2,
               command=self.on_find_card_number).pack(pady=10)
        Button(self.root, text="Check the report", width=30, height=2,
               command=self.on_check_the_report).pack(pady=10)
        Button(self.root, text="Luhn algorithm", width=30, height=2,
               command=self.on_luhn_algorithm).pack(pady=10)
        Button(self.root, text="Experiment", width=30, height=2,
               command=self.on_experiment).pack(pady=10)

        Button(self.root, text="Clear", width=10, height=1,
               command=self.on_clear).place(x=770,y=590)


    def setup_console(self):
        console_frame = Frame(self.root, bg='#202020', bd=1)
        console_frame.place(relx=0.5, rely=0.7, anchor="center", width=800, height=250)

        console_scroll = Scrollbar(console_frame)
        console_scroll.pack(side=RIGHT)

        self.console = Text(console_frame,
                       bg='#202020',
                       fg='white',
                       font=('Courier', 10),
                       yscrollcommand=console_scroll.set,
                       wrap=WORD)
        self.console.pack(expand=True, fill=BOTH)
        console_scroll.config(command=self.console.yview)


    def write_to_console(self, text: str, color = "white"):
        def typewriter_effect():
            delay = 0.05

            self.console.tag_config("green", foreground="green")
            self.console.tag_config("red", foreground="red")
            self.console.tag_config("yellow", foreground="yellow")
            self.console.tag_config("white", foreground="white")

            tag_colors = {
                "success": "green",
                "fail": "red",

                "status": "yellow",
                "card_number": "yellow",
                "bin": "yellow",
                "hash": "yellow",
                "last_four_numbers": "yellow",
                "processes_used": "yellow"
            }

            words = text.split()
            for word in words:
                clean_word = word.strip(":").lower()
                new_color = tag_colors.get(clean_word)

                if len(word) >=50:
                    delay = 0.01

                for char in word:
                    if new_color:
                        self.console.insert("end", char, new_color)
                    else:
                        self.console.insert("end", char, color)

                    self.console.see("end")
                    time.sleep(delay)
                    self.console.update()

                self.console.insert("end", " ")
                delay = 0.025

            self.console.insert("end","\n")

            self.console.see("end")

        Thread(target=typewriter_effect(), daemon=True).start()


    def on_find_card_number(self):
        self.write_to_console("\n")
        self.write_to_console("The card number is being selected...")

        card_number = None

        try:
            for bin in CONST.SBERBANK_VISA_DEBIT_BINS:
                card_number = CardFinder.find_card_number(self.card_finder, bin)
                if card_number:
                    self.write_to_console(f"Result was found: {card_number}")
                    CardFinder.write_report(self.card_finder,CardFinder.get_report(self.card_finder, card_number, bin))
                    self.write_to_console(f"Result was saved to report")
                    break
        except Exception as e:
            self.write_to_console(f"Result wasn't found! {e}", "red")


    def on_luhn_algorithm(self):
        self.write_to_console("\n")
        try:
            card_number = CardFinder.get_json_data(self.card_finder)["card_number"]

            self.write_to_console(f"Initialization of the algorithm: ", "yellow")
            self.write_to_console(f"card_number: {card_number}")

            numbers = [int(d) for d in reversed(card_number)]
            self.write_to_console(f"Reverse: {numbers}")

            for i in range(1, len(card_number), 2):
                doubled = numbers[i] * 2
                numbers[i] = doubled - 9 if doubled > 9 else doubled
            self.write_to_console(f"Doubling digits in odd positions: {numbers}")
            self.write_to_console(f"Sum: {sum(numbers)}")
            self.write_to_console(f"Remainder of the division by 10: {sum(numbers) % 10}")

            if sum(numbers)%10 == 0:
                self.write_to_console(f"Card number is correct", "green")
            else:
                self.write_to_console(f"Card number isn't correct", "red")
        except KeyError:
            self.write_to_console(f"Report is empty!", "red")


    def on_check_the_report(self):
        self.write_to_console("\n")
        data = CardFinder.get_json_data(self.card_finder)

        if data == {}:
            self.write_to_console(f"Report is empty!", "red")
            return

        for key, value in data.items():
            self.write_to_console(f"{key}: {value}")


    def on_experiment(self):
        self.write_to_console("\n")
        self.write_to_console("Running experiment...", "yellow")

        settings = {
            "hash": self.card_finder.hash,
            "last_four": self.card_finder.last_four,
            "json_res": self.card_finder.json_res
        }

        settings_path = "temp_settings.json"
        with open(settings_path, 'w') as f:
            json.dump(settings, f)

        python_exe = sys.executable
        subprocess.Popen(
            [python_exe, "experiment_runner.py", settings_path, str(int(CardFinder.get_num_processes() * 1.5))])


    def on_clear(self):
        self.console.delete("1.0", "end")


    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    card_finder = CardFinder()
    CardFinder.clear_report(card_finder)
    app = App(card_finder)
    app.run()
