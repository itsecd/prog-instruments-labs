import tkinter as tk
import datetime
import time
from threading import Thread
from pygame import mixer
from style import STYLE


class AlarmClock(tk.Frame):
    """
    Alarm Clock GUI class
    """
    def __init__(self, master):
        super().__init__(master, bg=STYLE["colors"]["bg"])
        self.pack(pady=STYLE["padding"]["medium"])

        self.__running = False
        self.__hour = tk.StringVar(value="00")
        self.__minute = tk.StringVar(value="00")
        self.__second = tk.StringVar(value="00")

        mixer.init()
        self.__create_widgets()

    def __create_widgets(self):
        """
        Create the widgets
        :return: None
        """
        tk.Label(
            self,
            text="Alarm Clock",
            font=STYLE["fonts"]["title"],
            fg=STYLE["colors"]["highlight"],
            bg=STYLE["colors"]["bg"],
        ).pack(pady=STYLE["padding"]["medium"])

        tk.Label(
            self,
            text="Set time (HH:MM:SS)",
            font=STYLE["fonts"]["label"],
            bg=STYLE["colors"]["bg"],
            fg=STYLE["colors"]["fg"],
        ).pack()

        frame = tk.Frame(self, bg=STYLE["colors"]["bg"])
        frame.pack(pady=STYLE["padding"]["small"])

        hours = [f"{i:02d}" for i in range(24)]
        minutes_seconds = [f"{i:02d}" for i in range(60)]

        tk.OptionMenu(frame, self.__hour, *hours).pack(side="left", padx=5)
        tk.OptionMenu(frame, self.__minute, *minutes_seconds).pack(side="left", padx=5)
        tk.OptionMenu(frame, self.__second, *minutes_seconds).pack(side="left", padx=5)

        tk.Button(
            self,
            text="Set Alarm",
            font=STYLE["fonts"]["button"],
            bg=STYLE["colors"]["primary"],
            fg=STYLE["colors"]["fg"],
            command=self.__start_alarm_thread,
        ).pack(pady=STYLE["padding"]["medium"])

        tk.Button(
            self,
            text="Stop Alarm",
            bg=STYLE["colors"]["danger"],
            fg=STYLE["colors"]["fg"],
            font=STYLE["fonts"]["button"],
            command=self.__stop_alarm,
        ).pack(pady=STYLE["padding"]["small"])

    def __start_alarm_thread(self):
        """
        Start the alarm clock thread
        :return: None
        """
        if not self.__running:
            self.__running = True
            Thread(target=self.__alarm_loop, daemon=True).start()
            print("Alarm Clock started")

    def __alarm_loop(self):
        """
        Checks the current time every second and print when it's time
        :return: None
        """
        set_time = f"{self.__hour.get()}:{self.__minute.get()}:{self.__second.get()}"
        print(f"Alarm set for: {set_time}")

        while self.__running:
            time.sleep(1)
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            if current_time == set_time:
                print("Wake Up!")
                break

    def __stop_alarm(self):
        """
        Stop the alarm clock thread
        :return: None
        """
        if self.__running:
            self.__running = False
            try:
                mixer.music.stop()
            except Exception:
                pass
            print("Alarm Clock stopped")
