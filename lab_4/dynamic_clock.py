import tkinter as tk
from time import strftime
from style import STYLE


class Clock(tk.Frame):
    """
    Clock
    """
    def __init__(self, master):
        super().__init__(master, bg=STYLE["colors"]["bg"])
        self.pack(pady=STYLE["padding"]["medium"])

        self.__clock_label = None

        self.__create_widgets()
        self.__update_time()

    def __create_widgets(self):
        """
        Create widgets for users
        :return: None
        """
        self.__clock_label = tk.Label(
            self,
            font=STYLE["fonts"]["clock"],
            bg=STYLE["colors"]["bg"],
            fg=STYLE["colors"]["fg"],
            padx=STYLE["padding"]["large"],
            pady=STYLE["padding"]["medium"],
        )
        self.__clock_label.pack(anchor="center")

    def __get_time_of_day(self, hour: int) -> str:
        """
        Get current time of day
        :param hour: current time
        :return: time of day
        """
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        return "Evening"

    def __update_time(self):
        """
        Update time of day
        :return: None
        """
        current_time = strftime("%H:%M:%S")
        hour = int(strftime("%H"))
        time_of_day = self.__get_time_of_day(hour)

        self.__clock_label.config(text=f"{current_time}\nGood {time_of_day}!")

        bg_color = (
            STYLE["colors"]["morning"]
            if time_of_day == "Morning"
            else STYLE["colors"]["afternoon"]
            if time_of_day == "Afternoon"
            else STYLE["colors"]["evening"]
        )

        self.master.configure(bg=bg_color)

        self.after(1000, self.__update_time)
