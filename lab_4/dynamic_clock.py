import tkinter as tk
from time import strftime

class Clock(tk.Frame):
    """
    Clock
    """
    def __init__(self, master):
        super().__init__(master)
        self.pack(pady=10)
        self.create_widgets()
        self.update_time()


    def create_widgets(self):
        """
        Create widgets for users
        :return: None
        """
        self.clock_label = tk.Label(self, font=("Courier New", 40, "bold"), bg="black", fg="white",padx=20, pady=10)
        self.clock_label.pack(anchor="center")


    def get_time_of_day(self,hour):
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

    def update_time(self):
        """
        Update time of day
        :return: None
        """
        current_time = strftime("%H:%M:%S")
        hour = int(strftime("%H"))
        time_of_day = self.get_time_of_day(hour)

        self.clock_label.config(text=f"{current_time}\nGood {time_of_day}")

        color = ("lightblue"
            if time_of_day == "Morning"
            else "lightyellow" if time_of_day == "Afternoon" else "lightcoral")

        self.master.configure(bg=color)
        self.after(1000, self.update_time)



