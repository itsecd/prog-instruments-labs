import tkinter as tk
from alarm_clock import AlarmClock
from dynamic_clock import Clock


class MainApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Alarm & Dynamic Clock")
        self.geometry("500x500")
        self.resizable(False, False)

        tk.Label(
            self,
            text="Clock Application",
            font=("Helvetica", 22, "bold")
        ).pack(pady=10)

        self.alarm_app = AlarmClock(self)
        self.dynamic_clock = Clock(self)


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
