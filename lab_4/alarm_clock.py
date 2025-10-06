import datetime
import time
import tkinter as tk
from threading import Thread
from pygame import mixer


class Alarm_Clock(tk.Frame):
    """
    Alarm Clock GUI class
    """
    def __init__(self, master):
        super().__init__(master)
        self.pack(pady=10)
        self.create_widgets()
        self.running = False

    def create_widgets(self):
        """
        Create the widgets
        :return: None
        """
        tk.Label(self, text="Alarm Clock", font=("Helvetica", 20, "bold"),fg="blue").pack(pady=10)
        tk.Label(self,text="Set time (HH:MM:SS)",font=("Helvetica",14))

        frame = tk.Frame(self)
        frame.pack()

        self.hour = tk.StringVar(value ="00")
        self.minute = tk.StringVar(value ="00")
        self.second = tk.StringVar(value ="00")

        hours = [f"{i:02d}" for i in range(0, 24)]
        minutes_seconds = [f"{i:02d}" for i in range(0, 60)]

        tk.OptionMenu(frame,self.hour, *hours).pack(side="left",padx=5)
        tk.OptionMenu(frame, self.minute, *minutes_seconds).pack(side="left", padx=5)
        tk.OptionMenu(frame, self.second, *minutes_seconds).pack(side="left", padx=5)

        tk.Button(
            self,
            text="Set Alarm",
            font=("Helvetica", 14),
            bg="#4CAF50",
            fg="white",
            command=self.start_alarm_thread).pack(pady=10)

        tk.Button(
            self,
            text="Stop Alarm",
            bg="red",
            fg="white",
            font=("Helvetica", 12),
            command=self.stop_alarm,
        ).pack(pady=5)

    def start_alarm_thread(self):
        """
        Start the alarm clock thread
        :return: None
        """
        if not self.running:
            self.running = True
            Thread(target=self.alarm_loop, daemon=True).start()
            print("Alarm Clock started")

    def alarm_loop(self):
        """
        Checks the current time every second and print when it's time
        :return: None
        """
        set_time = f"{self.hour.get()}:{self.minute.get()}:{self.second.get()}"
        print(f"Alarm set for: {set_time}")

        while self.running:
            time.sleep(1)
            current_time = datetime.datetime.now().strftime("%H:%M:%S")

            if current_time == set_time:
                print("Wake Up!")
                break


    def stop_alarm(self):
        """
        Stop the alarm clock thread
        :return: None
        """
        if self.running:
            self.running = False
            try:
                mixer.music.stop()
            except:
                pass
            print("Alarm Clock stopped")