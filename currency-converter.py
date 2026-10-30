# Importing the required Modules
import time
import tkinter.messagebox as messagebox
from tkinter import (
    RIDGE,
    VERTICAL,
    Button,
    Canvas,
    DoubleVar,
    Entry,
    Frame,
    Label,
    Scale,
    StringVar,
    Tk,
    ttk,
)

import requests as req
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Convert:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Currency Converter | Foreign Exchange Rates")
        self.root.geometry("1350x800+0+0")
        self.root.configure(background="#F2C496")

        # Инициализация переменных и парсинга как атрибутов класса
        self.url = "https://www.x-rates.com/table/?from=INR&amount=1"
        self.header = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/78.0.3904.87 Safari/537.36"
            )
        }
        self.webpage = req.get(self.url, headers=self.header)
        self.soup = BeautifulSoup(self.webpage.text, "html.parser")

        style.use("ggplot")
        self.fig3 = plt.Figure(figsize=(5, 3), dpi=100)
        self.ax3 = self.fig3.add_subplot(111)

        self.date_of_convert = StringVar()
        self.value_0 = StringVar()
        self.convert_var = DoubleVar()
        self.currency_var = DoubleVar()
        self.var1 = DoubleVar()
        self.var2 = DoubleVar()

        self.convert_var.set(1)
        self.date_of_convert.set(time.strftime("%d/%m/%y"))

        # ===================== Frames ==========================================
        title_frame = Frame(
            self.root,
            bd=10,
            width=1350,
            height=100,
            padx=10,
            pady=10,
            bg="#F2C496",
            relief=RIDGE,
        )
        title_frame.grid(row=0, column=0)

        self.lbl_Title = Label(
            title_frame,
            text="Currency Converter | Foreign Exchange Rates",
            padx=17,
            pady=4,
            bd=1,
            font=("arial", 30, "bold"),
            bg="#02A9FC",
            width=50,
        )
        self.lbl_Title.pack()

        main_frame = Frame(
            self.root,
            bd=10,
            width=1350,
            height=700,
            padx=11,
            pady=10,
            bg="#F2C496",
            relief=RIDGE,
        )
        main_frame.grid(row=1, column=0)

        frame_1 = Frame(
            main_frame, bd=4, width=100, height=600, padx=5, pady=1, relief=RIDGE
        )
        frame_1.grid(row=0, column=0)

        frame_2 = Frame(
            main_frame, bd=4, width=800, height=600, padx=0, pady=2, relief=RIDGE
        )
        frame_2.grid(row=0, column=1)

        frame_2_top = Frame(
            frame_2, width=250, height=300, bd=4, padx=80, pady=2, relief=RIDGE
        )
        frame_2_top.grid(row=0, column=0)

        frame_2_bottom = Frame(
            frame_2, width=800, height=300, bd=4, padx=5, pady=2, relief=RIDGE
        )
        frame_2_bottom.grid(row=1, column=0)

        self.frame_2_bottom_l = Frame(
            frame_2_bottom, width=450, height=300, bd=4, padx=0, pady=2, relief=RIDGE
        )
        self.frame_2_bottom_l.grid(row=0, column=0)

        frame_2_bottom_r = Frame(
            frame_2_bottom, width=350, height=300, bd=4, padx=10, pady=2, relief=RIDGE
        )
        frame_2_bottom_r.grid(row=0, column=1)

        frame_3 = Frame(
            main_frame, bd=4, width=100, height=600, padx=5, pady=1, relief=RIDGE
        )
        frame_3.grid(row=0, column=2)

        # =============================== Widgets =================================
        self.lblToday = Label(
            frame_2_top,
            font=("arial", 20, "bold"),
            text="Todays Date",
            padx=2,
            pady=10,
            bd=2,
            width=18,
        )
        self.lblToday.grid(row=0, column=0)

        self.lblDate = Label(
            frame_2_top,
            font=("arial", 20, "bold"),
            textvariable=self.date_of_convert,
            padx=2,
            pady=10,
            bd=2,
            width=12,
            justify="center",
        )
        self.lblDate.grid(row=0, column=1)

        self.Converted = Scale(
            frame_3,
            variable=self.var1,
            from_=0,
            to=1000,
            length=500,
            tickinterval=50,
            orient=VERTICAL,
            bg="#EA2A97",
            label="Convert",
            font=("arial", 10, "bold"),
        )
        self.Converted.grid(row=0, column=0, rowspan=2)

        self.INR = Scale(
            frame_1,
            variable=self.var2,
            from_=0,
            to=1000,
            length=500,
            tickinterval=50,
            orient=VERTICAL,
            bg="#EA2A97",
            label="INR",
            font=("arial", 10, "bold"),
        )
        self.INR.grid(row=0, column=0, rowspan=2)

        self.lbl_INR = Label(
            frame_2_top,
            font=("arial", 20, "bold"),
            text="INR",
            padx=2,
            pady=10,
            width=19,
            bd=2,
        )
        self.lbl_INR.grid(row=1, column=0)

        self.EntCurrency = Entry(
            frame_2_top,
            font=("arial", 20, "bold"),
            textvariable=self.convert_var,
            bd=2,
            width=23,
            justify="center",
        )
        self.EntCurrency.grid(row=1, column=1, pady=10)

        self.Option = ttk.Combobox(
            frame_2_top,
            textvariable=self.value_0,
            state="readonly",
            font=("arial", 20, "bold"),
            width=20,
        )
        # Разбито на короткие строки для соответствия PEP 8
        self.Option["values"] = (
            "Select",
            "Argentine Peso",
            "Australian Dollar",
            "Bahraini Dinar",
            "Batwana Pula",
            "Brazilian Real",
            "British Pound",
            "Bruneian Dollar",
            "Bulgarian Lev",
            "Canadian Dollar",
            "Chilean Peso",
            "Chinese Yuan Renmimbi",
            "Colombian Peso",
            "Crotian Kuna",
            "Czech Koruna",
            "Danish Krone",
            "Emirati Dirhan",
            "Euro",
            "Hong Kong Dollar",
            "Hungarian Foint",
            "IceLandic Krona",
            "Indonesian Rupiah",
            "Iranian Rial",
            "Israeli Shekel",
            "Japanese Yen",
            "Kazakhstani Tenge",
            "Kuwaiti Dinar",
            "Libyan Dinar",
            "Malaysian Ringgit",
            "Mauritian Rupee",
            "Mexican Peso",
            "Nepalese Rupee",
            "New Zealand Dollar",
            "Norwegian Krone",
            "Omani Rial",
            "Pakistani Rupee",
            "Philippine Peso",
            "Polish Zloty",
            "Qatari Riyal",
            "Romanian New Leu",
            "Russian Ruble",
            "Saudi Arabian Riyal",
            "Singapore Dollar",
            "South African Rand",
            "South Korean Won",
            "Sri Lankan Rupee",
            "Swedish Krona",
            "Swiss Franc",
            "Taiwan New Dollar",
            "Thai Baht",
            "Trinidadian Dollar",
            "Turkish Lira",
            "US Dollar",
            "Venezuelan Bolivar",
            "",
        )
        self.Option.current(0)
        self.Option.grid(row=2, column=0, padx=38, pady=10)

        self.lblCurrency = Label(
            frame_2_top,
            font=("arial", 20, "bold"),
            textvariable=self.currency_var,
            bd=2,
            width=20,
            bg="white",
            pady=2,
            padx=2,
            relief="sunken",
        )
        self.lblCurrency.grid(row=2, column=1)

        self.canvas = Canvas(self.frame_2_bottom_l, width=600, height=300, bg="gray")

        self.btnconvert = Button(
            frame_2_bottom_r,
            text="Convert",
            padx=2,
            pady=8,
            bd=2,
            fg="black",
            font=("arial", 20, "bold"),
            width=14,
            height=2,
            command=self.convert_value,
        )
        self.btnconvert.grid(row=4, column=0)

        self.btnReset = Button(
            frame_2_bottom_r,
            text="Reset",
            padx=2,
            pady=7,
            fg="black",
            font=("arial", 20, "bold"),
            width=14,
            height=2,
            command=self.reset,
        )
        self.btnReset.grid(row=5, column=0)

        self.btnExit = Button(
            frame_2_bottom_r,
            text="Exit",
            padx=2,
            pady=8,
            fg="black",
            font=("arial", 20, "bold"),
            width=14,
            height=2,
            command=self.c_exit,
        )
        self.btnExit.grid(row=6, column=0)

    # ============================ Methods ==================================

    def c_exit(self) -> None:
        exit_confirmed = messagebox.askyesno(
            "Exit System", "Confirm if you want to exit"
        )
        if exit_confirmed:
            self.root.destroy()

    def reset(self) -> None:
        self.value_0.set("")
        self.convert_var.set(1)
        self.currency_var.set("0.00")
        self.var1.set(0)
        self.var2.set(0)

    def convert_value(self) -> None:
        s = self.value_0.get()
        if s == "Select" or not s:
            return

        rate = (
            self.soup.find("table", class_="tablesorter ratesTable")
            .find("td", text=lambda x: x == s)
            .find_next("td")
            .get_text()
        )
        self.canvas_cal()
        x = float(self.convert_var.get()) * float(rate)
        final_value = f"{x:.2f}"

        self.currency_var.set(final_value)
        self.var1.set(final_value)
        self.var2.set(self.convert_var.get())

    def canvas_cal(self) -> None:
        s = self.value_0.get()
        c1 = int(self.convert_var.get())
        u = (
            self.soup.find("table", class_="tablesorter ratesTable")
            .find("td", text=lambda x: x == s)
            .find_next("td")
            .get_text()
        )
        c2 = float(c1) * float(u)
        print(c1, c2)

        scatter3 = FigureCanvasTkAgg(self.fig3, self.frame_2_bottom_l)
        scatter3.get_tk_widget().grid(row=0, column=0)
        self.ax3.set_xlabel("INR")
        self.ax3.set_ylabel("Converted Value")
        self.ax3.set_title("Currency Conversion")


if __name__ == "__main__":
    root = Tk()
    application = Convert(root)
    root.mainloop()
