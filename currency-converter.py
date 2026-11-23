import tkinter as tk
from tkinter import ttk
import time
import requests
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import style


class Convert:
    URL = "https://www.x-rates.com/table/?from=INR&amount=1"
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/78.0.3904.87 Safari/537.36"
    )

    def __init__(self, root):
        self.root = root
        self.root.title("Currency Converter | Foreign Exchange Rates")
        self.root.geometry("1350x800+0+0")
        self.root.configure(background="#F2C496")

        # Оптимизация 1: парсим таблицу ОДИН раз при запуске,
        # сохраняем в словарь {имя: курс} — поиск становится O(1)
        self.rates_dict = self._load_rates_once()

        style.use("ggplot")
        self.fig = plt.Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.date_of_convert = tk.StringVar()
        self.selected_currency = tk.StringVar()
        self.amount_var = tk.DoubleVar()
        self.result_var = tk.DoubleVar()
        self.amount_var.set(1)
        self.date_of_convert.set(time.strftime("%d/%m/%y"))

        self._build_frames()
        self._build_widgets()

    def _load_rates_once(self):
        """Загружает и парсит курсы один раз, возвращает словарь {имя: курс}."""
        response = requests.get(
            self.URL, headers={"User-Agent": self.USER_AGENT}, timeout=10
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", class_="tablesorter ratesTable")

        rates = {}
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 2:
                name = cells[0].get_text(strip=True)
                try:
                    rates[name] = float(cells[1].get_text(strip=True))
                except ValueError:
                    continue
        return rates

    def _build_frames(self):
        self.main_frame = tk.Frame(
            self.root, bd=10, width=1350, height=700,
            padx=11, pady=10, bg="#F2C496", relief=tk.RIDGE
        )
        self.main_frame.grid(row=1, column=0)

        self.frame_2_top = tk.Frame(
            self.main_frame, width=250, height=300,
            bd=4, padx=80, pady=2, relief=tk.RIDGE
        )
        self.frame_2_top.grid(row=0, column=0)

        self.frame_2_bottom = tk.Frame(
            self.main_frame, width=800, height=300,
            bd=4, padx=5, pady=2, relief=tk.RIDGE
        )
        self.frame_2_bottom.grid(row=1, column=0)

        self.frame_2_bottom_l = tk.Frame(
            self.frame_2_bottom, width=450, height=300,
            bd=4, padx=0, pady=2, relief=tk.RIDGE
        )
        self.frame_2_bottom_l.grid(row=0, column=0)

        self.frame_2_bottom_r = tk.Frame(
            self.frame_2_bottom, width=350, height=300,
            bd=4, padx=10, pady=2, relief=tk.RIDGE
        )
        self.frame_2_bottom_r.grid(row=0, column=1)

    def _build_widgets(self):
        tk.Label(
            self.frame_2_top, font=("arial", 20, "bold"), text="Todays Date",
            padx=2, pady=10, bd=2, width=18
        ).grid(row=0, column=0)
        tk.Label(
            self.frame_2_top, font=("arial", 20, "bold"),
            textvariable=self.date_of_convert,
            padx=2, pady=10, bd=2, width=12, justify="center"
        ).grid(row=0, column=1)

        tk.Label(
            self.frame_2_top, font=("arial", 20, "bold"), text="INR",
            padx=2, pady=10, width=19, bd=2
        ).grid(row=1, column=0)
        tk.Entry(
            self.frame_2_top, font=("arial", 20, "bold"),
            textvariable=self.amount_var, bd=2, width=23, justify="center"
        ).grid(row=1, column=1, pady=10)

        self.option_cb = ttk.Combobox(
            self.frame_2_top, textvariable=self.selected_currency,
            state="readonly", font=("arial", 20, "bold"), width=20
        )
        self.option_cb["values"] = list(self.rates_dict.keys())
        self.option_cb.current(0)
        self.option_cb.grid(row=2, column=0, padx=38, pady=10)

        tk.Label(
            self.frame_2_top, font=("arial", 20, "bold"),
            textvariable=self.result_var, bd=2, width=20, bg="white",
            pady=2, padx=2, relief="sunken"
        ).grid(row=2, column=1)

        # Оптимизация 2: создаём matplotlib canvas 1 раз при инициализации
        self.canvas = FigureCanvasTkAgg(self.fig, self.frame_2_bottom_l)
        self.canvas.get_tk_widget().grid(row=0, column=0)
        self.ax.set_xlabel("INR")
        self.ax.set_ylabel("Converted Value")
        self.ax.set_title("Currency Conversion")

        tk.Button(
            self.frame_2_bottom_r, text="Convert",
            font=("arial", 20, "bold"), width=14, height=2,
            command=self.convert_value
        ).grid(row=4, column=0)
        tk.Button(
            self.frame_2_bottom_r, text="Reset",
            font=("arial", 20, "bold"), width=14, height=2,
            command=self.reset
        ).grid(row=5, column=0)
        tk.Button(
            self.frame_2_bottom_r, text="Exit",
            font=("arial", 20, "bold"), width=14, height=2,
            command=self.root.destroy
        ).grid(row=6, column=0)

    def reset(self):
        self.selected_currency.set("")
        self.amount_var.set(1)
        self.result_var.set("0.00")

    def convert_value(self):
        currency_name = self.selected_currency.get()
        # Оптимизация 1: мгновенный поиск O(1) вместо медленного Soup.find
        rate = self.rates_dict.get(currency_name, 1.0)

        result_value = self.amount_var.get() * rate
        self.result_var.set(f"{result_value:.2f}")

        self.update_chart(self.amount_var.get(), result_value)

    def update_chart(self, amount, result_value):
        # Оптимизация 2: обновляем данные существующего scatter
        if not hasattr(self, "_scatter_plot") or self._scatter_plot is None:
            self._scatter_plot = self.ax.scatter([amount * 2], [result_value])
        else:
            self._scatter_plot.set_offsets([[amount * 2, result_value]])
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    application = Convert(root)
    root.mainloop()