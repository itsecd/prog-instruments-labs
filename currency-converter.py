from tkinter import Tk, StringVar, DoubleVar, Frame, Label, Entry, Button, Scale, Canvas, RIDGE, VERTICAL
from tkinter import ttk
import tkinter.messagebox
import time
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import style

from rate_service import RateService

# Размеры окна
WINDOW_WIDTH = 1350
WINDOW_HEIGHT = 800
WINDOW_GEOMETRY = f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+0+0"

# Цвета
BACKGROUND_COLOR = "#F2C496"
TITLE_BACKGROUND_COLOR = "#02A9FC"
SCALE_BACKGROUND_COLOR = "#EA2A97"
CANVAS_BACKGROUND_COLOR = "gray"
LABEL_BACKGROUND_COLOR = "white"

# Шрифты
TITLE_FONT = ("arial", 30, "bold")
LABEL_FONT = ("arial", 20, "bold")
SCALE_FONT = ("arial", 10, "bold")

# Размеры фреймов (в пикселях)
TITLE_FRAME_WIDTH = 1350
TITLE_FRAME_HEIGHT = 100
MAIN_FRAME_WIDTH = 1350
MAIN_FRAME_HEIGHT = 700
SIDE_FRAME_WIDTH = 100
SIDE_FRAME_HEIGHT = 600
CENTER_FRAME_WIDTH = 800
CENTER_FRAME_HEIGHT = 600
TOP_SUBFRAME_WIDTH = 250
TOP_SUBFRAME_HEIGHT = 300
BOTTOM_SUBFRAME_WIDTH = 800
BOTTOM_SUBFRAME_HEIGHT = 300
BOTTOM_LEFT_SUBFRAME_WIDTH = 450
BOTTOM_RIGHT_SUBFRAME_WIDTH = 350

LARGE_PADDING = 10
MEDIUM_PADDING = 5
SMALL_PADDING = 2
TITLE_PADDING_X = 17
TITLE_PADDING_Y = 4
LABEL_PADDING = 2
LABEL_PADDING_Y = 10

TITLE_LABEL_WIDTH = 50
DATE_LABEL_WIDTH = 18
DATE_VALUE_WIDTH = 12
INR_LABEL_WIDTH = 19
CURRENCY_ENTRY_WIDTH = 23
COMBOBOX_WIDTH = 20
CURRENCY_LABEL_WIDTH = 20
BUTTON_WIDTH = 14
BUTTON_HEIGHT = 2

SCALE_FROM = 0
SCALE_TO = 1000
SCALE_LENGTH = 500
SCALE_TICK_INTERVAL = 50

FIGURE_WIDTH = 5
FIGURE_HEIGHT = 3
FIGURE_DPI = 100

LARGE_BORDER = 10
MEDIUM_BORDER = 4
SMALL_BORDER = 2
TITLE_BORDER = 1


class Convert:

    def __init__(self, root):
        self.root = root
        self._setup_window()
        self._init_variables()
        self._build_frames()
        self._build_widgets()

    def _setup_window(self):
        self.root.title("Currency Converter | Foreign Exchange Rates")
        self.root.geometry(WINDOW_GEOMETRY)
        self.root.configure(background=BACKGROUND_COLOR)

        self.rate_service = RateService()
        self.rate_service.load()

        style.use("ggplot")
        self.fig = plt.Figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=FIGURE_DPI)
        self.ax = self.fig.add_subplot(111)

    def _init_variables(self):
        self.date_of_convert = StringVar()
        self.value_0 = StringVar()
        self.convert = DoubleVar()
        self.currency = DoubleVar()
        self.var1 = DoubleVar()
        self.var2 = DoubleVar()
        self.convert.set(1)
        self.date_of_convert.set(time.strftime("%d/%m/%y"))

    def _build_frames(self):
        self.title_frame = Frame(
            self.root, bd=LARGE_BORDER, width=TITLE_FRAME_WIDTH, height=TITLE_FRAME_HEIGHT,
            padx=LARGE_PADDING, pady=LARGE_PADDING, bg=BACKGROUND_COLOR, relief=RIDGE
        )
        self.title_frame.grid(row=0, column=0)

        self.main_frame = Frame(
            self.root, bd=LARGE_BORDER, width=MAIN_FRAME_WIDTH, height=MAIN_FRAME_HEIGHT,
            padx=LARGE_PADDING, pady=LARGE_PADDING, bg=BACKGROUND_COLOR, relief=RIDGE
        )
        self.main_frame.grid(row=1, column=0)

        self.frame_1 = Frame(
            self.main_frame, bd=MEDIUM_BORDER, width=SIDE_FRAME_WIDTH, height=SIDE_FRAME_HEIGHT,
            padx=MEDIUM_PADDING, pady=SMALL_PADDING, relief=RIDGE
        )
        self.frame_1.grid(row=0, column=0)

        self.frame_2 = Frame(
            self.main_frame, bd=MEDIUM_BORDER, width=CENTER_FRAME_WIDTH, height=CENTER_FRAME_HEIGHT,
            padx=0, pady=SMALL_PADDING, relief=RIDGE
        )
        self.frame_2.grid(row=0, column=1)

        self.frame_2_top = Frame(
            self.frame_2, width=TOP_SUBFRAME_WIDTH, height=TOP_SUBFRAME_HEIGHT,
            bd=MEDIUM_BORDER, padx=SMALL_PADDING * 4, pady=SMALL_PADDING, relief=RIDGE
        )
        self.frame_2_top.grid(row=0, column=0)

        self.frame_2_bottom = Frame(
            self.frame_2, width=BOTTOM_SUBFRAME_WIDTH, height=BOTTOM_SUBFRAME_HEIGHT,
            bd=MEDIUM_BORDER, padx=MEDIUM_PADDING, pady=SMALL_PADDING, relief=RIDGE
        )
        self.frame_2_bottom.grid(row=1, column=0)

        self.frame_2_bottom_l = Frame(
            self.frame_2_bottom, width=BOTTOM_LEFT_SUBFRAME_WIDTH, height=BOTTOM_SUBFRAME_HEIGHT,
            bd=MEDIUM_BORDER, padx=0, pady=SMALL_PADDING, relief=RIDGE
        )
        self.frame_2_bottom_l.grid(row=0, column=0)

        self.frame_2_bottom_r = Frame(
            self.frame_2_bottom, width=BOTTOM_RIGHT_SUBFRAME_WIDTH, height=BOTTOM_SUBFRAME_HEIGHT,
            bd=MEDIUM_BORDER, padx=SMALL_PADDING * 5, pady=SMALL_PADDING, relief=RIDGE
        )
        self.frame_2_bottom_r.grid(row=0, column=1)

        self.frame_3 = Frame(
            self.main_frame, bd=MEDIUM_BORDER, width=SIDE_FRAME_WIDTH, height=SIDE_FRAME_HEIGHT,
            padx=MEDIUM_PADDING, pady=SMALL_PADDING, relief=RIDGE
        )
        self.frame_3.grid(row=0, column=2)

    def _build_widgets(self):
        self.lbl_Title = Label(
            self.title_frame, text="Currency Converter | Foreign Exchange Rates",
            padx=TITLE_PADDING_X, pady=TITLE_PADDING_Y, bd=TITLE_BORDER,
            font=TITLE_FONT, bg=TITLE_BACKGROUND_COLOR, width=TITLE_LABEL_WIDTH
        )
        self.lbl_Title.pack()

        self.lblToday = Label(
            self.frame_2_top, font=LABEL_FONT, text='Todays Date',
            padx=LABEL_PADDING, pady=LABEL_PADDING_Y, bd=SMALL_BORDER, width=DATE_LABEL_WIDTH
        )
        self.lblToday.grid(row=0, column=0)
        self.lblDate = Label(
            self.frame_2_top, font=LABEL_FONT, textvariable=self.date_of_convert,
            padx=LABEL_PADDING, pady=LABEL_PADDING_Y, bd=SMALL_BORDER,
            width=DATE_VALUE_WIDTH, justify='center'
        )
        self.lblDate.grid(row=0, column=1)

        self.Converted = Scale(
            self.frame_3, variable=self.var1, from_=SCALE_FROM, to=SCALE_TO,
            length=SCALE_LENGTH, tickinterval=SCALE_TICK_INTERVAL, orient=VERTICAL,
            bg=SCALE_BACKGROUND_COLOR, label="Convert", font=SCALE_FONT
        )
        self.Converted.grid(row=0, column=0, rowspan=2)

        self.INR = Scale(
            self.frame_1, variable=self.var2, from_=SCALE_FROM, to=SCALE_TO,
            length=SCALE_LENGTH, tickinterval=SCALE_TICK_INTERVAL, orient=VERTICAL,
            bg=SCALE_BACKGROUND_COLOR, label="INR", font=SCALE_FONT
        )
        self.INR.grid(row=0, column=0, rowspan=2)

        self.lbl_INR = Label(
            self.frame_2_top, font=LABEL_FONT, text="INR",
            padx=LABEL_PADDING, pady=LABEL_PADDING_Y, width=INR_LABEL_WIDTH, bd=SMALL_BORDER
        )
        self.lbl_INR.grid(row=1, column=0)

        self.EntCurrency = Entry(
            self.frame_2_top, font=LABEL_FONT, textvariable=self.convert,
            bd=SMALL_BORDER, width=CURRENCY_ENTRY_WIDTH, justify='center'
        )
        self.EntCurrency.grid(row=1, column=1, pady=LABEL_PADDING_Y)

        self.Option = ttk.Combobox(
            self.frame_2_top, textvariable=self.value_0, state='readonly',
            font=LABEL_FONT, width=COMBOBOX_WIDTH
        )
        self.Option['values'] = (
            "Select", "Argentine Peso", "Australian Dollar", "Bahraini Dinar",
            "Batwana Pula", "Brazilian Real", "British Pound", "Bruneian Dollar",
            "Bulgarian Lev", "Canadian Dollar", "Chilean Peso",
            "Chinese Yuan Renmimbi", "Colombian Peso", "Crotian Kuna",
            "Czech Koruna", "Danish Krone", "Emirati Dirhan", "Euro",
            "Hong Kong Dollar", "Hungarian Foint", "IceLandic Krona",
            "Indonesian Rupiah", "Iranian Rial", "Israeli Shekel",
            "Japanese Yen", "Kazakhstani Tenge", "Kuwaiti Dinar",
            "Libyan Dinar", "Malaysian Ringgit", "Mauritian Rupee",
            "Mexican Peso", "Nepalese Rupee", "New Zealand Dollar",
            "Norwegian Krone", "Omani Rial", "Pakistani Rupee",
            "Philippine Peso", "Polish Zloty", "Qatari Riyal",
            "Romanian New Leu", "Russian Ruble", "Saudi Arabian Riyal",
            "Singapore Dollar", "South African Rand", "South Korean Won",
            "Sri Lankan Rupee", "Swedish Krona", "Swiss Franc",
            "Taiwan New Dollar", "Thai Baht", "Trinidadian Dollar",
            "Turkish Lira", "US Dollar", "Venezuelan Bolivar", ""
        )
        self.Option.current(0)
        self.Option.grid(row=2, column=0, padx=38, pady=LABEL_PADDING_Y)

        self.lblCurrency = Label(
            self.frame_2_top, font=LABEL_FONT, textvariable=self.currency,
            bd=SMALL_BORDER, width=CURRENCY_LABEL_WIDTH, bg=LABEL_BACKGROUND_COLOR,
            pady=SMALL_PADDING, padx=SMALL_PADDING, relief='sunken'
        )
        self.lblCurrency.grid(row=2, column=1)

        self.canvas = Canvas(
            self.frame_2_bottom_l, width=600, height=300, bg=CANVAS_BACKGROUND_COLOR
        )

        self.btnconvert = Button(
            self.frame_2_bottom_r, text="Convert", padx=SMALL_PADDING,
            pady=LABEL_PADDING_Y, bd=SMALL_BORDER, fg="black", font=LABEL_FONT,
            width=BUTTON_WIDTH, height=BUTTON_HEIGHT, command=self.convert_value
        )
        self.btnconvert.grid(row=4, column=0)

        self.btnReset = Button(
            self.frame_2_bottom_r, text="Reset", padx=SMALL_PADDING,
            pady=LABEL_PADDING, fg="black", font=LABEL_FONT,
            width=BUTTON_WIDTH, height=BUTTON_HEIGHT, command=self.reset
        )
        self.btnReset.grid(row=5, column=0)

        self.btnExit = Button(
            self.frame_2_bottom_r, text="Exit", padx=SMALL_PADDING,
            pady=LABEL_PADDING_Y, fg="black", font=LABEL_FONT,
            width=BUTTON_WIDTH, height=BUTTON_HEIGHT, command=self.exit_app
        )
        self.btnExit.grid(row=6, column=0)

    def exit_app(self):
        cExit = tkinter.messagebox.askyesno("Exit System", "Confirm if you want to exit")
        if (cExit > 0):
            self.root.destroy()
            return

    def reset(self):
        self.value_0.set("")
        self.convert.set(1)
        self.currency.set("0.00")
        self.var1.set(0)
        self.var2.set(0)

    def convert_value(self):
        s = self.value_0.get()
        rate = self.rate_service.get_rate(s)
        self.update_chart()
        x = self.convert.get() * rate
        Final_value = str('%.2f' % (x))
        self.currency.set(Final_value)
        self.var1.set(Final_value)
        self.var2.set(self.convert.get())
        return Final_value

    def update_chart(self):
        s = self.value_0.get()
        c1 = int(self.convert.get())
        u = self.rate_service.get_rate(s)
        c2 = c1 * u
        print(c1, c2)

        scatter3 = FigureCanvasTkAgg(self.fig, self.frame_2_bottom_l)
        scatter3.get_tk_widget().grid(row=0, column=0)
        self.ax.clear()
        self.ax.scatter(c1 + c1, c2)
        self.ax.set_xlabel('INR')
        self.ax.set_ylabel('Converted Value')
        self.ax.set_title('Currency Conversion')


if __name__ == "__main__":
    root = Tk()
    application = Convert(root)
    root.mainloop()