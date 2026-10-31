from tkinter import Tk, StringVar, DoubleVar, Frame, Label, Entry, Button, Scale, Canvas, RIDGE, VERTICAL
from tkinter import ttk
import tkinter.messagebox
import time
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import style

from rate_service import RateService


class Convert:

    def __init__(self, root):
        self.root = root
        self._setup_window()
        self._init_variables()
        self._build_frames()
        self._build_widgets()

    def _setup_window(self):
        self.root.title("Currency Converter | Foreign Exchange Rates")
        self.root.geometry("1350x800+0+0")
        self.root.configure(background="#F2C496")

        self.rate_service = RateService()
        self.rate_service.load()

        style.use("ggplot")
        self.fig = plt.Figure(figsize=(5, 3), dpi=100)
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
        self.title_frame = Frame(self.root, bd=10, width=1350, height=100, padx=10, pady=10, bg="#F2C496", relief=RIDGE)
        self.title_frame.grid(row=0, column=0)

        self.main_frame = Frame(self.root, bd=10, width=1350, height=700, padx=11, pady=10, bg="#F2C496", relief=RIDGE)
        self.main_frame.grid(row=1, column=0)

        self.frame_1 = Frame(self.main_frame, bd=4, width=100, height=600, padx=5, pady=1, relief=RIDGE)
        self.frame_1.grid(row=0, column=0)

        self.frame_2 = Frame(self.main_frame, bd=4, width=800, height=600, padx=0, pady=2, relief=RIDGE)
        self.frame_2.grid(row=0, column=1)

        self.frame_2_top = Frame(self.frame_2, width=250, height=300, bd=4, padx=80, pady=2, relief=RIDGE)
        self.frame_2_top.grid(row=0, column=0)

        self.frame_2_bottom = Frame(self.frame_2, width=800, height=300, bd=4, padx=5, pady=2, relief=RIDGE)
        self.frame_2_bottom.grid(row=1, column=0)

        self.frame_2_bottom_l = Frame(self.frame_2_bottom, width=450, height=300, bd=4, padx=0, pady=2, relief=RIDGE)
        self.frame_2_bottom_l.grid(row=0, column=0)

        self.frame_2_bottom_r = Frame(self.frame_2_bottom, width=350, height=300, bd=4, padx=10, pady=2, relief=RIDGE)
        self.frame_2_bottom_r.grid(row=0, column=1)

        self.frame_3 = Frame(self.main_frame, bd=4, width=100, height=600, padx=5, pady=1, relief=RIDGE)
        self.frame_3.grid(row=0, column=2)

    def _build_widgets(self):
        self.lbl_Title = Label(self.title_frame, text="Currency Converter | Foreign Exchange Rates", padx=17, pady=4, bd=1, font=('arial', 30, 'bold'), bg="#02A9FC", width=50)
        self.lbl_Title.pack()

        self.lblToday = Label(self.frame_2_top, font=('arial', 20, 'bold'), text='Todays Date', padx=2, pady=10, bd=2, width=18)
        self.lblToday.grid(row=0, column=0)
        self.lblDate = Label(self.frame_2_top, font=('arial', 20, 'bold'), textvariable=self.date_of_convert, padx=2, pady=10, bd=2, width=12, justify='center')
        self.lblDate.grid(row=0, column=1)

        self.Converted = Scale(self.frame_3, variable=self.var1, from_=0, to=1000, length=500, tickinterval=50, orient=VERTICAL, bg='#EA2A97', label="Convert", font=('arial', 10, 'bold'))
        self.Converted.grid(row=0, column=0, rowspan=2)

        self.INR = Scale(self.frame_1, variable=self.var2, from_=0, to=1000, length=500, tickinterval=50, orient=VERTICAL, bg='#EA2A97', label="INR", font=('arial', 10, 'bold'))
        self.INR.grid(row=0, column=0, rowspan=2)

        self.lbl_INR = Label(self.frame_2_top, font=('arial', 20, 'bold'), text="INR", padx=2, pady=10, width=19, bd=2)
        self.lbl_INR.grid(row=1, column=0)

        self.EntCurrency = Entry(self.frame_2_top, font=('arial', 20, 'bold'), textvariable=self.convert, bd=2, width=23, justify='center')
        self.EntCurrency.grid(row=1, column=1, pady=10)

        self.Option = ttk.Combobox(self.frame_2_top, textvariable=self.value_0, state='readonly', font=('arial', 20, 'bold'), width=20)
        self.Option['values'] = ("Select", "Argentine Peso", "Australian Dollar", "Bahraini Dinar", "Batwana Pula", "Brazilian Real", "British Pound", "Bruneian Dollar", "Bulgarian Lev", "Canadian Dollar", "Chilean Peso", "Chinese Yuan Renmimbi", "Colombian Peso", "Crotian Kuna", "Czech Koruna", "Danish Krone", "Emirati Dirhan", "Euro", "Hong Kong Dollar", "Hungarian Foint", "IceLandic Krona", "Indonesian Rupiah", "Iranian Rial", "Israeli Shekel", "Japanese Yen", "Kazakhstani Tenge", "Kuwaiti Dinar", "Libyan Dinar", "Malaysian Ringgit", "Mauritian Rupee", "Mexican Peso", "Nepalese Rupee", "New Zealand Dollar", "Norwegian Krone", "Omani Rial", "Pakistani Rupee", "Philippine Peso", "Polish Zloty", "Qatari Riyal", "Romanian New Leu", "Russian Ruble", "Saudi Arabian Riyal", "Singapore Dollar", "South African Rand", "South Korean Won", "Sri Lankan Rupee", "Swedish Krona", "Swiss Franc", "Taiwan New Dollar", "Thai Baht", "Trinidadian Dollar", "Turkish Lira", "US Dollar", "Venezuelan Bolivar", "")
        self.Option.current(0)
        self.Option.grid(row=2, column=0, padx=38, pady=10)

        self.lblCurrency = Label(self.frame_2_top, font=('arial', 20, 'bold'), textvariable=self.currency, bd=2, width=20, bg='white', pady=2, padx=2, relief='sunken')
        self.lblCurrency.grid(row=2, column=1)

        self.canvas = Canvas(self.frame_2_bottom_l, width=600, height=300, bg='gray')

        self.btnconvert = Button(self.frame_2_bottom_r, text="Convert", padx=2, pady=8, bd=2, fg="black", font=('arial', 20, 'bold'), width=14, height=2, command=self.convert_value)
        self.btnconvert.grid(row=4, column=0)

        self.btnReset = Button(self.frame_2_bottom_r, text="Reset", padx=2, pady=7, fg="black", font=('arial', 20, 'bold'), width=14, height=2, command=self.reset)
        self.btnReset.grid(row=5, column=0)

        self.btnExit = Button(self.frame_2_bottom_r, text="Exit", padx=2, pady=8, fg="black", font=('arial', 20, 'bold'), width=14, height=2, command=self.exit_app)
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