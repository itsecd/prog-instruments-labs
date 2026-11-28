# Importing the required Modules

from tkinter import Tk, StringVar, ttk
from tkinter import *
import tkinter.messagebox          # For Message Box
import time                        # For Date and Time
import requests as req             # For requests 
from bs4 import *                  # For BeautifulSoup
from matplotlib import pyplot as plt                  
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from matplotlib import style

from config_loader import load_config


# Загружаем конфиг (окружение можно задать через переменную APP_ENV)
cfg = load_config()

# Сетевые настройки из конфига
url = cfg.network.url
header = {
    "User-Agent": cfg.network.user_agent
}

# Getting the access to webpage and Creating the Soup object
webpage = req.get(url, headers=header, timeout=cfg.network.timeout)
Soup = BeautifulSoup(webpage.text, 'html.parser')


style.use(cfg.ui.chart.style)

fig3 = plt.Figure(
    figsize=tuple(cfg.ui.chart.figure_size),
    dpi=cfg.ui.chart.dpi
)
ax3 = fig3.add_subplot(111)


# Class For Converter
class Convert:

    def __init__(self, root):
        self.root = root
        self.root.title(cfg.ui.window.title)
        self.root.geometry(cfg.ui.window.geometry)
        self.root.configure(background=cfg.ui.window.background_color)


        #=============================== VAriables ===============================

        
        DateofConvert = StringVar()
        value_0 = StringVar()
        convert = DoubleVar()
        currency = DoubleVar()
        var1 = DoubleVar()
        var2 = DoubleVar()
        convert.set(cfg.converter.default_amount)
        DateofConvert.set(time.strftime(cfg.converter.date_format))

        # ============================ Methods ==================================

            # ============= Method For Exit =================================
        def cExit():
            cExit = tkinter.messagebox.askyesno("Exit System", "Confirm if you want to exit")
            if (cExit > 0):
                root.destroy()
                return

            # ============= Method For Reset the Screen ====================
        def Reset():
            value_0.set("")
            convert.set(cfg.converter.default_amount)
            currency.set("0.00")
            var1.set(0)
            var2.set(0)
            
            # ============= Method For Converting the value ================
        def Converted_value():
            s = value_0.get()
            # ============== Fetch the value of rates from the table ======
            rate = Soup.find('table', class_ = 'tablesorter ratesTable').find('td', text = lambda x : x == s).find_next('td').get_text()
            canvasCal()
            x = float(convert.get() * float(rate))
            Final_value = str('%.2f' %(x))
            currency.set(Final_value)
            var1.set(Final_value)
            var2.set(convert.get())
            def animate():
                s = value_0.get()
                c1 = int(convert.get())
                u = Soup.find('table', class_ = 'tablesorter ratesTable').find('td', text = lambda x : x == s).find_next('td').get_text()
                c2 = (float(c1) * float(u)) 
                ax3.clear()
                ax3.scatter(c1 + c1, c2)
            animate()

            if cfg.logging.print_debug:
                print(f"[{cfg.environment}] Converted {convert.get()} {cfg.converter.base_currency} to {Final_value} {s}")

            return Final_value

        def canvasCal():
            s = value_0.get()
            c1 = int(convert.get())
            u = Soup.find('table', class_ = 'tablesorter ratesTable').find('td', text = lambda x : x == s).find_next('td').get_text()
            c2 = (float(c1) * float(u))

            if cfg.logging.print_debug:
                print(c1, c2)
            
            #ax3.scatter(c1, c2)
            scatter3 = FigureCanvasTkAgg(fig3, Frame_2ButtomL)
            scatter3.get_tk_widget().grid(row = 0, column = 0)
            #ax3.legend()
            ax3.set_xlabel(cfg.ui.chart.x_label)
            ax3.set_ylabel(cfg.ui.chart.y_label)
            ax3.set_title(cfg.ui.chart.title)


        # ===================== Frames ==========================================
        Title_frame = Frame(
            self.root,
            bd=cfg.ui.frames.title.border,
            width=cfg.ui.frames.title.width,
            height=cfg.ui.frames.title.height,
            padx=cfg.ui.frames.title.padding_x,
            pady=cfg.ui.frames.title.padding_y,
            bg=cfg.ui.window.background_color,
            relief=RIDGE
        )
        Title_frame.grid(row=0, column=0)

        self.lbl_Title = Label(
            Title_frame,
            text=cfg.ui.title_label.text,
            padx=cfg.ui.title_label.padding_x,
            pady=cfg.ui.title_label.padding_y,
            bd=cfg.ui.title_label.border,
            font=tuple(cfg.ui.title_label.font),
            bg=cfg.ui.title_label.background_color,
            width=cfg.ui.title_label.width
        )
        self.lbl_Title.pack()

        Main_frame = Frame(
            self.root,
            bd=cfg.ui.frames.main.border,
            width=cfg.ui.frames.main.width,
            height=cfg.ui.frames.main.height,
            padx=cfg.ui.frames.main.padding_x,
            pady=cfg.ui.frames.main.padding_y,
            bg=cfg.ui.window.background_color,
            relief=RIDGE
        )
        Main_frame.grid(row=1, column=0)

        Frame_1 = Frame(
            Main_frame,
            bd=cfg.ui.frames.side.border,
            width=cfg.ui.frames.side.width,
            height=cfg.ui.frames.side.height,
            padx=cfg.ui.frames.side.padding_x,
            pady=cfg.ui.frames.side.padding_y,
            relief=RIDGE
        )
        Frame_1.grid(row=0, column=0)

        Frame_2 = Frame(
            Main_frame,
            bd=cfg.ui.frames.center.border,
            width=cfg.ui.frames.center.width,
            height=cfg.ui.frames.center.height,
            padx=cfg.ui.frames.center.padding_x,
            pady=cfg.ui.frames.center.padding_y,
            relief=RIDGE
        )
        Frame_2.grid(row=0, column=1)

        Frame_2Top = Frame(
            Frame_2,
            width=cfg.ui.frames.top.width,
            height=cfg.ui.frames.top.height,
            bd=cfg.ui.frames.top.border,
            padx=cfg.ui.frames.top.padding_x,
            pady=cfg.ui.frames.top.padding_y,
            relief=RIDGE
        )
        Frame_2Top.grid(row=0, column=0)

        Frame_2Buttom = Frame(
            Frame_2,
            width=cfg.ui.frames.bottom.width,
            height=cfg.ui.frames.bottom.height,
            bd=cfg.ui.frames.bottom.border,
            padx=cfg.ui.frames.bottom.padding_x,
            pady=cfg.ui.frames.bottom.padding_y,
            relief=RIDGE
        )
        Frame_2Buttom.grid(row=1, column=0)

        Frame_2ButtomL = Frame(
            Frame_2Buttom,
            width=cfg.ui.frames.bottom_left.width,
            height=cfg.ui.frames.bottom_left.height,
            bd=cfg.ui.frames.bottom_left.border,
            padx=cfg.ui.frames.bottom_left.padding_x,
            pady=cfg.ui.frames.bottom_left.padding_y,
            relief=RIDGE
        )
        Frame_2ButtomL.grid(row=0, column=0)

        Frame_2ButtomR = Frame(
            Frame_2Buttom,
            width=cfg.ui.frames.bottom_right.width,
            height=cfg.ui.frames.bottom_right.height,
            bd=cfg.ui.frames.bottom_right.border,
            padx=cfg.ui.frames.bottom_right.padding_x,
            pady=cfg.ui.frames.bottom_right.padding_y,
            relief=RIDGE
        )
        Frame_2ButtomR.grid(row=0, column=1)

        Frame_3 = Frame(
            Main_frame,
            bd=cfg.ui.frames.side.border,
            width=cfg.ui.frames.side.width,
            height=cfg.ui.frames.side.height,
            padx=cfg.ui.frames.side.padding_x,
            pady=cfg.ui.frames.side.padding_y,
            relief=RIDGE
        )
        Frame_3.grid(row=0, column=2)

        #=============================== Widgets =================================
        self.lblToday = Label(
            Frame_2Top,
            font=tuple(cfg.ui.fonts.label),
            text=cfg.ui.widgets.date_label.text,
            padx=cfg.ui.widgets.date_label.padding_x,
            pady=cfg.ui.widgets.date_label.padding_y,
            bd=cfg.ui.widgets.date_label.border,
            width=cfg.ui.widgets.date_label.width
        )
        self.lblToday.grid(row=0, column=0)

        self.lblDate = Label(
            Frame_2Top,
            font=tuple(cfg.ui.fonts.label),
            textvariable=DateofConvert,
            padx=cfg.ui.widgets.date_value.padding_x,
            pady=cfg.ui.widgets.date_value.padding_y,
            bd=cfg.ui.widgets.date_value.border,
            width=cfg.ui.widgets.date_value.width,
            justify=cfg.ui.widgets.date_value.justify
        )
        self.lblDate.grid(row=0, column=1)

        #================================ Scale ==================================

        self.Converted = Scale(
            Frame_3,
            variable=var1,
            from_=cfg.ui.widgets.scale.from_,
            to=cfg.ui.widgets.scale.to,
            length=cfg.ui.widgets.scale.length,
            tickinterval=cfg.ui.widgets.scale.tick_interval,
            orient=VERTICAL,
            bg=cfg.ui.colors.scale,
            label="Convert",
            font=tuple(cfg.ui.fonts.scale)
        )
        self.Converted.grid(row=0, column=0, rowspan=2)

        self.INR = Scale(
            Frame_1,
            variable=var2,
            from_=cfg.ui.widgets.scale.from_,
            to=cfg.ui.widgets.scale.to,
            length=cfg.ui.widgets.scale.length,
            tickinterval=cfg.ui.widgets.scale.tick_interval,
            orient=VERTICAL,
            bg=cfg.ui.colors.scale,
            label=cfg.converter.base_currency,
            font=tuple(cfg.ui.fonts.scale)
        )
        self.INR.grid(row=0, column=0, rowspan=2)

        #=========================== Labels for Frame_2 Top=======================

        self.lbl_INR = Label(
            Frame_2Top,
            font=tuple(cfg.ui.fonts.label),
            text=cfg.ui.widgets.inr_label.text,
            padx=cfg.ui.widgets.inr_label.padding_x,
            pady=cfg.ui.widgets.inr_label.padding_y,
            width=cfg.ui.widgets.inr_label.width,
            bd=cfg.ui.widgets.inr_label.border
        )
        self.lbl_INR.grid(row=1, column=0)

        self.EntCurrency = Entry(
            Frame_2Top,
            font=tuple(cfg.ui.fonts.label),
            textvariable=convert,
            bd=cfg.ui.widgets.currency_entry.border,
            width=cfg.ui.widgets.currency_entry.width,
            justify=cfg.ui.widgets.currency_entry.justify
        )
        self.EntCurrency.grid(row=1, column=1, pady=10)

        self.Option = ttk.Combobox(
            Frame_2Top,
            textvariable=value_0,
            state='readonly',
            font=tuple(cfg.ui.fonts.label),
            width=cfg.ui.widgets.combobox.width
        )
        self.Option['values'] = tuple(cfg.currencies)
        self.Option.current(0)
        self.Option.grid(
            row=2, column=0,
            padx=cfg.ui.widgets.combobox.padding_x,
            pady=cfg.ui.widgets.combobox.padding_y
        )

        self.lblCurrency = Label(
            Frame_2Top,
            font=tuple(cfg.ui.fonts.label),
            textvariable=currency,
            bd=cfg.ui.widgets.result_label.border,
            width=cfg.ui.widgets.result_label.width,
            bg=cfg.ui.colors.result_label,
            pady=cfg.ui.widgets.result_label.padding_y,
            padx=cfg.ui.widgets.result_label.padding_x,
            relief=cfg.ui.widgets.result_label.relief
        )
        self.lblCurrency.grid(row=2, column=1)

        #============================== Canvas =================================

        self.canvas = Canvas(
            Frame_2ButtomL,
            width=cfg.ui.widgets.canvas.width,
            height=cfg.ui.widgets.canvas.height,
            bg=cfg.ui.colors.canvas
        )

        #============================= Buttons =================================

        self.btnconvert = Button(
            Frame_2ButtomR,
            text="Convert",
            padx=cfg.ui.widgets.button.padding_x,
            pady=cfg.ui.widgets.button.padding_y,
            bd=cfg.ui.widgets.button.border,
            fg=cfg.ui.colors.button_fg,
            font=tuple(cfg.ui.fonts.label),
            width=cfg.ui.widgets.button.width,
            height=cfg.ui.widgets.button.height,
            command=Converted_value
        )
        self.btnconvert.grid(row=4, column=0)

        self.btnReset = Button(
            Frame_2ButtomR,
            text="Reset",
            padx=cfg.ui.widgets.button.padding_x,
            pady=cfg.ui.widgets.button.padding_y - 1,
            fg=cfg.ui.colors.button_fg,
            font=tuple(cfg.ui.fonts.label),
            width=cfg.ui.widgets.button.width,
            height=cfg.ui.widgets.button.height,
            command=Reset
        )
        self.btnReset.grid(row=5, column=0)

        self.btnExit = Button(
            Frame_2ButtomR,
            text="Exit",
            padx=cfg.ui.widgets.button.padding_x,
            pady=cfg.ui.widgets.button.padding_y,
            fg=cfg.ui.colors.button_fg,
            font=tuple(cfg.ui.fonts.label),
            width=cfg.ui.widgets.button.width,
            height=cfg.ui.widgets.button.height,
            command=cExit
        )
        self.btnExit.grid(row=6, column=0)


# ============================ Driver Code For the program =========================
if __name__ == "__main__":
    print(f"Запуск приложения в окружении: {cfg.environment}")
    print(f"URL источника курсов: {cfg.network.url}")
    print(f"Уровень логирования: {cfg.logging.level}")

    root = Tk()                     # Object instance of Tk() Class
    application = Convert(root)     # Creating the Object of the Classs
    root.mainloop()                 # Running the main loop for the main window