# Стандартные библиотеки
from math import sqrt

# Сторонние библиотеки
from sympy import ln, log, sin, cos, pi, tan, cot
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import (
    QKeyEvent, 
    QMouseEvent, 
    QPalette, 
    QColor, 
    QGuiApplication
)
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QFrame
)
from loguru import logger

# Настройка логирования
logger.add("app.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="DEBUG", rotation="1 MB")

Mem1 = 0


class Color(QWidget):
    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)


class CustomQPushButton(QPushButton):
    def __init__(self, text, parent, if_eng:bool=False):
        super().__init__(text, parent=parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, \
                           QSizePolicy.Policy.MinimumExpanding)
        self.full_formula = self.parent().full_formula
        self.alone_sign = self.parent().alone_sign
        self.full_formula.setText("0")
        self.alone_sign.setText("0")
# указатель, что кнопка является частью инженерного режима
        self.is_eng = if_eng  
        logger.debug(f"Button '{text}' initialized. Engineer mode: {self.is_eng}")

        
    
    def eng_toggle(self, switcher=None):
        """ Переключение видимости кнопок """
        if self.is_eng and self.isVisible():
            self.hide()
        elif self.is_eng and not self.isVisible():
            self.show()


    def key_Press_Event(self, e_1: QKeyEvent) -> None:
        match e_1.key():
            case Qt.Key_1:
                if self.alone_sign.text() == "0":
                    self.alone_sign.clear()
                if not "=" in self.alone_sign.text():
                    self.alone_sign.setText(self.alone_sign.text() + "1")
            case Qt.Key_2:
                if self.alone_sign.text() == "0":
                    self.alone_sign.clear()
                if not "=" in self.alone_sign.text():
                    self.alone_sign.setText(self.alone_sign.text() + "2")
            case Qt.Key_3:
                if self.alone_sign.text() == "0":
                    self.alone_sign.clear()
                if not "=" in self.alone_sign.text():
                    self.alone_sign.setText(self.alone_sign.text() + "3")
            case Qt.Key_4:
                if self.alone_sign.text() == "0":
                    self.alone_sign.clear()
                if not "=" in self.alone_sign.text():
                    self.alone_sign.setText(self.alone_sign.text() + "4")
            case Qt.Key_5:
                if self.alone_sign.text() == "0":
                    self.alone_sign.clear()
                if not "=" in self.alone_sign.text():
                    self.alone_sign.setText(self.alone_sign.text() + "5")
            case Qt.Key_6:
                if self.alone_sign.text() == "0":
                    self.alone_sign.clear()
                if not "=" in self.alone_sign.text():
                    self.alone_sign.setText(self.alone_sign.text() + "6")
            case Qt.Key_7:
                if self.alone_sign.text() == "0":
                    self.alone_sign.clear()
                if not "=" in self.alone_sign.text():
                    self.alone_sign.setText(self.alone_sign.text() + "7")
            case Qt.Key_8:
                if self.alone_sign.text() == "0":
                    self.alone_sign.clear()
                if not "=" in self.alone_sign.text():
                    self.alone_sign.setText(self.alone_sign.text() + "8")
            case Qt.Key_9:
                if self.alone_sign.text() == "0":
                    self.alone_sign.clear()
                if not "=" in self.alone_sign.text():
                    self.alone_sign.setText(self.alone_sign.text() + "9")
            case Qt.Key_0:
                if self.alone_sign.text() == "0":
                    self.alone_sign.clear()
                if not "=" in self.alone_sign.text():
                    self.alone_sign.setText(self.alone_sign.text() + "0")
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            case Qt.Key_Plus:
                if self.alone_sign.text()[-1].isdigit():
                    if self.alone_sign.text()[-1] == ".":
                        self.alone_sign.setText(self.alone_sign.text() + "0")
                    if "=" in self.alone_sign.text():
                        self.alone_sign.clear()
                    if self.full_formula.text() == "0":
                        self.full_formula.clear()
                    if len(self.full_formula.text()) < 1:
                        self.full_formula.setText(self.full_formula.text()\
                                                +self.alone_sign.text() + "+")
                        self.alone_sign.setText("0")
                    elif  self.full_formula.text()[-1] == "0":
                        self.full_formula.setText(self.full_formula.text()\
                                                + "+")
                        self.alone_sign.setText("0")
                    else:
                        self.full_formula.setText(self.full_formula.text()\
                                                +self.alone_sign.text() + "+")
                        self.alone_sign.setText("0")
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            case Qt.Key_Minus:
                if self.alone_sign.text()[-1].isdigit():
                    if self.alone_sign.text()[-1] == ".":
                        self.alone_sign.setText(self.alone_sign.text() + "0")
                    if "=" in self.alone_sign.text():
                        self.alone_sign.clear()
                    if self.full_formula.text() == "0":
                        self.full_formula.clear()

                    if len(self.full_formula.text()) < 1:
                        self.full_formula.setText(self.full_formula.text()\
                                                +self.alone_sign.text() + "-")
                        self.alone_sign.setText("0")
                    elif  self.full_formula.text()[-1] == "0":
                        self.full_formula.setText(self.full_formula.text()\
                                                +"-")
                        self.alone_sign.setText("0")
                    else:
                        self.full_formula.setText(self.full_formula.text()\
                                                +self.alone_sign.text() + "-")
                        self.alone_sign.setText("0")
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            case Qt.Key_Asterisk:
                if self.alone_sign.text()[-1].isdigit():
                    if self.alone_sign.text()[-1] == ".":
                        self.alone_sign.setText(self.alone_sign.text() + "0")
                    if "=" in self.alone_sign.text():
                        self.alone_sign.clear()
                    if self.full_formula.text() == "0":
                        self.full_formula.clear()

                    if len(self.full_formula.text()) < 1:
                        self.full_formula.setText(self.full_formula.text()\
                                                +self.alone_sign.text() + "*")
                        self.alone_sign.setText("0")
                    elif  self.full_formula.text()[-1] == "0":
                        self.full_formula.setText(self.full_formula.text()\
                                                +"*")
                        self.alone_sign.setText("0")
                    else:
                        self.full_formula.setText(self.full_formula.text()\
                                                +self.alone_sign.text() + "*")
                        self.alone_sign.setText("0")
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            case Qt.Key_Slash:
                if self.alone_sign.text()[-1].isdigit():
                    if self.alone_sign.text()[-1] == ".":
                        self.alone_sign.setText(self.alone_sign.text() + "0")
                    if "=" in self.alone_sign.text():
                        self.alone_sign.clear()
                    if self.full_formula.text() == "0":
                        self.full_formula.clear()
                    if len(self.full_formula.text()) < 1:
                        self.full_formula.setText(self.full_formula.text()\
                                                +self.alone_sign.text() + "/")
                        self.alone_sign.setText("0")
                    elif  self.full_formula.text()[-1] == "0":
                        self.full_formula.setText(self.full_formula.text()\
                                                +"/")
                        self.alone_sign.setText("0")
                    else:
                        self.full_formula.setText(self.full_formula.text()\
                                                +self.alone_sign.text() + "/")
                        self.alone_sign.setText("0")
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            case Qt.Key_Equal: 
                if self.alone_sign.text()[-1] == ".":
                    self.alone_sign.setText(self.alone_sign.text() + "0")
                if self.full_formula.text() == "0":
                    self.full_formula.clear()
                if "=" in self.alone_sign.text() and \
                    ("+" in self.full_formula.text() \
                    or "-" in self.full_formula.text() \
                    or "*" in self.full_formula.text() or \
                    "/" in self.full_formula.text()):
                    time_alone = ""
                    fin = 0
                    i = len(self.full_formula.text()) - 1
                    while fin == 0:
                        if self.full_formula.text()[i] == "+" or \
                            self.full_formula.text()[i] == "-" or \
                            self.full_formula.text()[i] == "*" or \
                            self.full_formula.text()[i] == "/":
                            if self.full_formula.text()[i] == "*":
                                time_alone = self.full_formula.text()[i]+\
                                    "*" + time_alone
                            else: 
                                time_alone = self.full_formula.text()[i]+\
                                    time_alone
                            fin = 1
                        else:
                            time_alone = self.full_formula.text()[i]\
                            + time_alone
                            i -= 1
                    time_full = str(self.alone_sign.text())
                    time_full = time_full[1:]
                    time_full = time_full + time_alone
                    self.full_formula.setText(time_full)
                    self.alone_sign.setText("=" + \
                                        str(eval(self.full_formula.text())))
                elif "=" in self.alone_sign.text():
                    pass
                elif len(self.alone_sign.text()) > 0:
                    if not self.alone_sign.text() == "0":
                        self.full_formula.setText(self.full_formula.text() + \
                                                  self.alone_sign.text())
                        if str(eval(self.full_formula.text()))[-1] == "0" and \
                                str(eval(self.full_formula.text()))[-2] == ".":
                            self.alone_sign.setText(\
                                "=" + str(int(eval(self.full_formula.text()))))
                        else:
                            self.alone_sign.setText(\
                                "=" + str(eval(self.full_formula.text())))
                else:
                    if str(self.full_formula.text()[-1]).isdigit() or \
                        str(self.alone_sign.text()[-1]).isdigit():
                        self.full_formula.setText(self.full_formula.text() + \
                                                  self.alone_sign.text())
                        self.alone_sign.setText(\
                            "=" + str(eval(self.full_formula.text())))
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            case Qt.Key_Period:
                if len(self.alone_sign.text()) == 0:	
                    self.alone_sign.setText("0")
                if not "." in self.alone_sign.text():
                    if not self.alone_sign.text()[-1].isdigit():
                        self.alone_sign.setText(self.alone_sign.text()\
                                                 + "0" + ".")
                    else:
                        self.alone_sign.setText(self.alone_sign.text()\
                                                 + ".")
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            case Qt.Key_Backspace:
                if len(self.alone_sign.text()) > 1: 
                    it= self.alone_sign.text()[:len(self.alone_sign.text())-1]
                    self.alone_sign.setText(it)
                else:
                    self.alone_sign.setText("0")
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            case Qt.Key_Delete:
                if len(self.full_formula.text()) > 1: 
                    it= self.full_formula.text()[:len(self.full_formula.text())-1]
                    self.full_formula.setText(it)
                else:
                    self.full_formula.setText("0")        
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            case Qt.Key_Slash:
                if self.alone_sign.text()[-1].isdigit():
                    if self.alone_sign.text()[-1] == ".":
                        self.alone_sign.setText(self.alone_sign.text() + "0")
                    if "=" in self.alone_sign.text():
                        self.alone_sign.clear()
                    if self.full_formula.text() == "0":
                        self.full_formula.clear()
                    if not self.alone_sign.text()[-1] == "0":
                        if self.full_formula.text()[-1] == "0":
                            self.full_formula.setText(\
                                self.full_formula.text() + "/")
                            self.alone_sign.setText("0")
                        else:
                            self.full_formula.setText(\
                                self.full_formula.text() +\
                                self.alone_sign.text() + "/")
                            self.alone_sign.setText("0")
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        return super().keyPressEvent(e_1)
    

    def mousePressEvent(self, e: QMouseEvent) -> None:
        global Mem1
        logger.info(f"Button '{self.text()}' pressed.")
        if self.text() == 'C' or self.text() == 'CE':
                pass
        else:
            if len(str(self.full_formula.text())) > 65 or \
                len(str(self.alone_sign.text())) > 65:
                return 0    
        if self.text() == 'C':
                logger.debug("Clearing display.")
                self.alone_sign.setText("0")
        elif self.text() == 'CE':
                logger.debug("Clearing full formula and display.")
                self.alone_sign.setText("0")  
                self.full_formula.setText("0")
# ///////////////////////////////////////////////////////////////////////////////
        elif self.text() == "MRC":
             
            if not self.full_formula.text()[-1].isdigit():
                self.alone_sign.setText(str(Mem1))
                Mem1 = 0
        elif self.text() == "MC": 
            Mem1 = 0
        elif self.text() == "MR": 
            if not self.full_formula.text()[-1].isdigit() or \
                self.full_formula.text() == '0':
                self.alone_sign.setText(str(Mem1))
        elif self.text() == "MS": 
            if "=" in self.alone_sign.text():
                Mem1  = float(self.alone_sign.text()[1:])
            else:
                Mem1  = float(self.alone_sign.text())
        elif self.text() == "M+": 
            if "=" in self.alone_sign.text():
                self.Mem +=  float(self.alone_sign.text()[1:])
            else:
                Mem1 +=  float(self.alone_sign.text())   
        elif self.text() == "M-": 
            if "=" in self.alone_sign.text():
                Mem1 -=  float(self.alone_sign.text()[1:])
            else:
                Mem1 -=  float(self.alone_sign.text())
# //////////////////////////////////////////////////////////////////////////////
        elif self.text() == "sqrt" and not "log" in \
            self.alone_sign.text() and not "%" in self.alone_sign.text():
            if (not ("=" in self.alone_sign.text())):
                self.alone_sign.setText(str(sqrt(float(self.alone_sign.text()))))
        elif self.text() == "log" or self.text() == "ln":
            if not ("=" in self.alone_sign.text()) and \
                not ("log" in self.alone_sign.text()) and \
                not ("%" in self.alone_sign.text()):
                if self.text() == "log":
                    self.alone_sign.setText(self.alone_sign.text() + "log")
                if self.text() == "ln":
                    self.alone_sign.setText(str(ln(float(self.alone_sign.text()))))
        elif "log" in self.alone_sign.text():
            if str(self.text()).isdigit():
                self.alone_sign.setText(self.alone_sign.text() + self.text())
            else:
                if  any([self.text() == '=',
                      self.text() == '+',
                      self.text() == '-',
                      self.text() == '*',
                      self.text() == '/',
                      self.text() == "x^y",
                      self.text() == "sqrt"]):
                    j = 0
                    final = 0
                    right=""
                    down=""
                    while final == 0:
                        if self.alone_sign.text()[j] == "l":
                            final = 1
                        else:
                            right += self.alone_sign.text()[j]
                            j += 1
                    j = len(self.alone_sign.text()) - 1
                    final = 0
                    while final == 0:
                        if self.alone_sign.text()[j] == "g":
                                final = 1
                        else:
                            down = self.alone_sign.text()[j] + down
                            j -= 1
                    if right == "0" or down == "0":
                        if self.full_formula.text() == "0":
                            self.full_formula.clear()
                        self.full_formula.setText(self.full_formula.text()\
                                                   + "0")
                        self.alone_sign.setText(\
                            "=" + str(eval(self.full_formula.text())))
                    elif self.text() == '=':
                        if self.full_formula.text() == "0":
                            self.full_formula.clear()
                        self.full_formula.setText(\
                            self.full_formula.text() + \
                            str(log(float(down),float(right))))
                        self.alone_sign.setText(\
                            "=" + str(eval(self.full_formula.text())))
                    elif self.text() == 'sqrt':
                        if self.full_formula.text() == "0":
                            self.full_formula.clear()
                        self.full_formula.setText(\
                            self.full_formula.text() + \
                                str(sqrt(log(float(down),float(right)))))
                        self.alone_sign.setText(\
                            "=" + str(eval(self.full_formula.text())))
                    elif self.text() == 'x^y':
                        if self.full_formula.text() == "0":
                            self.full_formula.clear()
                        self.full_formula.setText(\
                            self.full_formula.text() + \
                            str(log(float(down),float(right))) + "**")
                        self.alone_sign.setText("0")
                    else: 
                        if self.full_formula.text() == "0":
                            self.full_formula.clear()
                        self.full_formula.setText(\
                            self.full_formula.text() + \
                            str(log(float(down),float(right))) + \
                            self.text())
                        self.alone_sign.setText("0")
        elif self.text() == "%":
            if self.alone_sign.text().isdigit():
                self.alone_sign.setText(self.alone_sign.text() + "%")
        elif "%" in self.alone_sign.text():
            if str(self.text()).isdigit():
                self.alone_sign.setText(self.alone_sign.text() + self.text())
            else:
                if self.text() == '=' or self.text() == '+' or \
                    self.text() == '-' or self.text() == '*' or \
                    self.text() == '/' or self.text() == "x^y" or \
                    self.text() == "sqrt":
                    j1 = 0
                    final1 = 0
                    right1=""
                    down1=""
                    while final1 == 0:
                        if self.alone_sign.text()[j1] == "%":
                            final1 = 1
                        else:
                            right1 += self.alone_sign.text()[j1]
                            j1 += 1
                    j1 = len(self.alone_sign.text()) - 1
                    final1 = 0
                    while final1 == 0:
                        if self.alone_sign.text()[j1] == "%":
                                final1 = 1
                        else:
                            down1 = self.alone_sign.text()[j1] + down1
                            j1 -= 1
                    if right1 == "0" or down1 == "0":
                        if self.full_formula.text() == "0":
                            self.full_formula.clear()
                        self.full_formula.setText(\
                            self.full_formula.text() + "0")
                        self.alone_sign.setText(\
                            "=" + str(eval(self.full_formula.text())))
                    elif self.text() == '=':
                        if self.full_formula.text() == "0":
                            self.full_formula.clear()
                        self.full_formula.setText(\
                            self.full_formula.text() + \
                            str(float(down1)/100*float(right1)))
                        self.alone_sign.setText(\
                            "=" + str(eval(self.full_formula.text())))
                    elif self.text() == 'sqrt':
                        if self.full_formula.text() == "0":
                            self.full_formula.clear()
                        self.full_formula.setText(\
                            self.full_formula.text() + \
                            str(sqrt(float(down1)/100*float(right1))))
                        self.alone_sign.setText(\
                            "=" + str(eval(self.full_formula.text())))
                    elif self.text() == 'x^y':
                        if self.full_formula.text() == "0":
                            self.full_formula.clear()
                        self.full_formula.setText(\
                            self.full_formula.text() + \
                            str(float(down1)/100*float(right1)) + "**")
                        self.alone_sign.setText("0")
                    else: 
                        if self.full_formula.text() == "0":
                            self.full_formula.clear()
                        self.full_formula.setText(\
                            self.full_formula.text() + \
                            str(float(down1)/100*float(right1)) + self.text())
                        self.alone_sign.setText("0")

        elif self.text() == "sin" or self.text() == "cos" or \
            self.text() == "tan" or self.text() == "cot":
            if not ("=" in self.alone_sign.text()):
                x = float(self.alone_sign.text())  
                self.alone_sign.clear()
                if self.text() == "sin":
                    if x==0:
                        self.alone_sign.setText(\
                        self.alone_sign.text() + "1")
                    else:
                        self.alone_sign.setText(\
                        self.alone_sign.text() + str((float(sin(pi/(180/x))))))
                if self.text() == "cos":
                    if x==0:
                        self.alone_sign.setText(self.alone_sign.text() + "0")
                    else:
                        self.alone_sign.setText(\
                        self.alone_sign.text() + str((float(cos(pi/(180/x))))))
                if self.text() == "tan":
                    if not (x%90 == 0 and x%180>0):
                        if x==0:
                            self.alone_sign.setText(\
                            self.alone_sign.text() + "0")
                        else:
                            self.alone_sign.setText(\
                            self.alone_sign.text() + \
                            str((float(tan(pi/(180/x))))))
                if self.text() == "cot":
                    if not (x%180 == 0):
                        if not (x == 0) :
                            self.alone_sign.setText(\
                                self.alone_sign.text() + \
                                str((float(cot(pi/(180/x))))))
                if self.alone_sign.text() == "":
                    self.alone_sign.setText("0")
        elif self.text() == ".":
            if "." in self.alone_sign.text():
                pass
            else:
                self.alone_sign.setText(self.alone_sign.text() + self.text())
        elif self.text() == "+/-":
            if "=" in self.alone_sign.text() or "^" in self.alone_sign.text():
                pass
            else:
                if self.alone_sign.text()[0] == "-":
                    rem = float(self.alone_sign.text()) * -1
                    self.alone_sign.setText(str(rem))
                else:
                    self.alone_sign.setText("-" + self.alone_sign.text())

        elif str(self.text()).isdigit():
                if "=" in self.alone_sign.text() or \
                    (str(self.full_formula.text()[-1]).isdigit() and \
                    len(self.full_formula.text())>1):
                    pass
                else:
                    if self.alone_sign.text() == "0" or \
                        self.alone_sign.text() == "-0":
                        self.alone_sign.clear()
                    self.alone_sign.setText(self.alone_sign.text() + \
                                            self.text())
        else:
            if not self.text() == 'Инжинерный' and \
                not self.text() == 'Обычный': 
                
                if self.text() == '=':
                    if self.alone_sign.text()[-1] == ".":
                        self.alone_sign.setText(self.alone_sign.text() + "0")
                    if self.full_formula.text() == "0":
                        self.full_formula.clear()
                    if "=" in self.alone_sign.text() and \
                        ("+" in self.full_formula.text() or \
                         "-" in self.full_formula.text() or \
                        "*" in self.full_formula.text() or \
                        "/" in self.full_formula.text()):
                        time_alone = ""
                        fin = 0
                        i = len(self.full_formula.text()) - 1
                        while fin == 0:
                            if self.full_formula.text()[i] == "+" or \
                                self.full_formula.text()[i] == "-" or \
                                self.full_formula.text()[i] == "*" or \
                                self.full_formula.text()[i] == "/":
                                if self.full_formula.text()[i] == "*" and \
                                    self.full_formula.text()[i-1] == "*":
                                    time_alone = self.full_formula.text()[i]\
                                        + "*" + time_alone 
                                else: 
                                    time_alone = self.full_formula.text()[i]\
                                        + time_alone
                                fin = 1
                            else:
                                time_alone = self.full_formula.text()[i]\
                                    + time_alone
                                i -= 1
                        time_full = str(self.alone_sign.text())
                        time_full = time_full[1:]
                        time_full = time_full + time_alone
                        self.full_formula.setText(time_full)
                        self.alone_sign.setText(\
                            "=" + str(eval(self.full_formula.text())))
                    elif "=" in self.alone_sign.text():
                        pass
                    elif len(self.alone_sign.text()) > 0:
                        if not self.alone_sign.text() == "0":
                            self.full_formula.setText(\
                                self.full_formula.text() + \
                                self.alone_sign.text())
                            if str(eval(\
                                self.full_formula.text()))[-1] == "0" and \
                                str(eval(self.full_formula.text()))[-2] == ".":
                                self.alone_sign.setText(\
                                "=" + str(int(eval(self.full_formula.text()))))
                            else:
                                self.alone_sign.setText(\
                                "=" + str(eval(self.full_formula.text())))
                    else:
                        if str(self.full_formula.text()[-1]).isdigit() or \
                            str(self.alone_sign.text()[-1]).isdigit():
                            self.full_formula.setText(\
                                self.full_formula.text() + \
                                self.alone_sign.text())
                            self.alone_sign.setText(\
                            "=" + str(eval(self.full_formula.text())))
                else:
                        if self.alone_sign.text()[-1] == ".":
                            self.alone_sign.setText(\
                            self.alone_sign.text() + "0")
                        if "=" in self.alone_sign.text():
                            self.alone_sign.clear()
                        if self.full_formula.text() == "0":
                            self.full_formula.clear()
                        if self.text() == "x^y":
                            self.full_formula.setText(\
                            self.full_formula.text() + \
                            self.alone_sign.text() + "**")
                            self.alone_sign.setText("0")
                        else: 
                            self.full_formula.setText(\
                            self.full_formula.text() + \
                            self.alone_sign.text() + self.text())
                            self.alone_sign.setText("0")
        return super().mousePressEvent(e)


class ClickableLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        logger.info("Clickable label initialized.")

    def mouse_Press_Event(self, event):
        if event.button() == Qt.LeftButton:
            clipboard = QGuiApplication.clipboard()
            clipboard.setText(self.text())
            logger.info(f"Text copied to clipboard: {self.text()}")


class MainWindow(QMainWindow):  
    def __init__(self):
        super(MainWindow, self).__init__()
        logger.info("Main window initialized.")
        self.setFixedSize(QSize(550, 300))
        self.setWindowTitle("кальклюкатор")
        self.full_formula = ClickableLabel()
        self.full_formula.setFixedWidth(450)
        self.full_formula.setAlignment(Qt.AlignRight)
        self.alone_sign = ClickableLabel()
        self.alone_sign.setAlignment(Qt.AlignRight)
        self.alone_sign.setFixedWidth(450)
#main widget
        main_widget = QWidget()
        main_widget.setLayout(QVBoxLayout())
#labels
        labels_frame = QFrame()
        labels_frame.setLineWidth(3)
        labels_frame.setFrameStyle(QFrame.WinPanel | QFrame.Plain)
        labels_frame.setLayout(QVBoxLayout())
        labels_frame.layout().addWidget(self.full_formula, \
                                        alignment=Qt.AlignmentFlag.AlignRight)
        labels_frame.layout().addWidget(self.alone_sign, \
                                        alignment=Qt.AlignmentFlag.AlignRight)
        labels_frame.setSizePolicy(QSizePolicy.Policy.Expanding, \
                                   QSizePolicy.Policy.Fixed)
        main_widget.layout().addWidget(labels_frame)
        labels_frame = QFrame()
        layout_butt = QGridLayout()
        layout_butt.addWidget(CustomQPushButton("log", self, True), 0, 0)
        layout_butt.addWidget(CustomQPushButton("ln", self,True), 1, 0)
        layout_butt.addWidget(CustomQPushButton("sin", self,True), 2, 0)
        layout_butt.addWidget(CustomQPushButton("cos", self,True), 3, 0)
        layout_butt.addWidget(CustomQPushButton("tan", self,True), 4, 0)
        layout_butt.addWidget(CustomQPushButton("cot", self,True), 5, 0)
        layout_butt.addWidget(CustomQPushButton("sqrt", self), 0, 1)
        layout_butt.addWidget(CustomQPushButton("x^y", self), 1, 1)
        layout_butt.addWidget(CustomQPushButton("1", self), 2, 1)
        layout_butt.addWidget(CustomQPushButton("4", self), 3, 1)
        layout_butt.addWidget(CustomQPushButton("7", self), 4, 1)
        layout_butt.addWidget(CustomQPushButton("+/-", self), 5, 1)
        layout_butt.addWidget(CustomQPushButton("%", self), 0, 2)
        layout_butt.addWidget(CustomQPushButton("C", self), 1, 2)
        layout_butt.addWidget(CustomQPushButton("2", self), 2, 2)
        layout_butt.addWidget(CustomQPushButton("5", self), 3, 2)
        layout_butt.addWidget(CustomQPushButton("8", self), 4, 2)
        layout_butt.addWidget(CustomQPushButton("0", self), 5, 2)
        layout_butt.addWidget(CustomQPushButton("(", self), 0, 3)
        layout_butt.addWidget(CustomQPushButton("CE", self), 1, 3)
        layout_butt.addWidget(CustomQPushButton("3", self), 2, 3)
        layout_butt.addWidget(CustomQPushButton("6", self), 3, 3)
        layout_butt.addWidget(CustomQPushButton("9", self), 4, 3)
        layout_butt.addWidget(CustomQPushButton(".", self), 5, 3)
        layout_butt.addWidget(CustomQPushButton(")", self), 0, 4)
        layout_butt.addWidget(CustomQPushButton("/", self), 1, 4)
        layout_butt.addWidget(CustomQPushButton("*", self), 2, 4)
        layout_butt.addWidget(CustomQPushButton("-", self), 3, 4)
        layout_butt.addWidget(CustomQPushButton("+", self), 4, 4)
        layout_butt.addWidget(CustomQPushButton("=", self), 5, 4)
        switcher = CustomQPushButton("Инжинерный", self)
        switcher.clicked.connect(lambda: [\
            button.eng_toggle(switcher) for button\
                in self.findChildren(CustomQPushButton)])
        switcher.clicked.connect(lambda: switcher.setText('Обычный')\
                                if switcher.text() == "Инжинерный" else\
                                 switcher.setText("Инжинерный"))
        layout_butt.addWidget(switcher, 0, 5)
        layout_butt.addWidget(CustomQPushButton("MC", self), 1, 5)
        layout_butt.addWidget(CustomQPushButton("MR", self), 2, 5)
        layout_butt.addWidget(CustomQPushButton("MS", self), 3, 5)
        layout_butt.addWidget(CustomQPushButton("M+", self), 4, 5)
        layout_butt.addWidget(CustomQPushButton("M-", self), 5, 5)
        main_widget.layout().addLayout(layout_butt)
        self.setCentralWidget(main_widget)
            

if __name__ == "__main__":
    logger.info("Application started.")
    try:
        app = QApplication([])
        window = MainWindow()
        window.show()
        app.exec()
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")