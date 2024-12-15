import datetime
import json
import subprocess
import sys
from os import mkdir

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTreeWidgetItemIterator


class MainWindow(QMainWindow):
    """
    Главное окно приложения.
    """
    def __init__(self) -> None:
        """
        Инициализация главного окна.
        """
        super().__init__()
        uic.loadUi("src/forms/main_window.ui", self)
        self.about_window = AboutWindow(self)
        self.about_quit = AboutQuit(self)
        self.unsaved = False
        self.notes_dict = {}
        self.last = None
        self.lastIndex = 0
        self.file = {
            "notes_dict": self.notes_dict,
            "last": self.last
        }
        
        try: ## open .json with notes
            with open("userFiles/data.json", "r", encoding="utf8") as file:
                self.file = json.load(file)
                self.notes_dict = self.file["notes_dict"]
                self.last = self.file["last"]
                self.redraw_list_menu()
                self.selectItem(self.last)
        except: ## create .json if there is no such
            with open("userFiles/data.json", "w", encoding="utf8") as file:
                self.file = {
                    "notes_dict": self.notes_dict,
                    "last": self.last
                }
                json.dump(self.file, file, indent=4)
                self.redraw_list_menu()

        self.noteTitleEdit.textChanged.connect(lambda: self.note_changed())
        self.noteTextEdit.textChanged.connect(lambda: self.note_changed())
        self.saveButton.clicked.connect(lambda: self.about_quit.save_changes())
        self.deleteButton.clicked.connect(lambda: self.delete_note())
        self.notesList.currentItemChanged.connect(self.load_note)
        self.actionNewNote.triggered.connect(lambda: self.new_note())
        self.actionSaveNote.triggered.connect(
            lambda: self.about_quit.save_changes()
            )
        self.actionRemoveNote.triggered.connect(
            lambda: self.delete_note()
            )
        self.actionAbout.triggered.connect(
            lambda: self.about_window.help_show()
            )
        self.actionQuit.triggered.connect(lambda: self.close_app())

    def new_note(self) -> None:
        """
        Создание новой заметки.
        """
        if self.unsaved == True:
            self.about_quit.save_changes_window()
        self.last = None
        self.noteTitleEdit.setText("")
        self.noteTextEdit.setPlainText("")
        self.setWindowTitle("MyMemo")
        self.note_unsaved(False)
        self.notesList.setCurrentItem(None)
        self.deleteButton.setDisabled(True)
        self.actionRemoveNote.setDisabled(True)
        self.saveButton.setDisabled(True)
        self.actionSaveNote.setDisabled(True)
    
    def select_item(self, item_name: Optional[str]) -> None:
        """
        Выбор заметки из списка.

        Args:
            item_name (str): Название заметки.
        """
        if item_name != None:
            iterator = QTreeWidgetItemIterator(self.notesList, QTreeWidgetItemIterator.All)
            while iterator.value():
                item = iterator.value()
                if item.text(0) == item_name:
                    self.notesList.setCurrentItem(item, 1)
                    self.load_note(item, None)
                iterator += 1
    
    def note_changed(self) -> None:
        """
        Обработка изменения заметки.
        """
        if self.get_note_title().strip() != "":
            self.setWindowTitle("MyMemo - " + self.get_note_title().strip())
        else:
            self.setWindowTitle("MyMemo - untitled")
        self.note_unsaved()
        if self.get_note_title().strip() == "":
            self.saveButton.setDisabled(True)
            self.actionSaveNote.setDisabled(True)
        else:
            self.saveButton.setEnabled(True)
            self.actionSaveNote.setEnabled(True)

    def redraw_list_menu(self) -> None:
        """
        Перерисовка списка заметок.
        """
        self.notesList.clear()
        for element in list(self.notes_dict.keys()):
            self.last = QtWidgets.QTreeWidgetItem(self.notesList, 
                                      [self.notes_dict[element]["title"],
                                       self.notes_dict[element]["date"]]
                                      ).text(0)

    def note_unsaved(self, status: bool = True) -> None:
        """
        Пометка состояния несохраненной заметки.

        Args:
            status (bool): Состояние несохраненной заметки.
        """
        if status:
            if self.windowTitle()[-1] != "*":
                self.setWindowTitle(self.windowTitle() + "*")
            self.unsaved = True
            self.saveButton.setDisabled(True)
            self.actionSaveNote.setDisabled(True)
        else:
            if self.windowTitle()[-1] == "*":
                self.setWindowTitle(self.windowTitle()[:-1])
            self.unsaved = False
            self.actionSaveNote.setEnabled(True)
            self.saveButton.setEnabled(True)

    def load_note(self, item: Optional[QTreeWidgetItem], last_item: Optional[QTreeWidgetItem]) -> None:
        """
        Загрузка заметки из памяти.

        Args:
            item (QTreeWidgetItem): Текущий элемент.
            last_item (QTreeWidgetItem): Предыдущий элемент.
        """
        if item != last_item and item != None:
            self.last = item.text(0)
            if item != None:
                current_note_name = item.text(0)
                if current_note_name == "":
                    current_note_name = "untitled"
                current_note_date = item.text(1)
                current_note = self.notes_dict[current_note_name]
                self.noteTitleEdit.setText(current_note["title"])
                self.noteTextEdit.setPlainText(current_note["text"])
                self.setWindowTitle("MyMemo - " + current_note_name)
                self.note_unsaved(False)
                self.deleteButton.setEnabled(True)
                self.actionRemoveNote.setEnabled(True)
                self.saveButton.setDisabled(True)
                self.actionSaveNote.setDisabled(True)

    def delete_note(self) -> None:
        """
        Удаление заметки.
        """
        self.last = None
        if len(self.notesList.selectedItems()) != 0:
            note_name = self.notesList.selectedItems()[0].text(0)
            if note_name.strip() == "":
                note_name = "untitled"
            self.notesList.takeTopLevelItem(
                self.notesList.indexOfTopLevelItem(
                    self.notesList.selectedItems()[0]
                    )
                )
            self.notes_dict.pop(note_name)
                
        self.noteTitleEdit.setText("")
        self.noteTextEdit.setPlainText("")
        self.setWindowTitle("MyMemo")
        self.note_unsaved(False)
        self.redraw_list_menu()
        self.actionRemoveNote.setDisabled(True)
        self.deleteButton.setDisabled(True)
        self.actionSaveNote.setDisabled(True)
        self.saveButton.setDisabled(True)
        with open("userFiles/data.json", "w", encoding="utf8") as file:
            self.file = {
                "notes_dict": self.notes_dict,
                "last": self.last
            }
            json.dump(self.file, file, indent=4)

    def get_note_title(self) -> str:
        """
        Получение заголовка заметки.

        Returns:
            str: Заголовок заметки.
        """
        return str(self.noteTitleEdit.text())
    
    def get_note_text(self) -> str:
        """
        Получение текста заметки.

        Returns:
            str: Текст заметки.
        """
        return self.noteTextEdit.toPlainText() # [note] returns [' ', '\t', '\n']
    
    def close_event(self, event: QtGui.QCloseEvent) -> None:
        """
        Обработка закрытия окна.

        Args:
            event (QtGui.QCloseEvent): Событие закрытия.
        """
        if self.unsaved == False:
            app.closeAllWindows()
            event.accept()
            with open("userFiles/data.json", "w", encoding="utf8") as file:
                self.file = {
                    "notes_dict": self.notes_dict,
                    "last": self.last
                }
                json.dump(self.file, file, indent=4)
        else:
            self.close_app()
            event.ignore()

    def close_app(self) -> None:
        """
        Закрытие приложения.
        """
        with open("userFiles/data.json", "w", encoding="utf8") as file:
            self.file = {
                "notes_dict": self.notes_dict,
                "last": self.last
            }
            json.dump(self.file, file, indent=4)
        if self.about_window.isVisible():
            self.about_window.close()
        elif self.unsaved:
            self.about_quit.save_changes_window()
        else:
            app.closeAllWindows()   


class TableWidgetItem(QtWidgets.QTableWidgetItem):
    """
    Переопределение сортировки
    """
    def __lt__(self, other: 'TableWidgetItem') -> bool:
        """
        Переопределение оператора.

        Args:
            other (TableWidgetItem): Другой элемент для сравнения.
        
        Returns:
            bool: Результат сравнения.
        """
        print(0)
        if self.notesList.sortColumn == 1:
            print(1)
            key1 = self.text(0)
            key2 = other.text(0)
            return self.notesList[key2]["date_to_seconds"] < self.notesList[key1]["date_to_seconds"]
        

class AboutWindow(QWidget):
    """
    Окно "О программе", вызываемое из верхнего меню или по нажатию CTRL+H.
    """
    def __init__(self, *args: tuple) -> None:
        """
        Инициализация
        """
        super().__init__()
        uic.loadUi("src/forms/about_window.ui", self)

    def help_show(self) -> None:
        """
        Показать окно "О программе".
        """
        if self.isVisible():
            self.close()
        else:
            self.move(mainWin.x() + (mainWin.width() // 2 - self.width() // 2), 
                      mainWin.y() + (mainWin.height() // 2 - self.height() // 2)
                      )
            self.show()
    
    def close_event(self, event: QtGui.QCloseEvent) -> None:
        """
        Закрытие
        """
        self.close()


class AboutQuit(QWidget):
    """
    Окно "Сохранить изменения", если пользователь выходит без сохранения
    """
    def __init__(self, *args: tuple) -> None:
        """
        Инициализация окна "Сохранить изменения".
        """
        super().__init__()
        uic.loadUi("src/forms/about_quit.ui", self)
        self.saveButton.clicked.connect(lambda: self.save_changes())
        self.deleteButton.clicked.connect(lambda: self.delete_changes())
        self.cancelButton.clicked.connect(lambda: self.cancel_changes())

    def save_changes_window(self) -> None:
        """
        Показать окно "Сохранить изменения".
        """
        self.move(
            mainWin.x() + (mainWin.width() // 2 - self.width() // 2),
            mainWin.y() + (mainWin.height() // 2 - self.height() // 2)
            )
        self.show()

    def save_changes(self) -> None:
        """
        Сохранить изменения в заметке.
        """
        if mainWin.last == None:
            if mainWin.get_note_title() not in list(mainWin.notes_dict.keys()):
                time_now = datetime.datetime.now()
                current_date_seconds_from_start = time_now.timestamp()
                current_date = str(time_now.strftime("%H:%M %d.%m.%Y"))
                
                if mainWin.get_note_title().strip() == "":
                    mainWin.notes_dict["untitled"] = {
                        "title": mainWin.get_note_title(),
                        "date": current_date,
                        "text": mainWin.get_note_text(),
                        "date_to_seconds": current_date_seconds_from_start
                    }
                else:
                    mainWin.notes_dict[mainWin.get_note_title()] = {
                        "title": mainWin.get_note_title(),
                        "date": current_date,
                        "text": mainWin.get_note_text(),
                        "date_to_seconds": current_date_seconds_from_start
                    }
            else:
                time_now = datetime.datetime.now()
                current_date_seconds_from_start = time_now.timestamp()
                current_date = str(time_now.strftime("%H:%M %d.%m.%Y"))
                mainWin.notes_dict[mainWin.get_note_title()] = {
                    "title": mainWin.get_note_title(),
                    "date": current_date,
                    "text": mainWin.get_note_text(),
                    "date_to_seconds": current_date_seconds_from_start
                }
        else:
            time_now = datetime.datetime.now()
            current_date_seconds_from_start = time_now.timestamp()
            current_date = str(time_now.strftime("%H:%M %d.%m.%Y"))
            del(mainWin.notes_dict[mainWin.last])
            mainWin.notes_dict[mainWin.get_note_title()] = {
                "title": mainWin.get_note_title(),
                "date": current_date,
                "text": mainWin.get_note_text(),
                "date_to_seconds": current_date_seconds_from_start
            }
        mainWin.redraw_list_menu()
        mainWin.note_unsaved(False)
        mainWin.select_item(mainWin.last)
        mainWin.saveButton.setDisabled(True)
        mainWin.actionSaveNote.setDisabled(True)
        mainWin.deleteButton.setEnabled(True)
        mainWin.actionRemoveNote.setEnabled(True)
        
        with open("userFiles/data.json", "w", encoding="utf8") as file: ## writing to file
            mainWin.file = {
                "notes_dict": mainWin.notes_dict,
                "last": mainWin.last
            }
            json.dump(mainWin.file, file, indent=4)
        if self.isVisible(): ## app closes only if it asked about quit, else just saving without exiting
            mainWin.close_app()

    def delete_changes(self) -> None:
        """
        Удаление изменений
        """
        mainWin.note_unsaved(False)
        with open("userFiles/data.json", "w", encoding="utf8") as file:
            mainWin.file = {
                "notes_dict": mainWin.notes_dict,
                "last": mainWin.last
            }
            json.dump(mainWin.file, file, indent=4)
        app.closeAllWindows()
        
    def cancel_changes(self) -> None:
        """
        Отмена изменений
        """
        self.close()

    def close_event(self, event: QtGui.QCloseEvent) -> None:
        """
        Обработчик события закрытия окна.
        Args:
        event (QtGui.QCloseEvent): Событие закрытия окна.
        """
        self.cancel_changes()


def except_hook(cls: type, exception: Exception,
                traceback: Optional[traceback]) -> None:
    """
    Обработчик исключений.
    Args:
        cls (type): Тип исключения.
        exception (Exception): Экземпляр исключения.
        traceback (traceback): Объект трассировки исключения.
    """
    sys.__excepthook__(cls, exception, traceback)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.excepthook = except_hook
    sys.exit(app.exec_())