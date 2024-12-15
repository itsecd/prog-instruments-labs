# -*- coding: utf-8 -*-

from os import mkdir ## import block
import subprocess
import sys

try:
    import datetime
    import json

    from PyQt5 import QtGui, QtWidgets, uic
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTreeWidgetItemIterator
except:
    for library in ["datetime", "PyQt5", "time", "json"]:
        subprocess.run(["pip install", library])
        
    import datetime
    import json

    from PyQt5 import QtGui, QtWidgets, uic, QtCore
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTreeWidgetItemIterator


class MainWindow(QMainWindow):
    def __init__(self): ## initialization
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
    
    def new_note(self): ## creating new note
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
    
    def selectItem(self, item_name): ## func to get note from list
        if item_name != None:
            iterator = QTreeWidgetItemIterator(self.notesList, QTreeWidgetItemIterator.All)
            while iterator.value():
                item = iterator.value()
                if item.text(0) == item_name:
                    self.notesList.setCurrentItem(item, 1)
                    self.load_note(item, None)
                iterator += 1
    
    def note_changed(self): # if note title changes window title changes too
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
    
    def redraw_list_menu(self): ## redrawing list one by one
        self.notesList.clear()
        for element in list(self.notes_dict.keys()):
            self.last = QtWidgets.QTreeWidgetItem(self.notesList, 
                                      [self.notes_dict[element]["title"],
                                       self.notes_dict[element]["date"]]
                                      ).text(0)
        
    def note_unsaved(self, status=True): ## marks that note unsaved
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
    
    def load_note(self, item, last_item): ## load note from memory
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

    def delete_note(self): ## delete note
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
    
    def get_note_title(self):
        return str(self.noteTitleEdit.text())
    
    def get_note_text(self):
        return self.noteTextEdit.toPlainText() # [note] returns [' ', '\t', '\n']
    
    def closeEvent(self, event: QtGui.QCloseEvent): ## if user want to close window we save note [trigger]
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
    
    def close_app(self): ## closing app from code
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


class TableWidgetItem(QtWidgets.QTableWidgetItem): ## reorganization of sorting
    def __lt__(self, other):
        print(0)
        if self.notesList.sortColumn == 1:
            print(1)
            key1 = self.text(0)
            key2 = other.text(0)
            return self.notesList[key2]["date_to_seconds"] < self.notesList[key1]["date_to_seconds"]
        

class AboutWindow(QWidget): ## window can be called from upper bar or pressing CTRL+H
    def __init__(self, *args):
        super().__init__()
        uic.loadUi("src/forms/about_window.ui", self)
    
    def help_show(self):
        if self.isVisible():
            self.close()
        else:
            self.move( ## middle align
                mainWin.x() + (mainWin.width() // 2 - self.width() // 2), mainWin.y() + (mainWin.height() // 2 - self.height() // 2)
                )
            self.show()
    
    def closeEvent(self, event: QtGui.QCloseEvent):
        self.close()


class AboutQuit(QWidget): ## window activating if you try to close app without saving
    def __init__(self, *args):
        super().__init__()
        uic.loadUi("src/forms/about_quit.ui", self)
        self.saveButton.clicked.connect(lambda: self.save_changes())
        self.deleteButton.clicked.connect(lambda: self.delete_changes())
        self.cancelButton.clicked.connect(lambda: self.cancel_changes())

    def save_changes_window(self): ## if note isn't saved - ask to save
        self.move(
            mainWin.x() + (mainWin.width() // 2 - self.width() // 2),
            mainWin.y() + (mainWin.height() // 2 - self.height() // 2)
            )
        self.show()
    
    def save_changes(self):
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
        mainWin.selectItem(mainWin.last)
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
    
    def delete_changes(self):
        mainWin.note_unsaved(False)
        with open("userFiles/data.json", "w", encoding="utf8") as file:
            mainWin.file = {
                "notes_dict": mainWin.notes_dict,
                "last": mainWin.last
            }
            json.dump(mainWin.file, file, indent=4)
        app.closeAllWindows()
        
    def cancel_changes(self):
        self.close()

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.cancel_changes()


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.excepthook = except_hook
    sys.exit(app.exec_())