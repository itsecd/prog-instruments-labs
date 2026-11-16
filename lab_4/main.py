import sys
from cli import run_cli
from gui import AppGui
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli()
    else:
        app = QApplication(sys.argv)
        gui = AppGui()
        gui.show()
        sys.exit(app.exec_())