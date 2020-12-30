import view.url_main
from PyQt5.QtWidgets import QApplication
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainwindow = view.url_main.MainWindow()
    mainwindow.show()
    sys.exit(app.exec_())