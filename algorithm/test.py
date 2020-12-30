import os
import subprocess
import threading
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QAction, QTextEdit, QTextBrowser
import sys


class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):  # real signature unknown; restored from __doc__
        """ flush(self) """
        pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.message = ''
        self.initUI()

    def initUI(self):
        self.resize(821, 615)
        self.setWindowTitle('恶意URL检测系统')
        #菜单栏
        self.menubar = self.menuBar()
        self.file = self.menubar.addMenu("功能")
        runAct = QAction('run', self)
        runAct.triggered.connect(self.run_train)
        self.file.addAction(runAct)

        self.showEdit = QTextBrowser(self)
        self.showEdit.setGeometry(QtCore.QRect(50, 50, 700, 500))
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        sys.stderr = EmittingStream(textWritten=self.normalOutputWritten)
        self.my_thread = MyThread()

    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        # self.showEdit.append(text)
        cursor = self.showEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.showEdit.setTextCursor(cursor)
        self.showEdit.ensureCursorVisible()

    def run_train(self):
        self.my_thread.start()


class MyThread(QtCore.QThread):  # 线程类
    # my_signal = pyqtSignal(str)  # 自定义信号对象。参数str就代表这个信号可以传一个字符串
    def __init__(self):
        super(MyThread, self).__init__()
        # self.count = 0
        self.is_on = True

    def run(self):  # 线程执行函数
        self.handle = ctypes.windll.kernel32.OpenThread(  # @UndefinedVariable
            win32con.PROCESS_ALL_ACCESS, False, int(QtCore.QThread.currentThreadId()))
        while self.is_on:
            import parallel_train
            parallel_train.train(0.2, 128, 5, 200)
            self.is_on = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    mainwindow.show()
    sys.exit(app.exec_())
