# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\论文\恶意网站\项目\申请材料\1\url_train.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QAction, QFileDialog, QMessageBox
import view.url_detect
import view.help1
import view.help2
import pandas as pd
import sys
import os
import threading


class EmittingStream(QtCore.QObject):  ###
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):  # real signature unknown; restored from __doc__
        """ flush(self) """
        pass


class Train(QMainWindow):
    def __init__(self, *argv):
        super().__init__(*argv)
        self.csv = ""
        self.method = 0
        self.validate = 0.1
        self.batch_size = 32
        self.epoch = 20
        self.dims = 200
        self.my_thread = None
        self.InitUI()

    def InitUI(self):

        self.resize(825, 615)
        self.setWindowTitle("模型训练")

        self.groupBox = QtWidgets.QGroupBox("超参数选择", self)
        self.groupBox.setGeometry(QtCore.QRect(40, 40, 501, 281))

        self.label = QtWidgets.QLabel("交叉验证比例：", self.groupBox)
        self.label.setGeometry(QtCore.QRect(30, 40, 101, 16))

        self.label_2 = QtWidgets.QLabel("batch_size", self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(30, 100, 91, 16))

        self.label_5 = QtWidgets.QLabel("迭代次数：", self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(30, 160, 91, 16))

        self.label_6 = QtWidgets.QLabel("优化算法：", self.groupBox)
        self.label_6.setGeometry(QtCore.QRect(30, 220, 91, 16))

        self.comboBox = QtWidgets.QComboBox(self.groupBox)
        self.comboBox.setGeometry(QtCore.QRect(140, 40, 93, 22))
        self.comboBox.addItem("0.1")
        self.comboBox.addItem("0.2")
        self.comboBox.addItem("0.3")
        self.comboBox.addItem("0.4")
        self.comboBox.addItem("0.5")
        self.comboBox.currentIndexChanged[int].connect(self.select_validate)

        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_2.setGeometry(QtCore.QRect(140, 100, 93, 22))
        self.comboBox_2.addItem("32")
        self.comboBox_2.addItem("64")
        self.comboBox_2.addItem("128")
        self.comboBox_2.currentIndexChanged[int].connect(self.select_batch_size)
        self.comboBox_5 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_5.setGeometry(QtCore.QRect(140, 160, 93, 22))
        self.comboBox_5.addItem("20")
        self.comboBox_5.addItem("50")
        self.comboBox_5.addItem("100")
        self.comboBox_5.addItem("200")
        self.comboBox_5.currentIndexChanged[int].connect(self.select_epoch)

        self.comboBox_6 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_6.setGeometry(QtCore.QRect(140, 220, 93, 22))
        self.comboBox_6.addItem("adam")

        self.comboBox_9 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_9.setGeometry(QtCore.QRect(380, 40, 93, 22))
        self.comboBox_9.addItem("2")

        self.label_9 = QtWidgets.QLabel("IndRnn层数:", self.groupBox)
        self.label_9.setGeometry(QtCore.QRect(270, 40, 91, 16))

        self.comboBox_10 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_10.setGeometry(QtCore.QRect(380, 100, 93, 22))
        self.comboBox_10.addItem("185")
        self.comboBox_10.addItem("190")
        self.comboBox_10.addItem("195")
        self.comboBox_10.addItem("200")
        self.comboBox_10.currentIndexChanged[int].connect(self.select_dims)
        self.label_10 = QtWidgets.QLabel("特征维度:", self.groupBox)
        self.label_10.setGeometry(QtCore.QRect(270, 100, 91, 16))

        self.label_12 = QtWidgets.QLabel("注意力机制:", self.groupBox)
        self.label_12.setGeometry(QtCore.QRect(270, 160, 91, 16))

        self.comboBox_11 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_11.setGeometry(QtCore.QRect(380, 160, 93, 22))

        self.comboBox_11.addItem("Self-Attention")
        self.comboBox_11.addItem("Hi-Attention")
        self.comboBox_11.addItem("MultiHead-Attention")

        self.groupBox_2 = QtWidgets.QGroupBox("算法训练", self)
        self.groupBox_2.setGeometry(QtCore.QRect(570, 40, 211, 281))

        self.pushButton = QtWidgets.QPushButton("样本文件导入", self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(310, 215, 121, 31))
        self.pushButton.clicked.connect(self.btn_upload_clicked)

        self.label_13 = QtWidgets.QLabel("算法选择", self.groupBox_2)
        self.label_13.setGeometry(QtCore.QRect(30, 40, 101, 16))

        self.comboBox_12 = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox_12.setGeometry(QtCore.QRect(40, 70, 151, 21))
        self.comboBox_12.addItem("双向独立循环网络与胶囊网络串行联合算法")
        self.comboBox_12.addItem("独立循环神经网络与胶囊网络并行联合算法")
        self.comboBox_12.currentIndexChanged[int].connect(self.select_method)

        self.pushButton_2 = QtWidgets.QPushButton("开始训练", self.groupBox_2)
        self.pushButton_2.setGeometry(QtCore.QRect(40, 150, 131, 41))
        self.pushButton_2.clicked.connect(self.run_train)

        self.pushButton_2 = QtWidgets.QPushButton("结束训练", self.groupBox_2)
        self.pushButton_2.setGeometry(QtCore.QRect(40, 210, 131, 41))
        self.pushButton_2.clicked.connect(self.stop_train)

        self.textBrowser = QtWidgets.QTextBrowser(self)
        self.textBrowser.setGeometry(QtCore.QRect(40, 340, 741, 191))

        self.menubar = self.menuBar()
        self.file = self.menubar.addMenu("功能")
        self.file.addAction(QAction('URL检测', self))
        self.file.addAction(QAction('训练模型', self))
        self.file.triggered[QAction].connect(self.processtrigger)
        self.file = self.menubar.addMenu("帮助")
        self.file.addAction(QAction('URL检测操作指南', self))
        self.file.addAction(QAction('模型训练操作指南', self))
        self.file.triggered[QAction].connect(self.processtrigger)

        #self.pushButton_3

        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)  ###
        sys.stderr = EmittingStream(textWritten=self.normalOutputWritten)  ###

    def closeEvent(self, event):
        self.hide()
        self.parent().show()

    def processtrigger(self, i):
        if i.text() == "URL检测":
            Ui_detection = view.url_detect.URLdetection(self.parent())
            Ui_detection.show()
            self.hide()
            pass
        elif i.text() == "训练模型":
            # Ui_train = Train(self)
            # Ui_train.show()
            # Ui_train.exec_()
            pass
        elif i.text() == "URL检测操作指南":
            help = view.help1.detection_help(self)
            help.show()
        elif i.text() == "模型训练操作指南":
            help = view.help2.train_help(self)
            help.show()

    def normalOutputWritten(self, text):  ###
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        # self.showEdit.append(text)
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()

    def btn_upload_clicked(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'open file', '/', 'csv files(*.csv)')
        #fname为文件名，_为文件类型
        if fname:
            self.upload = 1
            df = pd.read_csv(fname)
            print(df)
            msg_box = QMessageBox.warning(self, '提示', '文件上传成功')
        else:
            self.upload = 0

    def select_method(self, i):
        #如果用户选择了第一个，则添加第一个到接口中
        if i == 0:
            self.method = 0
        else:
            self.method = 1

    #交叉验证比例选择
    def select_validate(self, i):
        if i == 0:
            self.validate = 0.1
        elif i == 1:
            self.validate = 0.2
        elif i == 2:
            self.validate = 0.3
        elif i == 3:
            self.validate = 0.4
        elif i == 4:
            self.validate = 0.5

    #batch_size选择
    def select_batch_size(self, i):
        if i == 0:
            self.batch_size = 32
        elif i == 1:
            self.batch_size = 64
        elif i == 2:
            self.batch_size = 128

    #epoch选择
    def select_epoch(self, i):
        if i == 0:
            self.epoch = 20
        elif i == 1:
            self.epoch = 50
        elif i == 2:
            self.epoch = 100
        elif i == 3:
            self.epoch = 200

    #dims选择
    def select_dims(self, i):
        if i == 0:
            self.dims = 185
        if i == 1:
            self.dims = 190
        if i == 2:
            self.dims = 195
        if i == 3:
            self.dims = 200

    def run_train(self):  ###
        if self.method == 0:  #串行
            #获取验证比例，batchsize，epochs，featuredim
            self.my_thread = Serial(self.validate, self.batch_size, self.epoch, self.dims)  ###
            self.my_thread.start()
        elif self.method == 1:  #并行
            self.my_thread = Parallel(self.validate, self.batch_size, self.epoch, self.dims)  ###
            self.my_thread.start()

    def stop_train(self):
        if self.my_thread != None:
            if self.my_thread.is_on == True:
                self.my_thread.terminate()
                print(self.my_thread.isRunning())
                self.my_thread = None
                print("训练已被中断")
            else:
                QMessageBox.warning(self, '提示', '未开始训练')
        else:
            QMessageBox.warning(self, '提示', '未开始训练')


class Serial(QtCore.QThread):  # 线程类
    # my_signal = pyqtSignal(str)  # 自定义信号对象。参数str就代表这个信号可以传一个字符串
    def __init__(self, val_rate, batchsize, epochs, featuredim):
        super(Serial, self).__init__()
        # self.count = 0
        self.validate = val_rate
        self.batch_size = batchsize
        self.epoch = epochs
        self.dims = featuredim
        self.is_on = True

    def run(self):  # 线程执行函数
        # self.handle = ctypes.windll.kernel32.OpenThread(  # @UndefinedVariable
        #     win32con.PROCESS_ALL_ACCESS, False, int(QtCore.QThread.currentThreadId()))
        print(os.getcwd())
        print('双向独立循环网络与胶囊网络串行联合算法')
        while self.is_on:
            import algorithm.serial_train
            # print(self.validate, self.batch_size, self.epoch, self.dims)
            algorithm.serial_train.train(self.validate, self.batch_size, self.epoch, self.dims)
            self.is_on = False


class Parallel(QtCore.QThread):  # 线程类
    # my_signal = pyqtSignal(str)  # 自定义信号对象。参数str就代表这个信号可以传一个字符串
    def __init__(self, val_rate, batchsize, epochs, featuredim):
        super(Parallel, self).__init__()
        # self.count = 0
        self.validate = val_rate
        self.batch_size = batchsize
        self.epoch = epochs
        self.dims = featuredim
        self.is_on = True

    def run(self):  # 线程执行函数
        # self.handle = ctypes.windll.kernel32.OpenThread(  # @UndefinedVariable
        #     win32con.PROCESS_ALL_ACCESS, False, int(QtCore.QThread.currentThreadId()))
        print(os.getcwd())
        print('独立循环神经网络与胶囊网络并行联合算法')
        while self.is_on:
            import algorithm.parallel_train
            # print(self.validate, self.batch_size, self.epoch, self.dims)
            algorithm.parallel_train.train(self.validate, self.batch_size, self.epoch, self.dims)
            self.is_on = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    UI = Train()
    UI.show()
    sys.exit(app.exec_())