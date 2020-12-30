# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\Desktop\malcious URL\url_main.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction
import view.url_detect
import view.url_train
import view.help1
import view.help2
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.methedid = 0
        self.InitUI()

    def InitUI(self):

        self.resize(821, 615)
        self.setWindowTitle('恶意URL检测系统')

        self.label = QtWidgets.QLabel('恶意URL检测系统', self)
        self.label.setGeometry(QtCore.QRect(200, 200, 431, 171))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(32)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        #菜单栏
        self.menubar = self.menuBar()
        self.file = self.menubar.addMenu("功能")
        self.file.addAction(QAction('URL检测', self))
        self.file.addAction(QAction('训练模型', self))
        self.file.triggered[QAction].connect(self.processtrigger)
        self.file = self.menubar.addMenu("帮助")
        self.file.addAction(QAction('URL检测操作指南', self))
        self.file.addAction(QAction('模型训练操作指南', self))
        self.file.triggered[QAction].connect(self.processtrigger)

    def processtrigger(self, i):
        if i.text() == "URL检测":
            Ui_detection = view.url_detect.URLdetection(self)
            Ui_detection.show()
            self.hide()  ###
        elif i.text() == "训练模型":
            Ui_train = view.url_train.Train(self)
            Ui_train.show()  ###
            self.hide()  ###
        elif i.text() == "URL检测操作指南":
            help = view.help1.detection_help(self)  ###
            help.show()  ###
        elif i.text() == "模型训练操作指南":
            help = view.help2.train_help(self)  ###
            help.show()  ###
