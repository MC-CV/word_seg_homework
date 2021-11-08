#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from main import demo,demo_ngram
from Ui_spilit import Ui_MainWindow


class ControlBoard(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(ControlBoard, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.bClicked)

    def bClicked(self):
        try:
            input = self.lineEdit.text()
            output = demo_ngram(input)
            self.textBrowser_2.setText("\n\n\n\n\n" + output)
        except:
            self.textBrowser_2.setText("请输入正确的语句。")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ControlBoard()
    win.show()
    sys.exit(app.exec_())

