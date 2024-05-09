# coding:utf-8
import sys

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication

from qfluentwidgets.components.widgets.acrylic_label import AcrylicLabel


app = QApplication(sys.argv)
w = AcrylicLabel(20, QColor(105, 114, 168, 102))
w.setImage(r'D:\Documents\university\competition\PyqtFluentApp\main\ui_app\resource\images\header.png')
w.show()
app.exec_()