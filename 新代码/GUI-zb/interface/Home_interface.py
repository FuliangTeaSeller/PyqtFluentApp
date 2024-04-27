from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QLabel, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QIcon, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtWidgets import QFileDialog, QScrollArea, QWidget, QFrame, QPlainTextEdit
from PyQt5 import QtCore, QtGui, QtWidgets
from GUI.UI.home_page import Ui_Form
from qfluentwidgets import FluentIcon as FIF

class HomeWidget(QFrame, Ui_Form):
    def __init__(self, parent=None):
        super().__init__()
        self.ui = Ui_Form
        self.setupUi(self)



