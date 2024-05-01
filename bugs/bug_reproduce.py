# coding:utf-8
import io, os
import sys
from PyQt5.QtCore import Qt, QSize,QUrl
from PyQt5.QtWidgets import QAction, QWidget, QVBoxLayout, QButtonGroup,QLabel,QStackedWidget,QApplication
from qfluentwidgets import (LineEdit,BodyLabel,PushButton,FlowLayout,VBoxLayout,Pivot)
from PyQt5.QtGui import QPainter,QFont
from PyQt5.QtWebEngineWidgets import QWebEngineView
from qfluentwidgets import FluentIcon as FIF
from PyQt5.QtCore import QUrl, QSize
from PyQt5.QtGui import QIcon, QDesktopServices,QPixmap
from PyQt5.QtWidgets import QApplication

from qfluentwidgets import (NavigationAvatarWidget, NavigationItemPosition, MessageBox, FluentWindow,MSFluentWindow,
                            SplashScreen)
from qfluentwidgets import FluentIcon as FIF

# from .predict_interface import PredictInterface
# from .gallery_interface import GalleryInterface
# from .home_interface import HomeInterface
# from .introduction_interface import IntroductionInterface
# from ..common.config import ZH_SUPPORT_URL, EN_SUPPORT_URL, cfg
# from ..common.icon import Icon
# from ..common.signal_bus import signalBus
# from ..common.translator import Translator
# from .gallery_interface import GalleryInterface
# from ..common.translator import Translator
# from GUI.interface.tox_result_interface import Result as tox_result

# from tox_21.prediction import resource_path

class Demo(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.setObjectName('predictInterface')
        self.tab = QWidget()
        # self.addSubInterface(self.tab, 'tab', '描述文字') 
               
        self.web_view = QWebEngineView(self)
        self.web_view.setMinimumSize(405, 350)
        
        
    def addSubInterface(self, widget: QLabel, objectName: str, text: str):
        widget.setObjectName(objectName)
        widget.setAlignment(Qt.AlignCenter)
        self.stackedWidget.addWidget(widget)

        # 使用全局唯一的 objectName 作为路由键
        self.pivot.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget)
        )

class MainWindow(FluentWindow):

    def __init__(self):
        super().__init__()
        
        self.resize(780, 780)
        self.setMinimumWidth(780)
        self.setMicaEffectEnabled(True)

        
        self.web_view = QWebEngineView(self)
        self.web_view.setMinimumSize(405, 350)

        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)
        self.show()
        QApplication.processEvents()
    


        
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec_()