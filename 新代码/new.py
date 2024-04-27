# coding:utf-8
import sys
from PyQt5.QtCore import Qt, QRect, QUrl, QSize
from PyQt5.QtGui import QIcon, QPainter, QImage, QBrush, QColor, QFont, QDesktopServices, QPixmap
from PyQt5.QtWidgets import QApplication, QFrame, QStackedWidget, QHBoxLayout, QLabel

from qfluentwidgets import (NavigationInterface, NavigationItemPosition, NavigationWidget, MessageBox,
                            isDarkTheme, setTheme, Theme, setThemeColor, qrouter, FluentWindow, NavigationAvatarWidget,
                            NavigationPanel, NavigationToolButton)
from qfluentwidgets import FluentIcon as FIF
from qframelesswindow import FramelessWindow, StandardTitleBar,AcrylicWindow
from GUI.interface.CIGIN_interface import Func1Widget as CIGINWidget
from GUI.interface.Home_interface import HomeWidget
from GUI.interface.Tox21_interface import Func2Widget as Tox21Widget
from GUI.interface.File_interface import Func3Widget as FileimportWidget
from tox_21.prediction import resource_path
class Window(FramelessWindow):

    def __init__(self):
        super().__init__()
        self.setTitleBar(StandardTitleBar(self))

        # use dark theme mode
        # setTheme(Theme.DARK)

        # change the theme color
        # setThemeColor('#0078d4')

        self.hBoxLayout = QHBoxLayout(self)
        self.navigationInterface = NavigationInterface(self, showMenuButton=True)
        self.navigationInterface.setExpandWidth(200)
        self.stackWidget = QStackedWidget(self)
        #self.stackWidget.setMinimumSize(QSize(440, 500))

        # create sub interface
        #self.settingInterface = Widget('Setting Interface', self)
        self.welcomeInterface = HomeWidget()
        self.func1Interface=CIGINWidget()
        self.func2Interface=Tox21Widget()
        self.fileimportInterface=FileimportWidget()

        # initialize layout
        self.initLayout()

        # add items to navigation interface
        self.initNavigation()

        self.initWindow()

    def initLayout(self):
        self.hBoxLayout.setSpacing(0)
        self.hBoxLayout.setContentsMargins(0, self.titleBar.height(), 0, 0)
        self.hBoxLayout.addWidget(self.navigationInterface)
        self.hBoxLayout.addWidget(self.stackWidget)
        self.hBoxLayout.setStretchFactor(self.stackWidget, 1)

    def initNavigation(self):

        self.addSubInterface(self.welcomeInterface, FIF.HOME, '首页')
        self.navigationInterface.addSeparator()
        pixmap1 = QPixmap(resource_path("GUI/icon/检验报告.png"))
        #icon.actualSize(QSize(32, 32))
        self.addSubInterface(self.func1Interface, QIcon(pixmap1), '溶剂化自由能预测')
        pixmap2 = QPixmap(resource_path("GUI/images/中毒.png"))
        self.addSubInterface(self.func2Interface, QIcon(pixmap2), 'Tox21毒性预测')
        self.navigationInterface.addSeparator()
        pixmap3 = QPixmap(resource_path("GUI/images/批量导入.png"))
        self.addSubInterface(self.fileimportInterface, QIcon(pixmap3), '批量导入')


        # add custom widget to bottom

        #self.addSubInterface(self.settingInterface, FIF.SETTING, 'Settings', NavigationItemPosition.BOTTOM)

        #!IMPORTANT: don't forget to set the default route key if you enable the return button
        #qrouter.setDefaultRouteKey(self.stackWidget, self.musicInterface.objectName())

        # set the maximum width
        self.navigationInterface.setExpandWidth(200)

        self.stackWidget.currentChanged.connect(self.onCurrentInterfaceChanged)
        self.stackWidget.setCurrentIndex(0)

        # always expand
        #self.navigationInterface.setCollapsible(False)

    def initWindow(self):
        self.resize(540, 480)
        self.setWindowIcon(QIcon(resource_path("title.png")))
        self.setWindowTitle('Detective Molecular Inspector 分子探长')
        self.titleBar.setAttribute(Qt.WA_StyledBackground)

        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)
        self.setQss()

    def addSubInterface(self, interface, icon, text: str, position=NavigationItemPosition.TOP, parent=None):
        """ add sub interface """
        self.stackWidget.addWidget(interface)
        self.navigationInterface.addItem(
            routeKey=interface.objectName(),
            icon=icon,
            text=text,
            onClick=lambda: self.switchTo(interface),
            position=position,
            tooltip=text,
            parentRouteKey=parent.objectName() if parent else None
        )

    def setQss(self):
        color = 'dark' if isDarkTheme() else 'light'
        with open(f'GUI/resource/{color}/demo.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def switchTo(self, widget):
        self.stackWidget.setCurrentWidget(widget)

    def onCurrentInterfaceChanged(self, index):
        widget = self.stackWidget.widget(index)
        self.navigationInterface.setCurrentItem(widget.objectName())

if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    '''QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)'''
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    w = Window()
    w.show()
    app.exec_()