# coding: utf-8
from PyQt5.QtCore import QUrl, QSize
from PyQt5.QtGui import QIcon, QDesktopServices,QPixmap
from PyQt5.QtWidgets import QApplication

from qfluentwidgets import (NavigationAvatarWidget, NavigationItemPosition, MessageBox, FluentWindow,MSFluentWindow,
                            SplashScreen)
from qfluentwidgets import FluentIcon as FIF

from .batchPredict_interface import BatchPredictInterface
from .predict_interface import PredictInterface
from .gallery_interface import GalleryInterface
from .home_interface import HomeInterface
from .introduction_interface import IntroductionInterface
from ..common.config import ZH_SUPPORT_URL, EN_SUPPORT_URL, cfg
from ..common.icon import Icon
from ..common.signal_bus import signalBus
from ..common.translator import Translator
# from ..common import resource
from ..resource import resource
# from GUI.interface.CIGIN_interface import Func1Widget as CIGINWidget
# from GUI.interface.Home_interface import HomeWidget
# from GUI.interface.Tox21_interface import Func2Widget as Tox21Widget
# from GUI.interface.File_interface import Func3Widget as FileimportWidget
# from tox_21.prediction import resource_path

class MainWindow(FluentWindow):

    def __init__(self):
        super().__init__()
        self.initWindow()

        # create sub interface
        self.homeInterface = HomeInterface(self)
        self.predictInterface= PredictInterface(self)
        self.BatchPredictInterface = BatchPredictInterface(self)
        self.introductionInterface = IntroductionInterface(self)

        # enable acrylic effect
        # self.navigationInterface.setAcrylicEnabled(True)

        self.connectSignalToSlot()
        # add items to navigation interface
        self.initNavigation()
        self.splashScreen.finish()
        self.updateFrameless()

    def connectSignalToSlot(self):
        signalBus.micaEnableChanged.connect(self.setMicaEffectEnabled)
        signalBus.switchToSampleCard.connect(self.switchToSample)
        signalBus.supportSignal.connect(self.onSupport)

    def initNavigation(self):
        # add navigation items
        t = Translator()
        self.addSubInterface(self.homeInterface, FIF.HOME, '首页')
        self.navigationInterface.addSeparator()

        pos = NavigationItemPosition.SCROLL
        self.addSubInterface(self.predictInterface, FIF.CHECKBOX,'预测界面', pos)
        self.addSubInterface(self.BatchPredictInterface, FIF.EDIT, '批量处理界面', pos)
        self.addSubInterface(self.introductionInterface, FIF.CAFE,'介绍界面', pos)
        
        

    def initWindow(self):
        # self.resize(960, 780)
        # self.setMinimumWidth(760)
        self.resize(780, 780)
        self.setMinimumWidth(780)
        self.setWindowIcon(QIcon(':/gallery/images/logo_new.png'))
        self.setWindowTitle('Detective Molecular Inspector 分子探长')

        # self.setMicaEffectEnabled(cfg.get(cfg.micaEnabled))

        # create splash screen
        self.splashScreen = SplashScreen(icon=self.windowIcon(), parent=self)
        self.splashScreen.setIconSize(QSize(106, 106))
        self.splashScreen.raise_()

        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)
        self.show()
        QApplication.processEvents()

    def onSupport(self):
        language = cfg.get(cfg.language).value
        if language.name() == "zh_CN":
            QDesktopServices.openUrl(QUrl(ZH_SUPPORT_URL))
        else:
            QDesktopServices.openUrl(QUrl(EN_SUPPORT_URL))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, 'splashScreen'):
            self.splashScreen.resize(self.size())

    def switchToSample(self, routeKey, index):
        """ switch to sample """
        interfaces = self.findChildren(GalleryInterface)
        for w in interfaces:
            if w.objectName() == routeKey:
                self.stackedWidget.setCurrentWidget(w, False)
                w.scrollToCard(index)
