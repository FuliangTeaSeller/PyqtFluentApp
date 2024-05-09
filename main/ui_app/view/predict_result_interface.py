# coding:utf-8
import io, os
from time import sleep
from PyQt5.QtCore import Qt, QSize,QUrl,QTimer
from PyQt5.QtWidgets import QAction, QWidget, QVBoxLayout, QButtonGroup,QLabel,QStackedWidget,QHBoxLayout,QFrame,QPushButton
from qfluentwidgets import LineEdit,BodyLabel,PushButton,FlowLayout,VBoxLayout,Pivot,PlainTextEdit,InfoBar,InfoBarPosition,FluentWindow,StrongBodyLabel,IconWidget,FluentIcon,ScrollArea,FlowLayout
from PyQt5.QtGui import QPainter,QFont,QPixmap,QBrush,QColor
from qfluentwidgets.components.widgets.acrylic_label import AcrylicLabel
# from ..common.style_sheet import StyleSheet
from qfluentwidgets import FluentIcon as FIF
# from .gallery_interface import GalleryInterface

class ExampleCard(QWidget):
    def __init__(self, title, subtitle, image_path, icon:FluentIcon, parent=None):
        super().__init__(parent=parent)
        self.title = title
        self.subtitle = subtitle
        self.image_path = image_path
        # self.avatar_path = avatar_path

        # Widgets
        self.card = QFrame(self)
        self.titleLabel = StrongBodyLabel(title, self.card)
        self.subtitleLabel = QLabel(subtitle, self.card)
        self.iconWidget = IconWidget(icon, self.card)

        # Layouts
        self.vBoxLayout = QVBoxLayout(self)
        self.cardLayout = QVBoxLayout(self.card)
        self.bottomLayout = QHBoxLayout()

        self.__init_widget()

    def __init_widget(self):
        # Setup Avatar

        self.card.setMinimumSize(300, 300)
        self.iconWidget.setFixedSize(50, 50)
        # Setup Background Image
        background_pixmap = QPixmap(self.image_path).scaled(600, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Setup Layouts
        self.vBoxLayout.setSpacing(10)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.addWidget(self.card)

        self.cardLayout.setSpacing(10)
        self.cardLayout.setContentsMargins(10, 10, 10, 10)
        # self.topLayout.addWidget(self.icon, 0, Qt.AlignLeft)
        # self.topLayout.addStretch(1)

        self.titleLabel.setAlignment(Qt.AlignLeft)
        self.subtitleLabel.setAlignment(Qt.AlignLeft)

        self.bottomLayout.setSpacing(5)
        self.bottomLayout.addWidget(self.titleLabel, 0, Qt.AlignLeft)
        self.bottomLayout.addWidget(self.subtitleLabel, 0, Qt.AlignLeft)
        self.bottomLayout.addStretch(1)

        self.cardLayout.addLayout(self.bottomLayout)
        self.card.setLayout(self.cardLayout)

class ExampleCard2(QWidget):
    def __init__(self,title:str,text:str,  icon:FluentIcon,parent: QWidget | None = ...):
        super().__init__()
        self.setMinimumSize(300, 300)
        self.acrylicLabel=AcrylicLabel(20,QColor(105, 114, 168, 102))
        self.acrylicLabel.setImage(r'D:\Documents\university\competition\PyqtFluentApp\main\ui_app\resource\images\header.png')
        self.titleLabel=StrongBodyLabel(title)
        self.textLabel=BodyLabel(text)
        self.iconWidget = IconWidget(FIF.ADD)
        
        self.vBoxLayout=VBoxLayout(self)
        self.vBoxLayout.addWidget(self.acrylicLabel)
        self.vBoxLayout.addWidget(self.titleLabel)
        self.vBoxLayout.addWidget(self.textLabel)
        self.vBoxLayout.addWidget(self.iconWidget)
        
        
        
        




class PredictResultWidget(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.view = QWidget(self)
        self.view.setObjectName('view')
        # self.view.setMinimumSize(800, 600)
        self.setWidget(self.view)
        self.setObjectName('predictResultWidget')
        
        self.setWidgetResizable(True)
        # self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # self.vBoxLayout.addWidget(self.view)
        self.vBoxLayout=VBoxLayout(self)
        self.vBoxLayout.setSpacing(30)
        self.vBoxLayout.setAlignment(Qt.AlignTop)
        self.vBoxLayout.setContentsMargins(36, 20, 36, 36)
        
        

        
        self.resultLabel = StrongBodyLabel("result")
        self.resultFlowLayout = FlowLayout()
        self.resultWidget = QWidget(self)
        self.resultWidget.setLayout(self.resultFlowLayout)
        
        
        self.infoLabel = StrongBodyLabel("info")
        self.infoFlowLayout = FlowLayout()
        self.infoWidget = QWidget(self)
        self.infoWidget.setLayout(self.infoFlowLayout)
        
        self.resultFlowLayout.addWidget(ExampleCard2("title","subtitle",FIF.ADD))
        self.resultFlowLayout.addWidget(ExampleCard2("title","subtitle",FIF.ADD))
        self.resultFlowLayout.addWidget(ExampleCard2("title","subtitle",FIF.ADD))
        self.infoFlowLayout.addWidget(ExampleCard2("title","subtitle",FIF.ADD))
        self.infoFlowLayout.addWidget(ExampleCard2("title","subtitle",FIF.ADD))
        
        self.resultFlowLayout.setContentsMargins(30, 30, 30, 30)
        self.resultFlowLayout.setVerticalSpacing(20)
        self.resultFlowLayout.setHorizontalSpacing(10)

        # self.resultFlowLayout.addWidget(QPushButton('aiko'))
        # self.resultFlowLayout.addWidget(QPushButton('刘静爱'))
        # self.resultFlowLayout.addWidget(QPushButton('柳井爱子'))
        # self.resultFlowLayout.addWidget(QPushButton('aiko 赛高'))
        # self.resultFlowLayout.addWidget(QPushButton('aiko 太爱啦😘'))

        
        # gray_block1 = QWidget(self)
        # gray_block1.setFixedSize(100, 100)
        # gray_block1.setStyleSheet("background-color: gray;")
        
        # gray_block2 = QWidget(self)
        # gray_block2.setFixedSize(100, 100)
        # gray_block2.setStyleSheet("background-color: green;")
        # self.resultFlowLayout.addWidget(gray_block1)
        # self.infoFlowLayout.addWidget(gray_block2)
        
        self.vBoxLayout.addWidget(self.resultLabel)
        self.vBoxLayout.addWidget(self.resultWidget)
        self.vBoxLayout.addWidget(self.infoLabel)
        self.vBoxLayout.addWidget(self.infoWidget)
        
        # StyleSheet.GALLERY_INTERFACE.apply(self)

class PredictResultInterface(FluentWindow):
    def __init__(self,result, parent=None):
        super().__init__(parent)
        self.setObjectName('predictResultInterface')
        self.result = result
        
        self.predictResultWidget=PredictResultWidget(self)
        self.predictResultWidget.setObjectName('predictResultWidget')
        self.addSubInterface(self.predictResultWidget,FIF.HOME ,'result')
        self.navigationInterface.hide()

        
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    result = PredictResultInterface("result")
    result.show()
    # card=ExampleCard2(title="title",text="text",icon=FIF.ADD)
    # card.show()
    sys.exit(app.exec_())
    
