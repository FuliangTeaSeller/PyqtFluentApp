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
    def __init__(self,title:str,text:str,  icon:FluentIcon,parent: QWidget | None = ...):
        super().__init__()
        self.setMinimumSize(300, 300)
        self.acrylicLabel=AcrylicLabel(20,QColor(105, 114, 168, 102))
        self.acrylicLabel.setImage(r'E:\Python\PyqtFluentApp\main\ui_app\resource\images\header.png')
        self.titleLabel=StrongBodyLabel(title)
        self.textLabel=BodyLabel(text)
        self.iconWidget = IconWidget(FIF.ADD)
        
        self.iconWidget.setFixedSize(64,64)
        
        self.vBoxLayout=VBoxLayout(self)
        self.vBoxLayout.addWidget(self.acrylicLabel)
        self.vBoxLayout.addWidget(self.titleLabel)
        self.vBoxLayout.addWidget(self.textLabel)
        self.vBoxLayout.addWidget(self.iconWidget)
        
        
        
        self.setStyleSheet('''
        # AcrylicLabel {
        #     /background-image: url("D:/Documents/university/competition/PyqtFluentApp/main/ui_app/resource/images/header.png");
        #     background-repeat: no-repeat;
        #     background-position: center;
        #     background-color: rgba(105, 114, 168, 102);
        #     border-radius: 20px;
        # }

        # IconWidget {
        #     qproperty-iconSize: 64px;
        #     qproperty-sizePolicy: QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed);
        #     margin: 0px auto;
        # }

        # StrongBodyLabel, BodyLabel {
        #     text-align: center;
        #     color: white;
        #     font-family: "Segoe UI", Arial, sans-serif;
        #     font-weight: bold;
        # }

        # StrongBodyLabel {
        #     font-size: 24px;
        # }

        # BodyLabel {
        #     font-size: 16px;
        # }

        VBoxLayout > StrongBodyLabel, VBoxLayout > BodyLabel {
            margin-top: auto;
            margin-bottom: 10px;
            text-align: center;
        }

        * {
            padding: 0px;
            margin: 0px;
        }
        ''')
        
        
        
        




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
        
        self.resultFlowLayout.addWidget(ExampleCard("title","subtitle",FIF.ADD))
        self.resultFlowLayout.addWidget(ExampleCard("title","subtitle",FIF.ADD))
        self.resultFlowLayout.addWidget(ExampleCard("title","subtitle",FIF.ADD))
        self.infoFlowLayout.addWidget(ExampleCard("title","subtitle",FIF.ADD))
        self.infoFlowLayout.addWidget(ExampleCard("title","subtitle",FIF.ADD))
        
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
    
