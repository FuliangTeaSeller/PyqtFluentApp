# coding:utf-8
import io, os
from time import sleep
from PyQt5.QtCore import Qt, QSize,QUrl,QTimer,QEasingCurve
from PyQt5.QtWidgets import QAction, QWidget, QVBoxLayout, QButtonGroup,QLabel,QStackedWidget,QHBoxLayout,QFrame,QPushButton,QScrollArea
from qfluentwidgets import LineEdit,BodyLabel,PushButton,FlowLayout,VBoxLayout,Pivot,PlainTextEdit,InfoBar,InfoBarPosition,FluentWindow,StrongBodyLabel,IconWidget,FluentIcon,ScrollArea,FlowLayout,SplashScreen
from PyQt5.QtGui import QPainter,QFont,QPixmap,QBrush,QColor,QIcon
from qfluentwidgets.components.widgets.acrylic_label import AcrylicLabel
# from ..common.style_sheet import StyleSheet
from qfluentwidgets import FluentIcon as FIF
# from .gallery_interface import GalleryInterface
from PIL import Image
import sys
import io
from PyQt5.QtWidgets import QLabel, QToolTip
from PyQt5.QtCore import Qt,QEvent
from PyQt5.QtGui import QPixmap, QColor, QIcon
from PyQt5.QtWidgets import QHBoxLayout
from Predictor.prediction import dict
class ExampleCard(QWidget):
    def __init__(self,title:str,text:str, imgpath:str, icon:FluentIcon,parent: QWidget | None = ...):
        super().__init__()
        self.setMinimumSize(300, 300)
        self.acrylicLabel=AcrylicLabel(10,QColor(255, 255, 255, 128))
        # self.acrylicLabel.setImage(r'E:\Python\PyqtFluentApp\main\ui_app\resource\images\header.png')
        # self.acrylicLabel.setImage('main/ui_app/resource/images/header.png')
        self.acrylicLabel.setImage(imgpath)
        self.titleLabel=StrongBodyLabel(title)
        self.textLabel=BodyLabel(text)
        # self.iconWidget = IconWidget(icon)
        
        self.acrylicLabel.setFixedSize(400,300)
        # self.iconWidget.setFixedSize(64,64)
        
        
        self.vBoxLayout=QVBoxLayout(self)
        self.hBoxLayout=QHBoxLayout(self.acrylicLabel)
        self.titleLayout=QVBoxLayout(self.acrylicLabel)
        
        self.vBoxLayout.addWidget(self.acrylicLabel)
        
        self.titleLayout.addWidget(self.titleLabel)
        self.titleLayout.addWidget(self.textLabel)
        
        self.hBoxLayout.addLayout(self.titleLayout)
        # self.hBoxLayout.addWidget(self.iconWidget)
        
        self.acrylicLabel.setStyleSheet("border-radius: 60px;")

        self.setStyleSheet('''
                        #    ExampleCard{
                        #        border-radius: 20px;
                        #        border: 1px solid black;
                        #             background-color: rgb(128, 128, 128);
                        #    }
                        AcrylicLabel {
                            # /background-image: url("D:/Documents/university/competition/PyqtFluentApp/main/ui_app/resource/images/header.png");
                            # background-repeat: no-repeat;
                            # background-position: center;
                            # background-color: rgba(105, 114, 168, 102);
                            # border-radius: 50px;
                        }

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

                        # VBoxLayout > StrongBodyLabel, VBoxLayout > BodyLabel {
                        #     margin-top: auto;
                        #     margin-bottom: 10px;
                        #     text-align: center;
                        # }

                        # * {
                        #     padding: 0px;
                        #     margin: 0px;
                        }
        ''')
        
class PredictResultWidget(QScrollArea):
    
    def __init__(self, result, parent=None):
        super().__init__(parent=parent)
        self.result = result
        
        self.setObjectName('predictResultWidget')
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.mainWidget = QWidget()
        self.setWidget(self.mainWidget)

        self.mainLayout = QHBoxLayout(self.mainWidget)
        
        self.vBoxLayout = QVBoxLayout()
        self.vBoxLayout.setSpacing(30)
        self.vBoxLayout.setAlignment(Qt.AlignTop)
        self.vBoxLayout.setContentsMargins(36, 20, 36, 36)

        self.imageLabel = QLabel()
        self.imageLabel.setFixedSize(400, 400)
        self.imageLabel.setStyleSheet("border: 1px solid black;")

        self.closeButton = QPushButton("Close")
        self.closeButton.clicked.connect(self.hideImage)

        self.overlayWidget = QWidget(self)
        self.overlayWidget.setGeometry(800, 0, 400, 450)  # 设置固定位置和大小
        self.overlayLayout = QVBoxLayout(self.overlayWidget)
        self.overlayLayout.addWidget(self.imageLabel)
        self.overlayLayout.addWidget(self.closeButton)

        self.mainLayout.addLayout(self.vBoxLayout)

        self.addResultWidgets(dict)
        # 图片区域
        self.picFlowLayout = FlowLayout()
        # self.addResultLabels()
        self.picWidget = QWidget()
        self.picWidget.setLayout(self.picFlowLayout)

        self.resultLabel = StrongBodyLabel("预测结果")
        self.resultFlowLayout = FlowLayout()
        self.resultWidget = QWidget()
        self.resultWidget.setLayout(self.resultFlowLayout)

        self.infoLabel = StrongBodyLabel("分子的物理化学信息")
        self.infoFlowLayout = FlowLayout()
        self.infoWidget = QWidget()
        self.infoWidget.setLayout(self.infoFlowLayout)

        # 添加示例卡片
        self.resultFlowLayout.addWidget(ExampleCard("分子的结构", "subtitle", 'main/ui_app/resource//images/result/background1.png', FIF.ADD))
        self.resultFlowLayout.addWidget(ExampleCard("分子的SMILES", "subtitle", 'main/ui_app/resource//images/result/background2.png', FIF.ADD))
        self.infoFlowLayout.addWidget(ExampleCard("吸收", "subtitle", 'main/ui_app/resource/images/result/background3.png', FIF.ADD))
        self.infoFlowLayout.addWidget(ExampleCard("分布", "subtitle", 'main/ui_app/resource/images/result/background4.png', FIF.ADD))
        self.infoFlowLayout.addWidget(ExampleCard("代谢","subtitle",'main/ui_app/resource/images/result/background5.png',FIF.ADD))
        self.infoFlowLayout.addWidget(ExampleCard("毒性","subtitle",'main/ui_app/resource/images/result/background6.png',FIF.ADD))
        
        self.resultFlowLayout.setContentsMargins(30, 30, 30, 30)
        self.resultFlowLayout.setVerticalSpacing(20)
        self.resultFlowLayout.setHorizontalSpacing(10)

        # 将各个部分添加到主布局中
        self.vBoxLayout.addWidget(self.picWidget)
        self.vBoxLayout.addWidget(self.resultLabel)
        self.vBoxLayout.addWidget(self.resultWidget)
        self.vBoxLayout.addWidget(self.infoLabel)
        self.vBoxLayout.addWidget(self.infoWidget)
        
            
    def addResultWidgets(self, titles):
        for i, (title, value) in enumerate(titles.items()):
            color = 'green' if self.result[0][i] == 1 else 'red'
            
            widget = QWidget()
            hBoxLayout = QHBoxLayout(widget)
            hBoxLayout.setSpacing(10)

            titleLabel = QLabel(title)
            hBoxLayout.addWidget(titleLabel)

            colorBlock = QLabel()
            colorBlock.setFixedSize(20, 20)
            colorBlock.setStyleSheet(f'background-color: {color};')
            hBoxLayout.addWidget(colorBlock)

            exclamationLabel = QLabel('!')
            exclamationLabel.setStyleSheet('color: orange; font-weight: bold; font-size: 16px;')
            hBoxLayout.addWidget(exclamationLabel)

            widget.setLayout(hBoxLayout)
            widget.mousePressEvent = lambda event, idx=i: self.showImage(idx)
            
            self.vBoxLayout.addWidget(widget)

    def showImage(self, index):
        byte_array = io.BytesIO()
        self.result[1][index].save(byte_array, format='PNG')
        qpixmap = QPixmap()
        qpixmap.loadFromData(byte_array.getvalue())
        self.imageLabel.setPixmap(qpixmap)
        self.overlayWidget.show()

    def hideImage(self):
        self.overlayWidget.hide()
    
    def addResultLabels(self):
        for i in range(len(self.result[0])):
            byte_array = io.BytesIO()
            self.result[1][i].save(byte_array, format='PNG')
            qpixmap = QPixmap()
            qpixmap.loadFromData(byte_array.getvalue())
            label = QLabel()
            label.setPixmap(qpixmap) 
            self.picFlowLayout.addWidget(label) 

class PredictResultInterface(FluentWindow):
    def __init__(self,result, parent=None):
        super().__init__(parent)
        self.result = result
        
        self.setWindowIcon(QIcon(':/gallery/images/logo_new.png'))
        self.setWindowTitle('预测结果')
        
        # self.splashScreen = SplashScreen(icon=self.windowIcon(), parent=self)
        # self.splashScreen.setIconSize(QSize(106, 106))
        # self.splashScreen.raise_()
        
        
        self.setObjectName('predictResultInterface')
        
        self.predictResultWidget=PredictResultWidget(result=self.result,parent=self)
        # self.predictResultWidget.setObjectName('predictResultWidget')
        # self.predictResultWidget = PredictResultWidget(result=self.result)
        
        self.addSubInterface(self.predictResultWidget,FIF.HOME ,'result')
        self.navigationInterface.hide()
        # self.splashScreen.finish()
        # self.updateFrameless()

        
# if __name__ == '__main__':
#     import sys
#     from PyQt5.QtWidgets import QApplication
#     app = QApplication(sys.argv)
#     result = PredictResultInterface("result")
#     result.show()
#     # card=ExampleCard2(title="title",text="text",icon=FIF.ADD)
#     # card.show()
#     sys.exit(app.exec_())
    
