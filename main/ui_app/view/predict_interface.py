# coding:utf-8
import io, os
from PyQt5.QtCore import Qt, QSize,QUrl,QTimer
from PyQt5.QtWidgets import QAction, QWidget, QVBoxLayout, QButtonGroup,QLabel,QStackedWidget,QHBoxLayout
from qfluentwidgets import (LineEdit,BodyLabel,PushButton,FlowLayout,VBoxLayout,Pivot,PlainTextEdit)
from PyQt5.QtGui import QPainter,QFont
from PyQt5.QtWebEngineWidgets import QWebEngineView

from qfluentwidgets import FluentIcon as FIF
#from qframelesswindow.webengine import FramelessWebEngineView


from .gallery_interface import GalleryInterface
from ..common.translator import Translator
from ..common.style_sheet import StyleSheet

import sys
sys.path.append("E:/Python/PyqtFluentApp/main/Predictor")
from Predictor.prediction import main as Predict
from GUI.interface.tox_result_interface import Result as tox_result

from tox_21.prediction import resource_path


class PredictInterface(GalleryInterface):

    def __init__(self, parent=None):
        translator = Translator()
        super().__init__(
            title=translator.predict,
            subtitle='预测界面',
            parent=parent
        )
        self.setObjectName('predictInterface')
        
        self.currentsmiles = ''
        
        self.pivot = Pivot(self)
        self.stackedWidget = QStackedWidget(self)
        

        # 设置定时器来定期从 JSME 编辑器获取 SMILES 字符串
        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.fetchSmiles)
        # self.timer.start(100)  # 每1秒获取一次

        # self.lineEdit = LineEdit()
        # self.lineEdit.setObjectName("lineEdit")
        # self.lineEdit.setPlaceholderText("请输入溶剂smiles")
        self.pushButton = PushButton()
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("预测")

        '''self.applybutton = PushButton('绘制完毕')
        self.applybutton.clicked.connect(self.fetchSmiles)'''
        
        self.Tab1_Jsme()
        self.Tab2_Input()
        
        self.vBoxLayout.addWidget(self.pivot, 0, Qt.AlignHCenter)
        self.vBoxLayout.addWidget(self.stackedWidget)
        # self.vBoxLayout.addWidget(self.lineEdit)
        # self.vBoxLayout.addWidget(self.smilelabel)
        #self.vBoxLayout.addWidget(self.applybutton)
        self.vBoxLayout.addWidget(self.pushButton)

        
        # 连接信号并初始化当前标签页
        self.stackedWidget.currentChanged.connect(self.onCurrentIndexChanged)
        self.stackedWidget.setCurrentWidget(self.tab_1)
        self.pivot.setCurrentItem('tab_1')


        # self.lineEdit.setClearButtonEnabled(True)
        self.pushButton.clicked.connect(self.Tox21)
        self.pushButton.setIcon(FIF.CHECKBOX)

        
    def Tox21(self):
        # 获取分子式
        current_index = self.stackedWidget.currentIndex()
        if current_index == 1:
            molecule = self.textEdit.toPlainText()
        elif current_index ==0:
            # 执行 JavaScript 代码获取 SMILES 字符串
            js_code = "jsmeApplet.smiles();"
            self.web_view.page().runJavaScript(js_code, self.setFromJSMESmiles)
            self.web_view.page().runJavaScript(js_code, self.setFromJSMESmiles)
            molecule = self.currentsmiles
        else :
            pass
        #self.tox_result = tox_result(molecule)
        #self.tox_result.show()
        print(molecule)
        result = Predict(molecule)
        
    def Tab1_Jsme(self):
        self.tab_1 = QWidget()
        self.addSubInterface(self.tab_1, 'tab_1', '使用JSME分子编辑器画出分子')
        # 创建WebEngineView部件
        '''self.smilelabel = BodyLabel("当前smiles:")
        self.smilelabel.setWordWrap(True)
        self.smilelabel.setMinimumSize(100,10)'''
        
        self.web_view = QWebEngineView(self)
        # self.web_view = QLabel("JSME编辑器")
        self.web_view.setMinimumSize(405, 350)
        
        '''self.applybutton = PushButton('绘制完毕')
        self.applybutton.clicked.connect(self.fetchSmiles)'''
        
        # 加载JSME编辑器的HTML文件
        # 获取当前文件的绝对路径
        current_dir = os.path.dirname(__file__)
        # 构建 JSME.html 的绝对路径
        jsme_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'JSME.html'))
        # 加载 HTML 文件
        self.web_view.load(QUrl.fromLocalFile(jsme_path))

        # self.layout_tab1 = QVBoxLayout()
        self.hlayout_tab1 = QHBoxLayout()
        #self.vlayout_tab1 = QVBoxLayout()
        #self.vlayout_tab1.addWidget(self.applybutton,alignment=Qt.AlignHCenter)
        #self.vlayout_tab1.addWidget(self.smilelabel,alignment=Qt.AlignHCenter)
        
        #self.vwidget_tab1 = QWidget()
        #self.vwidget_tab1.setLayout(self.vlayout_tab1)
        
        self.hlayout_tab1.addWidget(self.web_view,alignment=Qt.AlignHCenter)
        #self.hlayout_tab1.addWidget(self.vwidget_tab1,alignment=Qt.AlignHCenter)
        #self.hlayout_tab1.addWidget(self.applybutton,alignment=Qt.AlignHCenter)
        
        self.tab_1.setLayout(self.hlayout_tab1)
        
        # self.repaint()
        # StyleSheet.GALLERY_INTERFACE.apply(self)

    def Tab2_Input(self):
        self.tab_2 = QWidget()
        self.layout_tab2 = QVBoxLayout()
        self.textEdit = PlainTextEdit()
        self.textEdit.textChanged.connect(self.setFromTextEditSmiles)
        self.layout_tab2.addWidget(self.textEdit,alignment=Qt.AlignHCenter)
        self.tab_2.setLayout(self.layout_tab2)
        #self.textEdit.setPlainText("在此直接输入smiles")
        self.textEdit.setPlaceholderText("在此直接输入smiles")
        self.addSubInterface(self.tab_2, 'tab_2', '直接输入smiles')


    def plot_spider(self):
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(properties), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        # 绘制蛛网图
        ax = self.figure.add_subplot(111, polar=True)
        ax.plot(angles, molecule_data + molecule_data[:1], 'o-')
        ax.fill(angles, molecule_data + molecule_data[:1], alpha=0.25)

        # 设置刻度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(properties)

        # 设置图表标题
        ax.set_title('Molecule Properties')

        # 调整刻度范围
        ax.set_ylim(0, 1)

        # 显示蛛网图
        self.canvas.draw()
        


    def addSubInterface(self, widget: QWidget, objectName: str, text: str):
        widget.setObjectName(objectName)
        # widget.setAlignment(Qt.AlignCenter)
        self.stackedWidget.addWidget(widget)

        # 使用全局唯一的 objectName 作为路由键
        self.pivot.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget)
        )

    def onCurrentIndexChanged(self, index):
        widget = self.stackedWidget.widget(index)
        self.pivot.setCurrentItem(widget.objectName())
        
    def fetchSmiles(self):
        # 执行 JavaScript 代码获取 SMILES 字符串
        js_code = "jsmeApplet.smiles();"
        self.web_view.page().runJavaScript(js_code, self.setFromJSMESmiles)
        #self.smilelabel.setText(self.currentsmiles)

    def updateSmiles(self, smiles):
        # self.textEdit.setPlainText(self.currentsmiles)
        self.smilelabel.setText(self.currentsmiles)

    def setFromJSMESmiles(self, smiles):
        self.currentsmiles = smiles
        print("Current SMILES:", self.currentsmiles)
        
    def setFromTextEditSmiles(self):
        self.currentSmiles = self.textEdit.toPlainText()
        print("Updated SMILES:", self.currentSmiles)  # 可选：打印当前 SMILES 以验证