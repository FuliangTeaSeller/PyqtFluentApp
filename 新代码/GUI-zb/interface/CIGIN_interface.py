import sys, io, os

from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QLabel, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QIcon, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtWidgets import QFileDialog, QScrollArea, QWidget, QFrame, QPlainTextEdit
from PyQt5.QtWebEngineWidgets import QWebEngineView

from qfluentwidgets import FluentIcon as FIF


from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

IPythonConsole.ipython_useSVG = False  # 如果想展示PNG请设置为FALSE

from GUI.UI.func1_page import Ui_Form
from test_clone import delt_g_main, resource_path


class Func1Widget(QFrame, Ui_Form):
    def __init__(self, parent=None):
        super().__init__()
        self.ui = Ui_Form
        self.setupUi(self)
        self.setObjectName('Func1-Interface')
        self.pushButton.clicked.connect(self.Delt_g)
        self.pushButton.setIcon(FIF.CHECKBOX)
        self.lineEdit.setClearButtonEnabled(True)
        self.lineEdit_2.setClearButtonEnabled(True)
        # 将WebEngineView部件添加到布局中
        self.Jsme()
    def Delt_g(self, ):
        # 设置字体
        font = QFont()
        font.setPointSize(15)
        font.setBold(False)
        font.setFamily('微软雅黑')
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # 允许滚动区域自适应大小
        # 设置弹窗
        self.msg_box1 = QWidget()
        ICON = QIcon(resource_path("title.png"))
        self.msg_box1.setWindowIcon(ICON)
        self.msg_box1.setWindowTitle("result")
        self.msg_box1.resize(380, 500)
        self.frameg = QFrame(self)
        self.layout1 = QGridLayout(self.frameg)
        # 读取溶质，溶剂smiles
        solute = self.lineEdit_2.text()
        solvent = self.lineEdit.text()

        '''if any(c.isdigit() for c in solute) or any(c.isdigit() for c in solvent):
            QMessageBox.warning(self, "提示", "请输入正确的smiles")'''
        if not solute or not solvent:
            QMessageBox.warning(self, "提示", "输入为空")
        else:
            # 输入模型，返回预测值
            delt_g = delt_g_main(solute, solvent)
            output, map = delt_g[0], delt_g[1]
            # 获得分子二维结构图
            mol1 = Chem.MolFromSmiles(solute)
            mol2 = Chem.MolFromSmiles(solvent)

            photo1 = Draw.MolToImage(mol1)
            photo2 = Draw.MolToImage(mol2)

            # 转换为数据流
            byte_array1 = io.BytesIO()
            photo1.save(byte_array1, format='PNG')
            byte_array1.seek(0)

            byte_array2 = io.BytesIO()
            photo2.save(byte_array2, format='PNG')
            byte_array2.seek(0)

            q_img1 = QImage.fromData(byte_array1.read())
            pixmap1 = QPixmap.fromImage(q_img1)
            map_label1 = QLabel(self)
            name_label1 = QLabel(self)
            name_label1.setText('溶质分子的二维结构图：')
            name_label1.setFont(font)
            map_label1.setPixmap(pixmap1)

            q_img2 = QImage.fromData(byte_array2.read())
            pixmap2 = QPixmap.fromImage(q_img2)
            map_label2 = QLabel(self)
            name_label2 = QLabel(self)
            name_label2.setText('溶剂分子的二维结构图：')
            name_label2.setFont(font)

            map_label2.setPixmap(pixmap2)
            # 返回结果弹窗
            result_label = QLabel(self)
            result_label.setText(('溶剂化自由能为:{0} kcal/mol').format(output.item()))
            result_label.setFont(font)
            # 嵌套
            self.layout1.addWidget(name_label1, 0, 0)
            self.layout1.addWidget(name_label2, 5, 0)
            self.layout1.addWidget(map_label1, 1, 0, 3, 3)
            self.layout1.addWidget(map_label2, 6, 0, 3, 3)
            self.layout1.addWidget(result_label, 9, 0)

            scroll_area.setWidget(self.frameg)
            # 嵌套
            self.layout4 = QVBoxLayout(self)
            self.layout4.addWidget(scroll_area)
            self.msg_box1.setLayout(self.layout4)
            self.msg_box1.show()
    def Jsme(self):
        # 创建WebEngineView部件
        self.web_view = QWebEngineView(self)
        self.web_view.setMinimumSize(405, 350)
        # 加载JSME编辑器的HTML文件
        jsme_html = os.path.join(os.path.dirname(__file__), 'jsme.html')
        self.web_view.load(QUrl.fromLocalFile('E:\Python\pyqt5\JSME.html'))
        self.layout_tab2 = QVBoxLayout()
        self.layout_tab2.addWidget(self.web_view,alignment=Qt.AlignHCenter)
        self.tab_2.setLayout(self.layout_tab2)



