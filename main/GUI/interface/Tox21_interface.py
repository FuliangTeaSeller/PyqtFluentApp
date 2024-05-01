import io, os

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QScrollArea, QWidget, QFrame, QMessageBox, QLabel, QVBoxLayout
from PyQt5.QtGui import QIcon, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView

from qfluentwidgets import FluentIcon as FIF

from rdkit import Chem
from rdkit.Chem import Draw
from tox_21.prediction import resource_path
from tox_21.prediction import main as tox_21_main

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from GUI.UI.func2_page import Ui_Form
from GUI.interface.tox_result_interface import Result as tox_result
class Func2Widget(QFrame, Ui_Form):
    def __init__(self, parent=None):
        super().__init__()
        self.ui = Ui_Form
        self.setupUi(self)
        self.setObjectName('Func2-Interface')
        self.lineEdit.setClearButtonEnabled(True)
        self.pushButton.clicked.connect(self.Tox21)
        self.pushButton.setIcon(FIF.CHECKBOX)
        self.Jsme()

    def Tox21(self):
        # 获取分子式
        molecule = self.lineEdit.text()
        '''if molecule:
            # 设置字体
            font = QFont()
            font.setPointSize(15)
            font.setBold(False)
            font.setFamily('宋体')

            font1 = QFont()
            font1.setPointSize(18)
            font1.setBold(True)
            font1.setFamily('宋体')
            # 创建滚动区域
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)  # 允许滚动区域自适应大小
            # 设置messagebox
            self.msg_box2 = QWidget()
            self.msg_box2.setWindowIcon(QIcon(resource_path("title.png")))
            self.msg_box2.resize(500, 500)
            self.msg_box2.setWindowTitle("result")
            self.framex = QFrame(self)
            self.layout3 = QVBoxLayout(self.framex)
            # 输入模型,返回预测值
            output, photo = tox_21_main(molecule, 'single')
            # 获取分子二维结构图
            mol = Chem.MolFromSmiles(molecule)
            photo3 = Draw.MolToImage(mol)
            byte_array3 = io.BytesIO()
            photo3.save(byte_array3, format='PNG')
            byte_array3.seek(0)

            q_img3 = QImage.fromData(byte_array3.read())
            pixmap3 = QPixmap.fromImage(q_img3)

            # 弹出结果
            map_label3 = QLabel(self)
            map_label3.setPixmap(pixmap3)
            map_label3.setAlignment(Qt.AlignHCenter)
            name_label = QLabel(self)
            name_label.setText('{}分子的二维结构图：'.format(molecule))
            name_label.setAlignment(Qt.AlignHCenter)
            name_label.setFont(font)
            self.layout3.addWidget(name_label)
            self.layout3.addWidget(map_label3)

            attn_label = QLabel(self)
            attn_label.setText('(0代表无该毒性，1代表有该毒性)')
            attn_label.setAlignment(Qt.AlignCenter)
            attn_label.setFont(font1)
            self.layout3.addWidget(attn_label)

            num = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12']

            labels = ['NR-AR(雌激素受体激活与抑制活性):',
                      'NR-AR-LBD(雌激素受体亚型的配体结合结构域激活与抑制活性):',
                      'NR-AhR(芳香烃受体激活活性):',
                      'NR-Aromatase(芳香化酶抑制活性):',
                      'NR-ER(雌激素受体亚型激活活性):',
                      'NR-ER-LBD(雌激素受体亚型的配体结合结构域激活与抑制活性):',
                      'NR-PPAR-gamma(过氧化物酶体增殖物激活受体-gamma激活与抑制活性):',
                      'SR-ARE(抗氧化反应元件激活活性):',
                      'SR-ATAD5(ATAD5 核酸交联修复相关的ATP酶活性起动剂激活活性):',
                      'SR-HSE(热休克响应元件激活活性):',
                      'SR-MMP(基质金属蛋白酶激活与抑制活性):',
                      'SR-p53(肿瘤蛋白 p53 激活与抑制活性):']

            for i in range(12):
                num[i] = QLabel(self)
                dd = QLabel(self)
                ss = QLabel(self)
                msg = '{}{}'.format(labels[i], str(output[i]))
                num[i].setText(msg)
                num[i].setAlignment(Qt.AlignCenter)  # Qt.AlignRight)
                num[i].setFont(font)
                self.layout3.addWidget(num[i])
                # 预测毒性基团
                if photo[i] != []:
                    byte_array = io.BytesIO()
                    photo[i].save(byte_array, format='PNG')
                    byte_array.seek(0)
                    q_img = QImage.fromData(byte_array.read())
                    pixmap = QPixmap.fromImage(q_img)
                    ss.setText('提供毒性的子结构为:')
                    ss.setAlignment(Qt.AlignHCenter)
                    dd.setPixmap(pixmap)
                    dd.setAlignment(Qt.AlignCenter)
                    self.layout3.addWidget(ss)
                    self.layout3.addWidget(dd)

            # 设置布局
            scroll_area.setWidget(self.framex)
            # 嵌套
            self.layout4 = QVBoxLayout(self)
            self.layout4.addWidget(scroll_area)
            self.msg_box2.setLayout(self.layout4)
            self.msg_box2.show()
        else:
            QMessageBox.warning(self, "提示", "输入为空")'''
        self.tox_result = tox_result(molecule)
        self.tox_result.show()
    def Jsme(self):
        # 创建WebEngineView部件
        self.web_view = QWebEngineView(self)
        self.web_view.setMinimumSize(405, 350)
        # 加载JSME编辑器的HTML文件
        jsme_html = os.path.join(os.path.dirname(__file__), 'jsme.html')
        self.web_view.load(QUrl.fromLocalFile(resource_path('JSME.html')))
        self.layout_tab2 = QVBoxLayout()
        self.layout_tab2.addWidget(self.web_view,alignment=Qt.AlignHCenter)
        self.tab_2.setLayout(self.layout_tab2)

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

