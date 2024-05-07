# coding:utf-8
import io, os
from PyQt5.QtCore import Qt, QSize,QUrl,QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QPlainTextEdit, QWidget, QHBoxLayout ,QButtonGroup,QLabel,QStackedWidget,QFileDialog
from qfluentwidgets import (ProgressBar,PushButton,FlowLayout,VBoxLayout,MessageBox,PlainTextEdit,FolderListDialog)
from PyQt5.QtGui import QPainter,QFont
from PyQt5.QtWebEngineWidgets import QWebEngineView

from qfluentwidgets import FluentIcon as FIF
#from qframelesswindow.webengine import FramelessWebEngineView

import pandas as pd
from .gallery_interface import GalleryInterface
from ..common.translator import Translator
from ..common.style_sheet import StyleSheet
import sys
sys.path.append("E:/Python/PyqtFluentApp/main/Predictor")
from Predictor.prediction import main as Predict
class BatchPredictInterface(GalleryInterface):

    def __init__(self, parent=None):
        translator = Translator()
        super().__init__(
            title=translator.batchpredict,
            subtitle='批量处理界面',
            parent=parent
        )
        self.setObjectName('BatchPredictInterface')

        self.content_tab = QWidget()
        self.hlayout_pred = QHBoxLayout()

        self.plainTextEdit = PlainTextEdit()
        self.plainTextEdit.setObjectName("content")
        self.plainTextEdit.setReadOnly(True)


        self.impButton = PushButton()
        self.impButton.setObjectName("impButton")
        self.impButton.setText("导入")
        self.impButton.setIcon(FIF.FOLDER)

        self.saveButton = PushButton()
        self.saveButton.setObjectName("saveButton")
        self.saveButton.setText("保存")
        self.saveButton.setIcon(FIF.SAVE)

        self.beginButton = PushButton()
        self.beginButton.setObjectName("beginButton")
        self.beginButton.setText("开始")
        self.beginButton.setIcon(FIF.ZOOM)
        self.beginButton.clicked.connect(self.Batch_predict)

        self.progressbar = ProgressBar()
        self.progressbar.setObjectName('progressBar')

        self.impButton.clicked.connect(self.ImportFile)
        self.saveButton.clicked.connect(self.SaveFile)

        self.hlayout_pred.addWidget(self.impButton)
        self.hlayout_pred.addWidget(self.saveButton)

        self.content_tab.setLayout(self.hlayout_pred)

        self.vBoxLayout.addWidget(self.content_tab)
        self.vBoxLayout.addWidget(self.plainTextEdit)
        self.vBoxLayout.addWidget(self.progressbar)
        self.vBoxLayout.addWidget(self.beginButton)
        #self.vBoxLayout.addWidget(self.saveButton)


        self.content_pd = ''
    def ImportFile(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        files = file_dialog.getOpenFileNames(self, "选择要导入的文件", '', 'CSV文件(*.csv);;TXT文件(*.txt)')

        if files:
            for file_path in files[0]:
                file_ext = os.path.splitext(file_path)[1]
                if file_ext == ".csv":
                    self.content_pd = pd.read_csv(file_path,sep=';') # 读取CSV文件
                elif file_ext == ".txt":
                    self.content_pd = pd.read_csv(file_path, delimiter=";") # 读取文本文件
                font = QFont()
                font.setPointSize(12)
                font.setFamily('宋体')
                content = self.content_pd.to_string(index=range(1, len(self.content_pd) + 1))
                self.plainTextEdit.setLineWrapMode(QPlainTextEdit.NoWrap)
                self.plainTextEdit.setPlainText(content)
                self.plainTextEdit.setFont(font)

    def SaveFile(self):
        files = QFileDialog.getSaveFileName(self, 'Save File', '', 'CSV文件(*.csv);;TXT文件(*.txt)')
        if files[0]:
            file_path = files[0]
            self.content_pd.to_csv(file_path)
            w = MessageBox('提示', '文件保存完毕',self)

    def Batch_predict(self):
        self.column_names = self.content_pd.columns.tolist()
        if 'smiles' in self.column_names:
            self.labels = ['Carcinogenicity',
                    'Ames Mutagenicity',
                    'Respiratory toxicity',
                    'Eye irritation',
                    'Eye corrosion',
                    'Cardiotoxicity1',
                    'Cardiotoxicity10',
                    'Cardiotoxicity30',
                    'Cardiotoxicity5',
                    'CYP1A2',
                    'CYP2C19',
                    'CYP2C9',
                    'CYP2D6',
                    'CYP3A4',
                    'NR-AR',
                    'NR-AR-LBD',
                    'NR-AhR',
                    'NR-Aromatase',
                    'NR-ER',
                    'NR-ER-LBD',
                    'NR-PPAR-gamma',
                    'SR-ARE',
                    'SR-ATAD5',
                    'SR-HSE',
                    'SR-MMP',
                    'SR-p53']
            for i in range(len(self.labels)):
                self.content_pd[self.labels[i]] = None
            self.smiles_list = self.content_pd['smiles']
            self.worker_thread = predThread(self.smiles_list)
            self.worker_thread.update_progress.connect(self.update_progress_bar)
            self.worker_thread.result_ready.connect(self.handle_result)
            self.worker_thread.finished.connect(self.display_result)
            self.worker_thread.start()
        else:
            y = MessageBox('警告', '文件中无可匹配列名:smiles', self)

    def update_progress_bar(self, progress):
        self.progressbar.setValue(progress)
        if progress == '100':
            self.worker_thread.terminate()
    def display_result(self, result):
        x = MessageBox('提示', '进程已完成', self)

    def handle_result(self, processed_data):
        Vscrollbar = self.plainTextEdit.verticalScrollBar()
        Vscrollbar_pos = Vscrollbar.value()
        Hscrollbar = self.plainTextEdit.horizontalScrollBar()
        Hscrollbar_pos = Hscrollbar.value()

        for i in range(len(self.labels)):
            self.content_pd.loc[processed_data[1] - 1][i + 1] = processed_data[0][i]
        content = self.content_pd.to_string(index=range(1, len(self.content_pd) + 1))
        self.plainTextEdit.setPlainText(content)
        Vscrollbar.setValue(Vscrollbar_pos)
        Hscrollbar.setValue(Hscrollbar_pos)

class predThread(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    result_ready = pyqtSignal(list)

    def __init__(self, data):
        super().__init__()
        self.data = data

    def run(self):
        total = len(self.data)
        for i, item in enumerate(self.data, 1):
            # 模拟处理数据的操作，这里使用时间延迟来模拟处理耗时
            self.process_data(item, i)

            # 发送更新进度信号
            progress = int(i / total * 100)
            self.update_progress.emit(progress)

        self.finished.emit("处理完成")

    def process_data(self, item, index):
        # 实际的数据处理逻辑
        # 这里可以根据需要进行修改
        import time
        time.sleep(0.5)  # 模拟处理耗时
        output = Predict(item)
        self.result_ready.emit([output, index])


