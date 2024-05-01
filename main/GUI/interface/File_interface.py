import sys, io, os

from PyQt5.QtWidgets import QFileDialog, QPlainTextEdit, QWidget, QFrame, QMessageBox, QLabel, QVBoxLayout
from PyQt5.QtGui import QIcon, QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView

from qfluentwidgets import FluentIcon as FIF

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

from tox_21.prediction import resource_path
from tox_21.prediction import main as tox_21_main
from CIGIN.prediction import main as delt_g_main

import pandas as pd

from GUI.UI.func3_page import Ui_Form
class Func3Widget(QFrame, Ui_Form):
    def __init__(self, parent=None):
        super().__init__()
        self.ui = Ui_Form
        self.setupUi(self)
        self.setObjectName('Func3-Interface')
        self.pushButton.clicked.connect(self.Batch_import)
        self.pushButton.setIcon(FIF.FOLDER)
        self.pushButton_2.clicked.connect(self.Batch_predict)
        self.pushButton_2.setIcon(FIF.CHECKBOX)
        self.pushButton_3.clicked.connect(self.Batch_save)
        self.pushButton_3.setIcon(FIF.SAVE)

    def Batch_import(self):
        global content_pd
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        files = file_dialog.getOpenFileNames(self, "选择要导入的文件", '', 'CSV文件(*.csv);;TXT文件(*.txt)')
        font = QFont()
        font.setPointSize(12)
        font.setFamily('宋体')
        if files:
            for file_path in files[0]:
                file_ext = os.path.splitext(file_path)[1]
                if file_ext == ".csv":
                    content_pd = pd.read_csv(file_path,sep=';') # 读取CSV文件
                elif file_ext == ".txt":
                    content_pd = pd.read_csv(file_path, delimiter=";") # 读取文本文件

                content = content_pd.to_string(index=range(1, len(content_pd) + 1))
                self.plainTextEdit.setLineWrapMode(QPlainTextEdit.NoWrap)
                self.plainTextEdit.setPlainText(content)
                self.plainTextEdit.setFont(font)

    def Batch_predict(self):
        column_names = content_pd.columns.tolist()
        if self.radioButton_2.isChecked():
            if 'smiles' in column_names:
                labels = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                          'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
                for i in range(12):
                    content_pd[labels[i]] = None
                smiles_list = content_pd['smiles']

                self.worker_thread = ToxThread(smiles_list)
                self.worker_thread.update_progress.connect(self.update_progress_bar)
                self.worker_thread.result_ready.connect(self.handle_result)
                self.worker_thread.finished.connect(self.display_result)
                self.worker_thread.start()
            else:
                QMessageBox.warning(self, '警告', '文件中无可匹配列名:smiles')

        elif self.radioButton.isChecked():
            if 'SoluteSMILES' in column_names and 'SolventSMILES' in column_names:
                label = ['DeltaGsolv']
                solute = content_pd['SoluteSMILES']
                solvent = content_pd['SolventSMILES']
                content_pd[label] = None
                self.worker_thread = SolvThread(solute, solvent)
                self.worker_thread.update_progress.connect(self.update_progress_bar)
                self.worker_thread.result_ready.connect(self.handle_result)
                self.worker_thread.finished.connect(self.display_result)
                self.worker_thread.start()
            else:
                QMessageBox.warning(self, '警告', '文件中无可匹配列名:SoluteSMILES & SolventSMILES')

        else:
            QMessageBox.warning(self,'警告','请选择模型')
    def Batch_save(self):
        files = QFileDialog.getSaveFileName(self, 'Save File', '', 'CSV文件(*.csv);;TXT文件(*.txt)')
        if files[0]:
            file_path = files[0]
            content_pd.to_csv(file_path)
            QMessageBox.information(self,'提示','文件保存完毕')

    def update_progress_bar(self, progress):
        self.progressBar.setValue(progress)
        if progress == '100':
            self.worker_thread.terminate()

    # 显示处理结果
    def display_result(self, result):
        QMessageBox.information(self, '提示', '进程已完成')

    # 处理子线程传递的结果
    def handle_result(self, processed_data):
        Vscrollbar = self.plainTextEdit.verticalScrollBar()
        Vscrollbar_pos = Vscrollbar.value()
        Hscrollbar = self.plainTextEdit.horizontalScrollBar()
        Hscrollbar_pos = Hscrollbar.value()

        if self.radioButton_2.isChecked():
            for i in range(12):
                content_pd.loc[processed_data[1] - 1][i + 1] = processed_data[0][i]
            content = content_pd.to_string(index=range(1, len(content_pd) + 1))
            self.plainTextEdit.setPlainText(content)
        elif self.radioButton.isChecked():
            content_pd.loc[processed_data[1] - 1][-1] = processed_data[0]
            content = content_pd.to_string(index=range(1, len(content_pd) + 1))
            self.plainTextEdit.setPlainText(content)
        Vscrollbar.setValue(Vscrollbar_pos)
        Hscrollbar.setValue(Hscrollbar_pos)

# 子线程
class SolvThread(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    result_ready = pyqtSignal(list)

    def __init__(self, solute, solvent):
        super().__init__()
        self.solute = solute
        self.solvent = solvent

    def run(self):
        total = len(self.solute)
        for i in range(1, total + 1):
            # 模拟处理数据的操作，这里使用时间延迟来模拟处理耗时
            self.process_data(self.solute[i - 1], self.solvent[i - 1], i)

            # 发送更新进度信号
            progress = int(i / total * 100)
            self.update_progress.emit(progress)

        self.finished.emit("处理完成")

    def process_data(self, solute, solvent, index):
        # 实际的数据处理逻辑
        # 这里可以根据需要进行修改
        import time
        time.sleep(0.5)  # 模拟处理耗时
        delt_g = delt_g_main(solute, solvent)
        self.result_ready.emit([delt_g[0].item(), index])

class ToxThread(QThread):
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
        output, photo = tox_21_main(item, 'multitude')
        self.result_ready.emit([output, index])