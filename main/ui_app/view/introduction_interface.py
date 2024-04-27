# coding:utf-8
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QAction, QWidget, QVBoxLayout, QButtonGroup,QScrollArea,QLabel
from qfluentwidgets import (LineEdit,BodyLabel,PushButton,FlowLayout,VBoxLayout)
from PyQt5.QtGui import QPainter

from .gallery_interface import GalleryInterface
from .introduction_view import introductionView
from ..common.translator import Translator

class IntroductionInterface(GalleryInterface):
    def __init__(self, parent=None):
        translator = Translator()
        super().__init__(
            title='介绍',
            subtitle='介绍界面',
            parent=parent
        )
        self.setObjectName('introductionInterface')
        
        contents=[("模型介绍：",
                 "[这里添加模型介绍的占位文本]"),
                ("评价指标介绍：",
                 "[这里添加模型介绍的占位文本]"),
                ("和基线比较情况：",
                 "[这里添加模型介绍的占位文本]"),
                ("数据集信息：",
                 "[这里添加模型介绍的占位文本]"),
                ("文献引用信息：",
                 "[这里添加模型介绍的占位文本][这里添加模型介绍的占位文本][这里添加模型介绍的占位文本][这里添加模型介绍的占位文本][这里添加模型介绍的占位文本][这里添加模型介绍的占位文本][这里添加模型介绍的占位文本][这里添加模型介绍的占位文本][这里添加模型介绍的占位文本][这里添加模型介绍的占位文本][这里添加模型介绍的占位文本][这里添加模型介绍的占位文本][这里添加模型介绍的占位文本]"),
               ]
        # views=list(introductionView)
        # labels=list(BodyLabel)
        for title, text in contents:
            view = introductionView(title=self.tr(title), parent=self.view) 
            label = BodyLabel(self.tr(text)) 
            label.setObjectName('introductionLabel')
            label.setWordWrap(True)
            view.flowLayout.addWidget(label) 
            self.vBoxLayout.addWidget(view)
        # view = introductionView(title=self.tr("模型介绍："), parent=self.view)
        # label=BodyLabel(self.tr(
        #     "模型介绍：[这里添加模型介绍的占位文本]"
        #     ))
        # label.setObjectName('introductionLabel')
        # label.setWordWrap(True)
        # view.flowLayout.addWidget(label)
        # self.vBoxLayout.addWidget(view)

        # 创建各种 QLabel 来显示信息
        # model_intro_label = BodyLabel("模型介绍：[这里添加模型介绍的占位文本]")
        # evaluation_metrics_label = BodyLabel("评价指标介绍：[这里添加评价指标介绍的占位文本]")
        # baseline_comparison_label = BodyLabel("和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]和基线比较情况：[这里添加基线比较的占位文本]位文本]")
        # dataset_info_label = BodyLabel("数据集信息：[这里添加数据集信息的占位文本]")
        # citation_info_label = BodyLabel("文献引用信息：[这里添加文献引用信息的占位文本]")

        # # 设置标签的自动换行
        # model_intro_label.setWordWrap(True)
        # evaluation_metrics_label.setWordWrap(True)
        # baseline_comparison_label.setWordWrap(True)
        # dataset_info_label.setWordWrap(True)
        # citation_info_label.setWordWrap(True)

        # # 将标签添加到布局中
        # self.vBoxLayout.addWidget(model_intro_label)
        # self.vBoxLayout.addWidget(evaluation_metrics_label)
        # self.vBoxLayout.addWidget(baseline_comparison_label)
        # self.vBoxLayout.addWidget(dataset_info_label)
        # self.vBoxLayout.addWidget(citation_info_label)

        # container.setLayout(layout)

        # 创建一个 QScrollArea 并设置其滚动的 widget
        # scroll_area = QScrollArea(self)
        # scroll_area.setWidgetResizable(True)  # 允许滚动区域自适应大小
        # scroll_area.setWidget(container)  # 设置滚动区的内容为 container

        # scroll_area.setLayout(layout)
        # 主布局
        # main_layout = QVBoxLayout()
        # main_layout.addWidget(scroll_area)
        # self.setLayout(main_layout)

    
        

    
