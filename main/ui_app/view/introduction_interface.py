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

        for title, text in contents:
            view = introductionView(title=self.tr(title), parent=self.view) 
            label = BodyLabel(self.tr(text)) 
            label.setObjectName('introductionLabel')
            label.setWordWrap(True)
            view.flowLayout.addWidget(label) 
            self.vBoxLayout.addWidget(view)
        

    
        

    
