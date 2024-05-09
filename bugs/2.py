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
class Demo(QWidget):

    def __init__(self):
        super().__init__()
        layout = FlowLayout(self, needAni=True)  # 启用动画

        # 自定义动画参数
        # layout.setAnimation(250, QEasingCurve.OutQuad)

        layout.setContentsMargins(30, 30, 30, 30)
        layout.setVerticalSpacing(20)
        layout.setHorizontalSpacing(10)

        layout.addWidget(QPushButton('aiko'))
        layout.addWidget(QPushButton('刘静爱'))
        layout.addWidget(QPushButton('柳井爱子'))
        layout.addWidget(QPushButton('aiko 赛高'))
        layout.addWidget(QPushButton('aiko 太爱啦😘'))

        self.resize(250, 300)

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    demo = Demo()
    demo.show()
    sys.exit(app.exec_())