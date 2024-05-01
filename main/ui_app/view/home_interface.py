# coding:utf-8
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPixmap, QPainter, QColor, QBrush, QPainterPath, QLinearGradient, QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

from qfluentwidgets import ScrollArea, isDarkTheme, FluentIcon,BodyLabel,StrongBodyLabel
from .introduction_view import introductionView
from ..common.config import cfg, HELP_URL, REPO_URL, EXAMPLE_URL, FEEDBACK_URL
from ..common.icon import Icon, FluentIconBase
from ..components.link_card import LinkCardView
from ..components.sample_card import SampleCardView
from ..common.style_sheet import StyleSheet
from ..resource import resource


class BannerWidget(QWidget):
    """ Banner widget """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedHeight(336)

        self.vBoxLayout = QVBoxLayout(self)
        self.galleryLabel = QLabel('Detective Molecular Inspector 分子探长', self)
        self.banner = QPixmap(':/gallery/images/header1.png')
        self.linkCardView = LinkCardView(self)

        self.galleryLabel.setObjectName('galleryLabel')
        # self.galleryLabel.setFont(QFont('微软雅黑', 24, QFont.Bold))

        self.vBoxLayout.setSpacing(0)
        self.vBoxLayout.setContentsMargins(0, 20, 0, 0)
        self.vBoxLayout.addWidget(self.galleryLabel)
        self.vBoxLayout.addWidget(self.linkCardView, 1, Qt.AlignBottom)
        self.vBoxLayout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # self.linkCardView.addCard(
        #     ':/gallery/images/logo_old.png',
        #     self.tr('开始使用'),
        #     self.tr('开始使用智元'),
        #     HELP_URL
        # )
        
        self.linkCardView.addJumpCard(
            icon=':/gallery/images/logo_old.png',
            title=self.tr('开始使用'),
            content=self.tr('开始使用'),
            routeKey="predictInterface",
            index=0
        )

        self.linkCardView.addCard(
            FluentIcon.GITHUB,
            self.tr('GitHub仓库'),
            self.tr(
                'GitHub仓库'),
            REPO_URL
        )

        # self.linkCardView.addCard(
        #     FluentIcon.CODE,
        #     self.tr('Code samples'),
        #     self.tr(
        #         'Find samples that demonstrate specific tasks, features and APIs.'),
        #     EXAMPLE_URL
        # )

        # self.linkCardView.addCard(
        #     FluentIcon.FEEDBACK,
        #     self.tr('Send feedback'),
        #     self.tr('Help us improve PyQt-Fluent-Widgets by providing feedback.'),
        #     FEEDBACK_URL
        # )

    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHints(
            QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)
        w, h = self.width(), self.height()
        path.addRoundedRect(QRectF(0, 0, w, h), 10, 10)
        path.addRect(QRectF(0, h-50, 50, 50))
        path.addRect(QRectF(w-50, 0, 50, 50))
        path.addRect(QRectF(w-50, h-50, 50, 50))
        path = path.simplified()

        # init linear gradient effect
        gradient = QLinearGradient(0, 0, 0, h)

        # draw background color
        if not isDarkTheme():
            gradient.setColorAt(0, QColor(207, 216, 228, 255))
            gradient.setColorAt(1, QColor(207, 216, 228, 0))
        else:
            gradient.setColorAt(0, QColor(0, 0, 0, 255))
            gradient.setColorAt(1, QColor(0, 0, 0, 0))
            
        painter.fillPath(path, QBrush(gradient))

        # draw banner image
        pixmap = self.banner.scaled(
            self.size(), transformMode=Qt.SmoothTransformation)
        painter.fillPath(path, QBrush(pixmap))


class HomeInterface(ScrollArea):
    """ Home interface """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.banner = BannerWidget(self)
        self.view = QWidget(self)
        self.vBoxLayout = QVBoxLayout(self.view)

        self.__initWidget()
        self.initIntroduction()
        # self.loadSamples()

    def __initWidget(self):
        self.view.setObjectName('view')
        self.setObjectName('homeInterface')
        StyleSheet.HOME_INTERFACE.apply(self)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidget(self.view)
        self.setWidgetResizable(True)

        self.vBoxLayout.setContentsMargins(0, 0, 0, 36)
        self.vBoxLayout.setSpacing(40)
        self.vBoxLayout.addWidget(self.banner)
        self.vBoxLayout.setAlignment(Qt.AlignTop)

    def initIntroduction(self):
        view = introductionView(title=self.tr("Introduction"), parent=self.view)
        label=BodyLabel(self.tr(
            "IntelliMole, the cutting-edge molecular prediction software, is revolutionizing the field of scientific research. With its advanced algorithms and user-friendly interface, IntelliMole provides accurate and efficient molecular modeling capabilities, making it an indispensable tool for scientists and researchers. The software's ability to predict molecular structures and properties with high precision makes it a leader in the field of computational chemistry.\n\n尖端的分子预测软件IntelliMole正在彻底改变科学研究领域。凭借先进的算法和用户友好的界面，IntelliMole提供了准确和高效的分子建模能力，使其成为科学家和研究人员不可或缺的工具。该软件高精度预测分子结构和性质的能力使其在计算化学领域处于领先地位。"
            ))
        label.setObjectName('introductionLabel')
        label.setWordWrap(True)
        view.flowLayout.addWidget(label)
        self.vBoxLayout.addWidget(view)
        
        
    
    def loadSamples(self):
        """ load samples """
        # basic input samples
        basicInputView = SampleCardView(
            self.tr("Basic input samples"), self.view)
        basicInputView.addSampleCard(
            icon=":/gallery/images/controls/Button.png",
            title="Button",
            content=self.tr(
                "A control that responds to user input and emit clicked signal."),
            routeKey="basicInputInterface",
            index=0
        )
        basicInputView.addSampleCard(
            icon=":/gallery/images/controls/Checkbox.png",
            title="CheckBox",
            content=self.tr("A control that a user can select or clear."),
            routeKey="basicInputInterface",
            index=8
        )
        self.vBoxLayout.addWidget(basicInputView)

        # date time samples
        dateTimeView = SampleCardView(self.tr('Date & time samples'), self.view)
        dateTimeView.addSampleCard(
            icon=":/gallery/images/controls/CalendarDatePicker.png",
            title="CalendarPicker",
            content=self.tr("A control that lets a user pick a date value using a calendar."),
            routeKey="dateTimeInterface",
            index=0
        )
        dateTimeView.addSampleCard(
            icon=":/gallery/images/controls/DatePicker.png",
            title="DatePicker",
            content=self.tr("A control that lets a user pick a date value."),
            routeKey="dateTimeInterface",
            index=2
        )
        dateTimeView.addSampleCard(
            icon=":/gallery/images/controls/TimePicker.png",
            title="TimePicker",
            content=self.tr(
                "A configurable control that lets a user pick a time value."),
            routeKey="dateTimeInterface",
            index=4
        )
        self.vBoxLayout.addWidget(dateTimeView)

        
