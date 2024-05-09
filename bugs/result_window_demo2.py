import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGridLayout, QLineEdit, QFrame
)
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt

class MoleculePredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Molecule Prediction Software")
        self.setGeometry(100, 100, 1200, 800)

        main_layout = QHBoxLayout()
        sidebar = self.create_sidebar()
        main_content = self.create_main_content()

        main_layout.addWidget(sidebar)
        main_layout.addWidget(main_content)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_sidebar(self):
        sidebar_frame = QFrame()
        sidebar_frame.setFixedWidth(250)
        sidebar_frame.setStyleSheet("background-color: #444;")
        sidebar_layout = QVBoxLayout()

        logo_label = QLabel()
        logo_pixmap = QPixmap(50, 50)
        logo_pixmap.fill(Qt.transparent)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignCenter)

        home_button = QPushButton("Home")
        home_button.setIcon(QIcon.fromTheme("go-home"))
        home_button.setStyleSheet(self.button_style())

        predict_button = QPushButton("Prediction")
        predict_button.setIcon(QIcon.fromTheme("system-search"))
        predict_button.setStyleSheet(self.button_style())

        sidebar_layout.addWidget(logo_label)
        sidebar_layout.addWidget(home_button)
        sidebar_layout.addWidget(predict_button)
        sidebar_layout.addStretch()
        sidebar_frame.setLayout(sidebar_layout)

        return sidebar_frame

    def create_main_content(self):
        main_content_frame = QFrame()
        main_content_layout = QVBoxLayout()
        
        search_bar_layout = QHBoxLayout()
        search_input = QLineEdit()
        search_input.setPlaceholderText("Search...")
        search_button = QPushButton("Search")
        search_bar_layout.addWidget(search_input)
        search_bar_layout.addWidget(search_button)

        main_content_layout.addLayout(search_bar_layout)

        prediction_results_layout = QGridLayout()
        prediction_results_layout.setSpacing(10)

        cards_info = [
            ("Molecular Structure", "path/to/icon1.png"),
            ("Molecular SMILES", "path/to/icon2.png"),
            ("Chemical & Physical Information", "path/to/icon3.png"),
            ("Absorption", "path/to/icon4.png"),
            ("Distribution", "path/to/icon5.png"),
            ("Metabolism", "path/to/icon6.png"),
            ("Excretion", "path/to/icon7.png"),
            ("Toxicity", "path/to/icon8.png")
        ]

        for idx, (text, icon_path) in enumerate(cards_info):
            card_frame = QFrame()
            card_frame.setStyleSheet(self.card_style())
            card_layout = QVBoxLayout()
            card_icon = QLabel()
            card_pixmap = QPixmap(icon_path)
            card_icon.setPixmap(card_pixmap.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            card_icon.setAlignment(Qt.AlignCenter)
            card_title = QLabel(text)
            card_title.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(card_icon)
            card_layout.addWidget(card_title)
            card_frame.setLayout(card_layout)
            prediction_results_layout.addWidget(card_frame, idx // 2, idx % 2)

        main_content_layout.addLayout(prediction_results_layout)
        main_content_frame.setLayout(main_content_layout)

        return main_content_frame

    @staticmethod
    def button_style():
        return """
            QPushButton {
                background-color: #666;
                color: white;
                padding: 10px;
                border: none;
                text-align: left;
                padding-left: 20px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #888;
            }
        """

    @staticmethod
    def card_style():
        return """
            QFrame {
                background-color: #eee;
                border-radius: 10px;
                padding: 10px;
            }
        """

def main():
    app = QApplication(sys.argv)
    window = MoleculePredictionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
