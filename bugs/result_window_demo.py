from PyQt5 import QtWidgets, QtGui, QtCore

class MedicinalChemistryApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Set up the main layout
        self.main_layout = QtWidgets.QHBoxLayout(self)

        # Create and add the left panel layout
        self.left_panel = QtWidgets.QWidget()
        self.left_panel_layout = QtWidgets.QVBoxLayout(self.left_panel)

        # Create and add the header
        self.header_label = QtWidgets.QLabel("MEDICINAL CHEMISTRY")
        self.header_label.setAlignment(QtCore.Qt.AlignCenter)
        header_font = QtGui.QFont("Arial", 16, QtGui.QFont.Bold)
        self.header_label.setFont(header_font)
        self.header_label.setStyleSheet("background-color: #FF66B2; color: white;")
        self.left_panel_layout.addWidget(self.header_label)

        # Create and add the table widget
        self.table_widget = QtWidgets.QWidget()
        self.table_layout = QtWidgets.QGridLayout(self.table_widget)
        self.left_panel_layout.addWidget(self.table_widget)

        # Define table data
        self.dict = {
            'Carcinogenicity': 0.6230675578117371,
            'Ames Mutagenicity': 0.635084331035614,
            'Respiratory toxicity': 0.6175598502159119,
            'Eye irritation': 0.8082209825515747,
            'Eye corrosion': 0.7043989300727844,
            'Cardiotoxicity1': 0.14914588630199432,
            'Cardiotoxicity10': 0.6474915742874146,
            'Cardiotoxicity30': 0.8528580069541931,
            'Cardiotoxicity5': 0.4074292480945587,
            'CYP1A2': 0.3518486022949219,
            'CYP2C19': 0.2841881811618805,
            'CYP2C9': 0.3783678114414215,
            'CYP2D6': 0.18503925204277039,
            'CYP3A4': 0.19085778295993805,
            'NR-AR': 0.1093173548579216,
            'NR-AR-LBD': 0.009610038250684738,
            'NR-AhR': 0.049286000430583954,
            'NR-Aromatase': 0.016041185706853867,
            'NR-ER': 0.20859608054161072,
            'NR-ER-LBD': 0.04058457911014557,
            'NR-PPAR-gamma': 0.04711690917611122,
            'SR-ARE': 0.1094619631767273,
            'SR-ATAD5': 0.039146702736616135,
            'SR-HSE': 0.05485197901725769,
            'SR-MMP': 0.13969792425632477,
            'SR-p53': 0.013802326284348965
        }

        self.result = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Prepare table data by combining both dictionaries
        table_data = list(zip(self.dict.keys(), self.dict.values(), self.result))

        # Add table data to the layout
        for i, (label, value, result) in enumerate(table_data):
            color = "green" if result == 1 else "red"
            self.add_row_to_table(i, label, value, color)

        # Create and add the right panel for details
        self.right_panel = QtWidgets.QWidget()
        self.right_panel_layout = QtWidgets.QVBoxLayout(self.right_panel)
        self.details_label = QtWidgets.QLabel("Details View")
        self.details_label.setAlignment(QtCore.Qt.AlignCenter)
        self.details_label.setFont(header_font)
        self.details_label.setStyleSheet("background-color: #FF66B2; color: white;")
        self.right_panel_layout.addWidget(self.details_label)

        self.details_view = QtWidgets.QLabel("Select a row to see details.")
        self.details_view.setWordWrap(True)
        self.right_panel_layout.addWidget(self.details_view)

        # Add the panels to the main layout
        self.main_layout.addWidget(self.left_panel, 1)
        self.main_layout.addWidget(self.right_panel, 2)

    def add_row_to_table(self, index, label, value, color):
        # Create and add the metric label
        metric_label = QtWidgets.QLabel(label)
        self.table_layout.addWidget(metric_label, index, 0)

        # Create and add the value label
        value_label = QtWidgets.QLabel(str(value))
        self.table_layout.addWidget(value_label, index, 1)

        # Create and add the acceptance icon
        color_icon = QtGui.QPixmap(16, 16)
        color_icon.fill(QtGui.QColor(color))
        icon_label = QtWidgets.QLabel()
        icon_label.setPixmap(color_icon)
        self.table_layout.addWidget(icon_label, index, 2)

        # Create and add the "Details" button
        details_button = QtWidgets.QPushButton("Details")
        details_button.clicked.connect(lambda: self.show_details(label, value, color))
        self.table_layout.addWidget(details_button, index, 3)

    def show_details(self, label, value, color):
        status = "Accepted" if color == "green" else "Rejected"
        details_text = f"Metric: {label}\nValue: {value}\nStatus: {status}"
        self.details_view.setText(details_text)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    main_window = MedicinalChemistryApp()
    main_window.show()
    sys.exit(app.exec_())
