
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QMainWindow
from PyQt5.QtCore import Qt

class Main(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Surya Namaskar")
        self.setGeometry(400, 400, 600, 600)

        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #78d9ff;")
        
        label = QLabel("Surya Namaskar is completed")
        label.setStyleSheet("font-size: 40px; color: black;")

        layout = QHBoxLayout()
        layout.addWidget(label, alignment=Qt.AlignCenter)
        central_widget.setLayout(layout)

        self.setCentralWidget(central_widget)


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = Main()
#     window.show()
#     sys.exit(app.exec_())
