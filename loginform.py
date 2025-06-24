import sys
from database import mydatabase
import requests

from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox, QVBoxLayout, QLabel, QPushButton, QLineEdit, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor
from main import main



class login(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Login Page")
        self.setGeometry(300, 200, 800, 800)
        self.setStyleSheet("""
            background-color:#0D0D0D;
            color : white;
        """)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.box = QWidget()
        self.box.setFixedSize(650, 450)
        self.box.setStyleSheet("""
            background-color: #0D0D0D;
            border-radius :20px;
            border-color:#2B8A54;
            border-style:solid;
            border-width:5px;
        """)

        v_layout = QVBoxLayout()
        v_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)


        login_label = QLabel("Login")
        login_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        login_label.setStyleSheet("""
            padding-left:10px;
            padding-bottom:5px;
            margin-bottom:5px;
            color : #2B8A54;
            font-family : "Arial";
            font-weight : bold;
            font-size : 30px;
            margin-right:20px;
            border: none;  /* Ensure no border */
        """)

        self.user_line = QLineEdit()
        self.user_line.setPlaceholderText("Enter username or Email") 
        self.user_line.setStyleSheet("""
            border-radius: 20px;
            border: 2px solid white;
            font-size: 16px;
            margin-top: 5px;
            margin-bottom: 5px;
            font-family : "Arial";
            margin-left:20px;
            margin-right:20px;
            padding: 10px;
            background-color: white;
            color: black;
        """)

        self.pass_line = QLineEdit()
        self.pass_line.setPlaceholderText("Enter password")
        self.pass_line.setStyleSheet("""
            border-radius: 20px;
            border: 2px solid white;
            font-size: 16px;
            margin-top: 5px;
            margin-bottom: 5px;
            font-family : "Arial";
            margin-left:20px;
            margin-right:20px;
            padding: 10px;
            background-color: white;
            color: black;
        """)

        button1 = QPushButton("Login")
        button1.clicked.connect(self.login)
        button1.setStyleSheet("""
            font-family : "Arial";
            font-size:22px;
            color:white;
            padding:6px;
            margin-top: 10px;
            background-color:#2B8A54;
            border-radius: 20px;
            border-style:solid;
            border-color:#2B8A54;
            border-width:2px;
            margin-left:200px;
            margin-right:200px;
        """)

        # self.home = QPushButton("Home")
#         self.home.clicked.connect(self.home_button)
#         self.home.setStyleSheet("""
#             font-family : "Arial";
#             font-size:22px;
#             color:white;
#             padding:6px;
#             margin-top: 10px;
#             background-color:#2B8A54;
#             border-radius: 20px;
#             border-style:solid;
#             border-color:#2B8A54;
#             border-width:2px;
#             margin-left:200px;
#             margin-right:200px;           
# """)

        f_label = QLabel("Forgotten Password?")
        f_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f_label.setCursor(QCursor(Qt.PointingHandCursor))
        #f_label.mousePressEvent = self.forgot_password_clicked
        f_label.setStyleSheet("""
            color:white;
            font-size:18px;
            font-weight:bold;
            padding-top: 10px;
            margin-top:10px;
            margin-bottom:10px;
            border: none;  
        """)

        r_label = QLabel("Click here for Registration?")
        r_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        r_label.setCursor(QCursor(Qt.PointingHandCursor))
        r_label.mousePressEvent = self.register_form
        r_label.setStyleSheet("""
            color:white;
            font-size:18px;
            font-weight:bold;
            padding-top:5px;
            padding-bottom:10px;
            margin-bottom:15px;
            margin-top:5px;
            border: none;  /* Remove border */
        """)

        v_layout.addWidget(login_label)
        v_layout.addWidget(self.user_line)
        v_layout.addWidget(self.pass_line)
        v_layout.addWidget(button1)
        # v_layout.addWidget(self.home)
        v_layout.addWidget(f_label)
        v_layout.addWidget(r_label)

        self.box.setLayout(v_layout)
        main_layout.addWidget(self.box, alignment=Qt.AlignmentFlag.AlignCenter)
        self.central_widget.setLayout(main_layout)

        # self.mainconnect = mainwindow()
        # self.mainconnect.setVisible(False)

        # self.video_cap = VideoWidget()
        # self.video_cap.setVisible(False)

        # self.hand_cap = handtrack()
        
        
        self.check_username = None
        self.check_password = None
        self.register = None

    def login(self):
        
        db = mydatabase()
        username = self.user_line.text()
        password = self.pass_line.text()
        gmail = self.user_line.text()

        
        get_username_results = db.get_username()
        get_password_results = db.get_password()
        get_mail_results = db.email()

        #check user is in the database or not
        for row in get_username_results:
            db_username = str(row[0])
            if db_username == username:

                self.check_username = db_username

            # else:
            #     QMessageBox(self,"Wrong","Entered wrong username")

        #check password is in the database or not
        for row in get_password_results:
            db_password = str(row[0])

            if db_password == password:

                self.check_password = db_password

            # else:
            #     QMessageBox(self,"Wrong","Entered wrong Password")


        #check email is in the database or not
        for row in get_mail_results:
            db_mail = str(row[0])

            if db_mail == gmail:

                self.check_username = db_mail

            # else:
            #     QMessageBox(self,"Wrong","Entered wrong EMail")

        if self.check_username  and self.check_password:
            self.hide()
            main()        

        else:
            QMessageBox.about(self,"wrong login details","wrong login details")

        return True

    # def home_button(self):
    #     username = self.user_line.text()
    #     password = self.pass_line.text()
    #     if username == "Hemanth" and password == "hemanth":
    #         self.mainconnect.setVisible(True)

    def forgot_password_clicked(self, event):
        QMessageBox.information(self, "Forgot Password", "Password recovery feature coming soon!")

    def register_form(self,event):
        if self.register is None:
            from register import register
            self.register = register()
        self.register.show()
        self.hide()


app = QApplication(sys.argv)
window = login()
window.show()
app.exec()
