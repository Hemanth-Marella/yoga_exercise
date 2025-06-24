import sys
import mysql.connector
import requests

from PyQt5.QtWidgets import QApplication,QLineEdit,QHBoxLayout,QLabel,QMessageBox,QPushButton,QWidget,QSizePolicy,QMainWindow,QVBoxLayout,QComboBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor
from database import mydatabase
#import database 

class register(QMainWindow):
    
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Registration Form")
        self.setGeometry(300,200,800,800)
        self.setStyleSheet("""
                  background-color:#0D0D0D;
        """)

        self.central_widget=QWidget()
        self.central_widget.setObjectName("central_widget")
        self.setCentralWidget(self.central_widget)
        self.central_widget.setStyleSheet("""
                  background-color: #0D0D0D;                  
        """)

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        form_layout = QVBoxLayout()
        form_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        h_layout = QHBoxLayout()
        #h_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        r_label = QLabel("Registration")
        r_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        r_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        r_label.setStyleSheet("""
                                color:white;
                                font-size:20px;
                                font-weight:bold;
                                padding-top: 10px;
                                padding:10px;
                                margin-top:10px;
                                margin-bottom:15px;  
                                border:none; 
        """)

        self.f_line = QLineEdit()
        self.f_line.setPlaceholderText("First Name")
        self.f_line.setStyleSheet("""
                                color:black;
                                font-size:16px;
                                padding:10px;
                                margin-top:15px;
                                border-radius: 15px;
                                border: 2px solid #2B8A54;
                                background-color:white;
""")

        self.l_line = QLineEdit()
        self.l_line.setPlaceholderText("Last Name")
        self.l_line.setStyleSheet("""
                                color:black;
                                font-size:16px;
                                padding:10px;
                                margin-top:15px;
                                border-radius: 15px;
                                border: 2px solid #2B8A54;
                                background-color:white;
""")

        self.combo_box = QComboBox()
        self.combo_box.addItem("Gender")
        self.combo_box.addItem("Male")
        self.combo_box.addItem("Female")
        self.combo_box.addItem("Others")
        
        self.combo_box.setStyleSheet("""
                                padding:11px;
                                margin-top:15px;
                                color:black;
                                font-size:16px;
                                border-radius: 15px;
                                border: 2px solid #2B8A54;
                                background-color:white;
        """)

        self.a_line = QLineEdit()
        self.a_line.setPlaceholderText("Username")
        self.a_line.setStyleSheet("""
                                color:black;
                                font-size:16px;
                                padding:10px;
                                margin-top:15px;
                                border-radius: 15px;
                                border: 2px solid #2B8A54;
                                background-color:white;
""")

        self.h_line = QLineEdit()
        self.h_line.setPlaceholderText("Height")
        self.h_line.setStyleSheet("""
                                color:black;
                                font-size:16px;
                                padding:10px;
                                margin-top:15px;
                                border-radius: 15px;
                                border: 2px solid #2B8A54;
                                background-color:white;
""")

        self.w_line = QLineEdit()
        self.w_line.setPlaceholderText("Weight")
        self.w_line.setStyleSheet("""
                                color:black;
                                font-size:16px;
                                padding:10px;
                                margin-top:15px;
                                border-radius: 15px;
                                border: 2px solid #2B8A54;
                                background-color:white;
""")

        self.p_line = QLineEdit()
        self.p_line.setPlaceholderText("Phone Number")
        self.p_line.setStyleSheet("""
                                color:black;
                                font-size:16px;
                                padding:10px;
                                margin-top:15px;
                                border-radius: 15px;
                                border: 2px solid #2B8A54;
                                background-color:white;
""")

        self.m_line = QLineEdit()
        self.m_line.setPlaceholderText("Mail")
        self.m_line.setStyleSheet("""
                                color:black;
                                font-size:16px;
                                padding:10px;
                                margin-top:15px;
                                border-radius: 15px;
                                border: 2px solid #2B8A54;
                                background-color:white;
""")

        self.s_line = QLineEdit()
        self.s_line.setPlaceholderText("Set Password")
        self.s_line.setStyleSheet("""
                                color:black;
                                font-size:16px;
                                padding:10px;
                                margin-top:15px;
                                border-radius: 15px;
                                border: 2px solid #2B8A54;
                                background-color:white;
""")

        self.c_line = QLineEdit()
        self.c_line.setPlaceholderText("Confirm Password")
        self.c_line.setStyleSheet("""
                                color:black;
                                font-size:16px;
                                padding:10px;
                                margin-top:15px;
                                border-radius: 15px;
                                border: 2px solid #2B8A54;
                                background-color:white;
""")

#         self.list1=["f_line","l_name","a_line","h_line","w_line","p_line","m_line","s_line","c_line"]
#         self.place_holder=["First Name","Last Name","Address","Height","Weight","Phone Number","Mail","Set Password","Current Password"]

#         for lines in range(len(self.list1)):
#             self.list1[lines] = QLabel(self.list1[lines])
#             self.list1[lines].setPlaceholderText(self.place_holder[lines])
#             self.list1[lines].setStyleSheet("""
# #                                 color:black;
# #                                 font-size:16px;
# #                                 padding:10px;
# #                                 margin-top:15px;
# #                                 border-radius: 15px;
# #                                 border: 2px solid #2B8A54;
# # """)

        r_button=QPushButton("Register")
        r_button.clicked.connect(self.register_button_click)
        r_button.setStyleSheet("""
                                margin-top:20px;
                                margin-bottom:10px;
                                margin-left:190px;
                                margin-right:190px;
                                padding:10px;
                                border-radius:20px;
                                border-style:solid;
                                border-width:2px;
                                border-color:#2B8A54;
                                color:white;
                                background-color:#2B8A54;
                                font-size:16px;
                                font-weight:bold;

""")

        text_label = QLabel("Already have an account?")
        #text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_label.setStyleSheet("""
                  font-size:16px;
                  color:#2B8A54;
                  border:none;
""")
        
        go_login=QLabel("Go to Login")
        go_login.setAlignment(Qt.AlignmentFlag.AlignCenter)
        go_login.setCursor(Qt.CursorShape.PointingHandCursor)
        go_login.mousePressEvent = self.go_to_login
        go_login.setStyleSheet("""
                  font-size:16px;
                  color:#2B8A54;
                  border:none;
""")

        h1_layout=QHBoxLayout()
        h1_layout.setSpacing(10)
        h1_layout.addWidget(text_label)
        h1_layout.addWidget(go_login)
        h1_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        v_layout = QVBoxLayout()
        # v_layout.addWidget(r_label)
        v_layout.addWidget(self.f_line)
        v_layout.addWidget(self.combo_box)
        v_layout.addWidget(self.h_line)
        v_layout.addWidget(self.p_line)
        v_layout.addWidget(self.s_line)


        v1_layout = QVBoxLayout()
        v1_layout.addWidget(self.l_line)
        v1_layout.addWidget(self.a_line)
        v1_layout.addWidget(self.w_line)
        v1_layout.addWidget(self.m_line)
        v1_layout.addWidget(self.c_line)

        h_layout.addLayout(v_layout)
        h_layout.addLayout(v1_layout)

        form_layout.addWidget(r_label)
        form_layout.addLayout(h_layout)
        form_layout.addWidget(r_button)
        form_layout.addLayout(h1_layout)
       
        self.register_widget = QWidget()
        self.register_widget.setFixedSize(500,600)
        self.register_widget.setStyleSheet("""
            background-color: #0D0D0D;
            border-radius: 15px;
            border: 5px solid #2B8A54;

        """)


        self.register_widget.setLayout(form_layout)

        main_layout.addWidget(self.register_widget,alignment=Qt.AlignmentFlag.AlignCenter)

        self.central_widget.setLayout(main_layout)

        self.login = None

    def register_button_click(self):
        fname = self.f_line.text().strip()
        lname = self.l_line.text().strip()
        gender = self.combo_box.currentText()
        self.username = self.a_line.text().strip()
        height = self.h_line.text().strip()
        weight = self.w_line.text().strip()
        self.phone = self.p_line.text().strip()
        self.mail = self.m_line.text().strip()
        set_password = self.s_line.text()
        confirm_password = self.c_line.text()

        if not all([fname, lname, gender, self.username,height, weight, self.mail, set_password, confirm_password,self.phone]):
            QMessageBox.warning(self, "Input Error", "All fields are required!")
            return

        if set_password != confirm_password:
            QMessageBox.warning(self, "Password Error", "Passwords do not match!")
            return
        

        try:
            db =  mydatabase()
            rows = db.insert_user(fname, lname, height, weight, gender,self.mail,confirm_password ,self.username,self.phone)
            
            if rows > 0:
                QMessageBox.about(self, "Success", "Registration Successful!")
            else:

                # this is for phone number
                entered_phone = self.phone  
                phone_results = db.phone()

                for row in phone_results:
                    if str(row[0]) == str(entered_phone):
                        QMessageBox.information(self, "Phone Number Exists", "Phone number already exists")

                # this is for email id
                entered_mail = self.mail  
                email_results = db.email()

                for row in email_results:
                    if str(row[0]) == str(entered_mail):
                        QMessageBox.information(self, "EMail Exists", "EMail already exists")

                entered_user = self.username
                user_results = db.username()

                for row in user_results:
                    if str(row[0]) == str(entered_user):
                        QMessageBox.information(self, "UserName Exists", "User Name already exists")
                    
               # QMessageBox.critical(self, "Error", "Registration Failed!")
        
        # except Exception as e:
            
            #QMessageBox.critical(self, "Database Error", f"Error: {str(e)}")
        
        finally:
            db.close_connection()


    def go_to_login(self,event):
        
        if self.login is None:
            from loginform import login  
            self.login = login()

        self.login.show()
        self.hide()

    def button_click(self):
        set_password = self.s_line.text()
        confirm_password = self.c_line.text()

        if set_password == confirm_password:
            print("registration successfull")
        
        else:
            msg = QMessageBox(self)
            msg.setWindowTitle("message box")
            msg.setIcon(QMessageBox.Information)
            msg.setText("main message")
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg.exec()

app=QApplication(sys.argv)
window = register()
window.show()
app.exec()