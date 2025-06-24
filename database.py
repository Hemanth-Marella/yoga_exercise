import mysql.connector
from PyQt5.QtWidgets import QMessageBox

class mydatabase:
    
    def __init__(self):
         self.initialize()

    def initialize(self):

        try:
        
            self.mydb = mysql.connector.connect(host="localhost",user="root", password="Hemu@1234", database="final_year_project" )
            
            self.mycursor = self.mydb.cursor() 

        except mysql.connector.Error as err:
                QMessageBox.about(self,"connection","not connection successfully")

    def insert_user(self, fname, lname, height, weight, gender, mail, password,username,phone):

        try:
            query = """INSERT INTO Registration (F_name, L_name, Height, Weight, Gender, Mail, password,Username,phonenumber)
                       VALUES (%s, %s, %s, %s, %s, %s, %s,%s, %s)"""
            values = (fname, lname, height, weight, gender, mail, password,username,phone)
            self.mycursor.execute(query, values)
            self.mydb.commit()

            return self.mycursor.rowcount
        
        except mysql.connector.Error as err:
            print(f"Error inserting data: {err}")
            return 0
        
    def phone(self):
         
        query = "SELECT PhoneNumber FROM registration"
        self.mycursor.execute(query)
        results = self.mycursor.fetchall()

        return results
    
    def email(self):
         
        query = "SELECT mail FROM registration"
        self.mycursor.execute(query)
        results = self.mycursor.fetchall()

        return results
    
    def username(self):

        query = "SELECT Username FROM registration"
        self.mycursor.execute(query)
        results = self.mycursor.fetchall()

        return results

    def get_username(self):

        query = "select Username from registration"
        self.mycursor.execute(query)
        results = self.mycursor.fetchall()

        return results
    
    def get_password(self):

        query = "select password from registration"
        self.mycursor.execute(query)
        results = self.mycursor.fetchall()

        return results
        
    def close_connection(self):
        self.mycursor.close()
        self.mydb.close()
