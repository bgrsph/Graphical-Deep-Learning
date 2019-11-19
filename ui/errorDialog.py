from PyQt5.QtWidgets import *


class ErrorDialog:

    def __init__(self, message="Something went wrong. Please use ErrorDialog class to enter a error message"):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle("Error")
        msg.exec_()