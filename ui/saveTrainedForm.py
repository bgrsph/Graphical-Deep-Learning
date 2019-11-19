from PyQt5.QtWidgets import *
from tkinter import filedialog
import torch


class SaveTrainedModelForm(QDialog):
    def __init__(self, model, parent=None):
        super(SaveTrainedModelForm, self).__init__(parent)

        self.model = model
        layout = QFormLayout()

        self.enterModelNameLabel = QLabel("Enter the trained model name for saving (\"exampleModel.pth\"): ")
        self.enterModelNameLineEdit = QLineEdit()

        self.enterSavePathLabel = QLabel("Select a folder to save the model")
        self.enterSavePathButton = QPushButton("...")
        self.enterSavePathButton.clicked.connect(self.enterSavePathButtonPressed)

        self.allDoneButtonLabel = QLabel("Press the button after you entered the model name")
        self.allDoneButton = QPushButton("DONE")
        self.allDoneButton.clicked.connect(self.allDoneButtonButtonPressed)

        layout.addRow(self.enterModelNameLabel, self.enterModelNameLineEdit)
        layout.addRow(self.enterSavePathLabel, self.enterSavePathButton)
        layout.addRow(self.allDoneButtonLabel, self.allDoneButton)

        self.setLayout(layout)
        self.setWindowTitle("Choose a folder to save trained model (.pth file)")



    def enterSavePathButtonPressed(self):
        self.path = filedialog.askdirectory()
        self.enterSavePathButton.setText(self.path)


    def allDoneButtonButtonPressed(self):
        torch.save(self.model.state_dict(), self.path + "/" + self.enterModelNameLineEdit.text())


def startForm(form):
    app = QApplication([])
    form.show()
    app.exec_()

