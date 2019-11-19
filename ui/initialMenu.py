from ui.trainForm import *
from ui.testForm import *


class InitialMenu(QDialog):
    def __init__(self, parent=None):
        super(InitialMenu, self).__init__(parent)
        self.setWindowTitle("Welcome to the automized deep learning tool.")
        self.trainButton = QPushButton("TRAIN")
        self.trainButton.clicked.connect(self.trainButtonPressed)
        self.testButton = QPushButton("TEST WITH PRE-TRAINED DATA")
        self.testButton.clicked.connect(self.testButtonPressed)


        layout = QFormLayout()

        layout.addRow("------------------------------------>", self.trainButton)
        layout.addRow("------------------------------------>", self.testButton)
        self.setLayout(layout)
        self.trainForm = Form()
        self.testForm = TestForm()

    def trainButtonPressed(self):
        self.trainForm.show()
        self.close()


    def testButtonPressed(self):
        self.testForm.show()
        self.close()


