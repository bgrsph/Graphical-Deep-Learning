from domain.network import *
from PyQt5.QtWidgets import *
from domain.debugger import Debug
from tkinter.filedialog import askopenfilename
from ui.errorDialog import *

# All the network names in the project. Add model file into "models" dir, than add model name here
networkNamesList = ["alexnet", "googlenet", "inception_v3",
                    "mobilenet",
                    "densenet121",
                    "densenet169",
                    "densenet201",
                    "densenet161",
                    "mnasnet0_5",
                    "mnasnet1_0",
                    "resnet18",
                    "resnet34",
                    "resnet50",
                    "resnet101",
                    "resnet152",
                    "resnext50_32x4d",
                    "shufflenetv2_x0.5",
                    "shufflenetv2_x1.0", "shufflenetv2_x1.5", "shufflenetv2_x2.0", "squeezenet1_0", "squeezenet1_1",
                    "vgg11", "vgg13", "vgg16", "vgg19", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]

# Stores all the line edits for output class selection
outClassLineEdits = []

# Stroes all the user arguments
args = {}

# Make this true if debug
verbose = True


class TestForm(QDialog):

    def __init__(self, parent=None):
        super(TestForm, self).__init__(parent)
        args["test_mode"] = True
        self.setWindowTitle("Automated Deep Learning Tool: TEST")
        self.layout = QFormLayout()

        self.selectModelLabel = QLabel("Select Model:")
        self.selectModelComboBox = QComboBox()
        self.selectModelComboBox.addItems(networkNamesList)

        self.selectPretrainedModelLabel = QLabel("Select pre-trained model (.pth file)")
        self.selectModelButton = QPushButton("...")
        self.selectModelButton.clicked.connect(self.selectModelButtonPressed)




        self.testDataLabel = QLabel("Select TEST dataset folder")
        self.testDataButton = QPushButton("...")
        self.testDataButton.clicked.connect(self.testDataButtonPressed)

        self.batchNumLabel = QLabel("Enter batch-size:")
        self.batchNumLineEdit = QLineEdit()
        self.batchNumLineEdit.setText("2")  # Default value

        self.deviceLabel = QLabel("Enter the device (cpu or cuda):")
        self.deviceComboBox = QComboBox()
        self.deviceComboBox.addItems(["cpu","cuda"])

        self.selectNumClassesLabel = QLabel("Enter # of output classes:")
        self.selectNumClassesLineEdit = QLineEdit()

        self.enterClassesLabel = QLabel("Press the button to start typing class names")
        self.enterClassesButton = QPushButton("Enter Class Names")
        self.enterClassesButton.clicked.connect(self.outputClassButtonPressed)

        self.allDoneLabel = QLabel("Press this button when done.")
        self.allDoneButton = QPushButton("DONE")
        self.allDoneButton.clicked.connect(self.allDoneButtonPressed)



        self.layout.addRow(self.selectModelLabel, self.selectModelComboBox)
        self.layout.addRow(self.batchNumLabel, self.batchNumLineEdit)
        self.layout.addRow(self.selectPretrainedModelLabel, self.selectModelButton)
        self.layout.addRow(self.testDataLabel, self.testDataButton)
        self.layout.addRow(self.selectNumClassesLabel, self.selectNumClassesLineEdit)
        self.layout.addRow(self.enterClassesLabel, self.enterClassesButton)
        self.layout.addRow(self.allDoneLabel, self.allDoneButton)

        self.setLayout(self.layout)


    def testDataButtonPressed(self):
        args["test_data_path"] = filedialog.askdirectory()
        self.testDataButton.setText(args["test_data_path"])


    def selectModelButtonPressed(self):
        path = askopenfilename(filetypes=[(".pth files", "*.pth")])
        args["pretrained_data_path"] = path
        self.selectModelButton.setText(path)
        Debug(verbose=verbose, message="Pretrained data path: {}".format(args["pretrained_data_path"]))

    def allDoneButtonPressed(self):
        allClear = True
        args["model_name"] = str(self.selectModelComboBox.currentText())
        for i in range(0, len(outClassLineEdits)):
            if outClassLineEdits[i].text().replace(" ", "") == "":
                ErrorDialog(message="Please enter the output class names!")
                allClear = False
                break;

        if self.selectModelButton.text() == "..." or self.selectNumClassesLineEdit.text() == "":
            allClear = False
            ErrorDialog(message="Please fill the entire form!")
        if allClear:
            args["output_classes"] = []
            for i in range(0, len(outClassLineEdits)):
                args["output_classes"].append(outClassLineEdits[i].text())

            args["device"] = str(self.deviceComboBox.currentText())
            args["batch_size"] = int(self.batchNumLineEdit.text())
            Debug(verbose=verbose, message=args)
            for i in reversed(range(self.layout.count())):
                self.layout.itemAt(i).widget().setDisabled(True)
            self.testButton = QPushButton("START TESTING")
            self.layout.addRow("----------------------->", self.testButton)
            self.testButton.clicked.connect(self.startTest)

    def startTest(self):
        self.close()
        Network(args=args).test()


    def outputClassButtonPressed(self):
        str = self.selectNumClassesLineEdit.text()
        if str.replace(" ", "") == "":
            ErrorDialog(message="Please enter the number of output classes!")
        else:
            self.enterClassesButton.setDisabled(True)
            for i in range(0, int(str)):
                lineEdit = QLineEdit()
                outClassLineEdits.append(lineEdit)
                self.layout.addRow(QLabel("Enter the name of class #{}".format(i)), lineEdit)