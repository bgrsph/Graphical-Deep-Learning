from domain.network import *
from tkinter import filedialog
from PyQt5.QtWidgets import *
from domain.debugger import Debug
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

class Form(QDialog):
    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        args["test_mode"] = False
        self.layout = QFormLayout()
        self.setWindowTitle("Automated Deep Learning Tool: TRAIN")


        # Row 0: Select the model from a given list
        self.selectModelLabel = QLabel("Select Model:")
        self.selectModelComboBox = QComboBox()
        self.selectModelComboBox.addItems(networkNamesList)

        # Row 1: Select number of classes
        self.selectNumClassesLabel = QLabel("Enter # of output classes:")
        self.selectNumClassesLineEdit = QLineEdit()

        # Row 2: Enter all the output class names
        self.enterClassesLabel = QLabel("Press the button to start typing class names")
        self.enterClassesButton = QPushButton("Start")
        self.enterClassesButton.clicked.connect(self.outputClassButtonPressed)

        # Row 3-9: Enter hyper-parameters
        self.pretrainedLabel = QLabel("Pretrained? (t or f)")
        self.pretrainedLineEdit = QLineEdit()
        self.pretrainedLineEdit.setText("t")  # Default value

        self.progressLabel = QLabel("Progress? (t or f)")
        self.progressLineEdit = QLineEdit()
        self.progressLineEdit.setText("t")  # Default value

        self.batchNumLabel = QLabel("Enter batch-size:")
        self.batchNumLineEdit = QLineEdit()
        self.batchNumLineEdit.setText("2")  # Default value

        self.epochsLabel = QLabel("Enter # of epochs:")
        self.epochsLineEdit = QLineEdit()
        self.epochsLineEdit.setText("10")  # Default value

        self.learningRateLabel = QLabel("Enter learning rate:")
        self.learningRateLineEdit = QLineEdit()
        self.learningRateLineEdit.setText("0.001")  # Default value

        self.momentumLabel = QLabel("Enter momentum for SGD optimizer:")
        self.momentumLineEdit = QLineEdit()
        self.momentumLineEdit.setText("0")  # Default value

        self.weightDecayLabel = QLabel("Enter weight decay for SGD optimizer:")
        self.weightDecayLineEdit = QLineEdit()
        self.weightDecayLineEdit.setText("0")  # Default value

        # Row 10: Device Selection
        self.deviceLabel = QLabel("Enter the device (cpu or cuda):")
        self.deviceComboBox = QComboBox()
        self.deviceComboBox.addItems(["cpu","cuda"])


        # Row 11-14: Get paths of datasets
        self.testDataLabel = QLabel("Select TEST dataset folder")
        self.testDataButton = QPushButton("...")
        self.testDataButton.clicked.connect(self.testDataButtonPressed)

        self.trainDataLabel = QLabel("Select TRAIN dataset folder")
        self.trainDataButton = QPushButton("...")
        self.trainDataButton.clicked.connect(self.trainDataButtonPressed)

        self.validationDataLabel = QLabel("Select VALIDATION dataset folder")
        self.validationDataButton = QPushButton("...")
        self.validationDataButton.clicked.connect(self.validationDataButtonPressed)

        # Press this button when you stop finishing enter data
        self.allDoneButtonLabel = QLabel("Press when you are done")
        self.allDoneButton = QPushButton("DONE")
        self.allDoneButton.clicked.connect(self.allDoneButtonPressed)

        # Add all the rows to layout
        self.layout.addRow(self.selectModelLabel, self.selectModelComboBox)
        self.layout.addRow(self.selectNumClassesLabel, self.selectNumClassesLineEdit)
        self.layout.addRow(self.enterClassesLabel, self.enterClassesButton)
        self.layout.addRow(self.pretrainedLabel, self.pretrainedLineEdit)
        self.layout.addRow(self.progressLabel, self.progressLineEdit)
        self.layout.addRow(self.batchNumLabel, self.batchNumLineEdit)
        self.layout.addRow(self.epochsLabel, self.epochsLineEdit)
        self.layout.addRow(self.learningRateLabel, self.learningRateLineEdit)
        self.layout.addRow(self.momentumLabel, self.momentumLineEdit)
        self.layout.addRow(self.weightDecayLabel, self.weightDecayLineEdit)
        self.layout.addRow(self.deviceLabel, self.deviceComboBox)
        self.layout.addRow(self.testDataLabel, self.testDataButton)
        self.layout.addRow(self.trainDataLabel, self.trainDataButton)
        self.layout.addRow(self.validationDataLabel, self.validationDataButton)
        self.layout.addRow(self.allDoneButtonLabel, self.allDoneButton)

        self.setLayout(self.layout)

    def testDataButtonPressed(self):
        args["test_data_path"] = filedialog.askdirectory()
        self.testDataButton.setText(args["test_data_path"])
    def trainDataButtonPressed(self):
        args["train_data_path"] = filedialog.askdirectory()
        self.trainDataButton.setText(args["train_data_path"])
    def validationDataButtonPressed(self):
        args["validation_data_path"] = filedialog.askdirectory()
        self.validationDataButton.setText(args["validation_data_path"])

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

    # Get all the user input to args dict
    def allDoneButtonPressed(self):
        allClear = True

        for i in range(0, len(outClassLineEdits)):
            if outClassLineEdits[i].text().replace(" ", "") == "":
                allClear = False
                ErrorDialog(message="Please enter the output class names!")
                break

        if self.testDataButton.text() == "..." or self.trainDataButton.text() == "..." or \
                self.validationDataButton.text() == "..." or self.selectNumClassesLineEdit.text() == "" or not len(outClassLineEdits):
            allClear = False
            ErrorDialog(message="Please fill the entire form")
        if allClear:
            args["output_classes"] = []
            args["model_name"] = str(self.selectModelComboBox.currentText())
            for i in range(0,len(outClassLineEdits)):
                args["output_classes"].append(outClassLineEdits[i].text())
            args["pretrained"] = self.pretrainedLineEdit.text() == 't'
            args["progress"] = self.pretrainedLineEdit.text() == 't'
            args["batch_size"] = int(self.batchNumLineEdit.text())
            args["epochs"] = int(self.epochsLineEdit.text())
            args["learning_rate"] = float(self.learningRateLineEdit.text())
            args["weight_decay"] = float(self.weightDecayLineEdit.text())
            args["momentum"] = float(self.momentumLineEdit.text())
            args["device"] = str(self.deviceComboBox.currentText())
            Debug(verbose=verbose, message=args)

            for i in reversed(range(self.layout.count())):
                self.layout.itemAt(i).widget().setDisabled(True)
            self.trainButton = QPushButton("START TRAINING")
            self.layout.addRow("----------------------------------------------->", self.trainButton)
            self.trainButton.clicked.connect(self.startTrain)


    def startTrain(self):
        Debug(verbose=verbose,message="Training has been started.")
        self.close()
        Network(args=args).train()
