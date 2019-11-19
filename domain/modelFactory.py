from domain.models.alexnet import *
from domain.models.googlenet import *
from domain.models.inception import *
from domain.models.mobilenet import *
from domain.models.densenet import *
from domain.models.mnasnet import *
from domain.models.resnet import *
from domain.models.shufflenetv2 import *
from domain.models.squeezenet import *
from domain.models.vgg import *
from domain.debugger import *

"""This class gets user selections as input and produce the desired model as output"""

verbose = True


class ModelFactory:

    def __init__(self,args):
        self.args = args
        Debug(verbose=verbose,message="Model Factory has been initialized.")


    def get_model(self):
        Debug(verbose=verbose, message="Model Factory get_model() method has been initializedw with args: {}".format(self.args))
        networkName = self.args["model_name"]
        args = self.args
        if args["test_mode"]:
            Debug(verbose=verbose,message="Test Mode Selected")
            args["pretrained"] = True
            args["progress"] = True
        if networkName == "alexnet":
            Debug(verbose=verbose, message="Alexnet has been selected.")
            return alexnet(args["pretrained"], args["progress"])

        elif networkName == "googlenet":
            return googlenet(args["pretrained"], args["progress"])

        elif networkName == "inception_v3":
            return inception_v3(args["pretrained"], args["progress"])

        elif networkName == "mobilenet":
            return mobilenet_v2(args["pretrained"], args["progress"])

        elif networkName == "densenet121":
            return densenet121(args["pretrained"], args["progress"])

        elif networkName == "densenet169":
            return densenet169(args["pretrained"], args["progress"])

        elif networkName == "densenet201":
            return densenet201(args["pretrained"], args["progress"])

        elif networkName == "densenet161":
            return densenet161(args["pretrained"], args["progress"])

        elif networkName == "mnasnet0_5":
            return mnasnet0_5(args["pretrained"], args["progress"])

        elif networkName == "mnasnet1_0":
            return mnasnet1_0(args["pretrained"], args["progress"])

        elif networkName == "resnet18":
            return resnet18(args["pretrained"], args["progress"])

        elif networkName == "resnet34":
            return resnet34(args["pretrained"], args["progress"])

        elif networkName == "resnet50":
            return resnet50(args["pretrained"], args["progress"])

        elif networkName == "resnet101":
            return resnet101(args["pretrained"], args["progress"])

        elif networkName == "resnet152":
            return resnet152(args["pretrained"], args["progress"])

        elif networkName == "resnext50_32x4d":
            return resnext50_32x4d(args["pretrained"], args["progress"])

        elif networkName == "shufflenetv2_x0.5":
            return shufflenet_v2_x0_5(args["pretrained"], args["progress"])

        elif networkName == "shufflenetv2_x1.0":
            return shufflenet_v2_x1_0(args["pretrained"], args["progress"])

        elif networkName == "shufflenetv2_x1.5":
            return shufflenet_v2_x1_5(args["pretrained"], args["progress"])

        elif networkName == "shufflenetv2_x2.0":
            return shufflenet_v2_x2_0(args["pretrained"], args["progress"])

        elif networkName == "squeezenet1_0":
            return squeezenet1_0(args["pretrained"], args["progress"])

        elif networkName == "squeezenet1_1":
            return squeezenet1_1(args["pretrained"], args["progress"])

        elif networkName == "vgg11":
            return vgg11(args["pretrained"], args["progress"])

        elif networkName == "vgg13":
            return vgg13(args["pretrained"], args["progress"])

        elif networkName == "vgg16":
            return vgg16(args["pretrained"], args["progress"])

        elif networkName == "vgg19":
            return vgg19(args["pretrained"], args["progress"])

        elif networkName == "vgg11_bn":
            return vgg11_bn(args["pretrained"], args["progress"])

        elif networkName == "vgg13_bn":
            return vgg13_bn(args["pretrained"], args["progress"])

        elif networkName == "vgg16_bn":
            return vgg16_bn(args["pretrained"], args["progress"])

        elif networkName == "vgg19_bn":
            return vgg19_bn(args["pretrained"], args["progress"])