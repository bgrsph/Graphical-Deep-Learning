import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from domain.debugger import *

verbose = True
class DataLoader:
    def __init__(self, args):
        Debug(verbose=verbose, message="Initializing Data Loader...")
        self.args = args
        self.batch_size = args["batch_size"]
        Debug(verbose=verbose, message="Data Loader has been initialized")

    def get_transform_orientation1(self):
        return transforms.Compose([
            transforms.RandomVerticalFlip(),  # flipping the image vertically
            transforms.RandomHorizontalFlip(),  # flipping the image horizontally
            transforms.ToTensor(),  # convert the image to a Tensor
            # transforms.Normalize(
            #          mean = [0.537, 0.350, 0.599],
            #          std  = [0.246, 0.258, 0.212])
        ])  # normalize the image

    def get_transform_orientation2(self):
        return transforms.Compose([
            transforms.ToTensor(),  # convert the image to a Tensor
            # transforms.Normalize(
            #          mean =[0.537, 0.350, 0.599],
            #          std  =[0.246, 0.258, 0.212])
        ])  # normalize the image

    def get_data_set(self, mode):
        if mode is "test":
            return datasets.ImageFolder(root=self.args["test_data_path"], transform=self.get_transform_orientation2())
        elif mode is "train":
            return datasets.ImageFolder(root=self.args["train_data_path"], transform=self.get_transform_orientation1())
        elif mode is "validation":
            return datasets.ImageFolder(root=self.args["validation_data_path"],
                                        transform=self.get_transform_orientation2())

    def get_iterable_data_set(self, mode):
        return torch.utils.data.DataLoader(dataset=self.get_data_set(mode),
                                    batch_size=self.batch_size,
                                    shuffle=True)


