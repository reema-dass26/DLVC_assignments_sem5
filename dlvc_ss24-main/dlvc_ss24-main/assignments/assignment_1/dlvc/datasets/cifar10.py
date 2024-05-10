import pickle
from typing import Tuple
import numpy as np


from dlvc.datasets.dataset import  Subset, ClassificationDataset

class CIFAR10Dataset(ClassificationDataset):
    '''
    Custom CIFAR-10 Dataset.
    '''

    def __init__(self, fdir: str, subset: Subset, transform=None):
        '''
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        '''

        self.classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        ## TODO implement
        # See the CIFAR-10 website on how to load the data files
        pass

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''
        ## TODO implement
        pass

    def __getitem__(self, idx: int) -> Tuple:
        '''
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        '''
        ## TODO implement
        pass

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''
        ## TODO implement
        pass

