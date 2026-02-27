'''Fer2013 Dataset class'''

from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data

class Fer2013(data.Dataset):
    """FER2013 Dataset class

    Args:
        train(bool): If True, creates dataset from training set, otherwise creates from test set.
        transform(callable, optional): A function/transform that takes in a PIL image and returns
    """

    def __init__(self, split="train", transform=None):
        self.split = split
        self.transform = transform

        self.data = h5py.File("./data/fer2013.h5", "r", driver="core")

        if self.split == "train":
            self.train_data = self.data['train_pixels'][:]
            self.train_labels = self.data['train_labels'][:]
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((28709, 48, 48))

        elif self.split == "test":
            self.test_data = self.data['test_pixels'][:]
            self.test_labels = self.data['test_labels'][:]
            self.test_data = np.asarray(self.test_data)
            self.test_data = self.test_data.reshape((7178, 48, 48))
    
    def __getitem__(self, index):
        if self.split == "train":
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == "test":
            img, target = self.test_data[index], self.test_labels[index]
        
        img = img[:, :, np.newaxis] # add channel dimension
        # convert to 3 channels, this is necessary 
        # since most pre-trained models expect 3-channel input
        # even though the original images are grayscale
        img = np.concatenate((img, img, img), axis=2) 
        img = Image.fromarray(img.astype('uint8'), 'RGB') # convert to PIL image

        if self.transform is not None:
            img = self.transform(img)
        
        return img, int(target)
    
    def __len__(self):
        if self.split == "train":
            return len(self.train_data)
        elif self.split == "test":
            return len(self.test_data)