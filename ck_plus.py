from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data

class CKPlus48(data.Dataset):
    """CK+48 Dataset（全部作为测试集，兼容Fer2013接口）"""

    def __init__(self, split="test", transform=None):
        self.split = split
        self.transform = transform

        # 加载我们生成的h5文件
        self.data = h5py.File("./data/ck_plus.h5", "r", driver="core")

        if self.split == "test":
            # 全部作为测试集
            self.test_data = self.data['data_pixels'][:]
            self.test_labels = self.data['data_labels'][:]
            self.test_data = np.asarray(self.test_data).reshape((-1, 48, 48))
    
    def __getitem__(self, index):
        if self.split == "test":
            img, target = self.test_data[index], self.test_labels[index]
        
        # 和Fer2013完全一样的处理：灰度→3通道RGB
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img.astype('uint8'), 'RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, int(target)
    
    def __len__(self):
        if self.split == "test":
            return len(self.test_data)