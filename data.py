import pandas as pd
from skimage import io
import os
import torch
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils

class CassavaDataset():
    
    def __init__(self, data_csv, data_root, transforms=None, augmentation=None):
        
        self.data_source = data_root + '/train_images/'
        self.transforms = transforms 
        self.map_classes = {}
        
        from json import load
        with open(data_root+'/label_num_to_disease_map.json') as f:
             mm = load(f)
        for key in mm.keys():
            self.map_classes[int(key)] = mm[key]
        
        self.train = data_csv
        self.train['disease'] = self.train["label"].map(self.map_classes)     
    
    def __len__(self):
        return len(self.train)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.read_img(self.train.iloc[idx, 0])
        
        if self.transforms:
            image = self.transforms(image=image)['image']
        
        #image = image.transpose((2, 0, 1))
        label = self.train.iloc[idx, 1]
        
        return image.double(), int(label)
    
    def read_img(self, img_id):
        return io.imread(self.data_source+img_id)
        
    def print_random(self, label=None):
        
        idx = np.random.randint(low=0, high=self.__len__())
        img, label = self.__getitem__(idx)
        print(str(label), self.map_classes[label])
        print('Image shape:', img.shape, '. Therefore, unable to print.')
        #plt.imshow(img)
        
class TestDataset():
    
    def __init__(self, data_root, transforms=None):
        
        self.data_source = data_root + '/test_images/'
        self.transforms = transforms   
        self.test = os.listdir(self.data_source)
    
    def __len__(self):
        return len(self.test)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.read_img(self.test[idx])
        if self.transforms:
            image = self.transforms(image=image)['image']
        return image
    
    def read_img(self, img_id):
        return io.imread(self.data_source+img_id)
        
    def print_random(self):
        from random import choice
        img = self.read_img(choice(self.test))
        plt.imshow(img)

