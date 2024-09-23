import torch as t
from torch.utils import data
from PIL import Image
import os
import cv2
import numpy as np
import random


class ImagetNetTrainDataset(data.Dataset):
    # './data/ILSVRC2012/'
    def __init__(self, path_train_imagenet, num_classes, num_imgs_per_batch, transforms=None):
        self.path_img = []
        self.transforms = transforms

        dirs = [x[0] for x in os.walk(path_train_imagenet)]
        dirs = dirs[1:]
        # Sort the directory in alphabetical order (same as synset_words.txt)
        dirs = sorted(dirs)
        it = 0
        Matrix = [0 for x in range(1200)]

        for d in dirs:
            for root, dir, filename in os.walk(d):
                Matrix[it] = filename
            it = it + 1

        for k in range(num_classes):
            for u in range(num_imgs_per_batch):
                img_path = os.path.join(dirs[k], Matrix[k][u])
                self.path_img.append(img_path)
        
        # for num in range(dirs[150]):
        #     img_path = os.path.join(dirs[150], Matrix[150][num])
        #     self.path_img.append(img_path)
        
        random.shuffle(self.path_img)


    def __getitem__(self, index):
        path_img = self.path_img[index]
        # data = cv2.imread(path_img)
        # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = Image.open(path_img).convert('RGB')
        if self.transforms:
            data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.path_img)


class ImagetNetTestDataset(data.Dataset):
    # './data/ILSVRC2012/'
    def __init__(self, path_test_imagenet, num_imgs, transforms=None):
        self.path_img = []
        self.transforms = transforms
        
        for img_name in os.listdir(path_test_imagenet)[:num_imgs]:
            img_path = path_test_imagenet+img_name
            self.path_img.append(img_path) 
 
    def __getitem__(self, index):
        path_img = self.path_img[index]
        data = Image.open(path_img).convert('RGB')

        # data = cv2.imread(path_img)
        # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        # data = data.astype(np.float32)  # RGB image in range [0, 255]
        # data = data / 255.0
        # data = cv2.resize(data, (224, 224))
        
        if self.transforms:
            data = self.transforms(data)
        return data,self.path_img[index].split("/")[-1]

    def __len__(self):
        return len(self.path_img)




