import os
import random
from glob import glob
from copy import deepcopy

from PIL import Image
from PIL import ImageFile
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SynImageDataset(Dataset):
    def __init__(
        self,
        real_dir,
        fake_dir,
        opt,
        process_fn,
    ):
        self.opt = opt
        self.process_fn = process_fn

        self.reals = glob(real_dir+"/*.[jJp][pPEn]*[gG]") # recursive=True
        self.fakes = glob(fake_dir+"/*.[jJp][pPEn]*[gG]")

        self.images = self.reals + self.fakes
        self.labels = [0] * len(self.reals) + [1] * len(self.fakes)

        print(f'Load {len(self.reals)} reals and {len(self.fakes)} fakes.')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            image_path = self.images[idx]
            label = self.labels[idx]
            image = Image.open(image_path).convert('RGB')
            
            image = self.process_fn(image, self.opt, label, image_path)
            return image
        except Exception as e:
            print(e)
            return self[idx+1]
        

class CleanSampleDataset(Dataset):
    def __init__(self, 
        images: list, 
        labels: list, 
        opt, 
        process_fn
    ):
        super().__init__()
        self.images = images
        self.labels = labels
        self.opt = opt
        self.process_fn = process_fn

        print(f'Load {len(self.labels)-self.labels.sum()} reals and {self.labels.sum()} fakes.')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert("RGB")
        return self.process_fn(image, self.opt, label, image_path)