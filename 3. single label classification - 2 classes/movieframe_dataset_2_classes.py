import os
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF


class MovieFrameDataset(Dataset):

    def __init__(self, root_dir, csv, transform=None, raw=False):

        super().__init__()
        self.root_dir = root_dir
        self.csv = csv
        self.transform = transform
        self.raw = raw

        df = pd.read_csv(csv, sep="\t")

        self.elements = [root_dir + p.replace('\\', '/') for p in df['path'].to_list()]
        
        # print(self.elements)
        
        self.labels = [row.to_list() for _, row in df.iloc[:, -2:].iterrows()]
        self.length = len(df)
        
        self.labels_order = df.iloc[:, -2:].columns.tolist()

    def __len__(self):

        return self.length

    def __getitem__(self, index):

        img = Image.open(self.elements[index].rstrip())
        
        # print(self.elements[index])

        target = self.labels[index]
        
        if self.raw:
            # two following steps of transform are not enclosed in transform 
            # parameter as cropping in that way is not available in transforms
            # library
            # they are only used for raw images
            img = transforms.Resize((700, 1000))(img)
            img = TF.crop(img, 90, 0, 520, 1000)    
        
        if self.transform is not None:    
            img = self.transform(img)
            
        
        image, label = img, target
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
