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

    def __init__(self, root_dir, csv, transform=None):

        super().__init__()
        self.root_dir = root_dir
        self.csv = csv
        self.transform = transform

        df = pd.read_csv(csv, sep="\t")

        self.elements = [root_dir + p for p in df['path'].to_list()]
        self.labels = [row.to_list() for _, row in df.iloc[:, -8:].iterrows()]
        self.length = len(df)
        
        self.labels_order = df.iloc[:, -8:].columns.tolist()

    def __len__(self):

        return self.length

    def __getitem__(self, index):

        img = Image.open(self.elements[index].rstrip())
        
        print(self.elements[index])

        resized_img = transforms.Resize((700, 1000))(img)
        cropped_img = TF.crop(resized_img, 90, 0, 520, 1000)

        target = self.labels[index]
        

        
        if self.transform is not None:
            cropped_img = self.transform(cropped_img)
            
        
        image, label = cropped_img, target
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
