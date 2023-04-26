import os
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF

# from skimage import io
# import torch


class MovieFrameDataset(Dataset):

    def __init__(self, root_dir, multilabel=False, transform=None):

        super().__init__()
        self.root_dir = root_dir
        self.multilabel = multilabel
        self.transform = transform

        elements = []
        length = 0

        # Loop over subfolders in the parent folder
        for subfolder in os.listdir(root_dir):
            subfolder_path = os.path.join(root_dir, subfolder)
            if os.path.isdir(subfolder_path):
                # Loop over sub-subfolders in the subfolder
                count = 0
                for subsubfolder in os.listdir(subfolder_path):
                    subsubfolder_path = os.path.join(
                        subfolder_path, subsubfolder)

                    for frame in os.listdir(subsubfolder_path):
                        frame_path = os.path.join(subsubfolder_path, frame)
                        if os.path.exists(frame_path):
                            count += 1
                            elements.append(frame_path)

            length += count

        self.length = length
        self.elements = elements
        
        categories_list = []
        
        for subfolder in os.listdir(root_dir):
            categories_list += subfolder.split('_')
        
        categories_list = list(dict.fromkeys(categories_list))
        categories = dict(zip(categories_list, [i for i in range(len(categories_list))]))
        self.categories = categories

    def __len__(self):

        return self.length

    def __getitem__(self, index):

        img = Image.open(self.elements[index].rstrip())
        
        resized_img = transforms.Resize((700, 1000))(img)
        cropped_img = TF.crop(resized_img, 90, 0, 520, 1000)
        
        if self.multilabel:
            target = self.elements[index].rstrip().split('\\')[2]
        else:
            target = self.elements[index].rstrip().split('\\')[2].split('_')[0]
        
        image, label = cropped_img, self.categories[target]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label  