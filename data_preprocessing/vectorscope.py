from utilities import create_dir

from PIL import Image
import numpy as np
from sys import argv
from numpy import asarray
import matplotlib.pyplot as plt
import cv2
import os
import time
from tqdm import tqdm

origin_path = r'D:\frames'
destination_path = r'D:\vectorscope_frames'

path_list = []

start = time.time()

# create a list of all existing images
for subfolder in os.listdir(origin_path):
    subfolder_path = os.path.join(origin_path, subfolder)
    if os.path.isdir(subfolder_path):
        # Loop over sub-subfolders in the subfolder
        for subsubfolder in os.listdir(subfolder_path):
            subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
                
            for frame in os.listdir(subsubfolder_path):
                frame_path = os.path.join(subsubfolder_path, frame)
                if os.path.exists(frame_path):
                    path_list.append(frame_path)

# create the destination folder if it doesn't exist
create_dir(destination_path)

# vec_list=[]
count = 0

for src_path in tqdm(path_list):
    try:
        src = Image.open(src_path)
        src = src.resize((320, 180))
        src = asarray(src)
        
        if src.dtype == np.uint16:
            src = (src / 2**8).astype(np.uint8)
            
        R, G, B = src[:,:,0], src[:,:,1], src[:,:,2]
        
        Y = (0.299 * R) + (0.587 * G) + (0.114 * B)
        Cb = (-0.169 * R) - (0.331 * G) + (0.499 * B) + 128
        Cr = (0.499 * R) - (0.418 * G) - (0.0813 * B) + 128
        
        # traditional vectorscope orientation
        Cr = 256 - Cr
        
        dst = np.zeros((256, 256, 3), dtype=src.dtype)

        for x in range(src.shape[0]):
            for y in range(src.shape[1]):
                dst[int(Cr[x, y]), int(Cb[x, y])] = np.array([R[x, y], G[x, y], B[x, y]])
                #print(len(dst))
                

        # vec_list.append(dst)
        # vec_avg = np.mean(vec_list, axis=0)
        # plt.imshow(vec_avg, interpolation='nearest')
        
        # plt.imshow(dst, interpolation='nearest')
        # plt.show()
        
        dst_path = src_path.replace(origin_path, destination_path)
        dst_path_to_create = dst_path[:dst_path.rfind('\\')]
        
        create_dir(dst_path_to_create)
        cv2.imwrite(dst_path, dst)
    
    except Exception as e:
        print("error")
        pass 
       
    # count += 1
    
    # if count >= 1000:
    #     break
    
end = time.time()
print('time:', end - start)
