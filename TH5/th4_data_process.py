import numpy as np
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torch

import random
import math
import numpy as np
from sklearn.model_selection import train_test_split


def generate_images_per_dataset():
    labels = pd.read_csv("hirshfeld_data_feb24/csv/TH5_v2.csv")
    image_path = "hirshfeld_data_feb24/hirshfeld_origin/TH5"
    save_path = "th5_dataset/pretrain/pretrain_img"

    for i in tqdm(labels['name']):
        with open(f"{image_path}/{i}.json", "r") as fr:
            img = json.load(fr)[i.split('_')[-1]]
            img = np.array(img)[:,200:]
            # img = Image.fromarray(np.uint8(img))
            img = cv2.convertScaleAbs(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(f"{save_path}/{i}.jpg", img)

def generate_labels_per_dataset():
    with open("/home/qiming/Desktop/code/AI-for-Chemistry/TH5/th5_dataset/joint_labels_chan.txt", "a") as fr:
        labels = pd.read_csv("hirshfeld_data_feb24/csv/T2_v2.csv")
        for i in labels.iloc:
            fr.write(f"{i['name']},{i['chan_dim']}\n")
            

class CustomDataset(Dataset):
    def __init__(self, mode, transform=None):
        labels = {

        }
        with open("/home/qiming/Desktop/code/AI-for-Chemistry/TH5/th5_dataset/joint_labels_chan.txt", "r") as fr:
            content = fr.readlines()
            for i in content:
                labels[i.split(",")[0]] = float(i.split(",")[-1])

        image_path = "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/th5_dataset/pretrain/pretrain_img"
        self.img_path = []
        self.hb_labels = []
        for key, value in labels.items():
               self.img_path.append(f"{image_path}/{key}.jpg")
               self.hb_labels.append(value)
               
        self.mode = mode
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            self.img_path, self.hb_labels, test_size=0.8, random_state=42)        
           

    def __len__(self):
        if self.mode == "train":
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.mode == "train":
            image = cv2.imread(self.train_data[idx])
            label = int(self.train_labels[idx])
            
            image = torch.as_tensor(image).float().permute(2,0,1)
            label = torch.as_tensor(label).long()
        else:
            image = cv2.imread(self.train_data[idx])
            label = int(self.train_labels[idx])
            
            image = torch.as_tensor(image).float().permute(2,0,1)
            label = torch.as_tensor(label).long()            

        return image, label
    

train_ds = CustomDataset(mode="train")
test_ds = CustomDataset(mode="test")

# train_loader = DataLoader(train_ds, batch_size=32)
# test_loader = DataLoader(test_ds, batch_size=32)


# for x, y in train_loader:
#     print(x.shape, y.shape)
    
# generate_labels_per_dataset()

cot = [0,0,0,0]
for i in range(train_ds.__len__()):
    x, y = train_ds[i]
    cot[int(y)] += 1
print(cot)

cot = [0,0,0,0]
for i in range(test_ds.__len__()):
    x, y = test_ds[i]
    cot[int(y)] += 1
print(cot)
