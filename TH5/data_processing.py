import json 
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd

labels = pd.read_csv("./TH5_labels_cv.csv")

file_names = labels['name'].to_list()
density = labels['density'].to_list()
di = labels['di'].to_list()
facetoface = labels['facetoface'].to_list()
channel_dimension = labels['channel_dimension'].to_list()
HB2 = labels['HB2'].to_list()

# TH5_data = []

# for i in range(len(file_names)):
#     with open(f"TH5json_origin/{file_names[i]}.json", "r") as fr:
#         data = json.load(fr)[f'{file_names[i]}']
#         # data = np.arrray(data)
#         TH5_data.append(data)

# TH5_data = np.array(TH5_data)

# density
k = 4
density = np.array(density)
quantiles = np.percentile(density, np.linspace(0, 100, k+1))

# 使用cut函数将数据分配到不同的类别
categories = np.digitize(density, quantiles, right=True)
cots = [0 for i in range(k+1)]
for i in categories:
    cots[i] += 1
