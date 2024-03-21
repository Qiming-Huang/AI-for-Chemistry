# from torch import randint
# from torchmetrics.classification import MulticlassConfusionMatrix
# metric = MulticlassConfusionMatrix(num_classes=5)
# metric.update(randint(5, (4,)), randint(5, (4,)))
# metric.update(randint(5, (4,)), randint(5, (4,)))
# metric.update(randint(5, (4,)), randint(5, (4,)))
# fig_, ax_ = metric.plot()


import cv2
from tqdm import tqdm
import os


for root, dirs, files in os.walk("/home/qiming/Desktop/code/AI-for-Chemistry/TH5/th5_dataset/th4"):
    for file in tqdm(files):
        img = cv2.imread(f"/home/qiming/Desktop/code/AI-for-Chemistry/TH5/th5_dataset/th4/{file}")
        img = img[:,200:,:]
        cv2.imwrite(f"/home/qiming/Desktop/code/AI-for-Chemistry/TH5/th5_dataset/pretrain/pretrain_img/{file}", img)