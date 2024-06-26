import os

import lightning as L
import pandas as pd
import seaborn as sn
import torch
from IPython.display import display
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy, MulticlassROC
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset 
from torchvision.models import resnet18, ResNet18_Weights
import wandb 
import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import cv2
from lightning.pytorch.loggers import WandbLogger



BATCH_SIZE = 64
NUM_CLASSES = 4
LR = 1e-6
MAX_EPOCH = 5000

# define your model
class Th5_model_global_local(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.global_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.local_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # modify
        self.global_model.fc = nn.Linear(512, NUM_CLASSES)
        self.global_model.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, padding=1, bias=False
        )
        self.local_model.fc = nn.Linear(512, NUM_CLASSES)
        self.local_model.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, padding=1, bias=False
        )        

    def forward(self, x):
        x1 = x[:,:,:,:200]
        x2 = x[:,:,:,200:]

        out_1 = self.global_model(x1)
        out_2 = self.local_model(x2)

        return out_1 + out_2

# define your dataset
class TH5_data(Dataset): 
    def __init__(self, x, y): 
        self.x = x
        self.y = y
        self.sc = MinMaxScaler()
        b, w, h = self.x.shape
        self.sc.fit(self.x.reshape((b, w*h)))
        self.x = self.sc.transform(self.x.reshape((b, w*h))).reshape((b, w, h))
        
    def __len__(self): 
        return len(self.x) 
  
    def __getitem__(self, index): 
        x = self.x[index]
        y = self.y[index]
    
        x = torch.as_tensor(x).float().unsqueeze(0)
        y = torch.as_tensor(y).long()

        return x, y

class TH5_train(Dataset): 
    def __init__(self, label_type): 
        self.label_type = label_type
        self.img_path = "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/npz_data/train_imgs"
        img_lists = []
        for root, dirs, files in os.walk(self.img_path):
            for file in files:
                img_lists.append(file)
        self.train_labels = {
            "density" : [],
            "di" : [],
            "HB_class" : [],
            "f2f_class" : [],
            "chan_dim" : []
        }
        label_csv_set = [
            "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/hirshfeld_data_feb24/csv/TH5_v2.csv",
            "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/hirshfeld_data_feb24/csv/TH2_v2.csv",
            "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/hirshfeld_data_feb24/csv/TH1_v2.csv",
            "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/hirshfeld_data_feb24/csv/T2_v2.csv",
        ]
        for i in range(len(label_csv_set)):
            label = pd.read_csv(label_csv_set[i])
            
            self.train_labels['HB_class'].extend(label['HB_class'].to_numpy()) # 5777 17696 6355 306 2
            self.train_labels['f2f_class'].extend(label['f2f_class'].to_numpy()) # 12312 11279 5779 708 58
            self.train_labels['chan_dim'].extend(label['chan_dim'].to_numpy())  # 12965 9804 4009 3358              
                
        img_lists.sort(key = lambda x:int(x.split(".")[0]))
        self.img_lists = img_lists
        
    def __len__(self): 
        return len(self.img_lists) 
  
    def __getitem__(self, index): 
        x = cv2.imread(f"{self.img_path}/{self.img_lists[index]}")
        x = x[:,:,0] / 255
        y = self.train_labels[f'{self.label_type}'][index]
    
        x = torch.as_tensor(x).float().unsqueeze(0)
        y = torch.as_tensor(y).long()

        return x, y 

class TH5_test(Dataset): 
    # 8203
    def __init__(self, label_type): 
        self.label_type = label_type
        self.img_path = "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/npz_data/test_imgs"
        img_lists = []
        for root, dirs, files in os.walk(self.img_path):
            for file in files:
                img_lists.append(file)
        self.train_labels = {
            "density" : [],
            "di" : [],
            "HB_class" : [],
            "f2f_class" : [],
            "chan_dim" : []
        }
        label = pd.read_csv("/home/qiming/Desktop/code/AI-for-Chemistry/TH5/hirshfeld_data_feb24/csv/TH4_v2.csv")
        
        self.train_labels['HB_class'].extend(label['HB_class'].to_numpy()) # 529, 5102, 2433, 139
        self.train_labels['f2f_class'].extend(label['f2f_class'].to_numpy()) # 
        self.train_labels['chan_dim'].extend(label['chan_dim'].to_numpy())  # 
                
        img_lists.sort(key = lambda x:int(x.split(".")[0]))
        self.img_lists = img_lists
        
    def __len__(self): 
        return len(self.img_lists) 
  
    def __getitem__(self, index): 
        x = cv2.imread(f"{self.img_path}/{self.img_lists[index]}")
        x = x[:,:,0] / 255
        y = self.train_labels[f'{self.label_type}'][index]
    
        x = torch.as_tensor(x).float().unsqueeze(0)
        y = torch.as_tensor(y).long()

        return x, y 

# seen and unseen dataset
def prepare_joint_data(label_type):                                    
    label_csv_set = [
        "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/hirshfeld_data_feb24/csv/TH5_v2.csv",
        "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/hirshfeld_data_feb24/csv/TH4_v2.csv",
        "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/hirshfeld_data_feb24/csv/TH2_v2.csv",
        "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/hirshfeld_data_feb24/csv/TH1_v2.csv",
        "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/hirshfeld_data_feb24/csv/T2_v2.csv",
        "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/hirshfeld_data_feb24/csv/SH2_v2.csv",
        "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/hirshfeld_data_feb24/csv/SH1_v2.csv",
        "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/hirshfeld_data_feb24/csv/S2_v2.csv"
    ]
    imgs_path_set = [
        "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/npz_data/Th5_data.npz",
        "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/npz_data/TH4_data.npz",
        "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/npz_data/TH2_data.npz",
        "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/npz_data/TH1_data.npz",
        "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/npz_data/T2_data.npz",
        "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/npz_data/SH2_data.npz",
        "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/npz_data/SH1_data.npz",
        "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/npz_data/S2_data.npz"
    ]

    train_images = []

    train_labels = {
        "density" : [],
        "di" : [],
        "HB_class" : [],
        "f2f_class" : [],
        "chan_dim" : []
    }

    for i in range(2,8,1):
        label = pd.read_csv(label_csv_set[i])
        images = np.load(imgs_path_set[i], allow_pickle=True)['data']
        
        train_labels['HB_class'].extend(label['HB_class'].to_numpy())
        train_labels['f2f_class'].extend(label['f2f_class'].to_numpy())
        train_labels['chan_dim'].extend(label['chan_dim'].to_numpy())
        train_images.extend(images)
        
    train_images = np.array(train_images)

    test_images = []

    test_labels = {
        "density" : [],
        "di" : [],
        "HB_class" : [],
        "f2f_class" : [],
        "chan_dim" : []
    }

    for i in range(2):
        label = pd.read_csv(label_csv_set[i])
        images = np.load(imgs_path_set[i], allow_pickle=True)['data']
        
        test_labels['HB_class'].extend(label['HB_class'].to_numpy())
        test_labels['f2f_class'].extend(label['f2f_class'].to_numpy())
        test_labels['chan_dim'].extend(label['chan_dim'].to_numpy())
        test_images.extend(images)
        
    test_images = np.array(test_images)
    
    if label_type == "HB_class":
        test_labels = test_labels['HB_class']
        train_labels = train_labels['HB_class']
    if label_type == "f2f_class":
        test_labels = test_labels['f2f_class']
        train_labels = train_labels['f2f_class']    
    if label_type == "chan_dim":
        test_labels = test_labels['chan_dim']
        train_labels = train_labels['chan_dim']          
    
    return train_images, test_images, train_labels, test_labels

# template
class Lighting_Model(L.LightningModule):
    def __init__(self, data_dir="", hidden_size=64, learning_rate=LR):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes

        # Define PyTorch model
        self.model = Th5_model_global_local()

        self.val_accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        # self.val_accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES)
        # self.test_accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES)
        # self.mc_auroc = MulticlassROC(num_classes=NUM_CLASSES, average=None).to("cuda")
        self.metric_cf = MulticlassConfusionMatrix(num_classes=NUM_CLASSES).to("cuda")
        self.res = {
            "pred_logits" : [],
            "y" : []
        }

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        # x
        if batch_idx == 0:
            pred = torch.stack(self.res['pred_logits'])
            y = torch.stack(self.res['y'])
            # c, b, n = pred.shape
            res = self.metric_cf(pred, y)
            with open("/home/qiming/Desktop/code/AI-for-Chemistry/TH5/res.txt", "a") as fr:
                fr.write(f"{res.detach().cpu().numpy()}\n\n")
            self.res = {
                "pred_logits" : [],
                "y" : []
            }
        
        x, y = batch
        logits = self(x)
        class_weights = torch.as_tensor([1.4785569105691057, 0.8584493745574698, 6.583257918552037, 145.49]).float().to("cuda")
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)
        
        self.res['pred_logits'].extend(logits.softmax(dim=-1))
        self.res['y'].extend(y)
        
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True) 

        # muticlass-auc
        # print(mc_auroc(logits.softmax(dim=-1), y)) 
        # res = self.mc_auroc(logits.softmax(dim=-1), y)
        # for i in range(NUM_CLASSES):
        #     self.res[i].append(float(res[i].detach().cpu().numpy())) 

        # wandb
        # acc = (torch.sum(preds == y) / len(y)).detach().cpu().numpy()
        # wandb.log({"val_loss": loss})

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################
        
    def prepare_data(self):

        # self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True) 
        self.train_ds = TH5_train(label_type="chan_dim") 
        self.test_ds = TH5_test(label_type="chan_dim")  

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=BATCH_SIZE, num_workers=31, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_ds, batch_size=BATCH_SIZE, num_workers=31)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=BATCH_SIZE, num_workers=31)

model = Lighting_Model()
trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=MAX_EPOCH,
    logger=CSVLogger(save_dir="logs/"),
)
trainer.fit(model)
