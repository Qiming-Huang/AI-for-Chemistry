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


BATCH_SIZE = 64
NUM_CLASSES = 5
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

# define your dataset
def prepare_TH5_data(label_type):
    labels = pd.read_csv("./TH5_labels_cv.csv")
    
    density = labels['density'].to_list()
    di = labels['di'].to_list()
    facetoface = labels['facetoface'].to_list()
    channel_dimension = labels['channel_dimension'].to_list()
    HB2 = labels['HB2'].to_list() 

    imgs = np.load("./Th5_data.npz", allow_pickle=True)['data']

    if label_type == "facetoface":
        cots = [0 for i in range(len(np.unique(facetoface)))]
        for i in facetoface:
            cots[i] += 1
        idxs = []
        cot_0 = 0
        cot_1 = 0
        cot_2 = 0
        cot_3 = 0
        cot_4 = 0

        for i in range(len(facetoface)):
            if facetoface[i] == 0 and cot_0 < 200:
                cot_0 += 1
                idxs.append(i)
            if facetoface[i] == 1 and cot_1 < 200:
                cot_1 += 1
                idxs.append(i)
            if facetoface[i] == 2 and cot_2 < 156:
                cot_2 += 1
                idxs.append(i)
            if facetoface[i] == 3 and cot_3 < 200:
                cot_3 += 1
                idxs.append(i)
            if facetoface[i] == 4 and cot_4 < 200:
                cot_4 += 1
                idxs.append(i) 
        idxs = np.array(idxs)  
        x = imgs[idxs]
        y = np.array(facetoface)[idxs]   

        return x, y 
    if label_type == "density":
        k = 4
        density = np.array(density)
        quantiles = np.percentile(density, np.linspace(0, 100, k+1))

        # [1, 644, 643, 642, 644]
        # [0.3438  , 0.9873  , 1.095   , 1.194175, 1.4323  ]

        categories = np.digitize(density, quantiles, right=True)
        cots = [0 for i in range(k+1)]
        for i in categories:
            cots[i] += 1  
        return imgs, categories                                                    

# template
class Lighting_Model(L.LightningModule):
    def __init__(self, data_dir="", hidden_size=64, learning_rate=LR):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        self.model = Th5_model_global_local()

        self.val_accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        wandb.log({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)
        
        # wandb
        # acc = (torch.sum(preds == y) / len(y)).detach().cpu().numpy()
        wandb.log({"val_loss": loss})

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################
        
    def prepare_data(self):
        x, y = prepare_TH5_data(label_type="density")

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True) 
        self.train_ds = TH5_data(self.train_x, self.train_y) 
        self.test_ds = TH5_data(self.test_x, self.test_y)   

    # def setup(self, stage=None):
    #     # Assign train/val datasets for use in dataloaders
    #     if stage == "fit" or stage is None:
    #         mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
    #         self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    #     # Assign test dataset for use in dataloader(s)
    #     if stage == "test" or stage is None:
    #         self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.test_ds, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=BATCH_SIZE)

wandb.init(project="TH5")

model = Lighting_Model()
trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=MAX_EPOCH,
    logger=CSVLogger(save_dir="logs/"),
)
trainer.fit(model)