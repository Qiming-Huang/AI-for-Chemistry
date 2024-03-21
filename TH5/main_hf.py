# import torch
# from torchvision import transforms, datasets
# from torch.utils.data import DataLoader
# from transformers import ViTMAEForPreTraining, AutoImageProcessor, ViTMAEModel
# from transformers import AdamW
# import matplotlib.pyplot as plt
# from torchvision.utils import save_image

# # 数据集加载和处理
# class MyImageDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, feature_extractor):
#         self.dataset = datasets.ImageFolder(root_dir)
#         # self.feature_extractor = feature_extractor

#     def __getitem__(self, idx):
#         image, label = self.dataset[idx]
#         inputs = self.feature_extractor(images=image, return_tensors="pt")
#         return inputs, label

#     def __len__(self):
#         return len(self.dataset)

# # 初始化数据集和数据加载器
# # 初始化模型
# image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
# model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
# root_dir = "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/imgs"  # 修改为你的图像数据集路径

# dataset = MyImageDataset(root_dir, image_processor)
# dataloader = DataLoader(dataset, batch_size=32)

# # 训练设置
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# optimizer = AdamW(model.parameters(), lr=1e-5)

# # 训练循环
# model.train()
# for epoch in range(3):  # 假设训练3个epoch
#     total_loss = 0
#     for inputs, _ in dataloader:
#         optimizer.zero_grad()
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         inputs['pixel_values'] = inputs['pixel_values'].squeeze(1)
#         outputs = model(**inputs)
#         loss = outputs.loss
        
#         total_loss += loss.item()
        
#         loss.backward()
#         optimizer.step()
#     avg_loss = total_loss / len(dataloader)
#     print(f'Training Loss: {avg_loss:.4f}')


# from transformers import AutoImageProcessor, ViTMAEForPreTraining
# from PIL import Image
# import requests

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
# model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

# inputs = image_processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# loss = outputs.loss
# mask = outputs.mask
# ids_restore = outputs.ids_restore

from .model_mae import mae_vit_base_patch16