# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import torch

from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from timm.data import create_transform

from masking_generator import RandomMaskingGenerator
from dataset_folder import ImageFolder
from torch.utils.data import Dataset
import cv2
from sklearn.model_selection import train_test_split


class DataAugmentationForMAE(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.masked_position_generator = RandomMaskingGenerator(
            args.window_size, args.mask_ratio
        )

    def __call__(self, image):
        return self.transform(image), self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr
    
# TH5 dataset
class TH(Dataset):
    def __init__(self,transform=None):
        labels = {

        }
        with open("/home/qiming/Desktop/code/AI-for-Chemistry/TH5/th5_dataset/labels_train_hb_class.txt", "r") as fr:
            content = fr.readlines()
            for i in content:
                labels[i.split(",")[0]] = float(i.split(",")[-1])

        image_path = "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/th5_dataset/pretrain/pretrain_img"
        self.img_path = []
        self.hb_labels = []
        for key, value in labels.items():
               self.img_path.append(f"{image_path}/{key}.jpg")
               self.hb_labels.append(value)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_path[idx])
        image = cv2.resize(image, (224, 224))
        label = int(self.hb_labels[idx])
        
        image = torch.as_tensor(image).float().permute(2,0,1)
        label = torch.as_tensor(label).long()        

        return image, label
    
class TH4(Dataset):
    def __init__(self,transform=None):
        labels = {

        }
        with open("/home/qiming/Desktop/code/AI-for-Chemistry/TH5/th5_dataset/labels_test_hb_class.txt", "r") as fr:
            content = fr.readlines()
            for i in content:
                labels[i.split(",")[0]] = float(i.split(",")[-1])

        image_path = "/home/qiming/Desktop/code/AI-for-Chemistry/TH5/th5_dataset/th4"
        self.img_path = []
        self.hb_labels = []
        for key, value in labels.items():
               self.img_path.append(f"{image_path}/{key}.jpg")
               self.hb_labels.append(value)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_path[idx])
        image = cv2.resize(image, (224, 224))
        label = int(self.hb_labels[idx])
        
        image = torch.as_tensor(image).float().permute(2,0,1)
        label = torch.as_tensor(label).long()        

        return image, label

# joint without TH4
class THData_joint(Dataset):
    def __init__(self, mode, transform=None):
        labels = {

        }
        with open("/home/qiming/Desktop/code/AI-for-Chemistry/TH5/th5_dataset/joint_labels_f2f.txt", "r") as fr:
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
            self.img_path, self.hb_labels, test_size=0.2, random_state=42)        
           

    def __len__(self):
        if self.mode == "train":
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.mode == "train":
            image = cv2.imread(self.train_data[idx])
            image = cv2.resize(image, (224, 224))
            label = int(self.train_labels[idx])
            
            image = torch.as_tensor(image).float().permute(2,0,1)
            label = torch.as_tensor(label).long()
        else:
            image = cv2.imread(self.test_data[idx])
            image = cv2.resize(image, (224, 224))
            label = int(self.test_labels[idx])
            
            image = torch.as_tensor(image).float().permute(2,0,1)
            label = torch.as_tensor(label).long()            

        return image, label


def build_pretraining_dataset(args):
    transform = DataAugmentationForMAE(args)
    print("Data Aug = %s" % str(transform))
    return ImageFolder(args.data_path, transform=transform)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    # elif args.data_set == "TH5":
    #     if is_train:
    #         dataset = TH(transform=transform)
    #     else:
    #         dataset = TH4(transform=transform)
    #     nb_classes = 4
    elif args.data_set == "TH5":
        if is_train:
            dataset = THData_joint(mode="train", transform=transform)
        else:
            dataset = THData_joint(mode="test", transform=transform)
        nb_classes = 4    
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
