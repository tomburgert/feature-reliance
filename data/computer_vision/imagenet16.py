from typing import Optional
import os

import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision import datasets, transforms

from lightning.pytorch import LightningDataModule

from PIL import Image


# hack: target look up for non-imagenet images
name2target = {
    'airplane': 404,
    'knife': 499,
    'oven': 766
}


class ImageNet16Dataset(Dataset):
    def __init__(self, dataset_path: str, split: str = None, transform: Optional[transforms.Compose] = None):
        self.dataset_path = dataset_path
        self.image_names = os.listdir(dataset_path)
        self.image_names = [image for image in self.image_names if image.split('_')[1] != 'checkpoints']
        self.transform = transform
        imagenet = datasets.ImageNet(root='/data_read_only/imagenet2012/imagenet_pytorch', split='val')
        self.wnid_to_idx = imagenet.wnid_to_idx

    def __getitem__(self, idx: int):
        image_name = self.image_names[idx]
        wnid = image_name.split('_')[1]
        if wnid.endswith(".JPEG"):
            base_name = image_name.split('_')[0]
            target = name2target[base_name]
        else:
            target = self.wnid_to_idx[wnid]
        image_path = os.path.join(self.dataset_path, image_name)
        image = Image.open(image_path)
        target = torch.tensor(target)

        # some images (e.g. class car side) are grayscale, convert them to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.image_names)


class ImageNet16DataModule(LightningDataModule):
    def __init__(
        self,
        root_path: str,
        batch_size: int,
        num_workers: int,
        train_transform: Optional[transforms.Compose] = None,
        test_transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()
        self.dataset_path = root_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_transform = train_transform
        self.test_transform = test_transform

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageNet16Dataset(
                dataset_path=self.dataset_path,
                split='None',
                transform=self.train_transform,
            )
            self.val_dataset = ImageNet16Dataset(
                dataset_path=self.dataset_path,
                split='None',
                transform=self.test_transform,
            )
        if stage == 'test' or stage is None:
            self.test_dataset = ImageNet16Dataset(
                dataset_path=self.dataset_path,
                split='None',
                transform=self.test_transform,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
