from typing import Optional

import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision import datasets as torch_datasets

from lightning.pytorch import LightningDataModule


class ImageNetDataset(Dataset):
    def __init__(self, dataset_path: str, split: str = None, transform: Optional[transforms.Compose] = None):
        self.split = split
        self.transform = transform

        if split == 'train':
            self.dataset = torch_datasets.ImageNet(dataset_path, split='train')
            self.targets = np.array(self.dataset.targets)

        if split == 'validation':
            self.dataset = torch_datasets.ImageNet(dataset_path, split='val')
            self.targets = np.array(self.dataset.targets)

        if split == 'test':
            self.dataset = torch_datasets.ImageNet(dataset_path, split='val')
            self.targets = np.array(self.dataset.targets)

    def __getitem__(self, idx: int):
        image, target = self.dataset[idx]
        target = torch.tensor(target)

        # some images (e.g. class car side) are grayscale, convert them to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.dataset)


class ImageNetDataModule(LightningDataModule):
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
            self.train_dataset = ImageNetDataset(
                dataset_path=self.dataset_path,
                split='train',
                transform=self.train_transform,
            )
            self.val_dataset = ImageNetDataset(
                dataset_path=self.dataset_path,
                split='validation',
                transform=self.test_transform,
            )
        if stage == 'test' or stage is None:
            self.test_dataset = ImageNetDataset(
                dataset_path=self.dataset_path,
                split='test',
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
