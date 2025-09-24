from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Subset, Dataset

from torchvision import transforms
from torchvision import datasets as torch_datasets

from lightning.pytorch import LightningDataModule


class STL10Dataset(Dataset):
    def __init__(self, dataset_path: str, split: str = None, transform: Optional[transforms.Compose] = None):
        self.split = split
        self.transform = transform
        if split == 'train' or 'validation':
            self.dataset = torch_datasets.STL10(dataset_path, split='train')
            self.targets = np.array(self.dataset.labels)
            self.train_val_idx = np.arange(len(self.dataset))
            self.deterministic_train_val_split()

        if split == 'test':
            self.dataset = torch_datasets.STL10(dataset_path, split='test')
            self.targets = np.array(self.dataset.labels)
            self.test_idx = np.arange(len(self.dataset))

        self.initialize_split()

    def deterministic_train_val_split(self):
        self.train_idx, self.val_idx = train_test_split(
            self.train_val_idx, random_state=2024, test_size=0.10, train_size=0.90, stratify=self.targets
        )

    def initialize_split(self):
        if self.split == 'train':
            self.subset = Subset(self.dataset, self.train_idx)
        elif self.split == 'validation':
            self.subset = Subset(self.dataset, self.val_idx)
        elif self.split == 'test':
            self.subset = Subset(self.dataset, self.test_idx)

    def __getitem__(self, idx: int):
        image, target = self.subset[idx]
        target = torch.tensor(target)

        # some images (e.g. class car side) are grayscale, convert them to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.subset)


class STL10DataModule(LightningDataModule):
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
            self.train_dataset = STL10Dataset(
                dataset_path=self.dataset_path,
                split='train',
                transform=self.train_transform,
            )
            self.val_dataset = STL10Dataset(
                dataset_path=self.dataset_path,
                split='validation',
                transform=self.test_transform,
            )
        if stage == 'test' or stage is None:
            self.test_dataset = STL10Dataset(
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
