from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Subset, Dataset

from torchvision import transforms
from torchvision import datasets as torch_datasets

from lightning.pytorch import LightningDataModule


class Caltech101Dataset(Dataset):
    def __init__(self, dataset_path: str, split: str = None, transform: Optional[transforms.Compose] = None):
        self.dataset = torch_datasets.Caltech101(dataset_path)
        self.split = split
        self.transform = transform
        self.targets = np.array(self.dataset.y)
        self.all_idx = np.arange(len(self.dataset))
        self.deterministic_train_val_test_split()
        self.initialize_split()

    def deterministic_train_val_test_split(self):
        self.train_val_idx, self.test_idx = train_test_split(
            self.all_idx, random_state=2024, test_size=0.15, train_size=0.85, stratify=self.targets
        )
        self.train_idx, self.val_idx = train_test_split(
            self.train_val_idx, random_state=2024, test_size=0.1765, train_size=0.8235, stratify=self.targets[self.train_val_idx]
        )

    def initialize_split(self):
        if self.split is None:
            self.subset = Subset(self.dataset, self.all_idx)
        elif self.split == 'train':
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


class Caltech101DataModule(LightningDataModule):
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
            self.train_dataset = Caltech101Dataset(
                dataset_path=self.dataset_path,
                split='train',
                transform=self.train_transform,
            )
            self.val_dataset = Caltech101Dataset(
                dataset_path=self.dataset_path,
                split='validation',
                transform=self.test_transform,
            )
        if stage == 'test' or stage is None:
            self.test_dataset = Caltech101Dataset(
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
