from typing import Optional

import numpy as np

import torch

from torchvision import transforms

from lightning.pytorch import LightningDataModule

from medmnist import DermaMNIST


class DermaMNISTDataset():
    def __init__(self, dataset_path: str, split: str = None, transform: Optional[transforms.Compose] = None):
        self.dataset = DermaMNIST(root=dataset_path, split=split, download=False, size=224)
        self.split = split
        self.transform = transform
        self.targets = self.dataset.labels[:, 0]

    def __getitem__(self, idx: int):
        image, target = self.dataset[idx]
        target = torch.tensor(target[0])

        # some images (e.g. class car side) are grayscale, convert them to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.dataset)


class DermaMNISTDataModule(LightningDataModule):
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
            self.train_dataset = DermaMNISTDataset(
                dataset_path=self.dataset_path,
                split='train',
                transform=self.train_transform,
            )
            self.val_dataset = DermaMNISTDataset(
                dataset_path=self.dataset_path,
                split='val',
                transform=self.test_transform,
            )
        if stage == 'test' or stage is None:
            self.test_dataset = DermaMNISTDataset(
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
