from typing import Optional
import os
import glob

import numpy as np
from sklearn.model_selection import train_test_split

from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from lightning.pytorch import LightningDataModule


class PatternNetDataset(Dataset):
    def __init__(self, dataset_path: str, split: str = None, transform: Optional[transforms.Compose] = None):
        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, 'images')
        self.transform = transform
        self.split = split

        self.classes = sorted([d.name for d in os.scandir(self.image_dir) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.image_paths = []
        self.targets = []
        self.read_image_paths()

        self.all_idx = np.arange(len(self.image_paths))
        self.deterministic_train_val_test_split()
        self.initialize_split()

    def read_image_paths(self):
        for cl_name in self.classes:
            cls_folder = os.path.join(self.image_dir, cl_name)
            for ext in ('*.jpg', '*.png', '*.tif'):
                for img_path in glob.glob(os.path.join(cls_folder, ext)):
                    self.image_paths.append(img_path)
                    self.targets.append(self.class_to_idx[cl_name])

        self.image_paths = np.array(self.image_paths)
        self.targets = np.array(self.targets)

    def deterministic_train_val_test_split(self):
        self.train_val_idx, self.test_idx = train_test_split(
            self.all_idx, random_state=2024, test_size=0.15, train_size=0.85, stratify=self.targets
        )
        self.train_idx, self.val_idx = train_test_split(
            self.train_val_idx, random_state=2024, test_size=0.1765, train_size=0.8235, stratify=self.targets[self.train_val_idx]
        )

    def initialize_split(self):
        if self.split is None:
            self.subset_idx = self.all_idx
        elif self.split == 'train':
            self.subset_idx = self.train_idx
        elif self.split == 'validation':
            self.subset_idx = self.val_idx
        elif self.split == 'test':
            self.subset_idx = self.test_idx

    def __getitem__(self, idx):
        full_idx = self.subset_idx[idx]
        img_path = self.image_paths[full_idx]
        target = self.targets[full_idx]

        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        target = torch.tensor(target)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.subset_idx)


class PatternNetDataModule(LightningDataModule):
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
            self.train_dataset = PatternNetDataset(
                dataset_path=self.dataset_path,
                split='train',
                transform=self.train_transform,
            )
            self.val_dataset = PatternNetDataset(
                dataset_path=self.dataset_path,
                split='validation',
                transform=self.test_transform,
            )
        if stage == 'test' or stage is None:
            self.test_dataset = PatternNetDataset(
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
