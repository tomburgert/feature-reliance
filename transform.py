from typing import List, Tuple, Union, Optional

import cv2
import numpy as np

import albumentations as A

from torchvision import transforms


class CenterCrop(object):
    """Constant Transformation. PyTorch Wrapper for Albumentation VerticalFlip."""
    def __init__(self, size: Tuple[int, int], p: float = 1):
        self.CenterCrop = A.CenterCrop(size[0], size[1], p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.CenterCrop(image=image)['image']


class HorizontalFlip(object):
    """Constant Transformation. PyTorch Wrapper for Albumentation VerticalFlip."""
    def __init__(self, p: float = 0.5):
        self.HorizontalFlip = A.HorizontalFlip(p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.HorizontalFlip(image=image)['image']


class RandomResizedCrop:
    """Range Transformation."""
    def __init__(
        self,
        resize_size: Union[int, Tuple[int, int]] = (120, 120),
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio : Tuple[float, float] = (0.75, 1.3333333333333333),
        p: float = 1.0
    ):
        h, w = (resize_size, resize_size) if type(resize_size) != tuple else resize_size
        self.RandomResizedCrop = A.RandomResizedCrop(h, w, scale=scale, ratio=ratio, p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.RandomResizedCrop(image=image)['image']


class Resize:
    """Range Transformation. PyTorch Wrapper for Albumentation Flip."""
    def __init__(self, size: Tuple[int, int]):
        self.Resize = A.Resize(size[0], size[1])

    def __call__(self, image: np.array) -> np.array:
        return self.Resize(image=image)['image']


class ContinousGrayScale:
    def __init__(self, alpha: float = 1.0, p: float = 1.0):
        """
        Args:
            alpha (float): Strength of grayscale effect (0.0 = original, 1.0 = fully grayscale).
            p (float): Probability of applying the transformation.
        """
        assert 0.0 <= alpha <= 1.0, "Alpha must be in the range [0, 1]."
        self.alpha = alpha
        self.p = p

    def apply(self, img_np: np.ndarray) -> np.ndarray:
        """
        Args:
            img_np (np.ndarray): RGB image as (H, W, 3), dtype uint8 or float32 [0, 1] or [0, 255]
        Returns:
            np.ndarray: Blended image with grayscale effect.
        """
        is_uint8 = img_np.dtype == np.uint8

        # Convert to uint8 if needed for OpenCV
        if not is_uint8:
            img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        else:
            img_uint8 = img_np

        # OpenCV RGB → Gray → RGB
        gray_1c = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)              # shape: (H, W)
        gray_3c = cv2.cvtColor(gray_1c, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.

        # Normalize original image
        img_float = img_uint8.astype(np.float32) / 255.

        # Blend
        blended = (1 - self.alpha) * img_float + self.alpha * gray_3c

        # Return in original dtype
        if is_uint8:
            return np.clip(blended * 255, 0, 255).astype(np.uint8)
        else:
            return blended

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return self.apply(img_np)
        return img_np


class ChannelShuffle:
    def __init__(self, p: float = 1.0):
        self.p = p

    def apply(self, img_np: np.array) -> np.array:
        """
        Randomly shuffle the channels of an image with any number of channels.
        Args:
            img_np (numpy array): Image with shape (H, W, C).
        Returns:
            numpy array: Image with shuffled channels.
        """
        num_channels = img_np.shape[2]
        shuffled_indices = np.random.permutation(num_channels)  # Random permutation of channels
        return img_np[:, :, shuffled_indices]

    def __call__(self, img_np: np.array) -> np.array:
        if np.random.rand() < self.p:
            img_np = self.apply(img_np)
        return img_np


class BilateralFilter:
    def __init__(self, d: int = 5, sigma_color: int = 75, sigma_space: int = 75, p: float = 1.0):
        """
        Initialize bilateral filter parameters.
        Args:
            d (int): Diameter of pixel neighborhood.
            sigmaColor (float): Filter sigma in color space.
            sigmaSpace (float): Filter sigma in coordinate space.
        """
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.p = p

    def apply(self, img_np: np.array) -> np.array:
        img_np = cv2.bilateralFilter(img_np, d=self.d, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)
        # img_np = cv2.bilateral_filter_numpy(img_np, d=self.d, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)
        return img_np

    def __call__(self, img_np: np.array) -> np.array:
        if np.random.rand() < self.p:
            img_np = self.apply(img_np)
        # img_np = self.apply(img_np)
        return img_np


class FastNLMeansDenoising:
    def __init__(self, h: int = 5, template_window_size: int = 7, search_window_size: int = 21, p: float = 1.0):
        """
        Initialize fastNlMeansDenoising parameters.
        Args:
            d (int): Diameter of pixel neighborhood.
            sigmaColor (float): Filter sigma in color space.
            sigmaSpace (float): Filter sigma in coordinate space.
        """
        self.h = h
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size
        self.p = p

    def apply(self, img_np: np.array) -> np.array:
        img_np = cv2.fastNlMeansDenoising(
            img_np,
            h=self.h,
            templateWindowSize=self.template_window_size,
            searchWindowSize=self.search_window_size
        )
        return img_np

    def __call__(self, img_np: np.array) -> np.array:
        if np.random.rand() < self.p:
            img_np = self.apply(img_np)
        return img_np


class GaussianBlur:
    def __init__(self, k: int = 5, sigma: float = 1.0, p: float = 1.0):
        self.k = k
        self.sigma = sigma
        self.p = p

    def apply(self, img_np: np.array) -> np.array:
        img_np = cv2.GaussianBlur(img_np, (self.k, self.k), sigmaX=self.sigma)
        return img_np

    def __call__(self, img_np: np.array) -> np.array:
        img_np = self.apply(img_np)
        return img_np


class PatchShuffle:
    def __init__(self, grid_size: int = 3, p: float = 1.0):
        """
        Initialize GridShuffle transform.
        Args:
            grid_size (int): Number of grid cells along each axis.
            p (float): Probability of applying the transformation.
        """
        self.GridShuffle = A.RandomGridShuffle(grid=(grid_size, grid_size), p=p)
        self.p = p

    def apply(self, img_np: np.array) -> np.array:
        img_np = self.GridShuffle(image=img_np)['image']
        return img_np

    def __call__(self, img_np: np.array) -> np.array:
        if np.random.rand() < self.p:
            img_np = self.apply(img_np)
        return img_np


class PatchRotation:
    def __init__(self, grid_size: int = 3, p: float = 1.0, output_size: tuple = (224, 224), interpolation=cv2.INTER_LINEAR):
        """
        Args:
            grid_size (int): Number of patches along each axis.
            p (float): Probability of applying the transformation.
            output_size (tuple): Final output image size (H, W).
            interpolation: OpenCV interpolation method.
        """
        self.grid_size = grid_size
        self.p = p
        self.output_size = output_size
        self.interpolation = interpolation

    def _resize_to_square_grid_if_needed(self, img_np: np.ndarray) -> tuple:
        h, w, c = img_np.shape

        divisible_h = h % self.grid_size == 0
        divisible_w = w % self.grid_size == 0
        is_square = h == w

        if divisible_h and divisible_w and is_square:
            return img_np, (h, w)

        # Compute square side length that's divisible by grid_size
        max_dim = max(h, w)
        square_dim = ((max_dim + self.grid_size - 1) // self.grid_size) * self.grid_size
        resized = cv2.resize(img_np, (square_dim, square_dim), interpolation=self.interpolation)
        return resized, (h, w)

    def apply(self, img_np: np.ndarray) -> np.ndarray:
        img_np, _ = self._resize_to_square_grid_if_needed(img_np)
        h, w, c = img_np.shape
        patch_h = h // self.grid_size
        patch_w = w // self.grid_size  # == patch_h

        patches = []
        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                top = i * patch_h
                left = j * patch_w
                patch = img_np[top:top + patch_h, left:left + patch_w, :]
                angle = np.random.choice([0, 90, 180, 270])
                rotated = np.rot90(patch, k=angle // 90)
                row.append(rotated)
            patches.append(row)

        rows = [np.concatenate(patch_row, axis=1) for patch_row in patches]
        rotated_img = np.concatenate(rows, axis=0)

        # Resize to output size
        if rotated_img.shape[:2] != self.output_size:
            rotated_img = cv2.resize(rotated_img, (self.output_size[1], self.output_size[0]), interpolation=self.interpolation)

        return rotated_img

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return self.apply(img_np)
        # Resize to output size even if no patch rotation is applied
        return img_np


class CutOut(object):
    """Range Transformation."""
    def __init__(self, max_edge: float = 0.7, min_edge: float = 0.2, p: float = 0.5):
        self.CutOut = A.CoarseDropout(
            max_holes=1,
            max_height=max_edge,
            max_width=max_edge,
            min_height=min_edge,
            min_width=min_edge,
            p=p)

    def __call__(self, image: np.array) -> np.array:
        return self.CutOut(image=image)['image']


class CutMix:
    def __init__(self, alpha: float = 1.0, p: float = 1.0):
        self.alpha = alpha
        self.p = p

    def rand_bbox(self, size, lam):
        H, W = size[0], size[1]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, float]:
        if np.random.rand() > self.p:
            return img1, 1.0  # no mix, return image unchanged

        lam = np.random.beta(self.alpha, self.alpha)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(img1.shape, lam)
        mixed = img1.copy()
        mixed[bby1:bby2, bbx1:bbx2, :] = img2[bby1:bby2, bbx1:bbx2, :]
        lam_adjusted = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img1.shape[0] * img1.shape[1]))
        return mixed, lam_adjusted


class MixUp:
    def __init__(self, alpha: float = 1.0, p: float = 1.0):
        self.alpha = alpha
        self.p = p

    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, float]:
        if np.random.rand() > self.p:
            return img1, 1.0  # no mix

        lam = np.random.beta(self.alpha, self.alpha)
        mixed = lam * img1.astype(np.float32) + (1 - lam) * img2.astype(np.float32)
        mixed = np.clip(mixed, 0, 255).astype(np.uint8) if img1.dtype == np.uint8 else mixed
        return mixed, lam


def get_dataset_statistics(dataset):
    if dataset in ['imagenet', 'oxfordiiitpet', 'caltech101', 'flowers102', 'stl10', 'imagenet16']:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    elif dataset == 'bloodmnist':
        mean = [0.796, 0.659, 0.696]
        std  = [0.226, 0.259, 0.096]
    elif dataset == 'chestmnist':
        mean = [0.497, 0.497, 0.497]
        std  = [0.247, 0.247, 0.247]
    elif dataset == 'dermamnist':
        mean = [0.763, 0.538, 0.561]
        std  = [0.136, 0.158, 0.176]
    elif dataset == 'pathmnist':
        mean = [0.740, 0.532, 0.705]
        std  = [0.165, 0.217, 0.157]
    elif dataset == 'retinamnist':
        mean = [0.394, 0.241, 0.145]
        std  = [0.323, 0.210, 0.151]
    elif dataset == 'aid':
        mean = [0.397, 0.408, 0.368]
        std  = [0.216, 0.194, 0.191]
    elif dataset == 'patternnet':
        mean = [0.359, 0.360, 0.319]
        std  = [0.195, 0.185, 0.178]
    elif dataset == 'rsd46whu':
        mean = [0.378, 0.419, 0.373]
        std  = [0.207, 0.178, 0.171]
    elif dataset == 'ucmerced':
        mean = [0.483, 0.489, 0.450]
        std  = [0.217, 0.201, 0.195]
    elif dataset == 'deepglobe':
        mean = [0.407, 0.380, 0.283]
        std  = [0.150, 0.118, 0.108]
    return mean, std


def get_transform(
    train_augmentations: str,
    test_augmentations: str,
    p: float,
    p_list: Optional[List[int]],
    resize_size: Optional[int],
    grid_size: Optional[int],
    gray_alpha: Optional[float],
    bilateral_d: Optional[int],
    sigma_color: Optional[int],
    sigma_space: Optional[int],
    nlmeans_h: Optional[int],
    template_window_size: Optional[int],
    search_window_size: Optional[int],
    gaussian_k: Optional[int],
    gaussian_sigma: Optional[float],
    split: str = 'train',
    dataset: str = 'imagenet'
    
) -> List[object]:

    transform_names = list(train_augmentations.split('_')) if split == 'train' else list(test_augmentations.split('_'))
    compose = []

    if p_list is not None:
        assert len(transform_names) == len(p_list), "if p_list is provided, has to have same lentgh as num. augmentations"
        augment_ps = p_list
    else:
        augment_ps = [p] * len(transform_names)

    for transform_name, p in zip(transform_names, augment_ps):
        if 'grayscale' == transform_name:
            compose += [ContinousGrayScale(alpha=gray_alpha, p=p)]
        if 'channelshuffle' == transform_name:
            compose += [ChannelShuffle(p=p)]
        if 'bilateral' == transform_name:
            compose += [BilateralFilter(d=bilateral_d, sigma_color=sigma_color, sigma_space=sigma_space, p=p)]
        if 'gaussianblur' in transform_name:
            compose += [GaussianBlur(k=gaussian_k, sigma=gaussian_sigma, p=p)]
        if 'nlmeans' == transform_name:
            compose += [FastNLMeansDenoising(
                h=nlmeans_h,
                template_window_size=template_window_size,
                search_window_size=search_window_size,
                p=p
            )]
        if 'patchshuffle' == transform_name:
            compose += [PatchShuffle(grid_size=grid_size, p=p)]
        if 'patchrotation' == transform_name:
            compose += [PatchRotation(grid_size=grid_size, p=p)]

        if 'randomresizedcrop' == transform_name:
            compose += [RandomResizedCrop((resize_size, resize_size), scale=(0.3, 1.0))]
        if 'horizontalflip' == transform_name:
            compose += [HorizontalFlip()]

        if 'resize':
            compose += [Resize((resize_size, resize_size))]

        if 'resizecrop' == transform_name:
            compose += [Resize((256, 256)), CenterCrop((224, 224))]

        if 'crop' == transform_name:
            compose += [CenterCrop((224, 224))]

        if 'cutout' == transform_name:
            compose += [CutOut(p=p)]      

    mean, std = get_dataset_statistics(dataset)

    compose += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    return transforms.Compose(compose)
