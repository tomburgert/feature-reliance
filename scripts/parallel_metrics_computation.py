from typing import Tuple

import os
import cv2
import numpy as np
from PIL import Image
# from torchvision.transforms import Resize, CenterCrop
from skimage import color
from skimage.filters import sobel
from skimage.util import view_as_windows
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from functools import partial

from tqdm import tqdm
from concurrent.futures import as_completed

import albumentations as A


class Resize:
    """Range Transformation. PyTorch Wrapper for Albumentation Flip."""
    def __init__(self, size: Tuple[int, int]):
        self.Resize = A.Resize(size[0], size[1])

    def __call__(self, image: np.array) -> np.array:
        return self.Resize(image=image)['image']


# Texture suppression metrics
def local_variance_map(image_gray, window_size=15):
    windows = view_as_windows(image_gray, (window_size, window_size))
    var_map = np.var(windows, axis=(-2, -1))
    return np.mean(var_map)


def high_freq_energy(image_gray, radius=20):
    f = np.fft.fft2(image_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift) ** 2
    h, w = magnitude_spectrum.shape
    center = (h // 2, w // 2)
    y, x = np.ogrid[:h, :w]
    mask = ((x - center[1])**2 + (y - center[0])**2) >= radius**2
    high_freq_power = magnitude_spectrum[mask].sum()
    total_power = magnitude_spectrum.sum()
    return high_freq_power / total_power


def mean_gradient(image_gray):
    grad = sobel(image_gray)
    return np.mean(grad)


def sobel_cv(img, ksize=3):
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    return np.hypot(dx, dy)


def gradient_correlation(original_gray, filtered_gray, name, label):
    grad_orig = np.gradient(original_gray)
    grad_filt = np.gradient(filtered_gray)

    flat_x1 = grad_orig[0].flatten()
    flat_x2 = grad_filt[0].flatten()
    flat_y1 = grad_orig[1].flatten()
    flat_y2 = grad_filt[1].flatten()

    if np.std(flat_x1) == 0 or np.std(flat_x2) == 0:
        print("Warning: zero std in grad_x", np.std(flat_x1), np.std(flat_x2))
        print(f"Zero std for image={name}, transform={label}")
    if np.std(flat_y1) == 0 or np.std(flat_y2) == 0:
        print("Warning: zero std in grad_y", np.std(flat_y1), np.std(flat_y2))
        print(f"Zero std for image={name}, transform={label}")

    corr_x = np.corrcoef(flat_x1, flat_x2)[0, 1]
    corr_y = np.corrcoef(flat_y1, flat_y2)[0, 1]

    return (corr_x + corr_y) / 2


def process_image(name, kernel_sizes, eval_k):
    results = []

    # Load and preprocess image
    img_path = os.path.join(original_path, name)
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    resize = Resize((224, 224))
    img_np = resize(img_np)

    original_gray = color.rgb2gray(img_np)
    original_local_var = local_variance_map(original_gray, window_size=eval_k)
    original_high_freq_e = high_freq_energy(original_gray, radius=eval_k)

    def evaluate_transform(filtered, label, p1=None, p2=None, k=None):
        filtered_gray = color.rgb2gray(filtered)
        return {
            'image': name,
            'transform': label,
            'param1': p1,
            'param2': p2,
            'kernel': k,
            'local_variance': local_variance_map(filtered_gray, window_size=eval_k) / original_local_var,
            'high_freq_energy': high_freq_energy(filtered_gray, radius=eval_k) / original_high_freq_e,
            'gradient_corr': gradient_correlation(original_gray, filtered_gray, name, label),
        }

    # Bilateral Filter
    for sigma_color in [50, 80, 110, 140, 170, 200]:
        for k in kernel_sizes:
            filtered = cv2.bilateralFilter(img_np, d=k, sigmaColor=sigma_color, sigmaSpace=75)
            results.append(evaluate_transform(filtered, 'bilateral', sigma_color, 75, k))

    # Gaussian Blur
    for sigma in [0.66, 1.0, 1.33, 1.66, 2.0, 2.33]:
        for k in kernel_sizes:
            k_odd = k if k % 2 == 1 else k + 1
            filtered = cv2.GaussianBlur(img_np, (k_odd, k_odd), sigmaX=sigma)
            results.append(evaluate_transform(filtered, 'gaussian', sigma, k_odd, k_odd))

    # Median Filter
    for k in kernel_sizes:
        k_odd = k if k % 2 == 1 else k + 1
        filtered = cv2.medianBlur(img_np, k_odd)
        results.append(evaluate_transform(filtered, 'median', None, None, k_odd))

    # Non-Local Means
    for h in [5, 10, 15, 20, 25]:
        for k in kernel_sizes:
            filtered = cv2.fastNlMeansDenoising(img_np, h=h, templateWindowSize=k, searchWindowSize=21)
            results.append(evaluate_transform(filtered, 'nlmeans', h, 21, k))

    # Box Filter
    for k in kernel_sizes:
        filtered = cv2.blur(img_np, (k, k))
        results.append(evaluate_transform(filtered, 'box', None, None, k))

    return results


def benchmark_transformations_metric_static(sampled_names):
    kernel_sizes = [5, 7, 9, 11, 13, 15]
    eval_k = 11

    with ProcessPoolExecutor(32) as executor:
        func = partial(process_image, kernel_sizes=kernel_sizes, eval_k=eval_k)
        futures = [executor.submit(func, name) for name in sampled_names]
        all_results_nested = []

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            all_results_nested.append(future.result())

    # Flatten list of lists
    all_results = [row for result in all_results_nested for row in result]
    return pd.DataFrame(all_results)


base_path = '/data/tomburgert/data/datasets/imagenet16'
original_path = os.path.join(base_path, 'original')
image_names = os.listdir(original_path)
sampled_names = image_names[:-1]

df = benchmark_transformations_metric_static(sampled_names)

df.to_parquet('/data/tomburgert/data/additional_data/suppression_metrics_ablation.parquet')
