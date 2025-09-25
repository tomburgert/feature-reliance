from typing import Optional, List
from dataclasses import dataclass
from dataclasses import field


@dataclass
class GeneralParameter:
    seed: int = 1
    cuda_no: str = '2'
    max_epochs: int = 100
    num_workers: int = 4
    batch_size: int = 32
    dataset: str = 'food101'
    pin_memory: bool = True
    protocol_name: str = 'simple_test'
    slurm_bypass: bool = False


@dataclass
class DataAugmentationParameter:
    train_augmentations: str = 'none'
    test_augmentations: str = 'none'
    p: float = 1.0
    p_list: Optional[List[float]] = None
    resize_size: Optional[int] = 224
    grid_size: Optional[int] = 3
    gray_alpha: Optional[float] = 1.0
    bilateral_d: Optional[int] = 5
    sigma_color: Optional[int] = 75
    sigma_space: Optional[int] = 75
    nlmeans_h: Optional[int] = 5
    template_window_size: Optional[int] = 7
    search_window_size: Optional[int] = 21
    gaussian_k: Optional[int] = 11
    gaussian_sigma: Optional[float] = 1.0


@dataclass
class Dataset:
    domain: str = 'computer_vision'
    task: str = ''
    root_path: str = ''
    num_classes: Optional[int] = None
    num_channels: Optional[int] = None


@dataclass
class Network:
    name: str = 'resnet50'
    pretrained: bool = False
    pretrained_version: int = 0
    timm_pretrained: bool = False


@dataclass
class Optimizer:
    optimizer_name: str = 'adam_w'
    scheduler_name: str = 'cos_anneal_warm_restart'
    lr: float = 0.0001
    momentum: float = 0.9
    steps: int = 4
    weight_decay: float = 0.0001
    t_0: int = 10
    t_mult: int = 2
    eta_min: float = 0.000001


@dataclass
class Logging:
    exp_dir: str = ''
    ckpt_path: Optional[str] = None
    save_checkpoint : bool = False
    classwise_eval: bool = False
    track_test_probs: bool = False


@dataclass
class FeatureRelianceConfig:
    params: GeneralParameter = field(default_factory=GeneralParameter)
    dataset: Dataset = field(default_factory=Dataset)
    model: Network = field(default_factory=Network)
    dataaug: DataAugmentationParameter = field(default_factory=DataAugmentationParameter)
    optimizer: Optimizer = field(default_factory=Optimizer)
    logging: Logging = field(default_factory=Logging)
