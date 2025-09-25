from computer_vision.caltech import Caltech101DataModule
from computer_vision.flowers import Flowers102DataModule
from computer_vision.oxfordpet import OxfordIIITPetDataModule
from computer_vision.stl10 import STL10DataModule
from computer_vision.imagenet import ImageNetDataModule
from computer_vision.imagenet16 import ImageNet16DataModule

from medical_imaging.bloodmnist import BloodMNISTDataModule
from medical_imaging.chestmnist import ChestMNISTDataModule
from medical_imaging.dermamnist import DermaMNISTDataModule
from medical_imaging.pathmnist import PathMNISTDataModule
from medical_imaging.retinamnist import RetinaMNISTDataModule

from remote_sensing.rsd46whu import RSD46WHUDataModule
from remote_sensing.aid import AIDDataModule
from remote_sensing.ucmerced import UCMercedDataModule
from remote_sensing.patternnet import PatternNetDataModule
from remote_sensing.deepglobe import DeepGlobeDataModule


def get_datamodule(dataset):
    if dataset == 'caltech101':
        return Caltech101DataModule
    elif dataset == 'flowers102':
        return Flowers102DataModule
    elif dataset == 'oxfordiiitpet':
        return OxfordIIITPetDataModule
    elif dataset == 'stl10':
        return STL10DataModule
    elif dataset == 'imagenet':
        return ImageNetDataModule
    elif dataset == 'imagenet16':
        return ImageNet16DataModule

    elif dataset == 'bloodmnist':
        return BloodMNISTDataModule
    elif dataset == 'chestmnist':
        return ChestMNISTDataModule
    elif dataset == 'dermamnist':
        return DermaMNISTDataModule
    elif dataset == 'pathmnist':
        return PathMNISTDataModule
    elif dataset == 'retinamnist':
        return RetinaMNISTDataModule

    elif dataset == 'rsd46whu':
        return RSD46WHUDataModule
    elif dataset == 'aid':
        return AIDDataModule
    elif dataset == 'ucmerced':
        return UCMercedDataModule
    elif dataset == 'patternnet':
        return PatternNetDataModule
    elif dataset == 'deepglobe':
        return DeepGlobeDataModule

    else:
        raise ValueError(f'{dataset} not implemented')
