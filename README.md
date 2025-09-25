# Feature Reliance: Official Implementation

Official implementation of  
**“ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression”**  - Tom Burgert, Oliver Stoll, Paolo Rota, Begüm Demir | 
NeurIPS 2025 (oral)  
[Paper on arXiv](https://arxiv.org/abs/2509.20234)

## Abstract / Description

This repository provides code to reproduce the experiments in the paper. The work revisits the hypothesis that CNNs are inherently biased toward texture. It proposes a domain-agnostic framework for **controlled suppression** of cues (shape, texture, color) and quantifies feature reliance. Our results show that CNNs are not inherently texture-biased but mainly depend on local shape features. Extending the analysis across domains, we find systematic differences: vision models lean on shape, medical imaging on color, and remote sensing on texture.

## Installation

Install dependencies with pip. By default, the commands below install CPU wheels.  
For GPU support, uncomment the lines for CUDA 12.1.  
For other CUDA versions, please check the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

```bash
# CPU installation
pip install -r requirements.txt

# GPU installation (CUDA 12.1)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data Preparation

### Computer Vision Datasets
- **Caltech101**, **Flowers102**, **OxfordPet**, and **STL10** are included in `torchvision.datasets` and will be automatically downloaded when first used.  
- **ImageNet**: use the official ILSVRC 2012 split.

### Medical Imaging Datasets
- We use the `224×224` release from **MedMNIST**.  
- The [MedMNIST repository](https://github.com/MedMNIST/MedMNIST/tree/main) is integrated via the `medmnist` Python package (already included in `requirements.txt`), which handles dataset download in the same way as `torchvision.datasets`.

### Remote Sensing Datasets
- **AID**: [Hugging Face link](https://huggingface.co/datasets/blanchon/AID)  
- **RSD46-WHU**: [Hugging Face link](https://huggingface.co/datasets/jonathan-roberts1/RSD46-WHU/tree/main/data)  
  - After download, set the the path for base_path in the script and run: `scripts/preprocess_rsd46whu.py`
- **PatternNet**: [Download here](https://sites.google.com/view/zhouwx/dataset)  
- **UCMerced**: [Download here](http://weegee.vision.ucmerced.edu/datasets/landuse.html)  
- **DeepGlobe**: [Kaggle link](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)  
  - After download, set the the paths for source_path and destination_path in the script and run: `scripts/preprocess_deepglobe.py`
  - This script converts DeepGlobe into a **multi-label classification dataset**.

## Usage

### Training

To train a model, run:

```bash
python3 training.py \
  logging.exp_dir=<logging_path> \
  params.cuda_no=2 \
  params.dataset=caltech101 \
  optimizer.optimizer_name=adam_w \
  optimizer.lr=0.001 \
  params.max_epochs=300 \
  dataaug.train_augmentations=randomresizedcrop_horizontalflip \
  dataaug.test_augmentations=resizecrop \
  optimizer.weight_decay=0.01 \
  optimizer.scheduler_name=step_lr
```

Replace <logging_path> with your desired output directory.
Datasets can be swapped (e.g., imagenet, flowers102, etc.).

### Feature Reliance Protocol

To evaluate feature reliance of ResNet-50 across all Computer Vision datasets:

```bash
python3 reliance_protocol.py \
  -d imagenet caltech101 flowers102 stl10 oxfordiiitpet \
  -m resnet50 \
  -l <logging_path>
```

Here:
- `-d` : specifies the datasets (space-separated).  
- `-m` : specifies the model architecture.  
- `-l` : sets the logging directory.  

Replace <logging_path> with your desired output directory.

## Citation

If you use this code, please cite:

```bibtex
@misc{burgert2025featurereliance,
  title        = {ImageNet‐trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression},
  author       = {Tom Burgert and Oliver Stoll and Paolo Rota and Begüm Demir},
  year         = {2025},
  eprint       = {2509.20234},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV}
}
```
