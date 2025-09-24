import sys
import yaml
import os

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from config import FeatureRelianceConfig


print(hydra)

cs = ConfigStore.instance()
cs.store(name="base_config", node=FeatureRelianceConfig)


def update_dataset_parameter(cfg, dataset):

    with open('conf/datasets.yaml', "r") as f:
        yaml_file = yaml.safe_load(f)

    cfg.dataset.task         = yaml_file[dataset]['task']
    cfg.dataset.root_path    = yaml_file[dataset]['root_path']
    cfg.dataset.lmdb_path    = yaml_file[dataset]['lmdb_path']
    cfg.dataset.labels_path  = yaml_file[dataset]['labels_path']
    cfg.dataset.train_csv    = yaml_file[dataset]['train_csv']
    cfg.dataset.val_csv      = yaml_file[dataset]['val_csv']
    cfg.dataset.test_csv     = yaml_file[dataset]['test_csv']
    cfg.dataset.num_classes  = yaml_file[dataset]['num_classes']
    cfg.dataset.num_channels = yaml_file[dataset]['num_channels']

    return cfg


def get_monitoring_metric(task):
    if task == 'single_label':
        metric = 'val_accmac'
    elif task == 'multi_label':
        metric = 'val_APmac'
    elif task == 'ordinal_regression':
        metric = 'val_accmac'
    elif task == 'binary_classification':
        metric = 'val_accmac'
    return metric


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):

    if cfg.params.slurm_bypass:
        print('Bypassing slurm, using GPU: {}'.format(cfg.params.cuda_no))
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.params.cuda_no
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(OmegaConf.to_yaml(cfg))

    import torch

    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.callbacks import ModelCheckpoint

    from base import BaseModel
    from network import get_network
    from transform import get_transform

    sys.path.append('data')
    from data.utils import get_datamodule  # noqa: E402

    seed_everything(cfg.params.seed, workers=True)
    torch.set_float32_matmul_precision('high')

    cfg = update_dataset_parameter(cfg, cfg.params.dataset)
    DataModule = get_datamodule(cfg.params.dataset)
    dm = DataModule(
        root_path=cfg.dataset.root_path,
        batch_size=cfg.params.batch_size,
        num_workers=cfg.params.num_workers,
        train_transform=get_transform(**cfg.dataaug, split='train', dataset=cfg.params.dataset),
        test_transform=get_transform(**cfg.dataaug, split='test', dataset=cfg.params.dataset)
    )
    dm.setup()

    callbacks = []
    if cfg.logging.save_checkpoint:
        metric = get_monitoring_metric(cfg.dataset.task)
        callbacks += [ModelCheckpoint(monitor=metric, filename='best_model', mode='max')]

    network = get_network(cfg.model.name, cfg.dataset.num_channels, cfg.dataset.num_classes, cfg.model.timm_pretrained)
    model = BaseModel(cfg, dm, network)
    if cfg.model.pretrained:
        log_flag = 'pretrained'

        if not cfg.model.timm_pretrained:
            imagenet_path_suffix = 'computer_vision/imagenet/{}/from_scratch/version_0/checkpoints/best_model.ckpt'.format(cfg.model.name)
            imagenet_path_full = os.path.join('/data/tomburgert/data/logs_feature_bias', imagenet_path_suffix)

            # Step 2: Load the pretrained checkpoint
            checkpoint = torch.load(imagenet_path_full)["state_dict"]

            # Step 3: Filter out mismatched final layer (e.g., classifier head)
            filtered_checkpoint = {
                k: v for k, v in checkpoint.items() 
                if not (k.endswith("fc.weight") or k.endswith("fc.bias") or 
                        k.endswith("classifier.weight") or k.endswith("classifier.bias"))
            }

            # Step 4: Load weights with `strict=False` to ignore missing keys
            model.load_state_dict(filtered_checkpoint, strict=False)
    else:
        log_flag = 'from_scratch'

    logging_dir = os.path.join(cfg.logging.exp_dir, '{}/{}/{}'.format(cfg.params.dataset, cfg.model.name, log_flag))

    trainer = Trainer(
        # accelerator='cpu',
        accelerator='gpu',
        callbacks=callbacks,
        devices=[0],
        # devices=1,
        enable_checkpointing=cfg.logging.save_checkpoint,
        max_epochs=cfg.params.max_epochs,
        logger=CSVLogger(save_dir=logging_dir, name=''),
        deterministic=True
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
