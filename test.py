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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.params.cuda_no
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(OmegaConf.to_yaml(cfg))

    import torch

    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.loggers import CSVLogger

    from base import BaseModel
    from network import get_network
    from transform import get_transform

    sys.path.append('data')
    from data.utils import get_datamodule  # noqa: E402

    seed_everything(cfg.params.seed, workers=True)
    torch.set_float32_matmul_precision('high')

    if not cfg.params.dataset == 'imagenet16':
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

    network = get_network(cfg.model.name, cfg.dataset.num_channels, cfg.dataset.num_classes, cfg.model.timm_pretrained)
    log_flag = 'pretrained' if cfg.model.pretrained else 'from_scratch'
    base_logging_dir = os.path.join(cfg.logging.exp_dir, '{}/{}/{}'.format(cfg.params.dataset, cfg.model.name, log_flag))

    if cfg.model.timm_pretrained:
        model = BaseModel(cfg, dm, network)
    else:
        if cfg.logging.ckpt_path is None:
            path_part2 = 'version_{}/checkpoints/best_model.ckpt'.format(cfg.model.pretrained_version)
            model_path = os.path.join(base_logging_dir, path_part2)
        else:
            model_path = cfg.logging.ckpt_path

        model = BaseModel(cfg, dm, network)
        model.load_state_dict(torch.load(model_path)["state_dict"])

    logging_dir = os.path.join(base_logging_dir, cfg.params.protocol_name)

    trainer = Trainer(
        accelerator='gpu',
        callbacks=[],
        devices=[0],
        max_epochs=cfg.params.max_epochs,
        logger=CSVLogger(save_dir=logging_dir, name=''),
        deterministic=True
    )

    trainer.test(model)


if __name__ == "__main__":
    main()
