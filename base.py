import os
import numpy as np
import pandas as pd

import torch
import torch.optim as optim

from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AveragePrecision
from torchmetrics import CohenKappa

from timm.scheduler.cosine_lr import CosineLRScheduler

import pytorch_lightning as pl


class BaseModel(pl.LightningModule):
    def __init__(self, cfg, datamodule, network):
        super().__init__()
        self.cfg = cfg

        self.model = network
        self.save_hyperparameters('cfg')
        
        self.datamodule = datamodule
        self.criterion = self.init_criterion()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        metrics = self.init_metrics(self.cfg.dataset.num_classes)
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y        = batch
        logits      = self.forward(x)
        conv_logits = self.convert_logits(logits)
        loss        = self.criterion(logits, y)

        self.train_metrics.update(conv_logits, y.long())  # y should be long/int
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y        = batch
        logits      = self.forward(x)
        conv_logits = self.convert_logits(logits)
        loss        = self.criterion(logits, y)

        self.val_metrics.update(conv_logits, y.long())  # y should be long/int
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y      = batch
        logits    = self.forward(x)
        loss      = self.criterion(logits, y)
        output    = dict(y=y, loss=loss, logits=logits)
        self.test_step_outputs.append(output)
        return output

    def unpack_step_outputs(self, step_outs):
        y     = torch.cat(list(map(lambda x: x['y'], step_outs)), dim=0)
        logits = torch.cat(list(map(lambda x: x['logits'], step_outs)), dim=0)
        loss  = torch.stack(list(map(lambda x: x['loss'], step_outs)))
        return y, logits, loss

    def convert_logits(self, logits):
        if self.cfg.dataset.task == 'single_label':
            output = torch.argmax(logits, dim=1)
        elif self.cfg.dataset.task == 'multi_label':
            output = torch.sigmoid(logits)
        elif self.cfg.dataset.task == 'ordinal_regression':
            output = torch.argmax(logits, dim=1)
        elif self.cfg.dataset.task == 'binary_classification':
            probs = torch.sigmoid(logits)
            output = (probs > 0.5).float()
        return output

    def on_train_epoch_end(self):
        output = self.train_metrics.compute()
        self.log_dict(output)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        output = self.val_metrics.compute()
        self.log_dict(output)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        y, logits, loss = self.unpack_step_outputs(self.test_step_outputs)
        self.test_step_outputs.clear()
        converted_logits = self.convert_logits(logits)

        output = self.chunk_wise_metric_calculation(converted_logits, y.long(), self.test_metrics)
        self.log_metrics(loss, output, 'test')

        if self.cfg.logging.track_test_probs:
            probs = torch.softmax(logits, dim=1)
            self.track_probs(probs, y.long(), 'test')

    ########################
    # CRITERION & OPTIMIZER
    ########################

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # Standard optimizer step
        optimizer.step(closure=optimizer_closure)

        # Manual scheduler stepping
        if self.cfg.optimizer.scheduler_name == 'cosine':
            # Step timm's CosineLRScheduler with epoch if per-epoch stepping
            self.lr_scheduler.step(epoch)
        else:
            # Step standard PyTorch scheduler
            self.lr_scheduler.step()

    def configure_optimizers(self):
        if self.cfg.optimizer.optimizer_name == 'adam_w':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay
            )
        elif self.cfg.optimizer.optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.cfg.optimizer.lr,
                momentum=self.cfg.optimizer.momentum,
                weight_decay=self.cfg.optimizer.weight_decay
            )

        # Handle scheduler creation
        if self.cfg.optimizer.scheduler_name == 'cos_anneal_warm_restart':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=self.cfg.optimizer.t_0,
                T_mult=self.cfg.optimizer.t_mult,
                eta_min=self.cfg.optimizer.eta_min,
            )
        elif self.cfg.optimizer.scheduler_name == 'cosine':
            self.lr_scheduler = CosineLRScheduler(
                optimizer=optimizer,
                t_initial=self.cfg.params.max_epochs,     # Total epochs for cosine decay
                lr_min=0.0,
                warmup_t=5,                               # Warmup epochs
                warmup_lr_init=1e-6,                      # Warmup start lr
                warmup_prefix=True,
                cycle_limit=1                             # No restarts
            )
        elif self.cfg.optimizer.scheduler_name == 'one_cycle':
            self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.cfg.optimizer.lr,
                epochs=self.cfg.params.max_epochs,
                steps_per_epoch=int(len(self.datamodule.train_dataset) / self.cfg.params.batch_size) + 1,
                pct_start=0.3
            )
        elif self.cfg.optimizer.scheduler_name == 'step_lr':
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=int(self.cfg.params.max_epochs / self.cfg.optimizer.steps),
                gamma=0.1
            )
        return optimizer

    def init_criterion(self):
        if self.cfg.dataset.task == 'single_label':
            return torch.nn.CrossEntropyLoss()
        elif self.cfg.dataset.task == 'multi_label':
            return torch.nn.BCEWithLogitsLoss()
        elif self.cfg.dataset.task == 'ordinal_regression':
            return torch.nn.CrossEntropyLoss()
        elif self.cfg.dataset.task == 'binary_classification':
            return torch.nn.BCEWithLogitsLoss()

    #################
    # LOGGING MODULE
    #################

    def init_metrics(self, num_classes):
        metrics = {}
        if self.cfg.dataset.task == 'single_label':
            metrics.update({
                'accmac'        : Accuracy(num_classes=num_classes, task='multiclass', average='macro'),
                'accmic'        : Accuracy(num_classes=num_classes, task='multiclass', average='micro'),
                'f1mac'         : F1Score(num_classes=num_classes, task='multiclass', average='macro'),
                'precmac'       : Precision(num_classes=num_classes, task='multiclass', average='macro'),
                'recmac'        : Recall(num_classes=num_classes, task='multiclass', average='macro'),
            })
            if self.cfg.logging.classwise_eval:
                metrics.update({
                    'accclasses'    : Accuracy(num_classes=num_classes, task='multiclass', average=None),
                    'f1classes'     : F1Score(num_classes=num_classes, task='multiclass', average=None),
                    'precclasses'   : Precision(num_classes=num_classes, task='multiclass', average=None),
                    'recclasses'    : Recall(num_classes=num_classes, task='multiclass', average=None),
                })
        elif self.cfg.dataset.task == 'multi_label':
            metrics.update({
                'APmic'     : AveragePrecision(num_labels=num_classes, task='multilabel', average='micro'),
                'APmac'     : AveragePrecision(num_labels=num_classes, task='multilabel', average='macro'),
                'f1mic'     : F1Score(num_labels=num_classes, task='multilabel', average='micro', threshold=0.5),
                'f1mac'     : F1Score(num_labels=num_classes, task='multilabel', average='macro', threshold=0.5),
            })
            if self.cfg.logging.classwise_eval:
                metrics.update({
                    'APclasses' : AveragePrecision(num_labels=num_classes, task='multilabel', average=None),
                    'f1classes' : F1Score(num_labels=num_classes, task='multilabel', average=None, threshold=0.5)
                })
        elif self.cfg.dataset.task == 'ordinal_regression':
            metrics.update({
                'accmac'        : Accuracy(num_classes=num_classes, task='multiclass', average='macro'),
                'accmic'        : Accuracy(num_classes=num_classes, task='multiclass', average='micro'),
                'kappa'         : CohenKappa(num_classes=num_classes, task="multiclass"),
            })
        elif self.cfg.dataset.task == 'binary_classification':
            metrics.update({
                'accmac'        : Accuracy(task='binary'),
                'f1mac'         : F1Score(task='binary'),
                'precmac'       : Precision(task='binary'),
                'recmac'        : Recall(task='binary'),
            })
        return MetricCollection(metrics)

    def chunk_wise_metric_calculation(self, probs, y, metric, chunk_size=1000):
        # Reset the metrics before accumulating new results
        metric.reset()

        # Process data in chunks
        for i in range(0, len(probs), chunk_size):
            probs_chunk = probs[i:i + chunk_size]
            y_chunk = y[i:i + chunk_size].long()
            metric.update(probs_chunk, y_chunk)

        # Compute the final metric result after all updates
        return metric.compute()

    def log_metrics(self, loss, output, log_str='train'):
        if self.cfg.logging.classwise_eval:
            if self.cfg.dataset.task == 'single_label':
                acc_classes = output.pop('{}_accclasses'.format(log_str))
                f1_classes = output.pop('{}_f1classes'.format(log_str))
                rec_classes = output.pop('{}_recclasses'.format(log_str))
                prec_classes = output.pop('{}_precclasses'.format(log_str))

                self.log('{}_loss'.format(log_str), loss.mean(), prog_bar=True)
                self.log_dict(output)
                self.log_list('{}_acc_cl'.format(log_str), acc_classes)
                self.log_list('{}_f1_cl'.format(log_str), f1_classes)
                self.log_list('{}_rec_cl'.format(log_str), rec_classes)
                self.log_list('{}_prec_cl'.format(log_str), prec_classes)

            elif self.cfg.dataset.task == 'multi_label':
                ap_classes = output.pop('{}_APclasses'.format(log_str))
                f1_classes = output.pop('{}_f1classes'.format(log_str))

                self.log('{}_loss'.format(log_str), loss.mean(), prog_bar=True)
                self.log_dict(output)
                self.log_list('{}_AP_cl'.format(log_str), ap_classes)
                self.log_list('{}_f1_cl'.format(log_str), f1_classes)
        else:
            if self.cfg.dataset.task == 'single_label':
                self.log('{}_loss'.format(log_str), loss.mean(), prog_bar=True)
                self.log_dict(output)

            elif self.cfg.dataset.task == 'multi_label':
                self.log('{}_loss'.format(log_str), loss.mean(), prog_bar=True)
                self.log_dict(output)

            elif self.cfg.dataset.task == 'ordinal_regression':
                self.log('{}_loss'.format(log_str), loss.mean(), prog_bar=True)
                self.log_dict(output)

            elif self.cfg.dataset.task == 'binary_classification':
                self.log('{}_loss'.format(log_str), loss.mean(), prog_bar=True)
                self.log_dict(output)

    def track_probs(self, probs, targets, log_str='train'):
        self.log_inferences(probs, log_str)

    def log_list(self, prefix, metric_list):
        for cl_idx, cl_value in zip(np.arange(self.cfg.dataset.num_classes), metric_list):
            self.log('{}{}'.format(prefix, cl_idx), cl_value, on_epoch=True, on_step=False)

    def log_inferences(self, probs, split):
        probs = list(map(list, probs.detach().cpu().numpy().astype(float)))
        df = pd.DataFrame(data={str(self.current_epoch): probs})
        path = os.path.join(self.logger.log_dir, 'tracking_{}.parquet'.format(split))
        if os.path.exists(path):
            df_old = pd.read_parquet(path)
            df = pd.concat([df_old, df], axis=1)
        df.to_parquet(path)

    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()
