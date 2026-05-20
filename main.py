import os
import sys
import importlib

import torch
import pytorch_lightning as pl

from os import path as osp
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from metric import calculate_metrics, save_result_sen12, save_result_smile

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Config must contain a 'target' key")
    return get_obj_from_str(config["target"])(**config.get("params", {}))


class PlDataModule(pl.LightningDataModule):
    def __init__(self, train, val, test):
        super().__init__()
        self.train_config = train
        self.val_config = val
        self.test_config = test
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = instantiate_from_config(self.train_config)
            self.val_dataset = instantiate_from_config(self.val_config)
        if stage == "test" or stage is None:
            self.test_dataset = instantiate_from_config(self.test_config)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.train_config.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.val_config.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_config.batch_size,
            shuffle=False,
            num_workers=self.test_config.num_workers,
        )


class MetricsCallback(Callback):
    def __init__(self, data, device):
        super().__init__()
        self.data = data
        self.device = device

    def on_train_epoch_end(self, trainer, pl_module):
        # lr_ae, lr_disc = pl_module.lr_schedulers()
        # lr_ae.step()
        # lr_disc.step()
        with torch.no_grad():
            avg_ssim, avg_psnr, avg_mae, avg_sam = calculate_metrics(pl_module, self.data.train_dataloader(), self.device, is_train=True)
            pl_module.log("Metrics/train_ssim", avg_ssim, prog_bar=False, on_epoch=True)
            pl_module.log("Metrics/train_psnr", avg_psnr, prog_bar=False, on_epoch=True)
            pl_module.log("Metrics/train_mae", avg_mae, prog_bar=False, on_epoch=True)
            pl_module.log("Metrics/train_sam", avg_sam, prog_bar=False, on_epoch=True)
            os.system("cls" if sys.platform.startswith("win32") else "clear")

    def on_validation_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            avg_ssim, avg_psnr, avg_mae, avg_sam = calculate_metrics(pl_module, self.data.val_dataloader(), self.device)
            save_result_sen12(pl_module, self.data.val_dataset, "outputs", pl_module.global_step)
            # save_result_smile(pl_module, self.data.val_dataset, "outputs", pl_module.global_step)
            pl_module.log("Metrics/val_ssim", avg_ssim, prog_bar=True, on_epoch=True)
            pl_module.log("Metrics/val_psnr", avg_psnr, prog_bar=True, on_epoch=True)
            pl_module.log("Metrics/val_mae", avg_mae, prog_bar=True, on_epoch=True)
            pl_module.log("Metrics/val_sam", avg_sam, prog_bar=True, on_epoch=True)
            os.system("cls" if sys.platform.startswith("win32") else "clear")



if __name__ == "__main__":
    seed_everything(42)
    
    torch.set_float32_matmul_precision('high')
    
    config = OmegaConf.load("configs\SMILE_CR\HQD.yaml")
    model_config, data_config, logger_config = config.model, config.data, config.logger
    model = instantiate_from_config(model_config)
    data = instantiate_from_config(data_config)
    logger = instantiate_from_config(logger_config)

    data.setup(stage="fit")
    
    callbacks = []
    ckpt_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch}-{Metrics/val_psnr:.4f}",
        monitor="Metrics/val_psnr",
        auto_insert_metric_name=False,
        save_last=True,
        save_top_k=3,
        mode="max",
    )
    metric_callback = MetricsCallback(data, "cuda:0")
    lr_callback = LearningRateMonitor(logging_interval="step")
    es_callback = EarlyStopping(
        monitor="Metrics/val_psnr",
        patience=20,
        mode="max",
        verbose=True,
    )
    
    callbacks.extend([ckpt_callback, metric_callback, lr_callback, es_callback])
    
    trainer_config = config.lightning.trainer
    trainer_config["max_steps"] = len(data.train_dataset) // data.train_config.batch_size * 2 * trainer_config["max_epochs"]

    trainer = pl.Trainer(**trainer_config, logger=logger, callbacks=callbacks)

    import time
    time.sleep(5)
    trainer.fit(model, data)

    data.setup(stage="test")
    ckpt = os.listdir("checkpoints")[-2]
    model.load_state_dict(torch.load(osp.join("checkpoints", ckpt), weights_only=False)["state_dict"], strict=True)
    model.to("cuda:0")
    with torch.no_grad():
        _ = calculate_metrics(model, data.test_dataloader(), model.device)