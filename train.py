from typing import List, Optional

import os
import pickle
import shutil
import sys

import datasets
import ml_collections
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import yaml
from absl import flags, logging
from ml_collections import config_flags

import datasets
from models import utils, training_utils
from models.atg_mb import AutoregTreeGenMB

logging.set_verbosity(logging.INFO)


def train(
    config: ml_collections.ConfigDict, workdir: str = "./logging/"
):
    # set up work directory
    if not hasattr(config, "name"):
        name = utils.get_random_name()
    else:
        name = config["name"]
    logging.info("Starting training run {} at {}".format(name, workdir))

    workdir = os.path.join(workdir, name)
    checkpoint_path = None
    if os.path.exists(workdir):
        if config.overwrite:
            shutil.rmtree(workdir)
        elif config.get('checkpoint', None) is not None:
            checkpoint_path = os.path.join(
                workdir, 'lightning_logs/checkpoints', config.checkpoint)
        else:
            raise ValueError(
                f"Workdir {workdir} already exists. Please set overwrite=True "
                "to overwrite the existing directory.")

    os.makedirs(workdir, exist_ok=True)

    # Save the configuration to a YAML file
    config_dict = config.to_dict()
    config_path = os.path.join(workdir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # load dataset and prepare dataloader
    logging.info("Loading dataset ...")
    data = datasets.read_dataset(
        dataset_root=config.data.root,
        dataset_name=config.data.name,
        max_num_files=config.data.get("num_files", 1),
    )

    logging.info("Preparing dataloader...")
    train_loader, val_loader, norm_dict = datasets.prepare_dataloader(
        data,
        train_frac=config.data.train_frac,
        train_batch_size=config.training.train_batch_size,
        eval_batch_size=config.training.eval_batch_size,
        num_workers=config.training.get("num_workers", 0),
        seed=config.seed.data,
        norm_dict=None,
        reverse_time=config.data.get("reverse_time", False),
    )

    # create the model
    logging.info("Creating model...")
    model_atg = AutoregTreeGenMB(
        d_in=config.model.d_in,
        encoder_args=config.model.encoder,
        npe_args=config.model.npe,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
        training_args=config.training,
        norm_dict=norm_dict,
    )
    # Create call backs and trainer objects
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor=config.training.get('monitor', 'val_loss'),
            patience=config.training.get('patience', 1000),
            mode='min',
            verbose=True
        ),
        pl.callbacks.ModelCheckpoint(
            filename="{epoch}-{step}-{val_loss:.4f}",
            monitor=config.training.get('monitor', 'val_loss'),
            save_top_k=config.training.get('save_top_k', 1),
            mode='min',
            save_weights_only=False,
            save_last=True
        ),
        pl.callbacks.LearningRateMonitor("step"),
    ]
    train_logger = pl_loggers.TensorBoardLogger(workdir, version='')
    trainer = pl.Trainer(
        default_root_dir=workdir,
        max_epochs=config.training.max_epochs,
        max_steps=config.training.max_steps,
        accelerator=config.accelerator,
        callbacks=callbacks,
        logger=train_logger,
        gradient_clip_val=config.training.get('gradient_clip_val', 0),
        enable_progress_bar=config.get("enable_progress_bar", True),
        num_sanity_val_steps=0,
    )

    # train the model
    logging.info("Training model...")
    pl.seed_everything(config.seed.training)
    trainer.fit(
        model_atg,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=checkpoint_path,
    )

    # Done
    logging.info("Done!")

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training or sampling hyperparameter configuration.",
        lock_config=True,
    )
    flags.DEFINE_boolean(
        "debug",
        False,
        "If set to True, activates debug configuration.",
    )
    # Parse flags
    FLAGS(sys.argv)

    if FLAGS.debug:
        print("Debug mode activated.")
        # copy the config with some argument changes
        config = FLAGS.config
        config.workdir = 'debug'
        config.name = 'debug'
        config.overwrite = True
        config.enable_progress_bar = True
        config.data.num_files = 1
    else:
        config = FLAGS.config

    # Start training run
    train(config=config, workdir=config.workdir)
