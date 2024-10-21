
import os
import pickle
import shutil
import sys

import datasets
import ml_collections
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import yaml
from absl import flags, logging
from ml_collections import config_flags

from florah import utils
from florah.autoregressive import AutoregressiveModel

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
    logging.info("Preparing dataloader...")
    feats, times, mask = datasets.read_dataset(
        config.data.root,
        config.data.name,
        config.data.features,
        config.data.time_features,
        reverse_time=config.data.reverse_time,
    )
    train_loader, val_loader, norm_dict = datasets.prepare_dataloader(
        (feats, times, mask),
        train_batch_size=config.training.train_batch_size,
        eval_batch_size=config.training.eval_batch_size,
        train_frac=config.training.train_frac,
        num_workers=config.training.num_workers,
        seed=config.seed_data,
        norm_dict=None,
    )

    # create the model
    logging.info("Creating model...")
    model = AutoregressiveModel(
        d_in=config.model.d_in,
        d_time=config.model.d_time,
        encoder_args=config.model.encoder,
        npe_args=config.model.npe,
        optimizer_args=config.optimizer,
        scheduler_args=config.scheduler,
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
            filename="best-{epoch}-{step}-{val_loss:.4f}",
            monitor=config.training.get('monitor', 'val_loss'),
            save_top_k=config.training.get('save_top_k', 1),
            mode='min',
            save_weights_only=False,
            save_last=False
        ),
        pl.callbacks.ModelCheckpoint(
            filename="last-{epoch}-{step}-{val_loss:.4f}",
            save_top_k=config.training.get('save_last_k', 1),
            monitor='epoch',
            mode='max',
            save_weights_only=False,
            save_last=False
        ),
        pl.callbacks.LearningRateMonitor("step"),
    ]
    train_logger = pl_loggers.TensorBoardLogger(workdir, version='')
    trainer = pl.Trainer(
        default_root_dir=workdir,
        max_steps=config.training.max_steps,
        accelerator=config.training.accelerator,
        callbacks=callbacks,
        logger=train_logger,
        gradient_clip_val=config.training.get('gradient_clip_val', 0),
        enable_progress_bar=config.get("enable_progress_bar", True),
        num_sanity_val_steps=0,
        # log_every_n_steps=config.training.log_every_n_steps,
        # val_check_interval=config.training.val_check_interval,
        # check_val_every_n_epoch=None,
    )

    # train the model
    logging.info("Training model...")
    pl.seed_everything(config.seed_training)
    trainer.fit(
        model,
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
    # Parse flags
    FLAGS(sys.argv)

    # Start training run
    train(config=FLAGS.config, workdir=FLAGS.config.workdir)
