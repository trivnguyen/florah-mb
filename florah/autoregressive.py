
from typing import Union, Dict, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from ml_collections import ConfigDict

from florah.models import GRUEncoder
from florah.transformer import TransformerEncoder
from florah.flows import NPE
from florah import models_utils, training_utils

class AutoregressiveModel(pl.LightningModule):
    def __init__(
        self,
        d_in: int,
        d_time: int,
        encoder_args: Union[ConfigDict, Dict],
        npe_args: Union[ConfigDict, Dict],
        optimizer_args: Union[ConfigDict, Dict],
        scheduler_args: Union[ConfigDict, Dict],
        norm_dict: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_time = d_time
        self.encoder_args = encoder_args
        self.npe_args = npe_args
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args
        self.norm_dict = norm_dict
        self.save_hyperparameters()

        self._setup_model()

    def _setup_model(self) -> None:

        if self.encoder_args.name == 'transformer':
            self.encoder = TransformerEncoder(
                d_in=self.d_in,
                d_model=self.encoder_args.d_model,
                nhead=self.encoder_args.nhead,
                dim_feedforward=self.encoder_args.dim_feedforward,
                num_layers=self.encoder_args.num_layers,
                d_time=self.d_time,
                emb_size=self.encoder_args.emb_size,
                emb_dropout=self.encoder_args.emb_dropout,
                emb_type=self.encoder_args.emb_type,
                concat=self.encoder_args.concat,
            )
        elif self.encoder_args.name == 'gru':
            self.encoder = GRUEncoder(
                d_in=self.d_in,
                d_model=self.encoder_args.d_model,
                d_out=self.encoder_args.d_model,
                dim_feedforward=self.encoder_args.dim_feedforward,
                num_layers=self.encoder_args.num_layers,
                d_time=self.d_time,
                activation_fn=nn.ReLU(),
                concat=self.encoder_args.concat,
            )
        self.npe = NPE(
            input_size=self.d_in,
            context_size=self.encoder_args.d_model,
            hidden_sizes=self.npe_args.hidden_sizes,
            context_embedding_sizes=self.npe_args.context_embedding_sizes,
            num_transforms=self.npe_args.num_transforms,
            dropout=self.npe_args.dropout,
        )

    def _prepare_batch(self, batch):
        feats, times, padding_mask = batch
        max_len = padding_mask.eq(0).sum(-1).max().item()
        feats, times, padding_mask = feats[:, :max_len], times[:, :max_len], padding_mask[:, :max_len]

        # divide the data into input and target, each shifted by one time step
        src, tgt = feats[:, :-1], feats[:, 1:]
        src_t = torch.cat([times[:, :-1], times[:, 1:]], dim=-1)
        tgt_padding_mask = padding_mask[:, 1:]

        # move the data to the correct device
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        src_t = src_t.to(self.device)
        tgt_padding_mask = tgt_padding_mask.to(self.device)

        return {
            "src": src,
            "src_t": src_t,
            "tgt": tgt,
            "tgt_padding_mask": tgt_padding_mask,
            "batch_size": feats.size(0),
        }

    def forward(
        self, src, src_t, src_padding_mask=None,
    ):
        if self.encoder_args.name == 'transformer':
            x_enc = self.encoder(
                src,
                src_t,
                src_padding_mask=src_padding_mask
            )
        elif self.encoder_args.name == 'gru':
            if src_padding_mask is None:
                lengths = torch.tensor([src.size(1), ], dtype=torch.long).expand(src.size(0))
            else:
                lengths = src_padding_mask.eq(0).sum(-1).cpu()
            x_enc = self.encoder(
                x=src,
                t=src_t,
                lengths=lengths
            )
        return x_enc

    def training_step(self, batch, batch_idx):
        batch_dict = self._prepare_batch(batch)
        batch_size = batch_dict['batch_size']

        # forward pass
        context = self(
            src=batch_dict['src'],
            src_t=batch_dict['src_t'],
            src_padding_mask=batch_dict['tgt_padding_mask'],
        )
        lp = self.npe.log_prob(batch_dict['tgt'], context=context)
        lp = lp * batch_dict['tgt_padding_mask'].eq(0).float()
        loss = -lp.mean()
        self.log(
            'train_loss', loss, on_step=True, on_epoch=True,
            logger=True, prog_bar=True,  batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_dict = self._prepare_batch(batch)
        batch_size = batch_dict['batch_size']

        # forward pass
        context = self(
            src=batch_dict['src'],
            src_t=batch_dict['src_t'],
            src_padding_mask=batch_dict['tgt_padding_mask'],
        )
        lp = self.npe.log_prob(batch_dict['tgt'], context=context)
        lp = lp * batch_dict['tgt_padding_mask'].eq(0).float()
        loss = -lp.mean()
        self.log(
            'val_loss', loss, on_step=True, on_epoch=True,
            logger=True, prog_bar=True,  batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        """ Initialize optimizer and LR scheduler """
        return models_utils.configure_optimizers(
            self.parameters(), self.optimizer_args, self.scheduler_args)
