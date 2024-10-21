from typing import Dict, Optional, List, Any, Tuple

import time

import torch
import torch.nn as nn
from tqdm import tqdm

from florah import training_utils, models_utils

@torch.no_grad()
def generate_hist_batch(
    model,
    x_root: torch.Tensor,
    t_out: torch.Tensor,
    norm_dict: Dict[str, Any],
    device: torch.device,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = device or model.device
    model.eval()
    model.to(device)
    if not isinstance(x_root, torch.Tensor):
        x_root = torch.tensor(x_root, device=device).float()
    else:
        x_root = x_root.to(device).float()
    if not isinstance(t_out, torch.Tensor):
        t_out = torch.tensor(t_out, device=device).float()
    else:
        t_out = t_out.to(device).float()

    # normalizing data
    if norm_dict is not None:
        x_loc = torch.tensor(norm_dict['x_loc'], dtype=torch.float32, device=device)
        x_scale = torch.tensor(norm_dict['x_scale'], dtype=torch.float32, device=device)
        t_loc = torch.tensor(norm_dict['t_loc'], dtype=torch.float32, device=device)
        t_scale = torch.tensor(norm_dict['t_scale'], dtype=torch.float32, device=device)
    else:
        x_loc = torch.zeros(x_root.size(0), device=device)
        x_scale = torch.ones(x_root.size(0), device=device)
        t_loc = torch.zeros(1, device=device)
        t_scale = torch.ones(1, device=device)
    x_root = (x_root - x_loc) / x_scale
    t_out = (t_out - t_loc) / t_scale

    n_t = t_out.size(1)
    halo_feats = torch.zeros(
        (x_root.size(0), n_t, x_root.size(1)), dtype=torch.float32, device=device)
    halo_feats[:, 0] = x_root
    class_output = torch.zeros(
        (x_root.size(0), n_t, model.num_classes), dtype=torch.float32, device=device)

    for i in range(n_t-1):
        src = halo_feats[:, :i + 1]
        tgt_t = t_out[:, i + 1]
        if model.concat_time:
            src_t = torch.cat([t_out[:, :i + 1], t_out[:, 1:i+2]], dim=-1)
        else:
            src_t = t_out[:, :i + 1]

        # pss the source sequence through the encoder
        if model.encoder_args.name == 'transformer':
            x_enc = model.encoder(src, src_t, src_padding_mask=None)
            x_enc_reduced = models_utils.summarize_features(
                x_enc, reduction='last', padding_mask=None)
        elif model.encoder_args.name == 'gru':
            x_enc = model.encoder(
                src,
                src_t,
                lengths=torch.tensor([i + 1, ], dtype=torch.long).expand(src.size(0))
            )
            x_enc_reduced = models_utils.summarize_features(
                x_enc, reduction='last', padding_mask=None)

        # pass the encoded sequence through the decoder
        if model.use_sos_embedding:
            tgt_in = model.sos_embedding(x_enc_reduced).unsqueeze(1)
        else:
            tgt_in = torch.zeros((x_root.size(0), 1, x_root.size(1)), device=device)

        if model.decoder_args.name == 'transformer':
            x_dec = model.decoder(
                tgt_in,
                memory=x_enc_reduced.unsqueeze(1),
                context=tgt_t,
                tgt_padding_mask=None,
                memory_padding_mask=None
            )
        elif model.decoder_args.name == 'gru':
            x_dec = model.decoder(
                x=tgt_in,
                t=tgt_t.unsqueeze(1).expand(-1, tgt_in.size(1), -1),
                lengths=torch.ones((1, ), dtype=torch.long)
            )
        # sample the next halo from the flows
        if model.concat_npe_context:
            context_flows = torch.cat([
                x_dec,
                x_enc_reduced.unsqueeze(1).expand(1, x_dec.size(1), 1)
            ], dim=-1)
        else:
            context_flows = x_dec + x_enc_reduced.unsqueeze(1).expand(-1, x_dec.size(1), -1)
        context_flows = context_flows[:, -1]
        halo_feats[:, i+1] = model.npe.flow(model.npe(context_flows)).sample()

        # calculate and return x class
        x_class = x_enc_reduced + model.classifier_context_embed(tgt_t)
        x_class = model.classifier(x_class)
        class_output[:, i+1] += x_class

    # unnormalize the features
    halo_feats = halo_feats * x_scale + x_loc
    t_out = t_out * t_scale + t_loc
    halo_feats = torch.cat([halo_feats, t_out], dim=-1)

    return halo_feats, class_output
