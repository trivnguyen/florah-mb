
from typing import Dict, Optional, List, Any, Tuple
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from tqdm import tqdm

from models import training_utils, models_utils


@torch.no_grad()
def generate_mb_batch(
    model,
    x_root: torch.Tensor,
    t_out: torch.Tensor,
    norm_dict: Dict[str, Any],
    device: torch.device,
    numpy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:

    device = device or model.device
    model.eval()
    model.to(device)
    use_desc_mass_ratio = model.training_args.use_desc_mass_ratio

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

    n_t = t_out.size(0)
    halo_feats = torch.zeros(
        (x_root.size(0), n_t, x_root.size(1)), dtype=torch.float32, device=device)
    halo_feats[:, 0] = x_root

    for i in range(n_t-1):
        src = halo_feats[:, :i + 1]
        src_t = torch.cat([t_out[:i+1], t_out[1:i+2]], dim=-1)
        src_t = src_t.unsqueeze(0).expand(src.size(0), src_t.size(0), src_t.size(1))
        src_len = torch.tensor([i + 1, ], dtype=torch.long).expand(src.size(0))

        context = model.encoder(src, src_t, src_len)
        context = context[:, -1, :]  # only take the last time step
        prog_feat = model.npe.sample(context)
        if use_desc_mass_ratio:
            halo_feats[:, i + 1, 0] = prog_feat[:, 0] + halo_feats[:, i, 0]
            halo_feats[:, i + 1, 1:] = prog_feat[:, 1:]
        else:
            halo_feats[:, i + 1] = prog_feat

    # unnormalize the features
    halo_feats = halo_feats * x_scale + x_loc
    t_out = t_out * t_scale + t_loc
    t_out = t_out.unsqueeze(0).expand(halo_feats.size(0), halo_feats.size(1), 2)
    halo_feats = torch.cat([halo_feats, t_out], dim=-1)

    if numpy:
        halo_feats = halo_feats.cpu().numpy()
        t_out = t_out.cpu().numpy()

    return halo_feats, t_out

@torch.no_grad()
def generate_mb_ext_batch(
    model,
    x_br: torch.Tensor,
    x_mask: torch.Tensor,
    t_out: torch.Tensor,
    norm_dict: Dict[str, Any],
    device: torch.device,
    numpy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:


    device = device or model.device
    model.eval()
    model.to(device)
    use_desc_mass_ratio = model.training_args.use_desc_mass_ratio

    if not isinstance(x_br, torch.Tensor):
        x_br = torch.tensor(x_br, device=device).float()
    else:
        x_br = x_br.to(device).float()
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
        x_loc = torch.zeros(x_br.size(0), device=device)
        x_scale = torch.ones(x_br.size(0), device=device)
        t_loc = torch.zeros(1, device=device)
        t_scale = torch.ones(1, device=device)
    x_br = (x_br - x_loc) / x_scale
    t_out = (t_out - t_loc) / t_scale

    n_t = t_out.size(0)
    halo_feats = x_br.clone()

    for i in range(n_t-1):
        mask = x_mask[:, i+1]
        if not torch.any(mask):
            continue
        src = halo_feats[:, :i + 1]
        src_t = torch.cat([t_out[:i+1], t_out[1:i+2]], dim=-1)
        src_t = src_t.unsqueeze(0).expand(src.size(0), src_t.size(0), src_t.size(1))
        src_len = torch.tensor([i + 1, ], dtype=torch.long).expand(src.size(0))

        context = model.encoder(src, src_t, src_len)
        context = context[:, -1, :]  # only take the last time step
        prog_feat = model.npe.sample(context)

        if use_desc_mass_ratio:
            prog_feat[:, 0] = prog_feat[:, 0] + halo_feats[:, i, 0]

        halo_feats[mask, i + 1] = prog_feat[mask]

    # unnormalize the features
    halo_feats = halo_feats * x_scale + x_loc
    t_out = t_out * t_scale + t_loc
    t_out = t_out.unsqueeze(0).expand(halo_feats.size(0), halo_feats.size(1), 1)
    halo_feats = torch.cat([halo_feats, t_out], dim=-1)

    if numpy:
        halo_feats = halo_feats.cpu().numpy()
        t_out = t_out.cpu().numpy()

    return halo_feats, t_out
