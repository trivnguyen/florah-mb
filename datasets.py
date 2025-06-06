from typing import List, Optional, Tuple

import os
import pickle
from tqdm import tqdm

import ml_collections
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx

from models import training_utils

def read_dataset(
    dataset_name: str, dataset_root: str, max_num_files: int = 1,
    index_start: int=0, verbose: bool = True, prefix: str = "data"
):
    data = []
    data_dir = os.path.join(dataset_root, dataset_name)
    for i in tqdm(range(index_start, index_start + max_num_files)):
        data_path = os.path.join(data_dir, "{}.{}.pkl".format(prefix, i))
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data.extend(pickle.load(f))
            if verbose:
                print("Loading data from {}...".format(data_path))
        else:
            if verbose:
                print("Data file {} not found. Stopping".format(data_path))
            break
    if len(data) == 0:
        raise ValueError("No data found in the specified directory.")

    if not isinstance(data[0], Data):
        data = [from_networkx(d) for d in data]
    return data

def prepare_dataloader(
    data: List,
    train_frac: float = 0.8,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 0,
    norm_dict: dict = None,
    reverse_time: bool = False,
    seed: Optional[int] = None
):
    """
    Prepare the dataloader for training and evaluation.
    Args:
        data: list of PyTorch Geometric Data objects.
        train_frac: fraction of the data to use for training.
        train_batch_size: batch size for training.
        eval_batch_size: batch size for evaluation.
        num_workers: number of workers for data loading.
        norm_dict: dictionary containing normalization statistics.
        reverse_time: whether to reverse the time axis
        seed: random seed for shuffling the data.
    Returns:
        train_loader: PyTorch DataLoader for training.
        val_loader: PyTorch DataLoader for evaluation.
        norm_dict: dictionary containing normalization statistics.
    """

    rng = np.random.default_rng(seed)
    rng.shuffle(data)

    num_total = len(data)
    num_train = int(num_total * train_frac)

    # calculate the normaliziation statistics
    if norm_dict is None:
        x = torch.cat([d.x[..., :-1] for d in data[:num_train]])
        t = torch.cat([d.x[..., -1:] for d in data[:num_train]])

        # standardize the input features and min-max normalize the time
        x_loc = x.mean(dim=0)
        x_scale = x.std(dim=0)
        t_loc = t.min()
        t_scale = t.max() - t_loc
        if reverse_time:
            t_loc = t_scale + t_loc
            t_scale = -t_scale

        norm_dict = {
            "x_loc": list(x_loc.numpy()),
            "x_scale": list(x_scale.numpy()),
            "t_loc": t_loc.numpy(),
            "t_scale": t_scale.numpy(),
        }
    else:
        x_loc = torch.tensor(norm_dict["x_loc"], dtype=torch.float32)
        x_scale = torch.tensor(norm_dict["x_scale"], dtype=torch.float32)
        t_loc = torch.tensor(norm_dict["t_loc"], dtype=torch.float32)
        t_scale = torch.tensor(norm_dict["t_scale"], dtype=torch.float32)
    for d in data:
        d.x[..., :-1] = (d.x[..., :-1] - x_loc) / x_scale
        d.x[..., -1:] = (d.x[..., -1:] - t_loc) / t_scale

    print("Normalization statistics:")
    print("x_loc: {}".format(x_loc))
    print("x_scale: {}".format(x_scale))
    print("t_loc: {}".format(t_loc))
    print("t_scale: {}".format(t_scale))

    # pad the data
    max_len = max([d.x.shape[0] for d in data])
    mb_data, mb_len = training_utils.pad_sequences(
        [d.x for d in data], max_len=max_len, padding_value=0)
    train_dataset = TensorDataset(mb_data[:num_train], mb_len[:num_train])
    val_dataset = TensorDataset(mb_data[num_train:], mb_len[num_train:])

    # # create data loader
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, norm_dict
