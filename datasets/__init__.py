
from typing import List, Optional, Tuple

import os
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from datasets.preprocess_utils import pad_sequences, create_padding_mask

def read_dataset(
    dataset_root: str,
    dataset_name: str,
    features: List[str],
    time_features: List[str],
    reverse_time: bool = False,
):
    """ Read the dataset and extract the relevant features. """
    path = os.path.join(dataset_root, dataset_name + '.pickle')
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
    num_data = len(dataset['sfh_len'])

    # extract the relevant features
    feats_arr = []
    times_arr = []
    for i in range(num_data):
        feats = np.stack([dataset[feat][i] for feat in features]).T
        times = np.stack([dataset[feat][i] for feat in time_features]).T
        if reverse_time:
            times = times[::-1]
            feats = feats[::-1]
        feats_arr.append(feats)
        times_arr.append(times)

    padded_feats, lengths = pad_sequences(feats_arr, padding_value=-1)
    padded_times, _ = pad_sequences(times_arr, padding_value=-1)
    mask = create_padding_mask(lengths, padded_feats.shape[1])

    return padded_feats, padded_times, mask

def prepare_dataloader(
    data: List[np.ndarray],
    train_frac: float = 0.8,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 0,
    norm_dict: dict = None,
    seed: Optional[int] = None,
    reverse_time: bool = False,
):

    x, t, mask = data
    num_total = len(x)
    num_train = int(num_total * train_frac)

    # shuffle the data
    rng = np.random.default_rng(seed)
    shuffle = rng.permutation(len(x))
    x = x[shuffle]
    t = t[shuffle]
    mask = mask[shuffle]

    # convert to tensors
    x_ts = torch.tensor(x, dtype=torch.float32)
    t_ts = torch.tensor(t, dtype=torch.float32)
    mask_ts = torch.tensor(mask, dtype=torch.bool)

    # divide into train and validation sets
    x_train, t_train, mask_train = x_ts[:num_train], t_ts[:num_train], mask_ts[:num_train]
    x_val, t_val, mask_val = x_ts[num_train:], t_ts[num_train:], mask_ts[num_train:]

    # normalize the data
    if norm_dict is None:
        x_loc = torch.mean(x_train[~mask_train], dim=0)
        x_scale = torch.std(x_train[~mask_train], dim=0)

        norm_dict = {
            "x_loc": list(x_loc.numpy()),
            "x_scale": list(x_scale.numpy()),
        }
    else:
        x_loc = torch.tensor(norm_dict["x_loc"])
        x_scale = torch.tensor(norm_dict["x_scale"])

    print("Normalization statistics:")
    print("x_loc: {}".format(x_loc))
    print("x_scale: {}".format(x_scale))

    # normalize
    x_train = (x_train - x_loc) / x_scale
    x_val = (x_val - x_loc) / x_scale

    train_loader = DataLoader(
        TensorDataset(x_train, t_train, mask_train),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(x_val, t_val, mask_val),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, norm_dict
