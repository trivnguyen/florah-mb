
import torch
import torch.nn as nn


def pad_sequences(sequences, max_len=None, padding_value=0):
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    padded_sequences = []
    original_lengths = []
    for seq in sequences:
        original_lengths.append(len(seq))
        padding_length = max_len - len(seq)
        padded_seq = nn.functional.pad(seq, (0, 0, 0, padding_length), value=padding_value)
        padded_sequences.append(padded_seq)

    return torch.stack(padded_sequences), torch.tensor(original_lengths)

def create_padding_mask(lengths, max_len, batch_first=False):
    """ Create a padding mask. """
    batch_size = lengths.size(0)

    # Generate a mask with shape [batch_size, seq_len]
    mask = torch.arange(max_len).expand(batch_size, max_len) >= lengths.unsqueeze(1)
    if not batch_first:
        mask = mask.transpose(0, 1)
    return mask