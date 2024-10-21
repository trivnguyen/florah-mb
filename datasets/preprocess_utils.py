
import numpy as np

def pad_sequences(sequences, max_len=None, padding_value=np.nan):
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    padded_sequences = []
    original_lengths = []
    for seq in sequences:
        original_lengths.append(len(seq))
        padding_length = max_len - len(seq)
        padded_seq = np.pad(seq, ((0, padding_length), (0, 0)), constant_values=padding_value)
        padded_sequences.append(padded_seq)

    return np.stack(padded_sequences), np.array(original_lengths)

def create_padding_mask(lengths, max_len):
    """ Create a padding mask. """
    batch_size = lengths.shape[0]

    # Generate a mask with shape [batch_size, seq_len]
    mask = np.arange(max_len).reshape(1, -1) >= lengths.reshape(-1, 1)
    return mask
