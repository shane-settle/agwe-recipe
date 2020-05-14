import torch.nn as nn


def pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False):
  return nn.utils.rnn.pack_padded_sequence(x, lens,
                                           batch_first=batch_first,
                                           enforce_sorted=enforce_sorted)


def pad_packed_sequence(x, batch_first=True):
  return nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
