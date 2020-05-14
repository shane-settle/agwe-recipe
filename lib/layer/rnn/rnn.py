import logging as log

import torch
import torch.nn as nn

from layer.linear import Linear
from layer.utils import pack_padded_sequence, pad_packed_sequence
from saver.saver import Saver


class RNN_default(nn.Module, Saver):

  def __init__(self,
               cell,
               input_size,
               hidden_size,
               bidirectional,
               num_layers=1,
               dropout=0.,
               proj=None,
               num_embeddings=None):
    nn.Module.__init__(self)
    Saver.__init__(self)

    log.info(f" >> cell= {cell}")
    log.info(f" >> input_size= {input_size}")
    log.info(f" >> hidden_size= {hidden_size}")
    log.info(f" >> bidirectional= {bidirectional}")
    log.info(f" >> num_layers= {num_layers}")
    log.info(f" >> dropout between layers= {dropout}")

    if num_embeddings is not None:
      log.info(f" >> num input embeddings= {num_embeddings}")
      self.emb = nn.Embedding(num_embeddings=num_embeddings + 1,
                              embedding_dim=input_size,
                              padding_idx=0)

    self.rnn = getattr(nn, cell)(input_size=input_size,
                                 hidden_size=hidden_size,
                                 bidirectional=bidirectional,
                                 num_layers=num_layers,
                                 dropout=dropout if num_layers > 1 else 0.,
                                 batch_first=True)

    if proj is not None:
      log.info(f" >> proj after rnn= {proj}")
      self.proj = Linear(self.output_size, proj)

  @property
  def input_size(self):
    return self.rnn.input_size

  @property
  def hidden_size(self):
    return self.rnn.hidden_size

  @property
  def bidirectional(self):
    return self.rnn.bidirectional

  @property
  def output_size(self):
    if hasattr(self, "proj"):
      return self.proj.output_size
    elif self.bidirectional:
      return 2 * self.hidden_size
    else:
      return self.hidden_size

  @property
  def device(self):
    return next(self.parameters()).device

  def forward(self, x, lens):

    x = x.to(self.device)

    if hasattr(self, "emb"):
      x = self.emb(x)

    x = pack_padded_sequence(x, lens)

    x, h = self.rnn(x)
    if isinstance(h, tuple):
      h = h[0]

    x, lens = pad_packed_sequence(x)

    if self.bidirectional:
      h = torch.cat((h[-2], h[-1]), dim=1)
    else:
      h = h[-1]

    if hasattr(self, "proj"):
      x = self.proj(x)
      h = self.proj(h)

    return x, lens, h
