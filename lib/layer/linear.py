import logging as log
import torch
import torch.nn as nn

from saver.saver import Saver


class Linear(nn.Module, Saver):

  def __init__(self, in_features, out_features):
    nn.Module.__init__(self)
    Saver.__init__(self)

    log.info(f" >> in_features= {in_features}")
    log.info(f" >> out_features= {out_features}")

    self.lin = nn.Linear(in_features, out_features)

  @property
  def in_features(self):
    return self.lin.in_features

  @property
  def out_features(self):
    return self.lin.out_features

  @property
  def output_size(self):
    return self.out_features

  def forward(self, x):

    dims = list(x.shape)
    x = x.contiguous()

    if len(dims) > 2:
      d = dims.pop()
      n = torch.prod(torch.tensor(dims, dtype=torch.long))
      return self.lin(x.view(n, d)).view(*dims, -1)
    else:
      return self.lin(x)
