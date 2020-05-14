import logging as log

import torch
import torch.nn.functional as F


class MultiViewTripletLoss:

  def __init__(self, margin, k, min_k=None, average=True):

    log.info(f" >> margin= {margin}")
    log.info(f" >> k= {k}")
    log.info(f" >> min_k= {min_k}")
    log.info(f" >> average= {average}")

    self.margin = margin
    self.k = k
    self.min_k = min_k or k
    self.average = average

  def get_sims(self, x, y, inv):

    n, d = x.shape
    m = y.shape[0]

    same = F.cosine_similarity(x, y[inv]).unsqueeze(-1)

    perms = torch.cat([(inv + i) % m for i in range(1, m)])
    diff = F.cosine_similarity(
        x.view(n, 1, d), y[perms].view(n, m - 1, d), dim=2)

    return same, diff, perms

  def get_topk(self, x, k, dim=0):

    return x.topk(min(k, x.shape[dim]), dim=dim)[0]

  def __call__(self, x, y, inv):

    n, d = x.shape
    m = y.shape[0]

    if self.k > self.min_k:
      self.k -= 1

    k = min(self.k, m - 1)

    same, diff, perms = self.get_sims(x, y, inv)

    # Most offending words per utt
    diff_k = self.get_topk(diff, k=k, dim=1)
    obj0 = F.relu(self.margin + diff_k - same)

    # Most offending utts per word
    utt_diff_k = torch.zeros(m, k, device=diff.device)
    for i in range(m):
      utt_diff_k[i] = self.get_topk(diff.view(-1)[perms == i], k=k)
    obj2 = F.relu(self.margin + utt_diff_k[inv] - same)

    loss = (obj0 + obj2).mean(1)

    return loss.mean() if self.average else loss.sum()
