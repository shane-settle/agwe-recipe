import logging as log
import torch.utils.data as tud
import numpy as np

from saver.saver import Saver


class Dataset(tud.Dataset):

  @property
  def iter(self):
    return self.loader.batch_sampler.iter

  def __iter__(self):
    log.info(f"Iterating {self.__class__.__name__} (start_iter= {self.iter})")
    self.iterator = iter(self.loader)
    return self.iterator

  def __len__(self):
    return len(self.loader)

  def init_data_loader(self, batch_sampler):
    self.loader = tud.DataLoader(self, num_workers=1,
                                 batch_sampler=batch_sampler,
                                 collate_fn=self.collate_fn)
    log.info(f" >> {batch_sampler.__class__.__name__}; {len(self)} batches")


class SpeechDataset(Dataset, Saver):

  @staticmethod
  def add_deltas(feat):
    feat = np.pad(feat, ((2, 2), (0, 0)), 'edge')
    d = feat[2:, :] - feat[:-2, :]
    dd = d[2:, :] - d[:-2, :]
    return np.concatenate((feat[2:-2, :], d[1:-1], dd), axis=1)

  @staticmethod
  def stack_input_frames(feat, rstack=1, lstack=0):
    deci_rate = rstack + lstack + 1
    n, d = feat.shape
    feat_stack = np.zeros((n, d * deci_rate))
    for r in range(-lstack, rstack + 1):
      start = d * (r + lstack)
      stop = d * (r + lstack + 1)
      feat_stack[:, start:stop] = np.roll(feat, -r, axis=0)
    return feat_stack[::deci_rate]

  def state_dict(self):
    return self.loader.batch_sampler.state_dict(self.iterator)

  def load_state_dict(self, state_dict):
    self.loader.batch_sampler.load_state_dict(state_dict)
