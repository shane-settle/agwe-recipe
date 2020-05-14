import logging as log
import random
import numpy as np


class _StatefulBatchSampler:

  def __len__(self):
    return len(self.batches)

  def __iter__(self):
    while self.iter < len(self):
      batch = self.batches[self.iter]
      self.iter += 1
      yield batch
    self.init_iter()

  def state_dict(self, itr):
    return {
      "iter": self.iter - (itr._send_idx - itr._rcvd_idx),
      "batches": np.array(self.batches)
    }

  def load_state_dict(self, state_dict):
    self.iter = state_dict["iter"]
    self.batches = state_dict["batches"].tolist()


class BatchSampler(_StatefulBatchSampler):

  def __init__(self, examples, batch_size,
               shuffle=False):

    log.info(f" >> # examples= {len(examples)}")
    log.info(f" >> batch_size= {batch_size}")
    log.info(f" >> shuffle= {shuffle}")

    self.examples = list(examples)
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.init_iter()

  def init_iter(self):

    if self.shuffle:
      random.shuffle(self.examples)

    self.iter = 0
    self.batches = []

    batch = []
    for example in self.examples:
      if len(batch) < self.batch_size:
        batch.append(example)
      else:
        self.batches.append(batch)
        batch = [example]

    if len(batch) > 0:
      self.batches.append(batch)


class PackedBatchSampler(_StatefulBatchSampler):

  def __init__(self, examples, batch_size, sort_by,
               variable=False, shuffle=False):

    log.info(f" >> # examples= {len(examples)}")
    log.info(f" >> batch_size= {batch_size}")
    log.info(f" >> shuffle= {shuffle}")
    log.info(f" >> sort by {sort_by}")
    log.info(f" >> variable= {variable}")

    self.examples = examples
    self.batch_size = batch_size

    def get_size(k):
      return self.examples[k][sort_by]
    self.get_size = get_size

    self.variable = variable
    self.shuffle = shuffle
    self.init_iter()

  def init_iter(self):

    self.iter = 0

    batches = []
    batch = []
    batch_size = 0

    examples = sorted(self.examples, key=self.get_size, reverse=True)
    example_size = self.get_size(examples[0]) if self.variable else 1

    for example in examples:
      if batch_size + example_size <= self.batch_size:
        batch.append(example)
        batch_size += example_size
      else:
        batches.append(batch)
        batch = [example]
        example_size = self.get_size(example) if self.variable else 1
        batch_size = example_size

    if len(batch) > 0:
      batches.append(batch)

    self.batches = batches[::-1]

    if self.shuffle:
      random.shuffle(self.batches)
