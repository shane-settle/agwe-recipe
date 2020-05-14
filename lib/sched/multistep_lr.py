import logging as log

from lr_scheduler import _LRScheduler
from score_tracker import _ScoreTracker


class MultiStepLR(_LRScheduler, _ScoreTracker):

  def __init__(self,
               optim,
               eval_fn,
               gamma,
               milestones,
               epoch_len=None):
    _LRScheduler.__init__(self, optim, gamma)
    _ScoreTracker.__init__(self, eval_fn)

    if epoch_len is not None:
      milestones = list(map(lambda m: m * epoch_len, milestones))

    self.milestones = milestones

    log.info(f"Reducing lr by gamma={gamma} at each milestone")
    log.info(f" >> {milestones}")

  def step(self, global_step, score=None):

    if score is not None:
      self.update_score(global_step, score)

    if global_step in self.milestones:
      self.reduce_lr()
      self.optim.save(tag=global_step, best=True)

  def state_dict(self):
    state_dict = {}
    for k, v in self.__dict__.items():
      if k not in ["optim", "eval_fn"]:
        state_dict[k] = v
    return state_dict

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)
