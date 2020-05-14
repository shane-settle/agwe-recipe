import logging as log

from sched.lr_scheduler import LRScheduler
from sched.score_tracker import ScoreTracker
from saver.saver import Saver


class ReduceLROnPlateau(LRScheduler, ScoreTracker, Saver):

  def __init__(self,
               optim,
               eval_fn,
               gamma,
               patience,
               net=None):
    LRScheduler.__init__(self, optim, gamma)
    ScoreTracker.__init__(self, eval_fn)
    Saver.__init__(self)

    log.info(f" >> patience={patience}")
    log.info(f" >> revert={net is not None}")

    self.net = net
    self.patience = patience
    self.bad_evals = 0

  def step(self, global_step, score):

    self.update_score(global_step, score)

    if global_step == self.best_global_step:
      self.bad_evals = 0
      return

    self.bad_evals += 1

    if self.bad_evals >= self.patience:
      if self.net is not None:
        log.info(f"Reverting to previous best")
        self.net.load("best")
        self.optim.load("best")
      self.reduce_lr()
      self.optim.save(tag=global_step, best=True)
      self.bad_evals = 0

  def state_dict(self):
    state_dict = {}
    for k, v in self.__dict__.items():
      if k not in ["optim", "net", "eval_fn"]:
        state_dict[k] = v
    return state_dict

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)
