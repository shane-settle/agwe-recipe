import logging as log


class LRScheduler:

  def __init__(self,
               optim,
               gamma):

    log.info(f" >> optim= {optim.__class__.__name__}")
    log.info(f" >> decay lr by {gamma}")

    self.optim = optim
    self.gamma = gamma

  @property
  def converged(self):
    return self.optim.converged

  def reduce_lr(self):
    for g in self.optim.param_groups:
      old_lr = g["lr"]
      g["lr"] *= self.gamma
      log.info(f"Reducing group lr from {old_lr} to {g['lr']}")
