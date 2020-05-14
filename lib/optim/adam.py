import logging as log
import torch.optim as optim

from saver.saver import Saver


class Adam(optim.Adam, Saver):

  def __init__(self,
               params,
               lr=0.001,
               betas=(0.9, 0.999),
               eps=1e-08,
               weight_decay=0,
               amsgrad=False,
               min_lr=1e-10):

    log.info(f" >> lr= {lr}")
    log.info(f" >> betas= {betas}")
    log.info(f" >> eps= {eps}")
    log.info(f" >> weight_decay= {weight_decay}")
    log.info(f" >> amsgrad= {amsgrad}")
    log.info(f" >> min_lr= {min_lr}")

    optim.Adam.__init__(self,
                        params,
                        lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        amsgrad=amsgrad)
    Saver.__init__(self)

    self.min_lr = min_lr

  @property
  def converged(self):
    return max(g["lr"] for g in self.param_groups) < self.min_lr
