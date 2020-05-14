import logging as log
import torch.optim as optim

from saver.saver import Saver


class SGD(optim.SGD, Saver):

  def __init__(self,
               params,
               lr,
               momentum=0,
               dampening=0,
               weight_decay=0,
               nesterov=True,
               min_lr=1e-10):

    log.info(f" >> lr= {lr}")
    log.info(f" >> momentum= {momentum}")
    log.info(f" >> dampening= {dampening}")
    log.info(f" >> weight_decay= {weight_decay}")
    log.info(f" >> nesterov= {nesterov}")
    log.info(f" >> min_lr= {min_lr}")

    optim.SGD.__init__(self,
                       params,
                       lr=lr,
                       momentum=momentum,
                       dampening=dampening,
                       weight_decay=weight_decay,
                       nesterov=nesterov)
    Saver.__init__(self)

    self.min_lr = min_lr

  @property
  def converged(self):
    return max(g["lr"] for g in self.param_groups) < self.min_lr
