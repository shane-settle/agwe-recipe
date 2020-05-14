import logging as log


class ScoreTracker:

  def __init__(self,
               eval_fn):

    log.info(f" >> eval_fn= {eval_fn.__class__.__name__}")

    self.eval_fn = eval_fn
    self.best_score = float('-inf' if eval_fn.mode == "max" else 'inf')
    self.best_global_step = 0

  def is_better(self, score):

    if self.eval_fn.mode == "max":
      return score > self.best_score
    else:
      return score < self.best_score

  def update_score(self, global_step, score):

    if self.is_better(score):
      self.best_score = score
      self.best_global_step = global_step

    log.info(f"best_so_far= {self.best_score:.3f} at {self.best_global_step}")
