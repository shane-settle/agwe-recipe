import logging as log
import torch
import numpy as np

import metric.ap as ap


class MultiViewAP:

  def __init__(self, metric, acoustic_ap=True, crossview_ap=True,
               full_vocab=False):

    log.info(f" >> acoustic_ap= {acoustic_ap}")
    log.info(f" >> crossview_ap= {crossview_ap}")
    log.info(f" >> full_vocab= {full_vocab}")
    log.info(f" >> metric= {metric}")
    log.info(f" >> mode= max")

    self.acoustic_ap = (metric == "acoustic_ap") or acoustic_ap
    self.crossview_ap = (metric == "crossview_ap") or crossview_ap
    self.full_vocab = full_vocab
    self.metric = metric
    self.mode = "max"

  def __call__(self, net, data):

    outputs = {}

    net.eval()
    with torch.no_grad():

      embs1, ids1 = [], []
      for batch in data.loader:
        ids = batch.pop("ids")
        inv = batch.pop("inv")
        out1 = net.forward_view(batch, "view1")
        ids1.append(ids[inv.numpy()])
        embs1.append(out1.cpu().numpy())

      ids1 = np.hstack(ids1)
      words1 = list(map(data.vocab.i2w.get, ids1))
      embs1 = np.vstack(embs1)
      n = len(ids1)

      if self.crossview_ap:
        embs2, ids2 = [], []
        for batch in data.vocab.loader if self.full_vocab else data.loader:
          ids = batch.pop("ids")
          out2 = net.forward_view(batch, "view2")
          ids2.append(ids)
          embs2.append(out2.cpu().numpy())

        ids2, ind = np.unique(np.hstack(ids2), return_index=True)
        embs2 = np.vstack(embs2)[ind]
        words2 = list(map(data.vocab.i2w.get, ids2))

        m = len(ids2)
        crossview_ap = ap.cross_view(embs1, ids1, embs2, ids2)
        log.info(f"  >> crossview_ap={crossview_ap:.3f} ({n} x {m} pairs)")
        outputs.update({
          "crossview_ap": crossview_ap,
          "embs2": embs2, "ids2": ids2, "words2": words2
        })

      if self.acoustic_ap:
        acoustic_ap = ap.single_view(embs1, ids1)
        log.info(f"  >> acoustic_ap={acoustic_ap:.3f} ({n} choose 2 pairs)")
        outputs.update({
          "acoustic_ap": acoustic_ap,
          "embs1": embs1, "ids1": ids1, "words1": words1
        })

    return outputs[self.metric], outputs
