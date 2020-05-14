import logging as log
import os
import re
import json
import torch
import torch.nn as nn
import numpy as np
import filecmp


def save_many(*attrs, tag, best=False):
  for attr in attrs:
    attr.save(tag=tag, best=best)
    log.info(f"saved {attr.__class__.__name__} successfully")


def save_config(config):
  with open(config.config_file, "w") as f:
    json.dump(vars(config), f)
    log.info(f"saved config successfully")


def savez(name, **variables):
  np.savez(name, **variables)
  log.info(f"saved outputs to npz file successfully")


class Saver:

  def __init__(self, save_dir=os.path.join(os.getcwd(), "save"), keep=5):

    savepath = os.path.join(save_dir, self.__class__.__name__ + ".{}.pth")
    pattern = re.compile(savepath.format("[0-9]+"))
    os.makedirs(save_dir, exist_ok=True)

    self.savepath = savepath
    self.pattern = pattern
    self.save_dir = save_dir
    self.keep = keep

  def _link(self, tag, link):
    savepath = self.savepath.format(tag)
    link_savepath = self.savepath.format(link)
    if os.path.exists(link_savepath):
      os.remove(link_savepath)
    os.symlink(savepath, link_savepath)
    log.info(f"{link} linked {os.path.relpath(link_savepath)}")

  def _clean(self):
    best_savepath = self.savepath.format("best")
    files = {}
    for base in os.listdir(self.save_dir):
      file_ = os.path.join(self.save_dir, base)
      if os.path.exists(best_savepath) and filecmp.cmp(file_, best_savepath):
        continue
      if bool(re.match(self.pattern, file_)):
        files[file_] = os.path.getmtime(file_)
    if len(files) > self.keep:
      for old_file in sorted(files, key=files.get)[:-self.keep]:
        if os.path.exists(old_file):
          os.remove(old_file)
        log.info(f"removed {os.path.relpath(old_file)}")

  def save(self, tag, best=False):
    savepath = self.savepath.format(tag)
    if not os.path.exists(savepath):
      torch.save(self.state_dict(), savepath)
      log.info(f"saved {os.path.relpath(savepath)}")
      self._link(tag, link="most_recent")
    if best:
      self._link(tag, link="best")
    self._clean()

  def load(self, tag):
    loadpath = self.savepath.format(tag)
    if os.path.islink(loadpath):
      loadpath = os.readlink(loadpath)
    self.load_state_dict(torch.load(loadpath))
    log.info(f"loaded {os.path.relpath(loadpath)}")
    log.info(f"note: relative to {os.getcwd()}")
