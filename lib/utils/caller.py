import logging as log
import importlib


def call(path):

  log.info(f"Using module at {path}")

  mod, fun = path.rsplit(".", 1)
  mod = importlib.import_module(mod)
  fun = getattr(mod, fun)

  assert callable(fun)

  return fun
