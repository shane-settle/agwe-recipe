import logging as log
import re
import inspect

from utils.caller import call


def load(name, config, **kw):

  mod = call(getattr(config, f"{name}_fn"))

  log.info(f"Building {name} from config")
  for k, v in vars(config).items():
    if k.startswith(f"{name}") and not k.endswith("_fn"):
      k = re.sub(f"{name}_", "", k)
      if k not in kw:
        kw[k] = v

  valid = list(inspect.signature(mod).parameters)
  invalid = [k for k in kw if k not in valid]

  if len(invalid) > 0:
    raise ValueError(f"invalid keywords found {invalid}")

  return mod(**kw)
