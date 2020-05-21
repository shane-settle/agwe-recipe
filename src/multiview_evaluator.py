import logging as log

from utils.loader import load
from saver.saver import savez


def train(config):

  vocab = load("vocab_data", config)
  vocab_sampler = load("vocab_sampler", config, examples=vocab.examples)
  vocab.init_data_loader(vocab_sampler)

  dev_data = load("dev_data", config, vocab=vocab)
  dev_sampler = load("dev_sampler", config, examples=dev_data.examples)
  dev_data.init_data_loader(dev_sampler)

  test_data = load("test_data", config, vocab=vocab)
  test_sampler = load("test_sampler", config, examples=test_data.examples)
  test_data.init_data_loader(test_sampler)

  eval_fn = load("eval", config)

  net = load("net", config,
             view1_input_size=dev_data.input_feat_dim,
             view2_num_embeddings=dev_data.input_num_subwords)
  net.cuda()
  net.load(tag="best")

  log.info("Evaluating best model on dev and test:")

  log.info("dev score:")
  _, dev_out = eval_fn(net, dev_data)
  savez("save/dev-embs", **dev_out)

  log.info("test score:")
  _, test_out = eval_fn(net, test_data)
  savez("save/test-embs", **test_out)
