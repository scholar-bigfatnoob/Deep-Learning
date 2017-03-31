from __future__ import print_function, division
import sys
import os
import numpy as np
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True


__author__ = "bigfatnoob"


def softmax(x):
  exps = np.exp(x)
  return exps / np.sum(exps, 0)


def cross_entropy(predicted, labels):
  return -1 * np.sum(labels * np.log(predicted), axis=0)


def relu(x):
  return x if x > 0 else 0

delta = 0.000001
x = y = 1000000000
for _ in range(1000000):
  x += delta
print(x-y)