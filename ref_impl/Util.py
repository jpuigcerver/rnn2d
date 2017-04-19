import os
from theano.sandbox.cuda.basic_ops import (GpuContiguous, GpuFromHost)
import numpy as np
from itertools import groupby

def raw_variable(x):
  while x.owner is not None and (type(x.owner.op) == GpuContiguous or type(x.owner.op) == GpuFromHost):
    x = x.owner.inputs[0]
  return x

def get_c_support_code_common():
  base_path = os.path.dirname(__file__)
  with open(base_path + "/c_support_code_common.cpp") as f:
    return f.read()

def get_c_support_code_mdlstm():
  base_path = os.path.dirname(__file__)
  with open(base_path + "/c_support_code_mdlstm.cpp") as f:
    return f.read()

def get_c_support_code_cudnn():
  base_path = os.path.dirname(__file__)
  with open(base_path + "/c_support_code_cudnn.cpp") as f:
    return f.read()


def decode(output, mapping=None):
  W, N, D = output.shape
  maxi = np.argmax(output, axis=-1)

  dec = []
  for n in xrange(N):
    t = [x[0] - 1 for x in groupby(maxi[:,n].tolist()) if x[0] != 0]
    if mapping:
      dec.append(map(lambda x: mapping[x], t))
    else:
      dec.append(t)
  return dec
