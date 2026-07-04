import ctypes
import numpy as np
import torch

lib = ctypes.CDLL("../../src/kernals/softmax.o")

lib.softmax.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_uint32
]

lib.softmax.restype = None

a = np.random.rand(100)
c = np.empty_like(a)

lib.softmax(a, c, len(a))
x = torch.tensor(a)
res = torch.softmax(x, dim = 0)
print(res)
print(c)