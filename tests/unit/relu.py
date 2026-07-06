import ctypes
import numpy as np
import torch
import torch.nn.functional as F


lib = ctypes.CDLL("../../src/kernals/kernals.so")

lib.RELU.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_uint32,
    ctypes.c_uint32
]

lib.RELU.restype = None

num_test = 1000
length = 10
d = 10

passed = 0
for _ in range(num_test):
    a = np.random.rand(length * d)
    x = torch.tensor(a)
    y = F.relu(x)
    lib.RELU(a, length, d)
    
    try:
        assert (np.allclose(a, y, rtol=1e-9, atol=1e-12))
        passed +=1
    except AssertionError:
        pass

print(f"Passed: {passed}/{num_test}")
    