import ctypes
import numpy as np
import torch
import torch.nn.functional as F


lib = ctypes.CDLL("../../src/kernals/kernals.so")

lib.SILU.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_uint32
]

lib.SILU.restype = None

num_test = 1000
length = 100

passed = 0
for _ in range(num_test):
    a = np.random.rand(length)
    b = np.zeros_like(a)
    x = torch.tensor(a)
    y = F.silu(x)
    lib.SILU(a,b, length)
    
    try:
        assert (np.allclose(b, y, rtol=1e-9, atol=1e-12))
        passed +=1
    except AssertionError:
        pass

print(f"Passed: {passed}/{num_test}")
    