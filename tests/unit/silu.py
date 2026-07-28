import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import platform


_EXT = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}
lib = ctypes.CDLL(f"build/libkernels{_EXT[platform.system()]}")

lib.SILU.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    ctypes.c_uint32
]

lib.SILU.restype = None

num_test = 1000
length = 100

passed = 0
for _ in range(num_test):
    a = np.random.rand(length).astype(np.float32)
    b = np.zeros_like(a)
    x = torch.tensor(a)
    y = F.silu(x)
    lib.SILU(a,b, length)
    
    try:
        assert (np.allclose(b, y, rtol=1e-3, atol=1e-4))
        passed +=1
    except AssertionError:
        pass

print(f"Passed: {passed}/{num_test}")
    