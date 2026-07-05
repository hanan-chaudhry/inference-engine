import ctypes
import numpy as np
import torch
import platform

_EXT = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}
lib = ctypes.CDLL(f"build/libkernels{_EXT[platform.system()]}")

lib.softmax.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_uint32
]

lib.softmax.restype = None

num_test = 1000
length = 100

passed = 0
for _ in range(num_test):
    a = np.random.rand(length)
    c = np.empty_like(a)

    lib.softmax(a, c, len(a))
    x = torch.tensor(a)
    res = torch.softmax(x, dim = 0)
    torch.set_printoptions(precision=17)
    np.set_printoptions(precision=17)
    try:
        assert (np.allclose(c, res, rtol=1e-9, atol=1e-12))
        passed +=1
    except AssertionError:
        pass

if passed == num_test:
    print(f"\033[92m[PASS]\033[0m softmax ({passed}/{num_test} tests passed)")
else:
    print(f"\033[91m[FAIL]\033[0m softmax ({passed}/{num_test} tests passed)")
    