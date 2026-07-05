import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import platform

_EXT = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}
lib = ctypes.CDLL(f"build/libkernels{_EXT[platform.system()]}")

lib.attention.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
    ctypes.c_uint32,
    ctypes.c_uint32,
    np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS"),
    ctypes.c_bool
]

lib.attention.restype = None

num_test = 1000


passed = 0
L = 10
D = 20
for _ in range(num_test):
    Q_np = np.asfortranarray(np.random.randn(L, D))
    K_np = np.asfortranarray(np.random.randn(L, D))
    V_np = np.asfortranarray(np.random.randn(L, D))
    Q = torch.from_numpy(Q_np).unsqueeze(0).unsqueeze(0)
    K = torch.from_numpy(K_np).unsqueeze(0).unsqueeze(0)
    V = torch.from_numpy(V_np).unsqueeze(0).unsqueeze(0)

    res = F.scaled_dot_product_attention(
        Q,
        K,
        V,
        is_causal=True,
        dropout_p=0.0,
    )
    res = np.asfortranarray(res.squeeze(0).squeeze(0).numpy())
    out = np.empty((L, D), dtype=np.float64, order="F")
    lib.attention(
        Q_np,
        K_np,
        V_np,
        L,
        D,
        out,
        True,
    )
    try:
        assert (np.allclose(out, res, rtol=1e-9, atol=1e-12))
        passed +=1
    except AssertionError:
        print(out)
        print("\n")
        print(res)
print(f"(With masking) Passed: {passed}/{num_test}")

passed = 0
for _ in range(num_test):
    Q_np = np.asfortranarray(np.random.randn(L, D))
    K_np = np.asfortranarray(np.random.randn(L, D))
    V_np = np.asfortranarray(np.random.randn(L, D))
    Q = torch.from_numpy(Q_np).unsqueeze(0).unsqueeze(0)
    K = torch.from_numpy(K_np).unsqueeze(0).unsqueeze(0)
    V = torch.from_numpy(V_np).unsqueeze(0).unsqueeze(0)

    res = F.scaled_dot_product_attention(
        Q,
        K,
        V,
        is_causal=False,
        dropout_p=0.0,
    )
    res = np.asfortranarray(res.squeeze(0).squeeze(0).numpy())
    out = np.empty((L, D), dtype=np.float64, order="F")
    lib.attention(
        Q_np,
        K_np,
        V_np,
        L,
        D,
        out,
        False,
    )
    try:
        assert (np.allclose(out, res, rtol=1e-9, atol=1e-12))
        passed +=1
    except AssertionError:
        print(out)
        print("\n")
        print(res)
print(f"(Without masking) Passed: {passed}/{num_test}")
