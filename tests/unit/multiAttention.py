import ctypes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import platform

_EXT = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}
lib = ctypes.CDLL(f"build/libkernels{_EXT[platform.system()]}")

lib.multiAttention.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_uint32,
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_bool,
]

lib.multiAttention.restype = None

torch.set_grad_enabled(False)

num_tests = 1000

L = 10
D = 20
NUM_HEADS = 4

passed = 0

for test in range(num_tests):

    X_np = np.random.randn(L, D)

    W_Q_np = np.random.randn(D, D)
    W_K_np = np.random.randn(D, D)
    W_V_np = np.random.randn(D, D) 
    W_O_np = np.random.randn(D, D) 
    
    X = torch.from_numpy(X_np).unsqueeze(0)

    mha = nn.MultiheadAttention(
        embed_dim=D,
        num_heads=NUM_HEADS,
        bias=False,
        batch_first=True,
        dtype=torch.float64,
    )

    mha.in_proj_weight.copy_(
        torch.from_numpy(
            np.vstack((
                W_Q_np.T,
                W_K_np.T,
                W_V_np.T,
            ))
        )
    )

    mha.out_proj.weight.copy_(
        torch.from_numpy(W_O_np.T)
    )

    mask = nn.Transformer.generate_square_subsequent_mask(
        L,
        dtype=torch.float64,
    )

    res, _ = mha(
        X,
        X,
        X,
        attn_mask=mask,
        need_weights=False,
    )

    res = np.array(
        res.squeeze(0).numpy()
    )

    out = np.empty((L, D), dtype=np.float64, order="C")

    lib.multiAttention(
        W_Q_np,
        W_K_np,
        W_V_np,
        X_np,
        L,
        D,
        NUM_HEADS,
        NUM_HEADS,
        W_O_np,
        out,
        True,
    )

    try:
        np.testing.assert_allclose(
            out,
            res,
            rtol=1e-9,
            atol=1e-12,
        )
        passed += 1
    except AssertionError as e:
        print(f"\nTest {test} failed\n")
        print(e)
        print("\nC output:\n", out)
        print("\nPyTorch output:\n", res)
        break

print(f"\n (Masked MHA)Passed: {passed}/{num_tests}")

passed = 0

for test in range(num_tests):

    X_np = np.random.randn(L, D)

    W_Q_np = np.random.randn(D, D)
    W_K_np = np.random.randn(D, D)
    W_V_np = np.random.randn(D, D) 
    W_O_np = np.random.randn(D, D) 
    
    X = torch.from_numpy(X_np).unsqueeze(0)

    mha = nn.MultiheadAttention(
        embed_dim=D,
        num_heads=NUM_HEADS,
        bias=False,
        batch_first=True,
        dtype=torch.float64,
    )

    mha.in_proj_weight.copy_(
        torch.from_numpy(
            np.vstack((
                W_Q_np.T,
                W_K_np.T,
                W_V_np.T,
            ))
        )
    )

    mha.out_proj.weight.copy_(
        torch.from_numpy(W_O_np.T)
    )

    mask = nn.Transformer.generate_square_subsequent_mask(
        L,
        dtype=torch.float64,
    )

    res, _ = mha(
        X,
        X,
        X,
        attn_mask=None,
        need_weights=False,
    )

    res = np.array(
        res.squeeze(0).numpy()
    )

    out = np.empty((L, D), dtype=np.float64, order="C")

    lib.multiAttention(
        W_Q_np,
        W_K_np,
        W_V_np,
        X_np,
        L,
        D,
        NUM_HEADS,
        NUM_HEADS,
        W_O_np,
        out,
        False,
    )

    try:
        np.testing.assert_allclose(
            out,
            res,
            rtol=1e-9,
            atol=1e-12,
        )
        passed += 1
    except AssertionError as e:
        print(f"\nTest {test} failed\n")
        print(e)
        print("\nC output:\n", out)
        print("\nPyTorch output:\n", res)
        break

print(f"\n (Without Mask MHA)Passed: {passed}/{num_tests}")


B = 1  
L = 10
D = 20
passed = 0
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
num_tests = 1000

HEAD_DIM = D // NUM_Q_HEADS

for _ in range(num_tests):
    X_np = np.random.randn(L, D)
    W_Q_np = np.random.randn(D, D)
    W_K_np = np.random.randn(D, NUM_KV_HEADS * HEAD_DIM)
    W_V_np = np.random.randn(D, NUM_KV_HEADS * HEAD_DIM)
    W_O_np = np.random.randn(D, D)

    X = torch.from_numpy(X_np).unsqueeze(0)
    W_Q = torch.from_numpy(W_Q_np)
    W_K = torch.from_numpy(W_K_np)
    W_V = torch.from_numpy(W_V_np)
    W_O = torch.from_numpy(W_O_np)

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    Q = Q.view(B, L, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
    K = K.view(B, L, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    V = V.view(B, L, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

    attn = F.scaled_dot_product_attention(
        query=Q,
        key=K,
        value=V,
        enable_gqa=True,
        is_causal=False,
    )

    attn = attn.transpose(1, 2).reshape(B, L, D)

    res = attn @ W_O
    res = res.squeeze(0).numpy()
    out = np.empty((L, D), dtype=np.float64, order="C")
    lib.multiAttention(
        W_Q_np,
        W_K_np,
        W_V_np,
        X_np,
        L,
        D,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        W_O_np,
        out,
        False,
    )

    try:
        np.testing.assert_allclose(
            out,
            res,
            rtol=1e-9,
            atol=1e-12,
        )
        passed += 1
    except AssertionError as e:
        print(f"\nTest {test} failed\n")
        print(e)
        print("\nC output:\n", out)
        print("\nPyTorch output:\n", res)
print(f"\n (Without mask GQA)Passed: {passed}/{num_tests}")


passed = 0
for _ in range(num_tests):
    X_np = np.random.randn(L, D)
    W_Q_np = np.random.randn(D, D)
    W_K_np = np.random.randn(D, NUM_KV_HEADS * HEAD_DIM)
    W_V_np = np.random.randn(D, NUM_KV_HEADS * HEAD_DIM)
    W_O_np = np.random.randn(D, D)

    X = torch.from_numpy(X_np).unsqueeze(0)
    W_Q = torch.from_numpy(W_Q_np)
    W_K = torch.from_numpy(W_K_np)
    W_V = torch.from_numpy(W_V_np)
    W_O = torch.from_numpy(W_O_np)

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    Q = Q.view(B, L, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
    K = K.view(B, L, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    V = V.view(B, L, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

    attn = F.scaled_dot_product_attention(
        query=Q,
        key=K,
        value=V,
        enable_gqa=True,
        is_causal=True,
    )

    attn = attn.transpose(1, 2).reshape(B, L, D)

    res = attn @ W_O
    res = res.squeeze(0).numpy()
    out = np.empty((L, D), dtype=np.float64, order="C")
    lib.multiAttention(
        W_Q_np,
        W_K_np,
        W_V_np,
        X_np,
        L,
        D,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        W_O_np,
        out,
        True,
    )

    try:
        np.testing.assert_allclose(
            out,
            res,
            rtol=1e-9,
            atol=1e-12,
        )
        passed += 1
    except AssertionError as e:
        print(f"\nTest {test} failed\n")
        print(e)
        print("\nC output:\n", out)
        print("\nPyTorch output:\n", res)
print(f"\n (Masked GQA)Passed: {passed}/{num_tests}")