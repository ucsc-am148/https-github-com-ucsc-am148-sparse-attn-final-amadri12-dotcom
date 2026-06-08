"""STUDENT FILE: implement the three block-sparse rung functions."""
"""Alejandro Madrigal"""

import torch
import triton
import triton.language as tl


LOG2E = 1.4426950408889634

@triton.jit
def _dsd_matmul_kernel(
    values,
    row_offsets,
    column_indices,
    Bmat,
    C,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_block_row = tl.program_id(1)
    pid_sub_m = tl.program_id(2)

    offs_m_inner = pid_sub_m * BM + tl.arange(0, BM)
    offs_m = pid_block_row * BLOCK + offs_m_inner
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k_inner = tl.arange(0, BK)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    start = tl.load(row_offsets + pid_block_row)
    end = tl.load(row_offsets + pid_block_row + 1)

    p = start
    while p < end:
        k_block = tl.load(column_indices + p)

        k0 = 0
        while k0 < BLOCK:
            offs_k = k0 + offs_k_inner

            a = tl.load(
                values
                + p * BLOCK * BLOCK
                + offs_m_inner[:, None] * BLOCK
                + offs_k[None, :],
                mask=(offs_m_inner[:, None] < BLOCK) & (offs_k[None, :] < BLOCK),
                other=0.0,
            )

            b = tl.load(
                Bmat
                + (k_block * BLOCK + offs_k[:, None]) * N
                + offs_n[None, :],
                mask=((k_block * BLOCK + offs_k[:, None]) < K)
                & (offs_n[None, :] < N),
                other=0.0,
            )

            acc += tl.dot(a, b, input_precision="ieee")
            k0 += BK

        p += 1

    tl.store(
        C + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def dsd_matmul(values, row_offsets, column_indices, B, M, K, N, block):
    values = values.contiguous()
    row_offsets = row_offsets.contiguous()
    column_indices = column_indices.contiguous()
    B = B.contiguous()

    C = torch.empty((M, N), device=B.device, dtype=torch.float32)

    # Conservative tiles for T4 shared-memory limits.
    BM = 16
    BN = 32
    BK = 32

    grid = (
        triton.cdiv(N, BN),
        M // block,
        triton.cdiv(block, BM),
    )

    _dsd_matmul_kernel[grid](
        values,
        row_offsets,
        column_indices,
        B,
        C,
        M,
        K,
        N,
        block,
        BM,
        BN,
        BK,
        num_warps=4,
        num_stages=1,
    )

    return C


# ============================================================
# A2: sparse flash attention forward
# ============================================================

@triton.jit
def _sparse_flash_forward_kernel(
    Q,
    K,
    V,
    q_row_offsets,
    q_col_indices,
    O,
    L,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k_inner = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    base = pid_bh * T * D
    l_base = pid_bh * T

    q = tl.load(
        Q + base + offs_q[:, None] * D + offs_d[None, :],
        mask=(offs_q[:, None] < T) & (offs_d[None, :] < D),
        other=0.0,
    )

    m_i = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    start = tl.load(q_row_offsets + pid_q)
    end = tl.load(q_row_offsets + pid_q + 1)

    p = start
    while p < end:
        k_block = tl.load(q_col_indices + p)
        offs_k = k_block * BLOCK_K + offs_k_inner

        k = tl.load(
            K + base + offs_k[:, None] * D + offs_d[None, :],
            mask=(offs_k[:, None] < T) & (offs_d[None, :] < D),
            other=0.0,
        )

        v = tl.load(
            V + base + offs_k[:, None] * D + offs_d[None, :],
            mask=(offs_k[:, None] < T) & (offs_d[None, :] < D),
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE_LOG2
        scores = tl.where(
            (offs_q[:, None] < T) & (offs_k[None, :] < T),
            scores,
            -float("inf"),
        )

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p_ij = tl.exp2(scores - m_new[:, None])

        l_new = l_i * alpha + tl.sum(p_ij, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p_ij.to(tl.float16), v)

        m_i = m_new
        l_i = l_new
        p += 1

    out = acc / l_i[:, None]
    l_out = m_i + tl.log2(l_i)

    tl.store(
        O + base + offs_q[:, None] * D + offs_d[None, :],
        out,
        mask=(offs_q[:, None] < T) & (offs_d[None, :] < D),
    )

    tl.store(
        L + l_base + offs_q,
        l_out,
        mask=offs_q < T,
    )


def sparse_flash_forward(Q, K, V, q_row_offsets, q_col_indices,
                         sm_scale, BLOCK_Q, BLOCK_K):
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    q_row_offsets = q_row_offsets.contiguous()
    q_col_indices = q_col_indices.contiguous()

    B, H, T, d = Q.shape

    O = torch.empty_like(Q)
    L = torch.empty((B, H, T), device=Q.device, dtype=torch.float32)

    BLOCK_D = triton.next_power_of_2(d)
    grid = (triton.cdiv(T, BLOCK_Q), B * H)

    _sparse_flash_forward_kernel[grid](
        Q,
        K,
        V,
        q_row_offsets,
        q_col_indices,
        O,
        L,
        T,
        d,
        BLOCK_Q,
        BLOCK_K,
        BLOCK_D,
        sm_scale * LOG2E,
        num_warps=4,
        num_stages=1,
    )

    return O, L


# ============================================================
# A3: sparse flash attention backward
# ============================================================

@triton.jit
def _sparse_flash_backward_dq_kernel(
    Q,
    K,
    V,
    O,
    L,
    dO,
    q_row_offsets,
    q_col_indices,
    dQ,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k_inner = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    base = pid_bh * T * D
    l_base = pid_bh * T

    q = tl.load(
        Q + base + offs_q[:, None] * D + offs_d[None, :],
        mask=(offs_q[:, None] < T) & (offs_d[None, :] < D),
        other=0.0,
    )

    do = tl.load(
        dO + base + offs_q[:, None] * D + offs_d[None, :],
        mask=(offs_q[:, None] < T) & (offs_d[None, :] < D),
        other=0.0,
    )

    o = tl.load(
        O + base + offs_q[:, None] * D + offs_d[None, :],
        mask=(offs_q[:, None] < T) & (offs_d[None, :] < D),
        other=0.0,
    )

    l_i = tl.load(L + l_base + offs_q, mask=offs_q < T, other=0.0)

    # Di = sum_d dO_i[d] * O_i[d]
    D_i = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)

    dq_acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    start = tl.load(q_row_offsets + pid_q)
    end = tl.load(q_row_offsets + pid_q + 1)

    p = start
    while p < end:
        k_block = tl.load(q_col_indices + p)
        offs_k = k_block * BLOCK_K + offs_k_inner

        k = tl.load(
            K + base + offs_k[:, None] * D + offs_d[None, :],
            mask=(offs_k[:, None] < T) & (offs_d[None, :] < D),
            other=0.0,
        )

        v = tl.load(
            V + base + offs_k[:, None] * D + offs_d[None, :],
            mask=(offs_k[:, None] < T) & (offs_d[None, :] < D),
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE_LOG2
        scores = tl.where(
            (offs_q[:, None] < T) & (offs_k[None, :] < T),
            scores,
            -float("inf"),
        )

        # P_ij = exp2(score_log2 - L_i)
        prob = tl.exp2(scores - l_i[:, None])

        # dP = dO @ V^T
        dp = tl.dot(do, tl.trans(v), input_precision="ieee")

        # dS = P * (dP - D_i)
        ds = prob * (dp - D_i[:, None])

        # dQ += dS @ K * scale
        dq_acc += tl.dot(ds.to(tl.float16), k) * SCALE

        p += 1

    tl.store(
        dQ + base + offs_q[:, None] * D + offs_d[None, :],
        dq_acc,
        mask=(offs_q[:, None] < T) & (offs_d[None, :] < D),
    )


@triton.jit
def _sparse_flash_backward_dkdv_kernel(
    Q,
    K,
    V,
    O,
    L,
    dO,
    k_row_offsets,
    k_col_indices,
    dK,
    dV,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_q_inner = tl.arange(0, BLOCK_Q)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    base = pid_bh * T * D
    l_base = pid_bh * T

    k = tl.load(
        K + base + offs_k[:, None] * D + offs_d[None, :],
        mask=(offs_k[:, None] < T) & (offs_d[None, :] < D),
        other=0.0,
    )

    v = tl.load(
        V + base + offs_k[:, None] * D + offs_d[None, :],
        mask=(offs_k[:, None] < T) & (offs_d[None, :] < D),
        other=0.0,
    )

    dk_acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)

    start = tl.load(k_row_offsets + pid_k)
    end = tl.load(k_row_offsets + pid_k + 1)

    p = start
    while p < end:
        q_block = tl.load(k_col_indices + p)
        offs_q = q_block * BLOCK_Q + offs_q_inner

        q = tl.load(
            Q + base + offs_q[:, None] * D + offs_d[None, :],
            mask=(offs_q[:, None] < T) & (offs_d[None, :] < D),
            other=0.0,
        )

        do = tl.load(
            dO + base + offs_q[:, None] * D + offs_d[None, :],
            mask=(offs_q[:, None] < T) & (offs_d[None, :] < D),
            other=0.0,
        )

        o = tl.load(
            O + base + offs_q[:, None] * D + offs_d[None, :],
            mask=(offs_q[:, None] < T) & (offs_d[None, :] < D),
            other=0.0,
        )

        l_i = tl.load(L + l_base + offs_q, mask=offs_q < T, other=0.0)
        D_i = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)

        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE_LOG2
        scores = tl.where(
            (offs_q[:, None] < T) & (offs_k[None, :] < T),
            scores,
            -float("inf"),
        )

        prob = tl.exp2(scores - l_i[:, None])

        # dV += P^T @ dO
        dv_acc += tl.dot(tl.trans(prob.to(tl.float16)), do)

        # dP = dO @ V^T
        dp = tl.dot(do, tl.trans(v), input_precision="ieee")

        # dS = P * (dP - D_i)
        ds = prob * (dp - D_i[:, None])

        # dK += dS^T @ Q * scale
        dk_acc += tl.dot(tl.trans(ds.to(tl.float16)), q) * SCALE

        p += 1

    tl.store(
        dK + base + offs_k[:, None] * D + offs_d[None, :],
        dk_acc,
        mask=(offs_k[:, None] < T) & (offs_d[None, :] < D),
    )

    tl.store(
        dV + base + offs_k[:, None] * D + offs_d[None, :],
        dv_acc,
        mask=(offs_k[:, None] < T) & (offs_d[None, :] < D),
    )


def sparse_flash_backward(Q, K, V, O, L, dO,
                          k_row_offsets, k_col_indices,
                          q_row_offsets, q_col_indices,
                          sm_scale, BLOCK_Q, BLOCK_K):
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    O = O.contiguous()
    L = L.contiguous()
    dO = dO.contiguous()
    k_row_offsets = k_row_offsets.contiguous()
    k_col_indices = k_col_indices.contiguous()
    q_row_offsets = q_row_offsets.contiguous()
    q_col_indices = q_col_indices.contiguous()

    B, H, T, d = Q.shape

    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)

    BLOCK_D = triton.next_power_of_2(d)

    grid_q = (triton.cdiv(T, BLOCK_Q), B * H)
    grid_k = (triton.cdiv(T, BLOCK_K), B * H)

    _sparse_flash_backward_dq_kernel[grid_q](
        Q,
        K,
        V,
        O,
        L,
        dO,
        q_row_offsets,
        q_col_indices,
        dQ,
        T,
        d,
        BLOCK_Q,
        BLOCK_K,
        BLOCK_D,
        sm_scale,
        sm_scale * LOG2E,
        num_warps=4,
        num_stages=1,
    )

    _sparse_flash_backward_dkdv_kernel[grid_k](
        Q,
        K,
        V,
        O,
        L,
        dO,
        k_row_offsets,
        k_col_indices,
        dK,
        dV,
        T,
        d,
        BLOCK_Q,
        BLOCK_K,
        BLOCK_D,
        sm_scale,
        sm_scale * LOG2E,
        num_warps=4,
        num_stages=1,
    )

    return dQ, dK, dV
