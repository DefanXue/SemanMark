"""
从私钥派生 k 个在 R^d 空间内近似正交且单位范数的方向向量：
- secret_key: 固定字符串（当前按要求写死为 "XDF"）
- d: 维度（与编码器输出一致，默认 768）
- k: 比特数（默认 4）

实现：SHA-256(secret_key) 作为可复现种子 → 生成 k 个 U[-1,1] 随机向量 → Gram-Schmidt 正交化并归一化。
"""

from typing import Tuple
import hashlib
import numpy as np


def _gram_schmidt_orthonormalize(matrix: np.ndarray) -> np.ndarray:
    """
    对给定矩阵的行向量执行 Gram-Schmidt 正交化并单位化。
    要求：matrix 形状 [k, d]
    返回：形状 [k, d] 的单位正交向量组。
    """
    k, d = matrix.shape
    orth = np.zeros((k, d), dtype=np.float64)
    for i in range(k):
        vec = matrix[i].astype(np.float64)
        for j in range(i):
            proj = np.dot(orth[j], vec) * orth[j]
            vec = vec - proj
        norm = np.linalg.norm(vec) + 1e-12
        orth[i] = vec / norm
    return orth.astype(np.float32)


def derive_directions(secret_key: str = "XDF", d: int = 768, k: int = 4) -> np.ndarray:
    """
    从私钥派生 k 个方向（单位长度，近似正交），可复现。

    返回：np.ndarray，形状 [k, d]
    """
    if k <= 0:
        raise ValueError("k 必须为正整数")
    if d <= 0:
        raise ValueError("d 必须为正整数")

    # 使用 SHA-256 作为 PRNG 种子来源
    digest = hashlib.sha256(secret_key.encode("utf-8")).digest()
    # RandomState 仅接受 32 位无符号整数；使用前4字节作为种子
    seed32 = int.from_bytes(digest[:4], byteorder="big", signed=False)  # 0..2**32-1
    rng = np.random.RandomState(seed32)

    # 采样 k 个向量，均匀分布于 [-1,1]
    raw = rng.uniform(low=-1.0, high=1.0, size=(k, d)).astype(np.float32)
    ortho = _gram_schmidt_orthonormalize(raw)
    return ortho


