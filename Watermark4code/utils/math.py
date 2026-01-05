"""
投影与阈值判定等基础工具。
"""

from typing import Union, List
import numpy as np


def project_embeddings(
    embeddings: np.ndarray,  # [N, D]
    directions: np.ndarray,  # [K, D]
) -> np.ndarray:
    """
    计算每个嵌入在每个方向上的投影值：s = W · v（方向与嵌入均视为行向量）。
    返回形状 [N, K]
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings 必须是二维数组 [N, D]")
    if directions.ndim != 2:
        raise ValueError("directions 必须是二维数组 [K, D]")
    if embeddings.shape[1] != directions.shape[1]:
        raise ValueError("embeddings 与 directions 的维度 D 不一致")

    # W: [K,D], V: [N,D] -> S: [N,K]
    return np.matmul(embeddings, directions.T).astype(np.float32)


def compute_thresholds(s0: np.ndarray, t_margin: float) -> np.ndarray:
    """
    计算检测阈值 tau = s0 + t_margin。
    s0: [K] 或 [N,K]
    返回与 s0 同形状的阈值数组。
    """
    return (s0 + float(t_margin)).astype(np.float32)


def detect_bits(
    projections: np.ndarray,  # s'，形状 [K] 或 [N,K]
    s0: np.ndarray,           # 基线投影 s0，形状需可广播到 projections
    t_margin: float = 0.10,
) -> np.ndarray:
    """
    基于阈值 tau = s0 + t_margin 的逐位判定：s' >= tau 判 1，否则判 0。
    返回同形状的 {0,1} 数组。
    """
    tau = compute_thresholds(s0, t_margin)
    return (projections >= tau).astype(np.int32)









