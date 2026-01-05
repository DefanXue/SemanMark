"""
顶层 API 封装（第一阶段最小闭环）：
- 加载模型（默认量化）、批量嵌入
- 从固定私钥派生 4 个方向
- 基线/检测投影与判定

严格复用 contrastive_learning 评测路径：encoder-only、mean 池化、L2 归一化、max_length=512。
"""

from typing import Dict, List, Tuple
import os
import numpy as np
import torch

try:
    # 包内相对导入（作为包运行时）
    from .encoder import load_best_model, embed_codes
    from .keys import derive_directions
    from .utils import project_embeddings, compute_thresholds, detect_bits
except Exception:
    # 直接脚本导入的回退（在 Watermark4code 目录内运行 python test.py 时）
    import os, sys
    _CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    if _CUR_DIR not in sys.path:
        sys.path.append(_CUR_DIR)
    from encoder import load_best_model, embed_codes  # type: ignore
    from keys.directions import derive_directions  # type: ignore
    from utils.math import project_embeddings, compute_thresholds, detect_bits  # type: ignore


DEFAULT_SECRET = "XDF"
DEFAULT_D = 768
DEFAULT_K = 4
DEFAULT_T_MARGIN = 0.10


def load_encoder(model_dir: str, use_quantization: bool = True):
    """
    包装底层加载，返回 (model, tokenizer, device)
    """
    model, tokenizer = load_best_model(model_dir=model_dir, use_quantization=use_quantization)
    device = next(model.parameters()).device
    return model, tokenizer, device


def compute_baseline_s0(
    model_dir: str,
    codes: List[str],
    secret_key: str = DEFAULT_SECRET,
    use_quantization: bool = True,
    max_length: int = 512,
    batch_size: int = 32,
) -> Dict:
    """
    计算给定代码集合的基线投影：
    - 返回包含 embeddings, directions, s0 的字典（均为 numpy 数组）
    """
    model, tokenizer, device = load_encoder(model_dir, use_quantization)
    embs = embed_codes(model, tokenizer, codes, max_length=max_length, batch_size=batch_size, device=device)
    dirs = derive_directions(secret_key=secret_key, d=embs.shape[1], k=DEFAULT_K)
    s0 = project_embeddings(embs, dirs)  # [N,K]
    return {
        "embeddings": embs,
        "directions": dirs,
        "s0": s0,
    }


def detect_bits_for_codes(
    model_dir: str,
    codes: List[str],
    directions: np.ndarray,
    s0: np.ndarray,
    use_quantization: bool = True,
    t_margin: float = DEFAULT_T_MARGIN,
    max_length: int = 512,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对代码计算投影与比特判定：
    - 返回 (projections, bits)，形状均为 [N,K]
    """
    model, tokenizer, device = load_encoder(model_dir, use_quantization)
    embs = embed_codes(model, tokenizer, codes, max_length=max_length, batch_size=batch_size, device=device)
    projections = project_embeddings(embs, directions)
    bits = detect_bits(projections, s0, t_margin=t_margin)
    return projections, bits


