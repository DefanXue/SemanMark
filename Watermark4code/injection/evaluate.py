"""
Evaluate candidate gains against baseline.
"""

import numpy as np
from typing import List

from ..encoder import embed_codes
from ..utils import project_embeddings


def measure_gains(model, tokenizer, W: np.ndarray, s0: np.ndarray, candidates: List[str], max_length: int = 512, batch_size: int = 32, device=None) -> np.ndarray:
    if not candidates:
        return np.zeros((0, 4), dtype=np.float32)
    v_cands = embed_codes(model, tokenizer, candidates, max_length=max_length, batch_size=batch_size, device=device)
    s_cands = project_embeddings(v_cands, W)
    g = (s_cands - s0[None, :]).astype(np.float32)
    return g









