"""
Greedy selection loop: measure-then-add iteration.
"""

from typing import Dict, List, Tuple, Optional
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..api import load_encoder
from ..encoder import embed_codes
from ..utils import project_embeddings
from .plan import build_candidates_test_like, build_candidates_by_type, compute_baseline
from .evaluate import measure_gains


def select_and_inject(
    model_dir: str,
    anchor_code: str,
    bits: List[int],
    required_delta,
    secret_key: str = "WATERMARK_SECRET",
    K: int = 100,
    max_iters: int = 8,
    num_workers: int = 48,
    batch_size_for_parallel: int = 20,
    max_accept_per_round: int = 3,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Return dict with keys: final_code, s0, s_after, gains_trace[List[Dict]].
    """
    model, tokenizer, device = load_encoder(model_dir, use_quantization=True)

    # 使用簇中心作为baseline
    s0 = np.array(required_delta["s0"], dtype=np.float32)  # 簇中�?
    
    # 计算原始代码的投影和W矩阵
    baseline = compute_baseline(model_dir, anchor_code, secret_key=secret_key)
    s_original = baseline["s0"].astype(np.float32)
    W = baseline["W"].astype(np.float32)

    # 逐位方案：读�?个维度的独立阈�?
    bitwise_thresholds = required_delta.get("bitwise_thresholds", {})
    
    if not bitwise_thresholds:
        raise ValueError("bitwise_thresholds not found in required_delta")

    trace: List[Dict] = []
    current_code = anchor_code
    current_s = s_original.copy()  # 当前代码的投�?

    for it in range(max_iters):
        # 为不同类型生成不同数量的候选：
        # semantic_preserving: 3*K 个（静态规则，成本低）
        # llm_rewrite: K 个（LLM重写，成本高�?
        all_cands = []
        all_types = []
        aug_configs = [
            ("semantic_preserving", 10 * K),  # 静态规则：3倍候�?
            ("llm_rewrite", 0 * K),               # LLM重写：保持原K
        ]
        # 过滤掉k_count=0的配置，避免不必要的调用
        aug_configs = [(aug_type, k_count) for aug_type, k_count in aug_configs if k_count > 0]
        
        if not aug_configs:
            break  # 没有配置，停止迭�?
        
        with ThreadPoolExecutor(max_workers=len(aug_configs)) as ex:
            futs = {
                ex.submit(
                    build_candidates_by_type,
                    current_code,
                    k_count,
                    aug_type,
                    num_workers,
                    batch_size_for_parallel,
                ): aug_type for aug_type, k_count in aug_configs
            }
            for fut in as_completed(futs):
                aug_type = futs[fut]
                try:
                    cands = fut.result()
                except Exception:
                    cands = []
                # 过滤无变�?
                cands = [c for c in cands if isinstance(c, str) and c.strip() and c.strip() != current_code.strip()]
                if cands:
                    all_cands.extend(cands)
                    all_types.extend([aug_type] * len(cands))

        if not all_cands:
            break  # 无候选，停止迭代

        # 统一计算所有候选的增益
        g = measure_gains(model, tokenizer, W, current_s, all_cands, device=device)

        # 逐位打分：计算当前状态和剩余距离（相对于簇中心）
        offset_now = current_s - s0  # [4]，相对于簇中心的偏移
        
        # 计算每个维度的归一化剩余距�?
        normalized_remainders = []
        for i in range(4):
            m_pos = bitwise_thresholds[i]["m_pos"]
            m_neg = bitwise_thresholds[i]["m_neg"]
            
            if bits[i] == 1:
                # bit=1：目�?offset �?m_pos
                remainder = max(m_pos - offset_now[i], 0.0)
                norm_remainder = remainder / (m_pos + 1e-8)
            else:
                # bit=0：目�?offset �?-m_neg
                remainder = max(offset_now[i] + m_neg, 0.0)
                norm_remainder = remainder / (m_neg + 1e-8)
            
            normalized_remainders.append(norm_remainder)
        
        total_normalized_remainder = sum(normalized_remainders) + 1e-8
        
        # 对每个候选打�?
        scores = np.zeros(len(all_cands), dtype=np.float32)
        
        for cand_idx in range(len(all_cands)):
            gain = g[cand_idx]  # [4]
            score = 0.0
            
            # 逐维打分
            for i in range(4):
                m_pos = bitwise_thresholds[i]["m_pos"]
                m_neg = bitwise_thresholds[i]["m_neg"]
                
                # 权重：归一化剩余距离占比（距离越远权重越大�?
                weight = normalized_remainders[i] / total_normalized_remainder
                
                if bits[i] == 1:
                    # 目标：推正（offset >= m_pos�?
                    offset_after = offset_now[i] + gain[i]
                    
                    if offset_now[i] >= m_pos:
                        # 情况1：当前已达标
                        if gain[i] < 0:
                            # 倒退
                            if offset_after >= m_pos:
                                # 还在安全区，不惩�?
                                score += 0.0
                            else:
                                # 倒退到阈值以下，惩罚
                                remainder_after = m_pos - offset_after
                                norm_remainder_after = remainder_after / (m_pos + 1e-8)
                                temp_total = total_normalized_remainder + norm_remainder_after
                                temp_weight = norm_remainder_after / (temp_total + 1e-8)
                                score -= norm_remainder_after * temp_weight * 6.0
                        else:
                            # 继续推进，小奖励
                            norm_gain = gain[i] / (m_pos + 1e-8)
                            score += norm_gain * 0.1
                    else:
                        # 情况2：当前未达标
                        if gain[i] > 0:
                            # 正向推进，根据权重奖�?
                            norm_progress = min(gain[i] / (m_pos + 1e-8), normalized_remainders[i])
                            score += norm_progress * weight * 6.0
                        else:
                            # 反向倒退，对称惩�?
                            norm_backtrack = abs(gain[i]) / (m_pos + 1e-8)
                            score -= norm_backtrack * weight * 6.0
                
                else:
                    # 目标：推负（offset <= -m_neg�?
                    offset_after = offset_now[i] + gain[i]
                    
                    if offset_now[i] <= -m_neg:
                        # 情况1：当前已达标
                        if gain[i] > 0:
                            # 倒退（正向）
                            if offset_after <= -m_neg:
                                # 还在安全区，不惩�?
                                score += 0.0
                            else:
                                # 倒退到阈值以下，惩罚
                                remainder_after = offset_after - (-m_neg)
                                norm_remainder_after = remainder_after / (m_neg + 1e-8)
                                temp_total = total_normalized_remainder + norm_remainder_after
                                temp_weight = norm_remainder_after / (temp_total + 1e-8)
                                score -= norm_remainder_after * temp_weight * 6.0
                        else:
                            # 继续推进，小奖励
                            norm_gain = abs(gain[i]) / (m_neg + 1e-8)
                            score += norm_gain * 0.1
                    else:
                        # 情况2：当前未达标
                        if gain[i] < 0:
                            # 正向推进，根据权重奖�?
                            norm_progress = min(abs(gain[i]) / (m_neg + 1e-8), normalized_remainders[i])
                            score += norm_progress * weight * 6.0
                        else:
                            # 反向倒退，对称惩�?
                            norm_backtrack = gain[i] / (m_neg + 1e-8)
                            score -= norm_backtrack * weight * 6.0
            
            scores[cand_idx] = score

        try:
            trace.append({
                "iter": it + 1,
                "phase": "all",
                "candidates_gains": g.tolist(),
                "candidates_scores": scores.tolist(),
                "candidates_types": all_types
            })
        except Exception:
            pass

        # 全局选择分数最高且 > 0 的候�?
        order = np.argsort(-scores)
        best_idx = None
        for idx in order:
            if scores[idx] > 0:
                best_idx = int(idx)
                break
        if best_idx is None:
            break  # 无正分候选，停止迭代

        best_gain = g[best_idx]
        best_type = all_types[best_idx]
        trace.append({
            "iter": it + 1,
            "phase": best_type,
            "accepted_index": best_idx,
            "gain": best_gain.tolist(),
        })
        best_code = all_cands[best_idx]
        if save_dir:
            try:
                snap_path = os.path.join(save_dir, f"iter_{it+1:04d}.java")
                with open(snap_path, "w", encoding="utf-8") as f:
                    f.write(best_code)
            except Exception:
                pass
        current_code = best_code
        base2 = compute_baseline(model_dir, current_code, secret_key=secret_key)
        current_s = base2["s0"].astype(np.float32)

        # 达标检查（逐位方案）：相对于簇中心的offset
        offset_now = current_s - s0
        all_satisfied = True
        
        for i in range(4):
            m_pos = bitwise_thresholds[i]["m_pos"]
            m_neg = bitwise_thresholds[i]["m_neg"]
            
            if bits[i] == 1:
                if offset_now[i] < m_pos:
                    all_satisfied = False
                    break
            else:
                if offset_now[i] > -m_neg:
                    all_satisfied = False
                    break
        
        if all_satisfied:
            break

    s_after = current_s
    return {
        "final_code": current_code,
        "s0": s0.tolist(),  # 簇中�?
        "s_after": s_after.tolist(),  # 最终代码投�?
        "trace": trace,
    }


