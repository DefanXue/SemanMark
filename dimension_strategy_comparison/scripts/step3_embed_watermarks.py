"""
Step 3: 使用选定的维度嵌入水印

输入：
  - data/test_codes.jsonl
  - results/strategy_X/embedding/run_XXXX/selected_dimensions.json

输出：
  - results/strategy_X/embedding/run_XXXX/final.json
  - results/strategy_X/embedding/run_XXXX/watermarked.java
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Watermark4code.injection.plan import compute_required_delta_per_anchor, build_candidates_test_like
from Watermark4code.injection.greedy import select_and_inject
from Watermark4code.encoder.loader import load_best_model
from Watermark4code.keys.directions import derive_directions


def load_test_codes(base_config):
    """从原始数据集加载测试代码"""
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset_loader import DatasetLoader
    loader = DatasetLoader(base_config)

    start_idx = base_config['data_split']['embedding_and_testing']['start_index']
    num_codes = base_config['data_split']['embedding_and_testing']['num_codes']

    return loader.load_codes(num_codes, start_idx, split='test')


def compute_extreme_codes_for_strategy1(code, directions, K, num_workers, batch_size_for_parallel, model_dir, language='java'):
    """
    为策略1计算extreme_codes和extreme_cluster_centers（用于step4b，支持多语言）
    
    Args:
        code: str, 原始代码
        directions: np.ndarray, shape (k, 768)
        K: int, 候选数量
        num_workers: int
        batch_size_for_parallel: int
        model_dir: str
        language: 编程语言，默认'java'
    
    Returns:
        tuple: (extreme_codes, extreme_cluster_centers)
    """
    from Watermark4code.encoder.loader import load_best_model, embed_codes
    from Watermark4code.utils.math import project_embeddings
    import torch
    
    # ✅ 参数化 k
    k = directions.shape[0]
    
    # 生成候选
    cands = build_candidates_test_like(code, K, num_workers, batch_size_for_parallel, language=language)
    cands = [c for c in cands if isinstance(c, str) and c.strip() and c.strip() != code.strip()]
    
    if not cands:
        # ✅ 无候选时返回空的extreme_codes（参数化 k）
        empty_extreme_codes = {}
        empty_extreme_cluster_centers = {}
        for i in range(k):  # ✅ 参数化
            empty_extreme_codes[str(i)] = {"pos_code": code, "pos_projection": 0.0, "neg_code": code, "neg_projection": 0.0}
            empty_extreme_cluster_centers[str(i)] = {"pos_cluster": [0.0] * k, "neg_cluster": [0.0] * k}  # ✅ 参数化
        return empty_extreme_codes, empty_extreme_cluster_centers
    
    # 加载模型并编码
    model, tokenizer = load_best_model(model_dir)
    device = next(model.parameters()).device
    
    v_cands = embed_codes(model, tokenizer, cands, device=device)
    s_cands = project_embeddings(v_cands, directions)
    
    # 找到每个维度的极值变体
    extreme_codes = {}
    extreme_cluster_centers = {}
    
    for i in range(k):  # ✅ 参数化 k
        pos_idx = int(np.argmax(s_cands[:, i]))
        neg_idx = int(np.argmin(s_cands[:, i]))
        pos_code = cands[pos_idx]
        neg_code = cands[neg_idx]
        
        extreme_codes[str(i)] = {
            "pos_code": pos_code,
            "pos_projection": float(s_cands[pos_idx, i]),
            "neg_code": neg_code,
            "neg_projection": float(s_cands[neg_idx, i])
        }
        
        # 计算正极值的簇中心
        try:
            pos_cands = build_candidates_test_like(pos_code, K, num_workers, batch_size_for_parallel, language=language)
            pos_cands = [c for c in pos_cands if isinstance(c, str) and c.strip() and c.strip() != pos_code.strip()]
            if pos_cands:
                v_pos_cands = embed_codes(model, tokenizer, pos_cands, device=device)
                s_pos_cands = project_embeddings(v_pos_cands, directions)
                pos_cluster_center = np.array([np.median(s_pos_cands[:, j]) for j in range(k)])  # ✅ 参数化 k
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                v_pos = embed_codes(model, tokenizer, [pos_code], device=device)
                s_pos = project_embeddings(v_pos, directions)[0]
                pos_cluster_center = s_pos
        except Exception:
            v_pos = embed_codes(model, tokenizer, [pos_code], device=device)
            s_pos = project_embeddings(v_pos, directions)[0]
            pos_cluster_center = s_pos
        
        # 计算负极值的簇中心
        try:
            neg_cands = build_candidates_test_like(neg_code, K, num_workers, batch_size_for_parallel, language=language)
            neg_cands = [c for c in neg_cands if isinstance(c, str) and c.strip() and c.strip() != neg_code.strip()]
            if neg_cands:
                v_neg_cands = embed_codes(model, tokenizer, neg_cands, device=device)
                s_neg_cands = project_embeddings(v_neg_cands, directions)
                neg_cluster_center = np.array([np.median(s_neg_cands[:, j]) for j in range(k)])  # ✅ 参数化 k
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                v_neg = embed_codes(model, tokenizer, [neg_code], device=device)
                s_neg = project_embeddings(v_neg, directions)[0]
                neg_cluster_center = s_neg
        except Exception:
            v_neg = embed_codes(model, tokenizer, [neg_code], device=device)
            s_neg = project_embeddings(v_neg, directions)[0]
            neg_cluster_center = s_neg
        
        extreme_cluster_centers[str(i)] = {
            "pos_cluster": pos_cluster_center.tolist(),
            "neg_cluster": neg_cluster_center.tolist()
        }
    
    return extreme_codes, extreme_cluster_centers


def compute_thresholds_with_custom_directions(code, bits, directions, config, model, tokenizer, device, language='java'):
    """
    使用自定义方向计算阈值，基于簇中心s0（支持多语言）
    
    完全复刻原始compute_required_delta_per_anchor的逻辑，但使用自定义directions
    
    Args:
        language: 编程语言 ('java'/'javascript'/'cpp')，默认'java'
    """
    from Watermark4code.injection.plan import build_candidates_test_like
    from Watermark4code.utils.math import project_embeddings
    from Watermark4code.encoder.loader import embed_codes
    
    # 1) 生成候选（从config读取簇中心计算的变体数量）
    K_thr = config.get('cluster_variants', 100)  # 默认100，但优先从config读取
    try:
        cands, review_stats = build_candidates_test_like(
            code,
            max(1, K_thr),
            num_workers=config['num_workers'],
            batch_size_for_parallel=config['batch_size_for_parallel'],
            language=language,  # ✅ 传递language参数
        )
    except Exception:
        cands = []
        review_stats = {"passed_count": 0, "failed_reasons": {}}
    
    # 过滤无变化
    cands = [c for c in cands if isinstance(c, str) and c.strip() and c.strip() != code.strip()]
    
    if not cands:
        # 无候选时：使用原始代码作为median_code，返回完整的字典结构
        v_anchor = embed_codes(model, tokenizer, [code], device=device)
        s_anchor = project_embeddings(v_anchor, directions)[0]
        
        k = directions.shape[0]
        
        # 生成k维的零阈值和空极值配置
        bitwise_thresholds = {}
        extreme_codes = {}
        extreme_cluster_centers = {}
        
        for i in range(k):
            bitwise_thresholds[str(i)] = {
                "q_pos": 0.0, "m_pos": 0.0, "T_pos": 0.0, 
                "q_neg": 0.0, "m_neg": 0.0, "T_neg": 0.0
            }
            extreme_codes[str(i)] = {
                "pos_code": code,
                "pos_projection": float(s_anchor[i]),
                "neg_code": code,
                "neg_projection": float(s_anchor[i])
            }
            extreme_cluster_centers[str(i)] = {
                "pos_cluster": s_anchor.tolist(),
                "neg_cluster": s_anchor.tolist()
            }
        
        return {
            "s0": s_anchor.tolist(),
            "median_code": code,
            "median_cluster_center": s_anchor.tolist(),
            "extreme_codes": extreme_codes,
            "extreme_cluster_centers": extreme_cluster_centers,
            "bitwise_thresholds": bitwise_thresholds
        }
    
    # 2) 编码并投影（使用自定义方向）
    v_anchor = embed_codes(model, tokenizer, [code], device=device)
    v_cands = embed_codes(model, tokenizer, cands, device=device)
    
    s_anchor = project_embeddings(v_anchor, directions)[0]  # [4]
    s_cands = project_embeddings(v_cands, directions)       # [K,4]
    
    # ✅ 清理编码后的GPU缓存
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 3) 计算簇中心s0（✅ 关键：各维度的中位数）
    k = directions.shape[0]  # ✅ 参数化 k
    s0 = np.array([np.median(s_cands[:, i]) for i in range(k)])
    
    # === 新增：找到距离s0最近的变体（median_code）===
    distances = np.linalg.norm(s_cands - s0, axis=1)
    median_idx = int(np.argmin(distances))
    median_code = cands[median_idx]
    
    # === 计算median_code的簇中心 ===
    try:
        median_cands = build_candidates_test_like(
            median_code,
            max(1, K_thr),
            num_workers=config['num_workers'],
            batch_size_for_parallel=config['batch_size_for_parallel'],
            language=language,  # ✅ 传递language参数
        )
        median_cands = [c for c in median_cands if isinstance(c, str) and c.strip() and c.strip() != median_code.strip()]
        if median_cands:
            v_median_cands = embed_codes(model, tokenizer, median_cands, device=device)
            s_median_cands = project_embeddings(v_median_cands, directions)
            median_cluster_center = np.array([np.median(s_median_cands[:, i]) for i in range(k)])  # ✅ 参数化 k
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # 无候选时使用median_code自身的投影
            v_median = embed_codes(model, tokenizer, [median_code], device=device)
            s_median = project_embeddings(v_median, directions)[0]
            median_cluster_center = s_median
    except Exception:
        # 出错时使用median_code自身的投影
        v_median = embed_codes(model, tokenizer, [median_code], device=device)
        s_median = project_embeddings(v_median, directions)[0]
        median_cluster_center = s_median
    
    # === 新增：找到每个维度的极值变体（extreme_codes）===
    extreme_codes = {}
    extreme_cluster_centers = {}
    for i in range(k):  # ✅ 参数化 k
        pos_idx = int(np.argmax(s_cands[:, i]))
        neg_idx = int(np.argmin(s_cands[:, i]))
        pos_code = cands[pos_idx]
        neg_code = cands[neg_idx]
        
        extreme_codes[str(i)] = {
            "pos_code": pos_code,
            "pos_projection": float(s_cands[pos_idx, i]),
            "neg_code": neg_code,
            "neg_projection": float(s_cands[neg_idx, i])
        }
        
        # 计算正极值的簇中心
        try:
            pos_cands = build_candidates_test_like(
                pos_code,
                max(1, K_thr),
                num_workers=config['num_workers'],
                batch_size_for_parallel=config['batch_size_for_parallel'],
                language=language,  # ✅ 传递language参数
            )
            pos_cands = [c for c in pos_cands if isinstance(c, str) and c.strip() and c.strip() != pos_code.strip()]
            if pos_cands:
                v_pos_cands = embed_codes(model, tokenizer, pos_cands, device=device)
                s_pos_cands = project_embeddings(v_pos_cands, directions)
                pos_cluster_center = np.array([np.median(s_pos_cands[:, j]) for j in range(k)])  # ✅ 参数化 k
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                v_pos = embed_codes(model, tokenizer, [pos_code], device=device)
                s_pos = project_embeddings(v_pos, directions)[0]
                pos_cluster_center = s_pos
        except Exception:
            v_pos = embed_codes(model, tokenizer, [pos_code], device=device)
            s_pos = project_embeddings(v_pos, directions)[0]
            pos_cluster_center = s_pos
        
        # 计算负极值的簇中心
        try:
            neg_cands = build_candidates_test_like(
                neg_code,
                max(1, K_thr),
                num_workers=config['num_workers'],
                batch_size_for_parallel=config['batch_size_for_parallel'],
                language=language,  # ✅ 传递language参数
            )
            neg_cands = [c for c in neg_cands if isinstance(c, str) and c.strip() and c.strip() != neg_code.strip()]
            if neg_cands:
                v_neg_cands = embed_codes(model, tokenizer, neg_cands, device=device)
                s_neg_cands = project_embeddings(v_neg_cands, directions)
                neg_cluster_center = np.array([np.median(s_neg_cands[:, j]) for j in range(k)])  # ✅ 参数化 k
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                v_neg = embed_codes(model, tokenizer, [neg_code], device=device)
                s_neg = project_embeddings(v_neg, directions)[0]
                neg_cluster_center = s_neg
        except Exception:
            v_neg = embed_codes(model, tokenizer, [neg_code], device=device)
            s_neg = project_embeddings(v_neg, directions)[0]
            neg_cluster_center = s_neg
        
        extreme_cluster_centers[str(i)] = {
            "pos_cluster": pos_cluster_center.tolist(),
            "neg_cluster": neg_cluster_center.tolist()
        }
    
    # 4) 计算阈值（✅ 相对于簇中心s0）
    quantile = config['quantile']
    bitwise_thresholds = {}
    
    for i in range(k):  # ✅ 参数化 k
        # ✅ 相对于簇中心
        offsets = s_cands[:, i] - s0[i]
        
        pos_offsets = [float(d) for d in offsets if d > 0]
        neg_offsets = [float(abs(d)) for d in offsets if d < 0]
        
        if pos_offsets:
            radius_pos = max(pos_offsets)
            m_pos = radius_pos * quantile
            T_pos = radius_pos * quantile
        else:
            m_pos = T_pos = 0.0
        
        if neg_offsets:
            radius_neg = max(neg_offsets)
            m_neg = radius_neg * quantile
            T_neg = radius_neg * quantile
        else:
            m_neg = T_neg = 0.0
        
        bitwise_thresholds[str(i)] = {
            "q_pos": float(radius_pos) if pos_offsets else 0.0,
            "m_pos": float(m_pos),
            "T_pos": float(T_pos),
            "q_neg": float(radius_neg) if neg_offsets else 0.0,
            "m_neg": float(m_neg),
            "T_neg": float(T_neg)
        }
    
    return {
        "s0": s0.tolist(),
        "median_code": median_code,
        "median_cluster_center": median_cluster_center.tolist(),
        "extreme_codes": extreme_codes,
        "extreme_cluster_centers": extreme_cluster_centers,
        "bitwise_thresholds": bitwise_thresholds
    }


def greedy_inject_with_custom_directions(code, bits, directions, s0, bitwise_thresholds, config, model, tokenizer, device, save_dir=None, language='java'):
    """
    使用自定义方向的贪心注入（支持多语言）
    
    完全复刻原始select_and_inject的逻辑，但使用自定义directions和s0
    
    Args:
        save_dir: Optional[str], 保存中间变体的目录路径
        language: 编程语言 ('java'/'javascript'/'cpp')，默认'java'
    """
    from Watermark4code.injection.plan import build_candidates_by_type
    from Watermark4code.utils.math import project_embeddings
    from Watermark4code.injection.evaluate import measure_gains
    from Watermark4code.encoder.loader import embed_codes
    
    # 计算原始代码的投影
    v_original = embed_codes(model, tokenizer, [code], device=device)
    s_original = project_embeddings(v_original, directions)[0]  # [4]
    
    current_code = code
    current_s = s_original.copy()
    
    trace = []
    
    # === 修正1：添加历史变体记录 ===
    history_variants = [{
        "iter": 0,
        "code": current_code,
        "s": current_s.copy(),
    }]
    
    for it in range(config['max_iters']):
        # 生成候选（与原始一致：10*K）
        try:
            cands = build_candidates_by_type(
                current_code,
                10 * config['K'],  # ✅ 与原始一致：10*K
                "semantic_preserving",
                config['num_workers'],
                config['batch_size_for_parallel'],
                language=language  # ✅ 传递language参数
            )
        except Exception:
            cands = []
        
        # 过滤无变化
        cands = [c for c in cands if isinstance(c, str) and c.strip() and c.strip() != current_code.strip()]
        
        if not cands:
            break
        
        # 记录候选类型（与原代码一致）
        cand_types = ['semantic_preserving'] * len(cands)
        
        # 计算增益
        g = measure_gains(model, tokenizer, directions, current_s, cands, device=device)
        
        # ✅ 计算相对于簇中心s0的偏移
        offset_now = current_s - np.array(s0)
        
        # 计算每个维度的归一化剩余距离（完全复刻原始逻辑）
        normalized_remainders = []
        k = len(bits)  # ✅ 参数化 k
        for i in range(k):  # ✅ 参数化
            m_pos = bitwise_thresholds[str(i)]["m_pos"]
            m_neg = bitwise_thresholds[str(i)]["m_neg"]
            
            if bits[i] == 1:
                # bit=1：目标 offset ≥ m_pos
                remainder = max(m_pos - offset_now[i], 0.0)
                norm_remainder = remainder / (m_pos + 1e-8)
            else:
                # bit=0：目标 offset ≤ -m_neg
                remainder = max(offset_now[i] + m_neg, 0.0)
                norm_remainder = remainder / (m_neg + 1e-8)
            
            normalized_remainders.append(norm_remainder)
        
        total_normalized_remainder = sum(normalized_remainders) + 1e-8
        
        # 对每个候选打分（完全复刻原始逻辑）
        scores = np.zeros(len(cands), dtype=np.float32)
        
        for cand_idx in range(len(cands)):
            gain = g[cand_idx]  # [4]
            score = 0.0
            
            # 逐维打分
            for i in range(k):  # ✅ 参数化 k
                m_pos = bitwise_thresholds[str(i)]["m_pos"]
                m_neg = bitwise_thresholds[str(i)]["m_neg"]
                
                # 权重：归一化剩余距离占比（距离越远权重越大）
                weight = normalized_remainders[i] / total_normalized_remainder
                
                if bits[i] == 1:
                    # 目标：推正（offset >= m_pos）
                    offset_after = offset_now[i] + gain[i]
                    
                    if offset_now[i] >= m_pos:
                        # 情况1：当前已达标
                        if gain[i] < 0:
                            # 倒退
                            if offset_after >= m_pos:
                                # 还在安全区，不惩罚
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
                            # 正向推进，根据权重奖励
                            norm_progress = min(gain[i] / (m_pos + 1e-8), normalized_remainders[i])
                            score += norm_progress * weight * 6.0
                        else:
                            # 反向倒退，对称惩罚
                            norm_backtrack = abs(gain[i]) / (m_pos + 1e-8)
                            score -= norm_backtrack * weight * 6.0
                
                else:
                    # 目标：推负（offset <= -m_neg）
                    offset_after = offset_now[i] + gain[i]
                    
                    if offset_now[i] <= -m_neg:
                        # 情况1：当前已达标
                        if gain[i] > 0:
                            # 倒退（正向）
                            if offset_after <= -m_neg:
                                # 还在安全区，不惩罚
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
                            # 正向推进，根据权重奖励
                            norm_progress = min(abs(gain[i]) / (m_neg + 1e-8), normalized_remainders[i])
                            score += norm_progress * weight * 6.0
                        else:
                            # 反向倒退，对称惩罚
                            norm_backtrack = gain[i] / (m_neg + 1e-8)
                            score -= norm_backtrack * weight * 6.0
            
            scores[cand_idx] = score
        
        # === 记录所有候选的信息（与原代码一致）===
        try:
            trace.append({
                "iter": it + 1,
                "phase": "all",
                "candidates_gains": g.tolist(),
                "candidates_scores": scores.tolist(),
                "candidates_types": cand_types
            })
        except Exception:
            pass
        
        # === 修正2：允许负分选择（与原代码一致）===
        order = np.argsort(-scores)
        best_idx = int(order[0])  # 直接取最高分（允许负数）
        
        # 接受最佳候选
        current_code = cands[best_idx]
        best_type = cand_types[best_idx]
        
        # === 保存中间变体（与原始流程一致）===
        if save_dir:
            try:
                # ✅ 根据语言动态选择文件扩展名（支持Java、JavaScript和C++）
                if language == 'javascript':
                    file_extension = 'js'
                elif language == 'cpp':
                    file_extension = 'cpp'
                else:
                    file_extension = 'java'
                snap_path = os.path.join(save_dir, f"iter_{it+1:04d}.{file_extension}")
                with open(snap_path, "w", encoding="utf-8") as f:
                    f.write(current_code)
            except Exception:
                pass
        
        # === 修正3：重新编码而非增量更新 ===
        v_current = embed_codes(model, tokenizer, [current_code], device=device)
        current_s = project_embeddings(v_current, directions)[0]
        
        # === 修正4：记录历史变体 ===
        history_variants.append({
            "iter": it + 1,
            "code": current_code,
            "s": current_s.copy(),
        })
        
        # === 记录接受的候选信息（与原代码一致）===
        trace.append({
            "iter": it + 1,
            "phase": best_type,
            "accepted_index": best_idx,
            "gain": g[best_idx].tolist(),
        })
        
        # ✅ 清理GPU缓存（每次迭代后释放中间张量）
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 达标检查（与原代码一致）
        offset_now = current_s - np.array(s0)
        all_satisfied = True
        
        for i in range(4):
            m_pos = bitwise_thresholds[str(i)]["m_pos"]
            m_neg = bitwise_thresholds[str(i)]["m_neg"]
            
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
    
    # === 修正5：从历史中选择最优变体（与原代码一致）===
    def compute_variant_quality_score(s_var, s0, bits, bitwise_thresholds):
        """评估变体质量得分"""
        offset = s_var - np.array(s0)
        k = len(bits)  # ✅ 参数化 k
        
        normalized_remainders = []
        for i in range(k):  # ✅ 参数化 k
            m_pos = bitwise_thresholds[str(i)]["m_pos"]
            m_neg = bitwise_thresholds[str(i)]["m_neg"]
            
            if bits[i] == 1:
                remainder = max(m_pos - offset[i], 0.0)
                norm_remainder = remainder / (m_pos + 1e-8)
            else:
                remainder = max(offset[i] + m_neg, 0.0)
                norm_remainder = remainder / (m_neg + 1e-8)
            
            normalized_remainders.append(norm_remainder)
        
        total_normalized_remainder = sum(normalized_remainders) + 1e-8
        
        score = 0.0
        for i in range(k):  # ✅ 参数化 k
            m_pos = bitwise_thresholds[str(i)]["m_pos"]
            m_neg = bitwise_thresholds[str(i)]["m_neg"]
            weight = normalized_remainders[i] / total_normalized_remainder
            
            if bits[i] == 1:
                if offset[i] >= m_pos:
                    excess = offset[i] - m_pos
                    norm_excess = excess / (m_pos + 1e-8)
                    score += 1.0 + norm_excess * 0.1
                else:
                    score -= normalized_remainders[i] * weight * 6.0
            else:
                if offset[i] <= -m_neg:
                    excess = (-m_neg) - offset[i]
                    norm_excess = excess / (m_neg + 1e-8)
                    score += 1.0 + norm_excess * 0.1
                else:
                    score -= normalized_remainders[i] * weight * 6.0
        
        return score
    
    best_variant_idx = 0
    best_variant_score = compute_variant_quality_score(
        history_variants[0]["s"], s0, bits, bitwise_thresholds
    )
    
    for idx in range(1, len(history_variants)):
        variant_score = compute_variant_quality_score(
            history_variants[idx]["s"], s0, bits, bitwise_thresholds
        )
        if variant_score > best_variant_score:
            best_variant_score = variant_score
            best_variant_idx = idx
    
    # 使用最优变体
    final_code = history_variants[best_variant_idx]["code"]
    final_s = history_variants[best_variant_idx]["s"]
    
    return {
        "watermarked_code": final_code,
        "s0": s0,
        "s_after": final_s.tolist(),
        "offset": (final_s - np.array(s0)).tolist(),
        "trace": trace,
        "bitwise_thresholds": bitwise_thresholds,
        "bits": ''.join(map(str, bits)),
        "best_variant_iter": best_variant_idx
    }


def embed_watermark_with_custom_directions(code, bits, directions, config, model, tokenizer, device, save_dir=None, language='java'):
    """
    使用自定义方向嵌入水印（支持多语言）
    
    Args:
        code: str, 原始代码
        bits: list of int, 水印比特
        directions: np.ndarray, shape (k, 768)
        config: dict, 嵌入配置
        model, tokenizer, device: 模型相关
        save_dir: Optional[str], 保存中间变体的目录路径
        language: 编程语言 ('java'/'javascript'/'cpp')，默认'java'
    
    Returns:
        dict, 嵌入结果
    """
    # 步骤1：计算簇中心s0和阈值（使用自定义方向）
    threshold_result = compute_thresholds_with_custom_directions(
        code, bits, directions, config, model, tokenizer, device,
        language=language  # ✅ 传递language参数
    )
    
    # 步骤2：贪心注入（使用自定义方向和s0）
    inject_result = greedy_inject_with_custom_directions(
        code, bits, directions,
        threshold_result['s0'],  # ✅ 传入簇中心
        threshold_result['bitwise_thresholds'],
        config, model, tokenizer, device,
        save_dir=save_dir,  # ✅ 传入save_dir
        language=language  # ✅ 传递language参数
    )
    
    # 合并结果，包含median_code、extreme_codes和簇中心
    result = {
        "watermarked_code": inject_result['watermarked_code'],
        "s0": inject_result['s0'],
        "s_after": inject_result['s_after'],
        "offset": inject_result['offset'],
        "bits": inject_result['bits'],
        "bitwise_thresholds": inject_result['bitwise_thresholds'],
        "median_code": threshold_result['median_code'],
        "median_cluster_center": threshold_result['median_cluster_center'],
        "extreme_codes": threshold_result['extreme_codes'],
        "extreme_cluster_centers": threshold_result['extreme_cluster_centers'],
        "best_variant_iter": inject_result.get('best_variant_iter', -1),
        "trace": inject_result.get('trace', [])
    }
    
    return result


def process_one_run_strategy1(task):
    """
    Strategy 1: 使用原始API的单个任务处理函数（子进程，支持多语言）
    
    Args:
        task: (run_id, code, strategy_name, base_config)
    
    Returns:
        dict: 处理结果
    """
    run_id, code, strategy_name, base_config = task
    output_dir = f"dimension_strategy_comparison/results/{strategy_name}/embedding/run_{run_id:04d}"
    os.makedirs(output_dir, exist_ok=True)
    
    # ===== 新增：从配置获取language =====
    dataset_name = base_config.get('dataset', {}).get('name', 'csn-java')
    if 'js' in dataset_name.lower() or 'javascript' in dataset_name.lower():
        language = 'javascript'
    else:
        language = 'java'
    
    os.environ['WATERMARK_LANGUAGE'] = language
    
    try:
        # ✅ 使用原始函数（完全一致）
        k = len(base_config['embedding']['bits'])  # ✅ 参数化 k
        required_delta = compute_required_delta_per_anchor(
            model_dir=base_config['model_dir'],
            anchor_code=code,
            bits=base_config['embedding']['bits'],
            secret_key=base_config['embedding']['secret'],
            K=base_config['embedding']['K'],
            quantile=base_config['embedding']['quantile'],
            num_workers=base_config['embedding']['num_workers'],
            batch_size_for_parallel=base_config['embedding']['batch_size_for_parallel'],
            k=k,  # ✅ 传递 k 参数
        )
        
        result = select_and_inject(
            model_dir=base_config['model_dir'],
            anchor_code=code,
            bits=base_config['embedding']['bits'],
            required_delta=required_delta,
            secret_key=base_config['embedding']['secret'],
            K=base_config['embedding']['K'],
            max_iters=base_config['embedding']['max_iters'],
            num_workers=base_config['embedding']['num_workers'],
            batch_size_for_parallel=base_config['embedding']['batch_size_for_parallel'],
            save_dir=output_dir,
        )
        
        # 计算extreme_codes和extreme_cluster_centers（用于step4b）
        directions = derive_directions(secret_key=base_config['embedding']['secret'], d=768, k=4)
        extreme_codes, extreme_cluster_centers = compute_extreme_codes_for_strategy1(
            code=code,
            directions=directions,
            K=base_config['embedding']['K'],
            num_workers=base_config['embedding']['num_workers'],
            batch_size_for_parallel=base_config['embedding']['batch_size_for_parallel'],
            model_dir=base_config['model_dir'],
            language=language  # ✅ 传递language参数
        )
        
        # 保存结果
        with open(f"{output_dir}/final.json", 'w', encoding='utf-8') as f:
            json.dump({
                'bits': ''.join(map(str, base_config['embedding']['bits'])),
                's0': required_delta['s0'],
                'bitwise_thresholds': required_delta['bitwise_thresholds'],
                's_after': result['s_after'],
                'trace': result.get('trace', []),
                'median_code': required_delta.get('median_code'),
                'median_cluster_center': required_delta.get('median_cluster_center'),
                'extreme_codes': extreme_codes,
                'extreme_cluster_centers': required_delta.get('extreme_cluster_centers'),
                'lang': language  # 保存语言信息
            }, f, indent=2, ensure_ascii=False)
        
        # 根据语言确定文件扩展名
        file_extension = 'js' if language == 'javascript' else 'java'

        with open(f"{output_dir}/watermarked.{file_extension}", 'w', encoding='utf-8') as f:
            f.write(result['final_code'])

        with open(f"{output_dir}/original.{file_extension}", 'w', encoding='utf-8') as f:
            f.write(code)
        
        # 保存方向信息（用于提取）
        k = len(base_config['embedding']['bits'])  # ✅ 参数化 k
        directions = derive_directions(secret_key=base_config['embedding']['secret'], d=768, k=k)
        with open(f"{output_dir}/selected_dimensions.json", 'w', encoding='utf-8') as f:
            json.dump({
                'run_id': run_id,
                'strategy': strategy_name,
                'method': 'random_from_secret',
                'selected_dims': 'random',
                'directions': directions.tolist()
            }, f, indent=2, ensure_ascii=False)
        
        return {"run_id": run_id, "success": True, "error": None}
    
    except Exception as e:
        with open(f"{output_dir}/error.txt", 'w', encoding='utf-8') as f:
            f.write(f"嵌入失败: {str(e)}\n")
        return {"run_id": run_id, "success": False, "error": str(e)}


def process_one_run_custom(task):
    """
    Strategy 2/3/4: 使用自定义方向的单个任务处理函数（子进程，支持多语言）
    
    Args:
        task: (run_id, code, strategy_name, base_config)
    
    Returns:
        dict: 处理结果
    """
    import torch
    import gc
    
    run_id, code, strategy_name, base_config = task
    
    # ===== 新增：从配置获取language并设置环境变量 =====
    dataset_name = base_config.get('dataset', {}).get('name', 'csn-java')
    if 'js' in dataset_name.lower() or 'javascript' in dataset_name.lower():
        language = 'javascript'
    elif 'cpp' in dataset_name.lower() or 'c++' in dataset_name.lower():
        language = 'cpp'
    else:
        language = 'java'
    
    os.environ['WATERMARK_LANGUAGE'] = language
    
    # 加载选定的维度和方向
    dim_file = f"dimension_strategy_comparison/results/{strategy_name}/embedding/run_{run_id:04d}/selected_dimensions.json"
    with open(dim_file, 'r', encoding='utf-8') as f:
        dim_data = json.load(f)
    
    directions = np.array(dim_data['directions'])
    
    # 在子进程中加载模型
    model, tokenizer = load_best_model(base_config['model_dir'])
    device = next(model.parameters()).device
    
    output_dir = f"dimension_strategy_comparison/results/{strategy_name}/embedding/run_{run_id:04d}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        result = embed_watermark_with_custom_directions(
            code=code,
            bits=base_config['embedding']['bits'],
            directions=directions,
            config=base_config['embedding'],
            model=model,
            tokenizer=tokenizer,
            device=device,
            save_dir=output_dir,
            language=language  # ✅ 传递language参数
        )
        
        # 保存结果（包含median_code、extreme_codes和簇中心用于step4b）
        with open(f"{output_dir}/final.json", 'w', encoding='utf-8') as f:
            json.dump({
                'bits': result['bits'],
                's0': result['s0'],
                'bitwise_thresholds': result['bitwise_thresholds'],
                's_after': result['s_after'],
                'trace': result.get('trace', []),
                'median_code': result.get('median_code'),
                'median_cluster_center': result.get('median_cluster_center'),
                'extreme_codes': result.get('extreme_codes'),
                'extreme_cluster_centers': result.get('extreme_cluster_centers'),
                'lang': language  # 保存语言信息
            }, f, indent=2, ensure_ascii=False)
        
        # 根据语言确定文件扩展名
        if language == 'javascript':
            file_extension = 'js'
        elif language == 'cpp':
            file_extension = 'cpp'
        else:
            file_extension = 'java'

        with open(f"{output_dir}/watermarked.{file_extension}", 'w', encoding='utf-8') as f:
            f.write(result['watermarked_code'])

        with open(f"{output_dir}/original.{file_extension}", 'w', encoding='utf-8') as f:
            f.write(code)
        
        return_value = {"run_id": run_id, "success": True, "error": None}
        
    except Exception as e:
        with open(f"{output_dir}/error.txt", 'w', encoding='utf-8') as f:
            f.write(f"嵌入失败: {str(e)}\n")
        return_value = {"run_id": run_id, "success": False, "error": str(e)}
    
    finally:
        # ✅ 显式释放模型和GPU内存
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return return_value

def process_strategy(strategy_name, base_config, concurrency, resume=False):
    """处理单个策略的所有代码（支持并发和断点继续）"""
    print(f"\n处理策略: {strategy_name}")

    # 加载测试代码
    test_codes = load_test_codes(base_config)
    
    # 准备任务列表
    tasks = []
    skipped_count = 0
    for run_id, code in enumerate(test_codes):
        # 如果启用resume，检查final.json是否存在
        if resume:
            final_json_path = f"dimension_strategy_comparison/results/{strategy_name}/embedding/run_{run_id:04d}/final.json"
            if os.path.exists(final_json_path):
                skipped_count += 1
                continue
        tasks.append((run_id, code, strategy_name, base_config))
    
    if resume and skipped_count > 0:
        print(f"  跳过已完成: {skipped_count} 个")
    
    if not tasks:
        print(f"  所有任务已完成，无需处理")
        return
    
    # Strategy 1使用原始流程
    if strategy_name == "strategy_1_random":
        process_func = process_one_run_strategy1
    else:
        process_func = process_one_run_custom
    
    # 并发或串行执行
    if concurrency > 1:
        print(f"  使用并发模式 (max_workers={concurrency})")
        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_func, task) for task in tasks]
            
            # 使用tqdm显示进度
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  嵌入进度", ncols=80):
                result = future.result()
                results.append(result)
                if not result['success']:
                    print(f"\n[错误] run_{result['run_id']:04d} 嵌入失败: {result['error']}")
    else:
        print(f"  使用串行模式")
        results = []
        for task in tqdm(tasks, desc=f"  嵌入进度", ncols=80):
            result = process_func(task)
            results.append(result)
            if not result['success']:
                print(f"\n[错误] run_{result['run_id']:04d} 嵌入失败: {result['error']}")
    
    # 统计结果
    success_count = sum(1 for r in results if r['success'])
    print(f"  完成: {success_count}/{len(results)} 成功")


def main():
    parser = argparse.ArgumentParser(description="Step 3: 嵌入水印（支持并发）")
    parser.add_argument('--concurrency', type=int, default=5, help='并发处理的进程数（默认=5）')
    parser.add_argument('--resume', action='store_true', help='断点继续：跳过已有final.json的任务')
    parser.add_argument('--strategy', type=str, default=None, help='指定处理的策略（如果不指定，处理所有策略）')
    parser.add_argument('--config', type=str, 
                       default=None,
                       help='配置文件路径（相对于项目根目录）')
    args = parser.parse_args()

    print("="*80)
    print(f"Step 3: 嵌入水印 (concurrency={args.concurrency}, resume={args.resume})")
    if args.strategy:
        print(f"  仅处理策略: {args.strategy}")
    print("="*80)

    # 设置工作目录为项目根目录（与原始流程一致）
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    # 加载基础配置（保持原逻辑不变）
    if args.config:
        config_path = project_root / args.config
    else:
        config_path = project_root / "dimension_strategy_comparison" / "configs" / "base_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        base_config = json.load(f)

    # ===== 新增：语言和bit版本检测，设置环境变量 =====
    dataset_name = base_config.get('dataset', {}).get('name', 'csn-java')
    if 'js' in dataset_name.lower() or 'javascript' in dataset_name.lower():
        language = 'javascript'
        lang_suffix = '_js'
    elif 'cpp' in dataset_name.lower() or 'c++' in dataset_name.lower():
        language = 'cpp'
        lang_suffix = '_cpp'
    else:
        language = 'java'
        lang_suffix = ''
    
    # ✅ 计算 bit_suffix
    k = len(base_config['embedding']['bits'])
    bit_suffix = '_2bit' if k == 2 else ('_6bit' if k == 6 else ('_8bit' if k == 8 else ''))
    
    # 设置全局环境变量（供plan.py和子进程使用）
    os.environ['WATERMARK_LANGUAGE'] = language
    
    print(f"检测到语言: {language}")
    print(f"数据集: {dataset_name}")
    print(f"投影维度: {k}bit")

    # ✅ 确定要处理的策略（根据bit和语言动态调整后缀）
    if args.strategy:
        strategies = [args.strategy]
    else:
        strategies = [
            f"strategy_5_learned{bit_suffix}{lang_suffix}",
            f"strategy_6_adaptive{bit_suffix}{lang_suffix}",
        ]

    for strategy_name in strategies:
        process_strategy(strategy_name, base_config, args.concurrency, args.resume)

    print("\n" + "="*80)
    print("完成！水印嵌入结果已保存到 results/strategy_X/embedding/")
    print("="*80)


if __name__ == '__main__':
    os.chdir(Path(__file__).parent.parent)
    main()
