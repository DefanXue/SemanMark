"""
Watermark injection planning helpers.

Only uses components inside Watermark4code and the upstream generator entrypoint
to strictly match the test-split generation process.
"""

import os
import json
import tempfile
from typing import Dict, List, Tuple
import numpy as np

# 设置临时目录�?D �?tempfile.tempdir = r'D:\temp'
os.makedirs(tempfile.tempdir, exist_ok=True)

from ..api import load_encoder, compute_baseline_s0
from ..encoder import embed_codes
from ..keys import derive_directions
from ..utils import project_embeddings
import sys


def _ensure_srcmarker_on_path() -> None:
    """确保优先解析 WATERMARK_SECRET/SrcMarker-main 下的 contrastive_learning�?""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    WATERMARK_SECRET_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    srcmarker_root = os.path.join(WATERMARK_SECRET_root, "SrcMarker-main")
    if srcmarker_root in sys.path:
        sys.path.remove(srcmarker_root)
    sys.path.insert(0, srcmarker_root)


_ensure_srcmarker_on_path()

from contrastive_learning.java_augmentor import generate_java_training_data_parallel  # type: ignore
# �?C++ �?JavaScript 的导入将在运行时动态导入以避免循环依赖
# 参见 build_candidates_test_like() �?build_candidates_by_type() 中的条件导入


def _get_quantile_entry(obj: Dict, q: float):
    """
    Robustly fetch a quantile entry from an object that stores quantiles under string keys.
    Tries keys like "0.90", "0.9", str(q); if not found, tries tolerant numeric match.
    Returns the entry (could be float or dict) or None if not found.
    """
    if "quantiles" not in obj or not isinstance(obj["quantiles"], dict):
        return None
    qmap = obj["quantiles"]
    candidates = [f"{q:.2f}", f"{q:.1f}", str(q)]
    for k in candidates:
        if k in qmap:
            return qmap[k]
    # tolerant numeric match
    for k, v in qmap.items():
        try:
            if abs(float(k) - q) < 1e-8 or abs(round(float(k), 2) - round(q, 2)) < 1e-8:
                return v
        except Exception:
            continue
    return None


def compute_required_delta(epsilon_json_path: str, tmargin_json_path: str, quantile: float = 0.90) -> float:
    with open(epsilon_json_path, "r", encoding="utf-8") as f:
        eps_obj = json.load(f)
    with open(tmargin_json_path, "r", encoding="utf-8") as f:
        tm_obj = json.load(f)

    # epsilon: quantiles hold floats
    eps_entry = _get_quantile_entry(eps_obj, quantile)
    eps = float(eps_entry) if eps_entry is not None else float(eps_obj.get("epsilon_emp", 0.0))

    # t_margin: quantiles hold dict with {per_bit, scalar}
    tm_entry = _get_quantile_entry(tm_obj, quantile)
    if isinstance(tm_entry, dict) and "scalar" in tm_entry:
        tm_scalar = float(tm_entry["scalar"])
    else:
        tm_scalar = float(tm_obj.get("scalar", 0.0))

    return eps + tm_scalar


def compute_baseline(model_dir: str, anchor_code: str, secret_key: str = "WATERMARK_SECRET") -> Dict:
    base = compute_baseline_s0(model_dir, [anchor_code], secret_key=secret_key)
    return {
        "s0": base["s0"][0],
        "W": base["directions"],
    }


def _write_anchor_copies_tmp(anchor_code: str, copies: int) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="wm_inject_")
    tmp_path = os.path.join(tmp_dir, "anchors.jsonl")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for _ in range(max(1, copies)):
            f.write(json.dumps({"code": anchor_code}, ensure_ascii=False) + "\n")
    return tmp_path


def build_candidates_test_like(
    anchor_code: str,
    K: int,
    num_workers: int = 48,
    batch_size_for_parallel: int = 20,
    language: str = None,  # 新增：语言参数（可选，向后兼容�?) -> Tuple[List[str], Dict]:
    """
    Match test-split generation: split_type='test', positive_ratio=1.0, proportions 0.2/0.5/0.3.
    To get K variants, duplicate the same anchor K times as input.
    
    Args:
        language: 编程语言 ('java'/'javascript'/None)，None时从环境变量读取或默认为'java'
    
    Returns:
        (cands, stats): 
            cands - 通过审核的变体列�?            stats - {"passed_count": int, "failed_reasons": {"原因1": 计数, ...}}
    """
    # 确定语言（优先级：参�?> 环境变量 > 默认java�?    if language is None:
        language = os.environ.get('WATERMARK_LANGUAGE', 'java')
    
    # 设置环境变量（供augmentor使用�?    os.environ['WATERMARK_LANGUAGE'] = language
    
    aug_types = {
        "semantic_preserving": 1,  # 静态规�?50%
        "llm_rewrite": 0,           # LLM重写 50%
        "retranslate": 0.0,
    }

    in_file = _write_anchor_copies_tmp(anchor_code, K)
    out_dir = tempfile.mkdtemp(prefix="wm_aug_")
    out_file = os.path.join(out_dir, "augmented.jsonl")
    review_stats_file = os.path.join(out_dir, "review_stats.jsonl")

    # 通过环境变量传递统计文件路�?    os.environ['REVIEW_STATS_FILE'] = review_stats_file
    
    try:
        # �?根据语言选择正确的生成函�?        if language == 'cpp':
            from contrastive_learning.cpp_augmentor import generate_cpp_training_data_parallel  # type: ignore
            generate_cpp_training_data_parallel(
                input_file=in_file,
                output_file=out_file,
                model={"name": os.environ.get("NEWAPI_MODEL", "gpt-5-mini")},
                split_type="test",
                positive_ratio=1.0,
                augmentation_types=aug_types,
                max_samples=K,
                num_workers=num_workers,
                batch_size=batch_size_for_parallel,
                resume=False,
            )
        elif language == 'javascript':
            from contrastive_learning.js_augmentor import generate_js_training_data_parallel  # type: ignore
            generate_js_training_data_parallel(
                input_file=in_file,
                output_file=out_file,
                model={"name": os.environ.get("NEWAPI_MODEL", "gpt-5-mini")},
                split_type="test",
                positive_ratio=1.0,
                augmentation_types=aug_types,
                max_samples=K,
                num_workers=num_workers,
                batch_size=batch_size_for_parallel,
                resume=False,
            )
        else:  # java (默认)
            generate_java_training_data_parallel(
                input_file=in_file,
                output_file=out_file,
                model={"name": os.environ.get("NEWAPI_MODEL", "gpt-5-mini")},
                split_type="test",
                positive_ratio=1.0,
                augmentation_types=aug_types,
                max_samples=K,
                num_workers=num_workers,
                batch_size=batch_size_for_parallel,
                resume=False,
            )
    finally:
        if 'REVIEW_STATS_FILE' in os.environ:
            del os.environ['REVIEW_STATS_FILE']

    cands: List[str] = []
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "anchor" in obj and "positive" in obj and isinstance(obj["positive"], str):
                cand = obj["positive"].strip()
                if cand and cand != anchor_code.strip():
                    cands.append(cand)

    # 读取审核统计
    passed_count = 0
    failed_reasons = {}
    if os.path.exists(review_stats_file):
        with open(review_stats_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if record.get("passed"):
                        passed_count += 1
                    else:
                        reason = record.get("reason", "未知原因")
                        failed_reasons[reason] = failed_reasons.get(reason, 0) + 1
                except Exception:
                    continue
    
    stats = {
        "passed_count": passed_count,
        "failed_reasons": failed_reasons
    }

    # 去重并截�?K
    uniq: List[str] = []
    seen = set()
    for c in cands:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
        if len(uniq) >= K:
            break
    return uniq, stats


def build_candidates_by_type(
    anchor_code: str,
    K: int,
    aug_type: str,
    num_workers: int = 48,
    batch_size_for_parallel: int = 20,
    language: str = None,  # 新增：语言参数（可选，向后兼容�?) -> List[str]:
    """
    仅生成指定类别的等价候选，分布对齐 test-split：positive_ratio=1.0�?    aug_type �?{"semantic_preserving", "llm_rewrite", "retranslate"}
    
    Args:
        language: 编程语言 ('java'/'javascript'/None)，None时从环境变量读取或默认为'java'
    """
    # 确定语言（优先级：参�?> 环境变量 > 默认java�?    if language is None:
        language = os.environ.get('WATERMARK_LANGUAGE', 'java')
    
    # 设置环境变量（供augmentor使用�?    os.environ['WATERMARK_LANGUAGE'] = language
    assert aug_type in {"semantic_preserving", "llm_rewrite", "retranslate"}

    aug_types = {aug_type: 1.0}

    in_file = _write_anchor_copies_tmp(anchor_code, K)
    out_dir = tempfile.mkdtemp(prefix="wm_aug_type_")
    out_file = os.path.join(out_dir, "augmented.jsonl")

    # �?根据语言选择正确的生成函�?    if language == 'cpp':
        from contrastive_learning.cpp_augmentor import generate_cpp_training_data_parallel  # type: ignore
        generate_cpp_training_data_parallel(
            input_file=in_file,
            output_file=out_file,
            model={"name": os.environ.get("NEWAPI_MODEL", "gpt-5-mini")},
            split_type="test",
            positive_ratio=1.0,
            augmentation_types=aug_types,
            max_samples=K,
            num_workers=num_workers,
            batch_size=batch_size_for_parallel,
            resume=False,
        )
    elif language == 'javascript':
        from contrastive_learning.js_augmentor import generate_js_training_data_parallel  # type: ignore
        generate_js_training_data_parallel(
            input_file=in_file,
            output_file=out_file,
            model={"name": os.environ.get("NEWAPI_MODEL", "gpt-5-mini")},
            split_type="test",
            positive_ratio=1.0,
            augmentation_types=aug_types,
            max_samples=K,
            num_workers=num_workers,
            batch_size=batch_size_for_parallel,
            resume=False,
        )
    else:  # java (默认)
        generate_java_training_data_parallel(
            input_file=in_file,
            output_file=out_file,
            model={"name": os.environ.get("NEWAPI_MODEL", "gpt-5-mini")},
            split_type="test",
            positive_ratio=1.0,
            augmentation_types=aug_types,
            max_samples=K,
            num_workers=num_workers,
            batch_size=batch_size_for_parallel,
            resume=False,
        )

    cands: List[str] = []
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "anchor" in obj and "positive" in obj and isinstance(obj["positive"], str):
                cand = obj["positive"].strip()
                if cand and cand != anchor_code.strip():
                    cands.append(cand)

    # 去重并截�?K
    uniq: List[str] = []
    seen = set()
    for c in cands:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
        if len(uniq) >= K:
            break
    return uniq


def compute_required_delta_per_anchor(
    model_dir: str,
    anchor_code: str,
    bits: List[int],
    secret_key: str = "WATERMARK_SECRET",
    K: int = 50,
    quantile: float = 0.90,
    quantized: bool = True,
    max_length: int = 512,
    batch_size: int = 64,
    num_workers: int = 48,
    batch_size_for_parallel: int = 20,
) -> Dict:
    """
    针对单个 anchor，按测试分布生成 K_thr 等价候选，计算所�?6�?bit模式的分组阈值�?    """
    # 1) 生成候选（门槛由底层生成器负责），阈值估计固定用 50 个样�?    K_thr = 100
    try:
        cands, review_stats = build_candidates_test_like(
            anchor_code,
            max(1, K_thr),
            num_workers=num_workers,
            batch_size_for_parallel=batch_size_for_parallel,
        )
    except Exception:
        cands = []
        review_stats = {"passed_count": 0, "failed_reasons": {}}

    # 过滤无变�?    cands = [c for c in cands if isinstance(c, str) and c.strip() and c.strip() != anchor_code.strip()]
    if not cands:
        # 无候选时，返回所�?6种bits的零阈�?        all_bits_patterns = [
            f"{b3}{b2}{b1}{b0}"
            for b3 in [0, 1]
            for b2 in [0, 1]
            for b1 in [0, 1]
            for b0 in [0, 1]
        ]
        all_bits_thresholds = {}
        for bits_pattern in all_bits_patterns:
            bits_list = [int(b) for b in bits_pattern]
            pos_indices = [i for i in range(4) if bits_list[i] == 1]
            neg_indices = [i for i in range(4) if bits_list[i] == 0]
            all_bits_thresholds[bits_pattern] = {
                "q_pos_group": 0.0,
                "m_pos_group": 0.0,
                "T_pos_group": 0.0,
                "q_neg_group": 0.0,
                "m_neg_group": 0.0,
                "T_neg_group": 0.0,
                "pos_indices": pos_indices,
                "neg_indices": neg_indices,
            }
        return {"k": 4, "review_stats": review_stats, "all_bits_thresholds": all_bits_thresholds}

    # 2) 编码并投�?    model, tokenizer, device = load_encoder(model_dir, use_quantization=quantized)
    v_anchor = embed_codes(model, tokenizer, [anchor_code], max_length=max_length, batch_size=batch_size, device=device)
    v_cands = embed_codes(model, tokenizer, cands, max_length=max_length, batch_size=batch_size, device=device)

    d = v_anchor.shape[1]
    W = derive_directions(secret_key=secret_key, d=int(d), k=4)
    s_anchor = project_embeddings(v_anchor, W)[0]  # [4] 保留作为参�?    s_cands = project_embeddings(v_cands, W)       # [K,4]

    # 3) 计算两种簇中�?    cluster_centers_median = np.zeros(4)
    cluster_centers_balanced = np.zeros(4)
    for i in range(4):
        # 方法1：中位数
        cluster_centers_median[i] = float(np.median(s_cands[:, i]))
        
        # 方法2：平衡中心（正负半径相等�?        max_val = float(np.max(s_cands[:, i]))
        min_val = float(np.min(s_cands[:, i]))
        cluster_centers_balanced[i] = (max_val + min_val) / 2
    
    # 默认使用中位数方�?    cluster_centers = cluster_centers_median
    
    # 3.5) 找到最接近两种簇中心的变体代码
    distances_median = np.linalg.norm(s_cands - cluster_centers_median, axis=1)
    median_idx_median = int(np.argmin(distances_median))
    median_code_median = cands[median_idx_median]
    
    distances_balanced = np.linalg.norm(s_cands - cluster_centers_balanced, axis=1)
    median_idx_balanced = int(np.argmin(distances_balanced))
    median_code_balanced = cands[median_idx_balanced]
    
    # 保持向后兼容
    median_idx = median_idx_median
    median_code = median_code_median

    # 4) 计算簇半径（分别为两种中心计算）
    cluster_info_median = {}
    cluster_info_balanced = {}
    
    for i in range(4):
        # 方法1：中位数中心
        offsets_median = s_cands[:, i] - cluster_centers_median[i]
        pos_offsets_median = [o for o in offsets_median if o > 0]
        neg_offsets_median = [o for o in offsets_median if o < 0]
        
        radius_pos_median = float(max(pos_offsets_median)) if pos_offsets_median else 0.0
        radius_neg_median = float(abs(min(neg_offsets_median))) if neg_offsets_median else 0.0
        
        cluster_info_median[i] = {
            'center': float(cluster_centers_median[i]),
            'radius_pos': radius_pos_median,
            'radius_neg': radius_neg_median,
        }
        
        # 方法2：平衡中�?        offsets_balanced = s_cands[:, i] - cluster_centers_balanced[i]
        pos_offsets_balanced = [o for o in offsets_balanced if o > 0]
        neg_offsets_balanced = [o for o in offsets_balanced if o < 0]
        
        radius_pos_balanced = float(max(pos_offsets_balanced)) if pos_offsets_balanced else 0.0
        radius_neg_balanced = float(abs(min(neg_offsets_balanced))) if neg_offsets_balanced else 0.0
        
        cluster_info_balanced[i] = {
            'center': float(cluster_centers_balanced[i]),
            'radius_pos': radius_pos_balanced,
            'radius_neg': radius_neg_balanced,
        }
    
    # 保持向后兼容
    cluster_info = cluster_info_median

    # 5) 对抗阈值计算（分别为两种中心计算）
    bitwise_thresholds_median = {}
    bitwise_thresholds_balanced = {}
    
    for i in range(4):
        # 方法1：中位数中心的阈�?        T_pos_offset_median = cluster_info_median[i]['radius_pos'] * quantile
        T_pos_median = cluster_centers_median[i] + T_pos_offset_median
        T_neg_offset_median = cluster_info_median[i]['radius_neg'] * quantile
        T_neg_median = cluster_centers_median[i] - T_neg_offset_median
        
        bitwise_thresholds_median[i] = {
            "m_pos": T_pos_offset_median,
            "m_neg": T_neg_offset_median,
            "T_pos": T_pos_median,
            "T_neg": T_neg_median,
        }
        
        # 方法2：平衡中心的阈�?        T_pos_offset_balanced = cluster_info_balanced[i]['radius_pos'] * quantile
        T_pos_balanced = cluster_centers_balanced[i] + T_pos_offset_balanced
        T_neg_offset_balanced = cluster_info_balanced[i]['radius_neg'] * quantile
        T_neg_balanced = cluster_centers_balanced[i] - T_neg_offset_balanced
        
        bitwise_thresholds_balanced[i] = {
            "m_pos": T_pos_offset_balanced,
            "m_neg": T_neg_offset_balanced,
            "T_pos": T_pos_balanced,
            "T_neg": T_neg_balanced,
        }
    
    # 保持向后兼容
    bitwise_thresholds = bitwise_thresholds_median
    
    return {
        "k": 4,
        # 方法1：中位数中心（默认）
        "s0": cluster_centers_median.tolist(),
        "median_code": median_code_median,
        "cluster_info": {str(i): cluster_info_median[i] for i in range(4)},
        "bitwise_thresholds": bitwise_thresholds_median,
        # 方法2：平衡中�?        "s0_balanced": cluster_centers_balanced.tolist(),
        "median_code_balanced": median_code_balanced,
        "cluster_info_balanced": {str(i): cluster_info_balanced[i] for i in range(4)},
        "bitwise_thresholds_balanced": bitwise_thresholds_balanced,
        "review_stats": review_stats,
    }

