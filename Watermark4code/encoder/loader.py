"""
编码器加载与嵌入接口（第一阶段骨架）：
- 严格复用 contrastive_learning/scripts/evaluate.py 的加载与推理逻辑�?  - 检�?PEFT 适配器目录（adapter_config.json�?  - 使用基础模型 tokenizer
  - 构建 RobustEncoder(eval_mode=True) 并注�?PEFT 适配�?  - 仅使�?encoder-only 的表示（forward_encoder_only�?- 默认开启量化（与评测默认一致），max_length=512
- 批量嵌入接口 embed_codes

注意：本模块仅在 D:\kyl410\XDF\Watermark4code 内实现，不修改其他目录�?"""

import os
import sys
from typing import List, Tuple, Optional

import torch
import numpy as np
from transformers import AutoTokenizer


def _ensure_sys_path_for_contrastive_learning() -> None:
    """
    �?SrcMarker-main 加入 sys.path，确保可以导�?contrastive_learning 包�?    路径推导：Watermark4code 位于 XDF/Watermark4code，兄弟目录为 SrcMarker-main�?    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xdf_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    srcmarker_root = os.path.join(xdf_root, "SrcMarker-main")
    # 确保优先命中 XDF/SrcMarker-main 下的 contrastive_learning
    if srcmarker_root in sys.path:
        sys.path.remove(srcmarker_root)
    sys.path.insert(0, srcmarker_root)

_ensure_sys_path_for_contrastive_learning()

from contrastive_learning.model import RobustEncoder  # noqa: E402
from peft import PeftConfig, PeftModel  # noqa: E402

# 与评测脚本一致：仅在未预设时�?HF 缓存与离线标志提供默认�?_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_XDF_ROOT = os.path.abspath(os.path.join(_CUR_DIR, os.pardir, os.pardir))
_HF_CACHE_DEFAULT = os.path.join(_XDF_ROOT, "SrcMarker-main", "hf-cache")
os.environ.setdefault("HF_HOME", _HF_CACHE_DEFAULT)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_best_model(
    model_dir: str,
    use_quantization: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[RobustEncoder, AutoTokenizer]:
    """
    加载鲁棒编码器（�?contrastive_learning/scripts/evaluate.py 的加载流程对齐）�?    - 若检测到 PEFT 适配器（adapter_config.json）：
      * 从适配器配置读�?base_model_name
      * 使用�?base tokenizer（不指定 cache_dir/local_files_only�?      * 构建 RobustEncoder(eval_mode=True, use_quantization)
      * 注入 PeftModel.from_pretrained(model.encoder, model_dir)
    - 否则尝试加载完整模型（从 model_dir/pytorch_model.bin 回退），
      * tokenizer 从配置或默认模型名加载（与评测行为一致，不强�?cache_dir�?
    返回�?model, tokenizer)
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    adapter_cfg_path = os.path.join(model_dir, "adapter_config.json")
    is_peft = os.path.exists(adapter_cfg_path)

    if is_peft:
        # 与评测脚本一致：使用传入�?model_dir 作为适配器目�?        peft_config = PeftConfig.from_pretrained(model_dir)
        base_model_name = peft_config.base_model_name_or_path
        
        # 构建本地快照路径，避免尝试从远程加载
        # base_model_name 格式: "Salesforce/codet5-base"
        model_name_safe = base_model_name.replace("/", "--")  # "Salesforce--codet5-base"
        local_model_path = os.path.join(
            _HF_CACHE_DEFAULT,
            "hub",
            f"models--{model_name_safe}",
            "snapshots"
        )
        
        # 查找最新的快照目录
        if os.path.exists(local_model_path):
            snapshots = [d for d in os.listdir(local_model_path) if os.path.isdir(os.path.join(local_model_path, d))]
            if snapshots:
                # 使用第一个快照（通常只有一个）
                snapshot_path = os.path.join(local_model_path, snapshots[0])
                tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
            else:
                # 回退到远程名称（可能触发下载�?                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        else:
            # 回退到远程名�?            tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        model = RobustEncoder(
            model_name=base_model_name,
            projection_dim=128,
            projection_hidden_dim=512,
            pooling_strategy="mean",
            use_quantization=use_quantization,
            eval_mode=True,
        )
        model.encoder = PeftModel.from_pretrained(model.encoder, model_dir)
    else:
        # �?PEFT：与评测脚本相同策略，先按默认基础模型名加�?tokenizer，再从本地权重回退
        base_model_name = "Salesforce/codet5-base"
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = RobustEncoder(
            model_name=base_model_name,
            projection_dim=128,
            projection_hidden_dim=512,
            pooling_strategy="mean",
            use_quantization=use_quantization,
            eval_mode=True,
        )
        pt_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location=device)
            filtered = {k: v for k, v in state_dict.items() if not k.startswith("projection_head.")}
            model.load_state_dict(filtered, strict=False)

    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def embed_codes(
    model: RobustEncoder,
    tokenizer: AutoTokenizer,
    code_list: List[str],
    max_length: int = 512,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    将一组代码字符串编码�?L2 归一化的 768 维向量（encoder-only 表示）�?
    参数�?    - model / tokenizer：由 load_best_model 返回，模型已处于 eval 模式
    - code_list：代码字符串列表
    - max_length：分词器最�?token 长度（与评测一致为 512�?    - batch_size：批大小
    - device：可选设备，默认�?model 保持一�?
    返回�?    - numpy.ndarray，形�?[N, 768]
    """
    if not code_list:
        return np.zeros((0, 768), dtype=np.float32)

    device = device or next(model.parameters()).device
    all_embeddings: List[np.ndarray] = []

    total = len(code_list)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_codes = code_list[start:end]

        encodings = tokenizer(
            batch_codes,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        outputs = model.forward_encoder_only(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.detach().cpu().numpy()  # [B, 768]
        all_embeddings.append(embeddings.astype(np.float32))

    return np.concatenate(all_embeddings, axis=0)


