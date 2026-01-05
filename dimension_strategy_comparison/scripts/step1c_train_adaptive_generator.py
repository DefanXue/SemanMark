"""
Step 1c: 训练自适应方向生成器 (Strategy 6)

输入：data/dimension_analysis/run_XXXX_analysis.json（已有数据）
输出：results/strategy_6_adaptive/trained_generator.pth

自适应方向生成器根据代码嵌入动态生成最优的投影方向
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import gc
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目路径
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "dimension_strategy_comparison"))

# 添加contrastive_learning路径（用于Augmentor）
srcmarker_path = project_root / "SrcMarker-main"
sys.path.insert(0, str(srcmarker_path))
contrastive_path = srcmarker_path / "contrastive_learning"
sys.path.insert(0, str(contrastive_path))

from Watermark4code.encoder.loader import load_best_model, embed_codes
from Watermark4code.injection.plan import build_candidates_test_like

# ====== NEW: 为支持原始CodeT5的embedding ======
def embed_codes_universal(model, tokenizer, code_list, batch_size=32, device='cuda'):
    """
    通用的代码嵌入函数，支持 RobustEncoder 和原始 CodeT5
    """
    import torch
    import numpy as np
    from transformers import T5EncoderModel
    
    if not code_list:
        return np.zeros((0, 768), dtype=np.float32)
    
    if device == 'cpu':
        device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    all_embeddings = []
    total = len(code_list)
    
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_codes = code_list[start:end]
            
            encodings = tokenizer(
                batch_codes,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            
            # 检查是否是原始CodeT5还是RobustEncoder
            if isinstance(model, T5EncoderModel):
                # 原始CodeT5: 输出encoder的最后隐藏层
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token（T5用<s>）
            else:
                # RobustEncoder: 使用forward_encoder_only方法
                embeddings = model.forward_encoder_only(input_ids=input_ids, attention_mask=attention_mask)
            
            # L2归一化
            embeddings = embeddings / (torch.norm(embeddings, p=2, dim=1, keepdim=True) + 1e-8)
            embeddings = embeddings.detach().cpu().numpy()
            all_embeddings.append(embeddings.astype(np.float32))
    
    return np.concatenate(all_embeddings, axis=0)
from dimension_strategy_comparison.models.adaptive_direction_generator import AdaptiveDirectionGenerator

# 动态导入augmentor（根据语言自动选择）
def get_augmentor_for_language(language):
    """根据语言返回对应的代码增强器（并发安全版本）"""
    import sys
    
    # 确保contrastive_learning路径在sys.path中（对子进程也有效）
    contrastive_path = project_root / "SrcMarker-main" / "contrastive_learning"
    if str(contrastive_path) not in sys.path:
        sys.path.insert(0, str(contrastive_path))
    
    if language == 'java':
        # 使用绝对导入确保在子进程中也能找到
        from contrastive_learning.java_augmentor import JavaCodeAugmentor
        return JavaCodeAugmentor()
    elif language == 'javascript':
        # 使用绝对导入确保在子进程中也能找到
        from contrastive_learning.js_augmentor import JavaScriptCodeAugmentor
        return JavaScriptCodeAugmentor()
    elif language == 'cpp':
        # C++: 使用绝对导入确保在子进程中也能找到
        from contrastive_learning.cpp_augmentor import CppCodeAugmentor
        return CppCodeAugmentor()
    else:
        raise ValueError(f"Unsupported language: {language}")

# 导入MutableAST转换器（与step4完全一致）
from mutable_tree.transformers import (
    IfBlockSwapTransformer,
    CompoundIfTransformer,
    ConditionTransformer,
    LoopTransformer,
    InfiniteLoopTransformer,
    UpdateTransformer,
    SameTypeDeclarationTransformer,
    VarDeclLocationTransformer,
    VarInitTransformer,
    VarNameStyleTransformer,
)
from code_transform_provider import CodeTransformProvider
import tree_sitter

# 创建所有转换器实例（与step4完全一致）
transformers = [
    IfBlockSwapTransformer(),
    CompoundIfTransformer(),
    ConditionTransformer(),
    LoopTransformer(),
    InfiniteLoopTransformer(),
    UpdateTransformer(),
    SameTypeDeclarationTransformer(),
    VarDeclLocationTransformer(),
    VarInitTransformer(),
    VarNameStyleTransformer(),
]


def get_feasible_mutable_transforms(code, augmentor, language='java'):
    """检测对某个代码可行的MutableAST变换（支持Java和JavaScript）"""
    try:
        # 提取方法代码（去除类包装）
        method_code, metadata = augmentor._extract_method_from_full_class(code)

        # 初始化tree-sitter解析器
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent.parent
        lang_so = project_root / "SrcMarker-main" / "parser" / "languages.so"

        if not lang_so.exists():
            return {}

        parser_lang = tree_sitter.Language(str(lang_so), language)
        ts_parser = tree_sitter.Parser()
        ts_parser.set_language(parser_lang)

        # 初始化CodeTransformProvider
        transform_provider = CodeTransformProvider(language, ts_parser, transformers)

        # 枚举所有可行的转换键（与step4完全一致）
        feasible_transforms = []
        for transformer in transformers:
            transformer_name = transformer.__class__.__name__
            try:
                available_keys = transformer.get_available_transforms()
                for key in available_keys:
                    # 尝试应用单个转换验证可行性
                    new_code = transform_provider.code_transform(method_code, [key])
                    if new_code != method_code and len(new_code) > 0:
                        feasible_transforms.append((transformer_name, key))
            except:
                continue

        # 转换为与原函数兼容的格式
        feasible_map = {}
        for transformer_name, key in feasible_transforms:
            if transformer_name not in feasible_map:
                feasible_map[transformer_name] = []
            feasible_map[transformer_name].append(key)

        return feasible_map

    except Exception as e:
        return {}


def apply_single_mutable_transform(code, transformer_name, key, augmentor, language='java'):
    """应用单个MutableAST变换（支持Java和JavaScript）"""
    try:
        method_code, metadata = augmentor._extract_method_from_full_class(code)

        # 初始化tree-sitter解析器
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent.parent
        lang_so = project_root / "SrcMarker-main" / "parser" / "languages.so"

        if not lang_so.exists():
            return code

        parser_lang = tree_sitter.Language(str(lang_so), language)
        ts_parser = tree_sitter.Parser()
        ts_parser.set_language(parser_lang)

        # 初始化CodeTransformProvider
        transform_provider = CodeTransformProvider(language, ts_parser, transformers)

        # 应用转换
        transformed_method = transform_provider.code_transform(method_code, [key])

        # 包装回完整类
        if metadata is not None:
            return augmentor._wrap_method_back_to_class(transformed_method, metadata)
        else:
            return transformed_method

    except Exception:
        return code


def generate_attacks_via_mutable_ast(variant_code, augmentor, language='java'):
    """
    使用MutableAST变换、重命名和等价变体生成作为"攻击"（支持Java和JavaScript和C++）

    逻辑：
      1. 检测所有可行的MutableAST变换组合（所有transformer的所有key）
      2. 对每个可行的(transformer, key)组合生成一个攻击样本
      3. 加上8个full_variable_rename
      4. 生成15个等价变体作为攻击
      5. 返回实际生成的攻击（不填充）
    """
    import random
    
    # [验证] 确保 language 参数被正确传递
    augmentor_class = augmentor.__class__.__name__
    print(f"[DEBUG] generate_attacks_via_mutable_ast 入口: language={language}, augmentor={augmentor_class}")
    
    attacked_codes = []

    # 检测可行的MutableAST变换
    print(f"[DEBUG] 调用 get_feasible_mutable_transforms with language={language}")
    feasible_map = get_feasible_mutable_transforms(variant_code, augmentor, language)

    # 收集所有可行的(transformer_name, key)组合
    all_feasible_combinations = []
    for transformer_name, feasible_keys in feasible_map.items():
        for key in feasible_keys:
            all_feasible_combinations.append((transformer_name, key))

    # 对每个可行的组合生成攻击样本（完全覆盖所有可能的单个变换）
    for transformer_name, key in all_feasible_combinations:
        attacked_code = apply_single_mutable_transform(
            variant_code,
            transformer_name,
            key,
            augmentor,
            language
        )
        if attacked_code != variant_code:
            attacked_codes.append(attacked_code)

    # 应用8个full_variable_rename（根据语言选择重命名器，与Java完全一致）
    try:
        if language == 'java':
            from Watermark4code.experiments.Attack.Rename_Attack.java_variable_renamer import JavaVariableRenamer
            from Watermark4code.experiments.Attack.Rename_Attack.attack_config import AttackConfig

            for seed in range(8):
                try:
                    renamer = JavaVariableRenamer(variant_code)
                    config = AttackConfig(naming_strategy="random", seed=seed)
                    renamed_code = renamer.apply_renames(config, rename_ratio=1.0)
                    if renamed_code != variant_code:
                        attacked_codes.append(renamed_code)
                except:
                    pass
        elif language == 'javascript':
            # JS使用JavaScriptVariableRenamer，与Java保持完全一致
            # 明确传入seed=seed和rename_ratio=1.0，确保8次攻击使用不同的seed（0-7）
            from Watermark4code.experiments.Attack.Rename_Attack.js_variable_renamer import JavaScriptVariableRenamer
            from Watermark4code.experiments.Attack.Rename_Attack.attack_config import AttackConfig

            for seed in range(8):
                try:
                    renamer = JavaScriptVariableRenamer(variant_code)
                    config = AttackConfig(naming_strategy="random", seed=seed)
                    renamed_code = renamer.apply_renames(config, rename_ratio=1.0)
                    if renamed_code != variant_code:
                        attacked_codes.append(renamed_code)
                except:
                    pass
        elif language == 'cpp':
            # C++: 使用CppVariableRenamer，与Java/JS保持完全一致
            try:
                from Watermark4code.experiments.Attack.Rename_Attack.cpp_variable_renamer import CppVariableRenamer
                from Watermark4code.experiments.Attack.Rename_Attack.attack_config import AttackConfig

                for seed in range(8):
                    try:
                        renamer = CppVariableRenamer(variant_code)
                        config = AttackConfig(naming_strategy="random", seed=seed)
                        renamed_code = renamer.apply_renames(config, rename_ratio=1.0)
                        if renamed_code != variant_code:
                            attacked_codes.append(renamed_code)
                    except:
                        pass
            except ImportError:
                # 如果C++重命名器不可用，跳过该步骤
                pass
    except:
        pass

    # 生成15个等价变体作为攻击
    try:
        if language == 'javascript':
            # JS: 直接使用augmentor生成（避免硬编码的Java函数）
            for _ in range(10):
                try:
                    variant = augmentor._apply_mutable_ast_transforms(variant_code)
                    if variant.strip() and variant != variant_code:
                        attacked_codes.append(variant)
                except:
                    continue
        elif language == 'cpp':
            # C++: 使用augmentor生成（对标JavaScript）
            for _ in range(10):
                try:
                    variant = augmentor._apply_mutable_ast_transforms(variant_code)
                    if variant.strip() and variant != variant_code:
                        attacked_codes.append(variant)
                except:
                    # 备选方案：使用通用augment方法
                    try:
                        augmented = augmentor.augment(variant_code)
                        if augmented and isinstance(augmented, list) and len(augmented) > 0:
                            variant = augmented[0]
                            if variant.strip() and variant != variant_code:
                                attacked_codes.append(variant)
                    except:
                        continue
        else:
            # Java: 使用build_candidates_test_like
            from Watermark4code.injection.plan import build_candidates_test_like
            variant_attacks, _ = build_candidates_test_like(
                variant_code,
                K=10,
                num_workers=4,
                batch_size_for_parallel=2,
                language='java'  # ✅ 明确传递language参数
            )
            for v in variant_attacks:
                if isinstance(v, str) and v.strip() and v != variant_code:
                    attacked_codes.append(v)
    except:
        pass

    return attacked_codes


def compute_offset_thresholds(offset_before, quantile=0.6):
    """
    为每个维度计算offset阈值

    Args:
        offset_before: (N, 4) numpy array, 每个变体相对于簇中心的偏移
        quantile: float 分位数（0.6 = 60分位）

    Returns:
        thresholds: (4,) numpy array, 每个维度的阈值
    """
    thresholds = []
    for dim in range(offset_before.shape[1]):
        offsets_dim = np.abs(offset_before[:, dim])
        threshold = np.quantile(offsets_dim, quantile)
        thresholds.append(threshold)

    return np.array(thresholds)


def compute_offset_thresholds_torch(offset_before, quantile=0.6):
    """
    Torch版本的offset阈值计算，避免CPU往返
    Args:
        offset_before: (N, 4) torch.Tensor
    Returns:
        (4,) torch.Tensor
    """
    offsets_abs = torch.abs(offset_before)
    return torch.quantile(offsets_abs, quantile, dim=0)


def sign_preservation_loss_sample(Wb, embeddings_positive, embeddings_attacked, offset_thresholds_per_dim, num_attacks_per_variant):
    """
    单样本符号保持损失（选择性维度版）
    """
    device = embeddings_positive.device
    # ✅ 参数化 k
    k = Wb.shape[0]
    
    # 投影
    s_positive = embeddings_positive @ Wb.t()  # (N, k)
    s_attacked = embeddings_attacked @ Wb.t()  # (M, k)

    # 簇中心
    s0 = torch.median(s_positive, dim=0).values
    offset_before = s_positive - s0
    offset_after = s_attacked - s0

    if offset_after.numel() == 0:
        return torch.zeros(1, device=device)

    total_loss = torch.tensor(0.0, device=device)
    meaningful_dim_count = 0

    # 攻击索引映射
    num_attacks_tensor = torch.tensor(num_attacks_per_variant, device=device, dtype=torch.long)
    variant_ids = torch.repeat_interleave(torch.arange(len(num_attacks_per_variant), device=device), num_attacks_tensor)

    for dim in range(k):  # ✅ 参数化 k
        threshold = offset_thresholds_per_dim[dim]
        meaningful_mask = torch.abs(offset_before[:, dim]) >= threshold
        num_meaningful = meaningful_mask.sum().item()
        if num_meaningful == 0:
            continue
        meaningful_dim_count += 1

        valid_variant_ids = torch.nonzero(meaningful_mask, as_tuple=False).squeeze(-1)
        mask_attacks = (variant_ids.unsqueeze(1) == valid_variant_ids.unsqueeze(0)).any(dim=1)
        if mask_attacks.sum() == 0:
            continue

        offset_bef = offset_before[variant_ids[mask_attacks], dim]
        offset_att = offset_after[mask_attacks, dim]
        product = offset_bef * offset_att
        dim_loss = torch.relu(-product).mean()
        total_loss = total_loss + dim_loss

    if meaningful_dim_count > 0:
        return total_loss / meaningful_dim_count
    else:
        return torch.zeros(1, device=device)


def cluster_separation_loss_sample(Wb, embeddings_positive):
    """单样本簇分离损失：鼓励投影范围更大"""
    s_positive = embeddings_positive @ Wb.t()
    s_min = s_positive.min(dim=0).values
    s_max = s_positive.max(dim=0).values
    separation = (s_max - s_min).mean()
    return -separation


def compactness_loss_sample(Wb, embeddings_positive):
    """单样本紧致性损失：鼓励投影方差更小"""
    s_positive = embeddings_positive @ Wb.t()
    variance = torch.var(s_positive, dim=0).mean()
    return variance


def orthogonality_loss_W(Wb):
    """单样本正交性损失"""
    W_norm = torch.nn.functional.normalize(Wb, p=2, dim=1)
    gram = W_norm @ W_norm.t()
    identity = torch.eye(Wb.shape[0], device=Wb.device)
    return torch.mean((gram - identity) ** 2)


def sign_preservation_loss_selective_dims(W, embeddings_positive, embeddings_attacked, offset_thresholds_per_dim, num_attacks_per_variant=None):
    """
    改进的符号保持损失：只对超过阈值的维度计算

    Args:
        W: AdaptiveDirectionGenerator模块
        embeddings_positive: (N, 768) 等价变体嵌入
        embeddings_attacked: (M, 768) 攻击后嵌入
        offset_thresholds_per_dim: (k,) 每个维度的阈值
        num_attacks_per_variant: list of int，每个变体的实际攻击数
    """
    # ✅ 获取生成的方向和注意力权重
    W_directions, _ = W(embeddings_positive)  # (N, k, 768)
    k = W_directions.shape[1]  # ✅ 从 W_directions 动态获取 k

    # 投影
    s_positive = torch.bmm(embeddings_positive.unsqueeze(0), W_directions[0].t().unsqueeze(0)).squeeze(0)  # (N, k)

    # 简化：只对第一个batch处理
    if embeddings_positive.shape[0] > 1:
        s_positive = torch.stack([
            embeddings_positive[i] @ W_directions[i].t() for i in range(embeddings_positive.shape[0])
        ])

    s_attacked_list = []
    for i in range(embeddings_attacked.shape[0]):
        # 找到最相关的方向（这里简化处理，使用第一个）
        s_att = embeddings_attacked[i] @ W_directions[0].t()
        s_attacked_list.append(s_att)
    s_attacked = torch.stack(s_attacked_list)  # (M, k)

    # 簇中心
    s_positive_np = s_positive.detach().cpu().numpy()
    s0_np = np.array([np.median(s_positive_np[:, i]) for i in range(k)])  # ✅ 参数化 k
    s0 = torch.from_numpy(s0_np).to(s_positive.device).float()

    # 偏移
    offset_before = s_positive - s0  # (N, k)
    offset_after = s_attacked - s0   # (M, k)

    # 对每个维度分别计算
    total_loss = torch.tensor(0.0, device=s_positive.device)
    meaningful_dim_count = 0

    # 如果没有提供num_attacks_per_variant，假设均匀分布
    if num_attacks_per_variant is None:
        N_per_attack = embeddings_attacked.shape[0] // embeddings_positive.shape[0]
        num_attacks_per_variant = [N_per_attack] * embeddings_positive.shape[0]

    for dim in range(k):  # ✅ 参数化 k
        threshold = offset_thresholds_per_dim[dim]

        # 只选择超过阈值的变体
        meaningful_mask = torch.abs(offset_before[:, dim]) >= threshold  # (N,)
        num_meaningful = torch.sum(meaningful_mask).item()

        if num_meaningful == 0:
            continue

        meaningful_dim_count += 1

        # 计算这个维度的损失
        dim_loss = torch.tensor(0.0, device=s_positive.device)

        # 构建attack_idx到variant_idx的映射
        attack_idx = 0
        for var_idx in range(embeddings_positive.shape[0]):
            num_attacks = num_attacks_per_variant[var_idx]
            start_idx = attack_idx
            end_idx = attack_idx + num_attacks

            # 如果这个变体超过阈值，计算损失
            if meaningful_mask[var_idx]:
                # 维度级别的数据
                offset_bef = offset_before[var_idx, dim]  # scalar
                offset_att = offset_after[start_idx:end_idx, dim]  # (num_attacks,)

                # 符号保持检查
                product = offset_bef * offset_att
                if len(offset_att) > 0:
                    dim_loss = dim_loss + F.relu(-product).mean()

            attack_idx = end_idx

        # 平均到这个维度的变体数
        dim_loss = dim_loss / num_meaningful
        total_loss = total_loss + dim_loss

    # 除以有意义的维度数
    if meaningful_dim_count > 0:
        return total_loss / meaningful_dim_count
    else:
        return torch.zeros(1, device=s_positive.device, requires_grad=True)


def process_one_code(task):
    """
    处理单个代码的训练数据（用于并发处理）
    """
    run_id, code, base_config, num_attacks_subsample = task

    try:
        # 在子进程中确保路径正确设置（防御性措施）
        import sys
        from pathlib import Path
        
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent.parent
        srcmarker_path = project_root / "SrcMarker-main"
        contrastive_path = srcmarker_path / "contrastive_learning"
        
        # 添加必要的路径到sys.path
        for path in [str(project_root), str(srcmarker_path), str(contrastive_path)]:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        # 在每个进程中独立加载模型（避免跨进程共享问题）
        # ====== NEW: 支持 Variant A (w/o Contrastive Learning) ======
        import torch
        
        if not base_config.get('use_contrastive_learning', True):
            # Variant A: 加载原始 CodeT5（无对比学习微调）
            from transformers import T5EncoderModel, AutoTokenizer as HFAutoTokenizer
            print("[Variant A: w/o CL] 加载原始 CodeT5（无对比学习微调）")
            model_name = "Salesforce/codet5-base"
            # 确定设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # 加载模型并设置到正确的设备和模式
            model = T5EncoderModel.from_pretrained(model_name)
            model.to(device)  # ⚠️ 重要：设置到正确的设备
            model.eval()      # ⚠️ 重要：设置为评估模式，禁用Dropout/BatchNorm
            tokenizer = HFAutoTokenizer.from_pretrained(model_name)
        else:
            # Baseline: 加载微调后的编码器（有PEFT适配器）
            # load_best_model 已经处理了 .to(device) 和 .eval()
            model, tokenizer = load_best_model(base_config['model_dir'])
            device = next(model.parameters()).device
        
        # device 已在上方设置

        # 获取变体数量
        num_variants = base_config.get('embedding', {}).get('cluster_variants', 100)
        num_workers = base_config.get('embedding', {}).get('num_workers', 16)
        batch_size_for_parallel = base_config.get('embedding', {}).get('batch_size_for_parallel', 4)

        # 根据语言选择不同的变体生成方式
        dataset_name = base_config['dataset']['name']
        if 'js' in dataset_name.lower():
            language = 'javascript'
        elif 'cpp' in dataset_name.lower() or 'c++' in dataset_name.lower():
            language = 'cpp'
        else:
            language = 'java'
        
        # [检查点1] 验证语言检测
        print(f"[DEBUG] 变体生成阶段 - dataset_name={dataset_name}, detected language={language}")
        
        if language == 'javascript':
            # JavaScript: 直接使用augmentor生成变体（绕过硬编码的Java函数）
            try:
                # contrastive_path 已在函数开头设置
                if str(contrastive_path) not in sys.path:
                    sys.path.insert(0, str(contrastive_path))
                
                from contrastive_learning.js_augmentor import JavaScriptCodeAugmentor
                js_aug = JavaScriptCodeAugmentor()
                
                variants = []
                for _ in range(num_variants):
                    # 应用随机的语义保持变换
                    variant = code
                    if np.random.random() < 0.7:  # 70%概率应用变换
                        variant = js_aug._apply_mutable_ast_transforms(variant)
                    if variant.strip() and variant != code:
                        variants.append(variant)
                    else:
                        variants.append(code)
                
                # 确保正好num_variants个
                while len(variants) < num_variants:
                    variants.append(code)
                variants = variants[:num_variants]
            except Exception as e:
                print(f"\n[WARNING] JS变体生成失败: {e}，使用原代码")
                variants = [code] * num_variants
        elif language == 'cpp':
            # C++: 使用CppCodeAugmentor生成变体（对标JavaScript）
            try:
                # contrastive_path 已在函数开头设置
                if str(contrastive_path) not in sys.path:
                    sys.path.insert(0, str(contrastive_path))
                
                from contrastive_learning.cpp_augmentor import CppCodeAugmentor
                cpp_aug = CppCodeAugmentor()
                
                variants = []
                for _ in range(num_variants):
                    # 应用随机的语义保持变换（对标JavaScript的逻辑）
                    variant = code
                    if np.random.random() < 0.7:  # 70%概率应用变换
                        try:
                            variant = cpp_aug._apply_mutable_ast_transforms(variant)
                        except Exception:
                            # 如果MutableAST失败，尝试使用通用augment方法
                            augmented = cpp_aug.augment(variant)
                            if augmented and isinstance(augmented, list) and len(augmented) > 0:
                                variant = augmented[0]
                    
                    if variant.strip() and variant != code:
                        variants.append(variant)
                    else:
                        variants.append(code)
                
                # 确保正好num_variants个
                while len(variants) < num_variants:
                    variants.append(code)
                variants = variants[:num_variants]
            except Exception as e:
                print(f"\n[WARNING] C++变体生成失败: {e}，使用原代码")
                variants = [code] * num_variants
        else:
            # Java: 使用原有的build_candidates_test_like
            try:
                cands, stats = build_candidates_test_like(
                    code,
                    K=num_variants,
                    num_workers=num_workers,
                    batch_size_for_parallel=batch_size_for_parallel,
                    language='java'  # ✅ 明确传递language参数
                )
                # 过滤掉无效变体
                variants = [c for c in cands if isinstance(c, str) and c.strip() and c.strip() != code.strip()]
                # 如果生成的变体数量不足，用原代码填充
                while len(variants) < num_variants:
                    variants.append(code)
                variants = variants[:num_variants]  # 确保正好num_variants个
            except Exception as e:
                variants = [code] * num_variants

        # 编码变体（只编码变体，不包含原代码）
        embeddings_positive = embed_codes_universal(
                    model, tokenizer, variants,
                    batch_size=base_config['embedding']['batch_size_for_parallel'],
                    device=device
                )  # (num_variants, 768)

        # 生成攻击版本：使用MutableAST变换
        # 识别语言
        dataset_name = base_config['dataset']['name']
        if 'js' in dataset_name.lower():
            language = 'javascript'
        elif 'cpp' in dataset_name.lower() or 'c++' in dataset_name.lower():
            language = 'cpp'
        else:
            language = 'java'
        
        augmentor = get_augmentor_for_language(language)
        
        # [检查点2] 验证augmentor类型与语言匹配
        augmentor_class = augmentor.__class__.__name__
        print(f"[DEBUG] 攻击生成阶段 - language={language}, augmentor={augmentor_class}")
        
        if language == 'cpp' and augmentor_class != 'CppCodeAugmentor':
            print(f"[CRITICAL ERROR] C++ 语言应使用 CppCodeAugmentor，但获得了 {augmentor_class}！")
            raise ValueError(f"Language mismatch: language={language} but augmentor={augmentor_class}")
        elif language == 'javascript' and augmentor_class != 'JavaScriptCodeAugmentor':
            print(f"[WARNING] Expected JavaScriptCodeAugmentor but got {augmentor_class} for language={language}")
        elif language == 'java' and augmentor_class != 'JavaCodeAugmentor':
            print(f"[WARNING] Expected JavaCodeAugmentor but got {augmentor_class} for language={language}")
        
        attacked_codes = []
        num_attacks_per_variant = []

        # [检查点3] 日志记录即将传递的参数
        print(f"[DEBUG] 准备调用 generate_attacks_via_mutable_ast: language={language}, augmentor={augmentor_class}, num_variants={num_variants}")
        
        # 对每个变体生成MutableAST攻击
        for i in range(num_variants):
            variant_attacks = generate_attacks_via_mutable_ast(
                variants[i],
                augmentor,
                language
            )
            attacked_codes.extend(variant_attacks)
            num_attacks_per_variant.append(len(variant_attacks))

        # 编码攻击版本
        embeddings_attacked = embed_codes_universal(
            model, tokenizer, attacked_codes,
            batch_size=base_config['embedding']['batch_size_for_parallel'],
            device=device
        )  # (num_variants * num_attacks_per_variant, 768)

        # embed_codes 已经返回numpy数组，直接使用
        data = {
            'embedding_original': embeddings_positive[0],  # 第一个变体作为代表
            'embeddings_variants': embeddings_positive,  # 已经是numpy数组
            'embeddings_attacked': embeddings_attacked,   # 已经是numpy数组
            'num_attacks_per_variant': num_attacks_per_variant,  # 每个变体的实际攻击数
            'target_bits': base_config['embedding']['bits']  # 目标比特
        }

        # 清理GPU内存
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {'run_id': run_id, 'success': True, 'data': data, 'error': None}

    except Exception as e:
        # 清理资源
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {'run_id': run_id, 'success': False, 'data': None, 'error': str(e)}


def load_training_data(base_config, config, concurrency=1):
    """
    加载训练数据，支持MutableAST攻击
    """
    import sys
    from pathlib import Path as PathLib
    
    print("加载训练数据...")

    # 加载训练代码（支持data_split配置）
    if base_config.get('data_split', {}).get('enabled', False):
        split_config = base_config['data_split']['strategy_5_training']
        num_codes = split_config['num_codes']
        start_idx = split_config['start_index']
    else:
        num_codes = base_config['num_test_codes']
        start_idx = base_config['start_index']

    print(f"  加载 {num_codes} 个代码（start_index={start_idx}，使用train分割）...")
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset_loader import DatasetLoader
    loader = DatasetLoader(base_config)
    test_codes = loader.load_codes(num_codes, start_idx, split='train')

    # 准备任务列表
    tasks = [(run_id, code, base_config, 10)
             for run_id, code in enumerate(test_codes)]

    # 并发或串行执行
    if concurrency > 1:
        print(f"  使用并发模式 (max_workers={concurrency})")
        training_data_dict = {}

        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_one_code, task) for task in tasks]

            for future in tqdm(as_completed(futures), total=len(futures), desc="  处理代码", ncols=80):
                result = future.result()
                if result['success']:
                    training_data_dict[result['run_id']] = result['data']
                else:
                    print(f"\n[错误] 代码 {result['run_id']} 处理失败: {result['error']}")
                    # 失败时使用空数据
                    num_variants = base_config.get('embedding', {}).get('cluster_variants', 100)
                    training_data_dict[result['run_id']] = {
                        'embedding_original': np.zeros(768),
                        'embeddings_variants': np.zeros((num_variants, 768)),
                        'embeddings_attacked': np.zeros((0, 768)),
                        'num_attacks_per_variant': [0] * num_variants,
                        'target_bits': base_config['embedding']['bits']
                    }

        # 按run_id顺序排列
        training_data = [training_data_dict[i] for i in range(len(test_codes))]
    else:
        print(f"  使用串行模式")
        # ====== NEW: 支持 Variant A (w/o Contrastive Learning) ======
        import torch
        
        if not base_config.get('use_contrastive_learning', True):
            # Variant A: 加载原始 CodeT5
            from transformers import T5EncoderModel, AutoTokenizer as HFAutoTokenizer
            print("[Variant A: w/o CL] 加载原始 CodeT5（无对比学习微调）")
            model_name = "Salesforce/codet5-base"
            # 确定设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # 加载模型并设置到正确的设备和模式
            model = T5EncoderModel.from_pretrained(model_name)
            model.to(device)  # ⚠️ 重要：设置到正确的设备
            model.eval()      # ⚠️ 重要：设置为评估模式，禁用Dropout/BatchNorm
            tokenizer = HFAutoTokenizer.from_pretrained(model_name)
        else:
            # Baseline: 加载微调后的编码器
            # load_best_model 已经处理了 .to(device) 和 .eval()
            model, tokenizer = load_best_model(base_config['model_dir'])
            device = next(model.parameters()).device
        
        # device 已在上方设置

        num_variants = base_config.get('embedding', {}).get('cluster_variants', 100)
        num_workers = base_config.get('embedding', {}).get('num_workers', 16)
        batch_size_for_parallel = base_config.get('embedding', {}).get('batch_size_for_parallel', 4)

        # 识别语言
        dataset_name = base_config['dataset']['name']
        if 'js' in dataset_name.lower():
            language = 'javascript'
        elif 'cpp' in dataset_name.lower() or 'c++' in dataset_name.lower():
            language = 'cpp'
        else:
            language = 'java'
        
        # 为JavaScript和C++预加载augmentor
        js_aug = None
        cpp_aug = None
        if language == 'javascript':
            try:
                contrastive_path = project_root / "SrcMarker-main" / "contrastive_learning"
                if str(contrastive_path) not in sys.path:
                    sys.path.insert(0, str(contrastive_path))
                from contrastive_learning.js_augmentor import JavaScriptCodeAugmentor
                js_aug = JavaScriptCodeAugmentor()
            except Exception as e:
                print(f"\n[WARNING] 无法加载JS augmentor: {e}")
        elif language == 'cpp':
            # C++: 预加载CppCodeAugmentor（对标JavaScript）
            try:
                contrastive_path = project_root / "SrcMarker-main" / "contrastive_learning"
                if str(contrastive_path) not in sys.path:
                    sys.path.insert(0, str(contrastive_path))
                from contrastive_learning.cpp_augmentor import CppCodeAugmentor
                cpp_aug = CppCodeAugmentor()
            except Exception as e:
                print(f"\n[WARNING] 无法加载C++ augmentor: {e}")
        
        training_data = []
        for run_id in tqdm(range(len(test_codes)), desc="  处理代码", ncols=80):
            code = test_codes[run_id]

            if language == 'javascript' and js_aug is not None:
                # JavaScript: 使用augmentor生成变体
                try:
                    variants = []
                    for _ in range(num_variants):
                        variant = code
                        if np.random.random() < 0.7:  # 70%概率应用变换
                            variant = js_aug._apply_mutable_ast_transforms(variant)
                        if variant.strip() and variant != code:
                            variants.append(variant)
                        else:
                            variants.append(code)
                    while len(variants) < num_variants:
                        variants.append(code)
                    variants = variants[:num_variants]
                except Exception as e:
                    variants = [code] * num_variants
            elif language == 'cpp' and cpp_aug is not None:
                # C++: 使用augmentor生成变体（对标JavaScript）
                try:
                    variants = []
                    for _ in range(num_variants):
                        variant = code
                        if np.random.random() < 0.7:  # 70%概率应用变换
                            try:
                                variant = cpp_aug._apply_mutable_ast_transforms(variant)
                            except Exception:
                                # 如果MutableAST失败，尝试使用通用augment方法
                                augmented = cpp_aug.augment(variant)
                                if augmented and isinstance(augmented, list) and len(augmented) > 0:
                                    variant = augmented[0]
                        
                        if variant.strip() and variant != code:
                            variants.append(variant)
                        else:
                            variants.append(code)
                    while len(variants) < num_variants:
                        variants.append(code)
                    variants = variants[:num_variants]
                except Exception as e:
                    variants = [code] * num_variants
            else:
                # Java: 使用原有的build_candidates_test_like
                try:
                    cands, stats = build_candidates_test_like(
                        code,
                        K=num_variants,
                        num_workers=num_workers,
                        batch_size_for_parallel=batch_size_for_parallel,
                        language='java'  # ✅ 明确传递language参数
                    )
                    variants = [c for c in cands if isinstance(c, str) and c.strip() and c.strip() != code.strip()]
                    while len(variants) < num_variants:
                        variants.append(code)
                    variants = variants[:num_variants]
                except Exception as e:
                    variants = [code] * num_variants

            embeddings_positive = embed_codes_universal(
                    model, tokenizer, variants,
                    batch_size=base_config['embedding']['batch_size_for_parallel'],
                    device=device
                )

            # 识别语言并获取augmentor
            dataset_name = base_config['dataset']['name']
            if 'js' in dataset_name.lower():
                language = 'javascript'
            elif 'cpp' in dataset_name.lower() or 'c++' in dataset_name.lower():
                language = 'cpp'
            else:
                language = 'java'
            augmentor = get_augmentor_for_language(language)
            
            attacked_codes = []
            num_attacks_per_variant = []

            # 对每个变体生成MutableAST攻击
            for i in range(num_variants):
                variant_attacks = generate_attacks_via_mutable_ast(
                    variants[i],
                    augmentor,
                    language
                )
                attacked_codes.extend(variant_attacks)
                num_attacks_per_variant.append(len(variant_attacks))

            embeddings_attacked = embed_codes_universal(
                model, tokenizer, attacked_codes,
                batch_size=base_config['embedding']['batch_size_for_parallel'],
                device=device
            )

            # embed_codes 已经返回numpy数组，直接使用
            training_data.append({
                'embedding_original': embeddings_positive[0],
                'embeddings_variants': embeddings_positive,  # 已经是numpy数组
                'embeddings_attacked': embeddings_attacked,   # 已经是numpy数组
                'num_attacks_per_variant': num_attacks_per_variant,  # 每个变体的实际攻击数
                'target_bits': base_config['embedding']['bits']
            })

        # 清理模型
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return training_data


def train_adaptive_generator(training_data, config, device='cuda'):
    """训练自适应方向生成器"""
    print("\n开始训练...")

    # 初始化模型
    generator = AdaptiveDirectionGenerator(
        d=config['direction_generation']['d'],
        k=config['direction_generation']['k'],
        hidden_dims=config['learning']['hidden_dims'],
        use_attention=config['learning']['use_attention']
    ).to(device)

    # 优化器
    optimizer = optim.AdamW(
        generator.parameters(),
        lr=config['learning']['learning_rate'],
        weight_decay=config['learning']['weight_decay']
    )

    # 损失权重
    loss_weights = config['learning']['loss_weights']

    # 计算offset阈值（从前5个代码的等价变体）
    print("\n计算offset阈值...")
    offset_thresholds_per_dim = None

    with torch.no_grad():
        num_samples = min(5, len(training_data))
        for i in range(num_samples):
            embeddings_positive = torch.from_numpy(training_data[i]['embeddings_variants']).float().to(device)

            # 简化处理：使用mean作为简单投影
            s_positive_np = embeddings_positive.cpu().numpy()
            # 这里需要更复杂的处理，但为了简化，暂时使用mean
            s_positive_mean = np.mean(s_positive_np, axis=1, keepdims=True)

            # 简单的offset计算
            s0_np = np.median(s_positive_np, axis=0)
            offset_before = s_positive_np - s0_np

            # 计算阈值
            if offset_thresholds_per_dim is None:
                offset_thresholds_per_dim = compute_offset_thresholds(
                    offset_before,
                    quantile=config['learning'].get('threshold_quantile', 0.6)
                )

    print(f"  Offset阈值: {offset_thresholds_per_dim}")

    # 训练循环
    num_epochs = config['learning']['epochs']
    batch_size = config['learning']['batch_size']

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_sign = 0
        epoch_sep = 0
        epoch_comp = 0
        epoch_orth = 0
        epoch_attn = 0
        count_batches = 0

        # 随机打乱数据
        indices = np.random.permutation(len(training_data))

        # 分批训练
        for i in range(0, len(training_data), batch_size):
            batch_indices = indices[i:i+batch_size]

            # 简化版本的批处理
            batch_size_actual = len(batch_indices)

            # 收集batch数据
            embeddings_original_batch = []
            for idx in batch_indices:
                embedding = torch.from_numpy(training_data[idx]['embedding_original']).float()
                embeddings_original_batch.append(embedding)

            embeddings_original = torch.stack(embeddings_original_batch).to(device)  # (B, 768)

            # 前向传播：生成方向
            W, attention = generator(embeddings_original)  # (B, 4, 768)

            # 初始化各项损失
            loss_sign = torch.tensor(0.0, device=device)
            loss_sep = torch.tensor(0.0, device=device)
            loss_comp = torch.tensor(0.0, device=device)
            loss_orth = torch.tensor(0.0, device=device)
            loss_attn = torch.tensor(0.0, device=device)

            # 逐样本计算
            for local_idx, idx in enumerate(batch_indices):
                sample = training_data[idx]
                emb_var = torch.from_numpy(sample['embeddings_variants']).float().to(device)  # (Nv,768)
                emb_att = torch.from_numpy(sample['embeddings_attacked']).float().to(device)  # (Na,768)
                num_att = sample['num_attacks_per_variant']
                Wb = W[local_idx]

                # 阈值：基于变体投影
                s_pos = emb_var @ Wb.t()
                s0 = torch.median(s_pos, dim=0).values
                offset_before = s_pos - s0
                thresholds = compute_offset_thresholds_torch(offset_before, quantile=config['learning'].get('threshold_quantile', 0.6))

                # 各项损失
                loss_sign = loss_sign + sign_preservation_loss_sample(Wb, emb_var, emb_att, thresholds, num_att)
                loss_sep = loss_sep + cluster_separation_loss_sample(Wb, emb_var)
                loss_comp = loss_comp + compactness_loss_sample(Wb, emb_var)
                loss_orth = loss_orth + orthogonality_loss_W(Wb)

            # 平均到batch
            if batch_size_actual > 0:
                loss_sign = loss_sign / batch_size_actual
                loss_sep = loss_sep / batch_size_actual
                loss_comp = loss_comp / batch_size_actual
                loss_orth = loss_orth / batch_size_actual
                if attention is not None:
                    loss_attn = attention.abs().mean()
                else:
                    loss_attn = torch.tensor(0.0, device=device)

            # L2范数归一化各项损失（保留正负号，处理零值情况）
            if config.get('loss', {}).get('normalize_losses', False):
                # 收集各项损失（使用绝对值计算范数，但保持原始符号）
                loss_terms = torch.stack([loss_sign, loss_sep, loss_comp, loss_orth, loss_attn])

                # 计算L2范数（使用绝对值以正确处理负值）
                loss_norm = torch.norm(loss_terms, p=2)

                # 只有当范数大于阈值时才进行归一化，避免除零
                if loss_norm > config.get('loss', {}).get('eps', 1e-8):
                    # 归一化各项损失，保持原始符号
                    loss_sign = loss_sign / loss_norm
                    loss_sep = loss_sep / loss_norm
                    loss_comp = loss_comp / loss_norm
                    loss_orth = loss_orth / loss_norm
                    loss_attn = loss_attn / loss_norm

            # 按权重组合
            lw = loss_weights
            total_loss = (
                lw.get('sign_variants', 0.0) * loss_sign +
                lw.get('separation', 0.0) * loss_sep +
                lw.get('compactness', 0.0) * loss_comp +
                lw.get('orthogonality', 1.0) * loss_orth +
                lw.get('attention_sparsity', 0.0) * loss_attn
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_sign += loss_sign.item()
            epoch_sep += loss_sep.item()
            epoch_comp += loss_comp.item()
            epoch_orth += loss_orth.item()
            epoch_attn += loss_attn.item()
            count_batches += 1

        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if count_batches == 0:
                count_batches = 1
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"Loss={epoch_loss/count_batches:.4f} "
                  f"[Sign={epoch_sign/count_batches:.4f}, "
                  f"Sep={epoch_sep/count_batches:.4f}, "
                  f"Orth={epoch_orth/count_batches:.4f}, "
                  f"Comp={epoch_comp/count_batches:.4f}, "
                  f"Attn={epoch_attn/count_batches:.4f}]")

    return generator


def main():
    parser = argparse.ArgumentParser(description="Step 1c: 训练自适应方向生成器（策略6）")
    parser.add_argument('--concurrency', type=int, default=5, help='并发处理的进程数（默认=5）')
    parser.add_argument('--resume', action='store_true', help='加载已保存的训练数据')
    parser.add_argument('--config', type=str, 
                       default=None,
                       help='配置文件路径（相对于项目根目录）')
    args = parser.parse_args()

    print("="*80)
    print(f"Step 1c: 训练自适应方向生成器 (Strategy 6) (concurrency={args.concurrency})")
    print("="*80)

    # 设置工作目录
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    # 加载配置（与step3/step4保持一致的处理方式）
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            # 相对路径：加上 dimension_strategy_comparison 前缀
            if not str(config_path).startswith('dimension_strategy_comparison'):
                config_path = project_root / "dimension_strategy_comparison" / config_path
            else:
                config_path = project_root / config_path
        else:
            config_path = Path(args.config)
    else:
        config_path = project_root / "dimension_strategy_comparison" / "configs" / "base_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        base_config = json.load(f)

    # ✅ 根据 bits 长度动态读取对应的 strategy config
    k = len(base_config['embedding']['bits'])
    if k == 2:
        strategy_config_file = "dimension_strategy_comparison/configs/strategy_6_adaptive_2bit.json"
    elif k == 6:
        strategy_config_file = "dimension_strategy_comparison/configs/strategy_6_adaptive_6bit.json"
    elif k == 8:
        strategy_config_file = "dimension_strategy_comparison/configs/strategy_6_adaptive_8bit.json"
    else:  # 默认 4bit
        strategy_config_file = "dimension_strategy_comparison/configs/strategy_6_adaptive.json"
    
    with open(strategy_config_file, 'r', encoding='utf-8') as f:
        strategy_config = json.load(f)
    
    # 识别语言
    dataset_name = base_config['dataset']['name']
    if 'js' in dataset_name.lower():
        language = 'javascript'
    elif 'cpp' in dataset_name.lower() or 'c++' in dataset_name.lower():
        language = 'cpp'
    else:
        language = 'java'
    print(f"检测到语言: {language}")
    print(f"数据集: {dataset_name}")

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")

    # ===== 根据配置、bit和语言设置输出目录（与Step 2保持一致） =====
    # ✅ 计算 bit_suffix
    k = len(base_config['embedding']['bits'])
    bit_suffix = '_2bit' if k == 2 else ('_6bit' if k == 6 else ('_8bit' if k == 8 else ''))
    
    # 确定变体前缀
    use_cl = base_config.get('use_contrastive_learning', True)
    if not use_cl:
        variant_prefix = f"strategy_6_no_cl{bit_suffix}"  # ✅ 添加 bit_suffix
        print(f"[Variant A: w/o CL] 使用原始CodeT5（无对比学习微调）")
    else:
        variant_prefix = f"strategy_6_adaptive{bit_suffix}"  # ✅ 添加 bit_suffix
        print(f"[Baseline: w/ CL] 使用微调后的CodeT5（含对比学习）")
    
    # 根据语言确定后缀
    if language == 'javascript':
        strategy_name = f"{variant_prefix}_js"
    elif language == 'cpp':
        strategy_name = f"{variant_prefix}_cpp"
    else:
        strategy_name = variant_prefix
    
    print(f"结果目录: {strategy_name}\n")
    
    output_dir = Path(f"dimension_strategy_comparison/results/{strategy_name}")
    trained_generator_file = output_dir / "trained_generator.pth"
    cache_file = output_dir / "training_data.pkl"

    # 检查是否已训练完成
    if args.resume and trained_generator_file.exists():
        print("检测到已训练完成的模型:")
        print(f"  - {trained_generator_file}")
        print("\n跳过训练步骤。如需重新训练，请删除这些文件或不使用--resume参数。\n")
        print("="*80)
        print("训练已完成（跳过）")
        print("="*80)
        return

    # 尝试加载已保存的训练数据
    training_data = None
    if args.resume and cache_file.exists():
        print("尝试加载已保存的训练数据...")
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                training_data = pickle.load(f)
            print(f"[OK] 成功加载训练数据 ({len(training_data)}个代码)\n")
        except Exception as e:
            print(f"加载失败: {e}")
            training_data = None

    # 如果没有加载到数据，重新生成
    if training_data is None:
        if args.resume:
            print("未找到已保存的训练数据，将重新生成\n")

        training_data = load_training_data(
            base_config,
            strategy_config,
            concurrency=args.concurrency
        )

        # 保存训练数据
        print("\n保存训练数据...")
        output_dir.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(training_data, f)
        print(f"[OK] 训练数据已保存: {cache_file}")

    print(f"\n训练数据: {len(training_data)} 个代码，每个100个变体+多个MutableAST攻击样本")

    # 训练
    generator = train_adaptive_generator(training_data, strategy_config, device)

    # 保存模型（使用之前确定的 strategy_name）
    # strategy_name 在前面已根据语言设置（javascript → strategy_6_adaptive_js, java → strategy_6_adaptive）
    output_dir = Path(f"dimension_strategy_comparison/results/{strategy_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存生成器
    torch.save(generator.state_dict(), output_dir / "trained_generator.pth")
    print(f"\n[OK] 已保存生成器到: {output_dir / 'trained_generator.pth'}")

    # 打印最终结果
    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)


if __name__ == '__main__':
    main()
