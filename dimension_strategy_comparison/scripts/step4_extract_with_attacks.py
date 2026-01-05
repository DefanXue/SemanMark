"""
Step 4: 应用攻击并提取水印

输入：
  - results/strategy_X/embedding/run_XXXX/watermarked.java
  - results/strategy_X/embedding/run_XXXX/final.json

输出：
  - results/strategy_X/extraction/attack_X.XX/run_XXXX_seed_XXX.json
"""

import os
import sys
import json
import argparse
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import tree_sitter

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 添加SrcMarker路径
srcmarker_path = project_root / "SrcMarker-main"
sys.path.insert(0, str(srcmarker_path))

# 添加contrastive_learning路径（用于Augmentor）
contrastive_path = srcmarker_path / "contrastive_learning"
sys.path.insert(0, str(contrastive_path))

from Watermark4code.encoder.loader import load_best_model, embed_codes

# 动态导入augmentor和renamer（根据语言自动选择）
def get_augmentor_and_renamer(language):
    """根据语言返回对应的代码增强器和重命名器（并发安全版本）"""
    import sys
    from pathlib import Path
    
    # 确保contrastive_learning路径在sys.path中（对子进程也有效）
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    contrastive_path = project_root / "SrcMarker-main" / "contrastive_learning"
    if str(contrastive_path) not in sys.path:
        sys.path.insert(0, str(contrastive_path))
    
    if language == 'java':
        # 使用绝对导入确保在子进程中也能找到
        from contrastive_learning.java_augmentor import JavaCodeAugmentor
        from Watermark4code.experiments.Attack.Rename_Attack.java_variable_renamer import JavaVariableRenamer
        return JavaCodeAugmentor(), JavaVariableRenamer
    elif language == 'cpp':
        # C++增强器和重命名器
        from contrastive_learning.cpp_augmentor import CppCodeAugmentor
        from Watermark4code.experiments.Attack.Rename_Attack.cpp_variable_renamer import CppVariableRenamer
        return CppCodeAugmentor(), CppVariableRenamer
    elif language == 'javascript':
        # 使用绝对导入确保在子进程中也能找到
        from contrastive_learning.js_augmentor import JavaScriptCodeAugmentor
        # JS没有专门的renamer，返回None
        return JavaScriptCodeAugmentor(), None
    else:
        raise ValueError(f"Unsupported language: {language}")

# 导入CodeBLEU
sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics.calc_code_bleu import evaluate_per_example

# 导入MutableAST转换器
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

# 全局tree-sitter解析器
PARSER_LANG = None
ts_parser = None


def apply_rename_attack(code, ratio, seed, language='java', augmentor=None):
    """应用重命名攻击（支持Java、C++和JavaScript）"""
    try:
        if ratio == 0.0:
            return code

        if language == 'java':
            from Watermark4code.experiments.Attack.Rename_Attack.java_variable_renamer import JavaVariableRenamer
            from Watermark4code.experiments.Attack.Rename_Attack.attack_config import AttackConfig
            renamer = JavaVariableRenamer(code)
            config = AttackConfig(naming_strategy="random", seed=seed)
            attacked = renamer.apply_renames(config, rename_ratio=ratio)
            return attacked
        elif language == 'cpp':
            from Watermark4code.experiments.Attack.Rename_Attack.cpp_variable_renamer import CppVariableRenamer
            from Watermark4code.experiments.Attack.Rename_Attack.attack_config import AttackConfig
            renamer = CppVariableRenamer(code)
            config = AttackConfig(naming_strategy="random", seed=seed)
            attacked = renamer.apply_renames(config, rename_ratio=ratio)
            return attacked
        else:  # javascript
            from Watermark4code.experiments.Attack.Rename_Attack.js_variable_renamer import JavaScriptVariableRenamer
            from Watermark4code.experiments.Attack.Rename_Attack.attack_config import AttackConfig
            renamer = JavaScriptVariableRenamer(code)
            config = AttackConfig(naming_strategy="random", seed=seed)
            attacked = renamer.apply_renames(config, rename_ratio=ratio)
            return attacked
    except Exception as e:
        print(f"\n[警告] 攻击失败 (ratio={ratio}, seed={seed}): {e}")
        return code


def apply_t_transform_attack(code, n_transforms, seed=42, language='java', augmentor=None):
    """
    应用T Transform攻击：应用最多n个随机的MutableAST代码转换（支持Java、C++和JavaScript）

    如果可行转换少于n个，就应用所有可行转换

    Args:
        code: 原始代码
        n_transforms: 目标转换数量(1/2/3)
        seed: 随机种子
        language: 语言类型 ('java'、'cpp' 或 'javascript')
        augmentor: 代码增强器实例

    Returns:
        (attacked_code, attack_info) 元组
        - attacked_code: 攻击后的代码
        - attack_info: dict包含应用的转换信息
    """
    try:
        random.seed(seed)

        # 初始化augmentor用于提取/包装方法
        if augmentor is None:
            augmentor, _ = get_augmentor_and_renamer(language)

        # C++使用不同的方法
        if language == 'cpp':
            # C++使用augmentor的结构性变换，不依赖MutableAST
            attacked_code = code
            applied_transforms = []
            
            # 可用的结构性变换列表
            available_methods = [
                ('swap_if_condition', augmentor._swap_if_condition),
                ('convert_for_to_while', augmentor._convert_for_to_while),
                ('convert_while_to_for', augmentor._convert_while_to_for),
                ('swap_variable_declarations', augmentor._swap_variable_declarations),
            ]
            
            # 随机选择要应用的转换
            n_to_apply = min(n_transforms, len(available_methods))
            selected_methods = random.sample(available_methods, n_to_apply)
            
            # 应用选中的转换
            for transform_name, transform_func in selected_methods:
                try:
                    new_code = transform_func(attacked_code)
                    if new_code != attacked_code:
                        attacked_code = new_code
                        applied_transforms.append(transform_name)
                except:
                    pass
            
            return attacked_code, {'transforms': applied_transforms, 'num_transforms': len(applied_transforms)}

        # 提取方法代码（去除类包装）
        method_code, metadata = augmentor._extract_method_from_full_class(code)

        # 创建所有转换器实例
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

        # 在worker进程中本地初始化tree-sitter（不使用全局对象，避免跨进程问题）
        project_root = Path(__file__).parent.parent.parent
        lang_so = project_root / "SrcMarker-main" / "parser" / "languages.so"

        if not lang_so.exists():
            # languages.so不存在，无法进行转换
            return code, {'error': 'languages.so not found'}

        parser_lang = tree_sitter.Language(str(lang_so), language)
        ts_parser_local = tree_sitter.Parser()
        ts_parser_local.set_language(parser_lang)

        # 初始化CodeTransformProvider
        transform_provider = CodeTransformProvider(language, ts_parser_local, transformers)

        # 枚举所有可行的转换键（对方法代码进行测试）
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

        if len(feasible_transforms) == 0:
            # 没有可行转换，返回原代码
            return code, {}

        # 随机选择min(n_transforms, 可行数)个转换
        n_to_apply = min(n_transforms, len(feasible_transforms))
        selected = random.sample(feasible_transforms, n_to_apply)

        # 构建键列表并一次性应用（对方法代码）
        selected_keys = [key for _, key in selected]
        transformed_method = transform_provider.code_transform(method_code, selected_keys)

        # 包装回完整类
        if metadata is not None:
            attacked_code = augmentor._wrap_method_back_to_class(transformed_method, metadata)
        else:
            attacked_code = transformed_method

        # 记录应用的转换信息
        attack_info = {
            'transforms': [f"{name}:{key}" for name, key in selected],
            'num_transforms': len(selected),
            'feasible_total': len(feasible_transforms)
        }

        return attacked_code, attack_info

    except Exception as e:
        # 转换失败，返回原代码
        return code, {'error': str(e), 'feasible_total': 0, 'num_transforms': 0}


def apply_dual_channel_attack(code, seed=42, language='java', augmentor=None):
    """
    应用Dual Channel攻击：50%变量重命名 + 2个代码转换（支持Java和JavaScript）

    Args:
        code: 原始代码
        seed: 随机种子
        language: 语言类型
        augmentor: 代码增强器实例

    Returns:
        (attacked_code, attack_info)
    """
    try:
        # 第一阶段：50%变量重命名
        code_after_rename = apply_rename_attack(code, ratio=0.5, seed=seed, language=language, augmentor=augmentor)

        # 第二阶段：2个代码转换
        code_final, transform_info = apply_t_transform_attack(code_after_rename, n_transforms=2, seed=seed, language=language, augmentor=augmentor)

        # 合并攻击信息
        attack_info = {
            'phase1': 'rename_50%',
            'phase2_transforms': transform_info.get('transforms', []),
            'phase2_num_transforms': transform_info.get('num_transforms', 0)
        }

        return code_final, attack_info

    except Exception as e:
        return code, {'error': str(e)}


def extract_watermark(attacked_code, s0, directions, bitwise_thresholds, model, tokenizer, device):
    """
    提取水印（基于簇中心s0）
    
    完全复刻原始提取逻辑：offset = s_attacked - s0
    
    Args:
        attacked_code: str, 攻击后的代码
        s0: list, 簇中心投影（来自final.json）
        directions: np.ndarray, shape (k, 768)
        bitwise_thresholds: dict, 阈值信息
        model, tokenizer, device: 模型相关
    
    Returns:
        dict, 提取结果
    """
    # 编码并投影攻击后的代码
    attacked_embedding = embed_codes(model, tokenizer, [attacked_code], device=device)[0]  # (768,)
    attacked_projection = attacked_embedding @ directions.T  # (k,)
    
    # ✅ 计算相对于簇中心s0的偏移（关键修复！）
    offset = attacked_projection - np.array(s0)
    
    # ✅ 提取比特（基于offset符号）
    extracted_bits = []
    for i in range(len(offset)):
        if offset[i] > 0:
            extracted_bits.append(1)
        elif offset[i] < 0:
            extracted_bits.append(0)
        else:
            extracted_bits.append(-1)  # 未确定（极少见）
    
    return {
        'extracted_bits': extracted_bits,
        'offset': offset.tolist(),
        'attacked_projection': attacked_projection.tolist(),
        's0': s0
    }


def compute_offset_preservation(offset_before, offset_after):
    """
    计算偏移保持率（基于簇中心）
    
    Args:
        offset_before: np.ndarray, 嵌入后的偏移（相对s0）
        offset_after: np.ndarray, 攻击后的偏移（相对s0）
    
    Returns:
        dict, 包含各种保持率指标
    """
    k = len(offset_before)
    
    # 符号保持
    sign_preserved = sum([
        1 for i in range(k) 
        if np.sign(offset_before[i]) == np.sign(offset_after[i])
    ])
    
    # 幅度保持率（只计算符号保持的）
    magnitude_retention = []
    for i in range(k):
        if np.sign(offset_before[i]) == np.sign(offset_after[i]) and abs(offset_before[i]) > 1e-6:
            retention = abs(offset_after[i]) / abs(offset_before[i])
            magnitude_retention.append(retention)
    
    return {
        'sign_preservation_rate': float(sign_preserved / k),
        'avg_magnitude_retention': float(np.mean(magnitude_retention)) if magnitude_retention else 0.0,
        'per_dimension_sign_preserved': [
            bool(np.sign(offset_before[i]) == np.sign(offset_after[i]))  # 转换为Python bool
            for i in range(k)
        ]
    }


def process_one_run(task):
    """
    处理单个run的所有攻击和提取（子进程）
    
    Args:
        task: (run_id, strategy_name, base_config)
    
    Returns:
        dict: 处理结果
    """
    run_id, strategy_name, base_config = task
    
    embedding_dir = f"dimension_strategy_comparison/results/{strategy_name}/embedding/run_{run_id:04d}"
    
    # 识别语言并确定文件扩展名
    dataset_name = base_config['dataset']['name']
    if 'js' in dataset_name.lower():
        language = 'javascript'
        file_extension = 'js'
    elif 'cpp' in dataset_name.lower() or 'c++' in dataset_name.lower():
        language = 'cpp'
        file_extension = 'cpp'
    else:
        language = 'java'
        file_extension = 'java'

    # 检查是否嵌入成功
    if not os.path.exists(f"{embedding_dir}/watermarked.{file_extension}"):
        return {"run_id": run_id, "success": False, "error": "没有水印代码", "count": 0}

    try:
        with open(f"{embedding_dir}/watermarked.{file_extension}", 'r', encoding='utf-8') as f:
            watermarked_code = f.read()

        with open(f"{embedding_dir}/original.{file_extension}", 'r', encoding='utf-8') as f:
            original_code = f.read()
        
        with open(f"{embedding_dir}/final.json", 'r', encoding='utf-8') as f:
            embedding_result = json.load(f)
        
        # 加载方向和阈值
        with open(f"{embedding_dir}/selected_dimensions.json", 'r', encoding='utf-8') as f:
            dim_data = json.load(f)
        
        directions = np.array(dim_data['directions'])
        bitwise_thresholds = embedding_result['bitwise_thresholds']
        true_bits = base_config['embedding']['bits']
        
        # ✅ 获取簇中心s0和嵌入后的偏移
        s0 = embedding_result['s0']
        s_after = np.array(embedding_result['s_after'])
        offset_before = s_after - np.array(s0)  # 相对于簇中心
        
        # 在子进程中加载模型
        model, tokenizer = load_best_model(base_config['model_dir'])
        device = next(model.parameters()).device
        
        # 识别语言并初始化augmentor
        dataset_name = base_config['dataset']['name']
        if 'js' in dataset_name.lower():
            language = 'javascript'
        elif 'cpp' in dataset_name.lower() or 'c++' in dataset_name.lower():
            language = 'cpp'
        else:
            language = 'java'
        augmentor, _ = get_augmentor_and_renamer(language)
        
        extraction_count = 0

        # 定义攻击类型列表（包括新增的T Transform和Dual Channel）
        attack_types = [
            ('rename', 0.0),        # 无攻击（基准）
            ('rename', 0.25),       # 25%重命名
            ('rename', 0.50),       # 50%重命名
            ('rename', 0.75),       # 75%重命名
            ('rename', 1.0),        # 100%重命名
            ('t_transform', 1),     # T@1 - 1个转换
            ('t_transform', 2),     # T@2 - 2个转换
            ('t_transform', 3),     # T@3 - 3个转换
            ('dual_channel', None), # DualCh - 50%重命名+2个转换
        ]

        # 对每个攻击类型
        for attack_name, attack_param in attack_types:
            # 对每个随机种子
            for seed in range(base_config['extraction']['num_seeds_per_ratio']):
                # 应用相应的攻击
                if attack_name == 'rename':
                    attacked_code = apply_rename_attack(watermarked_code, attack_param, seed, language, augmentor)
                    attack_metadata = None
                elif attack_name == 't_transform':
                    attacked_code, attack_info = apply_t_transform_attack(watermarked_code, attack_param, seed, language, augmentor)
                    attack_metadata = attack_info
                elif attack_name == 'dual_channel':
                    attacked_code, attack_info = apply_dual_channel_attack(watermarked_code, seed, language, augmentor)
                    attack_metadata = attack_info
                else:
                    attacked_code = watermarked_code
                    attack_metadata = None

                # 提取水印
                extraction_result = extract_watermark(
                    attacked_code, s0,  # ✅ 传入簇中心s0
                    directions, bitwise_thresholds,
                    model, tokenizer, device
                )

                extracted_bits = extraction_result['extracted_bits']
                offset_after = np.array(extraction_result['offset'])  # 已经是相对s0的偏移

                # 计算成功率
                success = bool(np.all(extracted_bits == true_bits))  # 转换为Python bool
                bit_accuracy = sum([
                    1 for i in range(len(true_bits))
                    if extracted_bits[i] == true_bits[i]
                ]) / len(true_bits)

                # 计算偏移保持率（相对s0）
                offset_preservation = compute_offset_preservation(offset_before, offset_after)

                # 计算CodeBLEU（特殊处理rename_0.00）
                # rename_0.00作为基准，与原始代码比较，衡量水印侵入度
                # 其他攻击与水印代码比较，衡量攻击破坏力
                try:
                    if attack_name == 'rename' and attack_param == 0.0:
                        # ✅ rename_0.00：无任何攻击，与原始代码比较，衡量水印对代码的改动
                        reference_code = original_code
                        hypothesis_code = attacked_code  # attacked_code == watermarked_code（未被攻击）
                    else:
                        # ✅ 其他攻击：与水印代码比较，衡量攻击对代码的改动
                        reference_code = watermarked_code
                        hypothesis_code = attacked_code
                    
                    codebleu_result = evaluate_per_example(
                        reference=reference_code,
                        hypothesis=hypothesis_code,
                        lang=language
                    )
                    codebleu_score = codebleu_result['codebleu']
                except Exception:
                    codebleu_score = None

                # 生成输出目录和文件名
                if attack_name == 'rename':
                    attack_type_str = f"rename_{attack_param:.2f}"
                elif attack_name == 't_transform':
                    attack_type_str = f"t_transform_{attack_param}"
                elif attack_name == 'dual_channel':
                    attack_type_str = "dual_channel"
                else:
                    attack_type_str = "unknown"

                # 保存结果
                output_dir = f"dimension_strategy_comparison/results/{strategy_name}/extraction/{attack_type_str}"
                os.makedirs(output_dir, exist_ok=True)
                output_file = f"{output_dir}/run_{run_id:04d}_seed_{seed:03d}.json"

                # 构建结果字典
                result_dict = {
                    'run_id': run_id,
                    'attack_type': attack_type_str,
                    'attack_name': attack_name,
                    'attack_param': attack_param if attack_param is not None else None,
                    'seed': seed,
                    'true_bits': [int(b) for b in true_bits],
                    'extracted_bits': [int(b) for b in extracted_bits],
                    'success': success,
                    'bit_accuracy': float(bit_accuracy),
                    'codebleu': float(codebleu_score) if codebleu_score is not None else None,
                    'offset_preservation': offset_preservation,
                    'offset_before': offset_before.tolist(),
                    'offset_after': offset_after.tolist()
                }

                # 添加攻击元数据
                if attack_metadata is not None:
                    result_dict['attack_metadata'] = attack_metadata

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, indent=2)

                extraction_count += 1
        
        return {"run_id": run_id, "success": True, "error": None, "count": extraction_count}
    
    except Exception as e:
        return {"run_id": run_id, "success": False, "error": str(e), "count": 0}


def process_strategy(strategy_name, base_config, concurrency, resume=False):
    """处理单个策略（支持并发和断点继续）"""
    print(f"\n处理策略: {strategy_name}")

    # 创建提取输出目录（包括所有攻击类型）
    attack_types = [
        ('rename', 0.0),
        ('rename', 0.4),
        ('rename', 0.75),
        ('rename', 1.0),
        ('t_transform', 1),
        ('t_transform', 2),
        ('t_transform', 3),
        ('dual_channel', None),
    ]
    for attack_name, attack_param in attack_types:
        if attack_name == 'rename':
            attack_dir = f"dimension_strategy_comparison/results/{strategy_name}/extraction/rename_{attack_param:.2f}"
        elif attack_name == 't_transform':
            attack_dir = f"dimension_strategy_comparison/results/{strategy_name}/extraction/t_transform_{attack_param}"
        elif attack_name == 'dual_channel':
            attack_dir = f"dimension_strategy_comparison/results/{strategy_name}/extraction/dual_channel"
        os.makedirs(attack_dir, exist_ok=True)

    # 准备任务列表
    tasks = []
    skipped_count = 0
    for run_id in range(base_config['num_test_codes']):
        # 如果启用resume，检查是否存在已完成标记
        if resume:
            # 简化resume逻辑：检查至少一个rename攻击结果是否存在
            sample_output = f"dimension_strategy_comparison/results/{strategy_name}/extraction/rename_0.00/run_{run_id:04d}_seed_000.json"
            if os.path.exists(sample_output):
                skipped_count += 1
                continue
        tasks.append((run_id, strategy_name, base_config))
    
    if resume and skipped_count > 0:
        print(f"  跳过已完成: {skipped_count} 个")
    
    if not tasks:
        print(f"  所有任务已完成，无需处理")
        return
    
    # 并发或串行执行
    if concurrency > 1:
        print(f"  使用并发模式 (max_workers={concurrency})")
        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(process_one_run, task) for task in tasks]
            
            # 使用tqdm显示进度
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  提取进度", ncols=80):
                result = future.result()
                results.append(result)
                if not result['success']:
                    print(f"\n[错误] run_{result['run_id']:04d} 提取失败: {result['error']}")
    else:
        print(f"  使用串行模式")
        results = []
        for task in tqdm(tasks, desc=f"  提取进度", ncols=80):
            result = process_one_run(task)
            results.append(result)
            if not result['success']:
                print(f"\n[错误] run_{result['run_id']:04d} 提取失败: {result['error']}")
    
    # 统计结果
    success_count = sum(1 for r in results if r['success'])
    total_extractions = sum(r['count'] for r in results)
    print(f"  完成: {success_count}/{len(results)} 成功, 共 {total_extractions} 次提取")
    
    # ==================== 新增：计算单个水印提取时间 ====================
    try:
        import time
        from Watermark4code.encoder.loader import load_best_model, embed_codes
        
        print("\n  测试单个水印提取时间...")
        
        # 加载第一个水印代码
        dataset_name = base_config['dataset']['name']
        if 'js' in dataset_name.lower():
            language = 'javascript'
            file_extension = 'js'
        elif 'cpp' in dataset_name.lower() or 'c++' in dataset_name.lower():
            language = 'cpp'
            file_extension = 'cpp'
        else:
            language = 'java'
            file_extension = 'java'
        
        embedding_dir = f"dimension_strategy_comparison/results/{strategy_name}/embedding/run_0000"
        watermarked_file = f"{embedding_dir}/watermarked.{file_extension}"
        
        if os.path.exists(watermarked_file):
            with open(watermarked_file, 'r', encoding='utf-8') as f:
                watermarked_code = f.read()
            
            with open(f"{embedding_dir}/final.json", 'r', encoding='utf-8') as f:
                embedding_result = json.load(f)
            
            with open(f"{embedding_dir}/selected_dimensions.json", 'r', encoding='utf-8') as f:
                dim_data = json.load(f)
            
            # 加载模型
            model, tokenizer = load_best_model(base_config['model_dir'])
            device = next(model.parameters()).device
            
            # 准备数据
            directions = np.array(dim_data['directions'])
            bitwise_thresholds = embedding_result['bitwise_thresholds']
            s0 = embedding_result['s0']
            
            # 测量所有水印的提取时间
            total_time = 0
            num_watermarks = base_config['num_test_codes']
            
            for run_id in range(num_watermarks):
                embedding_dir_i = f"dimension_strategy_comparison/results/{strategy_name}/embedding/run_{run_id:04d}"
                watermarked_file_i = f"{embedding_dir_i}/watermarked.{file_extension}"
                final_file_i = f"{embedding_dir_i}/final.json"
                
                if os.path.exists(watermarked_file_i) and os.path.exists(final_file_i):
                    with open(watermarked_file_i, 'r', encoding='utf-8') as f:
                        code = f.read()
                    with open(final_file_i, 'r', encoding='utf-8') as f:
                        res = json.load(f)
                    
                    s0_i = res['s0']
                    
                    # 计时提取
                    start = time.perf_counter()
                    extraction_result = extract_watermark(
                        code, s0_i, directions, bitwise_thresholds,
                        model, tokenizer, device
                    )
                    end = time.perf_counter()
                    total_time += (end - start) * 1000  # 转换为毫秒
            
            avg_extraction_time = total_time / num_watermarks
            print(f"  单个水印提取时间: {avg_extraction_time:.2f} ms")
            
            # 保存到文件
            timing_file = f"dimension_strategy_comparison/results/{strategy_name}/extraction_timing.json"
            with open(timing_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'strategy': strategy_name,
                    'extraction_time_ms': round(avg_extraction_time, 2),
                    'num_samples': num_watermarks
                }, f, indent=2)
    
    except Exception as e:
        print(f"  [警告] 计时失败: {e}")


def main():
    global PARSER_LANG, ts_parser

    parser = argparse.ArgumentParser(description="Step 4: 攻击并提取水印（支持并发）")
    parser.add_argument('--concurrency', type=int, default=5, help='并发处理的进程数（默认=5）')
    parser.add_argument('--resume', action='store_true', help='断点继续：跳过已有extraction_result.json的任务')
    parser.add_argument('--strategy', type=str, default=None, help='指定处理的策略（如果不指定，处理所有策略）')
    parser.add_argument('--config', type=str, 
                       default=None,
                       help='配置文件路径（相对于项目根目录）')
    args = parser.parse_args()

    print("="*80)
    print(f"Step 4: 攻击并提取水印 (concurrency={args.concurrency}, resume={args.resume})")
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
    
    # ✅ 识别语言和bit版本
    dataset_name = base_config['dataset']['name']
    if 'js' in dataset_name.lower():
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
    
    print(f"检测到语言: {language}")
    print(f"数据集: {dataset_name}")
    print(f"投影维度: {k}bit")

    # 初始化tree-sitter解析器（根据语言选择）
    try:
        lang_so = project_root / "SrcMarker-main" / "parser" / "languages.so"
        if lang_so.exists():
            PARSER_LANG = tree_sitter.Language(str(lang_so), language)
            ts_parser = tree_sitter.Parser()
            ts_parser.set_language(PARSER_LANG)
            print(f"Tree-sitter {language}解析器初始化成功")
        else:
            print(f"[警告] languages.so 不存在，T Transform攻击可能无法使用")
    except Exception as e:
        print(f"[警告] Tree-sitter初始化失败: {e}")

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
    print("完成！提取结果已保存到 results/strategy_X/extraction/")
    print("="*80)


if __name__ == '__main__':
    os.chdir(Path(__file__).parent.parent)
    main()
