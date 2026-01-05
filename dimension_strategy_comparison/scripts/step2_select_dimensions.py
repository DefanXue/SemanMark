"""
Step 2: 根据策略选择维度并生成正交方向

输入：data/dimension_analysis/run_XXXX_analysis.json
输出：results/strategy_X/embedding/run_XXXX/selected_dimensions.json
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Watermark4code.keys.directions import derive_directions

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def construct_and_orthogonalize(selected_dims, d=768, guidance_strength=5.0):
    """
    将选择的维度索引作为引导，生成稠密的、正交的方向矩阵。
    """
    k = len(selected_dims)
    
    # 1. 生成 k 个随机的稠密向量。
    #    为了让这个过程可复现，我们可以用 selected_dims 的和作为随机种子。
    seed = sum(selected_dims)
    rng = np.random.RandomState(seed)
    random_vectors = rng.uniform(low=-1.0, high=1.0, size=(k, d))

    # 2. 【核心修改】对每个向量，放大其对应“最佳”维度的权重。
    guided_vectors = []
    for i in range(k):
        vec = random_vectors[i].copy()
        selected_dim_index = selected_dims[i]
        
        # 将被选中的维度的系数乘以一个增强因子。
        vec[selected_dim_index] *= guidance_strength
        
        guided_vectors.append(vec)
        
    guided_vectors = np.array(guided_vectors)

    # 3. 对这些“被引导”的稠密向量进行Gram-Schmidt正交化。
    ortho_vectors = []
    for i in range(k):
        v = guided_vectors[i].copy()
        
        # 减去在已正交化向量上的投影
        for u in ortho_vectors:
            v -= np.dot(v, u) * u
        
        # 归一化
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            v = v / norm
        else:
            # 如果向量几乎为0（高度线性相关），使用随机向量
            print(f"    [警告] 维度 {selected_dims[i]} 与已选维度高度相关，使用随机向量")
            v = np.random.randn(d)
            for u in ortho_vectors:
                v -= np.dot(v, u) * u
            v = v / np.linalg.norm(v)
        
        ortho_vectors.append(v)
    
    return np.array(ortho_vectors)  # (k, d)


def select_top_k_by_composite_score(analysis, k=4, weights=None):
    """
    按综合得分选择Top-k维度
    
    Args:
        analysis: dict, 包含 dimension_scores
        k: int, 选择维度数量
        weights: dict, 权重
    
    Returns:
        list of int, 选中的维度索引
    """
    if weights is None:
        weights = {
            'triple_order_preservation': 0.5,
            'overall_sign_preservation': 0.4,
            'left_vs_right_preservation': 0.1
        }
    
    scores = []
    for item in analysis['dimension_scores']:
        composite = (
            item['triple_order_preservation'] * weights['triple_order_preservation'] +
            item['overall_sign_preservation'] * weights['overall_sign_preservation'] +
            item['left_vs_right_preservation'] * weights['left_vs_right_preservation']
        )
        scores.append((item['dimension'], composite))
    
    # 按得分降序排序
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return [dim for dim, score in scores[:k]]


def select_top_k_by_single_metric(analysis, metric, k=4, weights=None):
    """
    按单一指标或组合指标选择Top-k维度
    
    Args:
        analysis: dict
        metric: str, 指标名称或'composite_score'
        k: int
        weights: dict, 当metric='composite_score'时使用的权重
    
    Returns:
        list of int
    """
    scores = []
    for item in analysis['dimension_scores']:
        if metric == 'composite_score' and weights:
            # 计算加权综合得分
            composite = 0.0
            for metric_name, weight in weights.items():
                if metric_name in item:
                    composite += item[metric_name] * weight
            scores.append((item['dimension'], composite))
        else:
            # 单一指标
            scores.append((item['dimension'], item[metric]))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return [dim for dim, score in scores[:k]]


def greedy_select_with_orthogonality(analysis, k=4, max_corr=0.90, fallback_thresholds=None):
    """
    贪心选择：优先鲁棒性，但避免高度相关的维度
    
    Args:
        analysis: dict
        k: int
        max_corr: float, 最大相关性阈值
        fallback_thresholds: list of float, 回退阈值
    
    Returns:
        list of int
    """
    if fallback_thresholds is None:
        fallback_thresholds = [0.93, 0.95, 0.98]
    
    # 按综合得分排序
    scores = []
    for item in analysis['dimension_scores']:
        composite = (
            item['triple_order_preservation'] * 0.5 +
            item['overall_sign_preservation'] * 0.4 +
            item['left_vs_right_preservation'] * 0.1
        )
        scores.append((item['dimension'], composite))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    correlation_matrix = np.array(analysis['correlation_matrix'])
    
    # 尝试不同的阈值
    thresholds_to_try = [max_corr] + fallback_thresholds
    
    for threshold in thresholds_to_try:
        selected = []
        
        for dim, score in scores:
            if len(selected) >= k:
                break
            
            # 检查与已选维度的相关性
            if len(selected) == 0:
                selected.append(dim)
            else:
                max_corr_with_selected = max([abs(correlation_matrix[dim, s]) for s in selected])
                if max_corr_with_selected < threshold:
                    selected.append(dim)
        
        if len(selected) >= k:
            print(f"    使用阈值 {threshold:.2f} 成功选出 {k} 个维度")
            return selected[:k]
    
    # 如果仍然失败，直接返回Top-k
    print(f"    [警告] 即使使用阈值 {thresholds_to_try[-1]:.2f} 也无法选出 {k} 个维度，返回综合得分Top-{k}")
    return [dim for dim, score in scores[:k]]


def process_strategy(strategy_name, strategy_config, base_config, resume=False):
    """处理单个策略"""
    print(f"\n处理策略: {strategy_name}")
    print(f"  描述: {strategy_config['description']}")
    
    # 创建输出目录
    output_dir = f"results/{strategy_name}/embedding"
    os.makedirs(output_dir, exist_ok=True)
    
    # 为策略6预加载所有代码（如果需要）
    codes_by_id = {}
    if strategy_config['dimension_selection']['method'] == "adaptive_learned":
        split_config = base_config.get('data_split', {}).get('embedding_and_testing', {})
        start_idx = split_config.get('start_index', 0)
        num_codes = split_config.get('num_codes', base_config['num_test_codes'])
        sys.path.insert(0, str(Path(__file__).parent))
        from dataset_loader import DatasetLoader
        loader = DatasetLoader(base_config)
        test_codes = loader.load_codes(num_codes, start_idx, split='test')
        codes_by_id = {i: code for i, code in enumerate(test_codes)}

    # 处理每个代码
    skipped = 0
    for run_id in range(base_config['num_test_codes']):
        run_output_dir = f"{output_dir}/run_{run_id:04d}"
        output_file = f"{run_output_dir}/selected_dimensions.json"

        if resume and os.path.exists(output_file):
            print(f"  run_{run_id:04d}: 已存在，跳过 [resume]")
            skipped += 1
            continue

        print(f"  run_{run_id:04d}: ", end='')

        # 根据策略选择维度
        method = strategy_config['dimension_selection']['method']

        # 只有策略1-4需要加载分析结果
        analysis = None
        if method in ["random_from_secret", "top_k_by_metric", "greedy_with_orthogonality_constraint"]:
            analysis_file = f"data/dimension_analysis/run_{run_id:04d}_analysis.json"
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
        
        if method == "random_from_secret":
            # 策略1：随机投影
            selected_dims = None
            directions = derive_directions(
                secret_key=base_config['embedding']['secret'],
                d=strategy_config['direction_generation']['d'],
                k=strategy_config['direction_generation']['k']
            )
            print("随机投影 [OK]")
        
        elif method == "top_k_by_metric":
            # 策略2/3/4：单一或组合指标Top-k
            selected_dims = select_top_k_by_single_metric(
                analysis,
                metric=strategy_config['dimension_selection']['metric'],
                k=strategy_config['dimension_selection']['k'],
                weights=strategy_config['dimension_selection'].get('weights')
            )
            directions = construct_and_orthogonalize(
                selected_dims,
                d=strategy_config['direction_generation']['d']
            )
            print(f"选出维度 {selected_dims} [OK]")
        
        elif method == "greedy_with_orthogonality_constraint":
            # 策略3：贪心平衡选择
            selected_dims = greedy_select_with_orthogonality(
                analysis,
                k=strategy_config['dimension_selection']['k'],
                max_corr=strategy_config['dimension_selection']['max_correlation_threshold'],
                fallback_thresholds=strategy_config['dimension_selection'].get('fallback_thresholds')
            )
            directions = construct_and_orthogonalize(
                selected_dims,
                d=strategy_config['direction_generation']['d']
            )
            print(f"选出维度 {selected_dims} [OK]")
        
        elif method == "learned_directions":
            # 策略5：可学习方向矩阵
            if not TORCH_AVAILABLE:
                raise ImportError("Strategy 5需要PyTorch，但未找到torch模块")

            trained_W_path = f"results/{strategy_name}/trained_W.pth"
            if not os.path.exists(trained_W_path):
                raise FileNotFoundError(
                    f"未找到训练好的W矩阵: {trained_W_path}\n"
                    f"请先运行: python scripts/step1b_train_learned_directions.py"
                )

            # 加载训练好的W矩阵
            W_tensor = torch.load(trained_W_path, map_location='cpu')
            directions = W_tensor.numpy()  # (k, 768)
            selected_dims = None  # learned方向不对应固定维度
            print(f"加载学习到的方向矩阵 {directions.shape} [OK]")

        elif method == "adaptive_learned":
            # 策略6：自适应方向生成器（Per-Code）
            if not TORCH_AVAILABLE:
                raise ImportError("Strategy 6需要PyTorch，但未找到torch模块")

            trained_generator_path = f"results/{strategy_name}/trained_generator.pth"
            if not os.path.exists(trained_generator_path):
                raise FileNotFoundError(
                    f"未找到训练好的自适应生成器: {trained_generator_path}\n"
                    f"请先运行: python scripts/step1c_train_adaptive_generator.py"
                )

            # 加载代码嵌入，用生成器生成自适应方向
            from Watermark4code.encoder.loader import load_best_model, embed_codes
            sys.path.insert(0, str(project_root / "dimension_strategy_comparison"))
            from models.adaptive_direction_generator import AdaptiveDirectionGenerator

            # 加载训练好的生成器
            generator = AdaptiveDirectionGenerator(
                d=strategy_config['direction_generation']['d'],
                k=strategy_config['direction_generation']['k'],
                hidden_dims=strategy_config['learning']['hidden_dims'],
                use_attention=strategy_config['learning']['use_attention']
            )
            generator.load_state_dict(torch.load(trained_generator_path, map_location='cpu'))
            generator.eval()

            # 加载代码并获取嵌入
            model, tokenizer = load_best_model(base_config['model_dir'])
            device = next(model.parameters()).device

            # 为当前代码生成自适应方向
            code = codes_by_id.get(run_id)
            if code is None:
                raise RuntimeError(f"无法加载代码 run_{run_id:04d}")
            code_embedding = embed_codes(model, tokenizer, [code], device=device)[0]  # (768,)
            code_embedding_tensor = torch.from_numpy(code_embedding).float().unsqueeze(0)  # (1, 768)

            with torch.no_grad():
                W, _ = generator(code_embedding_tensor)  # (1, 4, 768)
                directions = W.squeeze(0).cpu().numpy()  # (4, 768)

            selected_dims = None  # 自适应方向不对应固定维度
            print(f"生成自适应方向矩阵 {directions.shape} [OK]")

            # 清理模型
            del model, tokenizer, generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        else:
            raise ValueError(f"未知的选择方法: {method}")
        
        # 保存结果
        os.makedirs(run_output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'run_id': run_id,
                'strategy': strategy_name,
                'method': method,
                'selected_dims': selected_dims if selected_dims is not None else "random",
                'directions': directions.tolist()
            }, f, indent=2)
    if resume and skipped > 0:
        print(f"  跳过已完成: {skipped} 个")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, help='仅处理指定策略')
    parser.add_argument('--resume', action='store_true', help='断点继续：已存在selected_dimensions.json则跳过')
    parser.add_argument('--config', type=str, 
                       default=None,
                       help='配置文件路径（相对于当前目录）')
    args = parser.parse_args()

    print("="*80)
    print("Step 2: 选择维度并生成正交方向")
    print("="*80)

    # 加载基础配置（保持原逻辑不变）
    config_path = args.config if args.config else "configs/base_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        base_config = json.load(f)

    # ✅ 识别语言和bit版本，根据两者设置结果目录后缀
    dataset_name = base_config.get('dataset', {}).get('name', 'csn-java')
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

    # ✅ 处理所有策略（根据bit和语言调整策略名称，config文件也要同步调整）
    strategies = [
        (f"strategy_5_learned{bit_suffix}{lang_suffix}", "configs/strategy_5_learned.json"),
        (f"strategy_6_adaptive{bit_suffix}{lang_suffix}", f"configs/strategy_6_adaptive{bit_suffix}.json"),  # ✅ 加入 bit_suffix
    ]

    # 过滤策略
    if args.strategy:
        strategies = [(name, cfg) for name, cfg in strategies if name == args.strategy or name == args.strategy + lang_suffix]

    for strategy_name, config_file in strategies:
        with open(config_file, 'r', encoding='utf-8') as f:
            strategy_config = json.load(f)

        process_strategy(strategy_name, strategy_config, base_config, resume=args.resume)

    print("\n" + "="*80)
    print("完成！维度选择结果已保存到 results/strategy_X/embedding/")
    print("="*80)


if __name__ == '__main__':
    os.chdir(Path(__file__).parent.parent)
    main()

