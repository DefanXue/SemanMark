#!/usr/bin/env python3
"""
Strategy 6 完整测试脚本
从Step 1到Step 5运行完整的水印嵌入和提取流程

支持功能：
- --resume: 断点继续（对有此功能的步骤）
- --concurrency: 并发处理进程数
- --test-mode: 快速测试模式（用少量数据）
- --skip-steps: 跳过某些步骤
- --verbose: 详细输出
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 设置项目路径
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "dimension_strategy_comparison"))


class TestPipeline:
    """Strategy 6完整测试流程管理器"""

    def __init__(self, args):
        self.args = args
        self.project_root = project_root
        self.scripts_dir = project_root / "dimension_strategy_comparison" / "scripts"
        self.configs_dir = project_root / "dimension_strategy_comparison" / "configs"
        
        # 先加载配置以确定语言（用于设置结果目录）
        self.load_configs()
        
        # ✅ 根据 bits 长度和语言确定结果目录（区分 bit 版本和语言）
        dataset_name = self.base_config.get('dataset', {}).get('name', 'csn-java')
        k = len(self.base_config['embedding']['bits'])
        
        # 确定语言后缀
        if 'js' in dataset_name.lower() or 'javascript' in dataset_name.lower():
            self.language = 'javascript'
            lang_suffix = '_js'
        elif 'cpp' in dataset_name.lower() or 'c++' in dataset_name.lower():
            self.language = 'cpp'
            lang_suffix = '_cpp'
        else:
            self.language = 'java'
            lang_suffix = ''
        
        # ✅ 确定 bit 版本后缀（2bit、6bit、8bit）
        if k == 2:
            bit_suffix = '_2bit'
        elif k == 6:
            bit_suffix = '_6bit'
        elif k == 8:
            bit_suffix = '_8bit'
        else:  # 默认 4bit，无后缀
            bit_suffix = ''
        
        # ✅ 结合两个后缀生成完整的策略名称
        strategy_base = f"strategy_6_adaptive{bit_suffix}"
        self.results_dir = project_root / "dimension_strategy_comparison" / "results" / f"{strategy_base}{lang_suffix}"
        
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 测试报告
        self.report = {
            'start_time': datetime.now().isoformat(),
            'steps': {},
            'metrics': {},
            'errors': [],
            'summary': {}
        }

        # 设置编码以支持Windows
        os.environ['PYTHONIOENCODING'] = 'utf-8'

        # 时间统计
        self.step_times = {}

    def load_configs(self):
        """加载配置文件"""
        # 使用命令行指定的配置文件，或默认配置
        if hasattr(self.args, 'config') and self.args.config:
            base_config_path = Path(self.args.config)
            if not base_config_path.is_absolute():
                base_config_path = self.project_root / "dimension_strategy_comparison" / base_config_path
        else:
            base_config_path = self.configs_dir / "base_config.json"
        
        # ✅ 先加载 base_config 以获取 bits 长度
        with open(base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = json.load(f)
        
        # ✅ 根据 bits 长度动态选择 strategy config
        k = len(self.base_config['embedding']['bits'])
        if k == 2:
            strategy_config_filename = "strategy_6_adaptive_2bit.json"
        elif k == 6:
            strategy_config_filename = "strategy_6_adaptive_6bit.json"
        elif k == 8:
            strategy_config_filename = "strategy_6_adaptive_8bit.json"
        else:  # 默认 4bit
            strategy_config_filename = "strategy_6_adaptive.json"
        
        strategy_config_path = self.configs_dir / strategy_config_filename

        with open(strategy_config_path, 'r', encoding='utf-8') as f:
            self.strategy_config = json.load(f)

        if self.args.verbose or True:  # 总是显示配置信息
            print(f"[OK] 配置加载成功")
            print(f"  - Base config: {base_config_path}")
            print(f"  - Strategy config: {strategy_config_path}")
            print(f"  - Language: {getattr(self, 'language', 'detecting...')}")
            print(f"  - Results dir: {getattr(self, 'results_dir', 'detecting...')}")

    def run_step(self, step_num, script_name, step_name, extra_args=None):
        """运行单个步骤"""
        # 对 Step 1d 进行标签处理
        step_label = 'step1d' if script_name == 'step1d_generate_fixed_directions.py' else f"step{step_num}"
        
        if step_label in self.args.skip_steps:
            print(f"\n[SKIP] 跳过 {step_label}: {step_name}")
            return True

        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            error_msg = f"脚本不存在: {script_path}"
            self.report['errors'].append(f"{step_label}: {error_msg}")
            print(f"[FAIL] {step_label} ({step_name}): {error_msg}")
            return False

        print(f"\n{'='*80}")
        print(f"{step_label}: {step_name}")
        print(f"{'='*80}")

        # 构建命令
        cmd = [sys.executable, str(script_path)]

        # 添加 --config 参数（如果指定）
        if hasattr(self.args, 'config') and self.args.config:
            config_arg = self.args.config
            
            # 根据步骤调整配置路径格式（因为各步骤的工作目录不同）
            # 项目根目录：step1c(1), step1d(1), step3(3), step4(4) → 需要完整路径
            # dimension_strategy_comparison目录：step2(2), step5(5) → 需要相对路径
            if step_num in [1, 3, 4]:  # 切换到项目根目录
                # 需要完整路径: dimension_strategy_comparison/configs/...
                if not config_arg.startswith('dimension_strategy_comparison/') and not os.path.isabs(config_arg):
                    config_arg = f"dimension_strategy_comparison/{config_arg}"
            else:  # step2, step5 切换到 dimension_strategy_comparison 目录
                # 只需要: configs/...
                if config_arg.startswith('dimension_strategy_comparison/'):
                    config_arg = config_arg.replace('dimension_strategy_comparison/', '', 1)
            
            cmd.extend(['--config', config_arg])

        # 添加通用参数
        if extra_args:
            cmd.extend(extra_args)

        # 根据步骤添加特定参数
        # Step 1c 支持 concurrency，但 Step 1d 不需要（太快了）
        if step_num == 1 and script_name == 'step1c_train_adaptive_generator.py':
            cmd.extend(['--concurrency', str(self.args.concurrency)])
        
        if step_num in [3, 4]:  # step3, step4支持concurrency
            cmd.extend(['--concurrency', str(self.args.concurrency)])

        # Step 1d 不需要 --resume（太快了），Step 1c/2/3/4 支持 --resume
        if step_num in [1, 2, 3, 4] and script_name != 'step1d_generate_fixed_directions.py':
            if self.args.resume:
                cmd.append('--resume')

        if step_num in [2, 3, 4, 5]:  # step2, step3, step4, step5仅处理strategy
            # ✅ 根据配置、语言和 bit 版本设置策略名称
            lang_suffix = '_js' if self.language == 'javascript' else ('_cpp' if self.language == 'cpp' else '')
            k = len(self.base_config['embedding']['bits'])
            bit_suffix = '_2bit' if k == 2 else ('_6bit' if k == 6 else ('_8bit' if k == 8 else ''))
            
            if not self.base_config.get('use_contrastive_learning', True):
                # Variant A: w/o CL
                strategy_arg = f'strategy_6_no_cl{bit_suffix}{lang_suffix}'
            elif not self.base_config.get('use_adaptive_directions', True):
                # Variant B: w/o ADG
                strategy_arg = f'strategy_6_no_adg{bit_suffix}{lang_suffix}'
            else:
                # Baseline
                strategy_arg = f'strategy_6_adaptive{bit_suffix}{lang_suffix}'
            cmd.extend(['--strategy', strategy_arg])

        if step_num == 5:  # step5指定输出目录
            # ✅ 根据配置、语言和 bit 版本设置输出目录
            lang_suffix = '_js' if self.language == 'javascript' else ('_cpp' if self.language == 'cpp' else '')
            k = len(self.base_config['embedding']['bits'])
            bit_suffix = '_2bit' if k == 2 else ('_6bit' if k == 6 else ('_8bit' if k == 8 else ''))
            
            if not self.base_config.get('use_contrastive_learning', True):
                output_subdir = f'analysis/strategy_6_no_cl{bit_suffix}{lang_suffix}'
            elif not self.base_config.get('use_adaptive_directions', True):
                output_subdir = f'analysis/strategy_6_no_adg{bit_suffix}{lang_suffix}'
            else:
                output_subdir = f'analysis/strategy_6_adaptive{bit_suffix}{lang_suffix}'
            cmd.extend(['--output-dir', output_subdir])

        if self.args.verbose:
            print(f"执行命令: {' '.join(cmd)}")

        # 执行步骤
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                timeout=43200,  # 6小时超时
                encoding='utf-8',
                errors='replace'
            )

            elapsed_time = time.time() - start_time
            self.step_times[f"step{step_num}"] = elapsed_time

            if result.returncode == 0:
                print(f"[OK] Step {step_num} 完成 (耗时: {elapsed_time:.1f}s)")
                self.report['steps'][f"step{step_num}"] = {
                    'name': step_name,
                    'status': 'success',
                    'duration': elapsed_time
                }
                return True
            else:
                error_msg = f"返回码: {result.returncode}"
                self.report['errors'].append(f"Step {step_num}: {error_msg}")
                print(f"[FAIL] Step {step_num} 失败: {error_msg}")
                self.report['steps'][f"step{step_num}"] = {
                    'name': step_name,
                    'status': 'failed',
                    'duration': elapsed_time,
                    'error': error_msg
                }
                return False

        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            error_msg = "执行超时"
            self.report['errors'].append(f"Step {step_num}: {error_msg}")
            print(f"[FAIL] Step {step_num} 超时 (> 3600s)")
            return False

        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = str(e)
            self.report['errors'].append(f"Step {step_num}: {error_msg}")
            print(f"[FAIL] Step {step_num} 异常: {error_msg}")
            return False

    def verify_outputs(self):
        """验证各步骤的输出文件"""
        print(f"\n{'='*80}")
        print("验证输出文件")
        print(f"{'='*80}")

        verification = {
            'step1c': ['trained_generator.pth'],
            'step2': ['selected_dimensions.json'],
            'step3': ['embedded_watermarks.json'],
            'step4': ['extraction_results.json'],
            'step5': ['analysis_report.json']
        }

        all_valid = True
        for step, files in verification.items():
            if step in self.args.skip_steps:
                continue

            for file in files:
                file_path = self.results_dir / file

                if file_path.exists():
                    print(f"  [OK] {step}/{file}")
                else:
                    print(f"  [FAIL] {step}/{file} (不存在)")
                    all_valid = False

        return all_valid

    def collect_metrics(self):
        """收集性能指标"""
        print(f"\n{'='*80}")
        print("性能指标")
        print(f"{'='*80}")

        metrics = {
            'total_time': sum(self.step_times.values()),
            'step_times': self.step_times,
            'training_codes': self.base_config.get('data_split', {}).get('strategy_5_training', {}).get('num_codes', 200),
            'test_codes': self.base_config.get('data_split', {}).get('embedding_and_testing', {}).get('num_codes', 30),
        }

        # 尝试加载训练日志
        training_log_path = self.results_dir / "training_log.json"
        if training_log_path.exists():
            with open(training_log_path, 'r', encoding='utf-8') as f:
                training_log = json.load(f)
                if 'loss_history' in training_log and 'total' in training_log['loss_history']:
                    metrics['final_loss'] = training_log['loss_history']['total'][-1]

        # 尝试加载提取结果
        extraction_results_path = self.results_dir / "extraction_results.json"
        if extraction_results_path.exists():
            with open(extraction_results_path, 'r', encoding='utf-8') as f:
                extraction_results = json.load(f)
                if 'overall_accuracy' in extraction_results:
                    metrics['extraction_accuracy'] = extraction_results['overall_accuracy']

        self.report['metrics'] = metrics

        # 打印指标
        print(f"\n训练配置:")
        print(f"  - 训练代码数: {metrics['training_codes']}")
        print(f"  - 测试代码数: {metrics['test_codes']}")
        print(f"  - 每代码变体数: 100")

        print(f"\n执行时间:")
        for step, elapsed in sorted(self.step_times.items()):
            print(f"  - {step}: {elapsed:.1f}s")
        print(f"  - 总耗时: {metrics['total_time']:.1f}s ({metrics['total_time']/60:.1f} min)")

        if 'final_loss' in metrics:
            print(f"\n训练结果:")
            print(f"  - 最终损失: {metrics['final_loss']:.6f}")

        if 'extraction_accuracy' in metrics:
            print(f"\n水印提取:")
            print(f"  - 提取准确率: {metrics['extraction_accuracy']:.2%}")

    def generate_report(self):
        """生成最终报告"""
        print(f"\n{'='*80}")
        print("测试报告生成")
        print(f"{'='*80}")

        self.report['end_time'] = datetime.now().isoformat()

        # 确定整体状态
        failed_steps = [s for s, info in self.report['steps'].items() if info['status'] == 'failed']
        self.report['summary']['status'] = 'failed' if failed_steps else 'success'
        self.report['summary']['failed_steps'] = failed_steps
        self.report['summary']['error_count'] = len(self.report['errors'])

        # 保存报告
        report_path = self.results_dir / "test_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] 测试报告已保存: {report_path}")

        # 打印摘要
        print(f"\n{'='*80}")
        if self.report['summary']['status'] == 'success':
            print("[SUCCESS] 所有测试通过！")
        else:
            print(f"[WARN] 部分步骤失败: {', '.join(failed_steps)}")

        if self.report['errors']:
            print(f"\n错误列表 ({len(self.report['errors'])}):")
            for error in self.report['errors']:
                print(f"  - {error}")

        print(f"{'='*80}\n")

    def run(self):
        """执行完整测试流程"""
        print(f"\n{'#'*80}")
        
        # 检测是否是消融实验
        if not self.base_config.get('use_contrastive_learning', True):
            print(f"# Variant A: w/o Contrastive Learning (原始编码器) - 完整测试流程")
        elif not self.base_config.get('use_adaptive_directions', True):
            print(f"# Variant B: w/o ADG (固定随机方向) - 完整测试流程")
        else:
            print(f"# Baseline: Strategy 6 完整测试流程")
        
        print(f"# 开始时间: {self.report['start_time']}")
        print(f"{'#'*80}\n")

        # 运行各个步骤
        # 根据配置选择 Step 1c (自适应) 或 Step 1d (固定随机)
        if not self.base_config.get('use_adaptive_directions', True):
            # Variant B: w/o ADG
            steps = [
                (1, 'step1d_generate_fixed_directions.py', '生成固定随机方向'),
                (2, 'step2_select_dimensions.py', '选择维度'),
                (3, 'step3_embed_watermarks.py', '嵌入水印'),
                (4, 'step4_extract_with_attacks.py', '攻击提取'),
                (5, 'step5_analyze_results.py', '分析结果')
            ]
        else:
            # Baseline: Strategy 6 with ADG
            steps = [
                (1, 'step1c_train_adaptive_generator.py', '训练自适应方向生成器'),
                (2, 'step2_select_dimensions.py', '选择维度'),
                (3, 'step3_embed_watermarks.py', '嵌入水印'),
                (4, 'step4_extract_with_attacks.py', '攻击提取'),
                (5, 'step5_analyze_results.py', '分析结果')
            ]

        all_success = True
        for step_num, script_name, step_name in steps:
            # 构建step标签
            if step_num == 0:
                step_label = 'step1'
            elif 'step1b' in script_name:
                step_label = 'step1b'
            elif 'step1c' in script_name:
                step_label = 'step1c'
            else:
                step_label = f'step{step_num}'

            if step_label in self.args.skip_steps:
                continue

            success = self.run_step(step_num, script_name, step_name)
            if not success and not self.args.continue_on_error:
                print(f"\n[WARN] Step {step_num} 失败，停止执行")
                all_success = False
                break

        # 验证输出
        verify_success = self.verify_outputs()

        # 收集指标
        self.collect_metrics()

        # 生成报告
        self.generate_report()

        return all_success and verify_success


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Strategy 6 完整测试流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整测试（Java，默认）
  python test_strategy6_pipeline.py

  # JavaScript水印测试
  python test_strategy6_pipeline.py --config configs/base_config_js.json

  # 支持断点续传
  python test_strategy6_pipeline.py --config configs/base_config_js.json --resume

  # 跳过某些步骤
  python test_strategy6_pipeline.py --skip-steps step1,step2

  # 并发处理
  python test_strategy6_pipeline.py --config configs/base_config_js.json --concurrency 10

  # 详细输出
  python test_strategy6_pipeline.py --verbose
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（相对于项目根目录，如：configs/base_config_js.json）'
    )

    parser.add_argument(
        '--concurrency',
        type=int,
        default=5,
        help='并发处理的进程数（默认=5，对step1c/3/4有效）'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='断点继续：跳过已有结果的任务（对step3/4有效）'
    )

    parser.add_argument(
        '--skip-steps',
        type=str,
        default='',
        help='跳过的步骤，用逗号分隔（如：step1,step2）'
    )

    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='快速测试模式（用少量数据）'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )

    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='错误时继续执行（默认遇到错误停止）'
    )

    args = parser.parse_args()

    # 解析skip-steps
    args.skip_steps = set(s.strip() for s in args.skip_steps.split(',') if s.strip())

    return args


def main():
    """主入口函数"""
    args = parse_arguments()

    # 创建测试管道
    pipeline = TestPipeline(args)

    # 运行测试
    success = pipeline.run()

    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
