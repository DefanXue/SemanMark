#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C++代码增强器，用于生成对比学习的正样本和负样本。
对标JavaCodeAugmentor和JavaScriptCodeAugmentor
使用系统内置的MutableAST实现语义保持转换（完全对标Java和JavaScript）
"""

import random
import re
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# 导入基础增强器
from contrastive_learning.augmentor import CodeAugmentor


class CppCodeAugmentor(CodeAugmentor):
    """
    针对C++代码的增强器，继承自通用CodeAugmentor
    实现语义保持转换（使用MutableAST，对标Java和JavaScript）
    """
    
    def __init__(self, strategies: Optional[Dict[str, float]] = None):
        """
        初始化C++代码增强器
        
        参数:
            strategies: 增强策略字典，mapping策略名称到应用概率
        """
        # C++特定的默认策略（使用MutableAST，对标Java）
        self.strategies = strategies or {
            # === Tier 1：MutableAST结构性变换（最重要，对标Java）===
            "apply_mutable_ast": 0.7,           # MutableAST结构变换（最关键）
            
            # === Tier 2：变量级变换 ===
            "full_variable_rename": 0.7,        # 完整变量重命名
            
            # === Tier 3：表面变换（辅助）===
            "add_redundant_parens": 0.6,        # 添加冗余括号
            "transform_boolean_literal": 0.5,   # 布尔字面量等价替换
            "transform_zero_literal": 0.4,      # 零的等价表达式
            "modify_blank_lines": 0.6,          # 添加/删除空行
            "modify_comments": 0.3,             # 修改注释
        }
        
        # 注册C++特定增强方法
        self.augmentation_methods = {
            # Tier 1
            "apply_mutable_ast": self._apply_mutable_ast_transforms,
            
            # Tier 2
            "full_variable_rename": self._apply_full_variable_rename,
            
            # Tier 3
            "add_redundant_parens": self._add_redundant_parentheses,
            "transform_boolean_literal": self._transform_boolean_literal,
            "transform_zero_literal": self._transform_zero_literal,
            "modify_blank_lines": self._modify_blank_lines,
            "modify_comments": self._modify_comments,
        }
    
    # ===== C++特有关键字 =====
    CPP_KEYWORDS = {
        # C语言关键字
        "auto", "break", "case", "char", "const", "continue", "default", "do",
        "double", "else", "enum", "extern", "float", "for", "goto", "if",
        "inline", "int", "long", "register", "return", "short", "signed",
        "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned",
        "void", "volatile", "while",
        # C++关键字
        "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel",
        "atomic_commit", "atomic_noexcept", "bitand", "bitor", "bool",
        "char8_t", "char16_t", "char32_t", "class", "compl", "concept",
        "consteval", "constexpr", "constinit", "co_await", "co_return",
        "co_yield", "decltype", "delete", "explicit", "export", "false",
        "friend", "mutable", "namespace", "new", "noexcept", "not", "not_eq",
        "nullptr", "operator", "or", "or_eq", "private", "protected", "public",
        "reflexpr", "requires", "static_assert", "static_cast", "synchronized",
        "template", "this", "thread_local", "throw", "true", "try",
        "typeid", "typename", "using", "virtual", "wchar_t", "xor", "xor_eq"
    }
    
    def _protect_string_literals(self, code: str):
        """保护C++字符串和字符字面量"""
        string_map = {}
        counter = 0
        
        patterns = [
            r'R"([a-zA-Z0-9_]*)\((?:[^)]|\)(?![a-zA-Z0-9_"]*\))*\)\1"',
            r'R"\((?:[^)]|\)(?!"))*\)"',
            r'"(?:[^"\\]|\\.)*"',
            r"'(?:[^'\\]|\\.)'",
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, code):
                placeholder = f"__STRING_{counter}__"
                string_map[placeholder] = match.group(0)
                code = code[:match.start()] + placeholder + code[match.end():]
                counter += 1
        
        return code, string_map
    
    def _restore_string_literals(self, code: str, string_map: Dict[str, str]) -> str:
        """恢复受保护的字符串"""
        for placeholder, original in string_map.items():
            code = code.replace(placeholder, original)
        return code
    
    def _get_feasible_mutable_transforms(self, code: str) -> Dict[str, List[str]]:
        """
        动态检测代码片段的可行MutableAST变换（对标Java）
        
        注意：这个方法尝试检测哪些MutableAST变换对代码是可行的
        """
        import tree_sitter
        
        feasible_map = {}
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        srcmarker_root = os.path.dirname(current_dir)
        
        # 尝试导入MutableAST
        if srcmarker_root not in sys.path:
            sys.path.insert(0, srcmarker_root)
        
        try:
            import mutable_tree.transformers as ast_transformers
            from code_transform_provider import CodeTransformProvider
        except ImportError:
            print("[警告] MutableAST不可用，无法检测可行变换")
            return feasible_map
        
        # 初始化所有transformers
        transformers = [
            ast_transformers.IfBlockSwapTransformer(),
            ast_transformers.CompoundIfTransformer(),
            ast_transformers.ConditionTransformer(),
            ast_transformers.LoopTransformer(),
            ast_transformers.InfiniteLoopTransformer(),
            ast_transformers.UpdateTransformer(),
            ast_transformers.SameTypeDeclarationTransformer(),
            ast_transformers.VarDeclLocationTransformer(),
            ast_transformers.VarInitTransformer(),
        ]
        
        # 初始化C++ parser
        try:
            parser = tree_sitter.Parser()
            lib_path = os.path.join(srcmarker_root, "parser", "languages.so")
            parser_lang = tree_sitter.Language(lib_path, "cpp")
            parser.set_language(parser_lang)
        except Exception as e:
            lib_path = os.path.join(srcmarker_root, "parser", "languages.so")
            print(f"[错误] C++ parser加载失败！")
            print(f"  当前工作目录: {os.getcwd()}")
            print(f"  srcmarker_root: {srcmarker_root}")
            print(f"  languages.so路径: {lib_path}")
            print(f"  文件是否存在: {os.path.exists(lib_path)}")
            print(f"  错误详情: {e}")
            return feasible_map
        
        # 初始化CodeTransformProvider
        try:
            transform_provider = CodeTransformProvider("cpp", parser, transformers)
        except Exception as e:
            print(f"[警告] CodeTransformProvider初始化失败: {e}")
            return feasible_map
        
        # 检测可行的变换
        for transformer in transformers:
            transformer_name = transformer.name
            feasible_keys = []
            
            try:
                available_keys = transformer.get_available_transforms()
                for key in available_keys:
                    try:
                        # 尝试应用变换验证可行性
                        new_code = transform_provider.code_transform(code, [key])
                        if new_code != code and len(new_code) > 0:
                            feasible_keys.append(key)
                    except:
                        continue
            except:
                continue
            
            if feasible_keys:
                feasible_map[transformer_name] = feasible_keys
        
        return feasible_map
    
    def _extract_method_from_full_class(self, code: str) -> Tuple[str, Optional[Dict]]:
        """
        从完整代码中提取方法/函数（对标Java的实现）
        
        C++中的代码可能是：
        - 独立函数
        - 类中的方法
        - 模板函数
        
        返回：(方法代码, 元数据)
        """
        # 简化实现：直接返回代码（C++通常直接是函数/方法代码）
        return code, None
    
    def _wrap_method_back_to_class(self, method_code: str, metadata: Optional[Dict]) -> str:
        """将方法重新包装回类中（对标Java）"""
        if metadata is None:
            return method_code
        return method_code
    
    def _apply_mutable_ast_transforms(self, code: str) -> str:
        """
        应用MutableAST变换（对标Java的_apply_mutable_ast_transforms）
        使用系统内置的C++ adaptor、stringifier和transformers
        
        参数:
            code: 原始代码
        
        返回:
            变换后的代码
        """
        import tree_sitter
        
        # 提取方法（对标Java）
        method_code, metadata = self._extract_method_from_full_class(code)
        
        # 获取可行的transforms（对标Java）
        feasible_map = self._get_feasible_mutable_transforms(code)
        
        if not feasible_map:
            # MutableAST不可用，返回原代码（对标Java）
            return code
        
        # 导入必要模块（对标Java）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        srcmarker_root = os.path.dirname(current_dir)
        if srcmarker_root not in sys.path:
            sys.path.insert(0, srcmarker_root)
        
        try:
            import mutable_tree.transformers as ast_transformers
            from code_transform_provider import CodeTransformProvider
        except ImportError:
            return code
        
        # 重建transformers（顺序必须一致）- 对标Java
        transformers = [
            ast_transformers.IfBlockSwapTransformer(),
            ast_transformers.CompoundIfTransformer(),
            ast_transformers.ConditionTransformer(),
            ast_transformers.LoopTransformer(),
            ast_transformers.InfiniteLoopTransformer(),
            ast_transformers.UpdateTransformer(),
            ast_transformers.SameTypeDeclarationTransformer(),
            ast_transformers.VarDeclLocationTransformer(),
            ast_transformers.VarInitTransformer(),
        ]
        
        # 初始化parser（对标Java）
        parser = tree_sitter.Parser()
        try:
            parser_lang = tree_sitter.Language(
                os.path.join(srcmarker_root, "parser", "languages.so"), 
                "cpp"
            )
            parser.set_language(parser_lang)
        except Exception:
            return code
        
        # 初始化CodeTransformProvider（对标Java）
        try:
            transform_provider = CodeTransformProvider("cpp", parser, transformers)
        except Exception as e:
            return code
        
        # 从可行选项中随机选择组合（对标Java）
        selected_keys = []
        for transformer in transformers:
            transformer_name = transformer.name
            feasible_keys = feasible_map.get(transformer_name, [])
            if feasible_keys:
                selected_keys.append(random.choice(feasible_keys))
            else:
                # 回退到第一个选项
                available = transformer.get_available_transforms()
                selected_keys.append(available[0] if available else "")
        
        # 应用变换（对标Java）
        try:
            transformed_method = transform_provider.code_transform(method_code, selected_keys)
            
            # 重新包装回完整类
            if metadata is not None:
                return self._wrap_method_back_to_class(transformed_method, metadata)
            else:
                return transformed_method
                
        except Exception as e:
            return code
    
    def _apply_full_variable_rename(self, code: str, strategy: str = "random") -> str:
        """
        应用完整变量重命名
        
        参数:
            code: 代码
            strategy: 重命名策略
        
        返回:
            重命名后的代码
        """
        import sys
        import os
        
        # 导入重命名器
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xdf_root = os.path.dirname(os.path.dirname(current_dir))
        watermark_root = os.path.join(xdf_root, "Watermark4code")
        if watermark_root not in sys.path:
            sys.path.insert(0, watermark_root)
        
        try:
            from experiments.Attack.Rename_Attack.cpp_variable_renamer import CppVariableRenamer
            from experiments.Attack.Rename_Attack.attack_config import AttackConfig
            
            renamer = CppVariableRenamer(code)
            
            # 收集变量
            local_vars = renamer.collect_local_variables()
            parameters = renamer.collect_parameters()
            
            # 生成重命名映射
            for var_name in local_vars.keys():
                new_name = renamer.generate_new_name(var_name, strategy)
                renamer.rename_mapping[var_name] = new_name
            
            for param_name in parameters.keys():
                if param_name not in renamer.rename_mapping:
                    new_name = renamer.generate_new_name(param_name, strategy)
                    renamer.rename_mapping[param_name] = new_name
            
            # 应用重命名
            config = AttackConfig(rename_ratio=1.0, naming_strategy=strategy)
            renamed_code = renamer.apply_renames(config)
            
            return renamed_code
        except Exception as e:
            print(f"[警告] 完整重命名失败: {e}")
            return code
    
    def _add_redundant_parentheses(self, code: str) -> str:
        """添加冗余括号"""
        if random.random() > self.strategies.get("add_redundant_parens", 0.6):
            return code
        
        try:
            protected_code, string_map = self._protect_string_literals(code)
            
            # 简单启发式：在赋值和比较周围添加括号
            patterns = [
                (r'(\w+)\s*=\s*([^;]+);', r'\1 = (\2);'),
                (r'if\s*\(([^)]+)\)', r'if ((\1))'),
            ]
            
            result = protected_code
            for pattern, replacement in patterns:
                if random.random() < 0.5:
                    result = re.sub(pattern, replacement, result)
            
            result = self._restore_string_literals(result, string_map)
            return result if result != code else code
        except:
            return code
    
    def _transform_boolean_literal(self, code: str) -> str:
        """布尔字面量等价替换"""
        if random.random() > self.strategies.get("transform_boolean_literal", 0.5):
            return code
        
        try:
            protected_code, string_map = self._protect_string_literals(code)
            
            # 替换 true 和 false
            if random.random() < 0.5:
                protected_code = re.sub(r'\btrue\b', '(1==1)', protected_code)
            else:
                protected_code = re.sub(r'\bfalse\b', '(1!=1)', protected_code)
            
            result = self._restore_string_literals(protected_code, string_map)
            return result if result != code else code
        except:
            return code
    
    def _transform_zero_literal(self, code: str) -> str:
        """零的等价表达式"""
        if random.random() > self.strategies.get("transform_zero_literal", 0.4):
            return code
        
        try:
            protected_code, string_map = self._protect_string_literals(code)
            
            # 将数字0替换为等价表达式
            protected_code = re.sub(r'\b0\b', '(1-1)', protected_code)
            
            result = self._restore_string_literals(protected_code, string_map)
            return result if result != code else code
        except:
            return code
    
    def _modify_blank_lines(self, code: str) -> str:
        """修改空行"""
        if random.random() > self.strategies.get("modify_blank_lines", 0.6):
            return code
        
        try:
            lines = code.split('\n')
            result_lines = []
            
            for i, line in enumerate(lines):
                result_lines.append(line)
                
                # 随机在非空行后添加空行
                if line.strip() and random.random() < 0.1:
                    result_lines.append('')
            
            result = '\n'.join(result_lines)
            return result if result != code else code
        except:
            return code
    
    def _modify_comments(self, code: str) -> str:
        """修改注释"""
        if random.random() > self.strategies.get("modify_comments", 0.3):
            return code
        
        try:
            protected_code, string_map = self._protect_string_literals(code)
            
            lines = protected_code.split('\n')
            result_lines = []
            
            for i, line in enumerate(lines):
                result_lines.append(line)
                
                if line.strip() and not line.strip().startswith('//') and random.random() < 0.1:
                    if '{' in line or ';' in line:
                        indent = len(line) - len(line.lstrip())
                        comment = ' ' * indent + '// Processing'
                        result_lines.append(comment)
            
            result = '\n'.join(result_lines)
            result = self._restore_string_literals(result, string_map)
            return result if result != code else code
        except:
            return code
    
    def augment(self, code_string: str, num_augmentations: int = 1) -> List[str]:
        """
        生成正样本（多个增强变体）- 对标Java/JavaScript版本
        
        参数:
            code_string: 原始代码
            num_augmentations: 要生成的增强变体数量（默认1，对标base class）
        
        返回:
            增强代码列表，List[str]
        """
        augmented_samples = []
        
        for _ in range(num_augmentations):
            # 选择要应用的策略（基于概率）
            applied_strategies = [
                strategy for strategy in self.strategies
                if random.random() < self.strategies[strategy]
            ]
            
            # 应用选定的策略
            augmented_code = code_string
            for strategy in applied_strategies:
                if strategy in self.augmentation_methods:
                    try:
                        augmented_code = self.augmentation_methods[strategy](augmented_code)
                    except Exception as e:
                        # 跳过失败的增强
                        continue
            
            # 只有在代码改变时才添加
            if augmented_code != code_string:
                augmented_samples.append(augmented_code)
        
        # 如果无法生成任何有效增强，至少改变空行（对标base class）
        if not augmented_samples:
            try:
                whitespace_changed = self._modify_blank_lines(code_string)
                augmented_samples.append(whitespace_changed)
            except Exception:
                # 最后的保底：返回原始代码
                augmented_samples.append(code_string)
        
        return augmented_samples
    
    def create_hard_negative(self, code_string: str) -> str:
        """
        为C++代码生成语义不同但语法相似的困难负样本
        """
        # 替换一些关键操作符
        replacements = {
            " == ": " != ",
            " != ": " == ",
            " > ": " <= ",
            " < ": " >= ",
            " >= ": " < ",
            " <= ": " > ",
            " && ": " || ",
            " || ": " && ",
            "true": "false",
            "false": "true"
        }
        
        # 尝试替换操作符
        modified = code_string
        replacement_made = False
        
        for original, replacement in replacements.items():
            if original in modified:
                # 只替换一次
                pos = modified.find(original)
                modified = modified[:pos] + replacement + modified[pos + len(original):]
                replacement_made = True
                break
        
        # 如果没有找到操作符，尝试修改数值常量
        if not replacement_made:
            # 查找数值常量
            num_pattern = r'\b(\d+)\b'
            matches = list(re.finditer(num_pattern, code_string))
            
            if matches:
                # 选择一个随机数值进行修改
                match = random.choice(matches)
                num = int(match.group(1))
                
                if num == 0:
                    new_num = 1
                elif num == 1:
                    new_num = 0
                else:
                    # 改变符号或加/减一个小值
                    ops = [lambda x: -x, lambda x: x + 1, lambda x: x - 1, lambda x: x * 2]
                    new_num = random.choice(ops)(num)
                
                modified = code_string[:match.start()] + str(new_num) + code_string[match.end():]
                replacement_made = True
        
        # 如果还是没法修改，尝试移除一个重要语句
        if not replacement_made:
            lines = code_string.split('\n')
            for i, line in enumerate(lines):
                # 寻找包含return、赋值或条件语句的行
                if ("return" in line or "=" in line or 
                    "if" in line or "for" in line or 
                    "while" in line) and ";" in line:
                    # 注释掉这行，而不是删除
                    lines[i] = "// " + line
                    modified = '\n'.join(lines)
                    break
        
        return modified


# ===== 训练数据生成函数（对标java_augmentor.py）=====

def generate_cpp_training_data(
    input_file: str,
    output_file: str,
    model=None,
    split_type: str = "train",
    positive_ratio: float = 0.6,
    augmentation_types: Dict[str, float] = None,
    max_samples: int = None,
    parallel: bool = True,
    workers: int = 4,
    batch_size: int = 50,
    resume: bool = False
) -> int:
    """
    生成C++对比学习训练数据（完全对标generate_java_training_data）
    
    参数:
        input_file: 输入C++代码文件路径
        output_file: 输出训练数据文件路径
        model: LLM模型实例（C++版本暂不使用LLM）
        split_type: 数据集类型，'train'/'valid'/'test'
        positive_ratio: 正样本比例
        augmentation_types: 增强类型及其概率字典
        max_samples: 最大样本数
        parallel: 是否启用并行处理（C++版本暂不使用）
        workers: 并行工作线程数量（C++版本暂不使用）
        batch_size: 每批处理的样本数（C++版本暂不使用）
        resume: 是否从上次中断点恢复
        
    返回:
        处理的样本总数
    """
    return generate_cpp_training_data_parallel(
        input_file=input_file,
        output_file=output_file,
        split_type=split_type,
        positive_ratio=positive_ratio,
        augmentation_types=augmentation_types,
        max_samples=max_samples,
        num_workers=workers,
        batch_size=batch_size,
        resume=resume
    )


def generate_cpp_training_data_parallel(
    input_file: str,
    output_file: str,
    model=None,  # ✅ 添加 model 参数（对标Java/JavaScript），虽然C++版本不使用
    split_type: str = "train",
    positive_ratio: float = 0.6,
    augmentation_types: Dict[str, float] = None,
    max_samples: int = None,
    num_workers: int = 4,
    batch_size: int = 50,
    resume: bool = False
) -> int:
    """
    并行版本的C++训练数据生成（完全对标generate_java_training_data_parallel）
    """
    import json
    import random
    from tqdm import tqdm
    
    print(f"为{split_type}数据并行生成增强样本（使用{num_workers}个工作线程）...")
    
    # 进度记录文件
    progress_file = f"{output_file}.progress"
    
    # 默认增强类型概率（对标Java版本，仅使用语义保持）
    augmentation_types = augmentation_types or {
        "semantic_preserving": 1.0
    }
    
    # 加载源代码（对标Java版本）
    source_codes = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    code = data.get("code", "")
                    if code and len(code.strip()) > 0:
                        source_codes.append(code)
                except Exception:
                    continue
    
    if not source_codes:
        print(f"[警告] 未能从 {input_file} 加载任何代码")
        return 0
    
    # 限制样本数（对标Java版本）
    if max_samples and len(source_codes) > max_samples:
        random.shuffle(source_codes)
        source_codes = source_codes[:max_samples]
    
    print(f"加载了 {len(source_codes)} 个C++代码样本")
    
    # 初始化C++ augmentor
    cpp_augmentor = CppCodeAugmentor()
    
    # 生成训练数据
    processed = 0
    with open(output_file, 'w', encoding='utf-8', newline='\n') as out_file:
        try:
            for i, anchor_code in enumerate(tqdm(source_codes, desc=f"{split_type}处理进度")):
                # 按比例决定生成正样本还是负样本
                is_positive = random.random() < positive_ratio
                
                try:
                    if is_positive:
                        # 使用C++代码增强器生成正样本（对标Java）
                        # augment()现在返回List[str]，对标base class
                        positive_samples = cpp_augmentor.augment(anchor_code)
                        if positive_samples:
                            positive_code = positive_samples[0]
                            # 确保生成的代码与原代码不同
                            if positive_code != anchor_code:
                                out_file.write(json.dumps({
                                    "anchor": anchor_code,
                                    "positive": positive_code,
                                    "type": "augment"
                                }, ensure_ascii=False) + "\n")
                                processed += 1
                    
                    else:
                        # 仅使用简单负样本：从全量源代码中随机选择非自身的代码（对标Java）
                        available_negatives = [c for c in source_codes if c != anchor_code]
                        if available_negatives:
                            negative_code = random.choice(available_negatives)
                            out_file.write(json.dumps({
                                "anchor": anchor_code,
                                "negative": negative_code,
                                "type": "random_negative"
                            }, ensure_ascii=False) + "\n")
                            processed += 1
                            
                except Exception as e:
                    continue
        
        except KeyboardInterrupt:
            print(f"\n已中断。已生成 {processed} 个样本")
            raise
    
    print(f"共生成 {processed} 个处理后的C++样本")
    return processed
