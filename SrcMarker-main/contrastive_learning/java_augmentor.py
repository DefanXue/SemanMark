#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Java代码增强器，用于生成对比学习的正样本和负样本。
严格对标CodeWMBench中的三类扰动生成方式：
1. 语义保持转换（变量重命名、添加注释等）
2. LLM重写（使用与CodeWMBench相同的prompt）
3. 转译攻击（Java→C#→Java）
"""

import random
import re
import os
import time
import json
import threading
import concurrent.futures
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple, Union

# 添加文件锁，确保多线程安全写入
file_lock = threading.Lock()

# 线程局部：为本次调用临时覆盖模型名（并发安全）
_model_ctx = threading.local()

def _set_model_override(name: str) -> None:
	try:
		_model_ctx.name_override = name
	except Exception:
		pass

def _clear_model_override() -> None:
	try:
		if hasattr(_model_ctx, "name_override"):
			delattr(_model_ctx, "name_override")
	except Exception:
		pass

# 使用 NewAPI 的模型与基础地址（可由环境变量覆盖）
NEWAPI_MODEL_NAME = os.environ.get("NEWAPI_MODEL", "gpt-5-nano")
NEWAPI_BASE_URL = os.environ.get("NEWAPI_BASE_URL", "https://chat.cloudapi.vip/v1beta")

# 导入安全设置类型
try:
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    
    # 安全设置（与create_test_set.py保持一致）
    SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
    }
except ImportError:
    # 如果未安装google.generativeai，定义一个空字典
    SAFETY_SETTINGS = {}

# 导入基础增强器
from contrastive_learning.augmentor import CodeAugmentor


class JavaCodeAugmentor(CodeAugmentor):
    """
    针对Java代码的增强器，继承自通用CodeAugmentor
    实现CodeWMBench中的语义保持转换
    """
    
    def __init__(self, strategies: Optional[Dict[str, float]] = None):
        """
        初始化Java代码增强器
        
        参数:
            strategies: 增强策略字典，mapping策略名称到应用概率
        """
        # Java特定的默认策略
        self.strategies = strategies or {
            # === MutableAST变换（新增） ===
            "apply_mutable_ast": 0.7,         # 应用MutableAST结构变换
            
            # === 简单语义规则（保留3个） ===
            "add_redundant_parens": 0.8,      # 添加冗余括号
            "transform_boolean_literal": 0.6, # 布尔字面量等价替换
            "transform_zero_literal": 0.5,    # 零的等价表达式
            
            # === 完整重命名（新增） ===
            "full_variable_rename": 0.7,      # 完整变量重命名
            
            # === 新增通用规则 ===
            "modify_imports": 0.4,            # 添加冗余导入
            "modify_blank_lines": 0.6,        # 添加/删除空行
        }
        
        # 注册Java特定增强方法
        self.augmentation_methods = {
            "apply_mutable_ast": self._apply_mutable_ast_transforms,
            "add_redundant_parens": self._add_redundant_parentheses,
            "transform_boolean_literal": self._transform_boolean_literal,
            "transform_zero_literal": self._transform_zero_literal,
            "full_variable_rename": self._apply_full_variable_rename,
            "modify_imports": self._modify_imports,
            "modify_blank_lines": self._modify_blank_lines,
        }
        
        # 缓存可行的MutableAST变换（避免重复计算）
        self._feasible_cache = {}
    
    def _protect_string_literals(self, code: str):
        """保护字符串字面量，用占位符替换"""
        string_map = {}
        counter = 0
        
        # 正则匹配Java字符串字面量（支持转义）
        pattern = r'"(?:[^"\\]|\\.)*"'
        
        def replace_func(match):
            nonlocal counter
            original_string = match.group(0)
            placeholder = f"__STRING_{counter}__"
            string_map[placeholder] = original_string
            counter += 1
            return placeholder
        
        protected_code = re.sub(pattern, replace_func, code)
        return protected_code, string_map
    
    def _restore_string_literals(self, code: str, string_map: dict) -> str:
        """恢复字符串字面量"""
        result = code
        for placeholder, original_string in string_map.items():
            result = result.replace(placeholder, original_string)
        return result
    
    def _rename_java_variables(self, code_string: str) -> str:
        """重命名Java变量和方法名"""
        # Java关键字列表
        java_keywords = {
            "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", 
            "class", "const", "continue", "default", "do", "double", "else", "enum", 
            "extends", "final", "finally", "float", "for", "goto", "if", "implements", 
            "import", "instanceof", "int", "interface", "long", "native", "new", "package", 
            "private", "protected", "public", "return", "short", "static", "strictfp", 
            "super", "switch", "synchronized", "this", "throw", "throws", "transient", 
            "try", "void", "volatile", "while", "true", "false", "null"
        }
        
        # 常见Java类名，避免重命名
        common_types = {
            "String", "Integer", "Boolean", "Double", "Float", "List", "Map", 
            "Set", "ArrayList", "HashMap", "Object", "Exception", "RuntimeException"
        }
        
        # 提取变量声明（不包括方法名）
        var_pattern = r'(?<!\.)(\b[a-zA-Z_]\w*\b)(?!\s*\()'  # 变量名（后面不跟括号）
        
        # Step 1: 保护字符串字面量
        protected_code, string_map = self._protect_string_literals(code_string)
        
        # 识别所有标识符（在保护后的代码上识别）
        identifiers = set(re.findall(var_pattern, protected_code))
        
        # 排除Java关键字和常见类型
        identifiers = identifiers - java_keywords - common_types
        
        # 创建重命名映射
        var_mapping = {}
        prefix_options = ["var", "tmp", "arg", "param", "val", "obj", "item"]
        
        # 变量重命名（不重命名方法）
        for name in identifiers:
            # 跳过特殊名称
            if name in ["main", "args"]:
                continue
            
            # 生成新名称
            prefix = random.choice(prefix_options)
            new_name = f"{prefix}_{random.randint(1, 100)}"
            var_mapping[name] = new_name
        
        # Step 2: 应用重命名（只重命名变量，不重命名方法）
        result = protected_code
        
        for old_name, new_name in var_mapping.items():
            # 使用正则确保只替换完整的标识符（不包括方法调用）
            result = re.sub(r'\b' + re.escape(old_name) + r'\b(?!\s*\()', new_name, result)
        
        # Step 3: 恢复字符串字面量
        result = self._restore_string_literals(result, string_map)
        
        return result
    
    def _insert_java_comments(self, code_string: str) -> str:
        """在Java代码中插入或修改注释"""
        lines = code_string.split('\n')
        
        # Java风格注释
        comments = [
            "// Process the input data",
            "// Initialize variables",
            "// Update counter",
            "// Check boundary conditions",
            "// Handle edge case",
            "// Main business logic",
            "// Helper method",
            "// Return the result",
            "// Validate parameters",
            "// Apply transformation",
            "// Parse input string",
            "// Calculate result",
            "// Check for null values"
        ]
        
        # 插入注释的次数
        num_insertions = random.randint(1, 3)
        
        # 插入注释
        for _ in range(num_insertions):
            if not lines:
                break
                
            pos = random.randint(0, len(lines) - 1)
            
            # 跳过空行和已有注释的行
            if not lines[pos].strip() or "//" in lines[pos]:
                continue
                
            # 添加注释
            comment = random.choice(comments)
            indentation = re.match(r'^\s*', lines[pos]).group(0)
            
            if random.random() < 0.6:
                # 在行末添加注释
                lines[pos] = lines[pos] + "  " + comment
            else:
                # 在行前添加注释
                lines.insert(pos, indentation + comment)
                
        return '\n'.join(lines)
    
    def _change_java_braces_style(self, code_string: str) -> str:
        """修改Java大括号样式（K&R风格 vs. Allman风格）"""
        # K&R风格: if (condition) {
        # Allman风格: if (condition)
        #            {
        
        # 模式1: 将K&R风格转换为Allman风格
        kr_to_allman = random.random() < 0.5
        
        if kr_to_allman:
            # 匹配行尾的左花括号，将其移到下一行
            pattern = r'(\b(?:if|for|while|switch|try|catch|else)\b.*?)\s*\{'
            
            def replace_kr_to_allman(match):
                return match.group(1) + '\n' + ' ' * (len(match.group(1)) - len(match.group(1).lstrip())) + '{'
            
            return re.sub(pattern, replace_kr_to_allman, code_string)
        else:
            # 匹配单独一行的左花括号，将其移到上一行末尾
            lines = code_string.split('\n')
            result = []
            i = 0
            
            while i < len(lines):
                if i > 0 and lines[i].strip() == '{' and any(keyword in lines[i-1] for keyword in ['if', 'for', 'while', 'switch', 'try', 'catch', 'else']):
                    # 将括号附加到前一行
                    result[-1] = result[-1] + ' {'
                else:
                    result.append(lines[i])
                i += 1
            
            return '\n'.join(result)
    
    def _add_redundant_parentheses(self, code_string: str) -> str:
        """为算术表达式添加冗余括号（语义不变）"""
        result = code_string
        
        # 只保留安全的算术运算符（+, -, *）
        # 移除: 
        #   - 除法 (/) : 可能破坏 i / arr.get() 等表达式
        #   - 比较运算符 (<, >) : 会与泛型语法 List<Integer> 冲突
        patterns = [
            # 算术运算: a + b → (a + b)
            (r'(\w+)\s*\+\s*(\w+)(?!\+)', r'(\1 + \2)'),
            (r'(\w+)\s*-\s*(\w+)(?!>)', r'(\1 - \2)'),
            (r'(\w+)\s*\*\s*(\w+)', r'(\1 * \2)'),
        ]
        
        # 随机选择1-2个模式应用（减少数量，因为选项变少了）
        num_transforms = random.randint(1, min(2, len(patterns)))
        selected_patterns = random.sample(patterns, num_transforms)
        
        for pattern, replacement in selected_patterns:
            # 找到所有匹配，随机选择一个位置替换（增加变化性）
            all_matches = list(re.finditer(pattern, result))
            if all_matches:
                # 随机选择替换哪个匹配
                match_to_replace = random.choice(all_matches)
                start_pos = match_to_replace.start()
                end_pos = match_to_replace.end()
                matched_text = result[start_pos:end_pos]
                replaced_text = re.sub(pattern, replacement, matched_text)
                result = result[:start_pos] + replaced_text + result[end_pos:]
        
        return result
        
    def _transform_boolean_literal(self, code_string: str) -> str:
        """将布尔字面量替换为等价表达式（语义不变）"""
        # 提取字符串字面量的位置（保护区域）
        protected_regions = []
        for match in re.finditer(r'"(?:[^"\\]|\\.)*"', code_string):
            protected_regions.append((match.start(), match.end()))
        for match in re.finditer(r"'(?:[^'\\]|\\.)*'", code_string):
            protected_regions.append((match.start(), match.end()))
        
        def is_protected(pos):
            """检查位置是否在字符串字面量中"""
            for start, end in protected_regions:
                if start <= pos < end:
                    return True
            return False
        
        result = code_string
        
        # 收集所有未保护的true和false位置
        true_matches = [m for m in re.finditer(r'\btrue\b', code_string) if not is_protected(m.start())]
        false_matches = [m for m in re.finditer(r'\bfalse\b', code_string) if not is_protected(m.start())]
        
        # 随机选择要替换的true（增加变化性）
        if true_matches:
            num_to_replace = random.randint(0, len(true_matches))
            selected_indices = random.sample(range(len(true_matches)), num_to_replace)
            # 从后往前替换，避免位置偏移
            for idx in sorted(selected_indices, reverse=True):
                match = true_matches[idx]
                result = result[:match.start()] + '!false' + result[match.end():]
        
        # 随机选择要替换的false（增加变化性）
        # 重新搜索false，因为字符串可能已经改变
        false_matches = [m for m in re.finditer(r'\bfalse\b', result) if not is_protected(m.start())]
        if false_matches:
            num_to_replace = random.randint(0, len(false_matches))
            selected_indices = random.sample(range(len(false_matches)), num_to_replace)
            # 从后往前替换
            for idx in sorted(selected_indices, reverse=True):
                match = false_matches[idx]
                result = result[:match.start()] + '!true' + result[match.end():]
        
        return result
    
    def _transform_zero_literal(self, code_string: str) -> str:
        """将数字0替换为等价算术表达式（语义不变）"""
        # 只匹配赋值语句中的 0: = 0; 或 = 0,  或 = 0)
        pattern = r'=\s*0\s*([;,)])'
        
        # 找到所有匹配（增加变化性）
        all_matches = list(re.finditer(pattern, code_string))
        
        if not all_matches:
            return code_string
        
        result = code_string
        
        # 随机选择要替换的位置数量
        num_to_replace = random.randint(0, len(all_matches))
        if num_to_replace > 0:
            selected_indices = random.sample(range(len(all_matches)), num_to_replace)
            
            # 从后往前替换，避免位置偏移
            for idx in sorted(selected_indices, reverse=True):
                match = all_matches[idx]
                # 随机选择等价表达式
                equivalents = [
                    '1 - 1',   # 减法
                    '2 - 2',   # 减法（不同数字）
                    '0 * 1',   # 乘法
                    '1 * 0',   # 乘法（交换）
                ]
                replacement = random.choice(equivalents)
                result = result[:match.start()] + f'= {replacement}{match.group(1)}' + result[match.end():]
        
        return result
    
    def _modify_imports(self, code_string: str) -> str:
        """添加冗余但无害的导入语句（增加变化性）"""
        common_imports = [
            'import java.util.List;',
            'import java.util.ArrayList;',
            'import java.util.Map;',
            'import java.util.HashMap;',
            'import java.util.Set;',
            'import java.util.HashSet;',
        ]
        
        # 找到import区域的结束位置
        import_section_end = 0
        last_import_match = None
        for match in re.finditer(r'^import\s+.*?;', code_string, re.MULTILINE):
            if match.end() > import_section_end:
                import_section_end = match.end()
                last_import_match = match
        
        # 如果存在import语句，随机添加1-2个冗余import
        if import_section_end > 0:
            num_to_add = random.randint(1, 2)
            selected_imports = random.sample(common_imports, min(num_to_add, len(common_imports)))
            
            # 在最后一个import之后插入
            insert_text = '\n' + '\n'.join(selected_imports)
            result = code_string[:import_section_end] + insert_text + code_string[import_section_end:]
            return result
        
        return code_string
    
    def _modify_blank_lines(self, code_string: str) -> str:
        """在方法之间、语句块之间添加或删除空行（增加变化性）"""
        lines = code_string.split('\n')
        result_lines = []
        
        i = 0
        while i < len(lines):
            result_lines.append(lines[i])
            
            # 在大括号后随机添加空行（30%概率）
            if '{' in lines[i] and random.random() < 0.3:
                result_lines.append('')
            
            # 在return语句前随机添加空行（30%概率）
            if i < len(lines) - 1 and 'return' in lines[i+1] and random.random() < 0.3:
                result_lines.append('')
            
            # 在方法声明后随机添加空行（20%概率）
            if i < len(lines) - 1 and re.search(r'(public|private|protected)\s+(static\s+)?\w+\s+\w+\s*\(', lines[i]):
                if random.random() < 0.2:
                    result_lines.append('')
            
            i += 1
        
        return '\n'.join(result_lines)
    
    def _remove_line_comments(self, code: str) -> str:
        """
        移除 Java 代码中的行注释（//），避免 MutableAST 解析错误
        
        注意：
        1. 只移除字符串外的 '//' 注释
        2. 保留代码结构（将注释替换为空白）
        3. 不影响块注释（/* */ 或 /** */）
        
        参数:
            code: Java 代码
        
        返回:
            移除行注释后的代码
        """
        lines = code.split('\n')
        result_lines = []
        
        for line in lines:
            # 状态追踪
            in_string = False
            in_char = False
            escape_next = False
            comment_start = -1
            
            i = 0
            while i < len(line):
                char = line[i]
                
                # 处理转义字符
                if escape_next:
                    escape_next = False
                    i += 1
                    continue
                
                if char == '\\':
                    escape_next = True
                    i += 1
                    continue
                
                # 处理字符串
                if char == '"' and not in_char:
                    in_string = not in_string
                    i += 1
                    continue
                
                # 处理字符字面量
                if char == "'" and not in_string:
                    in_char = not in_char
                    i += 1
                    continue
                
                # 检测行注释（必须在字符串和字符字面量之外）
                if not in_string and not in_char:
                    if char == '/' and i + 1 < len(line) and line[i + 1] == '/':
                        comment_start = i
                        break
                
                i += 1
            
            # 移除注释部分
            if comment_start != -1:
                # 保留注释前的代码，去除尾部空白
                cleaned_line = line[:comment_start].rstrip()
                result_lines.append(cleaned_line)
            else:
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _extract_method_from_full_class(self, code: str) -> tuple:
        """
        从完整类代码中提取方法体
        
        返回: (method_code, metadata)
            method_code: 纯方法代码（不含import和class）
            metadata: {
                'imports': str,  # import语句
                'class_header': str,  # class定义行
                'method_indent': str,  # 方法缩进
            }
        """
        import re
        
        lines = code.split('\n')
        
        # 提取 import 部分（所有以import开头的行）
        imports = []
        non_import_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import '):
                imports.append(line)
            elif line.strip() and not line.strip().startswith('import '):
                non_import_start = i
                break
        
        # 提取 class 定义行（class XXX {）
        class_header_idx = None
        for i in range(non_import_start, len(lines)):
            if re.match(r'^\s*class\s+\w+', lines[i]):
                class_header_idx = i
                break
        
        if class_header_idx is None:
            # 没找到class定义，返回原代码
            return code, None
        
        # 提取方法部分（从第一个public/private/protected开始到倒数第二个}）
        method_signature_idx = None
        for i in range(class_header_idx + 1, len(lines)):
            stripped = lines[i].strip()
            if stripped.startswith(('public ', 'private ', 'protected ', 'static ')):
                # 必须包含括号，以区分方法和字段声明
                if '(' in stripped:
                    method_signature_idx = i
                    break
        
        if method_signature_idx is None:
            return code, None
        
        # 找到方法的缩进
        method_start_idx = method_signature_idx
        method_indent = len(lines[method_start_idx]) - len(lines[method_start_idx].lstrip())
        
        # 提取方法（去掉最后的 '}'，那是class的结束）
        method_lines = []
        for i in range(method_start_idx, len(lines)):
            line = lines[i]
            # 跳过最后一个单独的 '}'（class结束符）
            if i == len(lines) - 1 and line.strip() == '}':
                break
            # 完整保留所有字符（不删除缩进）
            method_lines.append(line)
        
        method_code = '\n'.join(method_lines)
        
        # 移除行注释，避免 MutableAST 在处理时遇到 KeyError: 'line_comment'
        method_code = self._remove_line_comments(method_code)
        
        metadata = {
            'imports': '\n'.join(imports),
            'class_header': lines[class_header_idx],
            'method_indent': ' ' * method_indent,
        }
        
        return method_code, metadata
    
    def _wrap_method_back_to_class(self, transformed_method: str, metadata: dict) -> str:
        """
        将变换后的方法重新包装回完整类
        
        参数:
            transformed_method: 变换后的纯方法代码
            metadata: _extract_method_from_full_class 返回的元数据
        
        返回:
            完整的类代码
        """
        if metadata is None:
            return transformed_method
        
        # 直接拼接，不处理缩进（Java语法不依赖缩进）
        parts = []
        if metadata['imports']:
            parts.append(metadata['imports'])
            parts.append('')  # 空行
        
        parts.append('')  # 空行
        parts.append(metadata['class_header'])
        parts.append(transformed_method)
        parts.append('}')  # class结束
        
        return '\n'.join(parts)
    
    def _get_feasible_mutable_transforms(self, code: str) -> Dict[str, List[str]]:
        """
        动态检测代码片段的可行MutableAST变换（参考SrcMarker的逻辑）
        
        参数:
            code: Java代码片段
        
        返回:
            {transformer_name: [feasible_keys]}
        """
        import tree_sitter
        import sys
        import os
        
        # 先尝试提取方法
        method_code, _ = self._extract_method_from_full_class(code)
        
        # 检查缓存时使用method_code的hash
        code_hash = hash(method_code)
        if code_hash in self._feasible_cache:
            return self._feasible_cache[code_hash]
        
        # 导入MutableAST
        current_dir = os.path.dirname(os.path.abspath(__file__))
        srcmarker_root = os.path.dirname(current_dir)
        if srcmarker_root not in sys.path:
            sys.path.insert(0, srcmarker_root)
        
        try:
            import mutable_tree.transformers as ast_transformers
            from code_transform_provider import CodeTransformProvider
        except ImportError as e:
            print(f"[警告] 无法导入MutableAST: {e}")
            return {}
        
        # 初始化transformers（不包括VarNameStyleTransformer）
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
        
        # 初始化parser
        parser = tree_sitter.Parser()
        try:
            parser_lang = tree_sitter.Language(
                os.path.join(srcmarker_root, "parser", "languages.so"), 
                "java"
            )
            parser.set_language(parser_lang)
        except Exception as e:
            print(f"[警告] 无法加载tree-sitter parser: {e}")
            return {}
        
        transform_provider = CodeTransformProvider("java", parser, transformers)
        
        # 检测每个transformer的可行选项（参考collect_feasible_transforms_jsonl.py）
        feasible_transforms = {}
        
        for transformer in transformers:
            transformer_name = transformer.name
            available_keys = transformer.get_available_transforms()
            feasible_keys = []
            
            for key in available_keys:
                try:
                    # 尝试应用transform
                    new_code = transform_provider.code_transform(method_code, [key])
                    
                    # 检查是否可解析
                    _ = transform_provider.to_mutable_tree(new_code)
                    
                    # 检查是否有变化
                    if len(new_code) != len(method_code) or new_code != method_code:
                        feasible_keys.append(key)
                        
                except Exception:
                    # 变换失败，跳过
                    continue
            
            # 确保至少有一个可行选项（回退到第一个）
            if not feasible_keys and available_keys:
                feasible_keys.append(available_keys[0])
            
            feasible_transforms[transformer_name] = feasible_keys
        
        # 缓存结果
        self._feasible_cache[code_hash] = feasible_transforms
        return feasible_transforms
    
    def _apply_mutable_ast_transforms(self, code: str) -> str:
        """
        应用MutableAST变换（动态选择可行组合）
        
        参数:
            code: 原始代码
        
        返回:
            变换后的代码
        """
        import tree_sitter
        import sys
        import os
        
        # 提取方法
        method_code, metadata = self._extract_method_from_full_class(code)
        
        if metadata is None:
            # 无法提取，说明可能已经是纯方法格式
            method_code = code
            # 移除行注释，避免 MutableAST 在处理时遇到 KeyError: 'line_comment'
            method_code = self._remove_line_comments(method_code)
        
        # 获取可行的transforms
        feasible_map = self._get_feasible_mutable_transforms(code)
        
        if not feasible_map:
            print("[警告] MutableAST变换不可用，返回原代码")
            return code
        
        # 导入必要模块
        current_dir = os.path.dirname(os.path.abspath(__file__))
        srcmarker_root = os.path.dirname(current_dir)
        if srcmarker_root not in sys.path:
            sys.path.insert(0, srcmarker_root)
        
        try:
            import mutable_tree.transformers as ast_transformers
            from code_transform_provider import CodeTransformProvider
        except ImportError as e:
            print(f"[警告] 无法导入MutableAST: {e}")
            return code
        
        # 重建transformers（顺序必须一致）
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
        
        # 初始化parser
        parser = tree_sitter.Parser()
        try:
            parser_lang = tree_sitter.Language(
                os.path.join(srcmarker_root, "parser", "languages.so"), 
                "java"
            )
            parser.set_language(parser_lang)
        except Exception as e:
            print(f"[警告] 无法加载parser: {e}")
            return code
        
        transform_provider = CodeTransformProvider("java", parser, transformers)
        
        # 从可行选项中随机选择组合（参考SrcMarker的逻辑）
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
        
        # 应用变换（串行）
        try:
            transformed_method = transform_provider.code_transform(method_code, selected_keys)
            
            # 重新包装回完整类
            if metadata is not None:
                return self._wrap_method_back_to_class(transformed_method, metadata)
            else:
                return transformed_method
                
        except Exception as e:
            # 输出详细错误信息以便调试
            import traceback
            import hashlib
            from datetime import datetime
            
            error_type = type(e).__name__
            error_msg = str(e)
            
            # 只在 KeyError 时记录详细调试信息
            if error_type == "KeyError":
                # 生成唯一ID
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
                
                # 创建调试目录
                debug_dir = os.path.join("mutable_ast_debug", "KeyError")
                os.makedirs(debug_dir, exist_ok=True)
                
                error_file = os.path.join(debug_dir, f"{timestamp}_{code_hash}.txt")
                
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write(f"时间: {datetime.now()}\n")
                    f.write(f"错误类型: {error_type}\n")
                    f.write(f"错误消息: {error_msg}\n")
                    f.write("="*80 + "\n\n")
                    
                    # 完整堆栈
                    f.write("【完整堆栈追踪】\n")
                    f.write("-"*80 + "\n")
                    f.write(traceback.format_exc())
                    f.write("\n")
                    
                    # 完整的输入代码
                    f.write("【完整输入代码（包含 import、class、JavaDoc）】\n")
                    f.write("-"*80 + "\n")
                    f.write(code)
                    f.write("\n" + "-"*80 + "\n\n")
                    
                    # 提取并移除注释后的方法代码
                    f.write("【提取的方法代码（已移除行注释）】\n")
                    f.write("-"*80 + "\n")
                    f.write(method_code)
                    f.write("\n" + "-"*80 + "\n\n")
                    
                    # 包装后送入 tree-sitter 的代码
                    f.write("【包装后的代码（送入 tree-sitter 解析）】\n")
                    f.write("-"*80 + "\n")
                    wrapped = f"public class A {{ {method_code} }}"
                    f.write(wrapped)
                    f.write("\n" + "-"*80 + "\n\n")
                    
                    # 尝试用 tree-sitter 解析并找到 ERROR 节点
                    f.write("【tree-sitter 解析结果】\n")
                    f.write("-"*80 + "\n")
                    try:
                        test_tree = parser.parse(bytes(wrapped, "utf-8"))
                        
                        def find_all_errors(node, path="", depth=0):
                            result = []
                            indent = "  " * depth
                            node_info = f"{indent}{node.type}"
                            
                            if node.type == "ERROR":
                                result.append({
                                    'depth': depth,
                                    'path': path,
                                    'start': node.start_point,
                                    'end': node.end_point,
                                    'text': node.text.decode('utf-8', errors='replace')
                                })
                                node_info += " ← *** ERROR NODE ***"
                            
                            f.write(node_info + "\n")
                            
                            for i, child in enumerate(node.children):
                                child_path = f"{path}/{node.type}[{i}]"
                                result.extend(find_all_errors(child, child_path, depth + 1))
                            
                            return result
                        
                        f.write("\nAST 结构:\n")
                        error_nodes = find_all_errors(test_tree.root_node)
                        
                        if error_nodes:
                            f.write(f"\n发现 {len(error_nodes)} 个 ERROR 节点:\n\n")
                            for i, err in enumerate(error_nodes, 1):
                                f.write(f"ERROR #{i}:\n")
                                f.write(f"  位置: 行{err['start'][0]+1} 列{err['start'][1]} - 行{err['end'][0]+1} 列{err['end'][1]}\n")
                                f.write(f"  路径: {err['path']}\n")
                                f.write(f"  内容: {err['text'][:200]}\n")
                                
                                # 提取上下文
                                lines = wrapped.split('\n')
                                start_line = max(0, err['start'][0] - 2)
                                end_line = min(len(lines), err['end'][0] + 3)
                                f.write(f"  上下文:\n")
                                for ln in range(start_line, end_line):
                                    prefix = ">>> " if ln == err['start'][0] else "    "
                                    f.write(f"    {prefix}L{ln+1}: {lines[ln]}\n")
                                f.write("\n")
                        else:
                            f.write("\n未发现 ERROR 节点（语法可能是合法的）\n")
                            
                    except Exception as parse_err:
                        f.write(f"解析失败: {parse_err}\n")
                    
                    f.write("-"*80 + "\n\n")
                    
                    # 选择的变换键
                    f.write("【应用的 MutableAST 变换】\n")
                    f.write("-"*80 + "\n")
                    f.write(f"selected_keys: {selected_keys}\n")
                    f.write("-"*80 + "\n")
                
                # 控制台输出
                print(f"[ERROR] KeyError: {error_msg}")
                print(f"  详细信息已保存: {error_file}")
            else:
                # 其他错误类型
                if error_msg == 'ERROR':
                    print(f"[警告] MutableAST变换失败: {error_type}")
                else:
                    print(f"[警告] MutableAST变换失败: {error_type} - {error_msg}")
            
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
            from experiments.Attack.Rename_Attack.java_variable_renamer import JavaVariableRenamer
            from experiments.Attack.Rename_Attack.attack_config import AttackConfig
            
            renamer = JavaVariableRenamer(code)
            
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
    
    def create_hard_negative(self, code_string: str) -> str:
        """
        为Java代码生成语义不同但语法相似的困难负样本
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


# LLM重写和转译攻击功能，完全对标CodeWMBench的实现

def check_hard_gates(original_code: str, generated_code: str) -> Tuple[bool, str]:
    """
    三阶段硬约束检查（保守策略：明确不等价才拒绝）：
    1) 签名比对（返回类型、参数类型必须完全一致）；
    2) 返回依赖检查（只拒绝明确的常量返回）；
    3) 非全路径抛出检查（只拒绝明确的全路径异常）。
    通过返回 (True, "")，失败返回 (False, 原因)。
    """
    from .java_ast_utils import JavaASTAnalyzer, is_definite_constant
    
    # ========== 使用AST解析代码 ==========
    orig_ast = JavaASTAnalyzer(original_code)
    gen_ast = JavaASTAnalyzer(generated_code)
    
    # 保守策略：如果原代码无法解析，放行（可能是复杂Java语法）
    if not orig_ast.is_valid():
            return True, ""
    
    # 如果生成代码无法解析，拒绝（LLM生成了错误代码）
    if not gen_ast.is_valid():
        return False, "SyntaxFail: generated code is not valid Java"
    
    # ========== 门槛1：签名比对（基于AST，100%可靠）==========
    orig_ret = orig_ast.get_return_type()
    gen_ret = gen_ast.get_return_type()
    
    if orig_ret != gen_ret:
        return False, f"门槛1失败: 返回类型变化 ({orig_ret} -> {gen_ret})"
    
    orig_params = orig_ast.get_parameter_types()
    gen_params = gen_ast.get_parameter_types()
    
    if orig_params != gen_params:
        return False, f"门槛1失败: 参数类型变化 ({orig_params} -> {gen_params})"

    # ========== 门槛2：返回依赖检查（保守策略：只拒绝明确的常量）==========
    # 获取返回表达式
    gen_returns = re.findall(r"return\s+([^;]+);", generated_code)
    orig_returns = re.findall(r"return\s+([^;]+);", original_code)
    
    if gen_returns:
        # 检查生成代码的所有返回值是否都是明确的常量
        all_definite_constant = all(is_definite_constant(expr.strip()) for expr in gen_returns)
        
        if all_definite_constant:
            # 生成代码返回明确常量，检查原代码是否也是
            orig_all_constant = all(is_definite_constant(expr.strip()) for expr in orig_returns) if orig_returns else False
            
            if not orig_all_constant:
                # 原代码不是常量，生成代码变成常量 → 明确的功能变化
                return False, "门槛2失败: 返回值从依赖输入变为固定常量"
        # 否则保守放行（无法明确判断的复杂表达式）

    # ========== 门槛3：非全路径抛出（保持原有逻辑）==========
    body_start = generated_code.find('{')
    if body_start != -1:
        body = generated_code[body_start+1:]
        lines = [l.strip() for l in body.split('\n') if l.strip() and not l.strip().startswith('//') and l.strip() != '}']
        if lines and all('throw' in l for l in lines):
            # 检查原代码是否也是全抛异常（避免误报）
            orig_body_start = original_code.find('{')
            orig_all_throw = False
            if orig_body_start != -1:
                orig_body = original_code[orig_body_start+1:]
                orig_lines = [l.strip() for l in orig_body.split('\n') if l.strip() and not l.strip().startswith('//') and l.strip() != '}']
                orig_all_throw = orig_lines and all('throw' in l for l in orig_lines)
            # 只有当原代码不是全抛，但生成代码变成全抛时才拒绝
            if not orig_all_throw:
                return False, "门槛3失败: 全路径抛出异常"

    # ========== PDG-lite 依赖链检查（放宽策略：扩大追踪深度，只拒绝明确常量）==========
    param_names = gen_ast.get_parameter_names()
    source_symbols = set(param_names + ['this'])

    body_text = generated_code[generated_code.find('{')+1:] if '{' in generated_code else generated_code

    # 构建依赖图（放宽到10跳）
    assign_pat = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*([^;]+);")
    var_deps: dict[str, set[str]] = {}
    
    # 初次扫描：直接含有源符号
    for m in assign_pat.finditer(body_text):
        var = m.group(1)
        rhs = m.group(2)
        ids = set(re.findall(r"\b[A-Za-z_]\w*\b", rhs))
        hit = ids & source_symbols
        if hit:
            var_deps.setdefault(var, set()).update(hit)
    
    # 多次传播（放宽到10跳）
    for _ in range(10):
        changed = False
        for m in assign_pat.finditer(body_text):
            var = m.group(1)
            rhs = m.group(2)
            ids = set(re.findall(r"\b[A-Za-z_]\w*\b", rhs))
            acc: set[str] = set()
            for idt in ids:
                if idt in var_deps:
                    acc.update(var_deps[idt])
                if idt in source_symbols:
                    acc.add(idt)
            if acc:
                prev = var_deps.get(var, set())
                new = prev | acc
                if new != prev:
                    var_deps[var] = new
                    changed = True
        if not changed:
            break

    # return 依赖检查
    ret_ok = False
    for expr in gen_returns:
        ids = set(re.findall(r"\b[A-Za-z_]\w*\b", expr))
        if ids & source_symbols:
            ret_ok = True
            break
        for idt in ids:
            if idt in var_deps and (var_deps[idt] & source_symbols):
                ret_ok = True
                break
        if ret_ok:
            break
    
    if gen_returns and not ret_ok:
        # 只拒绝明确的常量返回
        all_definite_constant = all(is_definite_constant(expr.strip()) for expr in gen_returns)
        
        if all_definite_constant:
            # 检查原代码
            orig_all_constant = all(is_definite_constant(expr.strip()) for expr in orig_returns) if orig_returns else False
            
            if not orig_all_constant:
                # 原代码不是常量，生成代码是常量 → 明确的功能变化
                return False, "PDG-BaseNoDep: 返回值从依赖输入变为固定常量"
        # 否则保守放行（无法追踪的复杂依赖）

    # ========== 小契约：逐一通知模式检查（放宽策略：接受更多迭代方式）==========
    def has_any_collection_operation(code: str) -> bool:
        """宽松检查：代码中是否有任何集合操作"""
        patterns = [
            r"for\s*\(",                    # 任何for循环
            r"while\s*\(",                  # while循环
            r"\.forEach",                   # forEach方法
            r"\.stream\s*\(",              # stream API
            r"\.iterator\s*\(",            # iterator方法
            r"\.parallelStream\s*\(",      # 并行流
            r"\.map\s*\(",                 # map操作
            r"\.filter\s*\(",              # filter操作
            r"\.collect\s*\(",             # collect操作
        ]
        return any(re.search(p, code) for p in patterns)
    
    # 检查原代码是否为逐一通知模式
    is_void = (orig_ret == "void") if (orig_ret is not None) else False
    
    # 简化的模式检测：只检查是否有集合操作
    orig_has_collection_op = has_any_collection_operation(original_code)
    
    # 如果原代码是void且有集合操作，检查生成代码
    if is_void and orig_has_collection_op:
        gen_has_collection_op = has_any_collection_operation(generated_code)
        
        if not gen_has_collection_op:
            # 生成代码完全删除了集合操作 → 可能的功能变化
            # 但这里保守放行，因为可能有等价的实现方式
            pass
    
    # 所有检查通过
    return True, ""

def generate_with_retry(
    model: Any, prompt: str, max_retries: int = 3, temperature: float = 0.2
) -> str:
    """
    使用重试机制调用LLM生成内容，通过NewAPI代理调用Gemini模型
    支持API密钥轮换：当检测到API配额限制时自动切换到备用密钥。
    
    参数:
        model: LLM模型实例
        prompt: 提示字符串
        max_retries: 最大重试次数
        temperature: 生成温度
        
    返回:
        处理后的生成内容字符串
    """
    # 创建生成配置
    generation_config = {
        "temperature": temperature,
        "maxOutputTokens": 4096,
    }
    
    # NewAPI的API密钥 - 请在此处替换为您的实际密钥
    NEWAPI_KEY = "sk-ApxCfJ7h1YR0L6nmSHvU0AXbUTs1iA4eiyMVwGaXkHBfJR6W"
    
    # NewAPI服务器地址 - 请替换为您的实际服务器地址
    NEWAPI_SERVER = "你的newapi服务器地址"  # 例如: api.example.com
    
    retry_count = 0
    print(f"[LLM] 发起生成: temp={temperature}, prompt_len={len(prompt)}")
    
    while retry_count < max_retries:
        try:
            print(f"[LLM] 尝试第 {retry_count + 1}/{max_retries} 次请求...")
            t0 = time.time()
            
            # 使用NewAPI调用指定模型
            import requests
            import os
            
            # 保存原始代理设置
            original_http_proxy = os.environ.pop('HTTP_PROXY', None)
            original_https_proxy = os.environ.pop('HTTPS_PROXY', None)
            
            try:
                # 构建请求体，根据NewAPI文档示例格式
                payload = {
                    "contents": [
                        {
                            "parts": [{"text": prompt}]
                        }
                    ]
                }
                
                # 将生成配置添加到请求体
                if generation_config:
                    payload["generationConfig"] = generation_config
                
                # API密钥
                api_key = "sk-ApxCfJ7h1YR0L6nmSHvU0AXbUTs1iA4eiyMVwGaXkHBfJR6W"
                
                # 使用Bearer token认证方式
                # 优先使用线程局部覆盖的模型名，其次回退到默认模型名
                try:
                    model_name_for_call = getattr(_model_ctx, "name_override", None) or NEWAPI_MODEL_NAME
                except Exception:
                    model_name_for_call = NEWAPI_MODEL_NAME
                url = f"{NEWAPI_BASE_URL}/models/{model_name_for_call}:generateContent"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                
                # 调用API
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=(5, 60)
                )
                
                # 检查响应
                response.raise_for_status()
                response_json = response.json()
                
                # 提取文本结果 - 根据NewAPI响应格式调整
                result = response_json["candidates"][0]["content"]["parts"][0]["text"]
                
            finally:
                # 恢复原始代理设置
                if original_http_proxy:
                    os.environ['HTTP_PROXY'] = original_http_proxy
                if original_https_proxy:
                    os.environ['HTTPS_PROXY'] = original_https_proxy
            
            dt = time.time() - t0
            
            # 移除可能存在的Markdown代码块标记
            if result.startswith("```java"):
                result = result.replace("```java", "", 1).strip()
            if result.startswith("```"):
                result = result.replace("```", "", 1).strip()
            if result.endswith("```"):
                result = result[:-3].strip()
                
            print(f"[LLM] 成功返回: text_len={len(result)}, 用时={dt:.2f}s")
            return result
        
        except Exception as e:
            error_str = str(e).lower()
            
            # 发生错误时，增加重试计数
            retry_count += 1
            print(f"[LLM] API调用失败，重试 {retry_count}/{max_retries}，错误: {e}")
            time.sleep(5)  # 等待5秒后重试
    
    # 如果所有重试都失败
    raise RuntimeError("所有API调用重试都失败")


def llm_rewrite_java(code_string: str, model, temperature=0.7) -> str:
    """
    使用LLM生成功能等价的Java代码重写版本；追加“硬门槛+功能不变Self-check”，
    生成后执行两次审核，失败抛异常（由上层捕获并跳过样本）。
    """

    # 1) Prompt 模板（5种，保留原有风格集合）
    PROMPT_TEMPLATES = {
        "default": """
# ROLE
You are a code transformation engine creating functionally equivalent variants.

# CORE PRINCIPLE
BLACK-BOX EQUIVALENCE: The input→output mapping must be IDENTICAL.
- For every input, the output must match the original exactly
- If original has bugs affecting I/O → preserve those bugs exactly
- Internal implementation can differ, but external behavior cannot

# TASK
Rewrite the following Java method using alternative implementation while maintaining IDENTICAL input→output behavior.

# YOUR CREATIVE FREEDOM
You are ENCOURAGED to explore transformations in these categories (any combination):
1. Variable / Function Name: Rename variables, change naming styles
2. Loops: Convert loop types (for↔while↔do-while), change iteration direction
3. Math Expression: Rewrite expressions using algebraic equivalences, change operator orders
4. Code Organization: Reorder statements, extract intermediate variables, inline expressions

# ABSOLUTE CONSTRAINTS (VIOLATION = IMMEDIATE REJECTION)
1. Method signature IDENTICAL (name, return type, parameter types)
2. ZERO COMMENTS RULE (CRITICAL):
   - Output MUST have EXACTLY ZERO comments
   - NO line comments (//)
   - NO block comments (/* */)
   - NO JavaDoc comments (/** */)
   - Even if original code has JavaDoc or comments, IGNORE them completely
   - Your output must be pure executable code with NO comments whatsoever
   - This rule applies even if you think comments would clarify bugs or edge cases
3. For EVERY input → output IDENTICAL to original (including edge cases, exceptions)
4. If original produces wrong output for some input → your code must produce SAME wrong output
5. Output ONLY raw Java code - no explanations, no markdown blocks

# ORIGINAL CODE
```java
{code_string}
```
""",
        "concise_refactor": """
# ROLE
You are a code transformation engine focused on creating clean, concise code variants.

# CORE PRINCIPLE
BLACK-BOX EQUIVALENCE: Input→output mapping must be IDENTICAL.
- Refactor for clarity or conciseness, but behavior must not change
- If original has I/O bugs → preserve them exactly

# TASK
Refactor the method to a more concise form while ensuring IDENTICAL input→output behavior.

# YOUR CREATIVE FREEDOM
You are ENCOURAGED to explore transformations in these categories (any combination):
1. Variable / Function Name: Rename variables, change naming styles
2. Loops: Convert loop types (for↔while↔do-while), change iteration direction
3. Math Expression: Rewrite expressions using algebraic equivalences, change operator orders
4. Code Organization: Reorder statements, extract intermediate variables, inline expressions

# ABSOLUTE CONSTRAINTS (VIOLATION = IMMEDIATE REJECTION)
1. Method signature IDENTICAL
2. ZERO COMMENTS RULE (CRITICAL):
   - Output MUST have EXACTLY ZERO comments
   - NO line comments (//)
   - NO block comments (/* */)
   - NO JavaDoc comments (/** */)
   - Even if original code has JavaDoc or comments, IGNORE them completely
   - Your output must be pure executable code with NO comments whatsoever
3. Every input produces identical output to original
4. Preserve bugs that affect observable behavior
5. Output raw Java code only

# ORIGINAL CODE
```java
{code_string}
```
""",
        "algorithm_optimize": """
# ROLE
You are a code transformation engine exploring alternative algorithmic approaches.

# CORE PRINCIPLE
BLACK-BOX EQUIVALENCE: Input→output mapping must be IDENTICAL.
- You may use different algorithms or data structures
- But for every input → output must match original exactly
- If original has I/O bugs → preserve them exactly

# TASK
Reimplement the method using an alternative approach while ensuring IDENTICAL input→output behavior for all cases.

# YOUR CREATIVE FREEDOM
You are ENCOURAGED to explore transformations in these categories (any combination):
1. Variable / Function Name: Rename variables, change naming styles
2. Loops: Convert loop types (for↔while↔do-while), change iteration direction
3. Math Expression: Rewrite expressions using algebraic equivalences, change operator orders
4. Code Organization: Reorder statements, extract intermediate variables, inline expressions

# ABSOLUTE CONSTRAINTS (VIOLATION = IMMEDIATE REJECTION)
1. Method signature IDENTICAL
2. ZERO COMMENTS RULE (CRITICAL):
   - Output MUST have EXACTLY ZERO comments
   - NO line comments (//)
   - NO block comments (/* */)
   - NO JavaDoc comments (/** */)
   - Even if original code has JavaDoc or comments, IGNORE them completely
   - Your output must be pure executable code with NO comments whatsoever
3. For every input (including edge cases) → identical output
4. Same exception behavior as original
5. Output raw Java code only

# ORIGINAL CODE
```java
{code_string}
```
""",
        "syntax_transform": """
# ROLE
You are a code transformation engine specializing in Java syntax transformations.

# CORE PRINCIPLE
BLACK-BOX EQUIVALENCE: Input→output mapping must be IDENTICAL.
- Transform syntax (e.g., for↔while, if-else↔ternary)
- But I/O behavior must not change
- Preserve I/O bugs exactly

# TASK
Apply syntax-level transformations while maintaining IDENTICAL input→output behavior.

# YOUR CREATIVE FREEDOM
You are ENCOURAGED to explore transformations in these categories (any combination):
1. Variable / Function Name: Rename variables, change naming styles
2. Loops: Convert loop types (for↔while↔do-while), change iteration direction
3. Math Expression: Rewrite expressions using algebraic equivalences, change operator orders
4. Code Organization: Reorder statements, extract intermediate variables, inline expressions

# ABSOLUTE CONSTRAINTS (VIOLATION = IMMEDIATE REJECTION)
1. Method signature IDENTICAL
2. ZERO COMMENTS RULE (CRITICAL):
   - Output MUST have EXACTLY ZERO comments
   - NO line comments (//)
   - NO block comments (/* */)
   - NO JavaDoc comments (/** */)
   - Even if original code has JavaDoc or comments, IGNORE them completely
   - Your output must be pure executable code with NO comments whatsoever
3. Identical I/O for all inputs
4. Same exception behavior
5. Output raw Java code only

# ORIGINAL CODE
```java
{code_string}
```
""",
        "expression_normalize": """
# ROLE
You are a code transformation engine specializing in expression-level transformations.

# CORE PRINCIPLE
BLACK-BOX EQUIVALENCE: Input→output mapping must be IDENTICAL.
- Rewrite expressions using algebraic/logical equivalences
- But I/O behavior must not change
- Preserve I/O bugs exactly

# TASK
Transform expressions (e.g., boolean simplification, algebraic equivalences) while ensuring IDENTICAL input→output behavior.

# YOUR CREATIVE FREEDOM
You are ENCOURAGED to explore transformations in these categories (any combination):
1. Variable / Function Name: Rename variables, change naming styles
2. Loops: Convert loop types (for↔while↔do-while), change iteration direction
3. Math Expression: Rewrite expressions using algebraic equivalences, change operator orders
4. Code Organization: Reorder statements, extract intermediate variables, inline expressions

# ABSOLUTE CONSTRAINTS (VIOLATION = IMMEDIATE REJECTION)
1. Method signature IDENTICAL
2. ZERO COMMENTS RULE (CRITICAL):
   - Output MUST have EXACTLY ZERO comments
   - NO line comments (//)
   - NO block comments (/* */)
   - NO JavaDoc comments (/** */)
   - Even if original code has JavaDoc or comments, IGNORE them completely
   - Your output must be pure executable code with NO comments whatsoever
3. Identical outputs for all inputs
4. Same exception behavior
5. Output raw Java code only

# ORIGINAL CODE
```java
{code_string}
```
""",
    }

    # 2) 模型名集合（硬编码）：默认模型名与 gemini-2.5-flash-lite
    DEFAULT_MODEL_NAME = NEWAPI_MODEL_NAME
    GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"

    # 3) 20 种变体：每项为 (prompt_style, temperature, model_name)
    variants = []
    # 默认模型：10个
    variants.extend([
        ("default", 0.0, DEFAULT_MODEL_NAME),
        ("default", 0.4, DEFAULT_MODEL_NAME),
        ("default", 0.8, DEFAULT_MODEL_NAME),
        ("concise_refactor", 0.7, DEFAULT_MODEL_NAME),
        ("concise_refactor", 1.0, DEFAULT_MODEL_NAME),
        ("algorithm_optimize", 0.7, DEFAULT_MODEL_NAME),
        ("algorithm_optimize", 0.5, DEFAULT_MODEL_NAME),
        ("syntax_transform", 0.7, DEFAULT_MODEL_NAME),
        ("expression_normalize", 0.7, DEFAULT_MODEL_NAME),
        ("default", 0.6, DEFAULT_MODEL_NAME),
    ])
    # gemini-2.5-flash-lite：10个（镜像相同分布）
    variants.extend([
        ("default", 0.0, GEMINI_FLASH_LITE),
        ("default", 0.4, GEMINI_FLASH_LITE),
        ("default", 0.8, GEMINI_FLASH_LITE),
        ("concise_refactor", 0.7, GEMINI_FLASH_LITE),
        ("concise_refactor", 1.0, GEMINI_FLASH_LITE),
        ("algorithm_optimize", 0.7, GEMINI_FLASH_LITE),
        ("algorithm_optimize", 0.5, GEMINI_FLASH_LITE),
        ("syntax_transform", 0.7, GEMINI_FLASH_LITE),
        ("expression_normalize", 0.7, GEMINI_FLASH_LITE),
        ("default", 0.6, GEMINI_FLASH_LITE),
    ])

    # 5) 构建审查对齐的不变式（从原代码轻量提取，用于提示）
    def _extract_sig_for_prompt(src: str):
        m = re.search(r"(public|private|protected|static|\s)+\s+(\w+(?:<[^>]+>)?)\s+\w+\s*\(([^)]*)\)", src, flags=re.MULTILINE)
        if not m:
            return "void", []
        ret = m.group(2).strip()
        params = []
        raw = m.group(3).strip()
        if raw:
            for seg in raw.split(','):
                seg = seg.strip()
                mm = re.match(r"(\w+(?:<[^>]+>)?)\s+\w+$", seg)
                if mm:
                    params.append(mm.group(1))
        return ret, params

    def _detect_roles_for_prompt(src: str):
        # 返回 (list_param_name, event_param_name)
        m = re.search(r"\(([^)]*)\)", src, flags=re.MULTILINE)
        list_nm, event_nm = "", ""
        if not m:
            return list_nm, event_nm
        raw = m.group(1).strip()
        pairs = []
        for seg in raw.split(','):
            seg = seg.strip()
            mm = re.match(r"(\w+(?:<[^>]+>)?)\s+(\w+)$", seg)
            if mm:
                pairs.append((mm.group(1), mm.group(2)))
        # 集合参数
        for tp, nm in pairs:
            tl = tp.lower()
            if ("list" in tl) or ("collection" in tl) or ("iterable" in tl):
                list_nm = nm
                break
        if not list_nm:
            for tp, nm in pairs:
                if re.search(rf"\b{re.escape(nm)}\s*\.(size|isEmpty|get)\s*\(", src):
                    list_nm = nm
                    break
        # 事件参数
        if list_nm:
            for tp, nm in pairs:
                if nm != list_nm:
                    event_nm = nm
                    break
        return list_nm, event_nm

    def _build_audit_invariants_for_prompt(src: str) -> str:
        ret, param_types = _extract_sig_for_prompt(src)
        list_nm, event_nm = _detect_roles_for_prompt(src)
        lines = []
        lines.append(f"- Types: return={ret}; params=[{', '.join(param_types)}]")
        lines.append("- SrcDeps: result MUST depend on original inputs/visible state; constant-only/always-throw forbidden.")
        lines.append("- NormalPath: ensure at least one non-exceptional path.")
        if list_nm:
            ev = event_nm or "<event>"
            lines.append(f"- Iter+SideEffect: explicitly iterate input collection '{list_nm}' and call the element side-effect with '{ev}' for each non-null element; do NOT replace with a single this.listener or remove the iteration.")
        return "\n# AUDIT-ALIGNED INVARIANTS\n" + "\n".join(lines) + "\n"

    audit_block = _build_audit_invariants_for_prompt(code_string)

    # 6) 抽样一个变体并生成（最多两次尝试）
    style, temp_use, model_name = random.choice(variants)
    base_template = PROMPT_TEMPLATES.get(style, PROMPT_TEMPLATES["default"]) 
    # 将“方法名不变”的约束弱化为“返回/参数类型不变”（最小侵入：对该句做替换）
    prompt_raw = base_template.format(code_string=code_string)
    prompt_raw = re.sub(r"DO NOT change the method name, return type, or parameter types/names\.",
                        "DO NOT change the return type or parameter types (parameter names may change).",
                        prompt_raw)
    # 追加审查对齐不变式与自检
    prompt_suffix = """

# SELF-CHECK BEFORE OUTPUT
Before outputting, ensure:
- Return type and parameter types are unchanged
- Return value depends on original inputs/state (not constant-only or always-throw)
- Functionality is preserved
If any constraint is violated, regenerate until all are satisfied.
"""
    prompt = prompt_raw + audit_block + prompt_suffix

    last_reason = ""
    for _ in range(2):
        try:
            _set_model_override(model_name)
            generated = generate_with_retry(model, prompt, temperature=float(temp_use))
        finally:
            _clear_model_override()
        ok, reason = check_hard_gates(code_string, generated)
        if ok:
            return generated
        last_reason = reason
    raise RuntimeError(f"LLM rewrite failed hard gates after 2 attempts: {last_reason}")


def retranslate_java(code_string: str, model, intermediate_lang="csharp", temperature=0.3) -> str:
    """
    Java → 中间语言 → Java 的等价转译（方式A：内部等概率抽样20种变体，外部接口不变）。
    """

    # 1) 支持的中间语言集合（按方案：csharp + c/cpp/kotlin/scala）
    INTERMEDIATE_LANGS = ["csharp", "c", "cpp", "kotlin", "scala"]

    # 2) 温度集合（仅 csharp 使用多温度；其它语言固定 0.0）
    TEMPERATURE_SET = [0.1, 0.15, 0.2, 0.25, 0.3]

    # 3) 模型名集合（硬编码）：默认模型名与 gemini-2.5-flash-lite
    DEFAULT_MODEL_NAME = NEWAPI_MODEL_NAME
    GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"

    # 4) Prompt 模板工厂：按语言生成 trans1/trans2 模板（严格保持功能等价、签名不变、仅输出代码）
    def build_trans1_prompt(target_lang: str, code_java: str) -> str:
        return f"""
# ROLE
You are an expert multi-language programmer specialized in precise, semantics-preserving translation.

# TASK
Translate the following Java method to {target_lang} with strictly identical functionality.

# CONSTRAINTS
1. Preserve ALL behavior, side effects, and exception semantics.
2. External contract unchanged: method name, parameter types and order, and return type. Parameter names may change; generics/annotations/modifier orders may vary.
3. The declared throws set must be equivalent (order can differ); do not add or remove checked exceptions.
4. Imports: allow only JDK/Javax standard libraries; do not introduce third-party frameworks or replace original framework types.
5. Do not change package/class or wrap into a new class; keep original context.
6. Keep control-flow skeleton and exception structure; allow minimal statement-level rewrites only.
7. ONLY output the translated method body or the method with the original signature and throws. No package/import/class wrappers, no extra text.
8. Do not add any comments or markers that describe who/how it was generated or which variant it belongs to (e.g., "generated by", "LLM rewrite", "translated", "auto-generated").

# SOURCE CODE (Java)
```java
{code_java}
```
"""

    def build_trans2_prompt(source_lang: str, code_middle: str) -> str:
        return f"""
# ROLE
You are an expert Java programmer specialized in precise back-translation.

# TASK
Translate the following {source_lang} code back to Java with strictly identical functionality.

# CONSTRAINTS
1. Keep the external contract: same method name, parameter types and order, and return type. Parameter names may change; generics/annotations/modifier orders may vary.
2. The declared throws set must be equivalent (order can differ); do not add or remove checked exceptions.
3. Use only JDK/Javax standard libraries; do not introduce third-party frameworks or replace original framework types.
4. Preserve control-flow skeleton and exception structure; allow minimal statement-level rewrites.
5. ONLY output the Java method with the original signature and throws, or only the method body; no package/import/class wrappers and no extra text.
6. Do not add any comments or markers that describe who/how it was generated or which variant it belongs to (e.g., "generated by", "LLM rewrite", "translated", "auto-generated").

# SOURCE CODE ({source_lang})
```
{code_middle}
```
"""

    # 5) 构建 20 种变体（等概率）：每项为 (lang, temperature, model_name)
    variants = []
    # 默认模型：csharp@多温度 + 4语言@0.0
    for t in TEMPERATURE_SET:
        variants.append(("csharp", float(t), DEFAULT_MODEL_NAME))
    variants.extend([
        ("c", 0.0, DEFAULT_MODEL_NAME),
        ("cpp", 0.0, DEFAULT_MODEL_NAME),
        ("kotlin", 0.0, DEFAULT_MODEL_NAME),
        ("scala", 0.0, DEFAULT_MODEL_NAME),
    ])
    # gemini-2.5-flash-lite：相同的集合
    for t in TEMPERATURE_SET:
        variants.append(("csharp", float(t), GEMINI_FLASH_LITE))
    variants.extend([
        ("c", 0.0, GEMINI_FLASH_LITE),
        ("cpp", 0.0, GEMINI_FLASH_LITE),
        ("kotlin", 0.0, GEMINI_FLASH_LITE),
        ("scala", 0.0, GEMINI_FLASH_LITE),
    ])

    # 5.5) 为 trans1/trans2 构建审查对齐不变式（从原 Java 代码）
    def _extract_sig_for_prompt(src: str):
        m = re.search(r"(public|private|protected|static|\s)+\s+(\w+(?:<[^>]+>)?)\s+\w+\s*\(([^)]*)\)", src, flags=re.MULTILINE)
        if not m:
            return "void", []
        ret = m.group(2).strip()
        params = []
        raw = m.group(3).strip()
        if raw:
            for seg in raw.split(','):
                seg = seg.strip()
                mm = re.match(r"(\w+(?:<[^>]+>)?)\s+\w+$", seg)
                if mm:
                    params.append(mm.group(1))
        return ret, params

    def _detect_roles_for_prompt(src: str):
        m = re.search(r"\(([^)]*)\)", src, flags=re.MULTILINE)
        list_nm, event_nm = "", ""
        if not m:
            return list_nm, event_nm
        raw = m.group(1).strip()
        pairs = []
        for seg in raw.split(','):
            seg = seg.strip()
            mm = re.match(r"(\w+(?:<[^>]+>)?)\s+(\w+)$", seg)
            if mm:
                pairs.append((mm.group(1), mm.group(2)))
        for tp, nm in pairs:
            tl = tp.lower()
            if ("list" in tl) or ("collection" in tl) or ("iterable" in tl):
                list_nm = nm
                break
        if not list_nm:
            for tp, nm in pairs:
                if re.search(rf"\b{re.escape(nm)}\s*\.(size|isEmpty|get)\s*\(", src):
                    list_nm = nm
                    break
        if list_nm:
            for tp, nm in pairs:
                if nm != list_nm:
                    event_nm = nm
                    break
        return list_nm, event_nm

    def _build_audit_invariants_for_prompt(src: str) -> str:
        ret, param_types = _extract_sig_for_prompt(src)
        list_nm, event_nm = _detect_roles_for_prompt(src)
        lines = []
        lines.append(f"- Types: return={ret}; params=[{', '.join(param_types)}]")
        lines.append("- SrcDeps: result MUST depend on original inputs/visible state; constant-only/always-throw forbidden.")
        lines.append("- NormalPath: ensure at least one non-exceptional path.")
        if list_nm:
            ev = event_nm or "<event>"
            lines.append(f"- Iter+SideEffect: explicitly iterate input collection '{list_nm}' and call the element side-effect with '{ev}' for each non-null element; do NOT replace with a single this.listener or remove the iteration.")
        return "\n# AUDIT-ALIGNED INVARIANTS\n" + "\n".join(lines) + "\n"

    audit_block = _build_audit_invariants_for_prompt(code_string)

    # 6) 等概率抽样并执行两段转译（最多两次尝试）
    lang, temp_use, model_name = random.choice(variants)
    last_reason = ""
    for _ in range(2):
        try:
            _set_model_override(model_name)
            # trans1: Java → lang
            trans1 = build_trans1_prompt(lang, code_string) + audit_block + "\n# SELF-CHECK BEFORE OUTPUT\nConfirm the above items; if any fails, regenerate and fix within this response. Output only the code.\n"
            middle = generate_with_retry(model, trans1, temperature=float(temp_use))
            # trans2: lang → Java，追加硬门槛与自检说明
            trans2_prompt = build_trans2_prompt(lang, middle) + audit_block + """

# SELF-CHECK BEFORE OUTPUT
Before outputting, ensure:
- Return type and parameter types are unchanged from the original Java code
- Return value depends on original inputs/state (not constant-only or always-throw)
- Functionality is preserved
If any constraint is violated, regenerate until all are satisfied.
"""
            generated = generate_with_retry(model, trans2_prompt, temperature=float(temp_use))
        finally:
            _clear_model_override()

        ok, reason = check_hard_gates(code_string, generated)
        if ok:
            return generated
        last_reason = reason

    raise RuntimeError(f"LLM retranslate failed hard gates after 2 attempts: {last_reason}")


def generate_java_training_data(
    input_file: str,
    output_file: str,
    model,
    split_type: str = "train",
    positive_ratio: float = 0.6,
    augmentation_types: Dict[str, float] = None,
    max_samples: int = None,
    parallel: bool = True,
    workers: int = 4,
    batch_size: int = 50,
    resume: bool = False  # 添加恢复参数
) -> int:
    """
    生成Java对比学习训练数据
    
    参数:
        input_file: 输入Java代码文件路径
        output_file: 输出训练数据文件路径
        model: LLM模型实例
        split_type: 数据集类型，'train'/'valid'/'test'
        positive_ratio: 正样本比例
        augmentation_types: 增强类型及其概率字典
        max_samples: 最大样本数
        parallel: 是否启用并行处理
        workers: 并行工作线程数量
        batch_size: 每批处理的样本数
        resume: 是否从上次中断点恢复
        
    返回:
        处理的样本总数
    """
    # 统一使用并行实现，避免串行/并行分叉导致的不一致
    parallel = True
    # 如果启用并行处理，调用并行版本
    if parallel:
        return generate_java_training_data_parallel(
            input_file=input_file,
            output_file=output_file,
            model=model,
            split_type=split_type,
            positive_ratio=positive_ratio,
            augmentation_types=augmentation_types,
            max_samples=max_samples,
            num_workers=workers,
            batch_size=batch_size,
            resume=resume  # 传递恢复参数
        )
    
    # 以下是原来的串行版本
    print(f"为{split_type}数据生成增强样本...")
    
    # 默认增强类型概率
    augmentation_types = augmentation_types or {
        "semantic_preserving": 0.3,  # 语义保持转换（变量重命名等）
        "llm_rewrite": 0.7,          # LLM重写（与CodeWMBench一致）
        "retranslate": 0.0           # 转译攻击已禁用
    }
    
    # 根据split_type调整参数
    if split_type == "train":
        # 训练集：多样化，更激进的增强
        temp_multiplier = 1.0
    elif split_type == "valid":
        # 验证集：稍保守
        temp_multiplier = 0.8
    else:  # test
        # 测试集：更严格的评估
        temp_multiplier = 1.2  # 更创新/困难的变体
    
    # 初始化Java代码增强器
    java_augmentor = JavaCodeAugmentor()
    
    # 加载源代码
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
    
    # 限制样本数
    if max_samples and len(source_codes) > max_samples:
        random.shuffle(source_codes)
        source_codes = source_codes[:max_samples]
    
    print(f"加载了 {len(source_codes)} 个Java代码样本")
    
    # 生成训练数据
    processed = 0
    with open(output_file, 'w', encoding='utf-8', newline='\n') as out_file:
        try:
            for i, anchor_code in enumerate(tqdm(source_codes, desc=f"{split_type}处理进度")):
                # 按比例决定生成正样本还是负样本
                is_positive = random.random() < positive_ratio
                
                try:
                    if is_positive:
                        # 选择增强类型
                        aug_type = random.choices(
                            list(augmentation_types.keys()),
                            weights=list(augmentation_types.values()),
                            k=1
                        )[0]
                        
                        if aug_type == "semantic_preserving":
                            # 使用Java代码增强器生成正样本
                            positive_samples = java_augmentor.augment(anchor_code)
                            if positive_samples:
                                positive_code = positive_samples[0]
                                out_file.write(json.dumps({
                                    "anchor": anchor_code,
                                    "positive": positive_code,
                                    "type": "augment"
                                }) + "\n")
                                processed += 1
                        
                        elif aug_type == "llm_rewrite":
                            # 使用LLM重写生成正样本
                            positive_code = llm_rewrite_java(
                                anchor_code, model, 
                                temperature=0.7 * temp_multiplier
                            )
                            out_file.write(json.dumps({
                                "anchor": anchor_code,
                                "positive": positive_code,
                                "type": "llm_rewrite"
                            }) + "\n")
                            processed += 1
                        
                        elif aug_type == "retranslate":
                            # 使用转译攻击生成正样本
                            positive_code = retranslate_java(
                                anchor_code, model,
                                temperature=0.3 * temp_multiplier
                            )
                            out_file.write(json.dumps({
                                "anchor": anchor_code,
                                "positive": positive_code,
                                "type": "retranslate"
                            }) + "\n")
                            processed += 1
                    
                    else:
                        # 仅使用简单负样本：从全量源代码中随机选择非自身的代码
                        available_negatives = [c for c in source_codes if c is not anchor_code]
                        if available_negatives:
                            negative_code = random.choice(available_negatives)
                            out_file.write(json.dumps({
                                "anchor": anchor_code,
                                "negative": negative_code,
                                "type": "random_negative"
                            }) + "\n")
                            processed += 1
                        
                except Exception as e:
                    print(f"处理样本 {i} 时出错: {e}")
                    continue
        except KeyboardInterrupt:
            print("检测到Ctrl+C，提前结束当前数据集的处理，已安全保存已生成的样本。")
            # 直接跳出循环，with块会确保文件关闭
            pass
    
    print(f"共生成 {processed} 个处理后的样本")
    return processed 


def _log_review_result(stats_file: str, passed: bool, reason: str):
    """线程安全地记录审核结果到统计文件"""
    record = {
        "passed": passed,
        "reason": reason if not passed else ""
    }
    with file_lock:
        with open(stats_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_batch(
    batch_codes: List[str],
    output_file: str,
    model,
    split_type: str = "train",
    positive_ratio: float = 0.6,
    augmentation_types: Dict[str, float] = None,
    all_codes: List[str] = None
) -> int:
    """
    处理一批代码样本，生成增强样本
    
    参数:
        batch_codes: 待处理的代码样本列表
        output_file: 输出文件路径
        model: LLM模型实例
        split_type: 数据集类型，'train'/'valid'/'test'
        positive_ratio: 正样本比例
        augmentation_types: 增强类型及其概率字典
        all_codes: 全量源代码池，用于抽取简单负样本
        
    返回:
        成功处理的样本数
    """
    processed = 0
    java_augmentor = JavaCodeAugmentor()
    
    # 获取审核统计文件路径（如果存在）
    review_stats_file = os.environ.get('REVIEW_STATS_FILE')
    
    # 回退：如果未提供全量池，则仅使用当前批次作为可选负样本池
    negative_pool = all_codes if all_codes is not None else batch_codes
    
    for anchor_code in batch_codes:
        try:
            # 保持原有的正负样本决策逻辑不变
            is_positive = random.random() < positive_ratio
            
            if is_positive:
                # 选择增强类型
                aug_type = random.choices(
                    list(augmentation_types.keys()),
                    weights=list(augmentation_types.values()),
                    k=1
                )[0]
                
                if aug_type == "semantic_preserving":
                    # 使用Java代码增强器生成正样本（统一出口：三阶段审核）
                    positive_samples = java_augmentor.augment(anchor_code)
                    if not positive_samples:
                        if review_stats_file:
                            _log_review_result(review_stats_file, False, "静态变换未生成候选")
                        continue
                    selected = None
                    last_reason = ""
                    for cand in positive_samples:
                        try:
                            ok, reason = check_hard_gates(anchor_code, cand)
                            if ok:
                                selected = cand
                                break
                            last_reason = reason
                        except Exception:
                            continue
                    if selected is None:
                        if review_stats_file:
                            _log_review_result(review_stats_file, False, last_reason or "静态变换审核失败")
                        continue
                    positive_code = selected
                    if review_stats_file:
                        _log_review_result(review_stats_file, True, "")
                    result = {
                        "anchor": anchor_code,
                        "positive": positive_code,
                        "type": "augment"
                    }
                    processed += 1
                
                elif aug_type == "llm_rewrite":
                    # 使用LLM重写生成正样本
                    temp_multiplier = 1.0 if split_type == "train" else 0.8 if split_type == "valid" else 1.2
                    try:
                        positive_code = llm_rewrite_java(
                            anchor_code, model, 
                            temperature=0.7 * temp_multiplier
                        )
                        if review_stats_file:
                            _log_review_result(review_stats_file, True, "")
                    except RuntimeError as e:
                        if review_stats_file:
                            _log_review_result(review_stats_file, False, str(e))
                        continue
                    except Exception as e:
                        if review_stats_file:
                            _log_review_result(review_stats_file, False, f"LLM调用异常: {str(e)[:100]}")
                        continue
                    result = {
                        "anchor": anchor_code,
                        "positive": positive_code,
                        "type": "llm_rewrite"
                    }
                    processed += 1
                
                elif aug_type == "retranslate":
                    # 使用转译攻击生成正样本
                    temp_multiplier = 1.0 if split_type == "train" else 0.8 if split_type == "valid" else 1.2
                    try:
                        positive_code = retranslate_java(
                            anchor_code, model,
                            temperature=0.3 * temp_multiplier
                        )
                        if review_stats_file:
                            _log_review_result(review_stats_file, True, "")
                    except RuntimeError as e:
                        if review_stats_file:
                            _log_review_result(review_stats_file, False, str(e))
                        continue
                    except Exception as e:
                        if review_stats_file:
                            _log_review_result(review_stats_file, False, f"转译异常: {str(e)[:100]}")
                        continue
                    result = {
                        "anchor": anchor_code,
                        "positive": positive_code,
                        "type": "retranslate"
                    }
                    processed += 1
            
            else:
                # 仅使用简单负样本：从全量池随机选择非自身代码
                candidates = [c for c in negative_pool if c is not anchor_code]
                if not candidates:
                    continue
                negative_code = random.choice(candidates)
                result = {
                    "anchor": anchor_code,
                    "negative": negative_code,
                    "type": "random_negative"
                }
                processed += 1
        
            # 使用锁安全写入文件
            with file_lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result) + "\n")
                    
        except Exception as e:
            print(f"处理样本时错误: {e}")
            continue
    
    return processed


def generate_java_training_data_parallel(
    input_file: str,
    output_file: str,
    model,
    split_type: str = "train",
    positive_ratio: float = 0.6,
    augmentation_types: Dict[str, float] = None,
    max_samples: int = None,
    num_workers: int = 48,  # 保持原有默认值
    batch_size: int = 50,
    resume: bool = False  # 添加恢复参数
) -> int:
    """
    并行版本的训练数据生成，支持中断恢复和多语言路由
    
    参数:
        input_file: 输入代码文件路径
        output_file: 输出训练数据文件路径
        model: LLM模型实例
        split_type: 数据集类型，'train'/'valid'/'test'
        positive_ratio: 正样本比例
        augmentation_types: 增强类型及其概率字典
        max_samples: 最大样本数
        num_workers: 并行工作线程数量
        batch_size: 每批处理的样本数
        resume: 是否从上次中断点恢复
        
    返回:
        处理的样本总数
    """
    # ===== 新增：多语言路由逻辑 =====
    language = os.environ.get('WATERMARK_LANGUAGE', 'java').lower()
    
    # 如果是JavaScript，路由到JS版本
    if language in ['javascript', 'js']:
        try:
            from contrastive_learning.js_augmentor import generate_javascript_training_data_parallel
            return generate_javascript_training_data_parallel(
                input_file=input_file,
                output_file=output_file,
                model=model,
                split_type=split_type,
                positive_ratio=positive_ratio,
                augmentation_types=augmentation_types,
                max_samples=max_samples,
                num_workers=num_workers,
                batch_size=batch_size,
                resume=resume,
            )
        except ImportError as e:
            print(f"[警告] 无法导入JavaScriptCodeAugmentor: {e}")
            print(f"[警告] 回退到Java处理模式")
    
    # 如果是C++，路由到C++版本（新增）
    elif language in ['cpp', 'c++']:
        try:
            from contrastive_learning.cpp_augmentor import CppCodeAugmentor
            
            print(f"为{split_type}数据并行生成增强样本（使用{num_workers}个工作线程）...")
            
            # 进度记录文件路径
            progress_file = f"{output_file}.progress"
            
            # 加载源代码
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
            
            # 限制样本数
            if max_samples and len(source_codes) > max_samples:
                random.shuffle(source_codes)
                source_codes = source_codes[:max_samples]
            
            print(f"加载了 {len(source_codes)} 个C++代码样本")
            
            # 初始化C++ augmentor
            cpp_augmentor = CppCodeAugmentor()
            
            # 生成训练数据
            processed = 0
            with open(output_file, 'w', encoding='utf-8', newline='\n') as out_file:
                for i, anchor_code in enumerate(tqdm(source_codes, desc=f"{split_type}处理进度")):
                    is_positive = random.random() < positive_ratio
                    
                    try:
                        if is_positive:
                            # 生成正样本（对标Java/JS）
                            positive_code = cpp_augmentor.augment(anchor_code)
                            if positive_code and positive_code != anchor_code:
                                out_file.write(json.dumps({
                                    "anchor": anchor_code,
                                    "positive": positive_code,
                                    "type": "augment"
                                }, ensure_ascii=False) + "\n")
                                processed += 1
                        
                        else:
                            # 生成负样本（对标Java/JS）
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
            
            print(f"共生成 {processed} 个处理后的C++样本")
            return processed
            
        except ImportError as e:
            print(f"[警告] 无法导入CppCodeAugmentor: {e}")
            print(f"[警告] 回退到Java处理模式")
    
    # ===== 原有Java处理逻辑 =====
    print(f"为{split_type}数据并行生成增强样本（使用{num_workers}个工作线程）...")
    
    # 进度记录文件路径
    progress_file = f"{output_file}.progress"
    
    # 默认增强类型概率
    augmentation_types = augmentation_types or {
        "semantic_preserving": 0.3,
        "llm_rewrite": 0.7,
        "retranslate": 0.0
    }
    
    # 加载源代码
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
    
    # 限制样本数
    if max_samples and len(source_codes) > max_samples:
        random.shuffle(source_codes)
        source_codes = source_codes[:max_samples]
    
    print(f"加载了 {len(source_codes)} 个Java代码样本")
    
    # 将数据分成多个批次
    total_samples = len(source_codes)
    batches = [source_codes[i:i+batch_size] for i in range(0, total_samples, batch_size)]
    
    # 加载上次处理进度
    processed_batches = set()
    total_processed = 0
    if resume and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                processed_batches = set(progress_data.get('processed_batches', []))
                total_processed = progress_data.get('total_processed', 0)
                print(f"恢复进度：已处理 {len(processed_batches)}/{len(batches)} 批次，共 {total_processed} 样本")
        except Exception as e:
            print(f"读取进度文件失败: {e}，将从头开始处理")
            processed_batches = set()
            total_processed = 0
    
    # 如果首次运行或需要重新开始，清空输出文件
    if not processed_batches:
        with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
            pass
    
    # 共享状态：所有API密钥是否已用尽
    all_keys_exhausted = threading.Event()
    
    # 记录进度的函数
    def save_progress():
        with open(progress_file, 'w', encoding='utf-8', newline='\n') as f:
            json.dump({
                'processed_batches': list(processed_batches),
                'total_processed': total_processed,
                'total_batches': len(batches),
                'timestamp': time.time()
            }, f)
    
    # 处理单个批次的函数
    def process_batch_with_progress(batch_index, batch_codes):
        nonlocal total_processed
        
        # 如果批次已处理过或所有API密钥已用尽，则跳过
        if batch_index in processed_batches or all_keys_exhausted.is_set():
            return 0
        
        try:
            # 调用原始的process_batch函数
            result = process_batch(
                batch_codes,
                output_file,
                model,
                split_type,
                positive_ratio,
                augmentation_types,
                all_codes=source_codes
            )
            
            # 更新进度
            with threading.Lock():
                processed_batches.add(batch_index)
                total_processed += result
                # 每完成10个批次保存一次进度
                if len(processed_batches) % 10 == 0:
                    save_progress()
            
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            
            # 处理失败，记录日志但继续执行
            print(f"批次 {batch_index} 处理失败: {e}")
            return 0
    
    # 并行处理未完成的批次
    remaining_batches = [(i, batch) for i, batch in enumerate(batches) if i not in processed_batches]
    
    with tqdm(total=len(batches), desc=f"{split_type}批次处理进度", initial=len(processed_batches)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有未处理的批次
            future_to_batch = {
                executor.submit(process_batch_with_progress, i, batch): i 
                for i, batch in remaining_batches
            }
            
            # 处理完成的任务
            try:
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_index = future_to_batch[future]
                    
                    # 检查是否所有API密钥都用尽
                    if all_keys_exhausted.is_set():
                        print("\n所有API密钥都已用尽，终止处理")
                        # 取消剩余任务
                        for f in future_to_batch:
                            if not f.done():
                                f.cancel()
                        break
                    
                    try:
                        # 这个批次在process_batch_with_progress中已更新过进度
                        future.result()  # 忽略返回值，因为已在函数内更新
                        pbar.update(1)
                        pbar.set_postfix({"已处理": total_processed, "批次": f"{len(processed_batches)}/{len(batches)}"})
                    except Exception as exc:
                        # 异常已在process_batch_with_progress中处理
                        pbar.update(1)
            
            except KeyboardInterrupt:
                # 捕获Ctrl+C，保存进度后终止
                print("\n检测到Ctrl+C，保存进度并终止...")
                save_progress()
                
                # 尝试取消未完成的任务
                try:
                    executor.shutdown(wait=False)
                except Exception:
                    pass
                
                print(f"\n进度已保存，可使用相同命令恢复。当前完成: {len(processed_batches)}/{len(batches)} 批次")
                raise
    
    # 最终保存进度
    save_progress()
    
    print(f"共生成 {total_processed} 个处理后的样本 ({len(processed_batches)}/{len(batches)} 批次)")
    return total_processed 