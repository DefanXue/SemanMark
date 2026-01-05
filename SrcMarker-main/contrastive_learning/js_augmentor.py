#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JavaScript代码增强器，对标java_augmentor.py（仅语义保持变换）
用于生成对比学习的正样本和负样本。
只实现语义保持转换（变量重命名、添加注释等）
"""

import random
import re
import os
import json
import threading
import concurrent.futures
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple, Union

# 添加文件锁，确保多线程安全写入（对标Java）
file_lock = threading.Lock()

# 导入基础增强器（对标Java）
from contrastive_learning.augmentor import CodeAugmentor


class JavaScriptCodeAugmentor(CodeAugmentor):
    """
    针对JavaScript代码的增强器，完全对标JavaCodeAugmentor
    实现相同的语义保持转换（不包含LLM功能）
    """
    
    def __init__(self, strategies: Optional[Dict[str, float]] = None):
        """
        初始化JavaScript代码增强器（对标Java）
        
        参数:
            strategies: 增强策略字典，mapping策略名称到应用概率
        """
        # JavaScript特定的默认策略（基于测试结果优化）
        self.strategies = strategies or {
            # === MutableAST变换（测试真变换率35.3%，降低概率） ===
            "apply_mutable_ast": 0.7,         # 应用MutableAST结构变换
            
            # === 简单语义规则（对标Java） ===
            "add_redundant_parens": 0.8,      # 添加冗余括号
            "transform_boolean_literal": 0.6, # 布尔字面量等价替换
            "transform_zero_literal": 0.5,    # 零的等价表达式
            
            # === 完整重命名（对标Java） ===
            "full_variable_rename": 0.7,      # 完整变量重命名
            
            # === 新增通用规则（对标Java） ===
            "modify_imports": 0.4,            # 添加冗余导入
            "modify_blank_lines": 0.6,        # 添加/删除空行
        }
        
        # 注册JavaScript特定增强方法（对标Java）
        self.augmentation_methods = {
            "apply_mutable_ast": self._apply_mutable_ast_transforms,
            "add_redundant_parens": self._add_redundant_parentheses,
            "transform_boolean_literal": self._transform_boolean_literal,
            "transform_zero_literal": self._transform_zero_literal,
            "full_variable_rename": self._apply_full_variable_rename,
            "modify_imports": self._modify_imports,
            "modify_blank_lines": self._modify_blank_lines,
        }
        
        # 缓存可行的MutableAST变换（对标Java）
        self._feasible_cache = {}
    
    def _protect_string_literals(self, code: str):
        """保护字符串字面量，用占位符替换（对标Java，增加模板字符串）"""
        string_map = {}
        counter = 0
        
        # 正则匹配JavaScript字符串字面量（单引号、双引号、模板字符串）
        patterns = [
            r'"(?:[^"\\]|\\.)*"',      # 双引号字符串
            r"'(?:[^'\\]|\\.)*'",      # 单引号字符串
            r'`(?:[^`\\]|\\.)*`',      # 模板字符串
        ]
        
        def replace_func(match):
            nonlocal counter
            original_string = match.group(0)
            placeholder = f"__STRING_{counter}__"
            string_map[placeholder] = original_string
            counter += 1
            return placeholder
        
        protected_code = code
        for pattern in patterns:
            protected_code = re.sub(pattern, replace_func, protected_code)
        
        return protected_code, string_map
    
    def _restore_string_literals(self, code: str, string_map: dict) -> str:
        """恢复字符串字面量（对标Java）"""
        result = code
        for placeholder, original_string in string_map.items():
            result = result.replace(placeholder, original_string)
        return result
    
    def _rename_js_variables(self, code_string: str) -> str:
        """重命名JavaScript变量（对标Java的_rename_java_variables）"""
        # JavaScript关键字列表
        js_keywords = {
            "await", "break", "case", "catch", "class", "const", "continue", 
            "debugger", "default", "delete", "do", "else", "enum", "export", 
            "extends", "false", "finally", "for", "function", "if", "implements", 
            "import", "in", "instanceof", "interface", "let", "new", "null", 
            "package", "private", "protected", "public", "return", "static", 
            "super", "switch", "this", "throw", "try", "true", "typeof", 
            "var", "void", "while", "with", "yield", "undefined"
        }
        
        # 常见JS类名/对象，避免重命名
        common_types = {
            "String", "Number", "Boolean", "Object", "Array", "Function", 
            "Date", "Math", "JSON", "console", "window", "document",
            "Promise", "Error", "TypeError", "ReferenceError"
        }
        
        # 提取变量声明（不包括函数名）- 对标Java
        var_pattern = r'(?<!\.)(\b[a-zA-Z_$]\w*\b)(?!\s*\()'  # JS允许$开头
        
        # Step 1: 保护字符串字面量（对标Java）
        protected_code, string_map = self._protect_string_literals(code_string)
        
        # 识别所有标识符
        identifiers = set(re.findall(var_pattern, protected_code))
        
        # 排除JS关键字和常见类型
        identifiers = identifiers - js_keywords - common_types
        
        # 创建重命名映射（对标Java）
        var_mapping = {}
        prefix_options = ["var", "tmp", "arg", "param", "val", "obj", "item"]
        
        # 变量重命名（不重命名函数）
        for name in identifiers:
            # 跳过特殊名称
            if name in ["main", "args", "exports", "module", "require"]:
                continue
            
            # 生成新名称（对标Java）
            prefix = random.choice(prefix_options)
            new_name = f"{prefix}_{random.randint(1, 100)}"
            var_mapping[name] = new_name
        
        # Step 2: 应用重命名（对标Java）
        result = protected_code
        
        for old_name, new_name in var_mapping.items():
            result = re.sub(r'\b' + re.escape(old_name) + r'\b(?!\s*\()', new_name, result)
        
        # Step 3: 恢复字符串字面量（对标Java）
        result = self._restore_string_literals(result, string_map)
        
        return result
    
    def _insert_js_comments(self, code_string: str) -> str:
        """在JavaScript代码中插入注释（对标Java的_insert_java_comments）"""
        lines = code_string.split('\n')
        
        # JavaScript风格注释（对标Java）
        comments = [
            "// Process the input data",
            "// Initialize variables",
            "// Update counter",
            "// Check boundary conditions",
            "// Handle edge case",
            "// Main business logic",
            "// Helper function",
            "// Return the result",
            "// Validate parameters",
            "// Apply transformation",
            "// Parse input string",
            "// Calculate result",
            "// Check for null/undefined values"
        ]
        
        # 插入注释的次数（对标Java）
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
    
    def _change_js_braces_style(self, code_string: str) -> str:
        """修改JavaScript大括号样式（对标Java的_change_java_braces_style）"""
        # K&R风格: if (condition) {
        # Allman风格: if (condition)
        #            {
        
        # 模式1: 将K&R风格转换为Allman风格
        kr_to_allman = random.random() < 0.5
        
        if kr_to_allman:
            # 匹配行尾的左花括号，将其移到下一行
            pattern = r'(\b(?:if|for|while|switch|try|catch|else|function)\b.*?)\s*\{'
            
            def replace_kr_to_allman(match):
                return match.group(1) + '\n' + ' ' * (len(match.group(1)) - len(match.group(1).lstrip())) + '{'
            
            return re.sub(pattern, replace_kr_to_allman, code_string)
        else:
            # 匹配单独一行的左花括号，将其移到上一行末尾
            lines = code_string.split('\n')
            result = []
            i = 0
            
            while i < len(lines):
                if i > 0 and lines[i].strip() == '{' and any(keyword in lines[i-1] for keyword in ['if', 'for', 'while', 'switch', 'try', 'catch', 'else', 'function']):
                    # 将括号附加到前一行
                    result[-1] = result[-1] + ' {'
                else:
                    result.append(lines[i])
                i += 1
            
            return '\n'.join(result)
    
    def _add_redundant_parentheses(self, code_string: str) -> str:
        """为算术表达式添加冗余括号（对标Java）"""
        result = code_string
        
        # 安全的算术运算符（对标Java）
        patterns = [
            # 算术运算: a + b → (a + b)
            (r'(\w+)\s*\+\s*(\w+)(?!\+)', r'(\1 + \2)'),
            (r'(\w+)\s*-\s*(\w+)(?!>)', r'(\1 - \2)'),
            (r'(\w+)\s*\*\s*(\w+)', r'(\1 * \2)'),
        ]
        
        # 随机选择1-2个模式应用（对标Java）
        num_transforms = random.randint(1, min(2, len(patterns)))
        selected_patterns = random.sample(patterns, num_transforms)
        
        for pattern, replacement in selected_patterns:
            # 找到所有匹配，随机选择一个位置替换（对标Java）
            all_matches = list(re.finditer(pattern, result))
            if all_matches:
                match_to_replace = random.choice(all_matches)
                start_pos = match_to_replace.start()
                end_pos = match_to_replace.end()
                matched_text = result[start_pos:end_pos]
                replaced_text = re.sub(pattern, replacement, matched_text)
                result = result[:start_pos] + replaced_text + result[end_pos:]
        
        return result
        
    def _transform_boolean_literal(self, code_string: str) -> str:
        """将布尔字面量替换为等价表达式（对标Java）"""
        # 提取字符串字面量的位置（保护区域）- 对标Java，增加模板字符串
        protected_regions = []
        for match in re.finditer(r'"(?:[^"\\]|\\.)*"', code_string):
            protected_regions.append((match.start(), match.end()))
        for match in re.finditer(r"'(?:[^'\\]|\\.)*'", code_string):
            protected_regions.append((match.start(), match.end()))
        for match in re.finditer(r'`(?:[^`\\]|\\.)*`', code_string):
            protected_regions.append((match.start(), match.end()))
        
        def is_protected(pos):
            """检查位置是否在字符串字面量中（对标Java）"""
            for start, end in protected_regions:
                if start <= pos < end:
                    return True
            return False
        
        result = code_string
        
        # 收集所有未保护的true和false位置（对标Java）
        true_matches = [m for m in re.finditer(r'\btrue\b', code_string) if not is_protected(m.start())]
        false_matches = [m for m in re.finditer(r'\bfalse\b', code_string) if not is_protected(m.start())]
        
        # 随机选择要替换的true（对标Java）
        if true_matches:
            num_to_replace = random.randint(0, len(true_matches))
            selected_indices = random.sample(range(len(true_matches)), num_to_replace)
            # 从后往前替换，避免位置偏移
            for idx in sorted(selected_indices, reverse=True):
                match = true_matches[idx]
                result = result[:match.start()] + '!false' + result[match.end():]
        
        # 随机选择要替换的false（对标Java）
        false_matches = [m for m in re.finditer(r'\bfalse\b', result) if not is_protected(m.start())]
        if false_matches:
            num_to_replace = random.randint(0, len(false_matches))
            selected_indices = random.sample(range(len(false_matches)), num_to_replace)
            for idx in sorted(selected_indices, reverse=True):
                match = false_matches[idx]
                result = result[:match.start()] + '!true' + result[match.end():]
        
        return result
    
    def _transform_zero_literal(self, code_string: str) -> str:
        """将数字0替换为等价算术表达式（对标Java）"""
        # 匹配赋值语句中的 0（对标Java）
        pattern = r'=\s*0\s*([;,)])'
        
        all_matches = list(re.finditer(pattern, code_string))
        
        if not all_matches:
            return code_string
        
        result = code_string
        
        # 随机选择要替换的位置数量（对标Java）
        num_to_replace = random.randint(0, len(all_matches))
        if num_to_replace > 0:
            selected_indices = random.sample(range(len(all_matches)), num_to_replace)
            
            # 从后往前替换（对标Java）
            for idx in sorted(selected_indices, reverse=True):
                match = all_matches[idx]
                # 随机选择等价表达式（对标Java）
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
        """添加冗余但无害的导入语句（对标Java的_modify_imports）"""
        # JavaScript/ES6 常见导入
        common_imports = [
            "const fs = require('fs');",
            "const path = require('path');",
            "const util = require('util');",
            "const crypto = require('crypto');",
        ]
        
        # 找到import/require区域的结束位置（对标Java）
        import_section_end = 0
        last_import_match = None
        
        # 匹配ES6 import
        for match in re.finditer(r'^import\s+.*?;', code_string, re.MULTILINE):
            if match.end() > import_section_end:
                import_section_end = match.end()
                last_import_match = match
        
        # 匹配CommonJS require
        for match in re.finditer(r'^const\s+\w+\s*=\s*require\(.*?\);', code_string, re.MULTILINE):
            if match.end() > import_section_end:
                import_section_end = match.end()
                last_import_match = match
        
        # 如果存在import语句，随机添加1-2个冗余import（对标Java）
        if import_section_end > 0:
            num_to_add = random.randint(1, 2)
            selected_imports = random.sample(common_imports, min(num_to_add, len(common_imports)))
            
            # 在最后一个import之后插入
            insert_text = '\n' + '\n'.join(selected_imports)
            result = code_string[:import_section_end] + insert_text + code_string[import_section_end:]
            return result
        
        return code_string
    
    def _modify_blank_lines(self, code_string: str) -> str:
        """在函数之间、语句块之间添加或删除空行（对标Java）"""
        lines = code_string.split('\n')
        result_lines = []
        
        i = 0
        while i < len(lines):
            result_lines.append(lines[i])
            
            # 在大括号后随机添加空行（30%概率）- 对标Java
            if '{' in lines[i] and random.random() < 0.3:
                result_lines.append('')
            
            # 在return语句前随机添加空行（30%概率）- 对标Java
            if i < len(lines) - 1 and 'return' in lines[i+1] and random.random() < 0.3:
                result_lines.append('')
            
            # 在函数声明后随机添加空行（20%概率）- 对标Java
            if i < len(lines) - 1 and re.search(r'function\s+\w+\s*\(', lines[i]):
                if random.random() < 0.2:
                    result_lines.append('')
            
            i += 1
        
        return '\n'.join(result_lines)
    
    def _remove_line_comments(self, code: str) -> str:
        """
        移除JavaScript代码中的行注释（//）
        完全对标Java的_remove_line_comments
        """
        lines = code.split('\n')
        result_lines = []
        
        for line in lines:
            # 状态追踪
            in_string_double = False
            in_string_single = False
            in_template = False
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
                
                # 处理双引号字符串
                if char == '"' and not in_string_single and not in_template:
                    in_string_double = not in_string_double
                    i += 1
                    continue
                
                # 处理单引号字符串
                if char == "'" and not in_string_double and not in_template:
                    in_string_single = not in_string_single
                    i += 1
                    continue
                
                # 处理模板字符串
                if char == '`' and not in_string_double and not in_string_single:
                    in_template = not in_template
                    i += 1
                    continue
                
                # 检测行注释（必须在字符串之外）
                if not in_string_double and not in_string_single and not in_template:
                    if char == '/' and i + 1 < len(line) and line[i + 1] == '/':
                        comment_start = i
                        break
                
                i += 1
            
            # 移除注释部分
            if comment_start != -1:
                cleaned_line = line[:comment_start].rstrip()
                result_lines.append(cleaned_line)
            else:
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _extract_method_from_full_class(self, code: str) -> tuple:
        """
        从完整代码中提取函数体（对标Java）
        JavaScript通常不需要类包装，但为了API兼容性保留此方法
        
        返回: (method_code, metadata)
            method_code: 纯函数代码
            metadata: None（JS通常不需要包装）
        """
        # JavaScript代码通常已经是纯函数，直接移除行注释即可
        clean_code = self._remove_line_comments(code)
        return clean_code, None
    
    def _wrap_method_back_to_class(self, transformed_method: str, metadata: dict) -> str:
        """
        将变换后的方法重新包装回完整结构（对标Java）
        JavaScript通常不需要包装
        
        参数:
            transformed_method: 变换后的纯方法代码
            metadata: _extract_method_from_full_class 返回的元数据
        
        返回:
            完整的代码
        """
        if metadata is None:
            return transformed_method
        
        # 如果有metadata，进行包装（对标Java）
        parts = []
        if metadata.get('imports'):
            parts.append(metadata['imports'])
            parts.append('')
        
        parts.append('')
        if metadata.get('class_header'):
            parts.append(metadata['class_header'])
        parts.append(transformed_method)
        parts.append('}')
        
        return '\n'.join(parts)
    
    def _get_feasible_mutable_transforms(self, code: str) -> Dict[str, List[str]]:
        """
        动态检测代码片段的可行MutableAST变换（对标Java）
        
        注意：MutableAST可能不完全支持JavaScript
        如果不支持，返回空字典，增强器会自动跳过此策略
        
        参数:
            code: JavaScript代码片段
        
        返回:
            {transformer_name: [feasible_keys]}
        """
        import tree_sitter
        import sys
        
        # 先尝试提取方法
        method_code, _ = self._extract_method_from_full_class(code)
        
        # 检查缓存
        code_hash = hash(method_code)
        if code_hash in self._feasible_cache:
            return self._feasible_cache[code_hash]
        
        # 导入MutableAST（对标Java）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        srcmarker_root = os.path.dirname(current_dir)
        if srcmarker_root not in sys.path:
            sys.path.insert(0, srcmarker_root)
        
        try:
            import mutable_tree.transformers as ast_transformers
            from code_transform_provider import CodeTransformProvider
        except ImportError:
            # MutableAST不可用，返回空字典
            self._feasible_cache[code_hash] = {}
            return {}
        
        # 初始化transformers（对标Java的列表）
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
        
        # 初始化parser（使用JavaScript）
        parser = tree_sitter.Parser()
        try:
            parser_lang = tree_sitter.Language(
                os.path.join(srcmarker_root, "parser", "languages.so"), 
                "javascript"
            )
            parser.set_language(parser_lang)
        except Exception:
            # Parser初始化失败
            self._feasible_cache[code_hash] = {}
            return {}
        
        # 检查MutableAST是否支持JavaScript
        try:
            transform_provider = CodeTransformProvider("javascript", parser, transformers)
        except Exception:
            # 不支持JavaScript，返回空字典
            self._feasible_cache[code_hash] = {}
            return {}
        
        # 检测每个transformer的可行选项（对标Java）
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
                    continue
            
            # 确保至少有一个可行选项（对标Java）
            if not feasible_keys and available_keys:
                feasible_keys.append(available_keys[0])
            
            feasible_transforms[transformer_name] = feasible_keys
        
        # 缓存结果
        self._feasible_cache[code_hash] = feasible_transforms
        return feasible_transforms
    
    def _apply_mutable_ast_transforms(self, code: str) -> str:
        """
        应用MutableAST变换（对标Java的_apply_mutable_ast_transforms）
        如果MutableAST不支持JavaScript，会自动返回原代码
        
        参数:
            code: 原始代码
        
        返回:
            变换后的代码
        """
        import tree_sitter
        import sys
        
        # 提取方法（对标Java）
        method_code, metadata = self._extract_method_from_full_class(code)
        
        # 获取可行的transforms（对标Java）
        feasible_map = self._get_feasible_mutable_transforms(code)
        
        if not feasible_map:
            # MutableAST不可用或不支持JS，返回原代码
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
                "javascript"
            )
            parser.set_language(parser_lang)
        except Exception:
            return code
        
        try:
            transform_provider = CodeTransformProvider("javascript", parser, transformers)
        except Exception:
            # 不支持JavaScript
            return code
        
        # 从可行选项中随机选择组合（对标Java）
        selected_keys = []
        for transformer in transformers:
            transformer_name = transformer.name
            feasible_keys = feasible_map.get(transformer_name, [])
            if feasible_keys:
                selected_keys.append(random.choice(feasible_keys))
            else:
                available = transformer.get_available_transforms()
                selected_keys.append(available[0] if available else "")
        
        # 应用变换（对标Java）
        try:
            transformed_method = transform_provider.code_transform(method_code, selected_keys)
            
            # 重新包装
            if metadata is not None:
                return self._wrap_method_back_to_class(transformed_method, metadata)
            else:
                return transformed_method
                
        except Exception as e:
            # 变换失败，返回原代码（简化错误处理）
            return code
    
    def _apply_full_variable_rename(self, code: str, strategy: str = "random") -> str:
        """
        应用完整变量重命名（对标Java的_apply_full_variable_rename）
        
        注意：需要JS版本的JavaScriptVariableRenamer
        如果不存在，回退到简单重命名
        
        参数:
            code: 代码
            strategy: 重命名策略
        
        返回:
            重命名后的代码
        """
        import sys
        
        # 导入重命名器（对标Java）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xdf_root = os.path.dirname(os.path.dirname(current_dir))
        watermark_root = os.path.join(xdf_root, "Watermark4code")
        if watermark_root not in sys.path:
            sys.path.insert(0, watermark_root)
        
        try:
            # 尝试导入JS版本的重命名器
            from experiments.Attack.Rename_Attack.js_variable_renamer import JavaScriptVariableRenamer
            from experiments.Attack.Rename_Attack.attack_config import AttackConfig
            
            renamer = JavaScriptVariableRenamer(code)
            
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
        except Exception:
            # 重命名器不可用，回退到简单重命名
            return self._rename_js_variables(code)
    
    def create_hard_negative(self, code_string: str) -> str:
        """
        为JavaScript代码生成语义不同但语法相似的困难负样本（对标Java）
        """
        # 替换关键操作符（对标Java）
        replacements = {
            " === ": " !== ",
            " !== ": " === ",
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
        
        # 尝试替换操作符（对标Java）
        modified = code_string
        replacement_made = False
        
        for original, replacement in replacements.items():
            if original in modified:
                pos = modified.find(original)
                modified = modified[:pos] + replacement + modified[pos + len(original):]
                replacement_made = True
                break
        
        # 如果没有找到操作符，尝试修改数值常量（对标Java）
        if not replacement_made:
            num_pattern = r'\b(\d+)\b'
            matches = list(re.finditer(num_pattern, code_string))
            
            if matches:
                match = random.choice(matches)
                num = int(match.group(1))
                
                if num == 0:
                    new_num = 1
                elif num == 1:
                    new_num = 0
                else:
                    ops = [lambda x: -x, lambda x: x + 1, lambda x: x - 1, lambda x: x * 2]
                    new_num = random.choice(ops)(num)
                
                modified = code_string[:match.start()] + str(new_num) + code_string[match.end():]
                replacement_made = True
        
        # 如果还是没法修改，尝试注释掉一个重要语句（对标Java）
        if not replacement_made:
            lines = code_string.split('\n')
            for i, line in enumerate(lines):
                if ("return" in line or "=" in line or 
                    "if" in line or "for" in line or 
                    "while" in line) and ";" in line:
                    lines[i] = "// " + line
                    modified = '\n'.join(lines)
                    break
        
        return modified


# ==================== 数据生成功能（只保留语义保持变换） ====================

def generate_js_training_data(
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
    resume: bool = False
) -> int:
    """
    生成JavaScript对比学习训练数据（完全对标Java版本，只保留语义保持变换）
    """
    # ===== 新增：检测语言，如果是C++则路由到Java的生成函数 =====
    language = os.environ.get('WATERMARK_LANGUAGE', 'javascript').lower()
    
    if language in ['cpp', 'c++']:
        # 路由到Java的generate_java_training_data_parallel
        # （它会检测到语言是C++并调用相应逻辑）
        try:
            from contrastive_learning.java_augmentor import generate_java_training_data_parallel
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
                resume=resume,
            )
        except ImportError as e:
            print(f"[警告] 无法路由到Java的生成函数: {e}")
            print(f"[警告] 继续使用JavaScript处理")
    
    # 统一使用并行实现（对标Java）
    parallel = True
    if parallel:
        return generate_js_training_data_parallel(
            input_file=input_file,
            output_file=output_file,
            model=model,
            split_type=split_type,
            positive_ratio=positive_ratio,
            augmentation_types=augmentation_types,
            max_samples=max_samples,
            num_workers=workers,
            batch_size=batch_size,
            resume=resume
        )
    
    # 串行版本（对标Java）
    print(f"为{split_type}数据生成增强样本...")
    
    augmentation_types = augmentation_types or {
        "semantic_preserving": 1.0  # 只使用语义保持变换
    }
    
    # 初始化JS代码增强器
    js_augmentor = JavaScriptCodeAugmentor()
    
    # 加载源代码（对标Java）
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
    
    # 限制样本数（对标Java）
    if max_samples and len(source_codes) > max_samples:
        random.shuffle(source_codes)
        source_codes = source_codes[:max_samples]
    
    print(f"加载了 {len(source_codes)} 个JavaScript代码样本")
    
    # 生成训练数据（对标Java）
    processed = 0
    with open(output_file, 'w', encoding='utf-8', newline='\n') as out_file:
        try:
            for i, anchor_code in enumerate(tqdm(source_codes, desc=f"{split_type}处理进度")):
                is_positive = random.random() < positive_ratio
                
                try:
                    if is_positive:
                        # 只使用语义保持变换（对标Java配置）
                        positive_samples = js_augmentor.augment(anchor_code)
                        if positive_samples:
                            positive_code = positive_samples[0]
                            out_file.write(json.dumps({
                                "anchor": anchor_code,
                                "positive": positive_code,
                                "type": "augment"
                            }) + "\n")
                            processed += 1
                    
                    else:
                        # 简单负样本（对标Java）
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
            pass
    
    print(f"共生成 {processed} 个处理后的样本")
    return processed


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
    处理一批代码样本（完全对标Java版本，只保留语义保持变换）
    """
    processed = 0
    js_augmentor = JavaScriptCodeAugmentor()
    
    negative_pool = all_codes if all_codes is not None else batch_codes
    
    for anchor_code in batch_codes:
        try:
            is_positive = random.random() < positive_ratio
            
            if is_positive:
                # 只使用语义保持变换（对标Java配置）
                positive_samples = js_augmentor.augment(anchor_code)
                if not positive_samples:
                    continue
                
                positive_code = positive_samples[0]
                result = {
                    "anchor": anchor_code,
                    "positive": positive_code,
                    "type": "augment"
                }
                processed += 1
            
            else:
                # 简单负样本（对标Java）
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
        
            # 安全写入文件（对标Java）
            with file_lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result) + "\n")
                    
        except Exception as e:
            print(f"处理样本时错误: {e}")
            continue
    
    return processed


def generate_js_training_data_parallel(
    input_file: str,
    output_file: str,
    model,
    split_type: str = "train",
    positive_ratio: float = 0.6,
    augmentation_types: Dict[str, float] = None,
    max_samples: int = None,
    num_workers: int = 48,
    batch_size: int = 50,
    resume: bool = False
) -> int:
    """
    并行版本的JavaScript训练数据生成（完全对标Java版本，只保留语义保持变换）
    """
    print(f"为{split_type}数据并行生成增强样本（使用{num_workers}个工作线程）...")
    
    # 进度记录文件（对标Java）
    progress_file = f"{output_file}.progress"
    
    # 默认增强类型概率（对标Java，只使用语义保持）
    augmentation_types = augmentation_types or {
        "semantic_preserving": 1.0
    }
    
    # 加载源代码（对标Java）
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
    
    # 限制样本数（对标Java）
    if max_samples and len(source_codes) > max_samples:
        random.shuffle(source_codes)
        source_codes = source_codes[:max_samples]
    
    print(f"加载了 {len(source_codes)} 个JavaScript代码样本")
    
    # 分批（对标Java）
    total_samples = len(source_codes)
    batches = [source_codes[i:i+batch_size] for i in range(0, total_samples, batch_size)]
    
    # 加载进度（对标Java）
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
    
    # 如果首次运行，清空输出文件（对标Java）
    if not processed_batches:
        with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
            pass
    
    # 保存进度（对标Java）
    def save_progress():
        with open(progress_file, 'w', encoding='utf-8', newline='\n') as f:
            json.dump({
                'processed_batches': list(processed_batches),
                'total_processed': total_processed,
                'total_batches': len(batches),
                'timestamp': time.time()
            }, f)
    
    import time
    
    # 处理单个批次（对标Java）
    def process_batch_with_progress(batch_index, batch_codes):
        nonlocal total_processed
        
        if batch_index in processed_batches:
            return 0
        
        try:
            result = process_batch(
                batch_codes,
                output_file,
                model,
                split_type,
                positive_ratio,
                augmentation_types,
                all_codes=source_codes
            )
            
            # 更新进度（对标Java）
            with threading.Lock():
                processed_batches.add(batch_index)
                total_processed += result
                if len(processed_batches) % 10 == 0:
                    save_progress()
            
            return result
            
        except Exception as e:
            print(f"批次 {batch_index} 处理失败: {e}")
            return 0
    
    # 并行处理（对标Java）
    remaining_batches = [(i, batch) for i, batch in enumerate(batches) if i not in processed_batches]
    
    with tqdm(total=len(batches), desc=f"{split_type}批次处理进度", initial=len(processed_batches)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_batch = {
                executor.submit(process_batch_with_progress, i, batch): i 
                for i, batch in remaining_batches
            }
            
            try:
                for future in concurrent.futures.as_completed(future_to_batch):
                    try:
                        future.result()
                        pbar.update(1)
                        pbar.set_postfix({"已处理": total_processed, "批次": f"{len(processed_batches)}/{len(batches)}"})
                    except Exception:
                        pbar.update(1)
            
            except KeyboardInterrupt:
                print("\n检测到Ctrl+C，保存进度并终止...")
                save_progress()
                
                try:
                    executor.shutdown(wait=False)
                except Exception:
                    pass
                
                print(f"\n进度已保存，可使用相同命令恢复。当前完成: {len(processed_batches)}/{len(batches)} 批次")
                raise
    
    # 最终保存进度（对标Java）
    save_progress()
    
    print(f"共生成 {total_processed} 个处理后的样本 ({len(processed_batches)}/{len(batches)} 批次)")
    return total_processed


# ===== 添加别名函数（供plan.py路由使用） =====
def generate_javascript_training_data_parallel(
    input_file: str,
    output_file: str,
    model,
    split_type: str = "train",
    positive_ratio: float = 0.6,
    augmentation_types: Dict[str, float] = None,
    max_samples: int = None,
    num_workers: int = 48,
    batch_size: int = 50,
    resume: bool = False
) -> int:
    """
    generate_js_training_data_parallel的别名函数
    用于从java_augmentor.py的语言路由机制调用
    """
    return generate_js_training_data_parallel(
        input_file=input_file,
        output_file=output_file,
        model=model,
        split_type=split_type,
        positive_ratio=positive_ratio,
        augmentation_types=augmentation_types,
        max_samples=max_samples,
        num_workers=num_workers,
        batch_size=batch_size,
        resume=resume
    )