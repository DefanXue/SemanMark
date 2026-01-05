# -*- coding: utf-8 -*-
"""JavaScript variable renamer for testing watermark robustness against rename attacks."""

import re
import random
import string
from typing import List, Dict, Set
from .attack_config import AttackConfig


class JavaScriptVariableRenamer:
    """Renames variables in JavaScript code while preserving semantics."""

    # JavaScript keywords that should not be renamed
    JS_KEYWORDS = {
        'abstract', 'arguments', 'await', 'boolean', 'break', 'byte', 'case', 'catch',
        'char', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do',
        'double', 'else', 'enum', 'eval', 'export', 'extends', 'false', 'final',
        'finally', 'float', 'for', 'function', 'goto', 'if', 'implements', 'import',
        'in', 'instanceof', 'int', 'interface', 'let', 'long', 'native', 'new',
        'null', 'package', 'private', 'protected', 'public', 'return', 'short',
        'static', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws',
        'transient', 'true', 'try', 'typeof', 'var', 'void', 'volatile', 'while',
        'with', 'yield'
    }

    # Common method names that should not be renamed
    COMMON_METHODS = {
        'constructor', 'toString', 'valueOf', 'hasOwnProperty', 'isPrototypeOf',
        'propertyIsEnumerable', 'toLocaleString', 'length', 'push', 'pop',
        'shift', 'unshift', 'splice', 'slice', 'concat', 'join', 'indexOf',
        'lastIndexOf', 'forEach', 'map', 'filter', 'reduce', 'find', 'findIndex',
        'some', 'every', 'includes', 'console', 'log', 'error', 'warn', 'info',
        'Math', 'Date', 'Array', 'Object', 'String', 'Number', 'Boolean',
        'RegExp', 'JSON', 'parse', 'stringify', 'setTimeout', 'setInterval',
        'clearTimeout', 'clearInterval', 'fetch', 'Promise', 'then', 'catch',
        'finally', 'async', 'await'
    }

    def __init__(self, code: str):
        """Initialize renamer with JavaScript code.

        Args:
            code: JavaScript source code to process
        """
        self.code = code
        self.variables = self._extract_variables()
        self.rename_mapping = {}  # Store rename mapping
        self._name_counter = 0    # Counter for unique names

    def _extract_variables(self) -> Set[str]:
        """Extract variable names from JavaScript code with per-pattern error handling.
        
        Key improvement for watermark robustness:
        - Each variable extraction pattern has independent try-except
        - Declaration variables can be extracted even if function params fail
        - Priority strategy: extract declaration variables first (like Java)

        Returns:
            Set of variable names found in the code
        """
        variables = set()

        # ===== Pattern 1: Variable declarations (most reliable, prioritized) =====
        try:
            declaration_pattern = r'\b(?:var|let|const)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\b'
            for match in re.finditer(declaration_pattern, self.code):
                var_name = match.group(1)
                if var_name not in self.JS_KEYWORDS and var_name not in self.COMMON_METHODS:
                    variables.add(var_name)
        except Exception as e:
            # Variable declaration extraction failed (rare), continue other patterns
            pass

        # ===== Pattern 2: Function parameters (split into two patterns for robustness) =====
        # Pattern 2a: function name(param1, param2)
        try:
            function_decl_pattern = r'\bfunction\s+\w*\s*\(([^)]*)\)'
            for match in re.finditer(function_decl_pattern, self.code):
                params_str = match.group(1)
                if params_str.strip():
                    # Extract parameter names from parameter string
                    for param in params_str.split(','):
                        param = param.strip()
                        if param and '=' not in param:  # Skip default parameters
                            param_match = re.search(r'[a-zA-Z_$][a-zA-Z0-9_$]*', param)
                            if param_match:
                                param_name = param_match.group(0)
                                if param_name not in self.JS_KEYWORDS and param_name not in self.COMMON_METHODS:
                                    variables.add(param_name)
        except Exception as e:
            # Function declaration parameter extraction failed
            pass

        # Pattern 2b: (...) => (arrow functions, allow spaces in =>)
        try:
            # More flexible arrow function pattern, allow = > format corrupted by watermark
            arrow_pattern_flexible = r'\(([^)]*?)\)\s*=\s*>'
            for match in re.finditer(arrow_pattern_flexible, self.code):
                params_str = match.group(1)
                if params_str.strip():
                    for param in params_str.split(','):
                        param = param.strip()
                        if param and '=' not in param:
                            param_match = re.search(r'[a-zA-Z_$][a-zA-Z0-9_$]*', param)
                            if param_match:
                                param_name = param_match.group(0)
                                if param_name not in self.JS_KEYWORDS and param_name not in self.COMMON_METHODS:
                                    variables.add(param_name)
        except Exception as e:
            # Arrow function parameter extraction failed
            pass

        # ===== Pattern 3: for loop variables (independent handling) =====
        try:
            for_pattern = r'\bfor\s*\(\s*(?:var|let|const)?\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s+(?:of|in)'
            for match in re.finditer(for_pattern, self.code):
                var_name = match.group(1)
                if var_name not in self.JS_KEYWORDS and var_name not in self.COMMON_METHODS:
                    variables.add(var_name)
        except Exception as e:
            # for loop variable extraction failed
            pass

        # ===== Pattern 4: catch block variables (independent handling) =====
        try:
            catch_pattern = r'\bcatch\s*\(\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\)'
            for match in re.finditer(catch_pattern, self.code):
                var_name = match.group(1)
                if var_name not in self.JS_KEYWORDS and var_name not in self.COMMON_METHODS:
                    variables.add(var_name)
        except Exception as e:
            # catch block variable extraction failed
            pass

        return variables

    def _generate_new_name(self, old_name: str, strategy: str, index: int, seed: int) -> str:
        """Generate a new variable name based on strategy.

        Args:
            old_name: Original variable name
            strategy: Naming strategy ('random', 'sequential', 'obfuscated')
            index: Index for sequential naming
            seed: Random seed

        Returns:
            New variable name
        """
        if strategy == 'random':
            # Use deterministic hash (avoid Python hash randomization)
            import hashlib
            deterministic_hash = int(hashlib.md5(old_name.encode()).hexdigest()[:8], 16)
            random.seed(seed + deterministic_hash)
            suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
            return f'var_{suffix}'

        elif strategy == 'sequential':
            return f'v{index}'

        elif strategy == 'obfuscated':
            # Use confusing characters: l (lowercase L), O (uppercase O), I (uppercase i)
            confusing_chars = ['l', 'O', 'I', 'll', 'lO', 'OI', 'Il']
            # Use deterministic hash
            import hashlib
            deterministic_hash = int(hashlib.md5(old_name.encode()).hexdigest()[:8], 16)
            random.seed(seed + deterministic_hash)
            return random.choice(confusing_chars) + str(index)

        else:
            return old_name

    def apply_renames(self, config: AttackConfig, rename_ratio: float = 1.0) -> str:
        """Apply variable renaming to the code.

        Args:
            config: Attack configuration
            rename_ratio: Ratio of variables to rename (0.0~1.0, default 1.0 = 100%)

        Returns:
            Renamed JavaScript code
        """
        if not self.variables:
            return self.code

        # Use existing rename_mapping if available, otherwise generate new one
        if self.rename_mapping:
            rename_map = self.rename_mapping
        else:
            # Create rename mapping
            rename_map: Dict[str, str] = {}
            sorted_vars = sorted(self.variables)

            # Determine which variables to rename based on rename_ratio
            if 0.0 < rename_ratio < 1.0:
                # Partial rename: use seed to deterministically select variables
                import random
                rng = random.Random(config.seed)
                num_to_rename = max(1, int(len(sorted_vars) * rename_ratio))
                vars_to_rename = set(rng.sample(sorted_vars, num_to_rename))
            else:
                # Rename all or none
                vars_to_rename = set(sorted_vars) if rename_ratio >= 1.0 else set()

            for idx, var in enumerate(sorted_vars):
                if var in vars_to_rename:
                    new_name = self._generate_new_name(
                        var,
                        config.naming_strategy,
                        idx,
                        config.seed
                    )
                    rename_map[var] = new_name

        # Apply renames
        renamed_code = self.code

        # Sort by length (descending) to avoid partial replacements
        # e.g., replace 'count' before 'c' to avoid 'c' -> 'v0' affecting 'count'
        for old_name in sorted(rename_map.keys(), key=len, reverse=True):
            new_name = rename_map[old_name]

            # Use word boundary to avoid partial matches
            # This ensures we only replace complete variable names
            pattern = r'\b' + re.escape(old_name) + r'\b'
            renamed_code = re.sub(pattern, new_name, renamed_code)

        return renamed_code

    def collect_local_variables(self) -> Dict[str, str]:
        """Collect local variable names (for compatibility with js_augmentor).

        Returns:
            Dictionary mapping variable names to their types (type info not available, so use 'var')
        """
        return {var: 'var' for var in self.variables}

    def collect_parameters(self) -> Dict[str, str]:
        """Collect function parameter names (for compatibility with js_augmentor).

        Returns:
            Dictionary mapping parameter names to their types (simplified extraction)
        """
        parameters = {}
        # Pattern to match function parameters: function name(param1, param2)
        # or arrow function: (param1, param2) =>
        function_patterns = [
            r'\bfunction\s+\w*\s*\(([^)]*)\)',  # function name(param1, param2)
            r'(?:\b(?:var|let|const)\s+\w+\s*=\s*)?\s*\(([^)]*)\)\s*=>'  # (param1, param2) =>
        ]

        for pattern in function_patterns:
            for match in re.finditer(pattern, self.code):
                params_str = match.group(1)
                if params_str.strip():
                    # Split by comma and extract parameter names
                    for param in params_str.split(','):
                        param = param.strip()
                        if param and '=' not in param:  # Skip default parameters for now
                            # Extract parameter name (handle destructuring simply)
                            param_match = re.search(r'[a-zA-Z_$][a-zA-Z0-9_$]*', param)
                            if param_match:
                                param_name = param_match.group(0)
                                if param_name not in self.JS_KEYWORDS and param_name not in self.COMMON_METHODS:
                                    parameters[param_name] = 'var'  # JavaScript doesn't have strong typing

        return parameters

    def generate_new_name(self, old_name: str, strategy: str = 'random') -> str:
        """Generate a new variable name (public interface for compatibility).

        Args:
            old_name: Original variable name
            strategy: Naming strategy

        Returns:
            New variable name
        """
        self._name_counter += 1
        # Use random seed instead of fixed 42, add variation
        dynamic_seed = random.randint(0, 1000000) + hash(old_name) % 1000000
        return self._generate_new_name(old_name, strategy, self._name_counter - 1, dynamic_seed)


def rename_variables(code: str, strategy: str = 'random', seed: int = 42) -> str:
    """Convenience function to rename variables in JavaScript code.

    Args:
        code: JavaScript source code
        strategy: Naming strategy ('random', 'sequential', 'obfuscated')
        seed: Random seed for reproducibility

    Returns:
        Renamed JavaScript code
    """
    config = AttackConfig(naming_strategy=strategy, seed=seed)
    renamer = JavaScriptVariableRenamer(code)
    return renamer.apply_renames(config)
