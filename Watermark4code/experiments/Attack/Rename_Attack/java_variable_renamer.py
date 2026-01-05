"""Java variable renamer for testing watermark robustness against rename attacks."""

import re
import random
import string
from typing import List, Dict, Set
from .attack_config import AttackConfig


class JavaVariableRenamer:
    """Renames variables in Java code while preserving semantics."""
    
    # Java keywords that should not be renamed
    JAVA_KEYWORDS = {
        'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char',
        'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum',
        'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements',
        'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', 'package',
        'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp',
        'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient',
        'try', 'void', 'volatile', 'while', 'true', 'false', 'null'
    }
    
    # Common method names that should not be renamed
    COMMON_METHODS = {
        'main', 'toString', 'equals', 'hashCode', 'clone', 'finalize',
        'notify', 'notifyAll', 'wait', 'getClass'
    }
    
    def __init__(self, code: str):
        """Initialize renamer with Java code.
        
        Args:
            code: Java source code to process
        """
        self.code = code
        self.variables = self._extract_variables()
        self.rename_mapping = {}  # 用于存储重命名映射
        self._name_counter = 0    # 用于生成唯一名称
    
    def _extract_variables(self) -> Set[str]:
        """Extract variable names from Java code.
        
        Returns:
            Set of variable names found in the code
        """
        variables = set()
        
        # Pattern to match variable declarations
        # Matches: type varName = ...
        declaration_pattern = r'\b(?:int|long|short|byte|float|double|boolean|char|String|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
        
        # Pattern to match for-loop variables
        # Matches: for (type var : ...)
        for_pattern = r'\bfor\s*\(\s*(?:int|long|short|byte|float|double|boolean|char|String|var)?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[=:]'
        
        # Find all matches
        for match in re.finditer(declaration_pattern, self.code):
            var_name = match.group(1)
            if var_name not in self.JAVA_KEYWORDS and var_name not in self.COMMON_METHODS:
                variables.add(var_name)
        
        for match in re.finditer(for_pattern, self.code):
            var_name = match.group(1)
            if var_name not in self.JAVA_KEYWORDS and var_name not in self.COMMON_METHODS:
                variables.add(var_name)
        
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
            # 使用确定性hash（避免Python Hash Randomization）
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
            # 使用确定性hash
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
            Renamed Java code
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
            
            # 根据rename_ratio决定重命名哪些变量
            if 0.0 < rename_ratio < 1.0:
                # 部分重命名：使用seed确定性地选择变量
                import random
                rng = random.Random(config.seed)
                num_to_rename = max(1, int(len(sorted_vars) * rename_ratio))
                vars_to_rename = set(rng.sample(sorted_vars, num_to_rename))
            else:
                # 全部重命名或不重命名
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
        """Collect local variable names (for compatibility with java_augmentor).
        
        Returns:
            Dictionary mapping variable names to their types (type info not available, so use 'var')
        """
        return {var: 'var' for var in self.variables}
    
    def collect_parameters(self) -> Dict[str, str]:
        """Collect function parameter names (for compatibility with java_augmentor).
        
        Returns:
            Dictionary mapping parameter names to their types (simplified extraction)
        """
        parameters = {}
        # Pattern to match function parameters: methodName(type param1, type param2)
        param_pattern = r'\b(?:public|private|protected|static)?\s*(?:\w+)\s+\w+\s*\(([^)]*)\)'
        
        for match in re.finditer(param_pattern, self.code):
            params_str = match.group(1).strip()
            if params_str:
                # Split by comma and extract parameter names
                for param in params_str.split(','):
                    param = param.strip()
                    if param:
                        # Extract parameter name (last word)
                        parts = param.split()
                        if len(parts) >= 2:
                            param_name = parts[-1]
                            param_type = ' '.join(parts[:-1])
                            if param_name not in self.JAVA_KEYWORDS:
                                parameters[param_name] = param_type
        
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
        # 使用随机seed而不是固定的42，增加变化性
        dynamic_seed = random.randint(0, 1000000) + hash(old_name) % 1000000
        return self._generate_new_name(old_name, strategy, self._name_counter - 1, dynamic_seed)


def rename_variables(code: str, strategy: str = 'random', seed: int = 42) -> str:
    """Convenience function to rename variables in Java code.
    
    Args:
        code: Java source code
        strategy: Naming strategy ('random', 'sequential', 'obfuscated')
        seed: Random seed for reproducibility
        
    Returns:
        Renamed Java code
    """
    config = AttackConfig(naming_strategy=strategy, seed=seed)
    renamer = JavaVariableRenamer(code)
    return renamer.apply_renames(config)

