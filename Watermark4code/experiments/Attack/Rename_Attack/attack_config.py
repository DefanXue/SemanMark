"""Configuration for rename attacks."""

from dataclasses import dataclass
from typing import Literal

@dataclass
class AttackConfig:
    """Configuration for variable renaming attack.
    
    Args:
        naming_strategy: Strategy for generating new variable names
            - 'random': Random alphanumeric names (e.g., 'var_a3b2')
            - 'sequential': Sequential names (e.g., 'v0', 'v1', 'v2')
            - 'obfuscated': Obfuscated names (e.g., 'l', 'O', 'I')
        preserve_semantics: Whether to preserve code semantics (always True for watermark testing)
        seed: Random seed for reproducibility
        rename_ratio: Ratio of variables to rename (1.0 = rename all, 0.5 = rename half)
    """
    naming_strategy: Literal['random', 'sequential', 'obfuscated'] = 'random'
    preserve_semantics: bool = True
    seed: int = 42
    rename_ratio: float = 1.0

