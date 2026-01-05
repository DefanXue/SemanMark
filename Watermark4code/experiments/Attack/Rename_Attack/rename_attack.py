"""Single file rename attack for testing watermark extraction."""

import os
import json
from pathlib import Path
from typing import Dict, Any
from .java_variable_renamer import JavaVariableRenamer
from .attack_config import AttackConfig


def apply_rename_attack(
    watermarked_code: str,
    strategy: str = 'random',
    seed: int = 42
) -> str:
    """Apply rename attack to watermarked code.
    
    Args:
        watermarked_code: The watermarked Java code
        strategy: Naming strategy ('random', 'sequential', 'obfuscated')
        seed: Random seed for reproducibility
        
    Returns:
        Attacked (renamed) code
    """
    config = AttackConfig(naming_strategy=strategy, seed=seed)
    renamer = JavaVariableRenamer(watermarked_code)
    attacked_code = renamer.apply_renames(config)
    return attacked_code


def attack_single_injection(
    injection_dir: str,
    output_dir: str,
    strategy: str = 'random',
    seed: int = 42
) -> Dict[str, Any]:
    """Apply rename attack to a single injection result.
    
    Args:
        injection_dir: Directory containing injection results (e.g., run_0000)
        output_dir: Directory to save attack results
        strategy: Naming strategy
        seed: Random seed
        
    Returns:
        Dictionary containing attack results and metadata
    """
    injection_dir = Path(injection_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load watermarked code
    watermarked_path = injection_dir / "final.java"
    if not watermarked_path.exists():
        raise FileNotFoundError(f"Watermarked code not found: {watermarked_path}")
    
    with open(watermarked_path, 'r', encoding='utf-8') as f:
        watermarked_code = f.read()
    
    # Apply attack
    attacked_code = apply_rename_attack(watermarked_code, strategy, seed)
    
    # Save attacked code
    attacked_path = output_dir / "attacked.java"
    with open(attacked_path, 'w', encoding='utf-8') as f:
        f.write(attacked_code)
    
    # Load injection metadata
    final_json_path = injection_dir / "final.json"
    if final_json_path.exists():
        with open(final_json_path, 'r', encoding='utf-8') as f:
            injection_info = json.load(f)
    else:
        injection_info = {}
    
    # Create attack metadata
    attack_info = {
        'strategy': strategy,
        'seed': seed,
        'injection_dir': str(injection_dir),
        'attacked_code_path': str(attacked_path),
        'original_bits': injection_info.get('bits', 'unknown')
    }
    
    # Save attack metadata
    attack_info_path = output_dir / "attack_info.json"
    with open(attack_info_path, 'w', encoding='utf-8') as f:
        json.dump(attack_info, f, indent=2, ensure_ascii=False)
    
    return attack_info


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply rename attack to watermarked code')
    parser.add_argument('--injection_dir', type=str, required=True,
                       help='Directory containing injection results')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save attack results')
    parser.add_argument('--strategy', type=str, default='random',
                       choices=['random', 'sequential', 'obfuscated'],
                       help='Naming strategy for attack')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    result = attack_single_injection(
        args.injection_dir,
        args.output_dir,
        args.strategy,
        args.seed
    )
    
    print(f"Attack completed!")
    print(f"Strategy: {result['strategy']}")
    print(f"Original bits: {result['original_bits']}")
    print(f"Attacked code saved to: {result['attacked_code_path']}")


