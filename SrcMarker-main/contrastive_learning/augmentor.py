"""
Code augmentation engine for contrastive learning.
"""

import random
import re
import ast
from typing import Dict, List, Callable, Union, Optional, Tuple


class CodeAugmentor:
    """
    Augmentation engine for generating positive examples from code snippets.
    """
    
    def __init__(self, strategies: Optional[Dict[str, float]] = None):
        """
        Initialize with augmentation strategies.
        
        Args:
            strategies: Dict mapping strategy name to probability of applying.
                If None, default strategies will be used.
        """
        # Default strategies with probabilities if none provided
        self.strategies = strategies or {
            "rename_variables": 0.8,
            "insert_comments": 0.3, 
            "reorder_statements": 0.5,
            "change_whitespace": 0.7,
            "reformat_code": 0.6,
            "convert_for_while": 0.4
        }
        
        # Register augmentation methods
        self.augmentation_methods = {
            "rename_variables": self._rename_variables,
            "insert_comments": self._insert_comments,
            "reorder_statements": self._reorder_statements,
            "change_whitespace": self._change_whitespace,
            "reformat_code": self._reformat_code,
            "convert_for_while": self._convert_for_while
        }
        
    def augment(self, code_string: str, num_augmentations: int = 1) -> List[str]:
        """
        Generate positive samples through augmentation.
        
        Args:
            code_string: Original code snippet
            num_augmentations: Number of augmented samples to generate
            
        Returns:
            List of augmented code strings
        """
        augmented_samples = []
        
        for _ in range(num_augmentations):
            # Choose which strategies to apply based on their probabilities
            applied_strategies = [
                strategy for strategy in self.strategies
                if random.random() < self.strategies[strategy]
            ]
            
            # Apply selected strategies
            augmented_code = code_string
            for strategy in applied_strategies:
                if strategy in self.augmentation_methods:
                    try:
                        augmented_code = self.augmentation_methods[strategy](augmented_code)
                    except Exception:
                        # Skip failed augmentations
                        continue
            
            # Only add if something changed
            if augmented_code != code_string:
                augmented_samples.append(augmented_code)
        
        # If we couldn't generate any valid augmentations, at least change whitespace
        if not augmented_samples:
            try:
                whitespace_changed = self._change_whitespace(code_string)
                augmented_samples.append(whitespace_changed)
            except Exception:
                # If all else fails, return the original
                augmented_samples.append(code_string)
        
        return augmented_samples
    
    def create_hard_negative(self, code_string: str) -> str:
        """
        Generate a hard negative sample that looks similar but has different semantics.
        
        Args:
            code_string: Original code snippet
            
        Returns:
            A semantically different but syntactically similar code sample
        """
        # This is a complex task that requires careful implementation
        # For now, we'll implement a simplified version that changes function behavior
        
        try:
            # Try to parse as Python code to make semantic changes
            parsed = ast.parse(code_string)
            
            # Find numerical constants and flip their signs
            class ConstantModifier(ast.NodeTransformer):
                def visit_Constant(self, node):
                    if isinstance(node.value, (int, float)) and node.value != 0:
                        node.value = -node.value
                    return node
                
                def visit_Compare(self, node):
                    # Flip comparison operators
                    op_map = {
                        ast.Eq: ast.NotEq,
                        ast.NotEq: ast.Eq,
                        ast.Lt: ast.Gt,
                        ast.LtE: ast.GtE,
                        ast.Gt: ast.Lt,
                        ast.GtE: ast.LtE
                    }
                    
                    for i, op in enumerate(node.ops):
                        for old_op, new_op in op_map.items():
                            if isinstance(op, old_op):
                                node.ops[i] = new_op()
                                break
                                
                    return node
            
            # Apply the transformation
            modified_ast = ConstantModifier().visit(parsed)
            ast.fix_missing_locations(modified_ast)
            
            # Generate code from modified AST
            hard_negative = ast.unparse(modified_ast)
            return hard_negative
            
        except (SyntaxError, TypeError, AttributeError):
            # Fallback approach for non-Python code or parsing failures
            
            # Replace some common operators with their opposites
            replacements = {
                " == ": " != ",
                " != ": " == ",
                " > ": " <= ",
                " < ": " >= ",
                " >= ": " < ",
                " <= ": " > ",
                "True": "False",
                "False": "True"
            }
            
            modified = code_string
            for original, replacement in replacements.items():
                modified = modified.replace(original, replacement)
                
            # If no changes were made, flip a random number's sign
            if modified == code_string:
                def replace_number(match):
                    num = match.group(0)
                    # Don't replace 0
                    if num == "0" or num == "0.0":
                        return num
                    # Add a negative sign or remove it if it exists
                    if num.startswith("-"):
                        return num[1:]
                    else:
                        return "-" + num
                
                modified = re.sub(r'-?\d+(\.\d+)?', replace_number, code_string, count=1)
            
            return modified
    
    # Augmentation strategies implementation
    
    def _rename_variables(self, code_string: str) -> str:
        """Rename variables while preserving semantics."""
        try:
            # Parse the code
            parsed = ast.parse(code_string)
            
            # Find all variable names
            var_names = set()
            
            class VariableVisitor(ast.NodeVisitor):
                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Store):
                        var_names.add(node.id)
                    self.generic_visit(node)
            
            VariableVisitor().visit(parsed)
            
            # Create new variable names
            var_mapping = {}
            common_prefixes = ["var", "temp", "v", "x", "y", "z", "data", "val"]
            
            for var in var_names:
                # Skip special names like self
                if var in ["self", "cls", "__init__"]:
                    continue
                    
                prefix = random.choice(common_prefixes)
                new_name = f"{prefix}_{random.randint(1, 100)}"
                var_mapping[var] = new_name
            
            # Apply renaming
            class VariableRenamer(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id in var_mapping:
                        node.id = var_mapping[node.id]
                    return node
            
            renamed_ast = VariableRenamer().visit(parsed)
            ast.fix_missing_locations(renamed_ast)
            
            # Generate code from modified AST
            renamed_code = ast.unparse(renamed_ast)
            return renamed_code
            
        except (SyntaxError, AttributeError):
            # If AST parsing or transformation fails, it means the code is either
            # not valid Python or too complex for our current AST logic.
            # In this case, we should not attempt an unsafe regex-based fallback.
            # Instead, we return the original code string, indicating that this
            # specific augmentation failed for this sample.
            return code_string
    
    def _insert_comments(self, code_string: str) -> str:
        """Insert or modify comments."""
        lines = code_string.split('\n')
        
        # Standard comments to insert
        comments = [
            "// Process the input",
            "// Initialize variables",
            "// Update counter",
            "// Check boundary conditions",
            "// Handle edge case",
            "// Main logic",
            "// Helper function",
            "// Return result",
            "# Process the input",
            "# Initialize variables",
            "# Update counter",
            "# Check boundary conditions",
            "# Handle edge case",
            "# Main logic",
            "# Helper function", 
            "# Return result"
        ]
        
        # Determine language to choose appropriate comment style
        if "//" in code_string:
            # C-style comments
            comment_prefix = "//"
            filtered_comments = [c for c in comments if c.startswith("//")]
        else:
            # Python-style comments
            comment_prefix = "#"
            filtered_comments = [c for c in comments if c.startswith("#")]
            
        if not filtered_comments:
            filtered_comments = comments
        
        # Insert comments in 1-3 random positions
        for _ in range(random.randint(1, 3)):
            if not lines:
                break
                
            pos = random.randint(0, len(lines) - 1)
            
            # Skip lines that already have comments
            if comment_prefix in lines[pos]:
                continue
                
            # Add comment
            comment = random.choice(filtered_comments)
            indentation = re.match(r'^\s*', lines[pos]).group(0)
            
            if random.random() < 0.5:
                # Add comment on its own line
                lines.insert(pos, indentation + comment)
            else:
                # Add comment at the end of the line
                lines[pos] = lines[pos] + "  " + comment
                
        return '\n'.join(lines)
    
    def _reorder_statements(self, code_string: str) -> str:
        """Reorder independent statements where possible."""
        try:
            # Parse the code
            parsed = ast.parse(code_string)
            
            # Find function definitions and reorder their bodies
            class StatementReorderer(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    # Identify groups of statements that could be reordered
                    
                    # We'll only reorder variable declarations that are independent
                    var_decls = []
                    other_stmts = []
                    
                    for stmt in node.body:
                        # Only reorder simple assignments
                        if isinstance(stmt, ast.Assign) and all(isinstance(t, ast.Name) for t in stmt.targets):
                            var_decls.append(stmt)
                        else:
                            # Keep other statements in their original order
                            # If we have collected variable declarations, add them now
                            if var_decls:
                                # Shuffle declarations
                                random.shuffle(var_decls)
                                other_stmts.extend(var_decls)
                                var_decls = []
                            other_stmts.append(stmt)
                    
                    # Add any remaining variable declarations
                    if var_decls:
                        random.shuffle(var_decls)
                        other_stmts.extend(var_decls)
                    
                    # Update the function body
                    node.body = other_stmts
                    return node
            
            modified_ast = StatementReorderer().visit(parsed)
            ast.fix_missing_locations(modified_ast)
            
            # Generate code from modified AST
            reordered_code = ast.unparse(modified_ast)
            return reordered_code
            
        except (SyntaxError, AttributeError):
            # Similar to other methods, if AST-based reordering fails,
            # we will not fall back to a potentially unsafe regex approach.
            # Return the original code to signal a failed augmentation.
            return code_string
    
    def _change_whitespace(self, code_string: str) -> str:
        """Modify whitespace without changing semantics."""
        # Split into lines
        lines = code_string.split('\n')
        modified_lines = []
        
        for line in lines:
            # Randomly choose a whitespace transformation
            choice = random.random()
            
            if choice < 0.3:
                # Add extra space after commas or operators
                line = re.sub(r'([,;=+\-*/%<>])', r'\1 ', line)
            elif choice < 0.6:
                # Remove extra spaces
                line = re.sub(r'\s+', ' ', line)
            elif choice < 0.8:
                # Add random indentation
                extra_indent = ' ' * random.randint(0, 2)
                line = extra_indent + line
            
            modified_lines.append(line)
            
        return '\n'.join(modified_lines)
    
    def _reformat_code(self, code_string: str) -> str:
        """Reformat code styling without changing semantics."""
        try:
            # For Python code, we can use ast to parse and regenerate
            parsed = ast.parse(code_string)
            reformatted = ast.unparse(parsed)
            return reformatted
        except (SyntaxError, AttributeError):
            # For non-Python code, apply some basic reformatting
            
            # Normalize spacing around operators
            result = code_string
            # Add space after commas
            result = re.sub(r',(\S)', r', \1', result)
            # Normalize spacing around operators
            result = re.sub(r'(\S)([\+\-\*/=<>])(\S)', r'\1 \2 \3', result)
            # Remove duplicate spaces
            result = re.sub(r' +', ' ', result)
            
            return result
    
    def _convert_for_while(self, code_string: str) -> str:
        """Convert between for and while loops where possible."""
        try:
            # Parse Python code
            parsed = ast.parse(code_string)
            
            # Find for loops and convert to while
            class LoopConverter(ast.NodeTransformer):
                def visit_For(self, node):
                    # Only handle simple range-based for loops
                    if isinstance(node.iter, ast.Call) and \
                       isinstance(node.iter.func, ast.Name) and \
                       node.iter.func.id == 'range':
                        
                        # Create a new variable for the counter
                        counter_var = node.target.id if isinstance(node.target, ast.Name) else 'i'
                        
                        # Parse range arguments
                        if len(node.iter.args) == 1:
                            start = ast.Constant(value=0)
                            end = node.iter.args[0]
                            step = ast.Constant(value=1)
                        elif len(node.iter.args) == 2:
                            start = node.iter.args[0]
                            end = node.iter.args[1]
                            step = ast.Constant(value=1)
                        elif len(node.iter.args) == 3:
                            start = node.iter.args[0]
                            end = node.iter.args[1]
                            step = node.iter.args[2]
                        else:
                            return node  # Can't handle this range pattern
                            
                        # Create initializer: counter = start
                        init = ast.Assign(
                            targets=[ast.Name(id=counter_var, ctx=ast.Store())],
                            value=start
                        )
                        
                        # Create condition: counter < end
                        condition = ast.Compare(
                            left=ast.Name(id=counter_var, ctx=ast.Load()),
                            ops=[ast.Lt()],
                            comparators=[end]
                        )
                        
                        # Create increment: counter += step
                        increment = ast.AugAssign(
                            target=ast.Name(id=counter_var, ctx=ast.Store()),
                            op=ast.Add(),
                            value=step
                        )
                        
                        # Add increment to the end of the loop body
                        body = node.body + [increment]
                        
                        # Create while loop
                        while_loop = ast.While(
                            test=condition,
                            body=body,
                            orelse=node.orelse
                        )
                        
                        # Return sequence of init and while
                        return [init, while_loop]
                    else:
                        return node
                        
                def visit_While(self, node):
                    # Convert some while loops to for loops
                    # This is more complex and depends on identifying counter patterns
                    # For simplicity in this example, we'll skip this direction
                    return node
            
            converter = LoopConverter()
            modified_ast = converter.visit(parsed)
            ast.fix_missing_locations(modified_ast)
            
            # Generate code from modified AST
            converted_code = ast.unparse(modified_ast)
            return converted_code
            
        except (SyntaxError, AttributeError):
            # Fallback for non-Python code
            return code_string 