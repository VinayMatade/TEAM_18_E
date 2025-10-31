#!/usr/bin/env python3
"""
Script to automatically fix code issues like unused imports and formatting.
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Set, List, Dict, Tuple


class ImportFixer:
    """Fix unused imports and formatting issues."""
    
    def __init__(self):
        self.imports_to_remove = {}
    
    def analyze_file(self, file_path: str) -> Dict:
        """Analyze file for issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=file_path)
            
            # Find unused imports
            imports = {}
            used_names = set()
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        if alias.name == '*':
                            continue
                        name = alias.asname if alias.asname else alias.name
                        imports[name] = (node.lineno, node)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
            
            unused_imports = []
            for name, (line, node) in imports.items():
                if name not in used_names:
                    unused_imports.append((name, line, node))
            
            return {
                'unused_imports': unused_imports,
                'content': content
            }
        
        except Exception as e:
            return {'error': str(e), 'content': ''}
    
    def fix_file(self, file_path: str) -> bool:
        """Fix issues in a single file."""
        analysis = self.analyze_file(file_path)
        
        if 'error' in analysis:
            print(f"Error analyzing {file_path}: {analysis['error']}")
            return False
        
        content = analysis['content']
        lines = content.splitlines(keepends=True)
        
        # Remove unused imports
        lines_to_remove = set()
        for name, line_num, node in analysis['unused_imports']:
            lines_to_remove.add(line_num - 1)  # Convert to 0-based indexing
        
        # Remove lines in reverse order to maintain line numbers
        for line_idx in sorted(lines_to_remove, reverse=True):
            if line_idx < len(lines):
                del lines[line_idx]
        
        # Fix formatting issues
        fixed_lines = []
        blank_line_count = 0
        
        for line in lines:
            # Remove trailing whitespace
            fixed_line = line.rstrip() + '\n' if line.endswith('\n') else line.rstrip()
            
            # Handle multiple blank lines
            if fixed_line.strip() == '':
                blank_line_count += 1
                if blank_line_count <= 2:  # Allow up to 2 consecutive blank lines
                    fixed_lines.append(fixed_line)
            else:
                blank_line_count = 0
                fixed_lines.append(fixed_line)
        
        # Write back to file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
            return True
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return False


def main():
    """Main function."""
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        # Find all Python files
        files = []
        for root, dirs, filenames in os.walk('.'):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for filename in filenames:
                if filename.endswith('.py') and filename != 'fix_code_issues.py':
                    files.append(os.path.join(root, filename))
    
    fixer = ImportFixer()
    fixed_count = 0
    
    for file_path in sorted(files):
        print(f"Fixing {file_path}...")
        if fixer.fix_file(file_path):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files.")


if __name__ == "__main__":
    main()