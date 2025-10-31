#!/usr/bin/env python3
"""
Script to check for unused imports and basic code quality issues.
"""

import ast
import os
import sys
from typing import Set, List, Dict, Tuple


class ImportChecker(ast.NodeVisitor):
    """AST visitor to check for unused imports."""

    def __init__(self):
        self.imports = {}  # name -> (module, line)
        self.used_names = set()
        self.current_line = 0

    def visit_Import(self, node):
        self.current_line = node.lineno
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports[name] = (alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.current_line = node.lineno
        for alias in node.names:
            if alias.name == '*':
                # Skip star imports
                continue
            name = alias.asname if alias.asname else alias.name
            module = f"{node.module}.{alias.name}" if node.module else alias.name
            self.imports[name] = (module, node.lineno)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            self.used_names.add(node.value.id)
        self.generic_visit(node)

    def get_unused_imports(self) -> List[Tuple[str, str, int]]:
        """Return list of (name, module, line) for unused imports."""
        unused = []
        for name, (module, line) in self.imports.items():
            if name not in self.used_names:
                unused.append((name, module, line))
        return unused


def check_file_imports(file_path: str) -> Dict:
    """Check a single Python file for unused imports."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=file_path)
        checker = ImportChecker()
        checker.visit(tree)

        unused_imports = checker.get_unused_imports()

        return {
            'file': file_path,
            'unused_imports': unused_imports,
            'total_imports': len(checker.imports),
            'used_imports': len(checker.imports) - len(unused_imports)
        }

    except Exception as e:
        return {
            'file': file_path,
            'error': str(e),
            'unused_imports': [],
            'total_imports': 0,
            'used_imports': 0
        }


def check_formatting_issues(file_path: str) -> Dict:
    """Check for basic formatting issues."""
    issues = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Check for multiple consecutive blank lines
        blank_line_count = 0
        for i, line in enumerate(lines, 1):
            if line.strip() == '':
                blank_line_count += 1
                if blank_line_count > 2:
                    issues.append(f"Line {i}: More than 2 consecutive blank lines")
            else:
                blank_line_count = 0

        # Check for trailing whitespace
        for i, line in enumerate(lines, 1):
            if line.rstrip() != line.rstrip('\n'):
                issues.append(f"Line {i}: Trailing whitespace")

        return {
            'file': file_path,
            'formatting_issues': issues
        }

    except Exception as e:
        return {
            'file': file_path,
            'error': str(e),
            'formatting_issues': []
        }


def main():
    """Main function to check all Python files."""
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        # Find all Python files
        files = []
        for root, dirs, filenames in os.walk('.'):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for filename in filenames:
                if filename.endswith('.py'):
                    files.append(os.path.join(root, filename))

    print("Checking Python files for unused imports and formatting issues...\n")

    total_unused = 0
    total_files = 0

    for file_path in sorted(files):
        total_files += 1

        # Check imports
        import_result = check_file_imports(file_path)

        # Check formatting
        format_result = check_formatting_issues(file_path)

        if import_result.get('error') or format_result.get('error'):
            print(f"❌ {file_path}")
            if import_result.get('error'):
                print(f"   Import check error: {import_result['error']}")
            if format_result.get('error'):
                print(f"   Format check error: {format_result['error']}")
            continue

        unused_imports = import_result['unused_imports']
        formatting_issues = format_result['formatting_issues']

        if unused_imports or formatting_issues:
            print(f"⚠️  {file_path}")

            if unused_imports:
                print(f"   Unused imports ({len(unused_imports)}):")
                for name, module, line in unused_imports:
                    print(f"     Line {line}: {name} (from {module})")
                total_unused += len(unused_imports)

            if formatting_issues:
                print(f"   Formatting issues ({len(formatting_issues)}):")
                for issue in formatting_issues[:5]:  # Show first 5 issues
                    print(f"     {issue}")
                if len(formatting_issues) > 5:
                    print(f"     ... and {len(formatting_issues) - 5} more")
        else:
            print(f"✅ {file_path}")

    print(f"\nSummary:")
    print(f"  Files checked: {total_files}")
    print(f"  Total unused imports: {total_unused}")


if __name__ == "__main__":
    main()