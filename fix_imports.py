"""
Import fix script for Railway deployment
Fixes relative import issues that cause runtime errors
"""

import os
import sys
from pathlib import Path

def fix_imports_in_file(filepath):
    """Fix relative imports in a Python file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix common relative import patterns
        fixes = [
            ('from .model_compatibility import', 'try:\n    from .model_compatibility import'),
            ('from .lstm_model import', 'try:\n    from .lstm_model import'),
            ('from .data_collector import', 'try:\n    from .data_collector import'),
            ('from .utils import', 'try:\n    from .utils import'),
            ('from .visualizations import', 'try:\n    from .visualizations import'),
        ]
        
        for old_pattern, new_pattern in fixes:
            if old_pattern in content and 'try:' not in content.split(old_pattern)[0].split('\n')[-1]:
                # Only replace if not already wrapped in try-except
                content = content.replace(
                    old_pattern,
                    new_pattern
                )
                # Add except block after the import line
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if new_pattern.split('\n')[-1] in line and i + 1 < len(lines):
                        # Find the module name
                        module_name = line.split('import ')[-1].split(' as')[0].strip()
                        except_line = f"except ImportError:\n    from {module_name.replace('.', '')} import {module_name.split('.')[-1]}"
                        lines.insert(i + 1, except_line)
                        break
                content = '\n'.join(lines)
        
        # Write back if changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed imports in {filepath}")
            return True
        else:
            print(f"ℹ️  No fixes needed in {filepath}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False

def main():
    """Fix all Python files in src directory"""
    print("🔧 Fixing relative imports for Railway deployment...")
    
    src_dir = Path("src")
    if not src_dir.exists():
        print("❌ src directory not found!")
        return
    
    fixed_files = 0
    total_files = 0
    
    for py_file in src_dir.glob("*.py"):
        if py_file.name != "__init__.py":
            total_files += 1
            if fix_imports_in_file(py_file):
                fixed_files += 1
    
    print(f"\n📊 Summary: Fixed {fixed_files}/{total_files} files")
    print("🚀 Ready for Railway deployment!")

if __name__ == "__main__":
    main()
