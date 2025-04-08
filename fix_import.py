#!/usr/bin/env python3
"""
Script to fix import statements in Python files to match the project structure
"""

import os
import re
import sys

def fix_imports_in_file(filepath):
    """Fix import statements in a file"""
    print(f"Processing {filepath}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Local imports that need to be fixed
    local_modules = [
        'data_service', 'model_service', 'signal_service', 'backtest_service',
        'strategy_manager', 'ninja_trader_client', 'data_store', 'monitoring',
        'portfolio_manager', 'trade_executor', 'multi_market_manager'
    ]
    
    # Map modules to their proper package
    module_packages = {
        'data_service': 'src.core',
        'model_service': 'src.core',
        'signal_service': 'src.core',
        'backtest_service': 'src.core',
        'strategy_manager': 'src.core',
        'ninja_trader_client': 'src.infrastructure',
        'data_store': 'src.infrastructure',
        'monitoring': 'src.infrastructure',
        'portfolio_manager': 'src.trading',
        'trade_executor': 'src.trading',
        'multi_market_manager': 'src.trading'
    }
    
    # Fix direct imports
    for module in local_modules:
        # Match: from module import X or import module
        pattern1 = fr'from\s+{module}\s+import'
        pattern2 = fr'import\s+{module}(?:\s+as\s+\w+)?$'
        
        package = module_packages.get(module)
        if package:
            # Replace with proper package
            content = re.sub(pattern1, f'from {package}.{module} import', content)
            content = re.sub(pattern2, f'import {package}.{module}', content)
    
    # Fix class imports
    patterns = {
        r'from (\w+) import (\w+Service|DataStore)': r'from src.{0}.\1 import \2',
        r'import (\w+) as': r'import src.{0}.\1 as',
    }
    
    for pattern, replace_template in patterns.items():
        matches = re.findall(pattern, content)
        for match in matches:
            if isinstance(match, tuple):
                module, cls = match
                package = module_packages.get(module)
                if package:
                    package_parts = package.split('.')
                    old = f"from {module} import {cls}"
                    new = f"from {package}.{module} import {cls}"
                    content = content.replace(old, new)
    
    # Write the changes back to the file
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Fixed imports in {filepath}")

def main():
    """Main function"""
    project_dir = 'smart-market-analyzer'
    
    # Find all Python files
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                fix_imports_in_file(filepath)
    
    print("All imports fixed!")

if __name__ == "__main__":
    main()
