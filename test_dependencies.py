#!/usr/bin/env python3
"""
Dependency validation script for Railway deployment
Tests that all required packages can be imported without conflicts
"""

import sys
import importlib
import traceback

# List of critical packages to test
CRITICAL_PACKAGES = [
    'tensorflow',
    'numpy',
    'pandas', 
    'scikit-learn',
    'streamlit',
    'plotly',
    'matplotlib',
    'seaborn',
    'requests',
    'alpha_vantage',
    'ta',
    'lxml',
    'beautifulsoup4',
    'joblib',
    'dotenv'
]

def test_import(package_name):
    """Test if a package can be imported"""
    try:
        if package_name == 'scikit-learn':
            importlib.import_module('sklearn')
        elif package_name == 'alpha_vantage':
            importlib.import_module('alpha_vantage.timeseries')
        elif package_name == 'dotenv':
            importlib.import_module('dotenv')
        else:
            importlib.import_module(package_name)
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    """Run dependency validation"""
    print("🔍 Testing package imports for Railway deployment...")
    print("=" * 50)
    
    failed_imports = []
    
    for package in CRITICAL_PACKAGES:
        success, error = test_import(package)
        if success:
            print(f"✅ {package:20} - OK")
        else:
            print(f"❌ {package:20} - FAILED: {error}")
            failed_imports.append((package, error))
    
    print("=" * 50)
    
    if failed_imports:
        print(f"❌ {len(failed_imports)} packages failed to import:")
        for package, error in failed_imports:
            print(f"   - {package}: {error}")
        sys.exit(1)
    else:
        print("✅ All packages imported successfully!")
        print("🚀 Ready for Railway deployment!")
        
        # Test TensorFlow specifically
        print("\n🧪 Testing TensorFlow compatibility...")
        try:
            import tensorflow as tf
            print(f"   TensorFlow version: {tf.__version__}")
            
            import numpy as np
            print(f"   NumPy version: {np.__version__}")
            
            # Test basic TensorFlow operation
            x = tf.constant([1.0, 2.0, 3.0])
            y = tf.constant([4.0, 5.0, 6.0])
            z = tf.add(x, y)
            print(f"   TensorFlow test operation: {z.numpy()}")
            print("✅ TensorFlow is working correctly!")
            
        except Exception as e:
            print(f"❌ TensorFlow test failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
