#!/usr/bin/env python3
"""
Test script to verify DataValidator functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test DataValidator class
def test_data_validator():
    try:
        from utils import DataValidator
        print("✅ DataValidator imported successfully")
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        data = pd.DataFrame({
            'Open': np.random.randn(len(dates)) * 10 + 100,
            'High': np.random.randn(len(dates)) * 10 + 105,
            'Low': np.random.randn(len(dates)) * 10 + 95,
            'Close': np.random.randn(len(dates)) * 10 + 100,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        print(f"✅ Sample data created: {data.shape}")
        
        # Test DataValidator
        validator = DataValidator()
        print("✅ DataValidator instance created")
        
        # Test validate_stock_data method
        result = validator.validate_stock_data(data, "TEST")
        print("✅ validate_stock_data method works")
        print(f"   Valid: {result['valid']}")
        print(f"   Issues: {result['issues']}")
        print(f"   Stats: {result['stats']}")
        
        # Test validate_and_fix_data method
        fixed_data = validator.validate_and_fix_data(data, "TEST")
        print("✅ validate_and_fix_data method works")
        print(f"   Fixed data shape: {fixed_data.shape}")
        print(f"   Columns: {fixed_data.columns.tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing DataValidator...")
    success = test_data_validator()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)
