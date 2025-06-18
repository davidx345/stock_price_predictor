#!/usr/bin/env python3
"""
Validation script to test the fixes for the stock predictor app
"""
import sys
import os
sys.path.append('./src')

def test_feature_engineering():
    """Test if feature engineering creates all required columns"""
    print("🧪 Testing Feature Engineering...")
    
    try:
        import pandas as pd
        import numpy as np
        from data_collector import FeatureEngineer
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.rand(100) * 100 + 100,
            'High': np.random.rand(100) * 100 + 110,
            'Low': np.random.rand(100) * 100 + 90,
            'Close': np.random.rand(100) * 100 + 100,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # Test feature engineering
        fe = FeatureEngineer()
        enhanced_data = fe.add_technical_indicators(sample_data)
        
        # Check required columns
        required_columns = ['SMA_20', 'EMA_12', 'RSI', 'MACD', 'MACD_signal', 
                          'BB_upper', 'BB_lower', 'volatility', 'price_change']
        
        missing_columns = [col for col in required_columns if col not in enhanced_data.columns]
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        else:
            print(f"✅ All required columns present: {required_columns}")
            return True
            
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        return False

def test_data_validator():
    """Test if data validator fixes missing columns"""
    print("🧪 Testing Data Validator...")
    
    try:
        import pandas as pd
        import numpy as np
        from utils import DataValidator
        
        # Create sample data with missing indicators
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.rand(100) * 100 + 100,
            'High': np.random.rand(100) * 100 + 110,
            'Low': np.random.rand(100) * 100 + 90,
            'Close': np.random.rand(100) * 100 + 100,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # Test data validator
        validator = DataValidator()
        fixed_data = validator.validate_and_fix_data(sample_data, "TEST")
        
        # Check required columns
        required_columns = ['SMA_20', 'EMA_12', 'RSI', 'MACD', 'MACD_signal', 
                          'BB_upper', 'BB_lower', 'volatility', 'price_change']
        
        missing_columns = [col for col in required_columns if col not in fixed_data.columns]
        
        if missing_columns:
            print(f"❌ Data validator failed to fix: {missing_columns}")
            return False
        else:
            print(f"✅ Data validator successfully fixed all columns")
            return True
            
    except Exception as e:
        print(f"❌ Data validator test failed: {e}")
        return False

def test_model_compatibility():
    """Test if model compatibility handler loads properly"""
    print("🧪 Testing Model Compatibility...")
    
    try:
        from model_compatibility import ModelCompatibilityHandler
        
        handler = ModelCompatibilityHandler()
        print("✅ Model compatibility handler loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🔍 Running Stock Predictor Validation Tests...")
    print("=" * 50)
    
    tests = [
        ("Feature Engineering", test_feature_engineering),
        ("Data Validator", test_data_validator),
        ("Model Compatibility", test_model_compatibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your fixes should work on Railway.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
