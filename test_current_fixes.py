"""
Test script to validate the current fixes and identify any remaining issues
"""
import sys
import os
sys.path.append('./src')

def test_imports():
    """Test if all imports work correctly"""
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        print("✅ Basic imports successful")
        
        # Test our custom modules
        from data_collector import StockDataCollector, FeatureEngineer
        from lstm_model import LSTMStockPredictor
        from utils import ModelManager, DataValidator
        from visualizations import StockVisualizer
        from error_handler import ErrorHandler
        print("✅ Custom module imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_validation():
    """Test data validation functionality"""
    try:
        from utils import DataValidator
        import pandas as pd
        import numpy as np
        
        validator = DataValidator()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000, 1100, 1200]
        })
        
        # Test validation
        result = validator.validate_stock_data(sample_data, "TEST")
        print(f"✅ Validation result: {result}")
        
        # Check if stats are properly included
        if 'stats' in result:
            print(f"✅ Stats included: {result['stats']}")
        else:
            print("❌ Stats missing from validation result")
            
        return True
    except Exception as e:
        print(f"❌ Validation test error: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_feature_engineering():
    """Test feature engineering"""
    try:
        from data_collector import FeatureEngineer
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.randn(100) * 10 + 100,
            'High': np.random.randn(100) * 10 + 105,
            'Low': np.random.randn(100) * 10 + 95,
            'Close': np.random.randn(100) * 10 + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        engineer = FeatureEngineer()
        enhanced_data = engineer.add_technical_indicators(sample_data)
        
        print(f"✅ Feature engineering successful. Columns: {enhanced_data.columns.tolist()}")
        
        # Check for required indicators
        required_indicators = ['SMA_20', 'EMA_12', 'RSI', 'MACD', 'MACD_signal', 
                              'BB_upper', 'BB_lower', 'volatility', 'price_change']
        missing = [col for col in required_indicators if col not in enhanced_data.columns]
        
        if missing:
            print(f"❌ Missing indicators: {missing}")
        else:
            print("✅ All required indicators present")
            
        return len(missing) == 0
    except Exception as e:
        print(f"❌ Feature engineering test error: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_error_handling():
    """Test error handling functionality"""
    try:
        from error_handler import ErrorHandler
        
        handler = ErrorHandler()
        
        # Test different error types
        test_error = Exception("Test error message")
        
        data_error = handler.handle_data_error(test_error, "AAPL")
        model_error = handler.handle_model_error(test_error, "LSTM")
        validation_error = handler.handle_validation_error(test_error, "stock")
        
        print("✅ Error handling functionality working")
        print(f"Data error: {data_error['user_message']}")
        print(f"Model error: {model_error['user_message']}")
        print(f"Validation error: {validation_error['user_message']}")
        
        return True
    except Exception as e:
        print(f"❌ Error handling test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 Testing Stock Predictor Components...")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Validation Test", test_data_validation),
        ("Feature Engineering Test", test_feature_engineering),
        ("Error Handling Test", test_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
        print(f"{'✅ PASSED' if result else '❌ FAILED'}")
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application should be working correctly.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
