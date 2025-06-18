"""
Quick test script to validate imports and core functionality
"""
import sys
import os

def test_imports():
    """Test if all imports work correctly"""
    print("🧪 Testing imports...")
    
    # Add src to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    sys.path.append('./src')
    
    try:
        print("   Testing data_collector...")
        from data_collector import StockDataCollector, FeatureEngineer
        print("   ✅ data_collector imported successfully")
        
        print("   Testing utils...")
        from utils import ModelManager, DataValidator
        print("   ✅ utils imported successfully")
        
        print("   Testing DataValidator methods...")
        validator = DataValidator()
        
        # Test validate_stock_data method
        if hasattr(validator, 'validate_stock_data'):
            print("   ✅ validate_stock_data method found")
        else:
            print("   ❌ validate_stock_data method NOT found")
            
        # Test validate_and_fix_data method  
        if hasattr(validator, 'validate_and_fix_data'):
            print("   ✅ validate_and_fix_data method found")
        else:
            print("   ❌ validate_and_fix_data method NOT found")
        
        print("   Testing model imports...")
        from lstm_model import LSTMStockPredictor
        print("   ✅ lstm_model imported successfully")
        
        print("   Testing visualizations...")
        from visualizations import StockVisualizer
        print("   ✅ visualizations imported successfully")
        
        print("   Testing config...")
        from config import POPULAR_STOCKS, APP_CONFIG, MODEL_CONFIG, DATA_CONFIG
        print("   ✅ config imported successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import test failed: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_component_initialization():
    """Test component initialization"""
    print("\n🧪 Testing component initialization...")
    
    try:
        from data_collector import StockDataCollector, FeatureEngineer
        from lstm_model import LSTMStockPredictor
        from visualizations import StockVisualizer
        from utils import ModelManager, DataValidator
        
        # Test each component
        print("   Creating StockDataCollector...")
        data_collector = StockDataCollector()
        print("   ✅ StockDataCollector created")
        
        print("   Creating FeatureEngineer...")
        feature_engineer = FeatureEngineer()
        print("   ✅ FeatureEngineer created")
        
        print("   Creating StockVisualizer...")
        visualizer = StockVisualizer()
        print("   ✅ StockVisualizer created")
        
        print("   Creating ModelManager...")
        model_manager = ModelManager("models")
        print("   ✅ ModelManager created")
        
        print("   Creating DataValidator...")
        data_validator = DataValidator()
        print("   ✅ DataValidator created")
        
        # Test DataValidator methods
        print("   Testing DataValidator.validate_stock_data...")
        result = data_validator.validate_stock_data(None, "TEST")
        print(f"   ✅ validate_stock_data returned: {result}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Component initialization failed: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all tests"""
    print("🔍 Railway Deployment Test")
    print("=" * 40)
    
    # Test imports
    import_success = test_imports()
    
    # Test component initialization
    component_success = test_component_initialization()
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"   Imports: {'✅ PASS' if import_success else '❌ FAIL'}")
    print(f"   Components: {'✅ PASS' if component_success else '❌ FAIL'}")
    
    if import_success and component_success:
        print("\n🎉 All tests passed! Ready for Railway deployment.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
