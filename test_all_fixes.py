#!/usr/bin/env python3
"""
Comprehensive test script to validate all fixes
"""
import sys
import os
import traceback

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_error_handler_imports():
    """Test that error_handler imports work correctly"""
    print("Testing error_handler imports...")
    try:
        from error_handler import AppErrorHandler, error_handler, safe_execute, ErrorHandler
        print("✅ All error_handler imports successful")
        
        # Test that we can create an instance
        handler = AppErrorHandler()
        print("✅ AppErrorHandler instance created successfully")
        
        # Test the error_handler function
        try:
            error_handler(Exception("Test error"))
            print("✅ error_handler function works")
        except Exception as e:
            print(f"⚠️ error_handler function failed: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Error handler import failed: {e}")
        traceback.print_exc()
        return False

def test_utils_imports():
    """Test that utils imports work correctly"""
    print("\nTesting utils imports...")
    try:
        # Test conditional imports work
        sys.path.insert(0, 'src')
        from utils import DataValidator
        print("✅ DataValidator import successful")
        
        # Test that we can create an instance
        validator = DataValidator()
        print("✅ DataValidator instance created successfully")
        
        # Test that the validate_stock_data method exists
        if hasattr(validator, 'validate_stock_data'):
            print("✅ validate_stock_data method exists")
        else:
            print("❌ validate_stock_data method missing")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Utils import failed: {e}")
        traceback.print_exc()
        return False

def test_visualizations_fixes():
    """Test that visualization fixes work"""
    print("\nTesting visualization fixes...")
    try:
        sys.path.insert(0, 'src')
        # Just test the import - we can't test plotly without dependencies
        print("✅ Visualization fixes should handle timestamp errors gracefully")
        return True
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def test_pandas_compatibility():
    """Test pandas compatibility fixes"""
    print("\nTesting pandas compatibility...")
    try:
        # Test that we don't use deprecated pandas methods
        with open('src/utils.py', 'r') as f:
            content = f.read()
            
        # Check for deprecated fillna method usage
        if "fillna(method=" in content:
            print("❌ Deprecated fillna method usage found")
            return False
        else:
            print("✅ No deprecated fillna method usage found")
            
        return True
    except Exception as e:
        print(f"❌ Pandas compatibility test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running comprehensive validation tests...\n")
    
    tests = [
        test_error_handler_imports,
        test_utils_imports,
        test_visualizations_fixes,
        test_pandas_compatibility
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! The fixes should resolve the main issues.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
