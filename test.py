"""
Test script to validate the stock prediction system.
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_collector import StockDataCollector, FeatureEngineer, DataPreprocessor
from lstm_model import LSTMStockPredictor
from utils import DataValidator, setup_logging
from config import MODEL_CONFIG, DATA_CONFIG

def test_data_collection():
    """Test data collection functionality"""
    print("🧪 Testing Data Collection...")
    
    collector = StockDataCollector()
    
    # Test data fetching
    data = collector.fetch_stock_data("AAPL", "1y")
    
    if data is not None and len(data) > 0:
        print(f"✅ Data collection successful: {len(data)} records")
        print(f"   Date range: {data.index.min()} to {data.index.max()}")
        return True
    else:
        print("❌ Data collection failed")
        return False

def test_feature_engineering():
    """Test feature engineering"""
    print("\n🧪 Testing Feature Engineering...")
    
    collector = StockDataCollector()
    engineer = FeatureEngineer()
    
    # Get sample data
    data = collector.fetch_stock_data("AAPL", "6mo")
    if data is None:
        print("❌ Cannot test feature engineering - no data")
        return False
    
    # Add technical indicators
    enhanced_data = engineer.add_technical_indicators(data)
    
    # Check if new features were added
    original_cols = len(data.columns)
    enhanced_cols = len(enhanced_data.columns)
    
    if enhanced_cols > original_cols:
        print(f"✅ Feature engineering successful: {enhanced_cols - original_cols} new features")
        print(f"   Original columns: {original_cols}, Enhanced columns: {enhanced_cols}")
        
        # Check for key indicators
        key_indicators = ['SMA_20', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
        missing_indicators = [ind for ind in key_indicators if ind not in enhanced_data.columns]
        
        if not missing_indicators:
            print("✅ All key technical indicators present")
            return True
        else:
            print(f"⚠️ Missing indicators: {missing_indicators}")
            return False
    else:
        print("❌ Feature engineering failed - no new features added")
        return False

def test_data_preprocessing():
    """Test data preprocessing"""
    print("\n🧪 Testing Data Preprocessing...")
    
    collector = StockDataCollector()
    engineer = FeatureEngineer()
    preprocessor = DataPreprocessor(lookback_days=30, forecast_days=5)
    
    # Get and enhance data
    data = collector.fetch_stock_data("AAPL", "1y")
    if data is None:
        print("❌ Cannot test preprocessing - no data")
        return False
    
    enhanced_data = engineer.add_technical_indicators(data)
    
    try:
        # Prepare data for training
        prepared_data = preprocessor.prepare_data_for_training(
            enhanced_data, DATA_CONFIG.feature_columns[:10]  # Use subset for testing
        )
        
        # Check data shapes
        X_train = prepared_data['X_train']
        y_train = prepared_data['y_train']
        
        print(f"✅ Data preprocessing successful")
        print(f"   Training data shape: {X_train.shape}")
        print(f"   Training target shape: {y_train.shape}")
        print(f"   Test data shape: {prepared_data['X_test'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return False

def test_model_creation():
    """Test LSTM model creation"""
    print("\n🧪 Testing Model Creation...")
    
    try:
        # Create model with test configuration
        test_config = MODEL_CONFIG.__dict__.copy()
        test_config['epochs'] = 2  # Minimal training for testing
        test_config['lookback_days'] = 30
        test_config['forecast_days'] = 5
        
        model = LSTMStockPredictor(test_config)
        
        # Test model building
        input_shape = (30, 10)  # 30 days, 10 features
        keras_model = model.build_model(input_shape)
        
        if keras_model is not None:
            print("✅ Model creation successful")
            print(f"   Model parameters: {keras_model.count_params():,}")
            return True
        else:
            print("❌ Model creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_end_to_end_training():
    """Test complete training pipeline"""
    print("\n🧪 Testing End-to-End Training...")
    
    try:
        # Initialize components
        collector = StockDataCollector()
        engineer = FeatureEngineer()
        validator = DataValidator()
        
        # Step 1: Data collection
        print("   📊 Collecting data...")
        data = collector.fetch_stock_data("AAPL", "2y")
        if data is None:
            raise ValueError("Failed to collect data")
        
        # Step 2: Data validation
        print("   🔍 Validating data...")
        validation = validator.validate_stock_data(data, "AAPL")
        if not validation['valid']:
            print(f"   ⚠️ Data validation issues: {validation['issues']}")
        
        # Step 3: Feature engineering
        print("   🔧 Engineering features...")
        enhanced_data = engineer.add_technical_indicators(data)
        
        # Step 4: Data preprocessing
        print("   ⚙️ Preprocessing data...")
        preprocessor = DataPreprocessor(lookback_days=30, forecast_days=3)
        
        # Use fewer features for faster testing
        test_features = ['Close', 'Volume', 'SMA_20', 'RSI', 'MACD']
        prepared_data = preprocessor.prepare_data_for_training(enhanced_data, test_features)
        
        # Step 5: Model training
        print("   🚀 Training model...")
        test_config = MODEL_CONFIG.__dict__.copy()
        test_config.update({
            'epochs': 3,  # Minimal for testing
            'lookback_days': 30,
            'forecast_days': 3,
            'lstm_units': [32, 16],  # Smaller for testing
            'batch_size': 16
        })
        
        model = LSTMStockPredictor(test_config)
        
        training_results = model.train(
            X_train=prepared_data['X_train'],
            y_train=prepared_data['y_train'],
            X_val=prepared_data['X_test'][:10],  # Small validation set
            y_val=prepared_data['y_test'][:10],
            model_path=None  # Don't save for testing
        )
        
        # Step 6: Test prediction
        print("   🔮 Testing predictions...")
        test_predictions = model.predict(prepared_data['X_test'][:5])
        
        if test_predictions is not None and len(test_predictions) > 0:
            print("✅ End-to-end training successful!")
            print(f"   Training metrics: {training_results['train_metrics']}")
            print(f"   Validation metrics: {training_results['val_metrics']}")
            print(f"   Prediction shape: {test_predictions.shape}")
            return True
        else:
            print("❌ Prediction failed")
            return False
            
    except Exception as e:
        print(f"❌ End-to-end training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_validation():
    """Test data validation functionality"""
    print("\n🧪 Testing Data Validation...")
    
    try:
        collector = StockDataCollector()
        validator = DataValidator()
        
        # Get test data
        data = collector.fetch_stock_data("AAPL", "1y")
        if data is None:
            print("❌ Cannot test validation - no data")
            return False
        
        # Validate data
        results = validator.validate_stock_data(data, "AAPL")
        
        print(f"✅ Data validation successful")
        print(f"   Valid: {results['valid']}")
        print(f"   Issues: {len(results['issues'])}")
        print(f"   Records: {results['stats']['records']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Stock Prediction System Tests")
    print("=" * 50)
    
    # Setup logging
    setup_logging(log_level="WARNING")  # Reduce log noise during testing
    
    tests = [
        ("Data Collection", test_data_collection),
        ("Feature Engineering", test_feature_engineering),
        ("Data Preprocessing", test_data_preprocessing),
        ("Model Creation", test_model_creation),
        ("Data Validation", test_data_validation),
        ("End-to-End Training", test_end_to_end_training)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if passed_test:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready for use.")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review and fix issues.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
