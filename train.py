"""
Training pipeline for stock price prediction models.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from data_collector import StockDataCollector, FeatureEngineer, DataPreprocessor
from lstm_model import LSTMStockPredictor
from utils import ModelManager, DataValidator, PerformanceTracker, setup_logging
from config import MODEL_CONFIG, DATA_CONFIG, MODELS_DIR, LOGS_DIR

class TrainingPipeline:
    """Complete training pipeline for stock prediction models"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize training pipeline
        
        Args:
            config: Configuration dictionary (uses defaults if None)
        """
        self.config = config or MODEL_CONFIG.__dict__
        self.logger = setup_logging(log_file=str(LOGS_DIR / "training.log"))
        
        # Initialize components
        self.data_collector = StockDataCollector()
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = DataPreprocessor(
            lookback_days=self.config['lookback_days'],
            forecast_days=self.config['forecast_days']
        )
        self.model_manager = ModelManager(str(MODELS_DIR))
        self.data_validator = DataValidator()
        self.performance_tracker = PerformanceTracker(str(LOGS_DIR / "performance_tracking.json"))
        
        self.logger.info("Training pipeline initialized")
    
    def train_single_stock(self, symbol: str, period: str = "5y", 
                          save_model: bool = True, use_attention: bool = False) -> Dict:
        """
        Train model for a single stock
        
        Args:
            symbol: Stock symbol to train on
            period: Data period ('1y', '2y', '5y', etc.)
            save_model: Whether to save the trained model
            use_attention: Whether to use attention mechanism
            
        Returns:
            Training results dictionary
        """
        try:
            self.logger.info(f"Starting training for {symbol}")
            start_time = datetime.now()
            
            # Step 1: Collect data
            self.logger.info(f"Collecting data for {symbol}")
            raw_data = self.data_collector.fetch_stock_data(symbol, period)
            
            if raw_data is None:
                raise ValueError(f"Failed to fetch data for {symbol}")
            
            # Step 2: Validate data
            self.logger.info("Validating data quality")
            validation_results = self.data_validator.validate_stock_data(raw_data, symbol)
            
            if not validation_results['valid']:
                raise ValueError(f"Data validation failed: {validation_results['issues']}")
            
            # Step 3: Feature engineering
            self.logger.info("Engineering features")
            enhanced_data = self.feature_engineer.add_technical_indicators(raw_data)
            
            # Step 4: Validate features
            feature_validation = self.data_validator.validate_features(
                enhanced_data, DATA_CONFIG.feature_columns
            )
            
            if not feature_validation['valid']:
                self.logger.warning(f"Feature validation issues: {feature_validation['issues']}")
            
            # Step 5: Prepare data for training
            self.logger.info("Preparing data for training")
            prepared_data = self.preprocessor.prepare_data_for_training(
                enhanced_data, DATA_CONFIG.feature_columns
            )
            
            # Step 6: Initialize and train model
            self.logger.info("Initializing LSTM model")
            model = LSTMStockPredictor(self.config)
            
            # Prepare model save path
            model_path = None
            if save_model:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_path = str(MODELS_DIR / f"{symbol}_lstm_{timestamp}.h5")
            
            # Train model
            self.logger.info("Starting model training")
            training_results = model.train(
                X_train=prepared_data['X_train'],
                y_train=prepared_data['y_train'],
                X_val=prepared_data['X_test'],
                y_val=prepared_data['y_test'],
                model_path=model_path,
                use_attention=use_attention
            )
            
            # Step 7: Compile results
            end_time = datetime.now()
            training_duration = (end_time - start_time).total_seconds()
            
            results = {
                'symbol': symbol,
                'training_duration': training_duration,
                'data_shape': prepared_data['data_shape'],
                'train_metrics': training_results['train_metrics'],
                'val_metrics': training_results['val_metrics'],
                'model_path': model_path,
                'data_validation': validation_results,
                'feature_validation': feature_validation,
                'config': self.config,
                'timestamp': start_time.isoformat(),
                'use_attention': use_attention
            }
            
            # Step 8: Save model with metadata if requested
            if save_model:
                self.logger.info("Saving model with metadata")
                model_dir = self.model_manager.save_model_with_metadata(
                    model, symbol, results
                )
                results['model_directory'] = model_dir
            
            # Step 9: Log experiment
            self.performance_tracker.log_experiment({
                'name': f"{symbol}_{'attention' if use_attention else 'lstm'}",
                'symbol': symbol,
                'model_type': 'LSTM_Attention' if use_attention else 'LSTM',
                'metrics': training_results['val_metrics'],
                'config': self.config,
                'duration': training_duration
            })
            
            self.logger.info(f"Training completed for {symbol} in {training_duration:.2f}s")
            self.logger.info(f"Validation metrics: {training_results['val_metrics']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error training model for {symbol}: {str(e)}")
            raise
    
    def train_multiple_stocks(self, symbols: List[str], period: str = "5y",
                             save_models: bool = True, use_attention: bool = False) -> Dict:
        """
        Train models for multiple stocks
        
        Args:
            symbols: List of stock symbols
            period: Data period
            save_models: Whether to save trained models
            use_attention: Whether to use attention mechanism
            
        Returns:
            Dictionary with results for each symbol
        """
        try:
            self.logger.info(f"Starting batch training for {len(symbols)} stocks")
            start_time = datetime.now()
            
            results = {}
            successful_trainings = 0
            failed_trainings = 0
            
            for i, symbol in enumerate(symbols):
                self.logger.info(f"Training {i+1}/{len(symbols)}: {symbol}")
                
                try:
                    symbol_results = self.train_single_stock(
                        symbol=symbol,
                        period=period,
                        save_model=save_models,
                        use_attention=use_attention
                    )
                    results[symbol] = symbol_results
                    successful_trainings += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to train {symbol}: {str(e)}")
                    results[symbol] = {
                        'error': str(e),
                        'status': 'failed'
                    }
                    failed_trainings += 1
            
            # Compile batch results
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            batch_results = {
                'symbols': symbols,
                'successful_trainings': successful_trainings,
                'failed_trainings': failed_trainings,
                'total_duration': total_duration,
                'average_duration': total_duration / len(symbols),
                'results': results,
                'timestamp': start_time.isoformat()
            }
            
            self.logger.info(f"Batch training completed: {successful_trainings} successful, "
                           f"{failed_trainings} failed in {total_duration:.2f}s")
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Error in batch training: {str(e)}")
            raise
    
    def hyperparameter_search(self, symbol: str, param_grid: Dict, 
                             max_trials: int = 10) -> Dict:
        """
        Perform hyperparameter search
        
        Args:
            symbol: Stock symbol to optimize for
            param_grid: Dictionary of parameter ranges
            max_trials: Maximum number of trials
            
        Returns:
            Best parameters and results
        """
        try:
            self.logger.info(f"Starting hyperparameter search for {symbol}")
            
            # Collect and prepare data once
            raw_data = self.data_collector.fetch_stock_data(symbol, "5y")
            if raw_data is None:
                raise ValueError(f"Failed to fetch data for {symbol}")
            
            enhanced_data = self.feature_engineer.add_technical_indicators(raw_data)
            
            best_score = 0
            best_params = None
            best_results = None
            trial_results = []
            
            # Generate parameter combinations (simple grid search)
            import itertools
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            # Limit combinations to max_trials
            combinations = list(itertools.product(*param_values))[:max_trials]
            
            for i, param_combination in enumerate(combinations):
                self.logger.info(f"Trial {i+1}/{len(combinations)}")
                
                # Create config for this trial
                trial_config = self.config.copy()
                for name, value in zip(param_names, param_combination):
                    trial_config[name] = value
                
                try:
                    # Update preprocessor with new parameters
                    preprocessor = DataPreprocessor(
                        lookback_days=trial_config['lookback_days'],
                        forecast_days=trial_config['forecast_days']
                    )
                    
                    # Prepare data
                    prepared_data = preprocessor.prepare_data_for_training(
                        enhanced_data, DATA_CONFIG.feature_columns
                    )
                    
                    # Train model
                    model = LSTMStockPredictor(trial_config)
                    training_results = model.train(
                        X_train=prepared_data['X_train'],
                        y_train=prepared_data['y_train'],
                        X_val=prepared_data['X_test'],
                        y_val=prepared_data['y_test'],
                        model_path=None,
                        use_attention=False
                    )
                    
                    # Evaluate performance
                    score = training_results['val_metrics']['directional_accuracy']
                    
                    trial_result = {
                        'trial': i + 1,
                        'params': dict(zip(param_names, param_combination)),
                        'score': score,
                        'metrics': training_results['val_metrics']
                    }
                    trial_results.append(trial_result)
                    
                    self.logger.info(f"Trial {i+1} score: {score:.4f}")
                    
                    # Update best if better
                    if score > best_score:
                        best_score = score
                        best_params = dict(zip(param_names, param_combination))
                        best_results = training_results
                        
                        self.logger.info(f"New best score: {best_score:.4f}")
                
                except Exception as e:
                    self.logger.error(f"Trial {i+1} failed: {str(e)}")
                    continue
            
            # Compile search results
            search_results = {
                'symbol': symbol,
                'best_params': best_params,
                'best_score': best_score,
                'best_metrics': best_results['val_metrics'] if best_results else None,
                'total_trials': len(trial_results),
                'trial_results': trial_results,
                'param_grid': param_grid
            }
            
            self.logger.info(f"Hyperparameter search completed. Best score: {best_score:.4f}")
            self.logger.info(f"Best parameters: {best_params}")
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter search: {str(e)}")
            raise
    
    def evaluate_model_performance(self, symbol: str, model_path: str) -> Dict:
        """
        Evaluate a trained model's performance
        
        Args:
            symbol: Stock symbol
            model_path: Path to trained model
            
        Returns:
            Evaluation results
        """
        try:
            self.logger.info(f"Evaluating model performance for {symbol}")
            
            # Load model
            model = LSTMStockPredictor.load_model(model_path)
            
            # Get recent data for evaluation
            raw_data = self.data_collector.fetch_stock_data(symbol, "1y")
            enhanced_data = self.feature_engineer.add_technical_indicators(raw_data)
            
            # Prepare data
            prepared_data = self.preprocessor.prepare_data_for_training(
                enhanced_data, DATA_CONFIG.feature_columns
            )
            
            # Make predictions
            predictions = model.predict(prepared_data['X_test'])
            actual = prepared_data['y_test']
            
            # Calculate additional metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            mae = mean_absolute_error(actual.flatten(), predictions.flatten())
            mse = mean_squared_error(actual.flatten(), predictions.flatten())
            rmse = np.sqrt(mse)
            r2 = r2_score(actual.flatten(), predictions.flatten())
            
            # Directional accuracy
            actual_direction = np.sign(np.diff(actual, axis=1))
            pred_direction = np.sign(np.diff(predictions, axis=1))
            directional_accuracy = np.mean(actual_direction == pred_direction)
            
            evaluation_results = {
                'symbol': symbol,
                'model_path': model_path,
                'metrics': {
                    'mae': float(mae),
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'r2_score': float(r2),
                    'directional_accuracy': float(directional_accuracy)
                },
                'data_points': len(predictions),
                'evaluation_date': datetime.now().isoformat()
            }
            
            self.logger.info(f"Model evaluation completed for {symbol}")
            self.logger.info(f"Metrics: {evaluation_results['metrics']}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise

def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train stock prediction models")
    parser.add_argument("--symbol", type=str, help="Stock symbol to train on")
    parser.add_argument("--symbols", nargs="+", help="Multiple stock symbols")
    parser.add_argument("--period", type=str, default="5y", help="Data period")
    parser.add_argument("--attention", action="store_true", help="Use attention mechanism")
    parser.add_argument("--hyperparameter-search", action="store_true", help="Perform hyperparameter search")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TrainingPipeline()
    
    try:
        if args.hyperparameter_search and args.symbol:
            # Hyperparameter search
            param_grid = {
                'lookback_days': [30, 60, 90],
                'lstm_units': [[50, 25], [100, 50, 25], [128, 64, 32]],
                'dropout_rate': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.0005, 0.002]
            }
            
            results = pipeline.hyperparameter_search(args.symbol, param_grid)
            print(f"Best parameters for {args.symbol}: {results['best_params']}")
            print(f"Best score: {results['best_score']:.4f}")
            
        elif args.symbols:
            # Train multiple stocks
            results = pipeline.train_multiple_stocks(
                symbols=args.symbols,
                period=args.period,
                use_attention=args.attention
            )
            print(f"Batch training completed: {results['successful_trainings']} successful")
            
        elif args.symbol:
            # Train single stock
            results = pipeline.train_single_stock(
                symbol=args.symbol,
                period=args.period,
                use_attention=args.attention
            )
            print(f"Training completed for {args.symbol}")
            print(f"Validation metrics: {results['val_metrics']}")
            
        else:
            print("Please specify --symbol or --symbols")
            
    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
