"""
Utility functions and helper classes for the stock prediction system.
"""
import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Optional imports with fallbacks
try:
    import joblib
except ImportError:
    joblib = None

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)

class ModelManager:
    """Manages model saving, loading, and versioning"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_model_with_metadata(self, model, symbol: str, metadata: Dict) -> str:
        """
        Save model with comprehensive metadata
        
        Args:
            model: Trained model instance
            symbol: Stock symbol
            metadata: Model metadata and metrics
            
        Returns:
            Path to saved model
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"{symbol}_lstm_{timestamp}"
            
            # Create model directory
            model_dir = self.models_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Save model
            model_path = model_dir / "model.h5"
            model.save_model(str(model_path))
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                clean_metadata = self._clean_metadata_for_json(metadata)
                json.dump(clean_metadata, f, indent=2, default=str)
            
            # Save training configuration
            config_path = model_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(model.config, f, indent=2)
            
            self.logger.info(f"Model saved with metadata at {model_dir}")
            return str(model_dir)
            
        except Exception as e:
            self.logger.error(f"Error saving model with metadata: {str(e)}")
            raise
    
    def _clean_metadata_for_json(self, metadata: Dict) -> Dict:
        """Clean metadata for JSON serialization"""
        clean_data = {}
        
        for key, value in metadata.items():
            if isinstance(value, (np.integer, np.floating)):
                clean_data[key] = float(value)
            elif isinstance(value, np.ndarray):
                clean_data[key] = value.tolist()
            elif isinstance(value, dict):
                clean_data[key] = self._clean_metadata_for_json(value)
            elif isinstance(value, (list, tuple)):
                clean_data[key] = [self._clean_value(v) for v in value]
            else:
                clean_data[key] = value
        
        return clean_data
    
    def _clean_value(self, value):
        """Clean individual values for JSON serialization"""
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            return value
    def list_models(self, symbol: Optional[str] = None) -> List[Dict]:
        """List available models with their metadata"""
        try:
            models = []
            
            # Check for directory-based models (new format)
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir():
                    metadata_path = model_dir / "metadata.json"
                    
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        model_info = {
                            'name': model_dir.name,
                            'path': str(model_dir),
                            'symbol': metadata.get('symbol', 'unknown'),
                            'created': metadata.get('created', 'unknown'),
                            'metrics': metadata.get('val_metrics', {}),
                            'size': self._get_directory_size(model_dir),
                            'type': 'directory'
                        }
                        
                        if symbol is None or model_info['symbol'] == symbol:
                            models.append(model_info)
            
            # Check for legacy .keras files (existing format)
            for model_file in self.models_dir.glob("*.keras"):
                # Parse symbol from filename (e.g., AAPL_lstm_20250612_191825.keras)
                filename_parts = model_file.stem.split('_')
                if len(filename_parts) >= 3:
                    file_symbol = filename_parts[0]
                    date_part = filename_parts[2] if len(filename_parts) >= 3 else 'unknown'
                    time_part = filename_parts[3] if len(filename_parts) >= 4 else ''
                    
                    # Convert to readable date
                    try:
                        if len(date_part) == 8:
                            created_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                            if time_part and len(time_part) == 6:
                                created_date += f" {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                        else:
                            created_date = 'unknown'
                    except:
                        created_date = 'unknown'
                    
                    model_info = {
                        'name': model_file.stem,
                        'path': str(model_file),
                        'symbol': file_symbol,
                        'created': created_date,
                        'metrics': {'note': 'Pre-trained model (no validation metrics available)'},
                        'size': f"{model_file.stat().st_size / (1024*1024):.1f} MB",
                        'type': 'file'
                    }
                    
                    if symbol is None or model_info['symbol'] == symbol:
                        models.append(model_info)
            
            # Sort by creation date (newest first)
            models.sort(key=lambda x: x['created'], reverse=True)
            return models
            
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            return []
    
    def _get_directory_size(self, directory: Path) -> str:
        """Get directory size in human-readable format"""
        try:
            total_size = sum(f.stat().st_size for f in directory.glob('**/*') if f.is_file())
            
            # Convert to human-readable format
            for unit in ['B', 'KB', 'MB', 'GB']:
                if total_size < 1024.0:
                    return f"{total_size:.1f} {unit}"
                total_size /= 1024.0
            
            return f"{total_size:.1f} TB"
            
        except Exception:
            return "Unknown"
    
    def load_best_model(self, symbol: str):
        """Load the best performing model for a symbol"""
        try:
            models = self.list_models(symbol)
            
            if not models:
                raise ValueError(f"No models found for symbol {symbol}")
              # Find model with best directional accuracy
            best_model = max(models, 
                           key=lambda x: x['metrics'].get('directional_accuracy', 0))
            
            model_path = Path(best_model['path']) / "model.h5"
            
            try:
                from .lstm_model import LSTMStockPredictor
            except ImportError:
                from lstm_model import LSTMStockPredictor
            model = LSTMStockPredictor.load_model(str(model_path))
            
            self.logger.info(f"Loaded best model for {symbol}: {best_model['name']}")
            return model, best_model
            
        except Exception as e:
            self.logger.error(f"Error loading best model: {str(e)}")
            raise

class DataValidator:
    """Validates and fixes data quality issues for stock prediction models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_and_fix_data(self, data: pd.DataFrame, symbol: str = "Unknown") -> pd.DataFrame:
        """
        Validate and fix common data issues, ensuring all required columns exist
        
        Args:
            data: DataFrame with stock data
            symbol: Stock symbol for logging
            
        Returns:
            Fixed DataFrame with all required columns
        """
        try:
            self.logger.info(f"Validating data for {symbol}, shape: {data.shape}")
            
            # Make a copy to avoid modifying original
            fixed_data = data.copy()
            
            # Required columns for technical analysis and model
            required_columns = ['SMA_20', 'EMA_12', 'RSI', 'MACD', 'MACD_signal', 
                              'BB_upper', 'BB_lower', 'volatility', 'price_change']
            
            # Check which columns are missing
            missing_columns = [col for col in required_columns if col not in fixed_data.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing columns for {symbol}: {missing_columns}")
                
                # Add missing columns with calculated values if base data exists
                if 'Close' in fixed_data.columns:
                    for col in missing_columns:
                        self.logger.info(f"Adding missing column: {col}")
                        
                        if col == 'SMA_20':
                            fixed_data[col] = fixed_data['Close'].rolling(20).mean()
                        elif col == 'EMA_12':
                            fixed_data[col] = fixed_data['Close'].ewm(span=12).mean()
                        elif col == 'RSI':
                            # Simple RSI calculation
                            delta = fixed_data['Close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            fixed_data[col] = 100 - (100 / (1 + rs))
                        elif col == 'MACD':
                            # Simple MACD calculation
                            ema_12 = fixed_data['Close'].ewm(span=12).mean()
                            ema_26 = fixed_data['Close'].ewm(span=26).mean()
                            fixed_data[col] = ema_12 - ema_26
                        elif col == 'MACD_signal':
                            if 'MACD' in fixed_data.columns:
                                fixed_data[col] = fixed_data['MACD'].ewm(span=9).mean()
                            else:
                                fixed_data[col] = 0
                        elif col == 'BB_upper':
                            sma_20 = fixed_data['Close'].rolling(20).mean()
                            std_20 = fixed_data['Close'].rolling(20).std()
                            fixed_data[col] = sma_20 + (std_20 * 2)
                        elif col == 'BB_lower':
                            sma_20 = fixed_data['Close'].rolling(20).mean()
                            std_20 = fixed_data['Close'].rolling(20).std()
                            fixed_data[col] = sma_20 - (std_20 * 2)
                        elif col == 'volatility':
                            fixed_data[col] = fixed_data['Close'].rolling(20).std()
                        elif col == 'price_change':
                            fixed_data[col] = fixed_data['Close'].pct_change()
                        else:
                            # Default to 0 for unknown indicators
                            fixed_data[col] = 0
              # Fill NaN values for all indicators
            for col in required_columns:
                if col in fixed_data.columns:
                    # Forward fill, then backward fill, then fill remaining with appropriate defaults
                    fixed_data[col] = fixed_data[col].ffill().bfill()
                    
                    # Fill remaining NaN with appropriate defaults
                    if col == 'RSI':
                        fixed_data[col] = fixed_data[col].fillna(50)  # Neutral RSI
                    elif col in ['MACD', 'MACD_signal', 'price_change']:
                        fixed_data[col] = fixed_data[col].fillna(0)
                    else:
                        # For price-based indicators, use the close price or 0
                        if 'Close' in fixed_data.columns:
                            fixed_data[col] = fixed_data[col].fillna(fixed_data['Close'].iloc[-1] if len(fixed_data) > 0 else 0)
                        else:
                            fixed_data[col] = fixed_data[col].fillna(0)
            
            # Ensure we have all required columns
            final_missing = [col for col in required_columns if col not in fixed_data.columns]
            if final_missing:
                self.logger.error(f"Still missing columns after validation: {final_missing}")
                # Add them with default values
                for col in final_missing:
                    fixed_data[col] = 0
            
            self.logger.info(f"Data validation complete for {symbol}. Final shape: {fixed_data.shape}")
            self.logger.info(f"Available columns: {fixed_data.columns.tolist()}")
            
            return fixed_data
            
        except Exception as e:
            self.logger.error(f"Data validation error for {symbol}: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return data  # Return original data if validation fails

    def validate_stock_data(self, data: pd.DataFrame, symbol: str = "Unknown") -> Dict:
        """
        Validate stock data and return validation results (for app.py compatibility)
        
        Args:
            data: DataFrame with stock data
            symbol: Stock symbol for logging
            
        Returns:
            Dictionary with validation results
        """
        try:
            self.logger.info(f"Validating stock data for {symbol}")
            
            issues = []
            
            # Check basic requirements
            if data is None or data.empty:
                issues.append("No data available")
                return {
                    'valid': False, 
                    'issues': issues,
                    'data_shape': (0, 0),
                    'available_columns': [],
                    'stats': {
                        'latest_price': 0,
                        'price_change': 0,
                        'volatility': 0,
                        'date_range': "N/A",
                        'total_records': 0
                    }
                }
            
            # Check required columns
            required_basic_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_basic = [col for col in required_basic_columns if col not in data.columns]
            if missing_basic:
                issues.append(f"Missing basic columns: {missing_basic}")
            
            # Check data quality
            if len(data) < 30:
                issues.append(f"Insufficient data: only {len(data)} records")
            
            # Check for excessive missing values
            for col in required_basic_columns:
                if col in data.columns:
                    null_pct = data[col].isnull().sum() / len(data) * 100
                    if null_pct > 50:
                        issues.append(f"High missing values in {col}: {null_pct:.1f}%")
            
            # Check for technical indicators
            technical_indicators = ['SMA_20', 'EMA_12', 'RSI', 'MACD', 'MACD_signal', 
                                  'BB_upper', 'BB_lower', 'volatility', 'price_change']
            missing_indicators = [col for col in technical_indicators if col not in data.columns]
            if missing_indicators:
                issues.append(f"Missing technical indicators: {len(missing_indicators)} indicators need calculation")
            
            # Calculate stats for the app
            stats = {}
            if not data.empty and 'Close' in data.columns:
                try:
                    stats.update({
                        'latest_price': float(data['Close'].iloc[-1]) if len(data) > 0 else 0,
                        'price_change': float(data['Close'].pct_change().iloc[-1]) if len(data) > 1 else 0,
                        'volatility': float(data['Close'].pct_change().std() * (252**0.5)) if len(data) > 20 else 0,
                        'date_range': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}" if len(data) > 0 else "N/A",
                        'total_records': len(data)
                    })
                except Exception as e:
                    self.logger.warning(f"Error calculating stats: {e}")
                    stats = {
                        'latest_price': 0,
                        'price_change': 0,
                        'volatility': 0,
                        'date_range': "N/A",
                        'total_records': len(data) if not data.empty else 0
                    }
            else:
                stats = {
                    'latest_price': 0,
                    'price_change': 0,
                    'volatility': 0,
                    'date_range': "N/A",
                    'total_records': 0
                }
            
            # Determine if data is valid
            is_valid = len(issues) == 0 or all('Missing technical indicators' in issue for issue in issues)
            
            self.logger.info(f"Data validation for {symbol}: {'Valid' if is_valid else 'Has issues'}")
            
            return {
                'valid': is_valid,
                'issues': issues,
                'data_shape': data.shape,
                'available_columns': data.columns.tolist(),
                'stats': stats
            }
            
        except Exception as e:
            self.logger.error(f"Stock data validation error for {symbol}: {str(e)}")
            return {
                'valid': False, 
                'issues': [f"Validation error: {str(e)}"],
                'data_shape': (0, 0),
                'available_columns': [],
                'stats': {
                    'latest_price': 0,
                    'price_change': 0,                    'volatility': 0,
                    'date_range': "N/A",
                    'total_records': 0
                }
            }

class PerformanceTracker:
    """Track and analyze model performance over time"""
    
    def __init__(self, tracking_file: str = "performance_tracking.json"):
        self.tracking_file = Path(tracking_file)
        self.logger = logging.getLogger(__name__)
        
        # Load existing tracking data
        self.data = self._load_tracking_data()
    
    def _load_tracking_data(self) -> Dict:
        """Load existing tracking data"""
        try:
            if self.tracking_file.exists():
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            else:
                return {'models': [], 'experiments': []}
        except Exception as e:
            self.logger.error(f"Error loading tracking data: {str(e)}")
            return {'models': [], 'experiments': []}
    
    def _save_tracking_data(self):
        """Save tracking data to file"""
        try:
            with open(self.tracking_file, 'w') as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving tracking data: {str(e)}")
    
    def log_experiment(self, experiment_data: Dict):
        """Log experiment results"""
        try:
            experiment_data['timestamp'] = datetime.now().isoformat()
            experiment_data['id'] = len(self.data['experiments'])
            
            self.data['experiments'].append(experiment_data)
            self._save_tracking_data()
            
            self.logger.info(f"Logged experiment: {experiment_data.get('name', 'Unknown')}")
            
        except Exception as e:
            self.logger.error(f"Error logging experiment: {str(e)}")
    
    def get_best_experiments(self, metric: str = 'directional_accuracy', top_n: int = 5) -> List[Dict]:
        """Get top performing experiments"""
        try:
            experiments = self.data['experiments']
            
            # Filter experiments with the specified metric
            valid_experiments = [
                exp for exp in experiments 
                if 'metrics' in exp and metric in exp['metrics']
            ]
            
            # Sort by metric (descending for accuracy, ascending for error metrics)
            ascending = metric in ['mae', 'mse', 'rmse']
            sorted_experiments = sorted(
                valid_experiments,
                key=lambda x: x['metrics'][metric],
                reverse=not ascending
            )
            
            return sorted_experiments[:top_n]
            
        except Exception as e:
            self.logger.error(f"Error getting best experiments: {str(e)}")
            return []
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        try:
            report = {
                'summary': {
                    'total_experiments': len(self.data['experiments']),
                    'symbols_tested': len(set(exp.get('symbol', 'unknown') 
                                           for exp in self.data['experiments'])),
                    'date_range': self._get_date_range()
                },
                'best_performers': {},
                'trends': {},
                'recommendations': []
            }
            
            # Get best performers for each metric
            metrics = ['directional_accuracy', 'mae', 'rmse']
            for metric in metrics:
                report['best_performers'][metric] = self.get_best_experiments(metric, 3)
            
            # Analyze trends
            if len(self.data['experiments']) > 5:
                report['trends'] = self._analyze_trends()
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations()
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
            return {'error': str(e)}
    
    def _get_date_range(self) -> Dict:
        """Get date range of experiments"""
        try:
            if not self.data['experiments']:
                return {}
            
            dates = [exp.get('timestamp') for exp in self.data['experiments'] if exp.get('timestamp')]
            if dates:
                return {
                    'first': min(dates),
                    'last': max(dates)
                }
            return {}
        except Exception:
            return {}
    
    def _analyze_trends(self) -> Dict:
        """Analyze performance trends over time"""
        # Placeholder for trend analysis
        # In a full implementation, this would analyze metrics over time
        return {
            'note': 'Trend analysis would be implemented here',
            'requires': 'More experiments for meaningful analysis'
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on experiment history"""
        recommendations = []
        
        if len(self.data['experiments']) < 5:
            recommendations.append("Run more experiments to get meaningful insights")
        
        # Analyze common patterns in successful experiments
        successful_experiments = [
            exp for exp in self.data['experiments']
            if exp.get('metrics', {}).get('directional_accuracy', 0) > 0.6
        ]
        
        if successful_experiments:
            recommendations.append("Focus on configurations similar to your best performing models")
        else:
            recommendations.append("Consider adjusting model architecture or features")
        
        return recommendations
