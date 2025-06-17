"""
Utility functions and helper classes for the stock prediction system.
"""
import logging
import os
import joblib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

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
    """Validates data quality and completeness"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_stock_data(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Validate stock data quality
        
        Args:
            df: Stock data DataFrame
            symbol: Stock symbol
            
        Returns:
            Validation results dictionary
        """
        try:
            results = {
                'symbol': symbol,
                'valid': True,
                'issues': [],
                'stats': {},
                'recommendations': []
            }
            
            # Check basic requirements
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                results['valid'] = False
                results['issues'].append(f"Missing required columns: {missing_columns}")
            
            # Check data size
            if len(df) < 100:
                results['valid'] = False
                results['issues'].append(f"Insufficient data: only {len(df)} records")
            elif len(df) < 500:
                results['issues'].append(f"Limited data: only {len(df)} records")
                results['recommendations'].append("Consider using longer time period for better predictions")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                results['issues'].append(f"Missing values found: {missing_values.to_dict()}")
            
            # Check for anomalies
            if 'Close' in df.columns:
                # Check for zero or negative prices
                zero_prices = (df['Close'] <= 0).sum()
                if zero_prices > 0:
                    results['valid'] = False
                    results['issues'].append(f"Found {zero_prices} zero or negative prices")
                
                # Check for extreme price changes
                price_changes = df['Close'].pct_change().abs()
                extreme_changes = (price_changes > 0.5).sum()  # More than 50% change
                if extreme_changes > 0:
                    results['issues'].append(f"Found {extreme_changes} extreme price changes (>50%)")
            
            # Check for duplicate dates
            if df.index.duplicated().any():
                results['issues'].append("Duplicate dates found in index")
            
            # Generate statistics
            if 'Close' in df.columns:
                results['stats'] = {
                    'records': len(df),
                    'date_range': {
                        'start': str(df.index.min().date()),
                        'end': str(df.index.max().date())
                    },
                    'price_range': {
                        'min': float(df['Close'].min()),
                        'max': float(df['Close'].max()),
                        'mean': float(df['Close'].mean())
                    },
                    'volatility': float(df['Close'].pct_change().std() * np.sqrt(252))  # Annualized
                }
            
            # Add recommendations based on findings
            if len(results['issues']) == 0:
                results['recommendations'].append("Data quality looks good for model training")
            elif results['valid']:
                results['recommendations'].append("Data has minor issues but should work for training")
            else:
                results['recommendations'].append("Data quality issues need to be resolved before training")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            return {
                'symbol': symbol,
                'valid': False,
                'issues': [f"Validation error: {str(e)}"],
                'stats': {},
                'recommendations': ["Fix data validation errors before proceeding"]
            }
    
    def validate_features(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict:
        """Validate feature engineering results"""
        try:
            results = {
                'valid': True,
                'issues': [],
                'feature_stats': {}
            }
            
            for column in feature_columns:
                if column not in df.columns:
                    results['valid'] = False
                    results['issues'].append(f"Missing feature column: {column}")
                    continue
                
                # Check for infinite values
                inf_count = np.isinf(df[column]).sum()
                if inf_count > 0:
                    results['issues'].append(f"Infinite values in {column}: {inf_count}")
                
                # Check for NaN values
                nan_count = df[column].isnull().sum()
                if nan_count > 0:
                    results['issues'].append(f"NaN values in {column}: {nan_count}")
                
                # Calculate basic statistics
                if df[column].dtype in ['float64', 'int64']:
                    results['feature_stats'][column] = {
                        'mean': float(df[column].mean()),
                        'std': float(df[column].std()),
                        'min': float(df[column].min()),
                        'max': float(df[column].max())
                    }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error validating features: {str(e)}")
            return {
                'valid': False,
                'issues': [f"Feature validation error: {str(e)}"],
                'feature_stats': {}
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
