"""
Data collection and processing utilities for stock price prediction.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from typing import Optional, Tuple, Dict, List
import logging
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class StockDataCollector:
    """Handles stock data collection and preprocessing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
      def fetch_stock_data(self, symbol: str, period: str = "5y") -> Optional[pd.DataFrame]:
        """
        Fetch stock data from Yahoo Finance with comprehensive error handling
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period ('1y', '2y', '5y', '10y', 'max')
            
        Returns:
            DataFrame with stock data or None if error
        """
        try:
            self.logger.info(f"Fetching data for {symbol} with period {period}")
            
            # Validate symbol format
            symbol = symbol.upper().strip()
            if not symbol or len(symbol) > 10:
                self.logger.error(f"Invalid symbol format: {symbol}")
                return None
              # Create ticker with enhanced configuration
            stock = yf.Ticker(symbol)
            data = None
            
            # Method 1: Try with period (most reliable)
            try:
                data = stock.history(
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    prepost=False,
                    threads=True,
                    proxy=None
                )
                self.logger.info(f"Method 1 successful for {symbol}")
            except Exception as e1:
                self.logger.warning(f"Method 1 failed for {symbol}: {e1}")
                
                # Method 2: Try with explicit date range
                try:
                    end_date = datetime.now()
                    if period == "1y":
                        start_date = end_date - timedelta(days=365)
                    elif period == "2y":
                        start_date = end_date - timedelta(days=730)
                    elif period == "5y":
                        start_date = end_date - timedelta(days=1825)
                    elif period == "10y":
                        start_date = end_date - timedelta(days=3650)
                    elif period == "max":
                        start_date = end_date - timedelta(days=7300)  # ~20 years
                    else:
                        start_date = end_date - timedelta(days=1825)  # Default 5y
                    
                    data = stock.history(
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        auto_adjust=True,
                        prepost=False
                    )
                    self.logger.info(f"Method 2 successful for {symbol}")
                except Exception as e2:
                    self.logger.error(f"Method 2 also failed for {symbol}: {e2}")
                      # Method 3: Try with download function
                    try:
                        data = yf.download(
                            symbol,
                            period=period,
                            interval="1d",
                            auto_adjust=True,
                            prepost=False,
                            threads=True,
                            group_by='ticker',
                            progress=False
                        )
                        
                        # If multi-level columns, flatten them
                        if hasattr(data.columns, 'levels'):
                            data.columns = data.columns.droplevel(0)
                        
                        self.logger.info(f"Method 3 successful for {symbol}")
                    except Exception as e3:
                        self.logger.error(f"All methods failed for {symbol}: {e3}")
                        return None
            
            # Validate data
            if data is None or data.empty:
                self.logger.error(f"No data returned for symbol {symbol}")
                return None
            
            # Check required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns for {symbol}: {missing_cols}")
                return None
            
            # Clean and validate data
            data = data.dropna()
            
            if len(data) < 30:  # Minimum data requirement
                self.logger.error(f"Insufficient data for {symbol}: only {len(data)} points")
                return None
            
            # Ensure numeric types
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove any remaining NaN after conversion
            data = data.dropna()
            
            # Add metadata
            data.attrs['symbol'] = symbol
            data.attrs['period'] = period
            data.attrs['last_update'] = datetime.now()
            
            self.logger.info(f"Successfully fetched {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
      def get_stock_info(self, symbol: str) -> Dict:
        """Get basic stock information with improved error handling"""
        try:
            stock = yf.Ticker(symbol)
            
            # Try to get info with timeout
            info = stock.info
            
            # Validate that we got actual data
            if not info or len(info) < 5:
                # Fallback to basic info
                return {
                    'name': symbol,
                    'sector': 'Unknown',
                    'market_cap': 0,
                    'pe_ratio': 0,
                    'dividend_yield': 0
                }
            
            return {
                'name': info.get('longName', info.get('shortName', symbol)),
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', info.get('forwardPE', 0)),
                'dividend_yield': info.get('dividendYield', 0)
            }
        except Exception as e:
            self.logger.error(f"Error getting info for {symbol}: {str(e)}")
            return {
                'name': symbol,
                'sector': 'Unknown', 
                'market_cap': 0,
                'pe_ratio': 0,
                'dividend_yield': 0
            }

class FeatureEngineer:
    """Creates technical indicators and features for ML models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators to the dataframe
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        try:
            data = df.copy()
            
            # Moving averages
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            data['EMA_12'] = ta.trend.ema_indicator(data['Close'], window=12)
            data['EMA_26'] = ta.trend.ema_indicator(data['Close'], window=26)
            
            # RSI
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
            data['MACD_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(data['Close'])
            data['BB_upper'] = bollinger.bollinger_hband()
            data['BB_middle'] = bollinger.bollinger_mavg()
            data['BB_lower'] = bollinger.bollinger_lband()
            data['BB_width'] = data['BB_upper'] - data['BB_lower']
            
            # Stochastic Oscillator
            data['Stoch_K'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
            data['Stoch_D'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'])
            
            # Average True Range (ATR)
            data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
            
            # Volume indicators
            data['Volume_SMA'] = ta.volume.volume_sma(data['Close'], data['Volume'], window=20)
            data['Volume_ratio'] = data['Volume'] / data['Volume_SMA']
            
            # Price-based features
            data['price_change'] = data['Close'].pct_change()
            data['high_low_ratio'] = data['High'] / data['Low']
            data['volatility'] = data['price_change'].rolling(window=20).std()
            
            # Gap indicators
            data['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
            
            # Time-based features
            data['day_of_week'] = data.index.dayofweek
            data['month'] = data.index.month
            data['quarter'] = data.index.quarter
            
            # Trend strength
            data['trend_20'] = np.where(data['Close'] > data['SMA_20'], 1, -1)
            data['trend_50'] = np.where(data['Close'] > data['SMA_50'], 1, -1)
            
            self.logger.info(f"Added {len(data.columns) - len(df.columns)} technical indicators")
            return data
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            return df
    
    def create_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """Create lagged features for specified columns"""
        data = df.copy()
        
        for col in columns:
            if col in data.columns:
                for lag in lags:
                    data[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        return data
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'minmax') -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize data for ML models
        
        Args:
            df: Input dataframe
            method: 'minmax', 'standard', or 'robust'
            
        Returns:
            Normalized dataframe and scaler parameters
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
        
        data = df.copy()
        scalers = {}
        
        # Select numeric columns only
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'standard':
                scaler = StandardScaler()
            else:  # robust
                scaler = RobustScaler()
            
            # Fit and transform
            data[column] = scaler.fit_transform(data[[column]])
            scalers[column] = scaler
        
        return data, scalers

class DataPreprocessor:
    """Handles data preprocessing for ML models"""
    
    def __init__(self, lookback_days: int = 60, forecast_days: int = 5):
        self.lookback_days = lookback_days
        self.forecast_days = forecast_days
        self.logger = logging.getLogger(__name__)
    
    def create_sequences(self, data: np.ndarray, target_col_idx: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            data: Normalized data array
            target_col_idx: Index of target column (Close price)
            
        Returns:
            X sequences and y targets
        """
        X, y = [], []
        
        for i in range(self.lookback_days, len(data) - self.forecast_days + 1):
            # Input sequence
            X.append(data[i-self.lookback_days:i])
            
            # Target (next forecast_days of close prices)
            y.append(data[i:i+self.forecast_days, target_col_idx])
        
        return np.array(X), np.array(y)
    
    def train_test_split(self, X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> Tuple:
        """Split data into train and test sets"""
        split_idx = int(len(X) * train_ratio)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        self.logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_data_for_training(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict:
        """
        Complete data preparation pipeline
        
        Args:
            df: DataFrame with stock data and indicators
            feature_columns: List of columns to use as features
            
        Returns:
            Dictionary with prepared data and metadata
        """
        try:
            # Remove rows with NaN values
            clean_data = df[feature_columns].dropna()
            
            if len(clean_data) < self.lookback_days + self.forecast_days:
                raise ValueError("Insufficient data for the specified lookback and forecast periods")
            
            # Normalize data
            preprocessor = FeatureEngineer()
            normalized_data, scalers = preprocessor.normalize_data(clean_data)
            
            # Convert to numpy array
            data_array = normalized_data.values
            
            # Create sequences
            target_idx = feature_columns.index('Close')
            X, y = self.create_sequences(data_array, target_idx)
            
            # Train-test split
            X_train, X_test, y_train, y_test = self.train_test_split(X, y)
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'scalers': scalers,
                'feature_columns': feature_columns,
                'data_shape': X.shape,
                'close_scaler': scalers['Close']
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
