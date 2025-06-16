"""
Data collection and processing utilities for stock price prediction.
Uses multiple reliable data sources with API keys from environment variables.
"""
import pandas as pd
import numpy as np
import ta
from typing import Optional, Tuple, Dict, List
import logging
from datetime import datetime, timedelta
import warnings
import requests
import json
import time
import traceback
import os
from io import StringIO

warnings.filterwarnings('ignore')

class StockDataCollector:
    """Handles stock data collection from multiple reliable sources with API keys"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Get API keys from environment variables
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        self.twelve_data_key = os.getenv('TWELVE_DATA_API_KEY', '')
        self.fmp_key = os.getenv('FMP_API_KEY', '')
        
        # Log which API keys are available
        self.logger.info(f"Alpha Vantage API: {'✅ Available' if self.alpha_vantage_key != 'demo' else '⚠️  Using demo key'}")
        self.logger.info(f"Twelve Data API: {'✅ Available' if self.twelve_data_key else '❌ No key'}")
        self.logger.info(f"FMP API: {'✅ Available' if self.fmp_key else '❌ No key'}")
        
    def fetch_stock_data(self, symbol: str, period: str = "5y") -> Optional[pd.DataFrame]:
        """
        Fetch stock data using multiple reliable sources (no Yahoo Finance)
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period ('1y', '2y', '5y', '10y', 'max')
            
        Returns:
            DataFrame with stock data or None if error
        """
        try:
            self.logger.info(f"STARTING: Fetching data for {symbol} with period {period}")
            
            # Validate symbol format
            symbol = symbol.upper().strip()
            if not symbol or len(symbol) > 10:
                self.logger.error(f"VALIDATION FAILED: Invalid symbol format: {symbol}")
                return None
            
            data = None
            
            # Method 1: Alpha Vantage (Free, reliable)
            self.logger.info(f"METHOD 1: Attempting Alpha Vantage for {symbol}")
            try:
                data = self._fetch_alpha_vantage(symbol, period)
                if data is not None and not data.empty:
                    self.logger.info(f"METHOD 1 SUCCESS: Alpha Vantage - {len(data)} records")
                else:
                    self.logger.warning(f"METHOD 1 EMPTY: Alpha Vantage returned no data")
                    data = None
            except Exception as e1:
                self.logger.error(f"METHOD 1 FAILED: Alpha Vantage - {type(e1).__name__}: {str(e1)}")
                
                # Method 2: Twelve Data (Free tier available)
                self.logger.info(f"METHOD 2: Attempting Twelve Data for {symbol}")
                try:
                    data = self._fetch_twelve_data(symbol, period)
                    if data is not None and not data.empty:
                        self.logger.info(f"METHOD 2 SUCCESS: Twelve Data - {len(data)} records")
                    else:
                        self.logger.warning(f"METHOD 2 EMPTY: Twelve Data returned no data")
                        data = None
                except Exception as e2:
                    self.logger.error(f"METHOD 2 FAILED: Twelve Data - {type(e2).__name__}: {str(e2)}")
                    
                    # Method 3: FMP (Financial Modeling Prep) Free tier
                    self.logger.info(f"METHOD 3: Attempting FMP for {symbol}")
                    try:
                        data = self._fetch_fmp_data(symbol, period)
                        if data is not None and not data.empty:
                            self.logger.info(f"METHOD 3 SUCCESS: FMP - {len(data)} records")
                        else:
                            self.logger.warning(f"METHOD 3 EMPTY: FMP returned no data")
                            data = None
                    except Exception as e3:
                        self.logger.error(f"METHOD 3 FAILED: FMP - {type(e3).__name__}: {str(e3)}")
                        
                        # Final fallback: Realistic sample data
                        self.logger.warning(f"ALL EXTERNAL SOURCES FAILED for {symbol}, using sample data")
                        try:
                            data = self._create_sample_data(symbol, period)
                        except Exception as e4:
                            self.logger.error(f"SAMPLE DATA FAILED: {symbol} - {type(e4).__name__}: {str(e4)}")
                            return None
            
            # Final Validation
            self.logger.info(f"VALIDATION: Starting for {symbol}")
            if data is None:
                self.logger.error(f"VALIDATION FAILED: {symbol} - data is None")
                return None
                
            if data.empty:
                self.logger.error(f"VALIDATION FAILED: {symbol} - data is empty")
                return None
            
            self.logger.info(f"VALIDATION: {symbol} has {len(data)} rows, columns: {list(data.columns)}")
            
            # Check required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self.logger.error(f"VALIDATION FAILED: {symbol} missing columns: {missing_cols}")
                return None
            
            # Clean and validate data
            data = data.dropna()
            
            if len(data) < 30:  # Minimum data requirement
                self.logger.error(f"VALIDATION FAILED: {symbol} insufficient data: {len(data)} points")
                return None
            
            # Ensure numeric types
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove any remaining NaN after conversion
            data = data.dropna()
            
            self.logger.info(f"VALIDATION SUCCESS: {symbol} - {len(data)} clean records")
              # Add metadata
            data.attrs['symbol'] = symbol
            data.attrs['period'] = period
            data.attrs['last_update'] = datetime.now()
            
            self.logger.info(f"FINAL SUCCESS: {symbol} ready with {len(data)} data points")
            return data
            
        except Exception as e:
            self.logger.error(f"CRITICAL ERROR: {symbol} - {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"TRACEBACK: {traceback.format_exc()}")
            return None
    
    def _fetch_alpha_vantage(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from Alpha Vantage with API key
        """
        try:
            if self.alpha_vantage_key == 'demo':
                self.logger.warning("Using Alpha Vantage demo key - limited to MSFT, IBM, etc.")
            
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "full" if period in ["5y", "10y", "max"] else "compact",
                "datatype": "json",
                "apikey": self.alpha_vantage_key
            }
            
            self.logger.info(f"Alpha Vantage API call for {symbol}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data_json = response.json()
            
            # Check for API limit errors
            if "Error Message" in data_json:
                self.logger.error(f"Alpha Vantage Error: {data_json['Error Message']}")
                return None
            
            if "Note" in data_json:
                self.logger.error(f"Alpha Vantage Rate Limit: {data_json['Note']}")
                return None
            
            if "Time Series (Daily)" in data_json:
                ts_data = data_json["Time Series (Daily)"]
                
                # Convert to DataFrame
                df_data = []
                for date_str, values in ts_data.items():
                    df_data.append({
                        'Date': pd.to_datetime(date_str),
                        'Open': float(values['1. open']),
                        'High': float(values['2. high']),
                        'Low': float(values['3. low']),
                        'Close': float(values['4. close']),
                        'Volume': int(values['5. volume'])
                    })
                
                df = pd.DataFrame(df_data).set_index('Date').sort_index()
                  # Filter by period
                df = self._filter_by_period(df, period)
                
                self.logger.info(f"Alpha Vantage SUCCESS: {len(df)} records for {symbol}")
                return df
            else:
                self.logger.error(f"Alpha Vantage unexpected response: {list(data_json.keys())}")
                return None
                
        except Exception as e:
            self.logger.error(f"Alpha Vantage error: {type(e).__name__}: {str(e)}")
            return None
            
    def _fetch_twelve_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from Twelve Data with API key
        """
        try:
            if not self.twelve_data_key:
                self.logger.warning("No Twelve Data API key - skipping")
                return None
            
            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol": symbol,
                "interval": "1day",
                "outputsize": "5000",
                "format": "json",
                "apikey": self.twelve_data_key
            }
            
            self.logger.info(f"Twelve Data API call for {symbol}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data_json = response.json()
            
            # Check for errors
            if "status" in data_json and data_json["status"] == "error":
                self.logger.error(f"Twelve Data Error: {data_json.get('message', 'Unknown error')}")
                return None
            
            if "values" in data_json and data_json["values"]:
                # Convert to DataFrame
                df_data = []
                for item in data_json["values"]:
                    df_data.append({
                        'Date': pd.to_datetime(item['datetime']),
                        'Open': float(item['open']),
                        'High': float(item['high']),
                        'Low': float(item['low']),
                        'Close': float(item['close']),
                        'Volume': int(item['volume']) if item['volume'] else 0
                    })
                
                df = pd.DataFrame(df_data).set_index('Date').sort_index()
                
                # Filter by period
                df = self._filter_by_period(df, period)
                
                self.logger.info(f"Twelve Data SUCCESS: {len(df)} records for {symbol}")
                return df
            else:
                self.logger.error(f"Twelve Data unexpected response: {list(data_json.keys())}")
                return None
                
        except Exception as e:
            self.logger.error(f"Twelve Data error: {type(e).__name__}: {str(e)}")
            return None

    def _fetch_fmp_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from Financial Modeling Prep (free tier: 250 calls/day)
        """
        try:
            # FMP free API
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
            params = {
                "serietype": "line"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data_json = response.json()
            
            if "historical" in data_json and data_json["historical"]:
                # Convert to DataFrame
                df_data = []
                for item in data_json["historical"]:
                    df_data.append({
                        'Date': pd.to_datetime(item['date']),
                        'Open': float(item['open']),
                        'High': float(item['low']),  # Note: FMP has low/high swapped in some responses
                        'Low': float(item['high']),   # So we correct it here
                        'Close': float(item['close']),
                        'Volume': int(item['volume']) if item['volume'] else 0
                    })
                
                df = pd.DataFrame(df_data).set_index('Date').sort_index()
                
                # Fix high/low if needed
                df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
                df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
                
                # Filter by period
                df = self._filter_by_period(df, period)
                
                return df
            else:
                self.logger.error(f"FMP API error: {data_json}")
                return None
                
        except Exception as e:
            self.logger.error(f"FMP error: {e}")
            return None

    def _filter_by_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Filter DataFrame by period"""
        if df.empty:
            return df
            
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
            return df  # Return all data
        else:
            start_date = end_date - timedelta(days=1825)  # Default 5y
        
        return df[df.index >= start_date]

    def _create_sample_data(self, symbol: str, period: str) -> pd.DataFrame:
        """
        Create sample stock data as absolute fallback when all APIs fail
        """
        self.logger.info(f"FALLBACK: Creating sample data for {symbol}")
        
        # Calculate number of days
        if period == "1y":
            days = 365
        elif period == "2y":
            days = 730
        elif period == "5y":
            days = 1825
        elif period == "10y":
            days = 3650
        else:
            days = 1825  # Default 5y
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
        
        # Generate realistic stock price data
        np.random.seed(hash(symbol) % 2**32)  # Consistent data per symbol
        
        # Starting values based on symbol
        if symbol in ['AAPL', 'MSFT', 'GOOGL']:
            base_price = 150
        elif symbol in ['AMZN']:
            base_price = 120
        elif symbol in ['TSLA']:
            base_price = 200
        else:
            base_price = 100
        
        # Generate price movement
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Small daily returns with volatility
        price_series = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        data = []
        for i, date in enumerate(dates):
            close = price_series[i]
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = close * (1 + np.random.normal(0, 0.005))
            volume = int(np.random.normal(50000000, 20000000))  # Random volume
            
            data.append({
                'Open': max(open_price, 0.01),
                'High': max(high, open_price, close),
                'Low': min(low, open_price, close),
                'Close': max(close, 0.01),
                'Volume': max(volume, 1000)
            })
        
        df = pd.DataFrame(data, index=dates)
        df.attrs['symbol'] = symbol
        df.attrs['period'] = period
        df.attrs['last_update'] = datetime.now()
        df.attrs['sample_data'] = True
        
        self.logger.info(f"FALLBACK SUCCESS: Created {len(df)} sample records for {symbol}")
        return df

    def get_stock_info(self, symbol: str) -> Dict:
        """Get basic stock information (simplified for fallback)"""
        try:
            # For now, return basic info since we're focusing on price data
            # In production, you could enhance this with additional API calls
            return {
                'name': f"{symbol} Inc.",
                'sector': 'Technology' if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'] else 'Unknown',
                'market_cap': 1000000000,  # 1B default
                'pe_ratio': 25.0,
                'dividend_yield': 0.02
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
            
            # Price change indicators
            data['Price_Change'] = data['Close'].pct_change()
            data['Price_Change_SMA'] = data['Price_Change'].rolling(window=5).mean()
            
            # Volatility
            data['Volatility'] = data['Close'].rolling(window=20).std()
            
            # Support and Resistance levels
            data['Support'] = data['Low'].rolling(window=20).min()
            data['Resistance'] = data['High'].rolling(window=20).max()
            
            self.logger.info(f"Added technical indicators, shape: {data.shape}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            return df
