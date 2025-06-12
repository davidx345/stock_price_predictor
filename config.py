"""
Configuration settings for the Stock Price Predictor application.
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

@dataclass
class ModelConfig:
    """Configuration for LSTM model"""
    lookback_days: int = 60
    forecast_days: int = 5
    lstm_units: List[int] = None
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    
    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [100, 50, 25]

@dataclass
class DataConfig:
    """Configuration for data processing"""
    default_period: str = "5y"
    min_data_points: int = 1000
    train_ratio: float = 0.8
    feature_columns: List[str] = None
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_20', 'EMA_12', 'RSI', 'MACD', 'MACD_signal',
                'BB_upper', 'BB_lower', 'volatility', 'price_change'
            ]

@dataclass
class AppConfig:
    """Configuration for Streamlit app"""
    title: str = "🚀 AI Stock Price Predictor"
    page_icon: str = "📈"
    layout: str = "wide"
    sidebar_width: int = 300
    chart_height: int = 500
    
# Application settings
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
APP_CONFIG = AppConfig()

# API Keys (load from environment)
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Popular stocks for quick selection
POPULAR_STOCKS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"],
    "Finance": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
    "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
    "Consumer": ["KO", "PEP", "WMT", "HD", "MCD"]
}

# Model parameters
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA"]

# Styling constants
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ff7f0e",
    "info": "#17a2b8",
    "background": "#ffffff",
    "sidebar": "#f8f9fa"
}

# Chart configurations
CHART_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "pan2d", "lasso2d", "select2d", "autoScale2d", "resetScale2d"
    ]
}
