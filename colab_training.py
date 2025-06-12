"""
Google Colab Training Notebook for Stock Price Prediction
Upload this notebook to Google Colab for free GPU training
"""

# ===============================================
# CELL 1: Setup and Installation
# ===============================================
# Run this cell first to install dependencies
!pip install yfinance ta tensorflow plotly streamlit scikit-learn joblib

# Mount Google Drive to save models
from google.colab import drive
drive.mount('/content/drive')

# Create project directory in Drive
!mkdir -p '/content/drive/MyDrive/stock_models'

# ===============================================
# CELL 2: Download Project Code
# ===============================================
# Clone your project or upload the src folder
# Option 1: If you have it on GitHub
# !git clone https://github.com/yourusername/stock_price_predictor.git
# %cd stock_price_predictor

# Option 2: Upload files manually to Colab, then:
import sys
sys.path.append('/content')  # Adjust path as needed

# ===============================================
# CELL 3: Import Required Modules
# ===============================================
import tensorflow as tf
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("TensorFlow Version:", tf.__version__)

# ===============================================
# CELL 4: Model Configuration
# ===============================================
MODEL_CONFIG = {
    'lookback_days': 60,
    'forecast_days': 5,
    'lstm_units': [100, 50, 25],
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'validation_split': 0.2
}

# Data configuration
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_20', 'EMA_12', 'RSI', 'MACD', 'MACD_signal',
    'BB_upper', 'BB_lower', 'volatility', 'price_change'
]

# ===============================================
# CELL 5: Data Collection Functions
# ===============================================
def fetch_stock_data(symbol, period="5y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        print(f"📊 Fetching data for {symbol}...")
        data = yf.download(symbol, period=period)
        print(f"✅ Fetched {len(data)} records")
        return data
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    print("🔧 Adding technical indicators...")
    data = df.copy()
    
    # Moving averages
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['EMA_12'] = ta.trend.ema_indicator(data['Close'], window=12)
    
    # RSI
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['BB_upper'] = bollinger.bollinger_hband()
    data['BB_lower'] = bollinger.bollinger_lband()
    
    # Price features
    data['price_change'] = data['Close'].pct_change()
    data['volatility'] = data['price_change'].rolling(window=20).std()
    
    print(f"✅ Added {len(data.columns) - len(df.columns)} technical indicators")
    return data

# ===============================================
# CELL 6: Data Preprocessing Functions
# ===============================================
def create_sequences(data, lookback_days, forecast_days, target_col_idx=3):
    """Create sequences for LSTM training"""
    X, y = [], []
    
    for i in range(lookback_days, len(data) - forecast_days + 1):
        X.append(data[i-lookback_days:i])
        y.append(data[i:i+forecast_days, target_col_idx])
    
    return np.array(X), np.array(y)

def prepare_data_for_training(df, feature_columns, config):
    """Complete data preparation pipeline"""
    print("⚙️ Preparing data for training...")
    
    # Clean data
    clean_data = df[feature_columns].dropna()
    print(f"Clean data shape: {clean_data.shape}")
    
    # Normalize data
    scalers = {}
    normalized_data = clean_data.copy()
    
    for column in clean_data.columns:
        scaler = MinMaxScaler()
        normalized_data[column] = scaler.fit_transform(clean_data[[column]])
        scalers[column] = scaler
    
    # Convert to numpy array
    data_array = normalized_data.values
    
    # Create sequences
    target_idx = feature_columns.index('Close')
    X, y = create_sequences(data_array, config['lookback_days'], config['forecast_days'], target_idx)
    
    # Train-test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"✅ Data prepared - Train: {X_train.shape}, Test: {X_test.shape}")
    
    return {
        'X_train': X_train,
        'X_test': X_test, 
        'y_train': y_train,
        'y_test': y_test,
        'scalers': scalers,
        'close_scaler': scalers['Close']
    }

# ===============================================
# CELL 7: LSTM Model Definition
# ===============================================
def build_lstm_model(input_shape, config):
    """Build LSTM model"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        # First LSTM layer
        LSTM(config['lstm_units'][0], return_sequences=True, input_shape=input_shape,
             dropout=config['dropout_rate'], recurrent_dropout=config['dropout_rate']),
        BatchNormalization(),
        
        # Second LSTM layer  
        LSTM(config['lstm_units'][1], return_sequences=True,
             dropout=config['dropout_rate'], recurrent_dropout=config['dropout_rate']),
        BatchNormalization(),
        
        # Third LSTM layer
        LSTM(config['lstm_units'][2], return_sequences=False,
             dropout=config['dropout_rate'], recurrent_dropout=config['dropout_rate']),
        BatchNormalization(),
        
        # Dense layers
        Dense(50, activation='relu'),
        Dropout(config['dropout_rate']),
        Dense(25, activation='relu'),
        Dropout(config['dropout_rate']),
        
        # Output layer
        Dense(config['forecast_days'], activation='linear')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss='huber',
        metrics=['mae', 'mse']
    )
    
    return model

# ===============================================
# CELL 8: Training Function
# ===============================================
def train_model(symbol, config=MODEL_CONFIG, period="5y"):
    """Complete training pipeline"""
    print(f"🚀 Starting training for {symbol}")
    start_time = datetime.now()
    
    # Step 1: Collect data
    raw_data = fetch_stock_data(symbol, period)
    if raw_data is None:
        return None
    
    # Step 2: Feature engineering
    enhanced_data = add_technical_indicators(raw_data)
    
    # Step 3: Prepare data
    prepared_data = prepare_data_for_training(enhanced_data, FEATURE_COLUMNS, config)
    
    # Step 4: Build model
    input_shape = (prepared_data['X_train'].shape[1], prepared_data['X_train'].shape[2])
    model = build_lstm_model(input_shape, config)
    
    print(f"🧠 Model built with {model.count_params():,} parameters")
    
    # Step 5: Train model
    print("🏋️ Training model...")
    
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-7)
    ]
    
    history = model.fit(
        prepared_data['X_train'], prepared_data['y_train'],
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_split=config['validation_split'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 6: Evaluate model
    train_loss = model.evaluate(prepared_data['X_train'], prepared_data['y_train'], verbose=0)
    test_loss = model.evaluate(prepared_data['X_test'], prepared_data['y_test'], verbose=0)
    
    # Calculate directional accuracy
    predictions = model.predict(prepared_data['X_test'], verbose=0)
    actual = prepared_data['y_test']
    
    # For next day prediction (first forecast day)
    actual_direction = np.sign(actual[:, 0] - actual[:, -1]) if actual.shape[1] > 1 else np.sign(actual.flatten())
    pred_direction = np.sign(predictions[:, 0] - predictions[:, -1]) if predictions.shape[1] > 1 else np.sign(predictions.flatten())
    directional_accuracy = np.mean(actual_direction == pred_direction)
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    results = {
        'model': model,
        'history': history,
        'prepared_data': prepared_data,
        'metrics': {
            'train_loss': float(train_loss[0]),
            'test_loss': float(test_loss[0]),
            'train_mae': float(train_loss[1]),
            'test_mae': float(test_loss[1]),
            'directional_accuracy': float(directional_accuracy)
        },
        'training_time': training_time,
        'symbol': symbol,
        'config': config
    }
    
    print(f"✅ Training completed in {training_time:.1f} seconds")
    print(f"📊 Test MAE: {test_loss[1]:.4f}")
    print(f"🎯 Directional Accuracy: {directional_accuracy:.2%}")
    
    return results

# ===============================================
# CELL 9: Save Model Function
# ===============================================
def save_model_to_drive(results, symbol):
    """Save trained model to Google Drive"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"{symbol}_lstm_{timestamp}"
    
    # Save model
    model_path = f'/content/drive/MyDrive/stock_models/{model_name}.h5'
    results['model'].save(model_path)
    
    # Save metadata
    import json
    metadata = {
        'symbol': symbol,
        'metrics': results['metrics'],
        'config': results['config'],
        'training_time': results['training_time'],
        'trained_at': timestamp
    }
    
    metadata_path = f'/content/drive/MyDrive/stock_models/{model_name}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Model saved to: {model_path}")
    print(f"📋 Metadata saved to: {metadata_path}")
    
    return model_path, metadata_path

# ===============================================
# CELL 10: Train Your Models
# ===============================================
# Train multiple stocks
STOCKS_TO_TRAIN = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

trained_models = {}

for symbol in STOCKS_TO_TRAIN:
    print(f"\n{'='*50}")
    print(f"Training {symbol}")
    print(f"{'='*50}")
    
    try:
        results = train_model(symbol, MODEL_CONFIG, "5y")
        if results:
            model_path, metadata_path = save_model_to_drive(results, symbol)
            trained_models[symbol] = {
                'model_path': model_path,
                'metadata_path': metadata_path,
                'metrics': results['metrics']
            }
            print(f"✅ {symbol} training successful!")
        else:
            print(f"❌ {symbol} training failed!")
    except Exception as e:
        print(f"❌ {symbol} training error: {e}")

# ===============================================
# CELL 11: Training Summary
# ===============================================
print(f"\n{'='*60}")
print("🎉 TRAINING SUMMARY")
print(f"{'='*60}")

for symbol, info in trained_models.items():
    metrics = info['metrics']
    print(f"\n📈 {symbol}:")
    print(f"   MAE: {metrics['test_mae']:.4f}")
    print(f"   Accuracy: {metrics['directional_accuracy']:.2%}")
    print(f"   Model: {info['model_path']}")

print(f"\n✅ {len(trained_models)} models trained successfully!")
print("💾 All models saved to Google Drive: /MyDrive/stock_models/")
print("\n📥 Download the models folder to your local project!")

# ===============================================
# CELL 12: Download Instructions
# ===============================================
print("""
🔽 TO USE THESE MODELS IN YOUR LOCAL PROJECT:

1. Download the 'stock_models' folder from Google Drive
2. Copy the .h5 model files to your local 'models/' directory  
3. Copy the _metadata.json files as well
4. Run your Streamlit app: streamlit run app.py
5. The models will be automatically detected and loaded!

🎯 Your models are now ready for production use!
""")
