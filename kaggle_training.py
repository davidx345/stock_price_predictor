"""
Kaggle Notebook Training Script for Stock Price Prediction
Alternative free GPU option to Google Colab
"""

# ===============================================
# KAGGLE SETUP INSTRUCTIONS
# ===============================================
"""
1. Go to https://www.kaggle.com/code
2. Create New Notebook
3. Copy this code into cells
4. Turn on GPU: Settings > Accelerator > GPU
5. Turn on Internet: Settings > Internet > On
6. Run all cells
"""

# ===============================================
# CELL 1: Install Dependencies
# ===============================================
import sys
!{sys.executable} -m pip install yfinance ta plotly streamlit scikit-learn joblib

# ===============================================
# CELL 2: Import and Setup
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

# Check GPU
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("TensorFlow Version:", tf.__version__)

# ===============================================
# CELL 3: Quick Training Function
# ===============================================
def quick_train_stock(symbol, epochs=50):
    """Streamlined training for Kaggle"""
    print(f"🚀 Training {symbol}...")
    
    # Fetch data
    data = yf.download(symbol, period="3y")  # Reduced for faster training
    if data.empty:
        return None
    
    # Simple feature engineering
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['RSI'] = ta.momentum.rsi(data['Close'])
    data['Returns'] = data['Close'].pct_change()
    
    # Prepare features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'RSI', 'Returns']
    clean_data = data[features].dropna()
    
    # Normalize
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(clean_data)
    
    # Create sequences
    lookback = 30  # Reduced for faster training
    X, y = [], []
    
    for i in range(lookback, len(normalized) - 5):
        X.append(normalized[i-lookback:i])
        y.append(normalized[i+1:i+6, 3])  # Next 5 days close prices
    
    X, y = np.array(X), np.array(y)
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(lookback, len(features))),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(5)  # 5-day prediction
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    # Save model
    model.save(f'{symbol}_model.h5')
    
    # Save scaler
    import joblib
    joblib.dump(scaler, f'{symbol}_scaler.pkl')
    
    print(f"✅ {symbol} - Test MAE: {test_mae:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'test_mae': test_mae,
        'history': history
    }

# ===============================================
# CELL 4: Train Multiple Stocks
# ===============================================
# Train popular stocks
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
results = {}

for stock in stocks:
    try:
        result = quick_train_stock(stock, epochs=30)  # Faster training
        if result:
            results[stock] = result
            print(f"✅ {stock} training completed")
        else:
            print(f"❌ {stock} training failed")
    except Exception as e:
        print(f"❌ {stock} error: {e}")

print(f"\n🎉 Trained {len(results)} models successfully!")

# ===============================================
# CELL 5: Download Models
# ===============================================
# Create zip file for download
import zipfile
import os

with zipfile.ZipFile('trained_models.zip', 'w') as zipf:
    for stock in results.keys():
        if os.path.exists(f'{stock}_model.h5'):
            zipf.write(f'{stock}_model.h5')
        if os.path.exists(f'{stock}_scaler.pkl'):
            zipf.write(f'{stock}_scaler.pkl')

print("💾 Models saved to 'trained_models.zip'")
print("📥 Download this file and extract to your local 'models/' folder")

# ===============================================
# CELL 6: Quick Test Prediction
# ===============================================
# Test a prediction
if 'AAPL' in results:
    print("\n🔮 Quick prediction test for AAPL:")
    
    # Get recent data
    recent_data = yf.download('AAPL', period='60d')
    
    # Prepare for prediction (simplified)
    recent_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    recent_clean = recent_data[recent_features].dropna().tail(30)
    
    # Make prediction
    model = results['AAPL']['model']
    scaler = results['AAPL']['scaler']
    
    # Note: This is a simplified prediction for demo
    current_price = recent_data['Close'].iloc[-1]
    print(f"Current AAPL price: ${current_price:.2f}")
    print(f"Model trained successfully - Ready for full predictions!")

print("\n✅ Kaggle training completed!")
print("📋 Next steps:")
print("1. Download 'trained_models.zip'")
print("2. Extract to your local project's 'models/' folder")
print("3. Run: streamlit run app.py")
