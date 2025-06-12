"""
FastAPI backend for stock price prediction API.
Optional deployment as REST API service.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
from pathlib import Path
import asyncio
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_collector import StockDataCollector, FeatureEngineer
from lstm_model import LSTMStockPredictor
from utils import ModelManager, setup_logging

# Initialize FastAPI
app = FastAPI(
    title="Stock Price Predictor API",
    description="Advanced LSTM-based stock price prediction API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
logger = setup_logging()
data_collector = StockDataCollector()
feature_engineer = FeatureEngineer()
model_manager = ModelManager()

# Pydantic models
class PredictionRequest(BaseModel):
    symbol: str
    forecast_days: int = 5
    period: str = "5y"
    use_attention: bool = False

class PredictionResponse(BaseModel):
    symbol: str
    predictions: List[float]
    dates: List[str]
    confidence: List[float]
    current_price: float
    model_accuracy: Optional[float] = None

class TrainingRequest(BaseModel):
    symbol: str
    period: str = "5y"
    epochs: int = 50
    use_attention: bool = False

class TrainingResponse(BaseModel):
    symbol: str
    status: str
    model_id: str
    training_metrics: Dict
    training_time: float

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Stock Price Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "Make price predictions",
            "/train": "Train new model",
            "/models": "List available models",
            "/health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test data connection
        test_data = data_collector.fetch_stock_data("AAPL", "5d")
        data_status = "healthy" if test_data is not None else "unhealthy"
        
        return {
            "status": "healthy",
            "data_connection": data_status,
            "timestamp": str(asyncio.get_event_loop().time())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock_price(request: PredictionRequest):
    """Make stock price predictions"""
    try:
        logger.info(f"Prediction request for {request.symbol}")
        
        # Try to load existing model
        try:
            model, model_info = model_manager.load_best_model(request.symbol)
            model_accuracy = model_info['metrics'].get('directional_accuracy', 0)
        except:
            # Train new model if none exists
            logger.info(f"No existing model for {request.symbol}, training new one...")
            model = await train_model_async(request.symbol, request.period, request.use_attention)
            model_accuracy = None
        
        # Get recent data
        data = data_collector.fetch_stock_data(request.symbol, request.period)
        if data is None:
            raise HTTPException(status_code=400, detail=f"Could not fetch data for {request.symbol}")
        
        # Add technical indicators
        enhanced_data = feature_engineer.add_technical_indicators(data)
        
        # Get current price
        current_price = float(data['Close'].iloc[-1])
        
        # Prepare data for prediction
        from data_collector import DataPreprocessor
        from config import DATA_CONFIG
        
        preprocessor = DataPreprocessor(
            lookback_days=60,
            forecast_days=request.forecast_days
        )
        
        # Use recent data
        recent_data = enhanced_data[DATA_CONFIG.feature_columns].dropna().tail(60)
        
        # Normalize data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(recent_data)
        
        # Make prediction
        close_scaler = MinMaxScaler()
        close_scaler.fit(recent_data[['Close']])
        
        predictions = model.predict_next_days(normalized_data, close_scaler)
        
        response = PredictionResponse(
            symbol=request.symbol,
            predictions=predictions['predictions'],
            dates=[str(date) for date in predictions['dates']],
            confidence=predictions['confidence'],
            current_price=current_price,
            model_accuracy=model_accuracy
        )
        
        logger.info(f"Prediction completed for {request.symbol}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed for {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train a new model for a stock"""
    try:
        logger.info(f"Training request for {request.symbol}")
        
        # Start training in background
        background_tasks.add_task(
            train_model_background,
            request.symbol,
            request.period,
            request.epochs,
            request.use_attention
        )
        
        return TrainingResponse(
            symbol=request.symbol,
            status="training_started",
            model_id="pending",
            training_metrics={},
            training_time=0.0
        )
        
    except Exception as e:
        logger.error(f"Training failed for {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def train_model_async(symbol: str, period: str, use_attention: bool = False):
    """Async model training"""
    from train import TrainingPipeline
    
    pipeline = TrainingPipeline()
    results = pipeline.train_single_stock(
        symbol=symbol,
        period=period,
        save_model=True,
        use_attention=use_attention
    )
    
    return results

def train_model_background(symbol: str, period: str, epochs: int, use_attention: bool):
    """Background training task"""
    try:
        from train import TrainingPipeline
        from config import MODEL_CONFIG
        
        # Customize config for API training
        config = MODEL_CONFIG.__dict__.copy()
        config['epochs'] = epochs
        
        pipeline = TrainingPipeline(config)
        results = pipeline.train_single_stock(
            symbol=symbol,
            period=period,
            save_model=True,
            use_attention=use_attention
        )
        
        logger.info(f"Background training completed for {symbol}")
        
    except Exception as e:
        logger.error(f"Background training failed for {symbol}: {str(e)}")

@app.get("/models")
async def list_models(symbol: Optional[str] = None):
    """List available models"""
    try:
        models = model_manager.list_models(symbol)
        return {
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stocks/popular")
async def get_popular_stocks():
    """Get list of popular stocks by category"""
    from config import POPULAR_STOCKS
    return POPULAR_STOCKS

@app.get("/stocks/{symbol}/info")
async def get_stock_info(symbol: str):
    """Get basic stock information"""
    try:
        info = data_collector.get_stock_info(symbol)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
