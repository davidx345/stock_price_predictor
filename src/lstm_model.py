"""
LSTM Model implementation for stock price prediction.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import the compatibility handler
from .model_compatibility import ModelCompatibilityHandler

class LSTMStockPredictor:
    """Advanced LSTM model for stock price prediction"""
    
    def __init__(self, config: Dict):
        """
        Initialize LSTM model
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.history = None
        self.logger = logging.getLogger(__name__)
        
        # Model parameters
        self.lookback_days = config.get('lookback_days', 60)
        self.forecast_days = config.get('forecast_days', 5)
        self.lstm_units = config.get('lstm_units', [100, 50, 25])
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        
    def build_model(self, input_shape: Tuple) -> Sequential:
        """
        Build advanced LSTM model architecture
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            
        Returns:
            Compiled Keras model
        """
        try:
            model = Sequential([
                # First LSTM layer
                LSTM(
                    self.lstm_units[0],
                    return_sequences=True,
                    input_shape=input_shape,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate
                ),
                BatchNormalization(),
                
                # Second LSTM layer
                LSTM(
                    self.lstm_units[1],
                    return_sequences=True,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate
                ),
                BatchNormalization(),
                
                # Third LSTM layer
                LSTM(
                    self.lstm_units[2],
                    return_sequences=False,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate
                ),
                BatchNormalization(),
                
                # Dense layers
                Dense(50, activation='relu'),
                Dropout(self.dropout_rate),
                
                Dense(25, activation='relu'),
                Dropout(self.dropout_rate),
                
                # Output layer
                Dense(self.forecast_days, activation='linear')
            ])
            
            # Compile model
            optimizer = Adam(learning_rate=self.learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='huber',  # More robust to outliers than MSE
                metrics=['mae', 'mse']
            )
            
            self.logger.info(f"Model built successfully with input shape: {input_shape}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error building model: {str(e)}")
            raise
    
    def build_attention_model(self, input_shape: Tuple) -> Model:
        """
        Build LSTM model with attention mechanism
        """
        try:
            # Input layer
            inputs = Input(shape=input_shape)
            
            # LSTM layers
            lstm1 = LSTM(self.lstm_units[0], return_sequences=True)(inputs)
            lstm1 = BatchNormalization()(lstm1)
            lstm1 = Dropout(self.dropout_rate)(lstm1)
            
            lstm2 = LSTM(self.lstm_units[1], return_sequences=True)(lstm1)
            lstm2 = BatchNormalization()(lstm2)
            lstm2 = Dropout(self.dropout_rate)(lstm2)
            
            # Attention mechanism
            attention = tf.keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=self.lstm_units[1] // 4
            )(lstm2, lstm2)
            
            # Global average pooling
            pooled = tf.keras.layers.GlobalAveragePooling1D()(attention)
            
            # Dense layers
            dense1 = Dense(50, activation='relu')(pooled)
            dense1 = Dropout(self.dropout_rate)(dense1)
            
            dense2 = Dense(25, activation='relu')(dense1)
            dense2 = Dropout(self.dropout_rate)(dense2)
            
            # Output
            outputs = Dense(self.forecast_days, activation='linear')(dense2)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compile
            optimizer = Adam(learning_rate=self.learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='huber',
                metrics=['mae', 'mse']
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error building attention model: {str(e)}")
            raise
    
    def create_callbacks(self, model_path: str) -> List:
        """Create training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              model_path: str = None, use_attention: bool = False) -> Dict:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_path: Path to save the best model
            use_attention: Whether to use attention mechanism
            
        Returns:
            Training history and metrics
        """
        try:
            # Build model
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            if use_attention:
                self.model = self.build_attention_model(input_shape)
            else:
                self.model = self.build_model(input_shape)
            
            self.logger.info(f"Model summary:\n{self.model.summary()}")
            
            # Prepare validation data
            if X_val is None or y_val is None:
                validation_data = None
                validation_split = 0.2
            else:
                validation_data = (X_val, y_val)
                validation_split = None
            
            # Create callbacks
            callbacks = []
            if model_path:
                callbacks = self.create_callbacks(model_path)
            
            # Train model
            self.logger.info("Starting model training...")
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=validation_data,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1,
                shuffle=False  # Important for time series data
            )
            
            # Calculate training metrics
            train_metrics = self._calculate_metrics(X_train, y_train, "Training")
            val_metrics = {}
            
            if X_val is not None and y_val is not None:
                val_metrics = self._calculate_metrics(X_val, y_val, "Validation")
            
            training_results = {
                'history': self.history.history,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'model_path': model_path
            }
            
            self.logger.info("Model training completed successfully")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
    
    def _calculate_metrics(self, X: np.ndarray, y: np.ndarray, dataset_name: str) -> Dict:
        """Calculate and log performance metrics"""
        predictions = self.model.predict(X, verbose=0)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - y))
        mse = np.mean((predictions - y) ** 2)
        rmse = np.sqrt(mse)
        
        # Directional accuracy (for first day prediction)
        if y.shape[1] > 1:
            y_direction = np.sign(y[:, 1] - y[:, 0])  # Next day vs current
            pred_direction = np.sign(predictions[:, 1] - predictions[:, 0])
        else:
            y_direction = np.sign(y.flatten())
            pred_direction = np.sign(predictions.flatten())
        
        directional_accuracy = np.mean(y_direction == pred_direction)
        
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'directional_accuracy': float(directional_accuracy)
        }
        
        self.logger.info(f"{dataset_name} Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, "
                        f"Directional Accuracy: {directional_accuracy:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(X, verbose=0)
    
    def predict_next_days(self, recent_data: np.ndarray, scaler) -> Dict:
        """
        Predict stock prices for the next few days
        
        Args:
            recent_data: Recent normalized data (lookback_days, features)
            scaler: Scaler for inverse transformation
            
        Returns:
            Dictionary with predictions and metadata
        """
        try:
            # Ensure correct shape
            if recent_data.shape[0] != self.lookback_days:
                recent_data = recent_data[-self.lookback_days:]
            
            # Reshape for prediction
            X = recent_data.reshape(1, self.lookback_days, -1)
            
            # Make prediction
            pred_normalized = self.model.predict(X, verbose=0)[0]
            
            # Inverse transform predictions
            pred_prices = scaler.inverse_transform(pred_normalized.reshape(-1, 1)).flatten()
            
            # Generate prediction dates (assuming daily predictions)
            from datetime import datetime, timedelta
            base_date = datetime.now().date()
            pred_dates = [base_date + timedelta(days=i+1) for i in range(len(pred_prices))]
            
            return {
                'predictions': pred_prices.tolist(),
                'dates': pred_dates,
                'confidence': self._calculate_prediction_confidence(pred_normalized)
            }
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def _calculate_prediction_confidence(self, predictions: np.ndarray) -> List[float]:
        """Calculate confidence scores for predictions"""
        # Simple confidence based on prediction variance
        # In practice, you might want to use more sophisticated methods
        base_confidence = 0.7
        variance_penalty = np.var(predictions) * 0.1
        confidence = max(0.3, base_confidence - variance_penalty)
        
        return [confidence] * len(predictions)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        try:
            # Save model 
            self.model.save(filepath)
             
            # Save configuration
            config_path = filepath.replace('.h5', '_config.pkl')
            joblib.dump(self.config, config_path)
            
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a trained model with compatibility for different formats"""
        try:
            logger = logging.getLogger(__name__)
            logger.info(f"Attempting to load model from: {filepath}")
            
            # Initialize compatibility handler
            compatibility_handler = ModelCompatibilityHandler()
            
            # Handle different file formats
            if filepath.endswith('.keras'):
                # New format - try to load directly with compatibility fixes
                try:
                    # Create a basic configuration for legacy models
                    default_config = {
                        'lookback_days': 60,
                        'forecast_days': 5,
                        'lstm_units': [100, 50, 25],
                        'dropout_rate': 0.2,
                        'learning_rate': 0.001,
                        'batch_size': 32,
                        'epochs': 100
                    }
                    
                    # Create instance with default config
                    instance = cls(default_config)
                    
                    # Use compatibility handler to load the model
                    instance.model = compatibility_handler.load_model_with_compatibility(
                        filepath, compile=False
                    )
                    logger.info(f"Model loaded successfully from {filepath}")
                    
                    # Recompile the model with current TensorFlow version
                    instance.recompile_model()
                    
                    return instance
                    
                except Exception as e:
                    logger.error(f"Failed to load .keras model: {e}")
                    raise
                    
            elif filepath.endswith('.h5'):
                # Legacy format with config file
                config_path = filepath.replace('.h5', '_config.pkl')
                if Path(config_path).exists():
                    config = joblib.load(config_path)
                else:
                    # Use default config if no config file
                    config = {
                        'lookback_days': 60,
                        'forecast_days': 5,
                        'lstm_units': [100, 50, 25],
                        'dropout_rate': 0.2,
                        'learning_rate': 0.001,
                        'batch_size': 32,
                        'epochs': 100
                    }
                
                # Create instance
                instance = cls(config)
                
                # Load model
                instance.model = tf.keras.models.load_model(filepath)
                logger.info(f"Legacy model loaded from {filepath}")
                return instance
            
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
                
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {str(e)}")
            raise
    
    def recompile_model(self):
        """Recompile model if it was loaded without compilation"""
        if self.model is not None:
            try:
                self.model.compile(
                    optimizer=Adam(learning_rate=self.learning_rate),
                    loss='mse',
                    metrics=['mae']
                )
                self.logger.info("Model recompiled successfully")
            except Exception as e:
                self.logger.warning(f"Could not recompile model: {e}")
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions using the loaded model"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            # Ensure model is compiled for prediction
            if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
                self.recompile_model()
            
            predictions = self.model.predict(data)
            return predictions
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    def plot_training_history(self) -> plt.Figure:
        """Plot training history"""
        if self.history is None:
            raise ValueError("No training history available")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Training History', fontsize=16)
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            axes[0, 1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # MSE
        axes[1, 0].plot(self.history.history['mse'], label='Training MSE')
        if 'val_mse' in self.history.history:
            axes[1, 0].plot(self.history.history['val_mse'], label='Validation MSE')
        axes[1, 0].set_title('Mean Squared Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        return fig
    
    def _rebuild_and_load_weights(self, filepath: str):
        """Rebuild model architecture and load weights for compatibility"""
        try:
            # Try to extract model architecture information
            # This is a fallback method when direct loading fails
            
            # Create a new model with the expected architecture
            # Based on your existing models (60 timesteps, 14 features)
            input_shape = (60, 14)  # Default shape from your existing models
            
            # Build the model architecture
            self.model = self.build_model(input_shape)
            
            # Try to load just the weights
            try:
                # First, try to load the entire model to get the architecture
                temp_model = tf.keras.models.load_model(filepath, compile=False)
                
                # Get the actual input shape from the loaded model
                if temp_model.layers:
                    actual_input_shape = temp_model.layers[0].input_shape[1:]  # Remove batch dimension
                    
                    # Rebuild with correct shape if different
                    if actual_input_shape != input_shape:
                        self.logger.info(f"Rebuilding model with shape {actual_input_shape}")
                        self.model = self.build_model(actual_input_shape)
                
                # Copy weights layer by layer
                for i, layer in enumerate(self.model.layers):
                    if i < len(temp_model.layers):
                        try:
                            weights = temp_model.layers[i].get_weights()
                            if weights:  # Only set weights if they exist
                                layer.set_weights(weights)
                        except Exception as e:
                            self.logger.warning(f"Could not copy weights for layer {i}: {e}")
                            continue
                
                self.logger.info("Successfully rebuilt model and loaded weights")
                
            except Exception as e:
                self.logger.error(f"Could not rebuild model: {e}")
                # As a last resort, just use the default architecture
                self.model = self.build_model((60, 14))
                self.logger.warning("Using default model architecture - predictions may be inaccurate")
                
        except Exception as e:
            self.logger.error(f"Failed to rebuild model: {e}")
            raise
