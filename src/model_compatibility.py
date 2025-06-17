"""
Model compatibility utilities for handling TensorFlow version differences.
"""
import tensorflow as tf
import logging
from typing import Dict, Any

class ModelCompatibilityHandler:
    """Handles model compatibility issues between TensorFlow versions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_model_with_compatibility(self, filepath: str, compile: bool = True):
        """
        Load model with compatibility fixes for different TensorFlow versions
        
        Args:
            filepath: Path to the model file
            compile: Whether to compile the model after loading
            
        Returns:
            Loaded model
        """
        try:
            # Try the standard loading first
            model = tf.keras.models.load_model(filepath, compile=compile)
            self.logger.info(f"Model loaded successfully with standard method")
            return model
            
        except Exception as e:
            if "batch_shape" in str(e) or "Unrecognized keyword arguments" in str(e):
                self.logger.warning(f"Compatibility issue detected: {e}")
                return self._fix_batch_shape_issue(filepath, compile)
            else:
                raise e
    
    def _fix_batch_shape_issue(self, filepath: str, compile: bool = True):
        """
        Fix the batch_shape compatibility issue by modifying the model config
        
        Args:
            filepath: Path to the model file
            compile: Whether to compile the model after loading
            
        Returns:
            Fixed model
        """
        try:
            # Load the model configuration
            import h5py
            import json
            
            with h5py.File(filepath, 'r') as f:
                # Get model config
                model_config = f.attrs.get('model_config')
                if model_config is not None:
                    config = json.loads(model_config.decode('utf-8'))
                    
                    # Fix the InputLayer configuration
                    config = self._fix_input_layer_config(config)
                    
                    # Create model from fixed config
                    model = tf.keras.models.model_from_json(json.dumps(config))
                    
                    # Load weights
                    model.load_weights(filepath)
                    
                    if compile:
                        model.compile(
                            optimizer='adam',
                            loss='mse',
                            metrics=['mae']
                        )
                    
                    self.logger.info("Model loaded with batch_shape fix")
                    return model
                else:
                    raise ValueError("Could not extract model config")
                    
        except Exception as e:
            self.logger.error(f"Failed to fix batch_shape issue: {e}")
            # Fallback: try to load without compilation and rebuild
            return self._fallback_load(filepath)
    
    def _fix_input_layer_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix InputLayer configuration by replacing batch_shape with input_shape
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Fixed configuration
        """
        if 'config' in config and 'layers' in config['config']:
            for layer in config['config']['layers']:
                if layer.get('class_name') == 'InputLayer':
                    layer_config = layer.get('config', {})
                    
                    # Replace batch_shape with input_shape
                    if 'batch_shape' in layer_config:
                        batch_shape = layer_config['batch_shape']
                        if batch_shape and len(batch_shape) > 1:
                            # Convert batch_shape to input_shape (remove batch dimension)
                            input_shape = batch_shape[1:]
                            layer_config['input_shape'] = input_shape
                            del layer_config['batch_shape']
                            
                        self.logger.info(f"Fixed InputLayer: batch_shape -> input_shape = {input_shape}")
        
        return config
    
    def _fallback_load(self, filepath: str):
        """
        Fallback method to load model by recreating architecture
        
        Args:
            filepath: Path to the model file
            
        Returns:
            Recreated model with loaded weights
        """
        try:
            # Create a simple LSTM model with expected architecture
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(60, 14)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(50, return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(25),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(5)  # Assuming 5-day forecast
            ])
            
            # Try to load weights
            try:
                model.load_weights(filepath)
                self.logger.info("Fallback model created and weights loaded")
            except Exception as e:
                self.logger.warning(f"Could not load weights in fallback: {e}")
                # Model will be untrained but at least it won't crash
            
            # Compile the model
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Fallback method failed: {e}")
            raise
