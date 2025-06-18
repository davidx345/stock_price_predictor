"""
User-friendly error handling and recovery for the stock predictor application
"""
import logging
import traceback
from typing import Dict, Any, Optional
import streamlit as st

class ErrorHandler:
    """Centralized error handling with user-friendly messages"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def handle_data_error(self, error: Exception, symbol: str = "Unknown") -> Dict[str, Any]:
        """Handle data collection/validation errors"""
        error_msg = str(error)
        
        # Map common errors to user-friendly messages
        if "404" in error_msg or "not found" in error_msg.lower():
            return {
                'user_message': f"❌ Stock symbol '{symbol}' not found",
                'suggestions': [
                    "Check if the symbol is correct (e.g., 'AAPL' for Apple)",
                    "Try a different stock symbol from major exchanges",
                    "Verify the symbol is actively traded"
                ],
                'severity': 'error'
            }
        elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            return {
                'user_message': "⚠️ Network connection issue",
                'suggestions': [
                    "Check your internet connection",
                    "Try again in a few moments",
                    "The data provider might be temporarily unavailable"
                ],
                'severity': 'warning'
            }
        elif "rate limit" in error_msg.lower() or "429" in error_msg:
            return {
                'user_message': "⚠️ Too many requests - rate limited",
                'suggestions': [
                    "Wait a few minutes before trying again",
                    "Consider using a different data source",
                    "Reduce the frequency of requests"
                ],
                'severity': 'warning'
            }
        elif "insufficient data" in error_msg.lower():
            return {
                'user_message': f"⚠️ Not enough data for '{symbol}'",
                'suggestions': [
                    "Try a longer time period (e.g., 2y instead of 1y)",
                    "Check if the stock has been trading long enough",
                    "Consider using a different stock symbol"
                ],
                'severity': 'warning'
            }
        else:
            return {
                'user_message': f"❌ Error loading data for '{symbol}'",
                'suggestions': [
                    "Try a different stock symbol",
                    "Check your internet connection",
                    "Try again in a few moments"
                ],
                'severity': 'error'
            }
    
    def handle_model_error(self, error: Exception, model_type: str = "prediction") -> Dict[str, Any]:
        """Handle model loading/prediction errors"""
        error_msg = str(error)
        
        if "model not found" in error_msg.lower() or "no such file" in error_msg.lower():
            return {
                'user_message': f"❌ {model_type.capitalize()} model not available",
                'suggestions': [
                    "The model may need to be trained first",
                    "Check if the model file exists",
                    "Try using a different prediction method"
                ],
                'severity': 'error'
            }
        elif "input shape" in error_msg.lower() or "dimension" in error_msg.lower():
            return {
                'user_message': "⚠️ Data format mismatch with model",
                'suggestions': [
                    "The model expects different input format",
                    "Try reprocessing the data",
                    "Check if technical indicators are properly calculated"
                ],
                'severity': 'warning'
            }
        elif "memory" in error_msg.lower() or "resource" in error_msg.lower():
            return {
                'user_message': "⚠️ Insufficient system resources",
                'suggestions': [
                    "Try reducing the data size or prediction period",
                    "Close other applications to free memory",
                    "Consider using a simpler model"
                ],
                'severity': 'warning'
            }
        else:
            return {
                'user_message': f"❌ Error in {model_type} processing",
                'suggestions': [
                    "Try reloading the page",
                    "Check if all required data is available",
                    "Contact support if the issue persists"
                ],
                'severity': 'error'
            }
    
    def handle_validation_error(self, error: Exception, data_type: str = "stock") -> Dict[str, Any]:
        """Handle data validation errors"""
        error_msg = str(error)
        
        if "missing columns" in error_msg.lower():
            return {
                'user_message': f"⚠️ Missing required data fields",
                'suggestions': [
                    "The data source may be incomplete",
                    "Try a different time period",
                    "Some technical indicators will be calculated automatically"
                ],
                'severity': 'warning'
            }
        elif "insufficient data" in error_msg.lower():
            return {
                'user_message': f"⚠️ Not enough {data_type} data",
                'suggestions': [
                    "Try extending the time period",
                    "Check if the symbol has sufficient trading history",
                    "Consider using a different stock"
                ],
                'severity': 'warning'
            }
        else:
            return {
                'user_message': f"⚠️ Data validation issue",
                'suggestions': [
                    "The data may need preprocessing",
                    "Try refreshing the data",
                    "Check the data source"
                ],
                'severity': 'warning'
            }
    
    def display_error(self, error_info: Dict[str, Any]) -> None:
        """Display error information to user using Streamlit"""
        severity = error_info.get('severity', 'error')
        message = error_info.get('user_message', 'An error occurred')
        suggestions = error_info.get('suggestions', [])
        
        if severity == 'error':
            st.error(message)
        elif severity == 'warning':
            st.warning(message)
        else:
            st.info(message)
        
        if suggestions:
            st.info("💡 **Suggestions:**")
            for suggestion in suggestions:
                st.info(f"• {suggestion}")
    
    def safe_execute(self, func, *args, error_type: str = "general", **kwargs) -> Any:
        """Safely execute a function with error handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"{error_type} error: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Handle different error types
            if error_type == "data":
                symbol = kwargs.get('symbol', args[0] if args else 'Unknown')
                error_info = self.handle_data_error(e, symbol)
            elif error_type == "model":
                model_type = kwargs.get('model_type', 'prediction')
                error_info = self.handle_model_error(e, model_type)
            elif error_type == "validation":
                data_type = kwargs.get('data_type', 'stock')
                error_info = self.handle_validation_error(e, data_type)
            else:
                error_info = {
                    'user_message': f"❌ An unexpected error occurred",
                    'suggestions': [
                        "Try refreshing the page",
                        "Check your input data",
                        "Contact support if the issue persists"
                    ],
                    'severity': 'error'
                }
            
            self.display_error(error_info)
            return None

# Global error handler instance
error_handler = ErrorHandler()
