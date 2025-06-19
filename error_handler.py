"""
Error handling and debugging utilities for the stock predictor app
"""
import logging
import streamlit as st
import pandas as pd
from typing import List, Dict, Any

class AppErrorHandler:
    """Handles common app errors and provides user-friendly messages"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def handle_missing_columns_error(self, available_columns: List[str], 
                                   required_columns: List[str], 
                                   symbol: str) -> bool:
        """
        Handle missing columns error with user-friendly messages
        
        Returns:
            bool: True if error was handled gracefully, False if critical error
        """
        missing_columns = [col for col in required_columns if col not in available_columns]
        
        if missing_columns:
            st.error("🚨 **Data Processing Error**")
            
            with st.expander("📋 **Error Details**", expanded=True):
                st.markdown(f"**Symbol:** {symbol}")
                st.markdown(f"**Issue:** Missing technical indicator columns")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**❌ Missing Columns:**")
                    for col in missing_columns:
                        st.write(f"• {col}")
                
                with col2:
                    st.markdown("**✅ Available Columns:**")
                    available_indicators = [col for col in available_columns 
                                          if col in required_columns]
                    for col in available_indicators:
                        st.write(f"• {col}")
            
            st.markdown("### 🛠️ **Troubleshooting Steps:**")
            st.markdown("""
            1. **Try refreshing the data** - Click 'Fetch Data' again
            2. **Check internet connection** - Ensure stable connection for data APIs
            3. **Try different stock symbol** - Some stocks may have limited data
            4. **Wait a moment** - Technical indicators are being calculated...
            """)
            
            # Show fix attempt button
            if st.button("🔧 **Try to Fix Data**", type="primary"):
                st.info("Attempting to fix missing indicators...")
                return True
            
            return False
        
        return True
    
    def handle_prediction_error(self, error: Exception, symbol: str) -> None:
        """Handle prediction errors with helpful messages"""
        st.error("🚨 **Prediction Error**")
        
        with st.expander("📋 **Error Details**", expanded=True):
            st.markdown(f"**Symbol:** {symbol}")
            st.markdown(f"**Error Type:** {type(error).__name__}")
            st.markdown(f"**Error Message:** {str(error)}")
        
        st.markdown("### 🛠️ **Possible Solutions:**")
        
        if "KeyError" in str(type(error)):
            st.markdown("""
            **This appears to be a data column issue:**
            1. The technical indicators may not have been calculated properly
            2. Try selecting a different stock symbol
            3. Refresh the page and try again
            4. Check if the stock symbol is valid
            """)
        elif "shape" in str(error).lower():
            st.markdown("""
            **This appears to be a data shape issue:**
            1. The model expects specific data dimensions
            2. Try using more historical data (increase lookback days)
            3. Ensure the stock has enough trading history
            """)
        else:
            st.markdown("""
            **General troubleshooting steps:**
            1. Refresh the page and try again
            2. Try a different stock symbol
            3. Check your internet connection
            4. Wait a moment and retry
            """)
    
    def show_data_debug_info(self, data: pd.DataFrame, symbol: str) -> None:
        """Show debugging information about the data"""
        with st.expander(f"🔍 **Debug Info for {symbol}**"):
            st.markdown("**Data Shape:**")
            st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
            
            st.markdown("**Available Columns:**")
            cols_display = ", ".join(data.columns.tolist())
            st.text(cols_display)
            
            st.markdown("**Data Sample:**")
            st.dataframe(data.tail(5))
            
            st.markdown("**Missing Values:**")
            missing_info = data.isnull().sum()
            missing_cols = missing_info[missing_info > 0]
            if len(missing_cols) > 0:
                st.write(missing_cols)
            else:
                st.write("No missing values found ✅")
    
    def handle_generic_error(self, error) -> None:
        """Handle generic errors with user-friendly messages"""
        self.logger.error(f"Generic error occurred: {str(error)}")
        
        st.error("🚨 **An Error Occurred**")
        
        with st.expander("📋 **Error Details**", expanded=False):
            st.code(str(error))
        
        # Provide troubleshooting suggestions based on error type
        if "timestamp" in str(error).lower() or "datetime" in str(error).lower():
            st.markdown("""
            **This appears to be a date/time related issue:**
            1. The data processing encountered a timestamp problem
            2. This is usually a temporary issue with data formatting
            3. Try refreshing the page and selecting the stock again
            """)
        elif "connection" in str(error).lower() or "timeout" in str(error).lower():
            st.markdown("""
            **This appears to be a connection issue:**
            1. Check your internet connection
            2. The data provider may be temporarily unavailable
            3. Try again in a few minutes
            """)
        elif "import" in str(error).lower() or "module" in str(error).lower():
            st.markdown("""
            **This appears to be a dependency issue:**
            1. Some required packages may not be installed properly
            2. This is usually resolved automatically on refresh
            3. If the problem persists, try refreshing the page
            """)
        else:
            st.markdown("""
            **General troubleshooting steps:**
            1. Refresh the page and try again
            2. Try a different stock symbol
            3. Check your internet connection
            4. Wait a moment and retry
            """)
    
def show_error_recovery_options():
    """Show error recovery options to users"""
    st.markdown("### 🔄 **Recovery Options**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 **Refresh Page**"):
            # Use modern rerun method
            try:
                st.rerun()
            except AttributeError:
                # Fallback for older Streamlit versions
                st.experimental_rerun()
            except Exception as e:
                st.warning(f"Refresh failed: {e}")
    
    with col2:
        if st.button("🏠 **Go to Home**"):
            try:
                # Safe session state clear with initialization check
                if hasattr(st, 'session_state'):
                    try:
                        if hasattr(st.session_state, 'clear'):
                            st.session_state.clear()
                    except Exception:
                        pass  # Continue even if clear fails
                
                # Use modern rerun method
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
            except Exception as e:
                st.warning(f"Navigation failed: {e}")
    
    with col3:
        if st.button("📊 **Try Different Stock**"):
            try:
                # Safe session state access with initialization check
                if hasattr(st, 'session_state'):
                    try:
                        if 'symbol' in st.session_state:
                            del st.session_state['symbol']
                    except Exception:
                        pass  # Continue even if deletion fails
                
                # Use modern rerun method
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
            except Exception as e:
                st.warning(f"Stock change failed: {e}")

# Create global instance for easy access
_error_handler_instance = AppErrorHandler()

# Export functions for backward compatibility
def error_handler(error):
    """Global error handler function"""
    return _error_handler_instance.handle_generic_error(error)

def safe_execute(func, *args, **kwargs):
    """Safely execute a function with error handling"""
    return _error_handler_instance.safe_execute(func, *args, **kwargs)

# Export the class as ErrorHandler for compatibility
ErrorHandler = AppErrorHandler

# Make all exports available
__all__ = ['AppErrorHandler', 'ErrorHandler', 'error_handler', 'safe_execute']
