"""
Modern Streamlit web application for stock price prediction.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
import logging
import traceback

# Configure page
st.set_page_config(
    page_title="AI Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

try:
    from data_collector import StockDataCollector, FeatureEngineer
    from lstm_model import LSTMStockPredictor
    from visualizations import StockVisualizer
    from utils import ModelManager, DataValidator
    from config import POPULAR_STOCKS, APP_CONFIG, MODEL_CONFIG, DATA_CONFIG
    
    # Try to setup additional logging if available
    try:
        from utils import setup_logging
        setup_logging()
    except (ImportError, AttributeError):
        # If setup_logging is not available, basic logging is already configured above
        logging.getLogger().info("Using basic logging configuration")
        
except ImportError as e:
    st.error(f"Import error: {e}")
    logging.error(f"Import error: {e}")
    st.stop()

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all components with caching"""
    try:
        components = {
            'data_collector': StockDataCollector(),
            'feature_engineer': FeatureEngineer(),
            'visualizer': StockVisualizer(),
            'model_manager': ModelManager("models"),
            'data_validator': DataValidator()
        }
        return components
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
        return None

# Custom CSS for modern styling
def load_custom_css():
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        border-radius: 10px;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stSelectbox > div > div > select {
        border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }    .info-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .info-card h4 {
        color: #333333;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .info-card p {
        color: #1a1a1a;
        font-weight: 600;
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display the main header with animation"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        ">🚀 AI Stock Price Predictor</h1>
        <p style="
            font-size: 1.2rem;
            color: #666;
            margin: 0;
        ">Advanced LSTM-based stock price forecasting with technical analysis</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create interactive sidebar"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
            <h2 style="color: white; margin: 0;">📊 Control Panel</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Stock selection
        st.markdown("### 🎯 Stock Selection")
        
        # Quick select from popular stocks
        category = st.selectbox(
            "Quick Select Category",
            options=["Custom"] + list(POPULAR_STOCKS.keys()),
            help="Choose a category for quick stock selection"
        )
        
        if category != "Custom":
            symbol = st.selectbox(
                "Select Stock",
                options=POPULAR_STOCKS[category],
                help="Choose from popular stocks in this category"
            )
        else:
            symbol = st.text_input(
                "Enter Stock Symbol",
                value="AAPL",
                help="Enter any valid stock symbol (e.g., AAPL, MSFT, GOOGL)"
            ).upper()
        
        # Data period
        period = st.selectbox(
            "📅 Data Period",
            options=["1y", "2y", "5y", "10y", "max"],
            index=2,
            help="Select the historical data period for analysis"
        )
        
        # Prediction settings
        st.markdown("### 🔮 Prediction Settings")
        
        forecast_days = st.slider(
            "Forecast Days",
            min_value=1,
            max_value=30,
            value=5,
            help="Number of days to predict into the future"
        )
        
        use_attention = st.checkbox(
            "Use Attention Mechanism",
            value=False,
            help="Enable attention mechanism for improved predictions"
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            lookback_days = st.slider(
                "Lookback Days",
                min_value=30,
                max_value=120,
                value=60,
                help="Number of historical days to use for prediction"
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.5,
                max_value=0.95,
                value=0.7,
                step=0.05,
                help="Minimum confidence level for predictions"
            )
        
        return {
            'symbol': symbol,
            'period': period,
            'forecast_days': forecast_days,
            'use_attention': use_attention,
            'lookback_days': lookback_days,
            'confidence_threshold': confidence_threshold
        }

def display_stock_info(components, symbol):
    """Display stock information card"""
    try:
        stock_info = components['data_collector'].get_stock_info(symbol)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="info-card">
                <h4>🏢 Company</h4>
                <p><strong>{stock_info.get('name', symbol)}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-card">
                <h4>🏭 Sector</h4>
                <p>{stock_info.get('sector', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            market_cap = stock_info.get('market_cap', 0)
            if market_cap > 1e9:
                market_cap_str = f"${market_cap/1e9:.1f}B"
            elif market_cap > 1e6:
                market_cap_str = f"${market_cap/1e6:.1f}M"
            else:
                market_cap_str = "N/A"
            
            st.markdown(f"""
            <div class="info-card">
                <h4>💰 Market Cap</h4>
                <p>{market_cap_str}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            pe_ratio = stock_info.get('pe_ratio', 0)
            pe_str = f"{pe_ratio:.2f}" if pe_ratio and pe_ratio > 0 else "N/A"
            
            st.markdown(f"""
            <div class="info-card">
                <h4>📊 P/E Ratio</h4>
                <p>{pe_str}</p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.warning(f"Could not fetch stock info: {e}")

def load_and_display_data(components, symbol, period):
    """Load and display stock data with validation"""
    try:
        with st.spinner(f"Loading data for {symbol}..."):            # Fetch data
            data = components['data_collector'].fetch_stock_data(symbol, period)
            
            if data is None:
                st.error(f"❌ Could not fetch data for {symbol}. Please check the symbol and try again.")
                st.info("💡 **Troubleshooting Tips:**")
                st.info("• Try a different stock symbol (e.g., MSFT, GOOGL)")
                st.info("• Check if the symbol is valid on major exchanges (NYSE, NASDAQ)")
                st.info("• Try reducing the data period (e.g., 1y instead of max)")
                logging.error(f"Failed to fetch data for symbol: {symbol}")
                return None
            
            # Check if sample data is being used
            if hasattr(data, 'attrs') and data.attrs.get('sample_data', False):
                st.warning("⚠️ **Demo Mode**: Using sample data because external API data sources are currently unavailable. This is for demonstration purposes only.")
            
            # Validate data
            validation = components['data_validator'].validate_stock_data(data, symbol)
            
            # Display validation results
            if validation['valid']:
                st.success(f"✅ Data loaded successfully: {len(data)} records")
            else:
                st.warning(f"⚠️ Data quality issues detected: {', '.join(validation['issues'])}")
            
            # Display data statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "📊 Records",
                    f"{len(data):,}",
                    help="Total number of data points"
                )
            
            with col2:
                price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                st.metric(
                    "💲 Latest Price",
                    f"${data['Close'].iloc[-1]:.2f}",
                    delta=f"{price_change:.2f}",
                    help="Most recent closing price"
                )
            
            with col3:
                volatility = validation['stats'].get('volatility', 0)
                st.metric(
                    "📈 Volatility",
                    f"{volatility:.1%}",
                    help="Annualized volatility"
                )
            
            with col4:
                date_range = validation['stats']['date_range']
                st.metric(
                    "📅 Date Range",
                    f"{len(data)} days",
                    help=f"From {date_range['start']} to {date_range['end']}"
                )
            
            return data
            
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.error(traceback.format_exc())
        return None

def create_technical_analysis(components, data, symbol):
    """Create and display technical analysis"""
    try:
        with st.spinner("Generating technical analysis..."):
            # Add technical indicators
            enhanced_data = components['feature_engineer'].add_technical_indicators(data)
            
            # Create comprehensive chart
            fig = components['visualizer'].create_price_chart_with_indicators(enhanced_data, symbol)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
            
            return enhanced_data
            
    except Exception as e:
        st.error(f"Error creating technical analysis: {e}")
        return data

def train_or_load_model(components, symbol, settings):
    """Train a new model or load existing one"""
    try:        # Check for existing models
        available_models = components['model_manager'].list_models(symbol)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if available_models:
                st.info(f"Found {len(available_models)} existing model(s) for {symbol}")
                
                # Display model options - only existing models
                model_options = [
                    f"{model['name']} (Created: {model['created']})"
                    for model in available_models[:5]  # Show top 5
                ]
                
                selected_option = st.selectbox(
                    "Choose Existing Model",
                    options=model_options,
                    help="Select an existing trained model"
                )
                
                # Load selected model
                model_index = model_options.index(selected_option)
                selected_model = available_models[model_index]
                return load_existing_model(components, selected_model)
            else:
                st.warning(f"⚠️ No existing models found for {symbol}")
                st.info("📝 **Available pre-trained models:**")
                st.info("• AAPL, AMZN, GOOGL, MSFT, TSLA")
                st.info("Please select one of the available symbols above.")
                return None
        
        with col2:
            if st.button("🔄 Refresh Models", help="Refresh the list of available models"):
                st.experimental_rerun()
                
    except Exception as e:
        st.error(f"Error managing models: {e}")
        return None
def load_existing_model(components, model_info):
    """Load an existing model"""   
    try:
        st.info(f"Loading model: {model_info['name']}")
        
        # Handle both directory-based models and individual .keras files
        if model_info.get('type') == 'file':
            # Legacy .keras file
            model_path = model_info['path']
        else:
            # Directory-based model
            model_path = Path(model_info['path']) / "model.h5"
        
        model = LSTMStockPredictor.load_model(str(model_path))
        
        # Display model info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            accuracy = model_info['metrics'].get('directional_accuracy', 0)
            st.metric("🎯 Accuracy", f"{accuracy:.2%}")
        
        with col2:
            mae = model_info['metrics'].get('mae', 0)
            st.metric("📊 MAE", f"{mae:.4f}")
        
        with col3:
            st.metric("📅 Created", model_info.get('created', 'Unknown')[:10])
        
        return {'model': model, 'model_info': model_info}
        
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def make_predictions(model_data, enhanced_data, symbol, settings, components):
    """Make and display predictions"""
    try:
        st.markdown("### 🔮 Price Predictions")
        
        model = model_data['model']
        
        # Prepare recent data for prediction
        recent_data = enhanced_data[DATA_CONFIG.feature_columns].dropna().tail(settings['lookback_days'])
        
        if len(recent_data) < settings['lookback_days']:
            st.error(f"Insufficient data for prediction. Need at least {settings['lookback_days']} days.")
            return
        
        # Normalize data
        from sklearn.preprocessing import MinMaxScaler
        scalers = {}
        normalized_data = recent_data.copy()
        
        for column in recent_data.columns:
            scaler = MinMaxScaler()
            normalized_data[column] = scaler.fit_transform(recent_data[[column]])
            scalers[column] = scaler
        
        # Make predictions
        predictions = model.predict_next_days(
            normalized_data.values, 
            scalers['Close']
        )
        
        # Display predictions
        display_predictions(predictions, enhanced_data, symbol, components)
        
        # Create prediction chart
        fig = components['visualizer'].create_prediction_chart(
            enhanced_data.tail(60), predictions, symbol
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.error(traceback.format_exc())

def display_predictions(predictions, historical_data, symbol, components):
    """Display prediction results in an attractive format"""
    current_price = historical_data['Close'].iloc[-1]
    pred_prices = predictions['predictions']
    pred_dates = predictions['dates']
    
    # Calculate metrics
    avg_prediction = np.mean(pred_prices)
    price_change = avg_prediction - current_price
    price_change_pct = (price_change / current_price) * 100
    
    # Main prediction card
    st.markdown(f"""
    <div class="prediction-card">
        <h2>🎯 {symbol} Prediction Summary</h2>
        <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
            <div>
                <h3>Current Price</h3>
                <h2>${current_price:.2f}</h2>
            </div>
            <div>
                <h3>Average Predicted</h3>
                <h2>${avg_prediction:.2f}</h2>
            </div>
            <div>
                <h3>Expected Change</h3>
                <h2 style="color: {'#4CAF50' if price_change > 0 else '#f44336'}">
                    {'+' if price_change > 0 else ''}{price_change_pct:.1f}%
                </h2>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed predictions table
    st.markdown("### 📅 Daily Predictions")
    
    # Create DataFrame for predictions
    pred_df = pd.DataFrame({
        'Date': pred_dates,
        'Predicted Price': [f"${price:.2f}" for price in pred_prices],
        'Change from Current': [f"{'+' if (price - current_price) > 0 else ''}{((price - current_price)/current_price)*100:.1f}%" 
                               for price in pred_prices],
        'Confidence': [f"{conf:.1%}" for conf in predictions.get('confidence', [0.7] * len(pred_prices))]
    })
    
    st.dataframe(pred_df, use_container_width=True)
    
    # Trend indicator
    if len(pred_prices) > 1:
        trend = "📈 Upward" if pred_prices[-1] > pred_prices[0] else "📉 Downward"
        st.markdown(f"**Trend Direction:** {trend}")

def main():
    """Main application"""
    # Load CSS and display header
    load_custom_css()
    display_header()
    
    # Initialize components
    components = initialize_components()
    if components is None:
        st.stop()
    
    # Create sidebar
    settings = create_sidebar()
    
    # Main content
    try:
        # Display stock information
        st.markdown("---")
        display_stock_info(components, settings['symbol'])
        
        # Load and display data
        st.markdown("---")
        st.markdown("## 📊 Stock Data Analysis")
        
        data = load_and_display_data(components, settings['symbol'], settings['period'])
        if data is None:
            st.stop()
        
        # Technical analysis
        enhanced_data = create_technical_analysis(components, data, settings['symbol'])
        
        # Model training/loading section
        st.markdown("---")
        st.markdown("## 🤖 AI Model")
        
        model_data = train_or_load_model(components, settings['symbol'], settings)
        if model_data is None:
            st.stop()
        
        # Make predictions
        st.markdown("---")
        make_predictions(model_data, enhanced_data, settings['symbol'], settings, components)
        
        # Additional features
        st.markdown("---") 
        st.markdown("## 📋 Additional Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "🔍 Feature Analysis", "📈 Market Insights"])
        
        with tab1:
            if 'results' in model_data:
                # Training history
                history = model_data['results'].get('history', {})
                if history:
                    fig = components['visualizer'].create_training_history_chart(history)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Feature correlation heatmap
            if enhanced_data is not None:
                numeric_cols = enhanced_data.select_dtypes(include=[np.number]).columns[:10]  # Limit for performance
                if len(numeric_cols) > 1:
                    fig = components['visualizer'].create_correlation_heatmap(enhanced_data[numeric_cols])
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("""
            ### 🎯 Key Insights
            - **Model Type**: Advanced LSTM with technical indicators
            - **Features Used**: Price data, volume, RSI, MACD, Bollinger Bands, and more
            - **Prediction Horizon**: Configurable (1-30 days)
            - **Update Frequency**: Real-time data from multiple financial APIs
            
            ### ⚠️ Disclaimer
            This is a demonstration application for educational purposes. 
            **Do not use these predictions for actual trading decisions.**
            Always consult with financial advisors and do your own research.
            """)
    
    except Exception as e:
        st.error(f"Application error: {e}")
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
