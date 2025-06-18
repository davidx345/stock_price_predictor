"""
Visualization utilities for stock price prediction analysis.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

class StockVisualizer:
    """Create interactive visualizations for stock analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Color scheme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'candlestick_up': '#00FF00',
            'candlestick_down': '#FF0000',
            'prediction': '#FF6B6B',
            'actual': '#4ECDC4'
        }
    
    def create_candlestick_chart(self, df: pd.DataFrame, title: str = "Stock Price") -> go.Figure:
        """Create an interactive candlestick chart"""
        try:
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                increasing_line_color=self.colors['candlestick_up'],
                decreasing_line_color=self.colors['candlestick_down'],
                name="Price"
            )])
            
            fig.update_layout(
                title=f"{title} - Candlestick Chart",
                yaxis_title="Price ($)",
                xaxis_title="Date",
                template="plotly_white",
                height=500,
                showlegend=True,
                xaxis_rangeslider_visible=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating candlestick chart: {str(e)}")
            raise
    
    def create_price_chart_with_indicators(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create comprehensive price chart with technical indicators"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price & Moving Averages', 'Volume', 'RSI', 'MACD'),
                row_heights=[0.5, 0.2, 0.15, 0.15]
            )
            
            # Price and Moving Averages
            fig.add_trace(
                go.Scatter(x=df.index, y=df['Close'], name='Close', 
                          line=dict(color=self.colors['primary'], width=2)),
                row=1, col=1
            )
            
            if 'SMA_20' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                              line=dict(color=self.colors['secondary'], width=1)),
                    row=1, col=1
                )
            
            if 'SMA_50' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                              line=dict(color=self.colors['success'], width=1)),
                    row=1, col=1
                )
            
            # Bollinger Bands
            if all(col in df.columns for col in ['BB_upper', 'BB_lower']):
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper',
                              line=dict(color='rgba(173,216,230,0.5)', width=1)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower',
                              line=dict(color='rgba(173,216,230,0.5)', width=1),
                              fill='tonexty', fillcolor='rgba(173,216,230,0.2)'),
                    row=1, col=1
                )
            
            # Volume
            colors = ['red' if close < open else 'green' 
                     for close, open in zip(df['Close'], df['Open'])]
            
            fig.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name='Volume',
                      marker_color=colors, opacity=0.7),
                row=2, col=1
            )
            
            # RSI
            if 'RSI' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                              line=dict(color=self.colors['warning'])),
                    row=3, col=1
                )
                
                # RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", 
                             annotation_text="Overbought", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", 
                             annotation_text="Oversold", row=3, col=1)
            
            # MACD
            if all(col in df.columns for col in ['MACD', 'MACD_signal']):
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                              line=dict(color=self.colors['info'])),
                    row=4, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal',
                              line=dict(color=self.colors['danger'])),
                    row=4, col=1
                )
                
                if 'MACD_histogram' in df.columns:
                    colors = ['green' if val >= 0 else 'red' for val in df['MACD_histogram']]
                    fig.add_trace(
                        go.Bar(x=df.index, y=df['MACD_histogram'], name='Histogram',
                              marker_color=colors, opacity=0.7),
                        row=4, col=1
                    )
            
            # Update layout
            fig.update_layout(
                title=f"{symbol} - Technical Analysis",
                height=800,
                template="plotly_white",
                showlegend=True,
                hovermode='x unified'
            )
            
            # Update axes
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
            fig.update_yaxes(title_text="MACD", row=4, col=1)
            fig.update_xaxes(title_text="Date", row=4, col=1)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating technical analysis chart: {str(e)}")
            raise
    
    def create_prediction_chart(self, historical_data: pd.DataFrame, 
                               predictions: Dict, symbol: str) -> go.Figure:
        """Create chart showing historical data and predictions"""
        try:
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=historical_data['Close'],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color=self.colors['actual'], width=2)
                )
            )
              # Predictions
            pred_dates = predictions['dates']
            
            # Ensure pred_dates is a proper datetime series
            if not isinstance(pred_dates, pd.DatetimeIndex):
                try:
                    pred_dates = pd.to_datetime(pred_dates)
                except Exception as e:
                    self.logger.warning(f"Could not convert prediction dates: {e}")
                    # Fallback: generate dates from last historical date
                    last_date = historical_data.index[-1]
                    pred_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=len(predictions['predictions']),
                        freq='D'
                    )
            
            pred_prices = predictions['predictions']
            
            # Connect last historical point to first prediction
            connection_x = [historical_data.index[-1], pred_dates[0]]
            connection_y = [historical_data['Close'].iloc[-1], pred_prices[0]]
            
            fig.add_trace(
                go.Scatter(
                    x=connection_x,
                    y=connection_y,
                    mode='lines',
                    name='Connection',
                    line=dict(color=self.colors['prediction'], width=2, dash='dot'),
                    showlegend=False
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=pred_dates,
                    y=pred_prices,
                    mode='lines+markers',
                    name='Predictions',
                    line=dict(color=self.colors['prediction'], width=3),
                    marker=dict(size=8, symbol='diamond')
                )
            )
            
            # Add confidence intervals if available
            if 'confidence' in predictions and len(predictions['confidence']) > 0:
                confidence = predictions['confidence'][0]  # Assuming uniform confidence
                upper_bound = [p * (1 + 0.1 * (1 - confidence)) for p in pred_prices]
                lower_bound = [p * (1 - 0.1 * (1 - confidence)) for p in pred_prices]
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_dates,
                        y=upper_bound,
                        mode='lines',
                        name='Upper Confidence',
                        line=dict(color='rgba(255,107,107,0.3)', width=1),
                        showlegend=False
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_dates,
                        y=lower_bound,
                        mode='lines',
                        name='Lower Confidence',
                        line=dict(color='rgba(255,107,107,0.3)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(255,107,107,0.2)',
                        showlegend=False
                    )
                )            # Add vertical line to separate historical and predicted data
            try:
                # Use the last date from historical data
                last_date = historical_data.index[-1]
                
                # Convert to a format that Plotly can handle reliably
                if hasattr(last_date, 'isoformat'):
                    last_date_str = last_date.isoformat()
                elif hasattr(last_date, 'strftime'):
                    last_date_str = last_date.strftime('%Y-%m-%d')
                else:
                    last_date_str = str(last_date)
                
                fig.add_vline(
                    x=last_date_str,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Prediction Start"
                )
            except Exception as e:
                self.logger.warning(f"Could not add vertical line: {e}")
                # Continue without the vertical line
            
            fig.update_layout(
                title=f"{symbol} - Price Prediction",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template="plotly_white",
                height=500,
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating prediction chart: {str(e)}")
            raise
    
    def create_performance_metrics_chart(self, metrics: Dict) -> go.Figure:
        """Create visualization of model performance metrics"""
        try:
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            # Create gauge charts for key metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=metric_names,
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]]
            )
            
            # Define ranges and colors for different metrics
            metric_configs = {
                'mae': {'range': [0, 10], 'color': 'red'},
                'rmse': {'range': [0, 15], 'color': 'orange'},
                'directional_accuracy': {'range': [0, 1], 'color': 'green'},
                'mse': {'range': [0, 100], 'color': 'blue'}
            }
            
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            
            for i, (metric, value) in enumerate(metrics.items()):
                if i >= 4:  # Only show first 4 metrics
                    break
                    
                config = metric_configs.get(metric, {'range': [0, 100], 'color': 'gray'})
                
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=value,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': metric.upper()},
                        gauge={
                            'axis': {'range': config['range']},
                            'bar': {'color': config['color']},
                            'steps': [
                                {'range': [config['range'][0], config['range'][1] * 0.5], 'color': "lightgray"},
                                {'range': [config['range'][1] * 0.5, config['range'][1] * 0.8], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': config['range'][1] * 0.9
                            }
                        }
                    ),
                    row=positions[i][0], col=positions[i][1]
                )
            
            fig.update_layout(
                title="Model Performance Metrics",
                height=600,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating performance metrics chart: {str(e)}")
            raise
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap of features"""
        try:
            # Calculate correlation matrix
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdYlBu',
                zmid=0,
                text=np.around(corr_matrix.values, decimals=2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Feature Correlation Matrix",
                template="plotly_white",
                height=600,
                width=800
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {str(e)}")
            raise
    
    def create_training_history_chart(self, history: Dict) -> go.Figure:
        """Create interactive training history visualization"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Loss', 'Mean Absolute Error', 'Mean Squared Error', 'Learning Rate'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            epochs = list(range(1, len(history['loss']) + 1))
            
            # Loss
            fig.add_trace(
                go.Scatter(x=epochs, y=history['loss'], name='Training Loss',
                          line=dict(color=self.colors['primary'])),
                row=1, col=1
            )
            if 'val_loss' in history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['val_loss'], name='Validation Loss',
                              line=dict(color=self.colors['secondary'])),
                    row=1, col=1
                )
            
            # MAE
            fig.add_trace(
                go.Scatter(x=epochs, y=history['mae'], name='Training MAE',
                          line=dict(color=self.colors['success'])),
                row=1, col=2
            )
            if 'val_mae' in history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['val_mae'], name='Validation MAE',
                              line=dict(color=self.colors['warning'])),
                    row=1, col=2
                )
            
            # MSE
            fig.add_trace(
                go.Scatter(x=epochs, y=history['mse'], name='Training MSE',
                          line=dict(color=self.colors['info'])),
                row=2, col=1
            )
            if 'val_mse' in history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['val_mse'], name='Validation MSE',
                              line=dict(color=self.colors['danger'])),
                    row=2, col=1
                )
            
            # Learning Rate
            if 'lr' in history:
                fig.add_trace(
                    go.Scatter(x=epochs, y=history['lr'], name='Learning Rate',
                              line=dict(color='purple')),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Training History",
                height=600,
                template="plotly_white",
                showlegend=True
            )
            
            # Update axes
            fig.update_xaxes(title_text="Epoch")
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="MAE", row=1, col=2)
            fig.update_yaxes(title_text="MSE", row=2, col=1)
            fig.update_yaxes(title_text="Learning Rate", row=2, col=2)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating training history chart: {str(e)}")
            raise
