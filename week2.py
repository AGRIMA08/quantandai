import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
# Machine Learning Libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
# Time Series Libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import glob
import os
def load_and_preprocess_data():
    txt_files = glob.glob("*.txt")
    if not txt_files:
        print("No .txt files found. Please ensure your stock data files are in the current directory.")
        return None
    dfs = []
    for file in txt_files:
        ticker = os.path.splitext(os.path.basename(file))[0]
        try:
            df = pd.read_csv(file, sep=',', parse_dates=['Date'])
            df['Ticker'] = ticker
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    if not dfs:
        print("No valid data files found.")
        return None
    # Concatenate all dataframes
    arr = pd.concat(dfs)
    arr['Date'] = pd.to_datetime(arr['Date'])
    arr.set_index(['Ticker', 'Date'], inplace=True)
    # Handle missing values
    print("Handling missing values...")
    missing_before = arr.isnull().groupby('Ticker').sum()
    arr = arr.groupby(level=0).apply(lambda g: g.interpolate().ffill().bfill())
    arr.index = arr.index.droplevel(0)
    arr.reset_index(inplace=True)
    arr.set_index(['Ticker', 'Date'], inplace=True)
    arr.sort_index(inplace=True)
    # Filter to last 10 years
    ten_years_ago = pd.Timestamp.today() - pd.DateOffset(years=10)
    arr = arr[arr.index.get_level_values('Date') >= ten_years_ago]
    # Calculate technical indicators
    print("Calculating technical indicators...")
    arr['Daily_Return'] = arr.groupby(level=0)['Close'].pct_change()
    arr['MA_7'] = arr.groupby(level=0)['Close'].transform(lambda x: x.rolling(window=7).mean())
    arr['MA_30'] = arr.groupby(level=0)['Close'].transform(lambda x: x.rolling(window=30).mean())
    arr['Volatility_30'] = arr.groupby(level=0)['Daily_Return'].transform(lambda x: x.rolling(window=30).std()) 
    # Additional technical indicators for Random Forest
    arr['RSI'] = arr.groupby(level=0).apply(lambda x: calculate_rsi(x['Close'])).reset_index(level=0, drop=True)
    arr['BB_Width'] = arr.groupby(level=0).apply(lambda x: calculate_bollinger_width(x['Close'])).reset_index(level=0, drop=True)
    arr['Volume_MA'] = arr.groupby(level=0)['Volume'].transform(lambda x: x.rolling(window=10).mean())
    arr['Volume_Ratio'] = arr['Volume'] / arr['Volume_MA']
    return arr
def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
def calculate_bollinger_width(prices, window=20):
    """Calculate Bollinger Band width"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    bb_upper = rolling_mean + (rolling_std * 2)
    bb_lower = rolling_mean - (rolling_std * 2)
    return bb_upper - bb_lower
# Load data
full_data = load_and_preprocess_data()
if full_data is None:
    print("Failed to load data. Exiting...")
    exit()
# Extract amzn.us data
if 'amzn.us' not in full_data.index.get_level_values('Ticker'):
    print("amzn.us data not found in the dataset. Available tickers:")
    print(full_data.index.get_level_values('Ticker').unique())
    exit()
amzn_data = full_data.loc['amzn.us'].copy()
amzn_data = amzn_data.dropna()
print(f"\namzn.us Data Summary:")
print(f"Date range: {amzn_data.index[0].date()} to {amzn_data.index[-1].date()}")
print(f"Total trading days: {len(amzn_data)}")
print(f"Columns: {list(amzn_data.columns)}")
# 2. DATA SPLITTING (Maintaining time order)
def split_data(data, train_ratio=0.8):
    """Split data maintaining chronological order"""
    split_index = int(len(data) * train_ratio)
    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()
    return train_data, test_data
train_data, test_data = split_data(amzn_data)
print(f"\nData Split:")
print(f"Training set: {len(train_data)} days ({train_data.index[0].date()} to {train_data.index[-1].date()})")
print(f"Testing set: {len(test_data)} days ({test_data.index[0].date()} to {test_data.index[-1].date()})")
# 3. MODEL IMPLEMENTATIONS
class StockPredictionModels:
    def __init__(self):
        self.linear_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.arima_model = None
        self.arima_fitted = None
    def prepare_linear_features(self, data, lookback=5):
        """Prepare features for Linear Regression (past 5 days closing prices)"""
        X, y = [], []
        close_prices = data['Close'].values
        for i in range(lookback, len(close_prices)):
            X.append(close_prices[i-lookback:i])
            y.append(close_prices[i])
        return np.array(X), np.array(y)
    def prepare_rf_features(self, data):
        """Prepare features for Random Forest using technical indicators"""
        feature_cols = ['MA_7', 'MA_30', 'RSI', 'BB_Width', 'Daily_Return', 'Volume_Ratio', 'Volatility_30']
        # Ensure all feature columns exist
        available_features = [col for col in feature_cols if col in data.columns]
        print(f"Using features for Random Forest: {available_features}")
        X = data[available_features].values
        y = data['Close'].shift(-1).values[:-1]  # Next day's closing price
        X = X[:-1]  # Remove last row to match y length
        return X, y, available_features
    def train_linear_regression(self, train_data):
        """Train Linear Regression model"""
        print("\n" + "="*40)
        print("TRAINING LINEAR REGRESSION")
        print("="*40)
        X_train, y_train = self.prepare_linear_features(train_data)
        print(f"Training samples: {len(X_train)}")
        print(f"Feature shape: {X_train.shape}")
        self.linear_model.fit(X_train, y_train)
        # Training performance
        train_pred = self.linear_model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        print(f"Training MAE: ${train_mae:.2f}")
        return X_train, y_train
    def train_random_forest(self, train_data):
        """Train Random Forest model"""
        print("\n" + "="*40)
        print("TRAINING RANDOM FOREST")
        print("="*40)
        X_train, y_train, feature_names = self.prepare_rf_features(train_data)
        # Remove rows with NaN values
        mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        X_train_clean = X_train[mask]
        y_train_clean = y_train[mask]
        print(f"Training samples: {len(X_train_clean)} (after removing NaN)")
        print(f"Feature shape: {X_train_clean.shape}")
        self.rf_model.fit(X_train_clean, y_train_clean)
        # Feature importance
        importance = self.rf_model.feature_importances_
        print(f"\nFeature Importance:")
        for feat, imp in zip(feature_names, importance):
            print(f"  {feat}: {imp:.4f}")
        # Training performance
        train_pred = self.rf_model.predict(X_train_clean)
        train_mae = mean_absolute_error(y_train_clean, train_pred)
        print(f"Training MAE: ${train_mae:.2f}")
        return X_train_clean, y_train_clean, feature_names
    def train_arima(self, train_data, order=(5,1,0)):
        """Train ARIMA model"""
        print("\n" + "="*40)
        print("TRAINING ARIMA")
        print("="*40)
        # Check for stationarity
        close_prices = train_data['Close'].dropna()
        adf_result = adfuller(close_prices)
        print(f"ADF Statistic: {adf_result[0]:.4f}")
        print(f"p-value: {adf_result[1]:.4f}")
        if adf_result[1] > 0.05:
            print("Series is not stationary, using differencing...")
        try:
            print(f"Fitting ARIMA{order}...")
            self.arima_model = ARIMA(close_prices, order=order)
            self.arima_fitted = self.arima_model.fit()
            print(f"ARIMA trained successfully")
            print(f"AIC: {self.arima_fitted.aic:.2f}")
            print(f"Training samples: {len(close_prices)}")
        except Exception as e:
            print(f"ARIMA training failed: {e}")
            # Try simpler model
            try:
                print("Trying simpler ARIMA(1,1,1)...")
                self.arima_model = ARIMA(close_prices, order=(1,1,1))
                self.arima_fitted = self.arima_model.fit()
                print(f"Simpler ARIMA trained successfully")
                print(f"AIC: {self.arima_fitted.aic:.2f}")
            except Exception as e2:
                print(f"Simpler ARIMA also failed: {e2}")
    def predict_linear(self, test_data):
        """Make predictions using Linear Regression"""
        X_test, y_test = self.prepare_linear_features(test_data)
        predictions = self.linear_model.predict(X_test)
        return predictions, y_test
    def predict_random_forest(self, test_data):
        """Make predictions using Random Forest"""
        X_test, y_test, _ = self.prepare_rf_features(test_data)
        # Handle NaN values
        mask = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y_test)
        X_test_clean = X_test[mask]
        y_test_clean = y_test[mask]
        predictions = self.rf_model.predict(X_test_clean)
        return predictions, y_test_clean
    
    def predict_arima(self, n_periods):
        """Make predictions using ARIMA"""
        if self.arima_fitted is None:
            return None, None
        
        try:
            forecast = self.arima_fitted.forecast(steps=n_periods)
            return forecast, None
        except Exception as e:
            print(f"ARIMA prediction failed: {e}")
            return None, None

# Initialize and train models
models = StockPredictionModels()

# Train all models
lr_train_data = models.train_linear_regression(train_data)
rf_train_data = models.train_random_forest(train_data)
models.train_arima(train_data)

# 4. MODEL EVALUATION

def evaluate_model(y_true, y_pred, model_name):
    """Comprehensive model evaluation"""
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Direction accuracy (key for trading)
    if len(y_true) > 1:
        actual_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(y_pred[1:] - y_true[:-1])
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        direction_accuracy = 0
    
    print(f"\n{model_name} Evaluation:")
    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Direction Accuracy: {direction_accuracy:.2f}%")
    
    return {
        'MAE': mae, 
        'RMSE': rmse, 
        'MAPE': mape,
        'Direction_Accuracy': direction_accuracy
    }

# Make predictions and evaluate
print("\n" + "="*60)
print("MODEL PREDICTIONS AND EVALUATION")
print("="*60)

# Linear Regression
print("Evaluating Linear Regression...")
lr_pred, lr_actual = models.predict_linear(test_data)
lr_metrics = evaluate_model(lr_actual, lr_pred, "Linear Regression")

# Random Forest
print("Evaluating Random Forest...")
rf_pred, rf_actual = models.predict_random_forest(test_data)
rf_metrics = evaluate_model(rf_actual, rf_pred, "Random Forest")

# ARIMA
print("Evaluating ARIMA...")
arima_pred, _ = models.predict_arima(len(test_data))
if arima_pred is not None:
    test_close = test_data['Close'].values
    # Align lengths
    min_len = min(len(test_close), len(arima_pred))
    arima_metrics = evaluate_model(test_close[:min_len], arima_pred[:min_len], "ARIMA")
else:
    arima_metrics = {'MAE': np.inf, 'RMSE': np.inf, 'MAPE': np.inf, 'Direction_Accuracy': 0}
    print("\nARIMA: Model failed to generate predictions")

# 5. MODEL COMPARISON
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

comparison_df = pd.DataFrame({
    'Linear Regression': lr_metrics,
    'Random Forest': rf_metrics,
    'ARIMA': arima_metrics
}).T

print(comparison_df.round(2))

# Find best model based on MAE
best_model_mae = comparison_df['MAE'].idxmin()
best_model_direction = comparison_df['Direction_Accuracy'].idxmax()

print(f"\nBest Model by MAE: {best_model_mae}")
print(f"Best Model by Direction Accuracy: {best_model_direction}")

# 6. BACKTESTING THE BEST MODEL

def backtest_strategy(predictions, actual_prices, dates, initial_capital=10000):
    """Backtest simple trading strategy"""
    capital = initial_capital
    position = 0  # 0: cash, 1: long position
    shares = 0
    trades = []
    portfolio_values = []
    
    for i in range(len(predictions)):
        current_price = actual_prices[i]
        predicted_price = predictions[i]
        
        # Simple strategy: Buy if predicted price > current price * 1.01 (1% threshold)
        # Sell if predicted price < current price * 0.99
        
        if predicted_price > current_price * 1.01 and position == 0:
            # Buy signal
            shares = capital / current_price
            position = 1
            capital = 0
            trades.append({
                'date': dates[i] if hasattr(dates[i], 'date') else dates[i],
                'action': 'BUY',
                'price': current_price,
                'shares': shares
            })
        elif predicted_price < current_price * 0.99 and position == 1:
            # Sell signal
            capital = shares * current_price
            position = 0
            shares = 0
            trades.append({
                'date': dates[i] if hasattr(dates[i], 'date') else dates[i],
                'action': 'SELL',
                'price': current_price,
                'capital': capital
            })
        
        # Calculate current portfolio value
        if position == 1:
            portfolio_values.append(shares * current_price)
        else:
            portfolio_values.append(capital)
    
    # Final portfolio value
    final_value = portfolio_values[-1] if portfolio_values else capital
    if position == 1:  # Still holding shares
        final_value = shares * actual_prices[-1]
    
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    return {
        'trades': trades,
        'portfolio_values': portfolio_values,
        'final_value': final_value,
        'total_return': total_return,
        'num_trades': len(trades)
    }

print("\n" + "="*60)
print("BACKTESTING THE BEST MODEL")
print("="*60)

# Use the best model (by MAE) for backtesting
if best_model_mae == 'Linear Regression':
    best_predictions = lr_pred
    best_actual = lr_actual
    test_dates = test_data.index[5:]  # Account for lookback period
elif best_model_mae == 'Random Forest':
    best_predictions = rf_pred
    best_actual = rf_actual
    # Get valid dates (removing NaN rows)
    X_test, y_test, _ = models.prepare_rf_features(test_data)
    mask = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y_test)
    test_dates = test_data.index[:-1][mask]  # Remove last date and apply mask
else:  # ARIMA
    best_predictions = arima_pred
    best_actual = test_data['Close'].values
    test_dates = test_data.index

# Ensure arrays have same length
min_len = min(len(best_predictions), len(best_actual), len(test_dates))
best_predictions = best_predictions[:min_len]
best_actual = best_actual[:min_len]
test_dates = test_dates[:min_len]

# Run backtesting
backtest_results = backtest_strategy(best_predictions, best_actual, test_dates)

print(f"Backtesting Results for {best_model_mae}:")
print(f"Initial Capital: $10,000.00")
print(f"Final Portfolio Value: ${backtest_results['final_value']:.2f}")
print(f"Total Return: {backtest_results['total_return']:.2f}%")
print(f"Number of Trades: {backtest_results['num_trades']}")

# Show some trades
if backtest_results['trades']:
    print(f"\nFirst 5 Trades:")
    for i, trade in enumerate(backtest_results['trades'][:5]):
        date_str = trade['date'].strftime('%Y-%m-%d') if hasattr(trade['date'], 'strftime') else str(trade['date'])
        print(f"  {i+1}. {date_str}: {trade['action']} at ${trade['price']:.2f}")
# Buy and hold comparison
buy_hold_return = ((best_actual[-1] - best_actual[0]) / best_actual[0]) * 100
print(f"\nBuy & Hold Return: {buy_hold_return:.2f}%")
print(f"Strategy vs Buy & Hold: {backtest_results['total_return'] - buy_hold_return:.2f}% {'outperformance' if backtest_results['total_return'] > buy_hold_return else 'underperformance'}")