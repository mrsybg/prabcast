# models.py
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

class SeasonalNaiveModel:
    def __init__(self, season_length=12):
        self.season_length = season_length

    def fit(self, data):
        self.data = data

    def predict(self, steps):
        last_season = self.data.iloc[-self.season_length:].values
        repeats = steps // self.season_length + 1
        forecast = np.tile(last_season, repeats)[:steps]
        return pd.Series(forecast.flatten(), index=pd.date_range(self.data.index[-1] + pd.DateOffset(months=1), periods=steps, freq='M'))

class MovingAverageModel:
    def __init__(self, window=12):
        self.window = window

    def fit(self, data):
        self.data = data

    def predict(self, steps):
        mean_value = self.data.iloc[-self.window:].mean()
        forecast = pd.Series([mean_value] * steps,
                             index=pd.date_range(self.data.index[-1] + pd.DateOffset(months=1),
                                                 periods=steps, freq='M'))
        return forecast

class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None

    def fit(self, data):
        self.model = ARIMA(data, order=self.order).fit()

    def predict(self, steps):
        forecast = self.model.forecast(steps=steps)
        return forecast

def standardize_data(data):
    """
    Standardizes input data for all models.
    Returns both original format and Prophet format (ds, y).
    """
    # Original format - datetime index and values
    std_data = data.copy()
    if not isinstance(std_data.index, pd.DatetimeIndex):
        std_data.index = pd.to_datetime(std_data.index)
    
    # Prophet format - ds and y columns
    prophet_data = pd.DataFrame({
        'ds': std_data.index,
        'y': std_data.values
    })
    
    return std_data, prophet_data

# Update ProphetModel class
class ProphetModel:
    def __init__(self):
        self.model = None
        self.data = None
        self.params = {}

    def fit(self, data, yearly_seasonality='auto', weekly_seasonality='auto', **kwargs):
        _, prophet_data = standardize_data(data)
        self.data = data
        self.params = {
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            **kwargs
        }
        self.model = Prophet(**self.params)
        self.model.fit(prophet_data)

    def predict(self, steps):
        future_dates = pd.date_range(
            start=self.data.index[-1] + pd.DateOffset(months=1),
            periods=steps,
            freq='M'
        )
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = self.model.predict(future_df)
        return pd.Series(forecast['yhat'].values, index=future_dates)

class LSTMModel:
    def __init__(self, epochs=50, batch_size=1):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
        self.last_sequence = None
        self.data = None

    def fit(self, data):
        self.data = data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        self.scaler = scaler

        X, y = [], []
        window_size = 12
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i-window_size:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        self.model.add(LSTM(50))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, steps):
        last_sequence = self.scaler.transform(self.data.values[-12:].reshape(-1, 1)).flatten().tolist()
        forecast = []
        for _ in range(steps):
            input_seq = np.array(last_sequence[-12:]).reshape((1, 12, 1))
            pred = self.model.predict(input_seq, verbose=0)
            forecast.append(pred[0,0])
            last_sequence.append(pred[0,0])
        forecast = self.scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        return pd.Series(forecast, index=pd.date_range(self.data.index[-1] + pd.DateOffset(months=1), 
                                                     periods=steps, freq='M'))

class GRUModel:
    def __init__(self, epochs=50, batch_size=1):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
        self.last_sequence = None
        self.data = None

    def fit(self, data):
        # Same implementation as before, but using self.epochs and self.batch_size
        self.data = data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        self.scaler = scaler

        X, y = [], []
        window_size = 12
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i-window_size:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        self.model = Sequential()
        self.model.add(GRU(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        self.model.add(GRU(50))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, steps):
        last_sequence = self.scaler.transform(self.data.values[-12:].reshape(-1, 1)).flatten().tolist()
        forecast = []
        for _ in range(steps):
            input_seq = np.array(last_sequence[-12:]).reshape((1, 12, 1))
            pred = self.model.predict(input_seq, verbose=0)
            forecast.append(pred[0,0])
            last_sequence.append(pred[0,0])
        forecast = self.scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        return pd.Series(forecast, index=pd.date_range(self.data.index[-1] + pd.DateOffset(months=1), 
                                                     periods=steps, freq='M'))

class SESModel:
    def __init__(self):
        self.model = None
        self.data = None

    def fit(self, data):
        self.model = SimpleExpSmoothing(data).fit()
        self.data = data

    def predict(self, steps):
        forecast = self.model.forecast(steps)
        return pd.Series(forecast, index=pd.date_range(self.data.index[-1] + pd.DateOffset(months=1), 
                                                     periods=steps, freq='M'))

class SARIMAModel:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None

    def fit(self, data):
        self.model = SARIMAX(data, order=self.order, seasonal_order=self.seasonal_order).fit()

    def predict(self, steps):
        forecast = self.model.forecast(steps=steps)
        return forecast

class HoltWintersModel:
    def __init__(self, seasonal_periods=12):
        self.seasonal_periods = seasonal_periods
        self.model = None

    def fit(self, data):
        self.model = ExponentialSmoothing(data, seasonal_periods=self.seasonal_periods, seasonal='add').fit()
        self.data = data

    def predict(self, steps):
        forecast = self.model.forecast(steps)
        return forecast

class XGBoostModel:
    def __init__(self, n_estimators=200, window_size=12):
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.model = XGBRegressor(n_estimators=self.n_estimators)

    def create_features(self, data):
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i-self.window_size:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def fit(self, data):
        self.data = data
        X, y = self.create_features(data.values)
        self.model.fit(X, y)

    def predict(self, steps):
        last_sequence = self.data.values[-self.window_size:].tolist()
        forecast = []
        for _ in range(steps):
            X_input = np.array(last_sequence[-self.window_size:]).reshape(1, -1)
            pred = self.model.predict(X_input)[0]
            forecast.append(pred)
            last_sequence.append(pred)
        return pd.Series(forecast, index=pd.date_range(self.data.index[-1] + pd.DateOffset(months=1), 
                                                     periods=steps, freq='M'))

class RandomForestModel:
    def __init__(self, n_estimators=200, window_size=12):
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.model = RandomForestRegressor(n_estimators=self.n_estimators)

    def create_features(self, data):
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i-self.window_size:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def fit(self, data):
        self.data = data
        X, y = self.create_features(data.values)
        self.model.fit(X, y)

    def predict(self, steps):
        last_sequence = self.data.values[-self.window_size:].tolist()
        forecast = []
        for _ in range(steps):
            X_input = np.array(last_sequence[-self.window_size:]).reshape(1, -1)
            pred = self.model.predict(X_input)[0]
            forecast.append(pred)
            last_sequence.append(pred)
        return pd.Series(forecast, index=pd.date_range(self.data.index[-1] + pd.DateOffset(months=1), 
                                                     periods=steps, freq='M'))

class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def fit(self, data):
        for model in self.models:
            model.fit(data)

    def predict(self, steps):
        forecasts = pd.DataFrame()
        for model in self.models:
            forecasts[model.__class__.__name__] = model.predict(steps)
        ensemble_forecast = forecasts.mean(axis=1)
        return ensemble_forecast