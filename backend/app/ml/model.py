import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from app.data.market_data import get_stock_data, preprocess_data, prepare_data_for_prediction

class NSDATradingModel:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(100, return_sequences=False),
            Dropout(0.2),
            Dense(50),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def train(self, symbol, start_date, end_date):
        df = get_stock_data(symbol, start_date, end_date)
        X, y, _ = preprocess_data(df)
        
        # Разделение на обучающую и валидационную выборки
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=200,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]
        )

    def predict(self, symbol, date):
        df = get_stock_data(symbol, date, date)
        X, scaler = prepare_data_for_prediction(df)
        prediction = self.model.predict(X)
        return scaler.inverse_transform(prediction)[0][0]

    def evaluate(self, symbol, start_date, end_date):
        df = get_stock_data(symbol, start_date, end_date)
        X, y, scaler = preprocess_data(df)
        
        predictions = self.model.predict(X)
        predictions = scaler.inverse_transform(predictions)
        y = scaler.inverse_transform(y.reshape(-1, 1))
        
        mse = np.mean((predictions - y) ** 2)
        rmse = np.sqrt(mse)
        
        return {"mse": float(mse), "rmse": float(rmse)}
