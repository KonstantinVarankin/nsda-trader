import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from ta import add_all_ta_features


class MarketAnalyzer:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lstm_model = None

    def prepare_data(self, data):
        """Prepare data for analysis by adding technical indicators."""
        df = add_all_ta_features(
            data, open="open", high="high", low="low", close="close", volume="volume"
        )
        return df.dropna()

    def split_data(self, X, y, test_size=0.2):
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=42)

    def train_random_forest(self, X, y):
        """Train a Random Forest model for classification."""
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        self.rf_model.fit(X_train, y_train)
        y_pred = self.rf_model.predict(X_test)
        return self.evaluate_model(y_test, y_pred)

    def predict_random_forest(self, X):
        """Make predictions using the trained Random Forest model."""
        return self.rf_model.predict(X)

    def create_lstm_model(self, input_shape):
        """Create and compile an LSTM model."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        self.lstm_model = model

    def train_lstm(self, X, y, epochs=50, batch_size=32):
        """Train the LSTM model."""
        if self.lstm_model is None:
            raise ValueError("LSTM model not created. Call create_lstm_model first.")
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        self.lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        y_pred = (self.lstm_model.predict(X_test) > 0.5).astype("int32")
        return self.evaluate_model(y_test, y_pred)

    def predict_lstm(self, X):
        """Make predictions using the trained LSTM model."""
        if self.lstm_model is None:
            raise ValueError("LSTM model not trained. Train the model first.")
        return (self.lstm_model.predict(X) > 0.5).astype("int32")

    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }

    def generate_trading_signals(self, data, threshold=0.5):
        """Generate trading signals based on model predictions and technical indicators."""
        signals = []
        rf_pred = self.predict_random_forest(data)
        lstm_pred = self.predict_lstm(data)

        for i in range(len(data)):
            if rf_pred[i] == 1 and lstm_pred[i] == 1 and data['rsi'][i] < 30:
                signals.append(1)  # Strong buy signal
            elif rf_pred[i] == 0 and lstm_pred[i] == 0 and data['rsi'][i] > 70:
                signals.append(-1)  # Strong sell signal
            else:
                signals.append(0)  # Hold

        return signals

    def backtest_strategy(self, data, initial_balance=10000):
        """Backtest the trading strategy."""
        balance = initial_balance
        position = 0
        trades = []

        for i, signal in enumerate(self.generate_trading_signals(data)):
            if signal == 1 and position == 0:  # Buy signal
                position = balance / data['close'][i]
                balance = 0
                trades.append(('buy', data['close'][i], position))
            elif signal == -1 and position > 0:  # Sell signal
                balance = position * data['close'][i]
                position = 0
                trades.append(('sell', data['close'][i], balance))

        if position > 0:  # Close any open position at the end
            balance = position * data['close'][-1]

        return {
            'final_balance': balance,
            'return': (balance - initial_balance) / initial_balance * 100,
            'trades': trades
        }