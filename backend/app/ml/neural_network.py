# backend/app/ml/neural_network.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


class TradingNeuralNetwork:
    def __init__(self):
        self.model = None

    def create_model(self, input_shape):
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    def train(self, X, y, epochs=100, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X):
        return self.model.predict(X)


def run_neural_network(data):
    nn = TradingNeuralNetwork()

    # Проверяем форму входных данных
    print(f"Input data shape: {data.shape}")

    if len(data.shape) < 2:
        raise ValueError("Input data must be 2D or 3D")

    # Если данные 2D, преобразуем их в 3D для LSTM
    if len(data.shape) == 2:
        X = data[:-1].reshape((data.shape[0] - 1, 1, data.shape[1]))
        y = data[1:, 0]  # Предполагаем, что целевая переменная - первый столбец
    else:
        X = data[:-1]
        y = data[1:, 0, 0]  # Предполагаем, что целевая переменная - первый элемент каждого временного шага

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    nn.create_model((X.shape[1], X.shape[2]))
    nn.train(X, y)
    return nn


def get_prediction(model, data):
    # Убедимся, что данные имеют правильную форму (batch_size, time_steps, features)
    if len(data.shape) == 2:
        data = data.reshape((1, data.shape[0], data.shape[1]))
    return model.predict(data)