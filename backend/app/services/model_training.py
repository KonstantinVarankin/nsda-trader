import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

from app.ml.neural_network import LSTMModel
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = MinMaxScaler()
        logger.info(f"Model trainer initialized. Using device: {self.device}")

    def prepare_data(self, data: pd.DataFrame, look_back: int = 60) -> Tuple[torch.Tensor, torch.Tensor]:
        scaled_data = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i:(i + look_back)])
            y.append(scaled_data[i + look_back, 3])  # Предсказываем цену закрытия
        return torch.FloatTensor(X).to(self.device), torch.FloatTensor(y).to(self.device)

    def train_model(self, symbol: str, data: pd.DataFrame, epochs: int = 100, batch_size: int = 64) -> LSTMModel:
        X, y = self.prepare_data(data)
        input_size = X.shape[2]

        model = LSTMModel(input_size).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

        return model

    def save_model(self, model: LSTMModel, symbol: str, interval: str):
        model_dir = os.path.join(settings.PROJECT_ROOT, 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{symbol}_{interval}.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model for {symbol} {interval} saved to {model_path}")

    def load_model(self, symbol: str, interval: str) -> LSTMModel:
        model_path = os.path.join(settings.PROJECT_ROOT, 'models', f"{symbol}_{interval}.pth")
        if os.path.exists(model_path):
            model = LSTMModel(input_size=5)  # Предполагаем, что у нас 5 входных признаков
            model.load_state_dict(torch.load(model_path))
            model.to(self.device)
            model.eval()
            logger.info(f"Model for {symbol} {interval} loaded from {model_path}")
            return model
        else:
            logger.error(f"Model file not found: {model_path}")
            return None


def train_and_save_model(symbol: str, interval: str, data: pd.DataFrame):
    trainer = ModelTrainer()
    model = trainer.train_model(symbol, data)
    trainer.save_model(model, symbol, interval)


# Пример использования:
if __name__ == "__main__":
    # Здесь должен быть код для загрузки данных
    # Например:
    # data = load_historical_data('BTCUSDT', '1h')
    # train_and_save_model('BTCUSDT', '1h', data)
    pass