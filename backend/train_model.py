# train_model.py
from app.ml.model import NSDATradingModel
from app.data.market_data import get_stock_data
from datetime import datetime, timedelta


def train_model():
    model = NSDATradingModel()
    symbol = "AAPL"  # Пример: обучаем на данных Apple
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # Данные за последние 2 года

    data = get_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    model.train(data)
    print(f"Модель обучена на данных {symbol}")


if __name__ == "__main__":
    train_model()