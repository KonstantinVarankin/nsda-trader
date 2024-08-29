import pandas as pd

from app.data.market_data import get_historical_data, get_latest_data, preprocess_data
from app.ml.neural_network import TradingNeuralNetwork, run_neural_network, get_prediction



class AITrader:
    def __init__(self, symbol="AAPL"):
        self.model = None
        self.symbol = symbol
        self.scaler = None

    def initialize(self):
        # Получаем исторические данные
        historical_data = get_historical_data(self.symbol)

        if historical_data is None or historical_data.empty:
            raise Exception(f"Не удалось получить исторические данные для {self.symbol}")

        # Предобработка данных
        X, y, self.scaler = preprocess_data(historical_data)

        # Запускаем нейронную сеть
        self.model = run_neural_network(X, y)
        print(f"Нейронная сеть инициализирована и обучена для {self.symbol}")

    def make_trading_decision(self):
        if self.model is None:
            raise Exception("Нейронная сеть не инициализирована")

        # Получаем последние данные рынка
        latest_data = get_latest_data(self.symbol)

        if latest_data is None or latest_data.empty:
            raise Exception(f"Не удалось получить последние данные для {self.symbol}")

        # Предобработка последних данных
        X_latest, _ = preprocess_data(latest_data, self.scaler)

        # Получаем прогноз от нейронной сети
        prediction = get_prediction(self.model, X_latest)

        # Простая торговая логика на основе прогноза
        current_price = latest_data['Close'].iloc[-1]
        if prediction > current_price * 1.01:  # Если прогноз на 1% выше текущей цены
            return "BUY"
        elif prediction < current_price * 0.99:  # Если прогноз на 1% ниже текущей цены
            return "SELL"
        else:
            return "HOLD"

    def set_symbol(self, symbol):
        self.symbol = symbol
        self.initialize()  # Переинициализируем модель для нового символа


# Создание и инициализация AI-трейдера
ai_trader = TradingNeuralNetwork()


# Функция для использования AI-трейдера
def use_ai_trader(symbol="AAPL"):
    ai_trader.set_symbol(symbol)
    decision = ai_trader.make_trading_decision()
    return decision


# Пример использования
if __name__ == "__main__":
    symbol = "AAPL"  # Можно изменить на любой другой символ
    try:
        decision = use_ai_trader(symbol)
        print(f"Торговое решение для {symbol}: {decision}")
    except Exception as e:
        print(f"Ошибка: {str(e)}")