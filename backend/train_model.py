# train_model.py
from datetime import datetime, timedelta
import os

try:
    from app.ml.model import NSDATradingModel
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что все необходимые модули установлены.")
    exit(1)


def train_model():
    try:
        model = NSDATradingModel()
        symbol = "AAPL"  # Пример: обучаем на данных Apple
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)  # Данные за последние 2 года

        print(
            f"Начало обучения модели на данных {symbol} с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}")
        model.train(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        print("Обучение модели завершено")

        # Оценка модели
        evaluation = model.evaluate(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        print(f"Оценка модели: MSE = {evaluation['mse']:.4f}, RMSE = {evaluation['rmse']:.4f}")

        # Пример предсказания
        prediction_date = end_date + timedelta(days=1)
        prediction = model.predict(symbol, prediction_date.strftime('%Y-%m-%d'))
        print(f"Предсказание на {prediction_date.strftime('%Y-%m-%d')}: {prediction:.2f}")

    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")


if __name__ == "__main__":
    train_model()