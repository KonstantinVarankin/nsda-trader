from fastapi import APIRouter, HTTPException
from app.services.analysis import MarketAnalyzer
from app.services.data_collection import DataCollector
from app.services.preprocessing import DataPreprocessor
import pandas as pd

router = APIRouter()
analyzer = MarketAnalyzer()
collector = DataCollector()
preprocessor = DataPreprocessor()


@router.get("/predict/{symbol}")
async def get_prediction(symbol: str):
    try:
        # Fetch and preprocess data
        data = collector.fetch_ohlcv(symbol)
        preprocessed_data = preprocessor.normalize_ohlcv(data)
        prepared_data = analyzer.prepare_data(preprocessed_data)

        # Prepare features and target
        features = prepared_data.drop(['close'], axis=1)
        target = (prepared_data['close'].pct_change().shift(-1) > 0).astype(int)

        # Train models and make predictions
        rf_metrics = analyzer.train_random_forest(features[:-1], target[:-1])

        analyzer.create_lstm_model((features.shape[1], 1))
        lstm_metrics = analyzer.train_lstm(features[:-1], target[:-1])

        latest_data = features.iloc[-1].values.reshape(1, -1)
        rf_prediction = analyzer.predict_random_forest(latest_data)
        lstm_prediction = analyzer.predict_lstm(latest_data)

        return {
            "symbol": symbol,
            "rf_prediction": int(rf_prediction[0]),
            "lstm_prediction": int(lstm_prediction[0]),
            "rf_metrics": rf_metrics,
            "lstm_metrics": lstm_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest/{symbol}")
async def backtest_strategy(symbol: str):
    try:
        # Fetch and preprocess data
        data = collector.fetch_ohlcv(symbol)
        preprocessed_data = preprocessor.normalize_ohlcv(data)
        prepared_data = analyzer.prepare_data(preprocessed_data)

        # Backtest the strategy
        backtest_results = analyzer.backtest_strategy(prepared_data)

        return {
            "symbol": symbol,
            "backtest_results": backtest_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))