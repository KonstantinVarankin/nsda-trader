import React, { useState, useEffect } from 'react';
import axios from 'axios';
import config from '../config';  // Импортируйте конфигурацию

function PredictionAnalysis() {
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [symbol, setSymbol] = useState('BTCUSDT');

    useEffect(() => {
        const fetchPrediction = async () => {
            try {
                setLoading(true);
                const response = await axios.get(`${config.API_URL}${config.API_V1_STR}/predictions/predict/${symbol}`);
                setPrediction(response.data);
                setLoading(false);
            } catch (err) {
                setError('Failed to fetch prediction');
                setLoading(false);
            }
        };

        fetchPrediction();
        const interval = setInterval(fetchPrediction, 300000); // каждые 5 минут
        return () => clearInterval(interval);
    }, [symbol]);

    if (loading) return <div>Loading prediction...</div>;
    if (error) return <div>Error: {error}</div>;

    return (
        <div>
            <h2>Prediction Analysis</h2>
            {prediction && (
                <div>
                    <p>Prediction type: {prediction.prediction_type}</p>
                    <p>Time horizon: {prediction.time_horizon}</p>
                    <p>Current price: ${prediction.current_price.toFixed(2)}</p>
                    <p>Predicted price: ${prediction.value.toFixed(2)}</p>
                    <p>Change: {((prediction.value - prediction.current_price) / prediction.current_price * 100).toFixed(2)}%</p>
                    <p>Timestamp: {new Date(prediction.timestamp).toLocaleString()}</p>
                    <p>Status: {prediction.status}</p>
                </div>
            )}
        </div>
    );
}

export default PredictionAnalysis;