import React, { useState } from 'react';
import axios from 'axios';

function AITrading() {
  const [symbol, setSymbol] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [tradeDecision, setTradeDecision] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const trainModel = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await axios.post(`/api/ai-trading/train?symbol=${symbol}`);
      alert(response.data.message);
    } catch (error) {
      setError('Error training model: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const getPrediction = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await axios.get(`/api/ai-trading/predict?symbol=${symbol}`);
      setPrediction(response.data);
    } catch (error) {
      setError('Error getting prediction: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const evaluateModel = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await axios.get(`/api/ai-trading/evaluate?symbol=${symbol}`);
      setEvaluation(response.data.evaluation);
    } catch (error) {
      setError('Error evaluating model: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const getTradeDecision = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await axios.get(`/api/ai-trading/trade-decision?symbol=${symbol}`);
      setTradeDecision(response.data);
    } catch (error) {
      setError('Error getting trade decision: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>AI Trading</h2>
      <input
        type="text"
        value={symbol}
        onChange={(e) => setSymbol(e.target.value)}
        placeholder="Enter stock symbol"
      />
      <button onClick={trainModel} disabled={loading || !symbol}>
        Train Model
      </button>
      <button onClick={getPrediction} disabled={loading || !symbol}>
        Get Prediction
      </button>
      <button onClick={evaluateModel} disabled={loading || !symbol}>
        Evaluate Model
      </button>
      <button onClick={getTradeDecision} disabled={loading || !symbol}>
        Get Trade Decision
      </button>
      {loading && <p>Loading...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {prediction && (
        <div>
          <h3>Prediction:</h3>
          <p>Symbol: {prediction.symbol}</p>
          <p>Date: {prediction.date}</p>
          <p>Predicted Price: </p>
        </div>
      )}
      {evaluation && (
        <div>
          <h3>Model Evaluation:</h3>
          <p>MSE: {evaluation.mse.toFixed(4)}</p>
          <p>RMSE: {evaluation.rmse.toFixed(4)}</p>
        </div>
      )}
      {tradeDecision && (
        <div>
          <h3>Trade Decision:</h3>
          <p>Symbol: {tradeDecision.symbol}</p>
          <p>Current Price: </p>
          <p>Predicted Price: </p>
          <p>Action: {tradeDecision.decision.action}</p>
          <p>Quantity: {tradeDecision.decision.quantity}</p>
          <p>Expected Return: {(tradeDecision.decision.expected_return * 100).toFixed(2)}%</p>
          <p>Sharpe Ratio: {tradeDecision.decision.sharpe_ratio.toFixed(4)}</p>
          <p>Volatility: {(tradeDecision.decision.volatility * 100).toFixed(2)}%</p>
          {tradeDecision.decision.stop_loss && (
            <p>Stop Loss: </p>
          )}
          {tradeDecision.decision.take_profit && (
            <p>Take Profit: </p>
          )}
        </div>
      )}
    </div>
  );
}

export default AITrading;
