import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';

const TradingHistory = () => {
  const [tradingHistory, setTradingHistory] = useState(null);

  useEffect(() => {
    const fetchTradingHistory = async () => {
      try {
        const response = await axios.get('/api/trading-history');
        setTradingHistory(response.data);
      } catch (error) {
        console.error('Error fetching trading history:', error);
      }
    };

    fetchTradingHistory();
  }, []);

  if (!tradingHistory) {
    return <div>Loading trading history...</div>;
  }

  const chartData = {
    labels: tradingHistory.dates,
    datasets: [
      {
        label: 'Portfolio Value',
        data: tradingHistory.portfolioValues,
        borderColor: 'green',
        fill: false,
      },
    ],
  };

  return (
    <div>
      <h2>Trading History</h2>
      <Line data={chartData} />
      <div>
        <h3>Performance Metrics</h3>
        <p>Total Profit/Loss: ${tradingHistory.totalProfitLoss.toFixed(2)}</p>
        <p>Win Rate: {tradingHistory.winRate.toFixed(2)}%</p>
        <p>Sharpe Ratio: {tradingHistory.sharpeRatio.toFixed(2)}</p>
      </div>
    </div>
  );
};

export default TradingHistory;