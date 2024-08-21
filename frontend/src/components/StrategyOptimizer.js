import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import axios from 'axios';

const StrategyOptimizer = () => {
  const [strategies, setStrategies] = useState([]);
  const [selectedStrategy, setSelectedStrategy] = useState('');
  const [paramGrid, setParamGrid] = useState({});
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [backtestResult, setBacktestResult] = useState(null);
  const [token, setToken] = useState('');

  useEffect(() => {
    // Загрузка списка стратегий при монтировании компонента
    fetchStrategies();
  }, []);

  const fetchStrategies = async () => {
    try {
      const response = await axios.get('/strategies', {
        headers: { Authorization: 'Bearer'  }
      });
      setStrategies(response.data);
    } catch (error) {
      console.error('Error fetching strategies:', error);
    }
  };

  const handleStrategyChange = (e) => {
    setSelectedStrategy(e.target.value);
  };

  const handleParamGridChange = (e) => {
    setParamGrid(JSON.parse(e.target.value));
  };

  const handleOptimize = async () => {
    try {
      const response = await axios.post('/optimize', {
        strategy_name: selectedStrategy,
        param_grid: paramGrid
      }, {
        headers: { Authorization: 'Bearer'  }
      });
      setOptimizationResult(response.data);
    } catch (error) {
      console.error('Error optimizing strategy:', error);
    }
  };

  const handleBacktest = async () => {
    try {
      const response = await axios.post('/backtest', {
        strategy_name: selectedStrategy,
        params: optimizationResult.best_params,
        start_date: '2020-01-01',
        end_date: '2021-12-31'
      }, {
        headers: { Authorization: 'Bearer'  }
      });
      setBacktestResult(response.data);
      renderBacktestChart(response.data.returns);
    } catch (error) {
      console.error('Error running backtest:', error);
    }
  };

  const renderBacktestChart = (returns) => {
    const chartData = {
      labels: Array.from({length: returns.length}, (_, i) => i + 1),
      datasets: [
        {
          label: 'Strategy Returns',
          data: returns,
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1
        }
      ]
    };

    const options = {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: 'Backtest Results'
        },
      },
      scales: {
        y: {
          beginAtZero: true
        }
      }
    };

    return <Line data={chartData} options={options} />;
  };

  return (
    <div>
      <h2>Strategy Optimizer</h2>
      <select value={selectedStrategy} onChange={handleStrategyChange}>
        <option value="">Select a strategy</option>
        {strategies.map(strategy => (
          <option key={strategy} value={strategy}>{strategy}</option>
        ))}
      </select>
      <textarea
        placeholder="Enter param grid (JSON format)"
        onChange={handleParamGridChange}
      />
      <button onClick={handleOptimize}>Optimize</button>
      {optimizationResult && (
        <div>
          <h3>Optimization Result</h3>
          <p>Best Sharpe Ratio: {optimizationResult.sharpe_ratio}</p>
          <p>Best Parameters: {JSON.stringify(optimizationResult.best_params)}</p>
          <button onClick={handleBacktest}>Run Backtest</button>
        </div>
      )}
      {backtestResult && (
        <div>
          <h3>Backtest Result</h3>
          <p>Sharpe Ratio: {backtestResult.sharpe_ratio}</p>
          {renderBacktestChart(backtestResult.returns)}
        </div>
      )}
    </div>
  );
};

export default StrategyOptimizer;

