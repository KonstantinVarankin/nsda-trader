import React, { useState } from 'react';
import axios from 'axios';

const StrategyOptimizer = () => {
  const [optimizationParams, setOptimizationParams] = useState({
    timeframe: '1h',
    symbol: 'BTCUSDT',
    startDate: '',
    endDate: '',
    populationSize: 50,
    generations: 100,
  });
  const [optimizationResults, setOptimizationResults] = useState(null);
  const [isOptimizing, setIsOptimizing] = useState(false);

  const handleInputChange = (e) => {
    setOptimizationParams({
      ...optimizationParams,
      [e.target.name]: e.target.value,
    });
  };

  const handleOptimize = async (e) => {
    e.preventDefault();
    setIsOptimizing(true);
    try {
      const response = await axios.post('/api/optimize-strategy', optimizationParams);
      setOptimizationResults(response.data);
    } catch (error) {
      console.error('Error optimizing strategy:', error);
    }
    setIsOptimizing(false);
  };

  return (
    <div>
      <h2>Strategy Optimizer</h2>
      <form onSubmit={handleOptimize}>
        <div>
          <label htmlFor="timeframe">Timeframe:</label>
          <select name="timeframe" value={optimizationParams.timeframe} onChange={handleInputChange}>
            <option value="1m">1 minute</option>
            <option value="5m">5 minutes</option>
            <option value="15m">15 minutes</option>
            <option value="1h">1 hour</option>
            <option value="4h">4 hours</option>
            <option value="1d">1 day</option>
          </select>
        </div>
        <div>
          <label htmlFor="symbol">Symbol:</label>
          <input type="text" name="symbol" value={optimizationParams.symbol} onChange={handleInputChange} />
        </div>
        <div>
          <label htmlFor="startDate">Start Date:</label>
          <input type="date" name="startDate" value={optimizationParams.startDate} onChange={handleInputChange} />
        </div>
        <div>
          <label htmlFor="endDate">End Date:</label>
          <input type="date" name="endDate" value={optimizationParams.endDate} onChange={handleInputChange} />
        </div>
        <div>
          <label htmlFor="populationSize">Population Size:</label>
          <input type="number" name="populationSize" value={optimizationParams.populationSize} onChange={handleInputChange} />
        </div>
        <div>
          <label htmlFor="generations">Generations:</label>
          <input type="number" name="generations" value={optimizationParams.generations} onChange={handleInputChange} />
        </div>
        <button type="submit" disabled={isOptimizing}>
          {isOptimizing ? 'Optimizing...' : 'Optimize Strategy'}
        </button>
      </form>
      {optimizationResults && (
        <div>
          <h3>Optimization Results</h3>
          <p>Best Fitness: {optimizationResults.bestFitness}</p>
          <p>Best Parameters:</p>
          <ul>
            {Object.entries(optimizationResults.bestParameters).map(([key, value]) => (
              <li key={key}>{key}: {value}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default StrategyOptimizer;