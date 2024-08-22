import React from 'react';
import PerformanceMetrics from './PerformanceMetrics';
import OpenOrders from './OpenOrders';
import AITrading from './AITrading';
import PredictionAnalysis from './PredictionAnalysis';
import NeuralNetworkVisualization from './NeuralNetworkVisualization';
import MarketSentiment from './MarketSentiment';
import TradingHistory from './TradingHistory';
import StrategyOptimizer from './StrategyOptimizer';

const Dashboard = () => {
  const neuralNetworkLayers = [4, 8, 8, 1]; // Example network architecture

  return (
    <div>
      <h1>NSDA-Trader Dashboard</h1>
      <div className="dashboard-grid">
        <div className="dashboard-item">
          <PerformanceMetrics />
        </div>
        <div className="dashboard-item">
          <OpenOrders />
        </div>
        <div className="dashboard-item">
          <AITrading />
        </div>
        <div className="dashboard-item">
          <PredictionAnalysis />
        </div>
        <div className="dashboard-item">
          <h2>Neural Network Architecture</h2>
          <NeuralNetworkVisualization layers={neuralNetworkLayers} />
        </div>
        <div className="dashboard-item">
          <MarketSentiment />
        </div>
        <div className="dashboard-item">
          <TradingHistory />
        </div>
        <div className="dashboard-item">
          <StrategyOptimizer />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;