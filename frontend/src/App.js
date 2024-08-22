import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import Login from './components/Login';
import Monitoring from './components/Monitoring';
import AITrading from './components/AITrading';
import OpenOrders from './components/OpenOrders';
import PerformanceMetrics from './components/PerformanceMetrics';
import Settings from './components/Settings';
import StrategyOptimizer from './components/StrategyOptimizer';
import TradingForm from './components/TradingForm';
import MarketSentiment from './components/MarketSentiment';
import NeuralNetworkVisualization from './components/NeuralNetworkVisualization';
import Notifications from './components/Notifications';
import PredictionAnalysis from './components/PredictionAnalysis';
import TradingHistory from './components/TradingHistory';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  return (
    <Router>
      <div className="App">
        {isLoggedIn ? (
          <>
            <nav>
              <ul>
                <li><Link to="/dashboard">Dashboard</Link></li>
                <li><Link to="/monitoring">Monitoring</Link></li>
                <li><Link to="/trading">Trading</Link></li>
                <li><Link to="/ai-trading">AI Trading</Link></li>
                <li><Link to="/performance">Performance</Link></li>
                <li><Link to="/market-sentiment">Market Sentiment</Link></li>
                <li><Link to="/strategy-optimizer">Strategy Optimizer</Link></li>
                <li><Link to="/neural-network">Neural Network</Link></li>
                <li><Link to="/prediction-analysis">Prediction Analysis</Link></li>
                <li><Link to="/trading-history">Trading History</Link></li>
                <li><Link to="/open-orders">Open Orders</Link></li>
                <li><Link to="/notifications">Notifications</Link></li>
                <li><Link to="/settings">Settings</Link></li>
              </ul>
            </nav>
            <Routes>
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/monitoring" element={<Monitoring />} />
              <Route path="/trading" element={<TradingForm />} />
              <Route path="/ai-trading" element={<AITrading />} />
              <Route path="/performance" element={<PerformanceMetrics />} />
              <Route path="/market-sentiment" element={<MarketSentiment />} />
              <Route path="/strategy-optimizer" element={<StrategyOptimizer />} />
              <Route path="/neural-network" element={<NeuralNetworkVisualization />} />
              <Route path="/prediction-analysis" element={<PredictionAnalysis />} />
              <Route path="/trading-history" element={<TradingHistory />} />
              <Route path="/open-orders" element={<OpenOrders />} />
              <Route path="/notifications" element={<Notifications />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </>
        ) : (
          <Login setIsLoggedIn={setIsLoggedIn} />
        )}
      </div>
    </Router>
  );
}

export default App;