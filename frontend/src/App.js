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
                <li><Link to="/settings">Settings</Link></li>
              </ul>
            </nav>
            <Routes>
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/monitoring" element={<Monitoring />} />
              <Route path="/trading" element={<TradingForm />} />
              <Route path="/ai-trading" element={<AITrading />} />
              <Route path="/performance" element={<PerformanceMetrics />} />
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
