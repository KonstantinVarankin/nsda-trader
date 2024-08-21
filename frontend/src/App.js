import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Switch, Link } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import Login from './components/Login';
import Monitoring from './components/Monitoring';
import Notifications from './components/Notifications';
import OpenOrders from './components/OpenOrders';
import PerformanceMetrics from './components/PerformanceMetrics';
import Settings from './components/Settings';
import StrategyOptimizer from './components/StrategyOptimizer';
import TradingForm from './components/TradingForm';

function App() {
  const [token, setToken] = useState(localStorage.getItem('token'));

  const handleLogin = (newToken) => {
    setToken(newToken);
    localStorage.setItem('token', newToken);
  };

  const handleLogout = () => {
    setToken(null);
    localStorage.removeItem('token');
  };

  return (
    <Router>
      <div className="App">
        {token ? (
          <>
            <nav>
              <ul>
                <li><Link to="/">Dashboard</Link></li>
                <li><Link to="/monitoring">Monitoring</Link></li>
                <li><Link to="/notifications">Notifications</Link></li>
                <li><Link to="/open-orders">Open Orders</Link></li>
                <li><Link to="/performance">Performance Metrics</Link></li>
                <li><Link to="/settings">Settings</Link></li>
                <li><Link to="/strategy-optimizer">Strategy Optimizer</Link></li>
                <li><Link to="/trading">Trading Form</Link></li>
              </ul>
            </nav>
            <button onClick={handleLogout}>Logout</button>
            <Switch>
              <Route exact path="/" component={Dashboard} />
              <Route path="/monitoring" component={Monitoring} />
              <Route path="/notifications" component={Notifications} />
              <Route path="/open-orders" component={OpenOrders} />
              <Route path="/performance" component={PerformanceMetrics} />
              <Route path="/settings" component={Settings} />
              <Route path="/strategy-optimizer" component={StrategyOptimizer} />
              <Route path="/trading" component={TradingForm} />
            </Switch>
          </>
        ) : (
          <Login onLogin={handleLogin} />
        )}
      </div>
    </Router>
  );
}

export default App;
