import React, { useState, useEffect } from 'react';
import PerformanceMetrics from './PerformanceMetrics';
import { Grid, Paper, Typography, Card, CardContent } from '@mui/material';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart, Pie, Cell } from 'recharts';
import axios from 'axios';

const Dashboard = () => {
  const [accountBalance, setAccountBalance] = useState({});
  const [performanceData, setPerformanceData] = useState([]);
  const [tradeHistory, setTradeHistory] = useState([]);
  const [assetAllocation, setAssetAllocation] = useState([]);

  useEffect(() => {
    fetchAccountBalance();
    fetchPerformanceData();
    fetchTradeHistory();
    fetchAssetAllocation();
  }, []);

  const fetchAccountBalance = async () => {
    try {
      const response = await axios.get('/api/v1/trading/account-balance');
      setAccountBalance(response.data);
    } catch (error) {
      console.error('Error fetching account balance:', error);
    }
  };

  const fetchPerformanceData = async () => {
    // This is a placeholder. In a real application, you would fetch actual performance data.
    const mockData = [
      { date: '2023-01-01', balance: 10000 },
      { date: '2023-02-01', balance: 10500 },
      { date: '2023-03-01', balance: 11000 },
      { date: '2023-04-01', balance: 10800 },
      { date: '2023-05-01', balance: 11500 },
    ];
    setPerformanceData(mockData);
  };

  const fetchTradeHistory = async () => {
    // This is a placeholder. In a real application, you would fetch actual trade history.
    const mockData = [
      { date: '2023-05-01', profit: 200 },
      { date: '2023-05-02', profit: -50 },
      { date: '2023-05-03', profit: 100 },
      { date: '2023-05-04', profit: 300 },
      { date: '2023-05-05', profit: -100 },
    ];
    setTradeHistory(mockData);
  };

  const fetchAssetAllocation = async () => {
    // This is a placeholder. In a real application, you would calculate this from the account balance.
    const mockData = [
      { name: 'BTC', value: 5000 },
      { name: 'ETH', value: 3000 },
      { name: 'USDT', value: 2000 },
      { name: 'Other', value: 1500 },
    ];
    setAssetAllocation(mockData);
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6">Account Balance</Typography>
            {Object.entries(accountBalance).map(([asset, balance]) => (
              <Typography key={asset}>{asset}: {balance}</Typography>
            ))}
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6">Performance Chart</Typography>
            <LineChart width={500} height={300} data={performanceData}>
              <XAxis dataKey="date" />
              <YAxis />
              <CartesianGrid strokeDasharray="3 3" />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="balance" stroke="#8884d8" />
            </LineChart>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6">Trade History</Typography>
            <BarChart width={500} height={300} data={tradeHistory}>
              <XAxis dataKey="date" />
              <YAxis />
              <CartesianGrid strokeDasharray="3 3" />
              <Tooltip />
              <Legend />
              <Bar dataKey="profit" fill="#82ca9d" />
            </BarChart>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6">Asset Allocation</Typography>
            <PieChart width={400} height={400}>
              <Pie
                data={assetAllocation}
                cx={200}
                cy={200}
                labelLine={false}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {assetAllocation.map((entry, index) => (
                  "<Cell 'key={`cell-\${index}`}', 'fill={COLORS\[index % COLORS.length\]}' />", "<Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />"
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default Dashboard;






