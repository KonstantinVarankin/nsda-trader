import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, Grid } from '@mui/material';
import axios from 'axios';

const PerformanceMetrics = () => {
  const [metrics, setMetrics] = useState({
    totalReturn: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    winRate: 0,
  });

  useEffect(() => {
    fetchPerformanceMetrics();
  }, []);

  const fetchPerformanceMetrics = async () => {
    try {
      // In a real application, you would fetch this data from your backend
      // const response = await axios.get('/api/v1/performance/metrics');
      // setMetrics(response.data);

      // For now, we'll use mock data
      setMetrics({
        totalReturn: 15.5,
        sharpeRatio: 1.2,
        maxDrawdown: -10.3,
        winRate: 60.5,
      });
    } catch (error) {
      console.error('Error fetching performance metrics:', error);
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Performance Metrics</Typography>
        <Grid container spacing={2}>
          <Grid item xs={6} md={3}>
            <Typography variant="subtitle1">Total Return</Typography>
            <Typography variant="h5">{metrics.totalReturn.toFixed(2)}%</Typography>
          </Grid>
          <Grid item xs={6} md={3}>
            <Typography variant="subtitle1">Sharpe Ratio</Typography>
            <Typography variant="h5">{metrics.sharpeRatio.toFixed(2)}</Typography>
          </Grid>
          <Grid item xs={6} md={3}>
            <Typography variant="subtitle1">Max Drawdown</Typography>
            <Typography variant="h5">{metrics.maxDrawdown.toFixed(2)}%</Typography>
          </Grid>
          <Grid item xs={6} md={3}>
            <Typography variant="subtitle1">Win Rate</Typography>
            <Typography variant="h5">{metrics.winRate.toFixed(2)}%</Typography>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default PerformanceMetrics;

