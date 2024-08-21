import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, Grid, CircularProgress } from '@mui/material';
import axios from 'axios';

const PerformanceMetrics = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchPerformanceMetrics();
  }, []);

  const fetchPerformanceMetrics = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://localhost:8000/api/v1/performance/metrics');
      setMetrics(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching performance metrics:', error);
      setError('Failed to fetch performance metrics. Please try again later.');
      setLoading(false);
    }
  };

  if (loading) {
    return <CircularProgress />;
  }

  if (error) {
    return <Typography color="error">{error}</Typography>;
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>Performance Metrics</Typography>
        {metrics ? (
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
        ) : (
          <Typography>No metrics available</Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default PerformanceMetrics;