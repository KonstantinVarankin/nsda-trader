import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  TextField, 
  Button, 
  Grid, 
  Switch, 
  FormControlLabel 
} from '@material-ui/core';
import axios from 'axios';

const Settings = () => {
  const [settings, setSettings] = useState({
  tradingEnabled: false,
  maxTradeAmount: 0,
  riskLevel: 'medium',
  tradingPairs: '',
  stopLossPercentage: 0,
  takeProfitPercentage: 0,
  strategy: 'MA_CROSSOVER',
  maPeriodsShort: 10,
  maPeriodsLong: 20,
  rsiPeriods: 14,
  rsiBuyThreshold: 30,
  rsiSellThreshold: 70,
  tradingEnabled: false,
  maxTradeAmount: 0,
  riskLevel: 'medium',
  tradingPairs: '',
  stopLossPercentage: 0,
  takeProfitPercentage: 0,
});
const [errors, setErrors] = useState({
    tradingEnabled: false,
    maxTradeAmount: 0,
    riskLevel: 'medium',
    tradingPairs: '',
    stopLossPercentage: 0,
    takeProfitPercentage: 0,
  });

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      const response = await axios.get('/api/v1/settings');
      setSettings(response.data);
    } catch (error) {
      console.error('Error fetching settings:', error);
    }
  };

  const handleChange = (event) => {
    const { name, value, checked } = event.target;
    setSettings(prevSettings => ({
      ...prevSettings,
      [name]: name === 'tradingEnabled' ? checked : value,
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      await axios.post('/api/v1/settings', settings);
      alert('Settings saved successfully!');
      setErrors({});
    } catch (error) {
      console.error('Error saving settings:', error);
      if (error.response && error.response.data && error.response.data.detail) {
        setErrors(error.response.data.detail.reduce((acc, curr) => {
          acc[curr.loc[1]] = curr.msg;
          return acc;
        }, {}));
      } else {
        alert('Error saving settings. Please try again.');
      }
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>Trading Settings</Typography>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.tradingEnabled}
                    onChange={handleChange}
                    name="tradingEnabled"
                    color="primary"
                  />
                }
                label="Enable Trading"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Strategy"
                name="strategy"
                select
                SelectProps={{ native: true }}
                value={settings.strategy}
                onChange={handleChange}
              >
                <option value="MA_CROSSOVER">Moving Average Crossover</option>
                <option value="RSI">Relative Strength Index (RSI)</option>
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Short MA Periods"
                name="maPeriodsShort"
                type="number"
                value={settings.maPeriodsShort}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Long MA Periods"
                name="maPeriodsLong"
                type="number"
                value={settings.maPeriodsLong}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Periods"
                name="rsiPeriods"
                type="number"
                value={settings.rsiPeriods}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Buy Threshold"
                name="rsiBuyThreshold"
                type="number"
                value={settings.rsiBuyThreshold}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Sell Threshold"
                name="rsiSellThreshold"
                type="number"
                value={settings.rsiSellThreshold}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Max Trade Amount"
                error={!!errors.maxTradeAmount}
                helperText={errors.maxTradeAmount}
                name="maxTradeAmount"
                type="number"
                value={settings.maxTradeAmount}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Strategy"
                name="strategy"
                select
                SelectProps={{ native: true }}
                value={settings.strategy}
                onChange={handleChange}
              >
                <option value="MA_CROSSOVER">Moving Average Crossover</option>
                <option value="RSI">Relative Strength Index (RSI)</option>
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Short MA Periods"
                name="maPeriodsShort"
                type="number"
                value={settings.maPeriodsShort}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Long MA Periods"
                name="maPeriodsLong"
                type="number"
                value={settings.maPeriodsLong}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Periods"
                name="rsiPeriods"
                type="number"
                value={settings.rsiPeriods}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Buy Threshold"
                name="rsiBuyThreshold"
                type="number"
                value={settings.rsiBuyThreshold}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Sell Threshold"
                name="rsiSellThreshold"
                type="number"
                value={settings.rsiSellThreshold}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Risk Level"
                error={!!errors.riskLevel}
                helperText={errors.riskLevel}
                name="riskLevel"
                select
                SelectProps={{ native: true }}
                value={settings.riskLevel}
                onChange={handleChange}
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </TextField>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Strategy"
                name="strategy"
                select
                SelectProps={{ native: true }}
                value={settings.strategy}
                onChange={handleChange}
              >
                <option value="MA_CROSSOVER">Moving Average Crossover</option>
                <option value="RSI">Relative Strength Index (RSI)</option>
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Short MA Periods"
                name="maPeriodsShort"
                type="number"
                value={settings.maPeriodsShort}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Long MA Periods"
                name="maPeriodsLong"
                type="number"
                value={settings.maPeriodsLong}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Periods"
                name="rsiPeriods"
                type="number"
                value={settings.rsiPeriods}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Buy Threshold"
                name="rsiBuyThreshold"
                type="number"
                value={settings.rsiBuyThreshold}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Sell Threshold"
                name="rsiSellThreshold"
                type="number"
                value={settings.rsiSellThreshold}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Trading Pairs (comma-separated)"
                name="tradingPairs"
                value={settings.tradingPairs}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Strategy"
                name="strategy"
                select
                SelectProps={{ native: true }}
                value={settings.strategy}
                onChange={handleChange}
              >
                <option value="MA_CROSSOVER">Moving Average Crossover</option>
                <option value="RSI">Relative Strength Index (RSI)</option>
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Short MA Periods"
                name="maPeriodsShort"
                type="number"
                value={settings.maPeriodsShort}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Long MA Periods"
                name="maPeriodsLong"
                type="number"
                value={settings.maPeriodsLong}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Periods"
                name="rsiPeriods"
                type="number"
                value={settings.rsiPeriods}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Buy Threshold"
                name="rsiBuyThreshold"
                type="number"
                value={settings.rsiBuyThreshold}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Sell Threshold"
                name="rsiSellThreshold"
                type="number"
                value={settings.rsiSellThreshold}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Stop Loss Percentage"
                error={!!errors.stopLossPercentage}
                helperText={errors.stopLossPercentage}
                name="stopLossPercentage"
                type="number"
                value={settings.stopLossPercentage}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Strategy"
                name="strategy"
                select
                SelectProps={{ native: true }}
                value={settings.strategy}
                onChange={handleChange}
              >
                <option value="MA_CROSSOVER">Moving Average Crossover</option>
                <option value="RSI">Relative Strength Index (RSI)</option>
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Short MA Periods"
                name="maPeriodsShort"
                type="number"
                value={settings.maPeriodsShort}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Long MA Periods"
                name="maPeriodsLong"
                type="number"
                value={settings.maPeriodsLong}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Periods"
                name="rsiPeriods"
                type="number"
                value={settings.rsiPeriods}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Buy Threshold"
                name="rsiBuyThreshold"
                type="number"
                value={settings.rsiBuyThreshold}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Sell Threshold"
                name="rsiSellThreshold"
                type="number"
                value={settings.rsiSellThreshold}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Take Profit Percentage"
                error={!!errors.takeProfitPercentage}
                helperText={errors.takeProfitPercentage}
                name="takeProfitPercentage"
                type="number"
                value={settings.takeProfitPercentage}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Strategy"
                name="strategy"
                select
                SelectProps={{ native: true }}
                value={settings.strategy}
                onChange={handleChange}
              >
                <option value="MA_CROSSOVER">Moving Average Crossover</option>
                <option value="RSI">Relative Strength Index (RSI)</option>
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Short MA Periods"
                name="maPeriodsShort"
                type="number"
                value={settings.maPeriodsShort}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Long MA Periods"
                name="maPeriodsLong"
                type="number"
                value={settings.maPeriodsLong}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Periods"
                name="rsiPeriods"
                type="number"
                value={settings.rsiPeriods}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Buy Threshold"
                name="rsiBuyThreshold"
                type="number"
                value={settings.rsiBuyThreshold}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Sell Threshold"
                name="rsiSellThreshold"
                type="number"
                value={settings.rsiSellThreshold}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12}>
              <Button type="submit" variant="contained" color="primary">
                Save Settings
              </Button>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Strategy"
                name="strategy"
                select
                SelectProps={{ native: true }}
                value={settings.strategy}
                onChange={handleChange}
              >
                <option value="MA_CROSSOVER">Moving Average Crossover</option>
                <option value="RSI">Relative Strength Index (RSI)</option>
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Short MA Periods"
                name="maPeriodsShort"
                type="number"
                value={settings.maPeriodsShort}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Long MA Periods"
                name="maPeriodsLong"
                type="number"
                value={settings.maPeriodsLong}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Periods"
                name="rsiPeriods"
                type="number"
                value={settings.rsiPeriods}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Buy Threshold"
                name="rsiBuyThreshold"
                type="number"
                value={settings.rsiBuyThreshold}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Sell Threshold"
                name="rsiSellThreshold"
                type="number"
                value={settings.rsiSellThreshold}
                onChange={handleChange}
              />
            </Grid>
          </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Strategy"
                name="strategy"
                select
                SelectProps={{ native: true }}
                value={settings.strategy}
                onChange={handleChange}
              >
                <option value="MA_CROSSOVER">Moving Average Crossover</option>
                <option value="RSI">Relative Strength Index (RSI)</option>
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Short MA Periods"
                name="maPeriodsShort"
                type="number"
                value={settings.maPeriodsShort}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Long MA Periods"
                name="maPeriodsLong"
                type="number"
                value={settings.maPeriodsLong}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Periods"
                name="rsiPeriods"
                type="number"
                value={settings.rsiPeriods}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Buy Threshold"
                name="rsiBuyThreshold"
                type="number"
                value={settings.rsiBuyThreshold}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="RSI Sell Threshold"
                name="rsiSellThreshold"
                type="number"
                value={settings.rsiSellThreshold}
                onChange={handleChange}
              />
            </Grid>
        </form>
      </CardContent>
    </Card>
  );
};

export default Settings;


