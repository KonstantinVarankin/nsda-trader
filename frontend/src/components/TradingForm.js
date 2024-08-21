import React, { useState } from 'react';
import { TextField, Button, Select, MenuItem, FormControl, InputLabel } from '@mui/material';
import axios from 'axios';

const TradingForm = () => {
  const [symbol, setSymbol] = useState('');
  const [action, setAction] = useState('buy');
  const [quantity, setQuantity] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('/api/v1/trading/execute', {
        symbol,
        action,
        quantity: parseFloat(quantity),
      });
      console.log('Trade executed:', response.data);
      // Here you would typically show a success message to the user
    } catch (error) {
      console.error('Error executing trade:', error);
      // Here you would typically show an error message to the user
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <TextField
        label="Symbol"
        value={symbol}
        onChange={(e) => setSymbol(e.target.value)}
        fullWidth
        margin="normal"
      />
      <FormControl fullWidth margin="normal">
        <InputLabel>Action</InputLabel>
        <Select value={action} onChange={(e) => setAction(e.target.value)}>
          <MenuItem value="buy">Buy</MenuItem>
          <MenuItem value="sell">Sell</MenuItem>
        </Select>
      </FormControl>
      <TextField
        label="Quantity"
        type="number"
        value={quantity}
        onChange={(e) => setQuantity(e.target.value)}
        fullWidth
        margin="normal"
      />
      <Button type="submit" variant="contained" color="primary">
        Execute Trade
      </Button>
    </form>
  );
};

export default TradingForm;

