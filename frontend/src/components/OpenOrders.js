import React, { useState, useEffect } from 'react';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Button } from '@material-ui/core';
import axios from 'axios';

const OpenOrders = () => {
  const [orders, setOrders] = useState([]);

  useEffect(() => {
    fetchOpenOrders();
  }, []);

  const fetchOpenOrders = async () => {
    try {
      const response = await axios.get('/api/v1/trading/open-orders');
      setOrders(response.data);
    } catch (error) {
      console.error('Error fetching open orders:', error);
    }
  };

  const handleCancelOrder = async (symbol, orderId) => {
    try {
      await axios.delete(/api/v1/trading/cancel-order//);
      fetchOpenOrders(); // Refresh the list after cancelling an order
    } catch (error) {
      console.error('Error cancelling order:', error);
    }
  };

  return (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Symbol</TableCell>
            <TableCell>Side</TableCell>
            <TableCell>Type</TableCell>
            <TableCell>Price</TableCell>
            <TableCell>Amount</TableCell>
            <TableCell>Action</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {orders.map((order) => (
            <TableRow key={order.orderId}>
              <TableCell>{order.symbol}</TableCell>
              <TableCell>{order.side}</TableCell>
              <TableCell>{order.type}</TableCell>
              <TableCell>{order.price}</TableCell>
              <TableCell>{order.origQty}</TableCell>
              <TableCell>
                <Button
                  variant="contained"
                  color="secondary"
                  onClick={() => handleCancelOrder(order.symbol, order.orderId)}
                >
                  Cancel
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default OpenOrders;
