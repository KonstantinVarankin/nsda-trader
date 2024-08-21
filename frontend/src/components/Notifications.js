import React, { useEffect } from 'react';
import { useSnackbar } from 'notistack';

const Notifications = () => {
  const { enqueueSnackbar } = useSnackbar();

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => {
      console.log('WebSocket ���������� �����������');
    };

    ws.onmessage = (event) => {
      const notification = JSON.parse(event.data);
      enqueueSnackbar(notification.message, { 
        variant: notification.type,
        autoHideDuration: 5000,
      });
    };

    ws.onerror = (error) => {
      console.error('WebSocket ������:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket ���������� �������');
    };

    return () => {
      ws.close();
    };
  }, [enqueueSnackbar]);

  return null; // ���� ��������� �� �������� UI ��������
};

export default Notifications;
