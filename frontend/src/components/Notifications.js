import React, { useEffect } from 'react';
import { useSnackbar } from 'notistack';

const Notifications = () => {
  const { enqueueSnackbar } = useSnackbar();

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => {
      console.log('WebSocket соединение установлено');
    };

    ws.onmessage = (event) => {
      const notification = JSON.parse(event.data);
      enqueueSnackbar(notification.message, { 
        variant: notification.type,
        autoHideDuration: 5000,
      });
    };

    ws.onerror = (error) => {
      console.error('WebSocket ошибка:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket соединение закрыто');
    };

    return () => {
      ws.close();
    };
  }, [enqueueSnackbar]);

  return null; // Ётот компонент не рендерит UI напр€мую
};

export default Notifications;
