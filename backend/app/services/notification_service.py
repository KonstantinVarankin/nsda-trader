from fastapi import WebSocket
from typing import List
import asyncio
import json

class NotificationManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_notification(self, message: str, notification_type: str):
        for connection in self.active_connections:
            await connection.send_text(json.dumps({
                "message": message,
                "type": notification_type
            }))

notification_manager = NotificationManager()

async def send_trade_notification(symbol: str, action: str, quantity: float, price: float):
    message = f"Trade executed: {action} {quantity} {symbol} at {price}"
    await notification_manager.send_notification(message, "success")

async def send_order_notification(symbol: str, action: str, quantity: float, price: float):
    message = f"New order placed: {action} {quantity} {symbol} at {price}"
    await notification_manager.send_notification(message, "info")

async def send_error_notification(error_message: str):
    await notification_manager.send_notification(error_message, "error")

async def send_warning_notification(warning_message: str):
    await notification_manager.send_notification(warning_message, "warning")
