from typing import List
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, webSocket: WebSocket): 
        await webSocket.accept()
        self.active_connections.append(webSocket)

    async def disconnect(self, webSocket: WebSocket): 
        if webSocket in self.active_connections:
            self.active_connections.remove(webSocket)

    async def send_personal_message(self, message: str, webSocket: WebSocket):
        try:
            await webSocket.send_text(message)
        except:
            print("WebSocket connection closed, cannot send message.")

    async def broadcast(self, message: str):
        disconnected_clients = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Mark disconnected WebSocket for removal
                disconnected_clients.append(connection)

        # Remove disconnected clients
        for connection in disconnected_clients:
            self.active_connections.remove(connection)
