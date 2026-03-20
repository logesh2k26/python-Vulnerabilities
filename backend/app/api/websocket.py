"""WebSocket handler for real-time analysis — hardened."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import logging

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Maximum message size (bytes) ──────────────────────────────────────────
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB


class ConnectionManager:
    """Manage WebSocket connections with a hard cap."""

    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> bool:
        """Accept a connection if under the limit, otherwise reject."""
        if len(self.active_connections) >= self.max_connections:
            await websocket.close(code=1013, reason="Max connections reached")
            logger.warning(
                "WebSocket connection rejected: %d/%d",
                len(self.active_connections),
                self.max_connections,
            )
            return False
        await websocket.accept()
        self.active_connections.append(websocket)
        return True

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_result(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)


manager = ConnectionManager(max_connections=settings.MAX_WS_CONNECTIONS)


@router.websocket("/analyze")
async def websocket_analyze(websocket: WebSocket):
    """Real-time code analysis via WebSocket."""
    connected = await manager.connect(websocket)
    if not connected:
        return

    try:
        while True:
            data = await websocket.receive_text()

            # Message size guard
            if len(data) > MAX_MESSAGE_SIZE:
                await manager.send_result(websocket, {"error": "Message too large"})
                continue

            try:
                request = json.loads(data)
            except json.JSONDecodeError:
                await manager.send_result(websocket, {"error": "Invalid JSON"})
                continue

            code = request.get("code", "")
            filename = request.get("filename", "code.py")

            if not code.strip():
                await manager.send_result(websocket, {"error": "Empty code"})
                continue

            # Get inference engine from app state
            engine = websocket.app.state.inference_engine

            # Send progress
            await manager.send_result(websocket, {
                "status": "analyzing",
                "message": "Parsing AST...",
            })

            result = engine.analyze(code, filename)

            await manager.send_result(websocket, {
                "status": "complete",
                "result": result,
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
