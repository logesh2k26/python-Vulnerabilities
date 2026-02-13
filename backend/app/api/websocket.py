"""WebSocket handler for real-time analysis."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_result(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)


manager = ConnectionManager()


@router.websocket("/analyze")
async def websocket_analyze(websocket: WebSocket):
    """Real-time code analysis via WebSocket."""
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            
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
                "message": "Parsing AST..."
            })
            
            result = engine.analyze(code, filename)
            
            await manager.send_result(websocket, {
                "status": "complete",
                "result": result
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
