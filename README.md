# Python Vulnerability Detector

AI-powered security scanner for Python code using semantic AST embeddings and Graph Neural Networks.

## Features

- 🔍 **Deep Code Analysis** - AST parsing with data flow and control flow analysis
- 🧠 **AI-Powered Detection** - Graph Neural Network with attention mechanism
- 🛡️ **6+ Vulnerability Types** - eval/exec, command injection, deserialization, secrets, SQL injection, path traversal
- 📊 **Confidence Scores** - Softmax-based probability scores
- 🎯 **Explainability** - Line-level highlighting of vulnerable code
- 🌐 **Modern Web UI** - Glassmorphism dark theme with real-time analysis

## Quick Start

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/analyze` | POST | Analyze single file |
| `/api/v1/analyze/batch` | POST | Analyze multiple files |
| `/api/v1/analyze/upload` | POST | Upload and analyze files |
| `/ws/analyze` | WebSocket | Real-time analysis |

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routes
│   │   ├── core/         # AST parsing, graph building
│   │   ├── detectors/    # Vulnerability detectors
│   │   ├── models/       # GNN model
│   │   └── explainability/
│   └── training/         # Model training scripts
├── frontend/
│   └── src/
│       └── components/   # React components
└── samples/              # Test files
```

## Detected Vulnerabilities

- **eval/exec** - Code injection via dynamic execution
- **Command Injection** - OS command execution
- **Unsafe Deserialization** - pickle, yaml, marshal
- **Hardcoded Secrets** - API keys, passwords, tokens
- **SQL Injection** - Unparameterized queries
- **Path Traversal** - File path manipulation
- **SSRF** - Server-side request forgery

## Tech Stack

- **Backend**: FastAPI, PyTorch, PyTorch Geometric
- **Frontend**: React, Vite, Monaco Editor
- **ML**: Graph Attention Network (GAT)
