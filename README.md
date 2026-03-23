# Python Vulnerability Detector  # #UNDER WORKING   # ITS NOT COMPLETED YET

AI-powered security scanner for Python code using semantic AST embeddings and Graph Neural Networks.

## Features

- 🔍 **Deep Code Analysis** - AST parsing with data flow and control flow analysis
- 🧠 **AI-Powered Detection** - Graph Neural Network with attention mechanism
- 🛡️ **11+ Vulnerability Types** - eval/exec, command injection, deserialization, secrets, SQL injection, path traversal, SSRF, Cryptography, XXE, ReDoS, XSS
- 📊 **Confidence Scores** - Softmax-based probability scores
- 🎯 **Explainability** - Line-level highlighting with GNN attention mapping
- 🌐 **Modern Web UI** - Glassmorphism dark theme with real-time analysis

## Documentation

- [System Architecture](system_architecture.md) - Deep dive into GNN and AST engine
- [Walkthrough & Verification](walkthrough.md) - Proof of work and testing results

## Quick Start

### 1. Environment Setup

```bash
# Copy the example env file and fill in your API keys
cp .env.example .env
cp backend/.env.example backend/.env
```

Edit the `.env` files and add your keys:
- `GEMINI_API_KEY` — Google Gemini API key ([get one here](https://aistudio.google.com/apikey))
- `OPENROUTER_API_KEY` — OpenRouter API key ([get one here](https://openrouter.ai/keys))
- `API_SECRET_KEY` — Any strong random string for API authentication

> ⚠️ **Never commit `.env` files** — they are excluded via `.gitignore`.

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend

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
- **SQL Injection** - Unparameterized queries (Optimized heuristics)
- **Path Traversal** - File path manipulation
- **SSRF** - Server-side request forgery
- **Insecure Cryptography** - [NEW] Weak hashes and insecure randomness
- **XXE** - [NEW] XML External Entity expansion
- **ReDoS** - [NEW] Regex Denial of Service patterns
- **XSS** - [NEW] Cross-Site Scripting in web responses


## Tech Stack

- **Backend**: FastAPI, PyTorch, PyTorch Geometric
- **Frontend**: React, Vite, Monaco Editor
- **ML**: Graph Attention Network (GAT)
