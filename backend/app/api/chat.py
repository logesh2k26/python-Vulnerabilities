"""Chat API for vulnerability explanation and multi-turn conversation."""
import logging
import httpx
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel

from app.config import settings
from app.security.auth import verify_api_key
from app.security.rate_limiter import limiter

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "google/gemini-2.0-flash-001"

SYSTEM_PROMPT = """You are an expert cybersecurity analyst embedded in a Python vulnerability detection tool. 
You help developers understand security vulnerabilities found in their code.
Be concise, precise, and actionable. Format your responses using Markdown.
When explaining vulnerabilities, cover: what makes it dangerous, how it can be exploited, and how to fix it."""


# --- One-shot explain endpoint (existing) ---

class ExplainRequest(BaseModel):
    vulnerability_type: str
    code_snippet: str
    description: str

class ExplainResponse(BaseModel):
    explanation: str


@router.post("/explain", response_model=ExplainResponse)
@limiter.limit(settings.RATE_LIMIT_ANALYZE)
async def explain_vulnerability(
    request: Request,
    payload: ExplainRequest,
    _key: str = Depends(verify_api_key),
):
    """Explain a vulnerability using OpenRouter AI."""
    
    if getattr(settings, "OPENROUTER_API_KEY", "") == "":
        raise HTTPException(status_code=503, detail="OpenRouter API Key is not configured.")

    try:
        prompt = f"""
        Explain the following vulnerability found in Python code.
        
        Vulnerability Type: {payload.vulnerability_type}
        Description: {payload.description}
        
        Code Snippet:
        ```python
        {payload.code_snippet}
        ```
        
        Provide a concise explanation of:
        1. Why this code is vulnerable.
        2. How an attacker might exploit it.
        3. How to fix or mitigate the issue.
        """

        response_text = await _call_openrouter([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        return {"explanation": response_text}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error communicating with OpenRouter API: {e}")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")


# --- Multi-turn chat endpoint (new) ---

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    vulnerability_context: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_ANALYZE)
async def chat(
    request: Request,
    payload: ChatRequest,
    _key: str = Depends(verify_api_key),
):
    """Multi-turn chat about a vulnerability."""

    if getattr(settings, "OPENROUTER_API_KEY", "") == "":
        raise HTTPException(status_code=503, detail="OpenRouter API Key is not configured.")

    try:
        # Build system prompt with vulnerability context
        system_content = SYSTEM_PROMPT
        if payload.vulnerability_context:
            system_content += f"\n\nVulnerability Context:\n{payload.vulnerability_context}"

        api_messages = [{"role": "system", "content": system_content}]

        for msg in payload.messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        response_text = await _call_openrouter(api_messages)
        return {"reply": response_text}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")


# --- Shared helper ---

async def _call_openrouter(messages: list) -> str:
    """Call OpenRouter API and return the assistant's text."""
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": MODEL_NAME,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(OPENROUTER_URL, headers=headers, json=body)

    if response.status_code != 200:
        logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
        if response.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="API rate limit exceeded. Please try again later."
            )
        raise HTTPException(status_code=500, detail=f"AI Error: {response.text}")

    data = response.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    if not text:
        raise HTTPException(status_code=500, detail="AI returned an empty response.")

    return text
