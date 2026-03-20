"""FastAPI application entry point — hardened."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.api.routes import router as api_router
from app.api.websocket import router as ws_router
from app.api.chat import router as chat_router
from app.models.inference import VulnerabilityInference
from app.security.rate_limiter import limiter, rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# ── Structured Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global inference engine
inference_engine: VulnerabilityInference = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    global inference_engine

    logger.info("Starting Python Vulnerability Detector...")
    logger.info("Device: %s", "GPU" if settings.USE_GPU else "CPU")

    # Initialize inference engine
    inference_engine = VulnerabilityInference()
    app.state.inference_engine = inference_engine

    logger.info("Inference engine initialized")

    yield

    # Cleanup
    logger.info("Shutting down...")


# ── Create FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    lifespan=lifespan,
    # Disable interactive docs in production
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# ── Rate Limiter ──────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# ── CORS — hardened ───────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# ── Routers ───────────────────────────────────────────────────────────────
app.include_router(api_router, prefix=settings.API_PREFIX)
app.include_router(chat_router, prefix=settings.API_PREFIX)
app.include_router(ws_router, prefix="/ws")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    from app.config import DEVICE
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": inference_engine is not None,
    }
