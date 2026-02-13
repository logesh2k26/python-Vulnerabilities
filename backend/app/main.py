"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.api.routes import router as api_router
from app.api.websocket import router as ws_router
from app.models.inference import VulnerabilityInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global inference engine
inference_engine: VulnerabilityInference = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    global inference_engine
    
    logger.info("Starting Python Vulnerability Detector...")
    logger.info(f"Device: {settings.USE_GPU and 'GPU' or 'CPU'}")
    
    # Initialize inference engine
    inference_engine = VulnerabilityInference()
    app.state.inference_engine = inference_engine
    
    logger.info("Inference engine initialized")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix=settings.API_PREFIX)
app.include_router(ws_router, prefix="/ws")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": settings.API_TITLE,
        "version": settings.API_VERSION
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    from app.config import DEVICE
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": inference_engine is not None
    }
