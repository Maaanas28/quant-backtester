
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn
from datetime import datetime
import os
from pathlib import Path
import sys

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.api import routes
from database.database import create_db_and_tables
from config import config
from utils.logging_config import setup_logger
from utils.exceptions import NeuroQuantException

# Setup logger
logger = setup_logger(__name__)

# Initialize database
logger.info("Initializing database...")
create_db_and_tables()

# Create FastAPI app
app = FastAPI(
    title=config.api.TITLE,
    version=config.api.VERSION,
    description=config.api.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors.ORIGINS,
    allow_credentials=config.cors.ALLOW_CREDENTIALS,
    allow_methods=config.cors.ALLOW_METHODS,
    allow_headers=config.cors.ALLOW_HEADERS,
)


# Exception handlers
@app.exception_handler(NeuroQuantException)
async def neuroquant_exception_handler(request, exc):
    """Handle custom NeuroQuant exceptions"""
    logger.error(f"NeuroQuant exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.__class__.__name__,
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors"""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Execute on application startup"""
    logger.info("=" * 60)
    logger.info("Starting NeuroQuant Trading System")
    logger.info(f"Version: {config.api.VERSION}")
    logger.info(f"Environment: {config.development.ENVIRONMENT}")
    logger.info(f"Debug Mode: {config.development.DEBUG}")
    logger.info(f"Database: {config.database.URL}")
    logger.info(f"Sentiment Analysis: {'Enabled' if config.sentiment.ENABLED else 'Disabled'}")
    logger.info(f"Initial Portfolio: ${config.financial.INITIAL_PORTFOLIO:,.2f}")
    logger.info(f"API running on http://{config.api.HOST}:{config.api.PORT}")
    logger.info(f"API Docs: http://{config.api.HOST}:{config.api.PORT}/docs")
    logger.info(f"Frontend: http://{config.api.HOST if config.api.HOST != '0.0.0.0' else 'localhost'}:{config.api.PORT}")
    logger.info("=" * 60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Execute on application shutdown"""
    logger.info("Shutting down NeuroQuant Trading System...")


# Mount static files (frontend)
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    logger.info(f"Frontend mounted at /static")
    
    # Serve HTML pages - these take precedence over API routes
    @app.get("/")
    @app.head("/")  # Add HEAD method support for Render health checks
    async def serve_frontend():
        """Serve the landing page"""
        return FileResponse(frontend_dir / "index.html")
    
    @app.get("/dashboard")
    async def serve_dashboard():
        """Serve the professional dashboard"""
        return FileResponse(frontend_dir / "dashboard.html")
    
    @app.get("/strategies")
    async def serve_strategies_page():
        """Serve the strategies page"""
        return FileResponse(frontend_dir / "strategies.html")
    
    @app.get("/backtest")
    async def serve_backtest_page():
        """Serve the backtest page"""
        return FileResponse(frontend_dir / "backtest.html")
    
    @app.get("/compare")
    async def serve_compare_page():
        """Serve the strategy comparison page"""
        return FileResponse(frontend_dir / "compare.html")
    
    @app.get("/lab")
    async def serve_lab_page():
        """Serve the financial laboratory page"""
        return FileResponse(frontend_dir / "lab.html")
else:
    logger.warning(f"Frontend directory not found at {frontend_dir}")


# Include API routers (with /api prefix, won't conflict with frontend routes)
app.include_router(routes.router)

# Include advanced features router
from backend.api import advanced_routes
app.include_router(advanced_routes.router, prefix="/api")

# Include institutional backtesting router
from backend.api import backtest_routes
app.include_router(backtest_routes.router, prefix="/api/v2", tags=["Institutional Backtesting"])

# Include advanced tools router (Portfolio Optimization, Financial Modeling, Market Data Pipeline)
from backend.api import advanced_tools_routes
app.include_router(advanced_tools_routes.router, tags=["Advanced Tools"])



if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=config.api.HOST,
        port=config.api.PORT,
        reload=config.development.DEBUG,
        log_level="info"
    )

if __name__ == "__main__":
    logger.info("Starting server...")
    
    uvicorn.run(
        app,
        host=config.api.HOST,
        port=config.api.PORT,
        reload=config.api.RELOAD,
        log_level=config.logging.LEVEL.lower()
    )

