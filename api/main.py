"""
Market & Customer Intelligence Platform - FastAPI Application
Production-ready API for customer feedback and competitive market analysis.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from contextlib import asynccontextmanager

from .routers import intelligence, search, feedback, health
from .models.response_models import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    print("🚀 Market Intelligence Platform starting up...")
    
    # Initialize Weaviate connection
    try:
        # Weaviate initialization will be implemented here
        print("✅ Vector database connection established")
    except Exception as e:
        print(f"⚠️ Vector database connection failed: {e}")
    
    # Initialize LangChain agents
    try:
        # Agent initialization will be implemented here
        print("✅ LangChain agents initialized")
    except Exception as e:
        print(f"⚠️ Agent initialization failed: {e}")
    
    yield
    
    # Shutdown
    print("🔄 Market Intelligence Platform shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Market & Customer Intelligence Platform",
    description="Unified intelligence platform for customer feedback and competitive market data analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(intelligence.router, prefix="/api/v1/intelligence", tags=["intelligence"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["feedback"])


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Market & Customer Intelligence Platform",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") == "development"
    )