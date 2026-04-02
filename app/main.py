"""
STEP 2: FastAPI Backend — Entry Point
======================================
FastAPI is a modern Python web framework.
It auto-generates API docs, validates inputs,
and handles async requests efficiently.

How it works:
  1. We create an `app` instance
  2. We register route handlers (our API endpoints)
  3. We add CORS so the frontend can call this backend
  4. Uvicorn (a server) runs this file
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.analyze import router as analyze_router
from app.routes.health import router as health_router

# --- Create the FastAPI app ---
# title/description appear in auto-generated docs at /docs
app = FastAPI(
    title="AI Health Risk Assistant API",
    description="NLP-powered symptom analysis and risk prediction",
    version="1.0.0"
)

# --- CORS Middleware ---
# CORS = Cross-Origin Resource Sharing
# Without this, browsers block frontend JS from calling our API
# (browsers enforce the "same-origin policy" for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # In production: specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],        # Allow GET, POST, PUT, etc.
    allow_headers=["*"],        # Allow Content-Type, Authorization, etc.
)

# --- Register route groups ---
# Each router is a collection of related endpoints
# prefix="/api/v1" means all routes start with /api/v1/...
app.include_router(health_router, prefix="/api/v1", tags=["Health Check"])
app.include_router(analyze_router, prefix="/api/v1", tags=["Symptom Analysis"])


# --- Root endpoint ---
@app.get("/")
async def root():
    return {
        "message": "AI Health Risk Assistant API is running",
        "docs": "/docs",        # Swagger UI — interactive API explorer
        "redoc": "/redoc"       # Alternative docs UI
    }