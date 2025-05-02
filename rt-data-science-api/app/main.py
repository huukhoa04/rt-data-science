"""
Main application module for the Movie Vector API.
This module initializes the FastAPI application, configures CORS,
and includes all API routes.
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.endpoints.movies import router as movies_router

app = FastAPI(
    title="Movie Vector API",
    description="API for movie recommendations and vector search",
    version="0.1.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(movies_router, prefix="/api/movies", tags=["movies"])

@app.get("/")
def read_root():
    """
    Root endpoint that provides basic API information and available endpoints.
    
    Returns:
        dict: Information about the API including available endpoints
    """
    return {
        "message": "Movie Vector API is running",
        "docs": "/docs",
        "endpoints": {
            "movies": "/api/movies"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)