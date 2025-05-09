from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from models.pinecone_model import Movie


# Model for embedding text queries
class TextQueryRequest(BaseModel):
    text: str
    top_k: int = Field(5, description="Number of results to return")

# Models for vector upsertion
class UpsertVectorRequest(BaseModel):
    id: str
    values: List[float]
    metadata: Movie

class PaginationRequest(BaseModel):
    pagination_token: Optional[str] = None
    limit: int = Field(100, ge=1, le=1000)


# Models for the API
class MoviePaginationRequest(BaseModel):
    """Request body for paginated movie list"""
    limit: int = Field(default=10, ge=1, le=50, description="Number of movies to return per page")
    pagination_token: Optional[str] = Field(default=None, description="Token for getting the next page")

class MovieSearchRequest(BaseModel):
    """Request body for movie search"""
    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")

class MovieRecommendRequest(BaseModel):
    """Request body for movie recommendations"""
    query: str = Field(..., min_length=1, description="Query text or movie title for recommendations")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")

class MoviePredictionRequest(BaseModel):
    """Request body for movie prediction"""
    Title: str = Field(..., description="Movie title")
    Year: Optional[int] = Field(None, description="Release year")
    Duration: Optional[str] = Field(None, description="Movie duration (e.g., '2h15m')")
    Rating: Optional[str] = Field(None, description="Movie rating (G, PG, PG-13, R, NC-17)")
    Director: Optional[str] = Field(None, description="Director name")
    Synopsis: Optional[str] = Field(None, description="Movie synopsis or description")
    
    # Character fields (up to 5)
    Character1: Optional[str] = Field(None, description="Main character 1")
    Character2: Optional[str] = Field(None, description="Main character 2")
    Character3: Optional[str] = Field(None, description="Main character 3")
    Character4: Optional[str] = Field(None, description="Main character 4")
    Character5: Optional[str] = Field(None, description="Main character 5")
    
    # Genre fields (binary 0/1)
    Action: Optional[int] = Field(0, ge=0, le=1, description="Action genre (1 if applicable)")
    Adventure: Optional[int] = Field(0, ge=0, le=1, description="Adventure genre (1 if applicable)")
    Animation: Optional[int] = Field(0, ge=0, le=1, description="Animation genre (1 if applicable)")
    Comedy: Optional[int] = Field(0, ge=0, le=1, description="Comedy genre (1 if applicable)")
    Crime: Optional[int] = Field(0, ge=0, le=1, description="Crime genre (1 if applicable)")
    Documentary: Optional[int] = Field(0, ge=0, le=1, description="Documentary genre (1 if applicable)")
    Drama: Optional[int] = Field(0, ge=0, le=1, description="Drama genre (1 if applicable)")
    Family: Optional[int] = Field(0, ge=0, le=1, description="Family genre (1 if applicable)")
    Fantasy: Optional[int] = Field(0, ge=0, le=1, description="Fantasy genre (1 if applicable)")
    Horror: Optional[int] = Field(0, ge=0, le=1, description="Horror genre (1 if applicable)")
    Musical: Optional[int] = Field(0, ge=0, le=1, description="Musical genre (1 if applicable)")
    Mystery: Optional[int] = Field(0, ge=0, le=1, description="Mystery genre (1 if applicable)")
    Romance: Optional[int] = Field(0, ge=0, le=1, description="Romance genre (1 if applicable)")
    SciFi: Optional[int] = Field(0, ge=0, le=1, description="Sci-Fi genre (1 if applicable)")
    Thriller: Optional[int] = Field(0, ge=0, le=1, description="Thriller genre (1 if applicable)")
    Western: Optional[int] = Field(0, ge=0, le=1, description="Western genre (1 if applicable)")

class MoviePredictionResponse(BaseModel):
    """Response for movie prediction"""
    title: str
    audience_score: float = Field(..., description="Audience score (percentage on 0-100 scale)")
    critics_score: float = Field(..., description="Critics score (percentage on 0-100 scale)") 
    explanation: str