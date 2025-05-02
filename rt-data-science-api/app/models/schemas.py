from pydantic import BaseModel, Field
from typing import List, Optional
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
    limit: int = Field(10, ge=1, le=100)
    pagination_token: Optional[str] = None
    
class MovieSearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")

class MovieRecommendRequest(BaseModel):
    query: str = Field(..., description="Text query for movie recommendations")
    top_k: int = Field(5, ge=1, le=20, description="Number of recommendations to return")