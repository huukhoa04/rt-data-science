from fastapi import APIRouter, HTTPException, Path, Body, Query
from typing import List, Dict, Optional, Any
from services.vector_service import VectorService
from core.config import RECORD_NAMESPACE
from models.pinecone_model import Movie, MovieResponse
from models.schemas import MoviePaginationRequest, MovieSearchRequest, MovieRecommendRequest

router = APIRouter()
vector_service = VectorService()

@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """
    Health check endpoint to verify if the service is running.
    """
    return {"status": "ok"}

@router.get("/", response_model=Dict[str, Any])
async def get_movies_by_page(
    limit: int = Query(10, ge=1, le=50, description="Number of movies to return per page"),
    pagination_token: Optional[str] = Query(None, description="Token for getting the next page")
):
    """
    Get paginated list of movies.
    
    Parameters:
    - limit: Number of movies to return per page
    - pagination_token: Token for getting the next page (null when there are no more pages)
    """
    try:
        # Get list of vector IDs
        result = await vector_service.list_vectors(
            namespace=RECORD_NAMESPACE,
            pagination_token=pagination_token,
            limit=limit
        )
        
        vector_ids = result.get("vector_ids", [])
        next_token = result.get("next_pagination_token")
        
        print(f"Retrieved {len(vector_ids)} vector IDs")
        
        # Empty response is valid - return early
        if not vector_ids:
            return {
                "movies": [],
                "pagination_token": next_token
            }
        
        # Process movies in smaller batches to avoid large requests
        movies = []
        batch_size = 10
        
        for i in range(0, len(vector_ids), batch_size):
            batch_ids = vector_ids[i:i + batch_size]
            try:
                vectors = await vector_service.fetch_vectors(ids=batch_ids)
               
                # Process each movie in the batch
                for movie_id, vector_data in vectors.items():
                    try:
                        movie_data = vector_data.get("metadata", {})
                        values = vector_data.get("values", [])
                        
                        # Create a Movie object from metadata
                        movie = Movie(**movie_data)
                        
                        # Add to the movies list
                        movies.append({
                            "id": movie_id,
                            "metadata": movie
                        })
                    except Exception as e:
                        print(f"Error processing movie {movie_id}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Error fetching batch of vectors: {str(e)}")
                # Continue processing other batches even if one fails
        
        return {
            "movies": movies,
            "pagination_token": next_token
        }
        
    except Exception as e:
        print(f"Error in get_movies_by_page: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch movies: {str(e)}")
    

@router.get("/{movie_id}", response_model=MovieResponse)
async def get_movie_by_id(movie_id: str = Path(..., description="The movie ID")):
    """
    Get a specific movie by its ID.
    """
    try:
        vectors = await vector_service.fetch_vectors(ids=[movie_id])
        
        if movie_id not in vectors:
            raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
        
        vector_data = vectors[movie_id]
        
        # Create a Movie object from metadata
        movie = Movie(**vector_data.get("metadata", {}))
        
        return MovieResponse(
            id=movie_id,
            values=vector_data.get("values", []),
            metadata=movie
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch movie: {str(e)}")

@router.post("/search", response_model=List[MovieResponse])
async def search_movies(request: MovieSearchRequest = Body(...)):
    """
    Search for movies by title and genres.
    """
    try:
        results = await vector_service.search_filtered(
            text=request.query,
            filter_fields=["title", "genres"],
            top_k=request.top_k
        )
        
        movies = []
        for match in results.get("matches", []):
            try:
                movie = Movie(**match.get("metadata", {}))
                movies.append(
                    MovieResponse(
                        id=match["id"],
                        values=match.get("values", [])[:10],  # Truncate values to reduce payload
                        metadata=movie
                    )
                )
            except Exception as e:
                print(f"Error processing search result: {str(e)}")
                continue
                
        return movies
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/recommend", response_model=List[MovieResponse])
async def get_recommendations(request: MovieRecommendRequest = Body(...)):
    """
    Get movie recommendations based on a text query.
    """
    try:
        # Generate embedding for the query text
        embedding = await vector_service.get_text_embedding(request.query)
        
        if not embedding:
            raise HTTPException(status_code=400, detail="Failed to generate embedding for query")
        
        # Search for movies similar to the embedding
        results = await vector_service.search_vector(vector=embedding, top_k=request.top_k)
        
        movies = []
        for match in results.get("matches", []):
            try:
                # Extract metadata and handle missing fields gracefully
                movie_data = match.get("metadata", {})
                
                # Check if any required fields are missing and add defaults
                required_fields = ["title", "description", "audienceConsensus", "cast", "criticsConsensus"]
                for field in required_fields:
                    if field not in movie_data:
                        movie_data[field] = ""
                
                # Create Movie object with relaxed validation
                movie = Movie.parse_obj(movie_data)
                
                movies.append(
                    MovieResponse(
                        id=match["id"],
                        values=match.get("values", [])[:10],  # Truncate values
                        metadata=movie
                    )
                )
            except Exception as e:
                print(f"Error processing recommendation: {str(e)}")
                continue
                
        return movies
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")